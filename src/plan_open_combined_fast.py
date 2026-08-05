import os
import sys

import numpy as np
from numpy.linalg import norm

from src.old.bellman_v3 import BARRIER_TYPE
from utils import (
    create_transition_open,
    get_complete_DR,
    place_walls,
    h_wall_edges,
    v_wall_edges,
    plan_with_M,
    plan_with_D,
    plan_random,
)
from models import build_features, woodbury_feat, woodbury_full, woodbury_td

# =============================================================================
# CONFIG
# =============================================================================
ROWS, COLS = 15, 15
K          = ROWS * COLS

# =============================================================================
# BARRIER GEOMETRY
# =============================================================================

H1_ROW       = 4
H1_COL_START = 1
H1_COL_END   = 9
H2_ROW       = 12
H2_COL_START = 6
H2_COL_END   = 14

# -- Evaluation regions ------------------------------------------------------
START_ROW_MIN, START_ROW_MAX = 1, 2
START_COL_MIN, START_COL_MAX = 3, 11
GOAL_ROW_MIN,  GOAL_ROW_MAX  = 13, 14
GOAL_COL_MIN,  GOAL_COL_MAX  = 3, 11


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result(name, mean, median, success):
    print(f"  {name:<24s}  mean = {mean:6.1f}  median = {median:5.1f}  success = {success:.2f}")


def _start_states():
    return [r * COLS + c
            for r in range(START_ROW_MIN, START_ROW_MAX + 1)
            for c in range(START_COL_MIN, START_COL_MAX + 1)]


def _goal_states():
    return [r * COLS + c
            for r in range(GOAL_ROW_MIN, GOAL_ROW_MAX + 1)
            for c in range(GOAL_COL_MIN, GOAL_COL_MAX + 1)]


def _grid_neighbors(s):
    """4-connected grid neighbours of state s (no wrap-around)."""
    r, c = divmod(s, COLS)
    nbrs = []
    if r > 0:        nbrs.append((r - 1) * COLS + c)
    if r < ROWS - 1: nbrs.append((r + 1) * COLS + c)
    if c > 0:        nbrs.append(r * COLS + c - 1)
    if c < COLS - 1: nbrs.append(r * COLS + c + 1)
    return nbrs


def _barrier_within_states(wall_edges, n_steps):
    dist = {s: 1 for edge in wall_edges for s in edge}   # touching cells = step 1
    frontier, step = list(dist), 1
    while step < n_steps and frontier:
        nxt = []
        for s in frontier:
            for j in _grid_neighbors(s):
                if j not in dist:
                    dist[j] = step + 1
                    nxt.append(j)
        frontier, step = nxt, step + 1
    return np.array(sorted(dist))


def _barrier_sampling_p(barrier_states, n_states, frac):
    is_barrier = np.zeros(n_states, dtype=bool)
    is_barrier[barrier_states] = True
    n_b = int(is_barrier.sum())
    n_r = n_states - n_b
    p = np.empty(n_states)
    p[is_barrier]  = frac / n_b
    p[~is_barrier] = (1.0 - frac) / n_r
    return p


def _train_M(F, T_new, emc, lam, p_states, n_samples, lr, rng, init_M):
    N, k = F.shape
    M = np.eye(k) if init_M is None else init_M.copy()
    lam_inv = 1.0 / lam[:k]

    for t in range(n_samples):
        s  = rng.choice(N, p=p_states)
        sp = rng.choice(N, p=T_new[s])

        f_s  = F[s]
        f_sp = F[sp]
        r_s = f_s * lam_inv

        delta = emc * (r_s + f_sp @ M) - f_s @ M
        M += lr * np.outer(f_s, delta)

    return M


def build_environments():
    """
    Build the open field and the three barrier environments.

    Returns
    -------
    T_os, D_os, F, lam, emc, T_h, T_l, T_both, edges
    """
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    D_os = (D_os + D_os.T) / 2          # symmetrise

    h1 = h_wall_edges(ROWS, COLS, H1_ROW, H1_COL_START, H1_COL_END)
    h2 = h_wall_edges(ROWS, COLS, H2_ROW, H2_COL_START, H2_COL_END)
    v1 = v2 = []

    l1_edges = h1 + v1
    l2_edges = h2 + v2

    T_h    = place_walls(T_os, l1_edges)          # L1 only (near start) -> M1
    T_l    = place_walls(T_os, l2_edges)          # L2 only (near goal)  -> M2
    T_both = place_walls(T_os, l1_edges + l2_edges)  # both -> TEST environment

    edges = dict(l1_h=h1, l1_v=v1, l2_h=h2, l2_v=v2)
    F, lam, _, emc = build_features(D_os, k=K)
    return T_os, D_os, F, lam, emc, T_h, T_l, T_both, edges


# =============================================================================
# LEARNING-CURVE DATA COLLECTION
# =============================================================================

LR         = 0.1
TEMP       = 0.2
SEED       = 42

def train_and_save(
    n_train_steps: int = 1_000,
    eval_every:    int = 200,
    n_eval_trials: int = 200,
    eval_max_steps: int = 1_000,
    grad_type: str = "semi",
    barrier_frac: float = 0.9,          # fraction of TD samples from barrier states
    n_steps: int = 1,                   # "within this many steps" of the barrier
    save_dir: str = "../data/planning_open/combine_barrier/",
):
    np.random.seed(SEED)
    os.makedirs(save_dir, exist_ok=True)
    n_checkpoints = n_train_steps // eval_every

    T_os, D_os, F, lam, emc, T_h, T_l, T_both, edges = build_environments()
    print(f"  M1 edges {len(edges['l1_h'])}+{len(edges['l1_v'])} | M2 edges "
          f"{len(edges['l2_h'])}+{len(edges['l2_v'])} | test env = both")
    I = np.eye(K)

    # barrier states (within n_steps) and the biased sampling distributions
    h1_within = _barrier_within_states(edges['l1_h'], n_steps)   # M1 barrier (H1)
    h2_within = _barrier_within_states(edges['l2_h'], n_steps)   # M2 barrier (H2)
    p_M1 = _barrier_sampling_p(h1_within, ROWS * COLS, barrier_frac)
    p_M2 = _barrier_sampling_p(h2_within, ROWS * COLS, barrier_frac)
    print(f"  sampling: {barrier_frac:.0%} from within-{n_steps}-step barrier states "
          f"(M1 {h1_within.size} states, M2 {h2_within.size} states), rest uniform")

    print("\n  [woodbury_feat] M1, M2, and M1+M2-I ...")
    M1_feat = woodbury_feat(T_os, T_h, F)
    M2_feat = woodbury_feat(T_os, T_l, F)
    M_mb_comb = M1_feat + M2_feat - I

    print("  [woodbury_full] D1, D2, D1+D2-D_os, and exact combined map ...")
    D1 = woodbury_full(T_os, D_os, T_h)
    D2 = woodbury_full(T_os, D_os, T_l)
    D_full_comb = D1 + D2 - D_os
    D_exact = woodbury_full(T_os, D_os, T_both)

    model_names = ("dfr_mb_combined", "dfr_td_combined",
                   "dfr_mb_M1_only", "dfr_mb_M2_only", "dfr_td_M1_only", "dfr_td_M2_only",
                   "open_field", "woodbury_full_combined", "woodbury_full_exact")

    # fixed planners for the compute-once models
    plan_mb_comb = lambda g, s: plan_with_M(F, M_mb_comb, T_both, g, s, max_steps=eval_max_steps, temp=TEMP)
    plan_mb_M1   = lambda g, s: plan_with_M(F, M1_feat,   T_both, g, s, max_steps=eval_max_steps, temp=TEMP)
    plan_mb_M2   = lambda g, s: plan_with_M(F, M2_feat,   T_both, g, s, max_steps=eval_max_steps, temp=TEMP)
    plan_open    = lambda g, s: plan_with_M(F, I,         T_both, g, s, max_steps=eval_max_steps, temp=TEMP)
    plan_full_comb  = lambda g, s: plan_with_D(D_full_comb, T_both, g, s, max_steps=eval_max_steps, temp=TEMP)
    plan_full_exact = lambda g, s: plan_with_D(D_exact,     T_both, g, s, max_steps=eval_max_steps, temp=TEMP)

    checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every
    records = {name: np.zeros((n_checkpoints, n_eval_trials), dtype=np.int32)
               for name in model_names}

    # warm-started TD operators, one per barrier
    M1_td = None
    M2_td = None

    # one continuous RNG stream per barrier (seeded once, not per chunk), so
    # eval_every no longer changes which states are sampled during training
    rng_1 = np.random.RandomState(SEED)
    rng_2 = np.random.RandomState(SEED + 7919)

    # fixed (start, goal) region pairs -- identical across all checkpoints
    start_states, goal_states = _start_states(), _goal_states()
    eval_rng = np.random.RandomState(SEED)
    eval_state = eval_rng.get_state()

    def _eval(plan_fn):
        eval_rng.set_state(eval_state)
        np.random.seed(SEED)
        steps = np.zeros(n_eval_trials, dtype=np.int32)
        for i in range(n_eval_trials):
            start = eval_rng.choice(start_states)
            goal  = eval_rng.choice(goal_states)
            st, _ = plan_fn(goal, start)
            steps[i] = st
        return steps

    for chunk in range(n_checkpoints):
        print(f"\n  Chunk {chunk+1}/{n_checkpoints}  "
              f"(steps/barrier {chunk*eval_every+1:,} -> {(chunk+1)*eval_every:,})")

        # continue each barrier's single sample stream for eval_every more steps
        M1_td = _train_M(F, T_h, emc, lam, p_M1,
                         n_samples=eval_every, lr=LR, rng=rng_1, init_M=M1_td)

        M2_td = _train_M(F, T_l, emc, lam, p_M2,
                         n_samples=eval_every, lr=LR, rng=rng_2, init_M=M2_td)

        M_td_comb = M1_td + M2_td - I
        plan_td_comb = lambda g, s, _M=M_td_comb: plan_with_M(F, _M, T_both, g, s, max_steps=eval_max_steps, temp=TEMP)
        plan_td_M1   = lambda g, s: plan_with_M(F, M1_td, T_both, g, s, max_steps=eval_max_steps, temp=TEMP)
        plan_td_M2   = lambda g, s: plan_with_M(F, M2_td, T_both, g, s, max_steps=eval_max_steps, temp=TEMP)

        records["dfr_mb_combined"][chunk]        = _eval(plan_mb_comb)
        records["dfr_td_combined"][chunk]        = _eval(plan_td_comb)
        records["dfr_mb_M1_only"][chunk]         = _eval(plan_mb_M1)
        records["dfr_mb_M2_only"][chunk]         = _eval(plan_mb_M2)
        records["dfr_td_M1_only"][chunk]         = _eval(plan_td_M1)
        records["dfr_td_M2_only"][chunk]         = _eval(plan_td_M2)
        records["open_field"][chunk]             = _eval(plan_open)
        records["woodbury_full_combined"][chunk] = _eval(plan_full_comb)
        records["woodbury_full_exact"][chunk]    = _eval(plan_full_exact)

        for name in model_names:
            vals = records[name][chunk]
            m  = vals.mean()
            ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))   # 95% CI half-width
            print(f"    {name:<24s}  mean_steps = {m:6.1f}  "
                  f"95% CI [{m - ci:6.1f}, {m + ci:6.1f}]")

    save_path = os.path.join(save_dir, "learning_curves_combine.npz")
    np.savez(save_path,
             checkpoint_steps=checkpoint_steps,
             **records,)

    operator_path = os.path.join(save_dir, "M1_M2_td.npz")
    np.savez(operator_path, M1_td=M1_td, M2_td=M2_td,
             h1_within=h1_within, h2_within=h2_within,
             barrier_frac=barrier_frac, n_steps=n_steps)

    print(f"\n  Saved -> {save_path}")
    print(f"  Saved -> {operator_path}  (M1_td, M2_td: {M1_td.shape})")
    print(f"  Arrays: checkpoint_steps {checkpoint_steps.shape}, "
          f"each model {records['dfr_td_combined'].shape}")


if __name__ == '__main__':
    train_and_save()
