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
LR         = 0.01
N_TRIALS   = 300
MAX_STEPS  = 500
TEMP       = 0.2
SEED       = 42

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


def _barrier_adjacent_states(wall_edges):
    states = {s for edge in wall_edges for s in edge}          # above/below cells
    for r in {s // COLS for s in states}:
        cols_in_row = [s % COLS for s in states if s // COLS == r]
        for c in (min(cols_in_row) - 1, max(cols_in_row) + 1):  # left / right ends
            if 0 <= c < COLS:
                states.add(r * COLS + c)
    return np.array(sorted(states))


def _train_M_walk(F, T_new, emc, lam, n_samples, lr, seed, init_M,
                  start_states=None):
    N, k = F.shape
    if start_states is None:
        start_states = np.arange(N)
    rng = np.random.RandomState(seed)
    M = np.eye(k) if init_M is None else init_M.copy()
    lam_inv = 1.0 / lam[:k]

    s = rng.choice(start_states)
    visited = np.empty(n_samples, dtype=np.int64)
    for t in range(n_samples):
        sp = rng.choice(N, p=T_new[s])

        f_s  = F[s]
        f_sp = F[sp]
        r_s = f_s * lam_inv

        delta = emc * (r_s + f_sp @ M) - f_s @ M
        M += lr * np.outer(f_s, delta)

        visited[t] = s
        s = sp
    return M, visited


def build_environments():
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

def train_and_save(
    n_train_steps: int = 100_000,
    eval_every:    int = 5_000,
    n_eval_trials: int = 200,
    eval_max_steps: int = 1_000,
    seed: int = None,
    run: int = 0,
    save_dir: str = "../data/planning_open/combine/fast/",
):
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    n_checkpoints = n_train_steps // eval_every

    T_os, D_os, F, lam, emc, T_h, T_l, T_both, edges = build_environments()
    print(f"  M1 edges {len(edges['l1_h'])}+{len(edges['l1_v'])} | M2 edges "
          f"{len(edges['l2_h'])}+{len(edges['l2_v'])} | test env = both")
    I = np.eye(K)

    valid_states_all = np.arange(ROWS * COLS)
    h1_adjacent = _barrier_adjacent_states(edges['l1_h'])   # M1 barrier (H1)
    h2_adjacent = _barrier_adjacent_states(edges['l2_h'])   # M2 barrier (H2)
    sampled_M1, sampled_M2 = [], []

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

    # fixed (start, goal) region pairs -- identical across all checkpoints
    start_states, goal_states = _start_states(), _goal_states()
    eval_rng = np.random.RandomState(seed)
    eval_state = eval_rng.get_state()

    def _eval(plan_fn):
        eval_rng.set_state(eval_state)
        steps = np.zeros(n_eval_trials, dtype=np.int32)
        for i in range(n_eval_trials):
            start = eval_rng.choice(start_states)
            goal  = eval_rng.choice(goal_states)
            st, _ = plan_fn(goal, start)
            steps[i] = st
        return steps

    for chunk in range(n_checkpoints):
        # distinct sample streams for the two barriers (independent learning)
        seed_1 = seed + chunk * 1009
        seed_2 = seed + chunk * 1009 + 7919

        print(f"\n  Chunk {chunk+1}/{n_checkpoints}  "
              f"(steps/barrier {chunk*eval_every+1:,} -> {(chunk+1)*eval_every:,})")

        M1_td, visited_1 = _train_M_walk(F, T_h, emc, lam,
                                         n_samples=eval_every, lr=LR, seed=seed_1,
                                         init_M=M1_td, start_states=valid_states_all)

        M2_td, visited_2 = _train_M_walk(F, T_l, emc, lam,
                                         n_samples=eval_every, lr=LR, seed=seed_2,
                                         init_M=M2_td, start_states=valid_states_all)

        sampled_M1.append(visited_1)
        sampled_M2.append(visited_2)

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
            print(f"    {name:<24s}  mean_steps = {records[name][chunk].mean():6.1f}")

    sampled_M1 = np.concatenate(sampled_M1)      # (n_train_steps,)
    sampled_M2 = np.concatenate(sampled_M2)
    n1_adj = int(np.isin(sampled_M1, h1_adjacent).sum())
    n2_adj = int(np.isin(sampled_M2, h2_adjacent).sum())

    print_header("TD samples drawn from states touching the trained barrier")
    print(f"  M1 (H1): {n1_adj:,} / {sampled_M1.size:,} "
          f"({100 * n1_adj / sampled_M1.size:.2f}%)   "
          f"[{h1_adjacent.size} adjacent states]")
    print(f"  M2 (H2): {n2_adj:,} / {sampled_M2.size:,} "
          f"({100 * n2_adj / sampled_M2.size:.2f}%)   "
          f"[{h2_adjacent.size} adjacent states]")

    save_path = os.path.join(save_dir, f"learning_curves_combine_{run}.npz")
    np.savez(save_path,
             checkpoint_steps=checkpoint_steps,
             **records,)

    operator_path = os.path.join(save_dir, "M1_M2_td.npz")
    np.savez(operator_path, M1_td=M1_td, M2_td=M2_td,
             sampled_states_M1=sampled_M1, sampled_states_M2=sampled_M2,
             h1_adjacent=h1_adjacent, h2_adjacent=h2_adjacent)

    print(f"\n  Saved -> {save_path}")
    print(f"  Saved -> {operator_path}  (M1_td, M2_td: {M1_td.shape})")
    print(f"  Arrays: checkpoint_steps {checkpoint_steps.shape}, "
          f"each model {records['dfr_td_combined'].shape}")


if __name__ == '__main__':
    for i, seed in enumerate([0, 1, 2, 4]):
        print(i, seed)
        train_and_save(seed=seed, run=i, n_train_steps=100_000, eval_every=5_000, n_eval_trials=200, eval_max_steps=1_000)