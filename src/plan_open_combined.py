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
# CONFIG   (hyper-parameters kept identical to the open-field scripts)
# =============================================================================
ROWS, COLS = 15, 15
K          = ROWS * COLS          # 225
N_SAMPLES  = 50_000              # outer single-shot budget (main())
LR         = 0.01
N_TRIALS   = 300
MAX_STEPS  = 500
TEMP       = 0.2
SEED       = 42

# =============================================================================
# BARRIER GEOMETRY
# =============================================================================

# -- Layout "horizontal" : two complementary horizontal walls (S-shaped detour) -
#    W1 sits just BELOW the start block and extends RIGHT from the left wall
#       (opening on the right).  W2 sits just ABOVE the goal block and extends
#       LEFT from the right wall (opening on the left).  The two spans tile the
#       full width (cols 0..6 | 7..14) so no straight vertical corridor exists ->
#       the agent must go down-right (past W1) then down-left (past W2).
H1_ROW       = 4       # boundary between rows 3 and 4 (just below start rows 1-3)
H1_COL_START = 1
H1_COL_END   = 9       # blocks cols 0..6  -> opening cols 7..14 (right)
H2_ROW       = 12      # boundary between rows 11 and 12 (just above goal rows 12-14)
H2_COL_START = 7       # blocks cols 7..14 -> opening cols 0..6 (left)
H2_COL_END   = 14

# -- Evaluation regions ------------------------------------------------------
START_ROW_MIN, START_ROW_MAX = 1, 2      # top band, rows 1-3
START_COL_MIN, START_COL_MAX = 3, 11     #           cols 3-11
GOAL_ROW_MIN,  GOAL_ROW_MAX  = 13, 14    # bottom band, rows 12-14
GOAL_COL_MIN,  GOAL_COL_MAX  = 3, 11     #              cols 3-11


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
# LEARNING-CURVE DATA COLLECTION  (mirrors train_and_save in the open-field scripts)
# =============================================================================

def train_and_save(
    n_train_steps: int = 100_000,
    eval_every:    int = 5_000,
    n_eval_trials: int = 200,
    eval_max_steps: int = 1_000,
    grad_type: str = "semi",
    save_dir: str = "../data/planning_open/combine/",
):
    np.random.seed(SEED)
    os.makedirs(save_dir, exist_ok=True)
    n_checkpoints = n_train_steps // eval_every

    # ------------------------------------------------------------------ envs
    T_os, D_os, F, lam, emc, T_h, T_l, T_both, edges = build_environments()
    print(f"  M1 edges {len(edges['l1_h'])}+{len(edges['l1_v'])} | M2 edges "
          f"{len(edges['l2_h'])}+{len(edges['l2_v'])} | test env = both")
    I = np.eye(K)

    # ---------------------------------------------------- analytic (compute once)
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

    # ------------------------------------------------------------- bookkeeping
    checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every
    records = {name: np.zeros((n_checkpoints, n_eval_trials), dtype=np.int32)
               for name in model_names}

    # warm-started TD operators, one per barrier
    M1_td = None
    M2_td = None

    # fixed (start, goal) region pairs -- identical across all checkpoints
    start_states, goal_states = _start_states(), _goal_states()
    eval_rng = np.random.RandomState(SEED)
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
        seed_1 = SEED + chunk * 1009
        seed_2 = SEED + chunk * 1009 + 7919

        print(f"\n  Chunk {chunk+1}/{n_checkpoints}  "
              f"(steps/barrier {chunk*eval_every+1:,} -> {(chunk+1)*eval_every:,})")

        # --- TD on the horizontal-only env  -> M1_td ---
        M1_td = woodbury_td(F, T_h, emc, lam,
                            valid_states=np.arange(ROWS * COLS),
                            grad_type=grad_type,
                            n_samples=eval_every,
                            lr=LR,
                            seed=seed_1,
                            init_M=M1_td)

        # --- TD on the L-only env  -> M2_td ---
        M2_td = woodbury_td(F, T_l, emc, lam,
                            valid_states=np.arange(ROWS * COLS),
                            grad_type=grad_type,
                            n_samples=eval_every,
                            lr=LR,
                            seed=seed_2,
                            init_M=M2_td)

        # --- zero-shot composition, planned on the BOTH-barrier env ---
        M_td_comb = M1_td + M2_td - I
        plan_td_comb = lambda g, s, _M=M_td_comb: plan_with_M(F, _M, T_both, g, s, max_steps=eval_max_steps, temp=TEMP)
        plan_td_M1   = lambda g, s: plan_with_M(F, M1_td, T_both, g, s, max_steps=eval_max_steps, temp=TEMP)
        plan_td_M2   = lambda g, s: plan_with_M(F, M2_td, T_both, g, s, max_steps=eval_max_steps, temp=TEMP)

        # --- evaluate every model at this checkpoint ---
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

    # ------------------------------------------------------------------ save
    save_path = os.path.join(save_dir, "learning_curves_combine.npz")
    np.savez(save_path,
             checkpoint_steps=checkpoint_steps,
             **records,)
    print(f"\n  Saved -> {save_path}")
    print(f"  Arrays: checkpoint_steps {checkpoint_steps.shape}, "
          f"each model {records['dfr_td_combined'].shape}")


if __name__ == '__main__':
    train_and_save()