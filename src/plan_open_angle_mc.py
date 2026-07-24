import os
import numpy as np
from numpy.linalg import norm

from utils import (
    create_transition_open,
    get_complete_DR,
    place_walls,
    h_wall_edges,
    v_wall_edges,
    plan_with_M,
    plan_with_V,
)
from models import build_features, woodbury_feat, woodbury_td, dyna_q

# =============================================================================
# CONFIG
# =============================================================================

ROWS, COLS = 15, 15
N_STATES   = ROWS * COLS
K          = ROWS * COLS
LR         = 0.01
TEMP       = 0.2
SEED       = 42

# L-barrier geometry
# Corner at (CORNER_ROW, CORNER_COL); one arm goes right, one goes down.
CORNER_ROW = 5    # horizontal arm sits on this row boundary (row 4 <-> row 5)
CORNER_COL = 5    # vertical arm sits on this col boundary  (col 4 <-> col 5)
H_END_COL  = 12   # horizontal arm: cols [CORNER_COL, H_END_COL)  = 5..11
V_END_ROW  = 12   # vertical arm:   rows [CORNER_ROW, V_END_ROW)  = 5..11

START_ROW_MAX = CORNER_ROW - 2   # start rows 0..3
START_COL_MAX = CORNER_COL - 2   # start cols 0..3
GOAL_ROW_MIN  = 6        # goal  rows 6..9
GOAL_COL_MIN  = 6        # goal  cols 6..9
GOAL_ROW_MAX  = 9        # goal  rows 6..9
GOAL_COL_MAX  = 9        # goal  cols 6..9

# Dyna-Q planning steps per real sample to sweep
K_PLANNING_VALS = [5, 10, 20, 50]


# =============================================================================
# LEARNING-CURVE DATA COLLECTION
# =============================================================================

def train_and_save(
    n_train_steps:  int = 50_000,
    eval_every:     int = 1_000,
    n_eval_trials:  int = 200,
    eval_max_steps: int = 1_000,
    save_dir:       str = "../data/planning_open/angle_mc/",
):
    np.random.seed(SEED)
    os.makedirs(save_dir, exist_ok=True)

    n_checkpoints = n_train_steps // eval_every

    # ------------------------------------------------------------------
    # 1. Environment
    # ------------------------------------------------------------------
    print("Building environment ...")
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)

    h_edges = h_wall_edges(ROWS, COLS,
                            barrier_row=CORNER_ROW,
                            col_start=CORNER_COL,
                            col_end=H_END_COL)
    v_edges = v_wall_edges(ROWS, COLS,
                            barrier_col=CORNER_COL,
                            row_start=CORNER_ROW,
                            row_end=V_END_ROW)
    wall_edges = h_edges + v_edges
    T_new = place_walls(T_os, wall_edges)
    print(f"  L-wall placed: {len(h_edges)} H edges + {len(v_edges)} V edges "
          f"= {len(wall_edges)} total blocked edges.")

    # ------------------------------------------------------------------
    # 2. Shared feature basis
    # ------------------------------------------------------------------
    F, lam, _, emc = build_features(D_os, k=K)
    print(f"  K={K}, emc={emc:.6f}")

    # ------------------------------------------------------------------
    # 3. Fixed goal for all models (center of goal region)
    # ------------------------------------------------------------------
    fixed_goal_row = (GOAL_ROW_MIN + GOAL_ROW_MAX) // 2
    fixed_goal_col = (GOAL_COL_MIN + GOAL_COL_MAX) // 2
    fixed_goal     = fixed_goal_row * COLS + fixed_goal_col
    print(f"  Fixed goal: state {fixed_goal} (row {fixed_goal_row}, col {fixed_goal_col})")

    # ------------------------------------------------------------------
    # 4. Analytic woodbury_feat (computed once, flat on learning curve)
    # ------------------------------------------------------------------
    print("\nComputing woodbury_feat ...")
    M_wb   = woodbury_feat(T_os, T_new, F)
    plan_wb = lambda g, s: plan_with_M(F, M_wb, T_new, g, s,
                                        max_steps=eval_max_steps, temp=TEMP)

    # ------------------------------------------------------------------
    # 5. Evaluation helpers
    # ------------------------------------------------------------------
    start_states = [r * COLS + c
                    for r in range(START_ROW_MAX + 1)
                    for c in range(START_COL_MAX + 1)]

    eval_rng   = np.random.RandomState(SEED)
    eval_state = eval_rng.get_state()

    def _eval(plan_fn):
        """200 trials: random start → fixed_goal. Same starts at every checkpoint."""
        eval_rng.set_state(eval_state)
        steps_arr = np.zeros(n_eval_trials, dtype=np.int32)
        for i in range(n_eval_trials):
            start     = eval_rng.choice(start_states)
            st, _     = plan_fn(fixed_goal, start)
            steps_arr[i] = st
        return steps_arr

    # ------------------------------------------------------------------
    # 6. Storage
    # ------------------------------------------------------------------
    dq_names     = [f'dyna_q_k{k}' for k in K_PLANNING_VALS]
    model_names  = ['woodbury_feat', 'woodbury_td'] + dq_names
    records      = {name: np.zeros((n_checkpoints, n_eval_trials), dtype=np.int32)
                    for name in model_names}
    checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every

    # ------------------------------------------------------------------
    # 7. Incremental training + evaluation
    # ------------------------------------------------------------------
    M_td    = None
    dq_state = {k: {'V': None, 'T_count': None} for k in K_PLANNING_VALS}

    for chunk in range(n_checkpoints):
        train_seed = SEED + chunk * 1009

        print(f"\nChunk {chunk+1}/{n_checkpoints}  "
              f"(real steps {chunk*eval_every+1:,} → {(chunk+1)*eval_every:,})")

        # --- woodbury_td ---
        M_td = woodbury_td(F, T_new, emc, lam,
                           valid_states=np.arange(N_STATES),
                           grad_type="full",
                           n_samples=eval_every,
                           lr=LR,
                           seed=train_seed,
                           init_M=M_td)
        plan_td = lambda g, s, _M=M_td: plan_with_M(F, _M, T_new, g, s,
                                                      max_steps=eval_max_steps,
                                                      temp=TEMP)

        # --- dyna_q for each k ---
        for k in K_PLANNING_VALS:
            V_dq, T_count_dq = dyna_q(T_new, fixed_goal,
                                       n_real_steps=eval_every,
                                       emc=emc,
                                       k_planning=k,
                                       lr=LR,
                                       r_goal=10,
                                       seed=train_seed,
                                       init_V=dq_state[k]['V'],
                                       init_T_count=dq_state[k]['T_count'])
            dq_state[k]['V']       = V_dq
            dq_state[k]['T_count'] = T_count_dq

        # --- evaluate all ---
        records['woodbury_feat'][chunk] = _eval(plan_wb)
        records['woodbury_td'][chunk]   = _eval(plan_td)
        for k in K_PLANNING_VALS:
            _V = dq_state[k]['V']
            plan_dq = lambda g, s, _Vk=_V: plan_with_V(_Vk, T_new, g, s,
                                                         max_steps=eval_max_steps,
                                                         temp=TEMP)
            records[f'dyna_q_k{k}'][chunk] = _eval(plan_dq)

        for name in model_names:
            print(f"    {name:<18s}  mean_steps = {records[name][chunk].mean():.1f}")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    save_path = os.path.join(save_dir, f"learning_curves_lr{LR}_TEST.npz")
    np.savez(save_path, checkpoint_steps=checkpoint_steps, **records)
    print(f"\nSaved → {save_path}")
    print(f"Arrays: checkpoint_steps {checkpoint_steps.shape}, "
          f"each model {records['woodbury_td'].shape}")
    print(f"Keys: checkpoint_steps, " + ", ".join(model_names))


if __name__ == '__main__':
    train_and_save()
