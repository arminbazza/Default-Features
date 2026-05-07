import os

import numpy as np

from utils import (
    create_transition_open,
    get_complete_DR,
    place_walls,
    h_wall_edges,
    plan_with_M,
)
from models import build_features, woodbury_feat, woodbury_td

# =============================================================================
# CONFIG
# =============================================================================
ROWS, COLS = 15, 15
K          = [50, 75, 100, 125, 150, 175, 200, 220, 225]
LR         = 0.01
TEMP       = 0.2
SEED       = 42

BARRIER_ROW = ROWS // 2
COL_START   = 3
COL_END     = 12


START_ROW_MIN, START_ROW_MAX = 3, 5
START_COL_MIN, START_COL_MAX = 5, 9
GOAL_ROW_MIN,  GOAL_ROW_MAX  = 8, 10
GOAL_COL_MIN,  GOAL_COL_MAX  = 5, 9


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result(name, mean, median, success):
    print(f"  {name:<20s}  mean = {mean:6.1f}  median = {median:5.1f}  success = {success:.2f}")


# =============================================================================
# LEARNING-CURVE DATA COLLECTION
# =============================================================================

def train_and_save(
    n_train_steps: int = 50_000,
    eval_every:    int = 1_000,
    n_eval_trials: int = 200,
    eval_max_steps: int = 1_000,
    save_dir: str = "../data/planning_open/horizontal/",
):
    np.random.seed(SEED)
    os.makedirs(save_dir, exist_ok=True)

    n_checkpoints = n_train_steps // eval_every

    # ------------------------------------------------------------------
    # 1. Environment
    # ------------------------------------------------------------------
    print_header("TRAIN & SAVE  –  open-field → horizontal barrier")
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)

    wall_edges = h_wall_edges(ROWS, COLS,
                               barrier_row=BARRIER_ROW,
                               col_start=COL_START,
                               col_end=COL_END)
    T_new = place_walls(T_os, wall_edges)
    print(f"  Grid {ROWS}x{COLS}, wall at row {BARRIER_ROW} cols {COL_START}–{COL_END-1}")
    
    all_records = []

    for k in K:
        print(f"\n  Running for K={k}...")
        # ------------------------------------------------------------------
        # 2. Feature basis
        # ------------------------------------------------------------------
        F, lam, _, emc = build_features(D_os, k=k)

        # ------------------------------------------------------------------
        # 3. Analytic models (computed once)
        # ------------------------------------------------------------------
        M_wb = woodbury_feat(T_os, T_new, F)
        plan_wb   = lambda g, s: plan_with_M(F, M_wb, T_new, g, s, max_steps=eval_max_steps, temp=TEMP)

        # ------------------------------------------------------------------
        # 4. Incremental training + evaluation
        # ------------------------------------------------------------------
        checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every

        # Storage: (n_checkpoints, n_eval_trials)
        records = {name: np.zeros((n_checkpoints, n_eval_trials), dtype=np.int32)
                for name in ("woodbury_feat", "woodbury_td")}
        M_td  = None

        start_states = [r * COLS + c
                        for r in range(START_ROW_MIN, START_ROW_MAX + 1)
                        for c in range(START_COL_MIN, START_COL_MAX + 1)]
        goal_states  = [r * COLS + c
                        for r in range(GOAL_ROW_MIN,  GOAL_ROW_MAX + 1)
                        for c in range(GOAL_COL_MIN,  GOAL_COL_MAX + 1)]
        eval_rng   = np.random.RandomState(SEED)
        eval_state = eval_rng.get_state()

        def _eval(plan_fn):
            eval_rng.set_state(eval_state)
            steps_arr = np.zeros(n_eval_trials, dtype=np.int32)
            for i in range(n_eval_trials):
                start = eval_rng.choice(start_states)
                goal  = eval_rng.choice(goal_states)
                st, _ = plan_fn(goal, start)
                steps_arr[i] = st
            return steps_arr

        for chunk in range(n_checkpoints):
            train_seed = SEED + chunk * 1009

            print(f"\n  Chunk {chunk+1}/{n_checkpoints}  "
                f"(steps {chunk*eval_every+1:,} → {(chunk+1)*eval_every:,})")

            # --- woodbury_td ---
            M_td = woodbury_td(F, T_new, emc, lam,
                            valid_states=np.arange(ROWS * COLS),
                            grad_type="full",
                            n_samples=eval_every,
                            lr=LR,
                            seed=train_seed,
                            init_M=M_td)
            plan_td = lambda g, s, _M=M_td: plan_with_M(F, _M, T_new, g, s,
                                                        max_steps=eval_max_steps,
                                                        temp=TEMP)

            # --- evaluate all models ---
            records["woodbury_feat"][chunk] = _eval(plan_wb)
            records["woodbury_td"][chunk]   = _eval(plan_td)

            # quick progress print
            for name in records:
                mean_steps = records[name][chunk].mean()
                print(f"    {name:<20s}  mean_steps = {mean_steps:.1f}")
                
        all_records.append(records)

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    save_path = os.path.join(save_dir, "num_steps_K.npz")
    save_dict = {"checkpoint_steps": checkpoint_steps}
    for k_val, records in zip(K, all_records):
        for model_name, arr in records.items():
            save_dict[f"{model_name}_K{k_val}"] = arr
    np.savez(save_path, **save_dict)
    print(f"\n  Saved → {save_path}")


if __name__ == '__main__':
    train_and_save()