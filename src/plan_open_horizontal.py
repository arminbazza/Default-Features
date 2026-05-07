import os
import sys

import numpy as np
from numpy.linalg import norm

from utils import (
    create_transition_open,
    get_complete_DR,
    place_walls,
    h_wall_edges,
    plan_with_M,
    plan_with_D,
    plan_with_SR,
    plan_with_SF,
    plan_random,
)
from models import build_features, woodbury_feat, woodbury_full, woodbury_td, sr_td, sr_sf

# =============================================================================
# CONFIG
# =============================================================================
ROWS, COLS = 15, 15
K          = ROWS * COLS
N_SAMPLES  = 400_000
LR         = 0.01
N_TRIALS   = 300
MAX_STEPS  = 500
TEMP       = 0.2
SEED       = 42

# Wall geometry: H wall between row (BARRIER_ROW-1) and BARRIER_ROW,
# spanning columns [COL_START, COL_END). Gaps at cols 0..2 and 12..14.
BARRIER_ROW = ROWS // 2        # 7  (wall between rows 6 and 7)
COL_START   = 3
COL_END     = 12

# Evaluation regions — start directly above wall, goal directly below,
# both restricted to cols 5–9 (within the wall span).
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
# MAIN
# =============================================================================

def main():
    np.random.seed(SEED)    # seed global state for softmax planning
    print_header("PLANNING COMPARISON")
    print(f"  Grid      : {ROWS}x{COLS} = {ROWS*COLS} states")
    print(f"  Wall      : H, row {BARRIER_ROW}, cols {COL_START}–{COL_END-1}")
    print(f"  Features  : K = {K}")
    print(f"  Woodbury-TD     : {N_SAMPLES:,} steps, lr = {LR}")
    print(f"  Trials    : {N_TRIALS} (start above wall, goal below)")

    # -------------------------------------------------------------------------
    # 1. Build base environment
    # -------------------------------------------------------------------------
    print_header("1. Environment")
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    D_os = (D_os + D_os.T) / 2          # symmetrise
    print(f"  Open-field T_os and D_os built.")

    # -------------------------------------------------------------------------
    # 2. Place wall
    # -------------------------------------------------------------------------
    wall_edges = h_wall_edges(ROWS, COLS,
                               barrier_row=BARRIER_ROW,
                               col_start=COL_START,
                               col_end=COL_END)
    T_new = place_walls(T_os, wall_edges)
    print(f"  Wall placed: {len(wall_edges)} blocked edges.")

    # -------------------------------------------------------------------------
    # 3. Shared feature basis
    # -------------------------------------------------------------------------
    print_header("2. Feature Basis")
    F, lam, ec, emc = build_features(D_os, k=K)
    recon = norm(F @ F.T - D_os, 'fro') / norm(D_os, 'fro')
    print(f"  K   = {K}")
    print(f"  ec  = {ec:.6f}   emc = {emc:.6f}")
    print(f"  ||FF^T - D_os|| / ||D_os|| = {recon:.4f}")

    # -------------------------------------------------------------------------
    # 4. Compute models
    # -------------------------------------------------------------------------
    print_header("3. Computing Models")

    # woodbury_feat — analytic, projects Woodbury update onto F
    print("\n  [woodbury_feat] computing M ...")
    M_wb = woodbury_feat(T_os, T_new, F)
    print(f"  ||M_wb - I|| = {norm(M_wb - np.eye(K)):.4f}")

    # woodbury_full — analytic, full N×N state space oracle
    print("\n  [woodbury_full] computing D_new ...")
    D_new = woodbury_full(T_os, D_os, T_new)
    print(f"  ||D_new||_F  = {norm(D_new, 'fro'):.4f}")

    # sf_td — online semi-gradient TD starting from the same F
    print(f"\n  [woodbury_td] running {N_SAMPLES:,} TD steps ...")
    M_td = woodbury_td(F, T_new, emc, lam,
                 valid_states=np.arange(ROWS * COLS),
                 grad_type="semi",
                 n_samples=N_SAMPLES,
                 lr=LR,
                 seed=SEED,
                 log_every=N_SAMPLES // 5)
    
    print(f"\n  [sr_td] running {N_SAMPLES} TD steps ...")
    D_sr_td = sr_td(T_new, emc, valid_states=np.arange(ROWS*COLS),
                    lr=LR, n_samples=N_SAMPLES, seed=SEED,
                    log_every=N_SAMPLES // 5)
    
    print(f"\n  [sr_sf] running {N_SAMPLES} TD steps ...")
    D_sr_sf = sr_sf(T_new, emc, F, valid_states=np.arange(ROWS*COLS),
                    lr=LR, n_samples=N_SAMPLES, seed=SEED,
                    log_every=N_SAMPLES // 5)

    # print(f"\n  [batch_lstq] running batch least squares ...")
    # M_blsq = batch_lstsq_M(F, T_new, ec, valid_states=np.arange(ROWS*COLS))

    # -------------------------------------------------------------------------
    # 5. Evaluate planning
    # -------------------------------------------------------------------------
    print_header("4. Planning Evaluation  (above-wall -> below-wall)")

    # Each plan_fn is a closure: (goal, start) -> (steps, path)
    plan_wb    = lambda g, s: plan_with_M(F, M_wb, T_new, g, s, max_steps=MAX_STEPS, temp=TEMP)
    plan_full  = lambda g, s: plan_with_D(D_new, T_new, g, s, max_steps=MAX_STEPS, temp=TEMP)
    plan_td    = lambda g, s: plan_with_M(F, M_td, T_new, g, s, max_steps=MAX_STEPS, temp=TEMP)
    plan_sr_td = lambda g, s: plan_with_SR(D_sr_td, T_new, g, s, max_steps=MAX_STEPS, temp=TEMP)
    plan_sr_sf = lambda g, s: plan_with_SF(D_sr_sf, F, lam, T_new, g, s, max_steps=MAX_STEPS, temp=TEMP)
    plan_rand  = lambda g, s: plan_random(T_new, g, s, SEED)

    start_states = [r * COLS + c
                    for r in range(START_ROW_MIN, START_ROW_MAX + 1)
                    for c in range(START_COL_MIN, START_COL_MAX + 1)]
    goal_states  = [r * COLS + c
                    for r in range(GOAL_ROW_MIN,  GOAL_ROW_MAX + 1)
                    for c in range(GOAL_COL_MIN,  GOAL_COL_MAX + 1)]
    eval_rng   = np.random.RandomState(SEED)
    eval_state = eval_rng.get_state()

    def _eval_main(plan_fn):
        eval_rng.set_state(eval_state)
        steps_list = []
        for _ in range(N_TRIALS):
            start = eval_rng.choice(start_states)
            goal  = eval_rng.choice(goal_states)
            st, _ = plan_fn(goal, start)
            steps_list.append(st)
        arr = np.array(steps_list)
        return arr.mean(), np.median(arr), np.mean(arr < MAX_STEPS)

    results = {}
    for name, plan_fn in [
        ("woodbury_feat", plan_wb),
        ("woodbury_full", plan_full),
        ("woodbury_td", plan_td),
        ("sr_td", plan_sr_td),
        ("sr_sf", plan_sr_sf),
        ("random", plan_rand)
    ]:
        mean, median, success = _eval_main(plan_fn)
        results[name] = dict(mean=mean, median=median, success=success)
        print_result(name, mean, median, success)

    # -------------------------------------------------------------------------
    # 6. Summary table
    # -------------------------------------------------------------------------
    print_header("5. Summary")
    print(f"  {'Model':<20s}  {'Mean':>6}  {'Median':>6}  {'Success':>7}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*7}")
    for name, r in results.items():
        print(f"  {name:<20s}  {r['mean']:6.1f}  {r['median']:6.1f}  {r['success']:7.2f}")


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
    """
    Train each TD model incrementally and evaluate every `eval_every` steps.

    For each checkpoint the planner is run for `n_eval_trials` episodes; the
    number of steps taken in each episode is stored.  Cost = -steps (every
    state has cost -1, terminal has cost 0).

    Analytic models (woodbury_feat, woodbury_full) are computed once and their
    evaluation is repeated at every checkpoint so all models share the same
    x-axis.

    Saved file: <save_dir>/learning_curves.npz
      checkpoint_steps  : (n_checkpoints,)          – TD steps at each eval
      <model_name>      : (n_checkpoints, n_trials)  – steps per trial
    """
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

    # ------------------------------------------------------------------
    # 2. Feature basis
    # ------------------------------------------------------------------
    F, lam, _, emc = build_features(D_os, k=K)

    # ------------------------------------------------------------------
    # 3. Analytic models (computed once)
    # ------------------------------------------------------------------
    print("\n  [woodbury_feat] computing analytic M ...")
    M_wb = woodbury_feat(T_os, T_new, F)

    print("  [woodbury_full] computing analytic D_new ...")
    D_new = woodbury_full(T_os, D_os, T_new)

    plan_wb   = lambda g, s: plan_with_M(F, M_wb, T_new, g, s, max_steps=eval_max_steps, temp=TEMP)
    plan_full = lambda g, s: plan_with_D(D_new,   T_new, g, s, max_steps=eval_max_steps, temp=TEMP)

    # ------------------------------------------------------------------
    # 4. Incremental training + evaluation
    # ------------------------------------------------------------------
    checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every

    # Storage: (n_checkpoints, n_eval_trials)
    records = {name: np.zeros((n_checkpoints, n_eval_trials), dtype=np.int32)
               for name in ("woodbury_feat", "woodbury_full",
                             "woodbury_td", "sr_td", "sr_sf")}

    # Running model states for warm-starting
    M_td  = None   # (k, k)  – starts as identity inside woodbury_td
    D_sr  = None   # (N, N)  – starts as identity inside sr_td
    Psi   = None   # (N, k)  – starts as zeros  inside sr_sf

    # Fixed (start, goal) pairs for evaluation — same across all checkpoints.
    # eval_rng is a separate RandomState; resetting it only fixes pair selection,
    # not the global np.random used by softmax planners.
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

        # --- sr_td ---
        D_sr = sr_td(T_new, emc,
                     valid_states=np.arange(ROWS * COLS),
                     lr=LR,
                     n_samples=eval_every,
                     seed=train_seed,
                     init_D=D_sr)
        plan_sr_td = lambda g, s, _D=D_sr: plan_with_SR(_D, T_new, g, s,
                                                          max_steps=eval_max_steps,
                                                          temp=TEMP)

        # --- sr_sf ---
        Psi = sr_sf(T_new, emc, F,
                    valid_states=np.arange(ROWS * COLS),
                    lr=LR,
                    n_samples=eval_every,
                    seed=train_seed,
                    init_Psi=Psi)
        plan_sr_sf = lambda g, s, _P=Psi: plan_with_SF(_P, F, lam, T_new, g, s,
                                                         max_steps=eval_max_steps,
                                                         temp=TEMP)

        # --- evaluate all models ---
        records["woodbury_feat"][chunk] = _eval(plan_wb)
        records["woodbury_full"][chunk] = _eval(plan_full)
        records["woodbury_td"][chunk]   = _eval(plan_td)
        records["sr_td"][chunk]         = _eval(plan_sr_td)
        records["sr_sf"][chunk]         = _eval(plan_sr_sf)

        # quick progress print
        for name in records:
            mean_steps = records[name][chunk].mean()
            print(f"    {name:<20s}  mean_steps = {mean_steps:.1f}")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    save_path = os.path.join(save_dir, "learning_curves_highlr.npz")
    np.savez(save_path,
             checkpoint_steps=checkpoint_steps,
             **records)
    print(f"\n  Saved → {save_path}")
    print(f"  Arrays: checkpoint_steps {checkpoint_steps.shape}, "
          f"each model {records['woodbury_td'].shape}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_and_save()
    else:
        main()
