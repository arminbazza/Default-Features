"""
Four-room planning with a fully blocked (inaccessible) doorway state.

Environment: 11x11 four-room grid (same layout as plan_4room_wall.py).
Start: state 0 = (0,0) — top-left corner of TL room (fixed).

Protocol (per scenario):
  1. Make goal absorbing → T_train.  Compute D_train, F, lam, ec, emc.
  2. Remove the doorway state entirely → T_new.
     (place_barrier zeros the entire row AND column of the doorway state,
      making it completely inaccessible from all neighbouring states.)
  3. Compute all 5 models using T_train as 'old' and T_new as 'new'.
  4. Plan from START to goal under T_new; record steps.

Scenarios
---------
  BL : goal = 110 = (10,0)  [bottom-left of BL room]
       blocks state (5,3)=58  — TL <-> BL doorway

  TR : goal = 10  = (0,10)  [top-right of TR room]
       blocks state (2,5)=27  — TL <-> TR doorway

Only BL is active by default.  Change ACTIVE in main() to run TR or both.

Run from repo root: python src/plan_4room_blocked.py
"""

import os
import sys

import numpy as np
from numpy.linalg import norm

from utils import (
    get_complete_DR,
    make_four_room,
    make_absorbing,
    place_barrier,
    place_walls,
    plan_with_M,
    plan_with_D,
    plan_with_SR,
    plan_with_SF,
    create_transition_open,
)
from models import build_features, woodbury_feat, woodbury_full, woodbury_td, sr_td, sr_sf

# =============================================================================
# CONFIG
# =============================================================================
ROWS, COLS = 11, 11
K          = ROWS * COLS
N_SAMPLES  = 800_000
LR         = 0.01
MAX_STEPS  = 500
TEMP       = 0.2
SEED       = 42

START = 0   # (0, 0) — TL corner

MODEL_NAMES = ['woodbury_feat', 'woodbury_full', 'woodbury_td', 'sr_td', 'sr_sf']

# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================
# Doorway states (completely removed in each scenario):
#   TL <-> BL doorway : state (5,3) = 58
#   TL <-> TR doorway : state (2,5) = 27

SCENARIOS = {
    'BL': {
        'goal':          10 * 11 + 0,    # state 110 = (10, 0)
        'barrier_state': 5  * 11 + 3,    # state  58 = (5,  3)  TL<->BL doorway
        'label':         'BL room — goal (10,0)=110, block state (5,3)=58',
    },
    'TR': {
        'goal':          0  * 11 + 10,   # state  10 = (0, 10)
        'barrier_state': 2  * 11 + 5,    # state  27 = (2,  5)  TL<->TR doorway
        'label':         'TR room — goal (0,10)=10,  block state (2,5)=27',
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# =============================================================================
# CORE: run one scenario
# =============================================================================

def run_scenario(scenario_key, T_base):
    """
    Full pipeline for a single blocked-state scenario:
      - build T_train (goal absorbing) and derive features from D_train
      - remove doorway state entirely → T_new
      - compute all 5 models
      - plan from START to goal under T_new
      - return {model_name: steps}
    """
    cfg           = SCENARIOS[scenario_key]
    goal          = cfg['goal']
    barrier_state = cfg['barrier_state']

    print_header(f"Scenario {scenario_key}  :  {cfg['label']}")

    # ------------------------------------------------------------------
    # Step 1: Training representation (goal absorbing, doorway still open)
    # ------------------------------------------------------------------
    T_train = make_absorbing(T_base, goal)
    D_train = get_complete_DR(T_train)
    F, lam, ec, emc = build_features(D_train, k=K)

    print(f"  Goal          : state {goal}")
    print(f"  Blocked state : state {barrier_state}")
    print(f"  ec = {ec:.5f}   emc = {emc:.5f}")
    print(f"  ||FF^T - D_train|| / ||D_train|| = "
          f"{norm(F @ F.T - D_train, 'fro') / norm(D_train, 'fro'):.4f}")

    # ------------------------------------------------------------------
    # Step 2: Remove doorway state entirely → T_new
    # ------------------------------------------------------------------
    T_new = place_barrier(T_train, [barrier_state])

    valid_td = np.array([s for s in range(ROWS * COLS) if T_new[s].sum() > 0])
    print(f"\n  Blocked state   : {barrier_state}  (row and column zeroed)")
    print(f"  TD valid states : {len(valid_td)}")

    # ------------------------------------------------------------------
    # Step 3: Compute all 5 models
    # ------------------------------------------------------------------
    print()
    M_wb = woodbury_feat(T_train, T_new, F)
    D_wb = woodbury_full(T_train, D_train, T_new)

    print("  woodbury_td ...", end=' ', flush=True)
    M_td = woodbury_td(F, T_new, emc, lam,
                       valid_states=valid_td,
                       grad_type='semi',
                       n_samples=N_SAMPLES, lr=LR, seed=SEED)
    print("done")

    print("  sr_td       ...", end=' ', flush=True)
    D_sr_td = sr_td(T_new, emc,
                    valid_states=valid_td,
                    n_samples=N_SAMPLES, lr=LR, seed=SEED)
    print("done")

    print("  sr_sf       ...", end=' ', flush=True)
    D_sr_sf = sr_sf(T_new, emc, F,
                    valid_states=valid_td,
                    n_samples=N_SAMPLES, lr=LR, seed=SEED)
    print("done")

    # ------------------------------------------------------------------
    # Step 4: Plan from start to goal under T_new
    # ------------------------------------------------------------------
    print()
    steps_wb,   _ = plan_with_M(F, M_wb, T_new, goal, START, max_steps=MAX_STEPS, temp=TEMP)
    steps_full, _ = plan_with_D(D_wb, T_new, goal, START, max_steps=MAX_STEPS, temp=TEMP)
    steps_td,   _ = plan_with_M(F, M_td, T_new, goal, START, max_steps=MAX_STEPS, temp=TEMP)
    steps_srtd, _ = plan_with_SR(D_sr_td, T_new, goal, START, max_steps=MAX_STEPS, temp=TEMP)
    steps_srsf, _ = plan_with_SF(D_sr_sf, F, lam, T_new, goal, START, max_steps=MAX_STEPS, temp=TEMP)

    step_vals = [steps_wb, steps_full, steps_td, steps_srtd, steps_srsf]

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print_header(f"Results — scenario {scenario_key}")
    print(f"  {'Model':<20s}  {'Steps':>6}")
    print(f"  {'-'*20}  {'-'*6}")
    for name, s in zip(MODEL_NAMES, step_vals):
        flag = "  [MAX]" if s == MAX_STEPS else ""
        print(f"  {name:<20s}  {s:>6}{flag}")

    return dict(zip(MODEL_NAMES, step_vals))


# =============================================================================
# MAIN
# =============================================================================

def main():
    np.random.seed(SEED)

    print_header("FOUR-ROOM: BLOCKED DOORWAY STATE PLANNING")
    print(f"  Grid      : {ROWS}x{COLS} = {ROWS*COLS} states")
    print(f"  Start     : state {START} = (0,0)  [TL room]")
    print(f"  K         : {K}")
    print(f"  N_SAMPLES : {N_SAMPLES:,}   LR : {LR}   TEMP : {TEMP}")

    T_base, wall_states = make_four_room(ROWS, COLS)
    print(f"  Wall cells: {len(wall_states)}")
    print(f"  Doorways  : (2,5)=27  (5,3)=58  (9,5)=104  (5,8)=63")

    # ---- Active scenarios: change to ['TR'], or ['BL', 'TR'] to run both ----
    ACTIVE = ['BL']

    all_results = {}
    for key in ACTIVE:
        all_results[key] = run_scenario(key, T_base)

    if len(all_results) > 1:
        print_header("CROSS-SCENARIO SUMMARY")
        print(f"  {'Model':<20s}  " + "  ".join(f"{k:>6}" for k in all_results))
        print(f"  {'-'*20}  " + "  ".join(["-"*6] * len(all_results)))
        for name in MODEL_NAMES:
            row = "  ".join(f"{all_results[k][name]:>6}" for k in all_results)
            print(f"  {name:<20s}  {row}")


# =============================================================================
# LEARNING-CURVE DATA COLLECTION
# =============================================================================

def train_and_save(
    n_train_steps:  int = 150_000,# 800_000,
    eval_every:     int = 2_500,# 10_000,
    n_eval_trials:  int = 20,
    eval_max_steps: int = 1_000,
    save_dir:       str = "../data/planning_4room/blocked/",
):
    """
    For each scenario (BL, TR), train each TD model incrementally and
    evaluate every eval_every steps.  Start is always START (state 0);
    goal is the fixed scenario goal.

    Saved file: <save_dir>/learning_curves.npz
      checkpoint_steps      : (n_checkpoints,)
      <model>_<scenario_key>: (n_checkpoints, n_eval_trials)
    """
    os.makedirs(save_dir, exist_ok=True)
    n_checkpoints = n_train_steps // eval_every
    checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every

    print_header("TRAIN & SAVE  —  four-room blocked doorway state")
    T_base, wall_states = make_four_room(ROWS, COLS)
    print(f"  Grid {ROWS}x{COLS},  {len(wall_states)} wall states,  "
          f"{n_checkpoints} checkpoints x {eval_every:,} steps")

    # Storage: (n_checkpoints, n_eval_trials) per model per scenario
    records = {
        f"{m}_{k}": np.zeros((n_checkpoints, n_eval_trials), dtype=np.int32)
        for m in MODEL_NAMES
        for k in SCENARIOS
    }

    # ------------------------------------------------------------------
    # Evaluation helper — fixed seed so every checkpoint is comparable
    # ------------------------------------------------------------------
    def _eval(plan_fn, goal):
        np.random.seed(SEED)
        steps_arr = np.zeros(n_eval_trials, dtype=np.int32)
        for i in range(n_eval_trials):
            st, _ = plan_fn(goal, START)
            steps_arr[i] = st
        return steps_arr

    # ------------------------------------------------------------------
    # Main loop: scenario -> chunks
    # ------------------------------------------------------------------
    for scenario_key, cfg in SCENARIOS.items():
        goal          = cfg['goal']
        barrier_state = cfg['barrier_state']
        print_header(f"Scenario {scenario_key}  :  {cfg['label']}")
        
        T_os = create_transition_open(shape=(ROWS, COLS))
        D_os = get_complete_DR(T=T_os)

        T_train = make_absorbing(T_base, goal)
        D_train = get_complete_DR(T_train)
        F, lam, _, emc = build_features(D_os, k=K)

        # Remove doorway state entirely
        T_new    = place_barrier(T_train, [barrier_state])
        valid_td = np.array([s for s in range(ROWS * COLS) if T_new[s].sum() > 0])
        
        # Payam suggestion
        M_init = woodbury_feat(T_os, T_train, F)
        M_wb = woodbury_feat(T_os, T_new, F)

        # Analytic models (computed once, constant across chunks)
        # M_wb = woodbury_feat(T_train, T_new, F)
        D_wb = woodbury_full(T_train, D_train, T_new)
        plan_wb   = lambda g, s, _M=M_wb: plan_with_M(F, _M, T_new, g, s,
                                                       max_steps=eval_max_steps, temp=TEMP)
        plan_full = lambda g, s, _D=D_wb: plan_with_D(_D, T_new, g, s,
                                                       max_steps=eval_max_steps, temp=TEMP)

        # TD warm-start states (reset per scenario)
        # M_td = None
        M_td = M_init
        D_sr = None
        Psi  = None

        print(f"  goal={goal}  barrier_state={barrier_state}  valid_td={len(valid_td)}",
              flush=True)

        for chunk in range(n_checkpoints):
            train_seed = SEED + chunk * 1009

            M_td = woodbury_td(F, T_new, emc, lam,
                               valid_states=valid_td, grad_type='full',
                               n_samples=eval_every, lr=LR,
                               seed=train_seed, init_M=M_td)
            D_sr = sr_td(T_new, emc, valid_states=valid_td,
                         n_samples=eval_every, lr=LR,
                         seed=train_seed, init_D=D_sr)
            Psi  = sr_sf(T_new, emc, F, valid_states=valid_td,
                         n_samples=eval_every, lr=LR,
                         seed=train_seed, init_Psi=Psi)

            plan_td    = lambda g, s, _M=M_td: plan_with_M(F, _M, T_new, g, s,
                                                            max_steps=eval_max_steps, temp=TEMP)
            plan_sr_td = lambda g, s, _D=D_sr: plan_with_SR(_D, T_new, g, s,
                                                              max_steps=eval_max_steps, temp=TEMP)
            plan_sr_sf = lambda g, s, _P=Psi:  plan_with_SF(_P, F, lam, T_new, g, s,
                                                              max_steps=eval_max_steps, temp=TEMP)

            records[f"woodbury_feat_{scenario_key}"][chunk] = _eval(plan_wb,     goal)
            records[f"woodbury_full_{scenario_key}"][chunk] = _eval(plan_full,   goal)
            records[f"woodbury_td_{scenario_key}"  ][chunk] = _eval(plan_td,     goal)
            records[f"sr_td_{scenario_key}"        ][chunk] = _eval(plan_sr_td,  goal)
            records[f"sr_sf_{scenario_key}"        ][chunk] = _eval(plan_sr_sf,  goal)

            if (chunk + 1) % 5 == 0:
                mean_full = records[f"woodbury_full_{scenario_key}"][chunk].mean()
                mean_feat = records[f"woodbury_feat_{scenario_key}"][chunk].mean()
                mean_td = records[f"woodbury_td_{scenario_key}"  ][chunk].mean()
                print(f"  chunk {chunk+1:>3}/{n_checkpoints}  "
                      f"full={mean_full:.1f}  feat={mean_feat:.1f}  td={mean_td:.1f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_path = os.path.join(save_dir, "learning_curves_highlr.npz")
    np.savez(save_path, checkpoint_steps=checkpoint_steps, **records)
    print(f"\n  Saved -> {save_path}")
    print(f"  Arrays per model/scenario: {(n_checkpoints, n_eval_trials)}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_and_save()
    else:
        #main()
        goal = 10 * 11 + 0
        # Step 1: F from open space (symmetric, clean)
        T_open = create_transition_open(shape=(ROWS, COLS))
        D_open = get_complete_DR(T_open)
        # D_open = (D_open + D_open.T) / 2
        F, lam, ec, emc = build_features(D_open, k=K)
        print(f"||FF^T - D_open|| / ||D_open|| = {norm(F @ F.T - D_open) / norm(D_open):.2e}")

        # Step 2: Analytical M_init from open -> four-room (with absorbing goal)
        T_base, wall_states = make_four_room(ROWS, COLS)
        T_train = make_absorbing(T_base, goal)
        M_init = woodbury_feat(T_open, T_train, F)

        # Check: can we plan in four-room with M_init?
        steps_init, _ = plan_with_M(F, M_init, T_train, goal, START, max_steps=500)
        print(f"M_init planning (four-room, no extra wall): {steps_init} steps")
        print(f"||M_init - I|| = {norm(M_init - np.eye(K)):.4f}")
        
        # Step 3: Add doorway wall
        barrier_state = 5  * 11 + 3
        T_new = place_barrier(T_train, [barrier_state])
        
        # wall_edges = [(4*11 + 3, 5*11 + 3), (5*11 + 3, 6*11 + 3)]
        # T_new = place_walls(T_train, wall_edges)

        # Step 4: Learn M_new via TD, starting from M_init
        valid_td = np.array([s for s in range(ROWS * COLS) if T_new[s].sum() > 0])
        lam_inv = 1.0 / lam[:K]

        M_new = M_init.copy()
        rng = np.random.RandomState(SEED)
        N = F.shape[0]

        for t in range(N_SAMPLES):
            s = rng.choice(valid_td)
            sp = rng.choice(N, p=T_new[s])
            f_s = F[s]
            f_sp = F[sp]
            r_s = f_s * lam_inv
            delta = emc * (r_s + f_sp @ M_new) - f_s @ M_new
            M_new += LR * np.outer(f_s, delta)
            # grad = 2.0 * np.outer(f_sp - 1/emc * f_s, delta)
            # M_new -= LR * grad

        # Step 5: Evaluate
        steps_new, _ = plan_with_M(F, M_new, T_new, goal, START, max_steps=500)
        print(f"M_new planning (four-room + wall): {steps_new} steps")
        print(f"||M_new - M_init|| = {norm(M_new - M_init):.4f}")

        # Does the analytical M for the FULL perturbation (open -> four-room + wall) work?
        M_direct = woodbury_feat(T_open, T_new, F)
        steps_direct, _ = plan_with_M(F, M_direct, T_new, goal, START, max_steps=500)
        print(f"M_direct (open -> four-room+wall): {steps_direct} steps")
        print(f"||M_direct - M_init|| = {norm(M_direct - M_init):.4f}")

        # Is TD moving toward or away from M_direct?
        print(f"||M_new - M_direct|| = {norm(M_new - M_direct):.4f}")
        print(f"||M_init - M_direct|| = {norm(M_init - M_direct):.4f}")
        
        D_new_exact = get_complete_DR(T_new)
        steps_exact, _ = plan_with_D(D_new_exact, T_new, goal, START, max_steps=500)
        print(f"Exact D_new: {steps_exact} steps")
        
        D_from_M = F @ M_direct @ F.T

        print(f"||F M F^T - D_new_exact|| / ||D_new|| = "
            f"{norm(D_from_M - D_new_exact) / norm(D_new_exact):.6f}")

        # Also check the condition number of the Woodbury inverse
        diffs = T_open != T_new
        delta_locs = np.unique(np.where(diffs)[0])
        R = (T_new - T_open)[delta_locs, :]
        C = np.eye(N)[:, delta_locs]
        H = R @ F
        G = F.T @ C
        Z = H @ G
        print(f"B = {len(delta_locs)}")
        print(f"cond(I - Z) = {np.linalg.cond(np.eye(len(delta_locs)) - Z):.2e}")
        print(f"min eigenvalue of (I - Z) = {np.min(np.abs(np.linalg.eigvals(np.eye(len(delta_locs)) - Z))):.2e}")