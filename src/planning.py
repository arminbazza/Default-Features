import numpy as np
from numpy.linalg import norm

from utils import (
    create_transition_open,
    get_complete_DR,
    place_walls,
    h_wall_edges,
    plan_with_M,
    plan_with_D,
    evaluate_wall_crossing,
)
from models import build_features, woodbury_feat, woodbury_full, woodbury_td, batch_lstsq_M

# =============================================================================
# CONFIG
# =============================================================================
ROWS, COLS = 15, 15
# K          = ROWS * COLS
K          = 200
N_SAMPLES  = 800_000
LR         = 0.001
N_TRIALS   = 300
MAX_STEPS  = 500
SEED       = 42

# Wall geometry: H wall between row (BARRIER_ROW-1) and BARRIER_ROW,
# spanning columns [COL_START, COL_END).
# Gaps are left at columns 0..COL_START-1 and COL_END..COLS-1.
BARRIER_ROW = ROWS // 2        # 7  (wall between rows 6 and 7)
COL_START   = COLS // 4        # 3
COL_END     = 3 * COLS // 4   # 11


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result(name, mean, median, success):
    print(f"  {name:<20s}  mean={mean:6.1f}  median={median:5.1f}  success={success:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
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
    M_sf = woodbury_td(F, T_new, emc, lam,
                 valid_states=np.arange(ROWS * COLS),
                 grad_type="full",
                 n_samples=N_SAMPLES,
                 lr=LR,
                 seed=SEED,
                 log_every=N_SAMPLES // 5)
    
    print(f"\n  [batch_lstq] running batch least squares ...")
    M_blsq = batch_lstsq_M(F, T_new, ec, valid_states=np.arange(ROWS*COLS))

    # -------------------------------------------------------------------------
    # 5. Evaluate planning
    # -------------------------------------------------------------------------
    print_header("4. Planning Evaluation  (above-wall -> below-wall)")

    # Each plan_fn is a closure: (goal, start) -> (steps, path)
    plan_wb   = lambda g, s: plan_with_M(F, M_wb, T_new, g, s, MAX_STEPS)
    plan_full = lambda g, s: plan_with_D(D_new, T_new, g, s, MAX_STEPS)
    plan_td   = lambda g, s: plan_with_M(F, M_sf, T_new, g, s, MAX_STEPS)
    plan_blsq = lambda g, s: plan_with_M(F, M_blsq, T_new, g, s, MAX_STEPS)

    results = {}
    for name, plan_fn in [
        ("woodbury_feat", plan_wb),
        ("woodbury_full", plan_full),
        ("woodbury_td", plan_td),
        ("batch_lstq", plan_blsq)
    ]:
        mean, median, success = evaluate_wall_crossing(
            plan_fn,
            barrier_row=BARRIER_ROW,
            rows=ROWS,
            cols=COLS,
            n_trials=N_TRIALS,
            max_steps=MAX_STEPS,
            seed=SEED,
        )
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


if __name__ == '__main__':
    main()
