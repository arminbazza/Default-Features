import os
import sys

import numpy as np
from numpy.linalg import norm

from utils import (
    get_complete_DR,
    make_four_room,
    make_absorbing,
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
K          = ROWS * COLS   # full-rank features (121); lower for speed
N_SAMPLES  = 20_000
LR         = 0.01
MAX_STEPS  = 500
TEMP       = 0.2
SEED       = 42

START = 0   # state (0, 0) — top-left corner of TL room

# ---------------------------------------------------------------------------
# Outer corners: 3 per non-TL room.
# State index = row * COLS + col.
# ---------------------------------------------------------------------------
OUTER_CORNERS = {
    'TR': [0*11+6,  0*11+10, 4*11+10],    # 6,  10,  54
    'BL': [6*11+0, 10*11+0, 10*11+4],     # 66, 110, 114
    'BR': [6*11+10, 10*11+6, 10*11+10],   # 76, 116, 120
}

ALL_GOALS    = [g for gs in OUTER_CORNERS.values() for g in gs]   # 9 goals
ROOM_OF_GOAL = {g: room for room, gs in OUTER_CORNERS.items() for g in gs}

MODEL_NAMES = ['woodbury_feat', 'woodbury_full', 'woodbury_td', 'sr_td', 'sr_sf']


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def empty_results():
    return {name: {'same': [], 'diff': []} for name in MODEL_NAMES}


# =============================================================================
# CORE: run one training goal
# =============================================================================

def run_one_training_goal(g_train, T_base):
    """
    For a single training goal g_train, compute D and F, then loop over all
    8 test goals, run every model, plan, and record steps.

    Returns dict: model_name -> {'same': [steps, ...], 'diff': [steps, ...]}
    """
    train_room = ROOM_OF_GOAL[g_train]
    test_goals = [g for g in ALL_GOALS if g != g_train]

    # ---- build training representation ----
    T_train  = make_absorbing(T_base, g_train)
    D_train  = get_complete_DR(T_train)
    # D_train  = (D_train + D_train.T) / 2          # symmetrise
    F, lam, ec, emc = build_features(D_train, k=K)

    print(f"  Training goal : state {g_train}  room {train_room}")
    print(f"  ec = {ec:.5f}   emc = {emc:.5f}")
    print(f"  ||FF^T - D_train||/||D_train|| = "
          f"{norm(F @ F.T - D_train,'fro') / norm(D_train,'fro'):.4f}")

    # States with non-zero rows in T_base (excludes wall states permanently)
    base_valid = np.array([s for s in range(ROWS * COLS) if T_base[s].sum() > 0])

    results = empty_results()

    for i, g_test in enumerate(test_goals):
        same_room = (ROOM_OF_GOAL[g_test] == train_room)
        key       = 'same' if same_room else 'diff'
        tag       = f"{'same' if same_room else 'diff'}-room"
        print(f"\n  [{i+1}/8] test goal {g_test} ({ROOM_OF_GOAL[g_test]}, {tag})")

        # T_new: g_test absorbing, g_train restored to normal
        T_new = make_absorbing(T_base, g_test)

        # Valid states for TD: exclude wall states and the absorbing goal state
        valid_td = base_valid[base_valid != g_test]

        # ---- model-based (analytic) ----
        M_wb = woodbury_feat(T_train, T_new, F)
        D_wb = woodbury_full(T_train, D_train, T_new)

        # ---- model-free (online TD) ----
        print("        woodbury_td ...", end=' ', flush=True)
        M_td = woodbury_td(F, T_new, emc, lam,
                           valid_states=valid_td,
                           grad_type='semi',
                           n_samples=N_SAMPLES, lr=LR, seed=SEED)
        print("done")

        print("        sr_td       ...", end=' ', flush=True)
        D_sr_td_val = sr_td(T_new, emc,
                            valid_states=valid_td,
                            n_samples=N_SAMPLES, lr=LR, seed=SEED)
        print("done")

        print("        sr_sf       ...", end=' ', flush=True)
        D_sr_sf_val = sr_sf(T_new, emc, F,
                            valid_states=valid_td,
                            n_samples=N_SAMPLES, lr=LR, seed=SEED)
        print("done")

        # ---- plan from START to g_test ----
        steps_wb,  _ = plan_with_M(F, M_wb, T_new, g_test, START, max_steps=MAX_STEPS, temp=TEMP)
        steps_full,_ = plan_with_D(D_wb, T_new, g_test, START, max_steps=MAX_STEPS, temp=TEMP)
        steps_td,  _ = plan_with_M(F, M_td, T_new, g_test, START, max_steps=MAX_STEPS, temp=TEMP)
        steps_srtd,_ = plan_with_SR(D_sr_td_val, T_new, g_test, START, max_steps=MAX_STEPS, temp=TEMP)
        steps_srsf,_ = plan_with_SF(D_sr_sf_val, F, lam, T_new, g_test, START, max_steps=MAX_STEPS, temp=TEMP)

        step_vals = [steps_wb, steps_full, steps_td, steps_srtd, steps_srsf]
        for name, s in zip(MODEL_NAMES, step_vals):
            results[name][key].append(s)

        print(f"        steps — wb:{steps_wb}  full:{steps_full}  "
              f"td:{steps_td}  sr_td:{steps_srtd}  sr_sf:{steps_srsf}")

    return results


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_summary(results_by_train_goal):
    """
    Aggregate over all training goals and print a table:
      model | same-room mean | diff-room mean
    """
    print_header("RESULTS SUMMARY")
    print(f"  {'Model':<20s}  {'Same-room':>12}  {'Diff-room':>12}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}")

    for name in MODEL_NAMES:
        same_all, diff_all = [], []
        for res in results_by_train_goal.values():
            same_all.extend(res[name]['same'])
            diff_all.extend(res[name]['diff'])

        same_mean = np.mean(same_all) if same_all else float('nan')
        diff_mean = np.mean(diff_all) if diff_all else float('nan')
        print(f"  {name:<20s}  {same_mean:>12.1f}  {diff_mean:>12.1f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    np.random.seed(SEED)

    print_header("FOUR-ROOM PLANNING COMPARISON")
    print(f"  Grid      : {ROWS}x{COLS} = {ROWS*COLS} states")
    print(f"  Start     : state {START} = (0, 0)  [TL room]")
    print(f"  Goals     : {ALL_GOALS}  ({len(ALL_GOALS)} total)")
    print(f"  K         : {K}")
    print(f"  N_SAMPLES : {N_SAMPLES:,}   LR : {LR}   TEMP : {TEMP}")

    # Build the base four-room environment (no absorbing state yet)
    T_base, wall_states = make_four_room(ROWS, COLS)
    print(f"  Wall cells: {len(wall_states)}")
    print(f"  Doorways  : (2,5)=27  (5,3)=58  (9,5)=104  (5,8)=63")

    # -------------------------------------------------------------------------
    # Step 8: run ONE training goal first to verify correctness
    # -------------------------------------------------------------------------
    g_train_test = ALL_GOALS[0]   # state 6 = (0,6), top-left corner of TR room

    print_header(f"Single-goal test run  [g_train = {g_train_test}]")
    res_one = run_one_training_goal(g_train_test, T_base)

    # Print per-test-goal breakdown
    print_header("Per-goal results")
    test_goals_ordered = [g for g in ALL_GOALS if g != g_train_test]
    same_idx = {n: 0 for n in MODEL_NAMES}
    diff_idx = {n: 0 for n in MODEL_NAMES}
    print(f"  {'Goal':>5}  {'Room':>4}  {'Tag':>4}  "
          + "  ".join(f"{n[:8]:>8}" for n in MODEL_NAMES))
    for g_test in test_goals_ordered:
        tag = 'same' if ROOM_OF_GOAL[g_test] == ROOM_OF_GOAL[g_train_test] else 'diff'
        row_vals = []
        for n in MODEL_NAMES:
            if tag == 'same':
                v = res_one[n]['same'][same_idx[n]]
                same_idx[n] += 1
            else:
                v = res_one[n]['diff'][diff_idx[n]]
                diff_idx[n] += 1
            row_vals.append(v)
        print(f"  {g_test:>5}  {ROOM_OF_GOAL[g_test]:>4}  {tag:>4}  "
              + "  ".join(f"{v:>8}" for v in row_vals))

    # Aggregate same vs diff for this one training goal
    print_header("Aggregate (one training goal)")
    print(f"  {'Model':<20s}  {'Same-room (2)':>14}  {'Diff-room (6)':>14}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*14}")
    for name in MODEL_NAMES:
        same_mean = np.mean(res_one[name]['same']) if res_one[name]['same'] else float('nan')
        diff_mean = np.mean(res_one[name]['diff']) if res_one[name]['diff'] else float('nan')
        print(f"  {name:<20s}  {same_mean:>14.1f}  {diff_mean:>14.1f}")

# =============================================================================
# LEARNING-CURVE DATA COLLECTION
# =============================================================================

def train_and_save(
    n_train_steps:  int = 20_000,
    eval_every:     int = 500,
    n_eval_trials:  int = 20,
    eval_max_steps: int = 1_000,
    save_dir:       str = "../data/planning_4room/goal/",
    train_all:      bool = False,
):
    """
    For every (g_train, g_test) pair, train each TD model incrementally and
    evaluate every eval_every steps.  Start is always START (state 0);
    goal is always g_test (fixed corner).

    Each evaluation runs n_eval_trials stochastic planning episodes with a
    fixed seed, so results are reproducible and comparable across checkpoints.

    Saved file: <save_dir>/learning_curves.npz
      checkpoint_steps        : (n_checkpoints,)
      same_pairs / diff_pairs : (n_pairs, 2)  — [g_train, g_test] index
      <model>_same            : (n_same_pairs, n_checkpoints, n_eval_trials)
      <model>_diff            : (n_diff_pairs, n_checkpoints, n_eval_trials)
    """
    os.makedirs(save_dir, exist_ok=True)
    n_checkpoints = n_train_steps // eval_every

    # ------------------------------------------------------------------
    # 1. Base environment
    # ------------------------------------------------------------------
    print_header("TRAIN & SAVE  —  four-room goal-change")
    T_base, wall_states = make_four_room(ROWS, COLS)
    base_valid = np.array([s for s in range(ROWS * COLS) if T_base[s].sum() > 0])
    print(f"  Grid {ROWS}x{COLS},  {len(wall_states)} wall states,  "
          f"{n_checkpoints} checkpoints x {eval_every:,} steps")

    # ------------------------------------------------------------------
    # 2. Build ordered lists of same / diff pairs
    # ------------------------------------------------------------------
    # state 10 = (row 0, col 10) — top-right corner of TR room
    train_goals = ALL_GOALS if train_all else [10]

    same_pairs, diff_pairs = [], []
    for g_train in train_goals:
        for g_test in ALL_GOALS:
            if g_test == g_train:
                continue
            if ROOM_OF_GOAL[g_test] == ROOM_OF_GOAL[g_train]:
                same_pairs.append((g_train, g_test))
            else:
                diff_pairs.append((g_train, g_test))

    n_same = len(same_pairs)   # 9 train x 2 same-room test = 18
    n_diff = len(diff_pairs)   # 9 train x 6 diff-room test = 54
    print(f"  Pairs: {n_same} same-room,  {n_diff} diff-room")

    checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every

    # Storage: (n_pairs, n_checkpoints, n_eval_trials)
    rec_same = {m: np.zeros((n_same, n_checkpoints, n_eval_trials), dtype=np.int32)
                for m in MODEL_NAMES}
    rec_diff = {m: np.zeros((n_diff, n_checkpoints, n_eval_trials), dtype=np.int32)
                for m in MODEL_NAMES}

    # ------------------------------------------------------------------
    # 3. Evaluation helper — fixed seed so every checkpoint is comparable
    # ------------------------------------------------------------------
    def _eval(plan_fn, g_test):
        np.random.seed(SEED)   # reset global state: same n_eval_trials each call
        steps_arr = np.zeros(n_eval_trials, dtype=np.int32)
        for i in range(n_eval_trials):
            st, _ = plan_fn(g_test, START)
            steps_arr[i] = st
        return steps_arr

    # ------------------------------------------------------------------
    # 4. Main loop: g_train -> g_test -> chunks
    # ------------------------------------------------------------------
    same_idx = 0
    diff_idx = 0

    for ti, g_train in enumerate(train_goals):
        train_room = ROOM_OF_GOAL[g_train]
        print_header(f"Training goal {ti+1}/9: state {g_train}  ({train_room})")
        
        T_os = create_transition_open(shape=(ROWS, COLS))
        D_os = get_complete_DR(T=T_os)

        T_train = make_absorbing(T_base, g_train)
        D_train = get_complete_DR(T_train)
        # F, lam, _, emc = build_features(D_train, k=K)
        F, lam, _, emc = build_features(D_os, k=K)

        test_goals = [g for g in ALL_GOALS if g != g_train]

        for g_test in test_goals:
            same_room = (ROOM_OF_GOAL[g_test] == train_room)
            pair_idx  = same_idx if same_room else diff_idx
            rec       = rec_same if same_room else rec_diff
            tag       = "same" if same_room else "diff"

            T_new    = make_absorbing(T_base, g_test)
            valid_td = base_valid[base_valid != g_test]

            # Analytic models (computed once per pair, constant across chunks)
            M_wb  = woodbury_feat(T_os, T_new, F)
            D_wb  = woodbury_full(T_train, D_train, T_new)
            plan_wb   = lambda g, s, _M=M_wb: plan_with_M(F, _M, T_new, g, s,
                                                           max_steps=eval_max_steps, temp=TEMP)
            plan_full = lambda g, s, _D=D_wb: plan_with_D(_D, T_new, g, s,
                                                           max_steps=eval_max_steps, temp=TEMP)

            # TD warm-start states (reset for each new pair)
            M_td = None
            D_sr = None
            Psi  = None

            print(f"  g_test ={g_test:3d} ({ROOM_OF_GOAL[g_test]}, {tag})")

            for chunk in range(n_checkpoints):
                train_seed = SEED + chunk * 1009

                M_td = woodbury_td(F, T_new, emc, lam,
                                   valid_states=valid_td, grad_type='semi',
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

                rec['woodbury_feat'][pair_idx, chunk] = _eval(plan_wb,     g_test)
                rec['woodbury_full'][pair_idx, chunk] = _eval(plan_full,   g_test)
                rec['woodbury_td'  ][pair_idx, chunk] = _eval(plan_td,     g_test)
                rec['sr_td'        ][pair_idx, chunk] = _eval(plan_sr_td,  g_test)
                rec['sr_sf'        ][pair_idx, chunk] = _eval(plan_sr_sf,  g_test)

                if (chunk + 1) % 5 == 0:
                    mean_full = rec[f"woodbury_full"][pair_idx, chunk].mean()
                    mean_feat = rec[f"woodbury_feat"][pair_idx, chunk].mean()
                    mean_td = rec[f"woodbury_td"  ][pair_idx, chunk].mean()
                    print(f"     chunk {chunk+1:>3}/{n_checkpoints}  "
                        f"full={mean_full:.1f}  feat={mean_feat:.1f}  td={mean_td:.1f}")

            mean_wb = rec['woodbury_feat'][pair_idx, -1].mean()
            mean_td = rec['woodbury_td'  ][pair_idx, -1].mean()
            print(f"  wb={mean_wb:.0f}  td={mean_td:.0f}")

            if same_room:
                same_idx += 1
            else:
                diff_idx += 1

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    save_path = os.path.join(save_dir, "learning_curves_highlr_newsf.npz")
    save_kwargs = {
        'checkpoint_steps': checkpoint_steps,
        'same_pairs':       np.array(same_pairs),
        'diff_pairs':       np.array(diff_pairs),
    }
    for m in MODEL_NAMES:
        save_kwargs[f'{m}_same'] = rec_same[m]
        save_kwargs[f'{m}_diff'] = rec_diff[m]

    np.savez(save_path, **save_kwargs)
    print(f"\n  Saved -> {save_path}")
    print(f"  same arrays : {rec_same['woodbury_td'].shape}  "
          f"(n_same_pairs, n_checkpoints, n_eval_trials)")
    print(f"  diff arrays : {rec_diff['woodbury_td'].shape}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_and_save()
    else:
        main()
