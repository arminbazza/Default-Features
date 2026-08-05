import argparse
import os
import time

import numpy as np

from utils import (
    create_transition_open,
    get_complete_DR,
    place_walls,
    h_wall_edges,
    v_wall_edges,
    neighbour_table,
    values_M,
    print_header,
    draw_starts,
    spawn_train_seeds,
    evaluate_starts,
    init_records,
    load_resume,
    save_records,
    report_multiseed,
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
R_GOAL     = 10
DQ_R_GOAL  = 10
SEED       = 42

CORNER_ROW = 5
CORNER_COL = 5
H_END_COL  = 12
V_END_ROW  = 12

START_ROW_MAX = CORNER_ROW - 2
START_COL_MAX = CORNER_COL - 2
GOAL_ROW_MIN  = 6
GOAL_COL_MIN  = 6
GOAL_ROW_MAX  = 9
GOAL_COL_MAX  = 9

K_PLANNING_VALS = [5, 10, 20, 50]

N_RUNS            = 5
PAIR_SEED         = 20_250_803
ROLLOUT_SEED_BASE = 960_000

DEFAULT_OUT = "../data/planning_open/angle_mc/"
FNAME       = "learning_curves_multiseed.npz"


def train_and_save(n_runs=N_RUNS,
                   n_train_steps=50_000,
                   eval_every=1_000,
                   n_eval_trials=200,
                   eval_max_steps=1_000,
                   save_dir=DEFAULT_OUT,
                   fresh=False):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, FNAME)
    n_checkpoints = n_train_steps // eval_every
    checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every

    print_header("TRAIN & SAVE (multi-seed)  --  L-barrier vs Dyna-Q")
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    h_edges = h_wall_edges(ROWS, COLS, barrier_row=CORNER_ROW,
                           col_start=CORNER_COL, col_end=H_END_COL)
    v_edges = v_wall_edges(ROWS, COLS, barrier_col=CORNER_COL,
                           row_start=CORNER_ROW, row_end=V_END_ROW)
    T_new = place_walls(T_os, h_edges + v_edges)
    nbrs = neighbour_table(T_new)

    F, lam, _, emc = build_features(D_os, k=K)

    fixed_goal = ((GOAL_ROW_MIN + GOAL_ROW_MAX) // 2) * COLS \
        + (GOAL_COL_MIN + GOAL_COL_MAX) // 2

    start_states = [r * COLS + c
                    for r in range(START_ROW_MAX + 1)
                    for c in range(START_COL_MAX + 1)]
    starts = draw_starts(start_states, n_eval_trials, PAIR_SEED)

    print(f"  runs        : {n_runs}")
    print(f"  L-wall      : {len(h_edges)} H + {len(v_edges)} V edges")
    print(f"  fixed goal  : state {fixed_goal}")
    print(f"  train steps : {n_train_steps:,} (eval every {eval_every:,})")
    print(f"  starts      : seed {PAIR_SEED}, {len(set(starts.tolist()))} unique "
          f"of {n_eval_trials} draws")

    train_seeds = spawn_train_seeds(SEED, n_runs, n_checkpoints)

    dq_names = [f"dyna_q_k{k}" for k in K_PLANNING_VALS]
    model_names = ["woodbury_feat", "woodbury_td"] + dq_names
    records = init_records(model_names, (n_runs, n_checkpoints, n_eval_trials))
    runs_done = load_resume(save_path, records, n_runs, fresh)

    M_wb = woodbury_feat(T_os, T_new, F)
    t_start = time.time()

    for run in range(runs_done, n_runs):
        print_header(f"RUN {run + 1}/{n_runs}")
        t_run = time.time()

        M_td = None
        dq_state = {k: {"V": None, "T_count": None} for k in K_PLANNING_VALS}

        for chunk in range(n_checkpoints):
            train_seed = int(train_seeds[run, chunk])

            M_td = woodbury_td(F, T_new, emc, lam,
                               valid_states=np.arange(N_STATES),
                               grad_type="full",
                               n_samples=eval_every, lr=LR,
                               seed=train_seed, init_M=M_td)
            records["woodbury_feat"][run, chunk] = evaluate_starts(
                values_M(F, M_wb, T_new, fixed_goal, R_GOAL),
                starts, fixed_goal, run, chunk, nbrs, eval_max_steps, TEMP,
                ROLLOUT_SEED_BASE)
            records["woodbury_td"][run, chunk] = evaluate_starts(
                values_M(F, M_td, T_new, fixed_goal, R_GOAL),
                starts, fixed_goal, run, chunk, nbrs, eval_max_steps, TEMP,
                ROLLOUT_SEED_BASE)

            for k in K_PLANNING_VALS:
                V_dq, T_count_dq = dyna_q(T_new, fixed_goal,
                                          n_real_steps=eval_every,
                                          emc=emc, k_planning=k, lr=LR,
                                          r_goal=DQ_R_GOAL, seed=train_seed,
                                          init_V=dq_state[k]["V"],
                                          init_T_count=dq_state[k]["T_count"])
                dq_state[k]["V"] = V_dq
                dq_state[k]["T_count"] = T_count_dq
                records[f"dyna_q_k{k}"][run, chunk] = evaluate_starts(
                    V_dq, starts, fixed_goal, run, chunk, nbrs, eval_max_steps,
                    TEMP, ROLLOUT_SEED_BASE)

            if (chunk + 1) % max(1, n_checkpoints // 10) == 0 or chunk == n_checkpoints - 1:
                msg = "  ".join(f"{m}={records[m][run, chunk].mean():6.1f}"
                                for m in ["woodbury_td"] + dq_names)
                print(f"    chunk {chunk+1:>3}/{n_checkpoints}  "
                      f"({(chunk+1)*eval_every:>7,})  {msg}", flush=True)

        save_records(save_path, records, run + 1,
                     checkpoint_steps=checkpoint_steps,
                     starts=starts,
                     fixed_goal=fixed_goal,
                     k_planning_vals=np.array(K_PLANNING_VALS),
                     dq_r_goal=DQ_R_GOAL,
                     train_seeds=train_seeds,
                     rollout_seed_base=ROLLOUT_SEED_BASE,
                     pair_seed=PAIR_SEED,
                     eval_max_steps=eval_max_steps,
                     n_runs=n_runs)
        print(f"  run {run+1} done in {time.time() - t_run:.1f}s  ->  saved")

    print(f"\n  total {time.time() - t_start:.1f}s   saved -> {save_path}")
    return save_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "report"], nargs="?", default="train")
    p.add_argument("--runs", type=int, default=N_RUNS)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=1_000)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    save_path = os.path.join(args.out, FNAME)
    if args.mode == "train":
        save_path = train_and_save(n_runs=args.runs,
                                   n_train_steps=args.steps,
                                   eval_every=args.eval_every,
                                   n_eval_trials=args.trials,
                                   eval_max_steps=args.max_steps,
                                   save_dir=args.out, fresh=args.fresh)

    names = ["woodbury_feat", "woodbury_td"] + [f"dyna_q_k{k}" for k in K_PLANNING_VALS]
    report_multiseed(save_path,
           groups=[("L-barrier, fixed goal", {m: m for m in names})],
           contrasts=[("woodbury_td", f"dyna_q_k{k}") for k in K_PLANNING_VALS]
                     + [("woodbury_td", "woodbury_feat")],
           within_label="starts")


if __name__ == "__main__":
    main()
