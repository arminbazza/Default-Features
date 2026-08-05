import argparse
import os
import time

import numpy as np

from utils import (
    create_transition_open,
    get_complete_DR,
    place_walls,
    h_wall_edges,
    neighbour_table,
    values_M,
    print_header,
    draw_pairs,
    spawn_train_seeds,
    evaluate_pairs,
    init_records,
    load_resume,
    save_records,
    report_multiseed,
)
from models import build_features, woodbury_feat, woodbury_td

# =============================================================================
# CONFIG
# =============================================================================

ROWS, COLS = 15, 15
K_VALUES   = [50, 75, 100, 125, 150, 175, 200, 225]
LR         = 0.01
TEMP       = 0.2
R_GOAL     = 10
SEED       = 42

BARRIER_ROW = ROWS // 2
COL_START   = 3
COL_END     = 12

START_ROW_MIN, START_ROW_MAX = 3, 5
START_COL_MIN, START_COL_MAX = 5, 9
GOAL_ROW_MIN,  GOAL_ROW_MAX  = 8, 10
GOAL_COL_MIN,  GOAL_COL_MAX  = 5, 9

N_RUNS            = 5
PAIR_SEED         = 20_250_803
ROLLOUT_SEED_BASE = 920_000

MODEL_NAMES = ("woodbury_feat", "woodbury_td")
DEFAULT_OUT = "../data/planning_open/horizontal/"
FNAME       = "num_steps_K_multiseed.npz"


def build_environment():
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    wall_edges = h_wall_edges(ROWS, COLS, barrier_row=BARRIER_ROW,
                              col_start=COL_START, col_end=COL_END)
    return T_os, D_os, place_walls(T_os, wall_edges)


def start_goal_pools():
    starts = [r * COLS + c
              for r in range(START_ROW_MIN, START_ROW_MAX + 1)
              for c in range(START_COL_MIN, START_COL_MAX + 1)]
    goals = [r * COLS + c
             for r in range(GOAL_ROW_MIN, GOAL_ROW_MAX + 1)
             for c in range(GOAL_COL_MIN, GOAL_COL_MAX + 1)]
    return starts, goals


def train_and_save(n_runs=N_RUNS,
                   k_values=None,
                   n_train_steps=50_000,
                   eval_every=1_000,
                   n_eval_trials=200,
                   eval_max_steps=1_000,
                   save_dir=DEFAULT_OUT,
                   fresh=False):
    k_values = k_values or K_VALUES
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, FNAME)
    n_checkpoints = n_train_steps // eval_every
    checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every

    print_header("TRAIN & SAVE (multi-seed)  --  horizontal barrier, K sweep")
    T_os, D_os, T_new = build_environment()
    nbrs = neighbour_table(T_new)

    starts, goals = start_goal_pools()
    pairs = draw_pairs(starts, goals, n_eval_trials, PAIR_SEED)
    print(f"  runs        : {n_runs}")
    print(f"  K values    : {k_values}")
    print(f"  train steps : {n_train_steps:,} (eval every {eval_every:,})")
    print(f"  pairs       : seed {PAIR_SEED}, "
          f"{len({tuple(p) for p in pairs})} unique of {n_eval_trials} draws")

    train_seeds = spawn_train_seeds(SEED, n_runs, n_checkpoints)

    keys = [f"{m}_K{k}" for k in k_values for m in MODEL_NAMES]
    records = init_records(keys, (n_runs, n_checkpoints, n_eval_trials))
    runs_done = load_resume(save_path, records, n_runs, fresh)

    t_start = time.time()

    for run in range(runs_done, n_runs):
        print_header(f"RUN {run + 1}/{n_runs}")
        t_run = time.time()

        for k in k_values:
            F, lam, _, emc = build_features(D_os, k=k)
            M_wb = woodbury_feat(T_os, T_new, F)

            M_td = None
            for chunk in range(n_checkpoints):
                M_td = woodbury_td(F, T_new, emc, lam,
                                   valid_states=np.arange(ROWS * COLS),
                                   grad_type="full",
                                   n_samples=eval_every, lr=LR,
                                   seed=int(train_seeds[run, chunk]), init_M=M_td)
                records[f"woodbury_feat_K{k}"][run, chunk] = evaluate_pairs(
                    lambda g: values_M(F, M_wb, T_new, g, R_GOAL),
                    pairs, run, chunk, nbrs, eval_max_steps, TEMP, ROLLOUT_SEED_BASE)
                records[f"woodbury_td_K{k}"][run, chunk] = evaluate_pairs(
                    lambda g, _M=M_td: values_M(F, _M, T_new, g, R_GOAL),
                    pairs, run, chunk, nbrs, eval_max_steps, TEMP, ROLLOUT_SEED_BASE)

            print(f"  K={k:>3d}  feat={records[f'woodbury_feat_K{k}'][run, -1].mean():7.1f}  "
                  f"td={records[f'woodbury_td_K{k}'][run, -1].mean():7.1f}", flush=True)

        save_records(save_path, records, run + 1,
                     checkpoint_steps=checkpoint_steps,
                     pairs=pairs,
                     k_values=np.array(k_values),
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
    p.add_argument("--k", type=str, default=None,
                   help="comma-separated K values (default: full sweep)")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=1_000)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    k_values = ([int(x) for x in args.k.split(",")] if args.k else K_VALUES)
    save_path = os.path.join(args.out, FNAME)
    if args.mode == "train":
        save_path = train_and_save(n_runs=args.runs, k_values=k_values,
                                   n_train_steps=args.steps,
                                   eval_every=args.eval_every,
                                   n_eval_trials=args.trials,
                                   eval_max_steps=args.max_steps,
                                   save_dir=args.out, fresh=args.fresh)

    report_multiseed(save_path,
           groups=[(f"K = {k}", {m: f"{m}_K{k}" for m in MODEL_NAMES})
                   for k in k_values],
           contrasts=[("woodbury_td", "woodbury_feat")],
           within_label="pairs")


if __name__ == "__main__":
    main()
