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
    values_D,
    values_SR,
    values_SF,
    print_header,
    draw_pairs,
    spawn_train_seeds,
    evaluate_pairs,
    init_records,
    load_resume,
    save_records,
    report_multiseed,
)
from models import build_features, woodbury_feat, woodbury_full, woodbury_td, sr_td, sr_sf

# =============================================================================
# CONFIG
# =============================================================================

ROWS, COLS = 15, 15
K          = ROWS * COLS
LR         = 0.01
TEMP       = 0.2
R_GOAL     = 10
SEED       = 42

CORNER_ROW = 5
CORNER_COL = 5
H_END_COL  = 12
V_END_ROW  = 12

START_ROW_MAX = CORNER_ROW - 2   # start rows 0..3
START_COL_MAX = CORNER_COL - 2   # start cols 0..3
GOAL_ROW_MIN  = 6
GOAL_COL_MIN  = 6

N_RUNS            = 5
PAIR_SEED         = 20_250_803
ROLLOUT_SEED_BASE = 940_000

TD_MODELS       = ("woodbury_td", "sr_td", "sr_sf")
ANALYTIC_MODELS = ("woodbury_feat", "woodbury_full")
ALL_MODELS      = ANALYTIC_MODELS + TD_MODELS

DEFAULT_OUT = "../data/planning_open/angle/"
FNAME       = "learning_curves_multiseed.npz"


def build_environment():
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    h_edges = h_wall_edges(ROWS, COLS, barrier_row=CORNER_ROW,
                           col_start=CORNER_COL, col_end=H_END_COL)
    v_edges = v_wall_edges(ROWS, COLS, barrier_col=CORNER_COL,
                           row_start=CORNER_ROW, row_end=V_END_ROW)
    T_new = place_walls(T_os, h_edges + v_edges)
    F, lam, _, emc = build_features(D_os, k=K)
    return T_os, D_os, T_new, F, lam, emc, len(h_edges), len(v_edges)


def start_goal_pools():
    starts = [r * COLS + c
              for r in range(START_ROW_MAX + 1)
              for c in range(START_COL_MAX + 1)]
    goals = [r * COLS + c
             for r in range(GOAL_ROW_MIN, ROWS)
             for c in range(GOAL_COL_MIN, COLS)]
    return starts, goals


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

    print_header("TRAIN & SAVE (multi-seed)  --  open field -> L-barrier")
    T_os, D_os, T_new, F, lam, emc, n_h, n_v = build_environment()
    nbrs = neighbour_table(T_new)

    starts, goals = start_goal_pools()
    pairs = draw_pairs(starts, goals, n_eval_trials, PAIR_SEED)

    print(f"  runs        : {n_runs}")
    print(f"  L-wall      : {n_h} H edges + {n_v} V edges")
    print(f"  train steps : {n_train_steps:,} (eval every {eval_every:,})")
    print(f"  regions     : {len(starts)} starts x {len(goals)} goals")
    print(f"  pairs       : seed {PAIR_SEED}, "
          f"{len({tuple(p) for p in pairs})} unique of {n_eval_trials} draws")

    train_seeds = spawn_train_seeds(SEED, n_runs, n_checkpoints)
    Psi0 = None

    print("\n  computing analytic solutions ...")
    M_wb = woodbury_feat(T_os, T_new, F)
    D_new = woodbury_full(T_os, D_os, T_new)

    records = init_records(ALL_MODELS, (n_runs, n_checkpoints, n_eval_trials))
    runs_done = load_resume(save_path, records, n_runs, fresh)

    t_start = time.time()

    for run in range(runs_done, n_runs):
        print_header(f"RUN {run + 1}/{n_runs}")
        t_run = time.time()

        M_td, D_sr = None, None
        Psi = Psi0.copy()

        for chunk in range(n_checkpoints):
            train_seed = int(train_seeds[run, chunk])

            M_td = woodbury_td(F, T_new, emc, lam,
                               valid_states=np.arange(ROWS * COLS),
                               grad_type="full",
                               n_samples=eval_every, lr=LR,
                               seed=train_seed, init_M=M_td)
            D_sr = sr_td(T_new, emc, valid_states=np.arange(ROWS * COLS),
                         lr=LR, n_samples=eval_every,
                         seed=train_seed, init_D=D_sr)
            Psi = sr_sf(T_new, emc, F, valid_states=np.arange(ROWS * COLS),
                        lr=LR, n_samples=eval_every,
                        seed=train_seed, init_Psi=Psi)

            # analytic models carry no training randomness but are re-evaluated
            # here so their curves show the same evaluation noise as the rest
            records["woodbury_feat"][run, chunk] = evaluate_pairs(
                lambda g: values_M(F, M_wb, T_new, g, R_GOAL),
                pairs, run, chunk, nbrs, eval_max_steps, TEMP, ROLLOUT_SEED_BASE)
            records["woodbury_full"][run, chunk] = evaluate_pairs(
                lambda g: values_D(D_new, T_new, g, R_GOAL),
                pairs, run, chunk, nbrs, eval_max_steps, TEMP, ROLLOUT_SEED_BASE)
            records["woodbury_td"][run, chunk] = evaluate_pairs(
                lambda g, _M=M_td: values_M(F, _M, T_new, g, R_GOAL),
                pairs, run, chunk, nbrs, eval_max_steps, TEMP, ROLLOUT_SEED_BASE)
            records["sr_td"][run, chunk] = evaluate_pairs(
                lambda g, _D=D_sr: values_SR(_D, g, R_GOAL),
                pairs, run, chunk, nbrs, eval_max_steps, TEMP, ROLLOUT_SEED_BASE)
            records["sr_sf"][run, chunk] = evaluate_pairs(
                lambda g, _P=Psi: values_SF(_P, F, lam, g, R_GOAL),
                pairs, run, chunk, nbrs, eval_max_steps, TEMP, ROLLOUT_SEED_BASE)

            if (chunk + 1) % max(1, n_checkpoints // 10) == 0 or chunk == n_checkpoints - 1:
                msg = "  ".join(f"{m}={records[m][run, chunk].mean():6.1f}"
                                for m in TD_MODELS)
                print(f"    chunk {chunk+1:>3}/{n_checkpoints}  "
                      f"({(chunk+1)*eval_every:>7,})  {msg}", flush=True)

        save_records(save_path, records, run + 1,
                     checkpoint_steps=checkpoint_steps,
                     pairs=pairs,
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

    report_multiseed(save_path,
           groups=[("L-barrier", {m: m for m in ALL_MODELS})],
           contrasts=[("woodbury_td", "sr_td"),
                      ("woodbury_td", "sr_sf"),
                      ("woodbury_td", "woodbury_feat")],
           within_label="pairs")


if __name__ == "__main__":
    main()