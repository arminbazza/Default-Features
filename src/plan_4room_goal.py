import argparse
import os
import sys
import time

import numpy as np

from utils import (
    get_complete_DR,
    make_four_room,
    make_absorbing,
    create_transition_open,
    neighbour_table,
    values_M,
    values_D,
    values_SR,
    values_SF,
    print_header,
    spawn_train_seeds,
    evaluate_fixed,
    init_records,
    load_resume,
    save_records,
    report_multiseed,
)
from models import build_features, woodbury_feat, woodbury_full, woodbury_td, sr_td, sr_sf

# =============================================================================
# CONFIG
# =============================================================================

ROWS, COLS = 11, 11
K          = ROWS * COLS
LR         = 0.01
TEMP       = 0.2
R_GOAL     = 10
SEED       = 42

START = 0   # state (0, 0) — top-left corner of TL room

OUTER_CORNERS = {
    'TR': [0*11+6,  0*11+10, 4*11+10],    # 6,  10,  54
    'BL': [6*11+0, 10*11+0, 10*11+4],     # 66, 110, 114
    'BR': [6*11+10, 10*11+6, 10*11+10],   # 76, 116, 120
}
ALL_GOALS    = [g for gs in OUTER_CORNERS.values() for g in gs]
ROOM_OF_GOAL = {g: room for room, gs in OUTER_CORNERS.items() for g in gs}

N_RUNS            = 5
ROLLOUT_SEED_BASE = 610_000

TD_MODELS   = ('woodbury_td', 'sr_td', 'sr_sf')
MODEL_NAMES = ['woodbury_feat', 'woodbury_full'] + list(TD_MODELS)

DEFAULT_OUT = "../data/planning_4room/goal/"
FNAME       = "learning_curves_multiseed.npz"


def train_and_save(n_runs=N_RUNS,
                   n_train_steps=20_000,
                   eval_every=500,
                   n_eval_trials=20,
                   eval_max_steps=1_000,
                   save_dir=DEFAULT_OUT,
                   train_all=False,
                   fresh=False):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, FNAME)
    n_checkpoints = n_train_steps // eval_every
    checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every

    print_header("TRAIN & SAVE (multi-seed)  --  four-room goal change")
    T_base, wall_states = make_four_room(ROWS, COLS)
    base_valid = np.array([s for s in range(ROWS * COLS) if T_base[s].sum() > 0])

    train_goals = ALL_GOALS if train_all else [10]
    same_pairs, diff_pairs = [], []
    for g_train in train_goals:
        for g_test in ALL_GOALS:
            if g_test == g_train:
                continue
            (same_pairs if ROOM_OF_GOAL[g_test] == ROOM_OF_GOAL[g_train]
             else diff_pairs).append((g_train, g_test))
    n_same, n_diff = len(same_pairs), len(diff_pairs)

    print(f"  runs        : {n_runs}")
    print(f"  train goals : {train_goals}")
    print(f"  pairs       : {n_same} same-room, {n_diff} diff-room")
    print(f"  train steps : {n_train_steps:,} (eval every {eval_every:,})")
    print(f"  eval        : {n_eval_trials} rollouts START={START} -> g_test, "
          f"cap {eval_max_steps}")

    train_seeds = spawn_train_seeds(SEED, n_runs, n_checkpoints)

    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    F, lam, _, emc = build_features(D_os, k=K)      # open-field basis

    keys_same = [f"{m}_same" for m in MODEL_NAMES]
    keys_diff = [f"{m}_diff" for m in MODEL_NAMES]
    records = {}
    records.update(init_records(keys_same, (n_runs, n_same, n_checkpoints, n_eval_trials)))
    records.update(init_records(keys_diff, (n_runs, n_diff, n_checkpoints, n_eval_trials)))
    runs_done = load_resume(save_path, records, n_runs, fresh)

    t_start = time.time()

    for run in range(runs_done, n_runs):
        print_header(f"RUN {run + 1}/{n_runs}")
        t_run = time.time()
        same_idx = diff_idx = 0

        for g_train in train_goals:
            train_room = ROOM_OF_GOAL[g_train]
            T_train = make_absorbing(T_base, g_train)
            D_train = get_complete_DR(T_train)

            for g_test in [g for g in ALL_GOALS if g != g_train]:
                same_room = (ROOM_OF_GOAL[g_test] == train_room)
                pair_idx = same_idx if same_room else diff_idx
                tag = "same" if same_room else "diff"

                T_new = make_absorbing(T_base, g_test)
                valid_td = base_valid[base_valid != g_test]
                nbrs = neighbour_table(T_new)

                M_wb = woodbury_feat(T_os, T_new, F)
                D_wb = woodbury_full(T_train, D_train, T_new)

                # TD warm starts reset for each new pair (as in the original)
                M_td = None
                D_sr = None
                Psi = None

                for chunk in range(n_checkpoints):
                    train_seed = int(train_seeds[run, chunk])

                    M_td = woodbury_td(F, T_new, emc, lam,
                                       valid_states=valid_td, grad_type='full',
                                       n_samples=eval_every, lr=LR,
                                       seed=train_seed, init_M=M_td)
                    D_sr = sr_td(T_new, emc, valid_states=valid_td,
                                 n_samples=eval_every, lr=LR,
                                 seed=train_seed, init_D=D_sr)
                    Psi = sr_sf(T_new, emc, F, valid_states=valid_td,
                                n_samples=eval_every, lr=LR,
                                seed=train_seed, init_Psi=Psi)

                    for name, vals in (
                        # ("woodbury_feat", values_M(F, M_wb, T_new, g_test, R_GOAL)),
                        ("woodbury_full", values_D(D_wb, T_new, g_test, R_GOAL)),
                        ("woodbury_td",   values_M(F, M_td, T_new, g_test, R_GOAL)),
                        ("sr_td",         values_SR(D_sr, g_test, R_GOAL)),
                        ("sr_sf",         values_SF(Psi, F, lam, g_test, R_GOAL)),
                    ):
                        records[f"{name}_{tag}"][run, pair_idx, chunk] = evaluate_fixed(
                            vals, g_test, START, run, chunk, n_eval_trials, nbrs,
                            eval_max_steps, TEMP, ROLLOUT_SEED_BASE)

                msg = "  ".join(
                    f"{m}={records[f'{m}_{tag}'][run, pair_idx, -1].mean():6.1f}"
                    for m in MODEL_NAMES)
                print(f"  [{tag}] g_train={g_train:3d} g_test={g_test:3d}  {msg}",
                      flush=True)

                if same_room:
                    same_idx += 1
                else:
                    diff_idx += 1

        save_records(save_path, records, run + 1,
                     checkpoint_steps=checkpoint_steps,
                     same_pairs=np.array(same_pairs),
                     diff_pairs=np.array(diff_pairs),
                     train_seeds=train_seeds,
                     rollout_seed_base=ROLLOUT_SEED_BASE,
                     start_state=START,
                     eval_max_steps=eval_max_steps,
                     n_runs=n_runs)
        print(f"  run {run+1} done in {time.time() - t_run:.1f}s  ->  saved")

    print(f"\n  total {time.time() - t_start:.1f}s   saved -> {save_path}")
    return save_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "report"], nargs="?", default="train")
    p.add_argument("--runs", type=int, default=N_RUNS)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=1_000)
    p.add_argument("--train-all", action="store_true",
                   help="use all 9 training goals (default: goal 10 only)")
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
                                   save_dir=args.out,
                                   train_all=args.train_all,
                                   fresh=args.fresh)

    report_multiseed(save_path,
           groups=[("Same-room transfer", {m: f"{m}_same" for m in MODEL_NAMES}),
                   ("Diff-room transfer", {m: f"{m}_diff" for m in MODEL_NAMES})],
           contrasts=[("woodbury_td", "sr_td"),
                      ("woodbury_td", "sr_sf"),
                      ("woodbury_td", "woodbury_feat")],
           within_label="rollouts")


if __name__ == "__main__":
    main()
