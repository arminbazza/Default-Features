import argparse
import os
import time

import numpy as np

from utils import (
    get_complete_DR,
    make_four_room,
    make_absorbing,
    place_barrier,
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

START = 0   # (0, 0) — TL corner

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

N_RUNS            = 5
ROLLOUT_SEED_BASE = 700_000

TD_MODELS       = ('woodbury_td', 'sr_td', 'sr_sf')
ANALYTIC_MODELS = ('woodbury_feat', 'woodbury_full')
MODEL_NAMES     = list(ANALYTIC_MODELS + TD_MODELS)

DEFAULT_OUT = "../data/planning_4room/blocked/"
FNAME       = "learning_curves_multiseed.npz"


def build_scenario(scenario_key, T_base):
    cfg = SCENARIOS[scenario_key]
    goal, barrier_state = cfg['goal'], cfg['barrier_state']

    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)

    T_train = make_absorbing(T_base, goal)
    D_train = get_complete_DR(T_train)
    F, lam, _, emc = build_features(D_os, k=K)

    T_new = place_barrier(T_train, [barrier_state])
    valid_td = np.array([s for s in range(ROWS * COLS) if T_new[s].sum() > 0])

    M_init = woodbury_feat(T_os, T_train, F)      # TD warm start
    M_wb   = woodbury_feat(T_os, T_new,   F)      # analytic (feature space)
    D_wb   = woodbury_full(T_train, D_train, T_new)

    return dict(goal=goal, barrier_state=barrier_state, label=cfg['label'],
                F=F, lam=lam, emc=emc, T_new=T_new, valid_td=valid_td,
                M_init=M_init, M_wb=M_wb, D_wb=D_wb,
                nbrs=neighbour_table(T_new))


def train_and_save(n_runs=N_RUNS,
                   n_train_steps=150_000,
                   eval_every=2_500,
                   n_eval_trials=20,
                   eval_max_steps=1_000,
                   save_dir=DEFAULT_OUT,
                   scenarios=None,
                   fresh=False):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, FNAME)
    scenarios = scenarios or list(SCENARIOS)
    n_checkpoints = n_train_steps // eval_every
    checkpoint_steps = np.arange(1, n_checkpoints + 1) * eval_every

    print_header("TRAIN & SAVE (multi-seed)  --  four-room blocked doorway state")
    print(f"  runs        : {n_runs}")
    print(f"  scenarios   : {', '.join(scenarios)}")
    print(f"  train steps : {n_train_steps:,}  (eval every {eval_every:,})")
    print(f"  eval        : {n_eval_trials} rollouts START={START} -> goal, "
          f"cap {eval_max_steps}")
    print(f"  grid        : {ROWS}x{COLS}, K = {K}, lr = {LR}, temp = {TEMP}")

    T_base, wall_states = make_four_room(ROWS, COLS)
    print(f"  wall states : {len(wall_states)}")

    setups = {}
    for key in scenarios:
        setups[key] = build_scenario(key, T_base)
        s = setups[key]
        print(f"  [{key}] goal={s['goal']}  barrier={s['barrier_state']}  ")

    train_seeds = spawn_train_seeds(SEED, n_runs, n_checkpoints)

    keys = [f"{m}_{k}" for m in MODEL_NAMES for k in scenarios]
    records = init_records(keys, (n_runs, n_checkpoints, n_eval_trials))
    runs_done = load_resume(save_path, records, n_runs, fresh)

    t_start = time.time()

    for run in range(runs_done, n_runs):
        print_header(f"RUN {run + 1}/{n_runs}")
        t_run = time.time()

        for key in scenarios:
            sc = setups[key]
            F, lam, emc = sc['F'], sc['lam'], sc['emc']
            T_new, goal, nbrs, valid_td = sc['T_new'], sc['goal'], sc['nbrs'], sc['valid_td']

            # TD warm starts, reset per scenario (as in the original)
            M_td = sc['M_init'].copy()
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
                    ("woodbury_feat", values_M(F, sc['M_wb'], T_new, goal, R_GOAL)),
                    ("woodbury_full", values_D(sc['D_wb'], T_new, goal, R_GOAL)),
                    ("woodbury_td",   values_M(F, M_td, T_new, goal, R_GOAL)),
                    ("sr_td",         values_SR(D_sr, goal, R_GOAL)),
                    ("sr_sf",         values_SF(Psi, F, lam, goal, R_GOAL)),
                ):
                    records[f"{name}_{key}"][run, chunk] = evaluate_fixed(
                        vals, goal, START, run, chunk, n_eval_trials, nbrs,
                        eval_max_steps, TEMP, ROLLOUT_SEED_BASE)

                if (chunk + 1) % 5 == 0 or chunk == n_checkpoints - 1:
                    msg = "  ".join(
                        f"{m}={records[f'{m}_{key}'][run, chunk].mean():6.1f}"
                        for m in TD_MODELS)
                    print(f"    [{key}] chunk {chunk+1:>3}/{n_checkpoints}  "
                          f"({(chunk+1)*eval_every:>8,} steps)  {msg}", flush=True)

        save_records(save_path, records, run + 1,
                     checkpoint_steps=checkpoint_steps,
                     scenarios=np.array(scenarios),
                     train_seeds=train_seeds,
                     rollout_seed_base=ROLLOUT_SEED_BASE,
                     start_state=START,
                     goals=np.array([SCENARIOS[k]['goal'] for k in scenarios]),
                     eval_max_steps=eval_max_steps,
                     n_runs=n_runs)
        print(f"  run {run+1} done in {time.time() - t_run:.1f}s  ->  saved")

    print(f"\n  total {time.time() - t_start:.1f}s   saved -> {save_path}")
    print(f"  arrays: {(n_runs, n_checkpoints, n_eval_trials)} per model/scenario")
    return save_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "report"], nargs="?", default="train")
    p.add_argument("--runs", type=int, default=N_RUNS)
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--eval-every", type=int, default=2_500)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=1_000)
    p.add_argument("--scenarios", type=str, default="BL,TR",
                   help="comma-separated subset of BL,TR")
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    save_path = os.path.join(args.out, FNAME)
    if args.mode == "train":
        save_path = train_and_save(n_runs=args.runs,
                                   n_train_steps=args.steps,
                                   eval_every=args.eval_every,
                                   n_eval_trials=args.trials,
                                   eval_max_steps=args.max_steps,
                                   save_dir=args.out,
                                   scenarios=scenarios,
                                   fresh=args.fresh)

    report_multiseed(save_path,
                     groups=[(f"Scenario {k}",
                              {m: f"{m}_{k}" for m in MODEL_NAMES})
                             for k in scenarios],
                     contrasts=[("woodbury_td", "sr_td"),
                                ("woodbury_td", "sr_sf"),
                                ("woodbury_td", "woodbury_feat")],
                     within_label="rollouts")


if __name__ == "__main__":
    main()
