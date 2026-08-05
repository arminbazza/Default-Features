import os
import sys
import time
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (create_transition_open, get_complete_DR, place_barrier,
                   place_walls, make_four_room, make_absorbing, h_wall_edges,
                   v_wall_edges, plan_with_M, plan_with_D, _pick_next)
from models import build_features, woodbury_td, woodbury_full

# ------------------------------------------------------------------ FIXED PARAMS
LR             = 0.01
TEMP           = 0.2
EVAL_MAX_STEPS = 1000
R_GOAL         = 10
EVAL_SEED      = 42
DEFAULT_SEEDS  = 5
INTERP_ALPHAS  = [0.25, 0.5, 0.75]
RANDOM_LRS     = [0.003, 0.03]

save_dir = "../data/basis_alignment/"
os.makedirs(save_dir, exist_ok=True)
RESULTS_PATH = save_dir + "results_basis_ablation.pkl"


# ============================================================ ENVIRONMENT SETUPS
def build_env(name):
    if name == "horizontal":
        rows = cols = 15
        n = rows * cols
        T_warm = create_transition_open(shape=(rows, cols))
        D = get_complete_DR(T=T_warm)
        T_new = place_walls(T_warm, h_wall_edges(rows, cols, barrier_row=rows // 2,
                                                 col_start=3, col_end=12))
        valid_states = np.arange(n)
        env = dict(name=name, rows=rows, cols=cols, n=n,
                   n_samples=50_000, eval_every=1_000, n_eval_trials=200,
                   eval_mode="region",
                   starts=[r * cols + c for r in range(3, 6) for c in range(5, 10)],
                   goals=[r * cols + c for r in range(8, 11) for c in range(5, 10)])

    elif name == "four_room":
        rows = cols = 11
        n = rows * cols
        goal = 10 * cols + 0
        T_base, _walls = make_four_room(rows, cols)
        T_warm = make_absorbing(T_base, goal)
        D = get_complete_DR(T=T_warm)
        T_new = place_barrier(T_warm, [5 * cols + 3])
        valid_states = np.array([s for s in range(n) if T_new[s].sum() > 0])
        env = dict(name=name, rows=rows, cols=cols, n=n,
                   n_samples=150_000, eval_every=2_500, n_eval_trials=20,
                   eval_mode="single_pair",
                   start=0, goal=goal)
    else:
        raise ValueError(name)

    env.update(T_warm=T_warm, D_warm=D,
               T_new=T_new, valid_states=valid_states)
    return env


# =================================================================== BASIS SETUP
def make_basis(env, kind):
    n, D_warm = env["n"], env["D_warm"]
    rows, cols = env["rows"], env["cols"]
    F_base, lam_base, _, emc_base = build_features(D_warm, k=n)

    def warm_start(F):
        Finv = np.linalg.inv(F)
        return Finv @ D_warm @ Finv.T

    if kind in ("spectral", "four_room"):
        return F_base, lam_base, emc_base, np.eye(n)

    if kind == "L_barrier":                       # horizontal: L-shaped barrier env
        edges = (h_wall_edges(rows, cols, barrier_row=5, col_start=5, col_end=12)
                 + v_wall_edges(rows, cols, barrier_col=5, row_start=5, row_end=12))
        D = get_complete_DR(place_walls(create_transition_open(shape=(rows, cols)),
                                             edges))
        F, lam, _, emc = build_features(D, k=n)
        return F, lam, emc, warm_start(F)

    if kind == "one_door_blocked":                # four-room: same goal, a DIFFERENT door blocked
        T_base, _ = make_four_room(rows, cols)
        T_alt = place_barrier(make_absorbing(T_base, env["goal"]), [2 * cols + 5])  # block TL<->TR (27)
        F, lam, _, emc = build_features(get_complete_DR(T_alt), k=n)
        return F, lam, emc, warm_start(F)

    if kind == "open_field":                      # four-room: open-field basis
        D = get_complete_DR(create_transition_open(shape=(rows, cols)))
        F, lam, _, emc = build_features(D, k=n)
        return F, lam, emc, warm_start(F)

    if kind == "random" or kind.startswith("random_lr"):   # matched-spectrum control
        V_rand = _random_orthonormal(n, seed=0)
        F = V_rand * np.sqrt(np.clip(lam_base, 0, None))[np.newaxis, :]
        return F, lam_base, emc_base, warm_start(F)

    if kind.startswith("interp_"):                # synthetic aligned->random (spectrum fixed)
        alpha = float(kind.split("_")[1])
        V_base = F_base / np.sqrt(np.clip(lam_base, 1e-12, None))[np.newaxis, :]
        V_rand = _random_orthonormal(n, seed=0)
        blend = V_base * np.cos(alpha * np.pi / 2) + V_rand * np.sin(alpha * np.pi / 2)
        V_a, _ = np.linalg.qr(blend)
        F = V_a * np.sqrt(np.clip(lam_base, 0, None))[np.newaxis, :]
        return F, lam_base, emc_base, warm_start(F)

    raise ValueError(kind)


def _random_orthonormal(n, seed):
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    return Q


def basis_lr(kind):
    if kind.startswith("random_lr"):
        return float(kind.split("lr")[1])
    return LR


def basis_list(env_name):
    if env_name == "horizontal":
        real = ["spectral", "L_barrier", "random"]
    else:
        real = ["four_room", "one_door_blocked", "open_field", "random"]
    rand_lr = [f"random_lr{lr}" for lr in RANDOM_LRS]
    interp = [f"interp_{a}" for a in INTERP_ALPHAS]
    return real + rand_lr + interp


# ======================================================================= EVAL
def _values_M(F, M, T_new, goal, n):
    nt = np.array([s for s in range(n) if s != goal])
    t_vec = T_new[np.ix_(nt, [goal])].flatten()
    F_nt = F[nt]
    exp_v = np.maximum(F_nt @ M @ F_nt.T @ t_vec, 1e-10)
    values = np.full(n, -np.inf)
    values[nt] = np.log(exp_v)
    values[goal] = R_GOAL
    return values


def _values_D(D, T_new, goal, n):
    nt = np.array([s for s in range(n) if s != goal])
    t_vec = T_new[np.ix_(nt, [goal])].flatten()
    exp_v = np.maximum(D[np.ix_(nt, nt)] @ t_vec, 1e-10)
    values = np.full(n, -np.inf)
    values[nt] = np.log(exp_v)
    values[goal] = R_GOAL
    return values


def _rollout(values, T_new, start, goal):
    state, steps = start, EVAL_MAX_STEPS
    for step in range(EVAL_MAX_STEPS):
        if state == goal:
            steps = step
            break
        nb = np.where(T_new[state] > 0)[0]
        if len(nb) == 0:
            break
        state = _pick_next(nb, values, TEMP)
    return steps


def _eval_from_values(value_fn, env):
    n, T_new = env["n"], env["T_new"]
    np.random.seed(EVAL_SEED)                       # softmax rollout RNG
    if env["eval_mode"] == "single_pair":
        goal, start = env["goal"], env["start"]
        values = value_fn(goal)
        steps = np.array([_rollout(values, T_new, start, goal)
                          for _ in range(env["n_eval_trials"])])
    else:  # region
        pair_rng = np.random.RandomState(EVAL_SEED)
        cache = {g: value_fn(g) for g in set(env["goals"])}
        steps = np.empty(env["n_eval_trials"])
        for i in range(env["n_eval_trials"]):
            start = pair_rng.choice(env["starts"])
            goal = pair_rng.choice(env["goals"])
            steps[i] = _rollout(cache[goal], T_new, start, goal)
    return steps.mean(), np.median(steps), np.mean(steps < EVAL_MAX_STEPS)


def evaluate(F, M, env):
    return _eval_from_values(lambda g: _values_M(F, M, env["T_new"], g, env["n"]), env)


def evaluate_oracle(env):
    """Analytic model-based oracle: exact D_new via woodbury_full (seed-free)."""
    D_new = woodbury_full(env["T_warm"], env["D_warm"], env["T_new"])
    return _eval_from_values(lambda g: _values_D(D_new, env["T_new"], g, env["n"]), env)


def _self_test(env):
    """Lightweight sanity check: warm-start _rollout runs and returns valid step counts.
    (Equivalence with plan_with_M was verified previously; we no longer re-run it.)"""
    F, lam, emc, M0 = make_basis(env, basis_list(env["name"])[0])
    rng = np.random.RandomState(1)
    goals = [env["goal"]] if env["eval_mode"] == "single_pair" else env["goals"]
    starts = [env["start"]] if env["eval_mode"] == "single_pair" else env["starts"]
    for _ in range(6):
        start = int(rng.choice(starts))
        goal = int(rng.choice(goals))
        np.random.seed(123)
        steps = _rollout(_values_M(F, M0, env["T_new"], goal, env["n"]),
                         env["T_new"], start, goal)
        assert 0 <= steps <= EVAL_MAX_STEPS, f"bad rollout step count: {steps}"


# ==================================================================== TRAINING
def run_basis(env, kind, seeds, eval_stride=None):
    """
    Multi-seed, checkpointed training for one basis.
    """
    F, lam, emc, M0 = make_basis(env, kind)
    lr = basis_lr(kind)
    eval_every = env["eval_every"]
    n_chunks = env["n_samples"] // eval_every
    if not eval_stride or eval_stride < 1:
        eval_stride = max(1, n_chunks // 20)          # ~20 curve points by default
    eval_chunks = sorted({c for c in range(1, n_chunks + 1)
                          if c % eval_stride == 0 or c == n_chunks})
    eval_set = set(eval_chunks)
    steps_axis = np.array([c * eval_every for c in eval_chunks])

    out = np.empty((len(seeds), len(eval_chunks), 3))
    t0 = time.time()
    for si, seed in enumerate(seeds):
        M = M0.copy()
        ei = 0
        for c in range(1, n_chunks + 1):
            M = woodbury_td(F, env["T_new"], emc, lam,
                            valid_states=env["valid_states"], grad_type='full',
                            n_samples=eval_every, lr=lr,
                            seed=42 + seed * 100_003 + (c - 1) * 1009, init_M=M)
            if c in eval_set:
                out[si, ei] = evaluate(F, M, env)
                ei += 1
        print(f"      seed {si+1:2d}/{len(seeds)}  "
              f"final {out[si,-1,0]:6.1f} / {out[si,-1,2]:.2f}  "
              f"[{time.time()-t0:.0f}s elapsed]", flush=True)
    return steps_axis, out


# ===================================================================== DRIVER
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "rb") as fh:
            return pickle.load(fh)
    return {"meta": {}, "data": {}, "oracle": {}}


def save_results(res):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh)
    os.replace(tmp, RESULTS_PATH)


def run(env_names, seeds, eval_stride=1):
    res = load_results()
    for env_name in env_names:
        env = build_env(env_name)
        _self_test(env)
        res["data"].setdefault(env_name, {})
        res["oracle"][env_name] = evaluate_oracle(env)     # seed-free reference
        print(f"\n=== {env_name}  (N={env['n']}, {env['n_samples']:,} steps, "
              f"{len(seeds)} seeds) ===", flush=True)
        print(f"  oracle (woodbury_full): mean={res['oracle'][env_name][0]:.1f} "
              f"success={res['oracle'][env_name][2]:.2f}", flush=True)

        for kind in basis_list(env_name):
            prev = res["data"][env_name].get(kind)
            if prev is not None and prev["results"].shape[0] >= len(seeds):
                print(f"  [skip] {kind}", flush=True)
                continue
            t0 = time.time()
            steps_axis, arr = run_basis(env, kind, seeds, eval_stride)
            res["data"][env_name][kind] = dict(steps=steps_axis, results=arr)
            res["meta"][env_name] = dict(
                n_samples=env["n_samples"], eval_every=env["eval_every"],
                n=env["n"], lr=LR, grad_type='full', temp=TEMP,
                eval_max_steps=EVAL_MAX_STEPS, seeds=list(seeds),
                eval_mode=env["eval_mode"])
            save_results(res)
            f = arr[:, -1, :]
            print(f"  {kind:16s} lr={basis_lr(kind):<5g} final "
                  f"mean={f[:,0].mean():7.1f} +/- {f[:,0].std():5.1f}  "
                  f"success={f[:,2].mean():.2f}   ({time.time()-t0:.0f}s)", flush=True)
    return res


def report_table(res):
    """Rebuttal table: basis x quarter-budget checkpoints + success@final."""
    for env_name, data in res.get("data", {}).items():
        if not data:
            continue
        any_key = next(iter(data))
        steps = data[any_key]["steps"]
        n_pts = len(steps)
        q_idx = sorted(set(max(0, round(f * n_pts) - 1) for f in (0.25, 0.5, 0.75, 1.0)))
        q_steps = [steps[i] for i in q_idx]
        print(f"\n================ {env_name}  (mean accumulated cost) ================")
        hdr = "".join(f"{s//1000:>7}k" for s in q_steps) + "   succ@final"
        print(f"{'basis':<20}{hdr}")
        print("-" * (20 + 8 * len(q_idx) + 12))
        for kind in basis_list(env_name):
            if kind not in data:
                continue
            r = data[kind]["results"]
            means = r[:, :, 0].mean(axis=0)
            succ = r[:, -1, 2].mean()
            row = "".join(f"{means[i]:8.1f}" for i in q_idx)
            print(f"{kind:<20}{row}   {succ:.2f}")
        if env_name in res.get("oracle", {}):
            o = res["oracle"][env_name]
            print(f"{'woodbury_full oracle':<20}{'':>{8*len(q_idx)}}   {o[2]:.2f}"
                  f"   (mean {o[0]:.1f})")


# ======================================================================== MAIN
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("env", choices=["horizontal", "four_room", "both"], nargs="?",
                    default="both")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--test", action="store_true", help="quick 2-seed smoke test")
    ap.add_argument("--eval-stride", type=int, default=1,
                    help="evaluate every N chunks (1 = every chunk, full curve)")
    ap.add_argument("--report", action="store_true", help="print table from saved results")
    args = ap.parse_args()

    if args.report:
        report_table(load_results())
        sys.exit(0)

    envs = ["horizontal", "four_room"] if args.env == "both" else [args.env]
    seeds = list(range(2 if args.test else args.seeds))
    results = run(envs, seeds, eval_stride=args.eval_stride)
    report_table(results)
    print(f"\nSaved to {RESULTS_PATH}")