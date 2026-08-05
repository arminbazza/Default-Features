import os
import argparse

import numpy as np
from numpy.linalg import eigh, norm

from utils import create_transition_open, get_complete_DR, make_wall_maze

# =============================================================================
# CONFIG
# =============================================================================

ROWS, COLS     = 10, 10
BARRIER_TYPE   = 'H'
K_VALUES       = [10, 25, 50, 75, 100]
N_SAMPLES      = 400_000      # TD training steps
LR             = 0.001
SEED           = 42
N_SEEDS        = 20
MEASURE_STEPS  = 1_000        # default-policy steps used for the sampled measure
BURN_IN        = 500          # discarded before collecting, so the walk has mixed
SAVE_DIR       = "../data/td_error/"


# =============================================================================
# BASIS
# =============================================================================

def full_eigendecomposition(D_os):
    w, V = eigh(D_os)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def basis_at(lam_all, V_all, K):
    lam = lam_all[:K]
    F = V_all[:, :K] * np.sqrt(lam)[np.newaxis, :]
    gamma = 1.0 / (1.0 + 1.0 / lam_all[0])
    return F, lam, gamma


# =============================================================================
# MEASURE
# =============================================================================

def delta_matrix(F, lam, gamma, T, M):
    """(N, K) array whose row s is delta_s, expectation over successors exact."""
    R = F * (1.0 / lam)[np.newaxis, :]           # reward features r_s
    EF = T @ F                                   # expected successor features
    return gamma * (R + EF @ M) - F @ M


def measure_sampled(Delta, T, n_steps, burn_in, seed):
    """Average ||delta_s||^2 / K over states visited by the default policy."""
    N, K = Delta.shape
    rng = np.random.RandomState(seed)
    sq = np.sum(Delta ** 2, axis=1)
    s = rng.randint(N)
    for _ in range(burn_in):
        s = rng.choice(N, p=T[s])
    total = 0.0
    for _ in range(n_steps):
        total += sq[s]
        s = rng.choice(N, p=T[s])
    return float(total / n_steps / K)


def stationary_distribution(T, tol=1e-13, max_iter=200_000):
    """mu(s): left eigenvector of T with eigenvalue 1, by power iteration."""
    N = T.shape[0]
    mu = np.ones(N) / N
    for _ in range(max_iter):
        nxt = mu @ T
        tot = nxt.sum()
        if tot <= 0:
            break
        nxt /= tot
        if np.abs(nxt - mu).max() < tol:
            return nxt
        mu = nxt
    return mu


def measure_analytic(Delta, mu):
    """sum_s mu(s) ||delta_s||^2 / K over all states."""
    K = Delta.shape[1]
    return float(np.dot(mu, np.sum(Delta ** 2, axis=1)) / K)


def counts_to_T(counts, T_prior):
    rs = counts.sum(axis=1, keepdims=True)
    return np.where(rs > 0, counts / np.maximum(rs, 1), T_prior)


# =============================================================================
# TRAINING
# =============================================================================

def train_M(F, lam, gamma, T_new, n_samples, lr, seed, estimate_T):
    N, K = F.shape
    lam_inv = 1.0 / lam
    M = np.eye(K)
    rng = np.random.RandomState(seed)
    counts = np.zeros((N, N)) if estimate_T else None
    for _ in range(n_samples):
        s = rng.randint(N)
        sp = rng.choice(N, p=T_new[s])
        if estimate_T:
            counts[s, sp] += 1
        f_s, f_sp = F[s], F[sp]
        delta = gamma * (f_s * lam_inv + f_sp @ M) - f_s @ M
        M += lr * np.outer(f_s, delta)
    return M, counts


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--measure-steps", type=int, default=MEASURE_STEPS)
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--estimate-T", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    N = ROWS * COLS
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    D_os = (D_os + D_os.T) / 2
    T_new, _ = make_wall_maze(T_os, BARRIER_TYPE, ROWS, COLS)
    D_new = get_complete_DR(T=T_new)
    lam_all, V_all = full_eigendecomposition(D_os)
    mu = stationary_distribution(T_new)
    seeds = [SEED + i for i in range(args.n_seeds)]

    print("=" * 88)
    print(f"TD-error measure after learning   {ROWS}x{COLS}, N={N}, "
          f"T={'estimated' if args.estimate_T else 'known'}")
    print(f"{args.n_samples:,} TD steps, lr={args.lr}, {args.n_seeds} seeds; "
          f"sampled measure over {args.measure_steps:,} default-policy steps "
          f"(burn-in {args.burn_in})")
    print(f"mu(s) range [{mu.min():.4f}, {mu.max():.4f}] "
          f"(uniform would be {1.0/N:.4f})")
    print("=" * 88)

    results = {}
    for K in K_VALUES:
        F, lam, gamma = basis_at(lam_all, V_all, K)
        s_vals, a_vals, r_vals = [], [], []
        for sd in seeds:
            M, counts = train_M(F, lam, gamma, T_new,
                                args.n_samples, args.lr, sd, args.estimate_T)
            T_use = counts_to_T(counts, T_os) if args.estimate_T else T_new
            Delta = delta_matrix(F, lam, gamma, T_use, M)
            s_vals.append(measure_sampled(Delta, T_use, args.measure_steps,
                                          args.burn_in, sd + 10_000))
            a_vals.append(measure_analytic(Delta, mu))
            r_vals.append(norm(F @ M @ F.T - D_new) / norm(D_new))
        results[K] = dict(sampled=np.array(s_vals),
                          analytic=np.array(a_vals),
                          recon=np.array(r_vals))
        s, a, r = results[K]['sampled'], results[K]['analytic'], results[K]['recon']
        print(f"  K = {K:>3}   sampled {s.mean():.4e}   "
              f"analytic {a.mean():.4e}   recon {r.mean():.4f}", flush=True)

    def sem(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    print("\n" + "=" * 88)
    print(f"FINAL  (mean +/- sem over {args.n_seeds} seeds)")
    print("=" * 88)
    print(f"  {'K':>4}{'E (sampled)':>16}{'sem':>12}"
          f"{'E (analytic)':>16}{'sem':>12}{'recon err':>13}{'sem':>12}")
    for K in K_VALUES:
        s, a, r = results[K]['sampled'], results[K]['analytic'], results[K]['recon']
        print(f"  {K:>4}{s.mean():16.4e}{sem(s):12.2e}"
              f"{a.mean():16.4e}{sem(a):12.2e}{r.mean():13.4f}{sem(r):12.2e}")

    if args.save:
        os.makedirs(SAVE_DIR, exist_ok=True)
        p = os.path.join(SAVE_DIR, "td_K.npz")
        arrays = {'k_values': np.array(K_VALUES), 'seeds': np.array(seeds), 'mu': mu}
        for K in K_VALUES:
            for name in ('sampled', 'analytic', 'recon'):
                arrays[f'{K}_{name}'] = results[K][name]
        np.savez(p, **arrays)
        print(f"\n  Saved -> {p}")


if __name__ == "__main__":
    main()