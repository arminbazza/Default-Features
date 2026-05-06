"""
Bellman TD learning of M with arbitrary feature basis F.

Learns M such that D_new ≈ F @ M @ F^T from sampled transitions in
the new environment. Works with ANY F, not just orthonormal eigenvectors.

The Bellman equation for the DR in feature space:
    e^c * f(s) @ M = f(s) + E_{s'~T_new}[f(s') @ M]

TD error for a sampled transition s -> s':
    delta = f(s) + f(s') @ M - e^c * f(s) @ M

Three update strategies to handle non-orthonormal F:
1. Preconditioned semi-gradient: accounts for F^T F
2. Batch least-squares: solves for M from collected transitions
3. Whitened features: transforms F to orthonormal before learning

T_old is NEVER used. Only F, e^c, and sampled transitions from T_new.
"""

import numpy as np
from numpy.linalg import inv, norm, eigh
from utils import create_transition_open, get_complete_DR, make_maze

ROWS, COLS = 20, 20
NUM_STATES = ROWS * COLS
BARRIER_TYPE = 'H'


# =========================================================================
# UTILITIES
# =========================================================================

def woodbury(T_old, T_new, F):
    """Computes exact M using F"""
    diffs = T_old != T_new
    delta_locs = np.unique(np.where(diffs)[0])
    N = T_old.shape[0]
    dd = F.shape[1]
    R = (T_new - T_old)[delta_locs, :]
    C = np.eye(N)[:, delta_locs]
    H = R @ F
    G = F.T @ C
    B = H.shape[0]
    Z = H @ G
    A = inv(np.eye(B) - Z)
    return np.eye(dd) + G @ A @ H


def plan_with_F_M(F, M, T_new, goal, start, max_steps=500):
    """Plan using D_new ≈ F @ M @ F^T."""
    N = F.shape[0]
    nonterminal = [s for s in range(N) if s != goal]
    nt_idx = np.array(nonterminal)
    t_vec = T_new[np.ix_(nt_idx, [goal])].flatten()
    F_nt = F[nt_idx]
    exp_v = np.maximum(F_nt @ M @ F_nt.T @ t_vec, 1e-10)
    v = np.log(exp_v)
    values = np.full(N, -np.inf)
    for i, s in enumerate(nonterminal):
        values[s] = v[i]
    values[goal] = 10.0
    state = start
    path = [state]
    for _ in range(max_steps):
        if state == goal:
            break
        neighbors = np.where(T_new[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = neighbors[np.argmax(values[neighbors])]
        path.append(state)
    return (len(path) - 1 if state == goal else max_steps), path


def eval_barrier_crossing(F, M, T_new, barrier_states,
                           rows, cols, n_trials=200, seed=42):
    """Evaluate on paths that must cross the barrier."""
    N = rows * cols
    barrier_row = rows // 2
    valid = [s for s in range(N) if s not in barrier_states]
    rng = np.random.RandomState(seed)
    steps = []
    n = 0
    while n < n_trials:
        s = rng.choice(valid)
        g = rng.choice(valid)
        if s == g:
            continue
        if not ((s // cols < barrier_row and g // cols >= barrier_row) or
                (s // cols >= barrier_row and g // cols < barrier_row)):
            continue
        st, _ = plan_with_F_M(F, M, T_new, g, s)
        steps.append(st)
        n += 1
    arr = np.array(steps)
    return np.mean(arr), np.median(arr), np.sum(arr < 500) / n_trials


# =========================================================================
# METHOD 1: PRECONDITIONED SEMI-GRADIENT TD
# =========================================================================

def train_preconditioned_td(F, T_new, ec, valid_states,
                             M_true=None,
                             n_samples=200000, lr=0.01,
                             log_every=20000, seed=42):
    """
    Semi-gradient TD with preconditioning by (F^T F)^{-1}.
    
    The raw semi-gradient update is: M += lr * outer(f(s), delta)
    
    With non-orthonormal F, this is biased because f(s) vectors
    overlap. Preconditioning corrects for this:
    
        M += lr * (F^T F)^{-1} @ outer(f(s), delta)
    
    This is equivalent to learning in the whitened space but
    without explicitly transforming F.
    """
    N = F.shape[0]
    dd = F.shape[1]
    M = np.eye(dd)
    rng = np.random.RandomState(seed)

    # Precompute (F^T F)^{-1}
    gram_inv = inv(F.T @ F)

    history = {'samples': [], 'M_error': [], 'td_loss': []}
    running_loss = 0.0

    for t in range(n_samples):
        s = rng.choice(valid_states)
        sp = rng.choice(N, p=T_new[s, :])

        f_s = F[s, :]
        f_sp = F[sp, :]

        delta = f_s + f_sp @ M - ec * f_s @ M
        loss = np.sum(delta ** 2)
        running_loss = 0.99 * running_loss + 0.01 * loss

        # Preconditioned semi-gradient update
        update = gram_inv @ np.outer(f_s, delta)
        M += lr * update

        if (t + 1) % log_every == 0:
            m_err = norm(M - M_true) / norm(M_true) if M_true is not None else None
            m_str = f", M_err={m_err:.4f}" if m_err is not None else ""
            history['samples'].append(t + 1)
            history['M_error'].append(m_err)
            history['td_loss'].append(running_loss)
            print(f"  {t+1:>7d}: loss={running_loss:.4f}{m_str}")

    return M, history


# =========================================================================
# METHOD 2: BATCH LEAST-SQUARES
# =========================================================================

def train_batch_lstsq(F, T_new, ec, valid_states,
                       M_true=None,
                       n_batches=10, batch_size=10000,
                       log_every=1, seed=42):
    """
    Batch least-squares learning of M.
    
    Collect a batch of transitions, form the linear system, solve.
    
    From the Bellman equation: e^c * f(s) @ M = f(s) + f(s') @ M
    Rearranging:  f(s) = (e^c * f(s) - f(s')) @ M
    
    Over a batch of K transitions:
        targets[k, :] = f(s_k)                        (K x d)
        features[k, :] = e^c * f(s_k) - f(s'_k)      (K x d)
    
    Solve: targets = features @ M  =>  M = lstsq(features, targets)
    
    Each batch gives a full estimate of M. Average across batches
    for stability.
    """
    N = F.shape[0]
    dd = F.shape[1]
    rng = np.random.RandomState(seed)

    history = {'batch': [], 'M_error': []}
    M_avg = None
    total_samples = 0

    for b in range(n_batches):
        # Sample batch of transitions
        states = rng.choice(valid_states, size=batch_size)
        nexts = np.array([rng.choice(N, p=T_new[s, :]) for s in states])

        # Build system: f(s) = (e^c * f(s) - f(s')) @ M
        targets = F[states, :]                              # K x d
        features = ec * F[states, :] - F[nexts, :]          # K x d

        # Solve via least squares
        M_batch, _, _, _ = np.linalg.lstsq(features, targets, rcond=None)

        total_samples += batch_size

        # Running average of M estimates
        if M_avg is None:
            M_avg = M_batch.copy()
        else:
            M_avg = (b * M_avg + M_batch) / (b + 1)

        if (b + 1) % log_every == 0:
            m_err_batch = norm(M_batch - M_true) / norm(M_true) if M_true is not None else None
            m_err_avg = norm(M_avg - M_true) / norm(M_true) if M_true is not None else None
            m_str = ""
            if M_true is not None:
                m_str = (f", batch_err={m_err_batch:.4f}"
                         f", avg_err={m_err_avg:.4f}")
            history['batch'].append(b + 1)
            history['M_error'].append(m_err_avg)
            print(f"  Batch {b+1} ({total_samples} samples){m_str}")

    return M_avg, history


# =========================================================================
# METHOD 3: WHITENED FEATURES
# =========================================================================

def train_whitened_td(F, T_new, ec, valid_states,
                      M_true=None,
                      n_samples=200000, lr=0.01,
                      log_every=20000, seed=42):
    """
    Whiten F to make it orthonormal, then run standard semi-gradient TD.
    
    Given F, compute F_white = F @ W where W whitens the Gram matrix:
        F_white^T @ F_white = I
    
    Learn M_white in whitened space, then transform back:
        M_original = W @ M_white @ W^T
    
    This is mathematically equivalent to preconditioning but
    avoids the per-step matrix multiply.
    """
    N = F.shape[0]
    dd = F.shape[1]
    rng = np.random.RandomState(seed)

    # Whiten: F_w = F @ W such that F_w^T @ F_w = I
    gram = F.T @ F
    eigvals_g, eigvecs_g = eigh(gram)
    idx = np.argsort(eigvals_g)[::-1]
    eigvals_g = eigvals_g[idx]
    eigvecs_g = eigvecs_g[:, idx]

    # W = eigvecs @ diag(1/sqrt(eigvals))
    W = eigvecs_g @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_g, 1e-10)))
    W_inv = np.diag(np.sqrt(np.maximum(eigvals_g, 1e-10))) @ eigvecs_g.T

    F_w = F @ W  # (N, d) whitened features

    # Verify orthonormality
    gram_check = norm(F_w.T @ F_w - np.eye(dd))
    print(f"    ||F_w^T @ F_w - I|| = {gram_check:.2e}")

    # Transform M_true to whitened space for comparison
    M_true_w = None
    if M_true is not None:
        M_true_w = W_inv @ M_true @ W_inv.T

    # Initialize M_white
    # In whitened space, D_old = F_w @ M_w_old @ F_w^T
    # M_w_old = W^{-1} @ I @ W^{-T} = W_inv @ W_inv^T (when M_old = I in F-space)
    M_w = W_inv @ W_inv.T

    history = {'samples': [], 'M_error': [], 'td_loss': []}
    running_loss = 0.0

    for t in range(n_samples):
        s = rng.choice(valid_states)
        sp = rng.choice(N, p=T_new[s, :])

        fw_s = F_w[s, :]
        fw_sp = F_w[sp, :]

        delta = fw_s + fw_sp @ M_w - ec * fw_s @ M_w
        loss = np.sum(delta ** 2)
        running_loss = 0.99 * running_loss + 0.01 * loss

        # Standard semi-gradient (works because F_w is orthonormal)
        M_w += lr * np.outer(fw_s, delta)

        if (t + 1) % log_every == 0:
            # Transform back to compare
            M_original = W @ M_w @ W.T
            m_err = norm(M_original - M_true) / norm(M_true) if M_true is not None else None
            m_str = f", M_err={m_err:.4f}" if m_err is not None else ""
            history['samples'].append(t + 1)
            history['M_error'].append(m_err)
            history['td_loss'].append(running_loss)
            print(f"  {t+1:>7d}: loss={running_loss:.4f}{m_str}")

    # Transform back to original F-space
    M_final = W @ M_w @ W.T
    return M_final, history


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("BELLMAN TD WITH ARBITRARY FEATURES F")
    print("=" * 70)

    # === Setup ===
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    T_new, barrier_states = make_maze(T_os, BARRIER_TYPE, ROWS, COLS)
    valid_states = np.array([s for s in range(NUM_STATES)
                             if s not in barrier_states])

    # Eigenbasis
    eigenvalues, eigvecs = eigh(D_os)
    idx = np.argsort(eigenvalues)[::-1]
    lam = eigenvalues[idx]
    V = eigvecs[:, idx]
    ec = 1.0 + 1.0 / lam[0]

    print(f"Grid: {ROWS}x{COLS} = {NUM_STATES}")
    print(f"e^c = {ec:.6f}")

    # === Test at different d values ===
    for test_d in [50, 100, 400]:
        # Absorbed eigenbasis F = V * sqrt(Lambda)
        F = V[:, :test_d] * np.sqrt(lam[:test_d])[np.newaxis, :]
        dd = test_d

        recon = norm(F @ F.T - D_os) / norm(D_os)
        M_true = woodbury(T_os, T_new, F)

        print(f"\n{'='*70}")
        print(f"d = {test_d}, ||FF^T - D||/||D|| = {recon:.4f}")
        print(f"||M_true - I|| = {norm(M_true - np.eye(dd)):.4f}")
        print(f"{'='*70}")

        # Baselines
        mean_gt, med_gt, suc_gt = eval_barrier_crossing(
            F, M_true, T_new, barrier_states, ROWS, COLS)
        mean_nr, med_nr, suc_nr = eval_barrier_crossing(
            F, np.eye(dd), T_new, barrier_states, ROWS, COLS)
        print(f"True M:    mean={mean_gt:.1f}, median={med_gt:.1f}, success={suc_gt:.2f}")
        print(f"No repair: mean={mean_nr:.1f}, median={med_nr:.1f}, success={suc_nr:.2f}")

        # --- Method 1: Preconditioned semi-gradient TD ---
        print(f"\n  [Method 1] Preconditioned semi-gradient TD:")
        M1, h1 = train_preconditioned_td(
            F, T_new, ec, valid_states, M_true=M_true,
            n_samples=200000, lr=0.01, log_every=50000)
        m1, d1, s1 = eval_barrier_crossing(
            F, M1, T_new, barrier_states, ROWS, COLS)
        print(f"  Planning: mean={m1:.1f}, median={d1:.1f}, success={s1:.2f}")

        # --- Method 2: Batch least-squares ---
        print(f"\n  [Method 2] Batch least-squares:")
        M2, h2 = train_batch_lstsq(
            F, T_new, ec, valid_states, M_true=M_true,
            n_batches=10, batch_size=20000)
        m2, d2, s2 = eval_barrier_crossing(
            F, M2, T_new, barrier_states, ROWS, COLS)
        print(f"  Planning: mean={m2:.1f}, median={d2:.1f}, success={s2:.2f}")

        # --- Method 3: Whitened features ---
        print(f"\n  [Method 3] Whitened semi-gradient TD:")
        M3, h3 = train_whitened_td(
            F, T_new, ec, valid_states, M_true=M_true,
            n_samples=200000, lr=0.01, log_every=50000)
        m3, d3, s3 = eval_barrier_crossing(
            F, M3, T_new, barrier_states, ROWS, COLS)
        print(f"  Planning: mean={m3:.1f}, median={d3:.1f}, success={s3:.2f}")

        # --- Summary for this d ---
        print(f"\n  Summary (d={test_d}):")
        print(f"    True M:           mean={mean_gt:.1f}, success={suc_gt:.2f}")
        print(f"    Preconditioned:   mean={m1:.1f}, success={s1:.2f}, M_err={h1['M_error'][-1]:.4f}")
        print(f"    Batch lstsq:      mean={m2:.1f}, success={s2:.2f}, M_err={h2['M_error'][-1]:.4f}")
        print(f"    Whitened:         mean={m3:.1f}, success={s3:.2f}, M_err={h3['M_error'][-1]:.4f}")
        print(f"    No repair:        mean={mean_nr:.1f}, success={suc_nr:.2f}")


if __name__ == '__main__':
    main()