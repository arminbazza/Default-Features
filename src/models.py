"""
Models for cognitive map planning.

All feature-based models share the same F = V * sqrt(Lambda), derived from
the top-k eigenvectors of the open-field DR matrix D_os. This ensures a fair
comparison: the only difference between models is *how* they update M.

API
---
build_features(D_os, k)          ->  F, lam, ec, emc
woodbury_feat(T_old, T_new, F)   ->  M          (k x k)
woodbury_full(T_old, D_os, T_new)->  D_new      (N x N)
sf_td(F, T_new, emc, ...)        ->  M          (k x k)
"""

import numpy as np
from numpy.linalg import inv, eigh


# =============================================================================
# SHARED FEATURE BASIS
# =============================================================================

def build_features(D_os, k):
    """
    Extract the top-k eigenvectors of D_os and scale by sqrt(eigenvalue).

        F[s, i] = V[s, i] * sqrt(lambda_i)

    so that F @ F^T ≈ D_os (exact when k == N).

    Returns
    -------
    F   : (N, k) feature matrix
    lam : (k,)   eigenvalues in descending order
    ec  : scalar  e^c from the DR relation  (ec = 1 + 1/lam[0])
    emc : scalar  e^{-c} = 1/ec  (the discount factor < 1)
    """
    eigenvalues, eigvecs = eigh(D_os)
    idx = np.argsort(eigenvalues)[::-1][:k]
    lam = eigenvalues[idx]
    V = eigvecs[:, idx]
    F = V * np.sqrt(lam)[np.newaxis, :]

    ec  = 1.0 + 1.0 / lam[0]
    emc = 1.0 / ec
    return F, lam, ec, emc


# =============================================================================
# MODEL 1 — Woodbury featyre (analytic, projected onto F, model-based)
# =============================================================================

def woodbury_feat(T_old, T_new, F):
    """
    Compute the exact correction operator M in feature space such that
    F @ M @ F^T approximates D_new.

    Uses the Woodbury identity applied to the rows of T that changed.
    Requires knowledge of both T_old and T_new (model-based).

    Returns M : (k, k)
    """
    diffs = T_old != T_new
    delta_locs = np.unique(np.where(diffs)[0])
    N = T_old.shape[0]
    k = F.shape[1]

    R = (T_new - T_old)[delta_locs, :]   # (B, N)
    C = np.eye(N)[:, delta_locs]         # (N, B)
    H = R @ F                            # (B, k)
    G = F.T @ C                          # (k, B)
    B = H.shape[0]

    Z = H @ G                            # (B, B)
    A = inv(np.eye(B) - Z)
    M = np.eye(k) + G @ A @ H
    return M


# =============================================================================
# MODEL 2 — Woodbury FULL (analytic, full state space)
# =============================================================================

def woodbury_full(T_old, D_os, T_new):
    """
    Compute the exact updated DR matrix D_new via the full-state-space
    Woodbury identity.

    This is the model-based oracle: it requires T_old, T_new, and D_os.

    Returns D_new : (N, N)
    """
    diffs = T_old != T_new
    delta_locs = np.unique(np.where(diffs)[0])
    N = T_old.shape[0]

    R = (T_new - T_old)[delta_locs, :]
    C = np.eye(N)[:, delta_locs]

    Z = R @ D_os @ C
    A = inv(np.eye(Z.shape[0]) - Z)
    D_new = D_os + D_os @ C @ A @ R @ D_os
    return D_new


# =============================================================================
# MODEL 3 — Woodbury features via TD (online, model-free)
# =============================================================================

def woodbury_td(F, T_new, emc, lam,
                valid_states=None, grad_type="semi",
                n_samples=200_000, lr=0.001, seed=42,
                log_every=None, init_M=None):
    """
    Learn M online via semi-gradient TD so that F @ M @ F^T ≈ D_new.

    Parameters
    ----------
    init_M : (k, k) array or None
        Warm-start value for M. If None, initialises to identity.
    """
    N, k = F.shape
    if valid_states is None:
        valid_states = np.arange(N)

    rng = np.random.RandomState(seed)
    M = np.eye(k) if init_M is None else init_M.copy()
    lam_inv = 1.0 / lam[:k]
    running_loss = 0.0

    for t in range(n_samples):
        s  = rng.choice(valid_states)
        sp = rng.choice(N, p=T_new[s])

        f_s  = F[s]
        f_sp = F[sp]
        r_s = f_s * lam_inv

        delta = emc * (r_s + f_sp @ M) - f_s @ M
        if grad_type == "full":
            grad = 2.0 * np.outer(f_sp - 1/emc * f_s, delta)
            M -= lr * grad
        elif grad_type == "semi":
            M += lr * np.outer(f_s, delta)
        
        # logging
        running_loss = 0.99 * running_loss + 0.01 * np.sum(delta ** 2)
        if log_every and (t + 1) % log_every == 0:
            print(f"  woodbury_td step {t+1:>7,}  td_loss={running_loss:.4f}")

    return M

# =============================================================================
# MODEL 4 — Successor Representation with TD learning
# =============================================================================

def sr_td(T_new, emc, valid_states, lr=0.001,
          n_samples=200_000, seed=42,
          log_every=None, init_D=None):
    """
    Parameters
    ----------
    init_D : (N, N) array or None
        Warm-start value for D. If None, initialises to identity.
    """
    N = len(T_new)
    I = np.eye(N)
    D = I.copy() if init_D is None else init_D.copy()

    rng = np.random.RandomState(seed)
    running_loss = 0.0
    
    for t in range(n_samples):
        s = rng.choice(valid_states)
        sp = rng.choice(N, p=T_new[s])
        
        td_target = I[s] + emc * D[sp]
        td_error = td_target - D[s]
        # D[s] += lr * td_error
        
        D[s] = (1 - lr) * D[s] + lr * (I[s] + emc * D[sp])
        
        # logging
        td_error = td_target - D[s]
        step_loss = np.mean(td_error ** 2)
        if t == 0:
            running_loss = step_loss
        else:
            running_loss = 0.99 * running_loss + 0.01 * step_loss
        if log_every and (t + 1) % log_every == 0:
            print(f"  sr_td step {t+1:>7,}  td_loss={running_loss:.4f}")

    return D


# =============================================================================
# MODEL 5 — Successor Features
# =============================================================================

def sr_sf(T_new, emc, F, valid_states,
          lr=0.001, n_samples=200_000,
          seed=42, log_every=None, init_Psi=None):
    """
    Parameters
    ----------
    init_Psi : (N, d) array or None
        Warm-start value for Psi. If None, initialises to zeros.
    """
    N, d = F.shape
    Psi = np.zeros((N, d)) if init_Psi is None else init_Psi.copy()
    rng = np.random.RandomState(seed)
    running_loss = 0.0

    for t in range(n_samples):
        s  = rng.choice(valid_states)
        sp = rng.choice(N, p=T_new[s])

        td_target = F[s] + emc * Psi[sp]
        td_error  = td_target - Psi[s]
        Psi[s]   += lr * td_error

        # logging
        td_error_post = td_target - Psi[s]
        step_loss = np.mean(td_error_post ** 2)
        if t == 0:
            running_loss = step_loss
        else:
            running_loss = 0.99 * running_loss + 0.01 * step_loss
        if log_every and (t + 1) % log_every == 0:
            print(f"  sr_sf step {t+1:>7,}  td_loss={running_loss:.4f}")

    return Psi


# =============================================================================
# MODEL 6 — Dyna-Q (online, fixed goal, real samples + k simulated backups)
# =============================================================================

def dyna_q(T_true, fixed_goal, n_real_steps, emc,
           k_planning=5, r_goal=1.0, lr=0.001, seed=42,
           init_V=None, init_T_count=None):
    """
    Dyna-Q for a single fixed goal.

    Real step : sample s, observe s' ~ T_true[s], update T_hat (counts),
                Bellman-optimality backup on V[s] using T_hat
    Planning  : k times — sample a visited state sv, same max-backup on V[sv]

    Backups use max over known T_hat neighbors (Bellman optimality), NOT a
    single-sample TD update. Single-sample TD converges to V^pi (random
    policy value); the max operator makes V converge to V*.

    Parameters
    ----------
    init_T_count : (N, N) int array or None — warm-start transition counts

    Returns
    -------
    V       : (N,)   value function for fixed_goal
    T_count : (N, N) accumulated transition counts (warm-start next chunk)
    """
    N = T_true.shape[0]
    V = np.zeros(N) if init_V is None else init_V.copy()
    V[fixed_goal] = r_goal   # fixed terminal anchor — never updated in _backup
    T_count = (np.zeros((N, N), dtype=np.int32)
               if init_T_count is None else init_T_count.copy())

    rng     = np.random.RandomState(seed)
    visited = list(np.where(T_count.sum(axis=1) > 0)[0])

    def _backup(s):
        if s == fixed_goal:
            return   # terminal — V[goal] = r_goal stays fixed
        nbrs = np.where(T_count[s] > 0)[0]
        if len(nbrs) == 0:
            return
        # Pure discount Bellman: V(s) = emc * max_{s'} V(s')
        # V[goal] = r_goal is the high anchor, all others decay as emc^distance
        V[s] += lr * (emc * V[nbrs].max() - V[s])

    for _ in range(n_real_steps):
        s  = rng.randint(N)
        sp = rng.choice(N, p=T_true[s])

        if T_count[s].sum() == 0:
            visited.append(s)
        T_count[s, sp] += 1

        _backup(s)

        if visited:
            for sv in rng.choice(visited, size=k_planning, replace=True):
                _backup(int(sv))

    return V, T_count