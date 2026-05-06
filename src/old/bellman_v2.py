"""
Cognitive map update via Bellman TD learning of M.

The agent learns the correction operator M such that D_new ≈ F @ M @ F^T,
using ONLY:
1. F (pre-learned feature basis from old environment)
2. e^c (from eigenvalues of old D)
3. Sampled transitions from T_new (the agent lives in this environment)

No T_old, no H, no G, no Woodbury rebuild, no Eq 16.
Just TD learning of M from the DR's defining equation.

The TD equation:
    For each sampled transition s -> s':
        delta = f(s) + f(s') @ M - e^c * f(s) @ M
    This is zero in expectation at the correct M.
    Update: M -= lr * 2 * outer(f(s') - e^c * f(s), delta)
"""

import numpy as np
from numpy.linalg import inv, norm, eigh
from utils import create_transition_open, get_complete_DR, make_maze

ROWS, COLS = 20, 20
NUM_STATES = ROWS * COLS
BARRIER_TYPE = 'H'
d = NUM_STATES


# =========================================================================
# UTILITIES
# =========================================================================

def woodbury_ground_truth(T_old, D, T_new, F):
    """Compute exact M from known transitions. ONLY for verification."""
    diffs = T_old != T_new
    delta_locs = np.unique(np.where(diffs)[0])
    N = T_old.shape[0]
    dd = F.shape[1]  # use F's dimension, not global d
    R = (T_new - T_old)[delta_locs, :]
    C = np.eye(N)[:, delta_locs]
    H = R @ F
    G = F.T @ C
    B = H.shape[0]
    Z = H @ G
    A = inv(np.eye(B) - Z)
    M = np.eye(dd) + G @ A @ H
    return M


def plan_with_V_M(V, M, T_new, goal, start, max_steps=500):
    """Plan using D_new = V @ M @ V^T."""
    N = V.shape[0]
    nonterminal = [s for s in range(N) if s != goal]
    nt_idx = np.array(nonterminal)
    t_vec = T_new[np.ix_(nt_idx, [goal])].flatten()
    V_nt = V[nt_idx]
    exp_v = np.maximum(V_nt @ M @ V_nt.T @ t_vec, 1e-10)
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


def eval_barrier_V(V, M, T_new, barrier_states, rows, cols, n_trials=200, seed=42):
    """Evaluate barrier-crossing with V-based planning."""
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
        st, _ = plan_with_V_M(V, M, T_new, g, s)
        steps.append(st)
        n += 1
    arr = np.array(steps)
    return np.mean(arr), np.median(arr), np.sum(arr < 500) / n_trials


# =========================================================================
# BELLMAN TD LEARNING OF M
# =========================================================================

def train_bellman_td_corrected(V, T_new, ec, lam, 
                               valid_states, M_true=None,
                                n_samples=100000, lr=0.01,
                                update_type='semi_gradient',
                                log_every=10000, seed=42):
    """
    Learn M via TD using RAW eigenvectors (not absorbed).
    
    The DR satisfies: e^c * V[s,:] @ M = V[s,:] + E[V[s',:] @ M]
    when M_old = Lambda (the eigenvalue matrix).
    
    TD error:  delta = v(s) + v(s') @ M - e^c * v(s) @ M
    This IS zero in expectation at the true M.
    """
    N = V.shape[0]
    dd = V.shape[1]
    
    # Initialize M to Lambda (the old eigenvalue matrix)
    # NOT identity — because D_old = V @ Lambda @ V^T, not V @ I @ V^T
    M = np.diag(lam[:dd]).copy()
    
    rng = np.random.RandomState(seed)
    history = {'samples': [], 'M_error': [], 'td_loss': []}
    running_loss = 0.0

    for t in range(n_samples):
        s = rng.choice(valid_states)
        sp = rng.choice(N, p=T_new[s, :])

        v_s = V[s, :]
        v_sp = V[sp, :]

        # TD error (zero in expectation at true M)
        delta = v_s + v_sp @ M - ec * v_s @ M

        loss = np.sum(delta ** 2)
        running_loss = 0.99 * running_loss + 0.01 * loss

        if update_type == 'semi_gradient':
            M += lr * np.outer(v_s, delta)
        elif update_type == 'full_gradient':
            grad = 2.0 * np.outer(v_sp - ec * v_s, delta)
            M -= lr * grad

        if (t + 1) % log_every == 0:
            m_err = norm(M - M_true) / norm(M_true) if M_true is not None else None
            m_str = f", M_err={m_err:.4f}" if m_err is not None else ""
            history['samples'].append(t + 1)
            history['M_error'].append(m_err)
            history['td_loss'].append(running_loss)
            print(f"  {t+1:>7d}: td_loss={running_loss:.4f}{m_str}")

    return M, history


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("BELLMAN TD LEARNING OF M")
    print("=" * 70)

    # === 1. Build environments ===
    print("\n--- 1. Environments ---")
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    T_new, barrier_states = make_maze(T_os, BARRIER_TYPE, ROWS, COLS)

    valid_states = np.array([s for s in range(NUM_STATES)
                             if s not in barrier_states])

    print(f"Grid: {ROWS}x{COLS} = {NUM_STATES} states")
    print(f"Barrier states: {len(barrier_states)}")
    print(f"Valid states: {len(valid_states)}")
    D_new_exact = get_complete_DR(T_new)
     
    # === 2. Feature basis ===
    print("\n--- 2. Feature basis ---")
    eigenvalues, eigvecs = eigh(D_os)
    idx = np.argsort(eigenvalues)[::-1][:d]
    lam = eigenvalues[idx]
    V = eigvecs[:, idx]
    F = V * np.sqrt(lam)[np.newaxis, :]    # F = V * sqrt(Lambda)

    ec = 1.0 + 1.0 / lam[0]

    recon_err = norm(F @ F.T - D_os) / norm(D_os)
    print(f"d = {d}")
    print(f"e^c = {ec:.6f}")
    print(f"||F @ F^T - D|| / ||D|| = {recon_err:.2e}")

    # === 3. Ground truth M (for verification only) ===
    print("\n--- 3. Ground truth ---")
    M_true = woodbury_ground_truth(T_os, D_os, T_new, F)
    print(f"||M_true - I|| = {norm(M_true - np.eye(d)):.4f}")

    # For different d values
    # for test_d in [50, 100, 400]:
    #     V_small = V[:, :test_d]
        
    #     # Ground truth M in V-space: M_true = V^T @ D_new @ V
    #     M_true_V = V_small.T @ D_new_exact @ V_small
    #     # M_old in V-space is Lambda (diagonal eigenvalues)
    #     M_old_V = np.diag(lam[:test_d])
        
    #     print(f"\n{'='*70}")
    #     print(f"d = {test_d}")
    #     print(f"{'='*70}")
    #     print(f"||M_true - Lambda|| = {norm(M_true_V - M_old_V):.4f}")
        
    #     # Ground truth planning
    #     mean_gt, med_gt, suc_gt = eval_barrier_V(
    #         V_small, M_true_V, T_new, barrier_states, ROWS, COLS)
    #     mean_nr, med_nr, suc_nr = eval_barrier_V(
    #         V_small, M_old_V, T_new, barrier_states, ROWS, COLS)
    #     print(f"True M:    mean={mean_gt:.1f}, median={med_gt:.1f}, success={suc_gt:.2f}")
    #     print(f"No repair: mean={mean_nr:.1f}, median={med_nr:.1f}, success={suc_nr:.2f}")
        
    #     # Semi-gradient TD
    #     print(f"\nSemi-gradient TD:")
    #     M_learned, hist = train_bellman_td_corrected(
    #         V_small, T_new, ec, lam,
    #         valid_states, M_true=M_true_V,
    #         n_samples=100000, lr=0.001,
    #         update_type='semi_gradient',
    #         log_every=20000,
    #     )
    #     mean_l, med_l, suc_l = eval_barrier_V(
    #         V_small, M_learned, T_new, barrier_states, ROWS, COLS)
    #     print(f"  Learned: mean={mean_l:.1f}, median={med_l:.1f}, success={suc_l:.2f}")
    
    print("\nd=400, semi-gradient TD, higher learning rates:")
    print("=" * 70)

    V_full = V[:, :400]
    M_true_V = V_full.T @ D_new_exact @ V_full

    for lr in [0.01, 0.05]:
        print(f"\n--- lr={lr} ---")
        M_learned, hist = train_bellman_td_corrected(
            V_full, T_new, ec, lam, 
            valid_states, M_true=M_true_V,
            n_samples=200000, lr=lr,
            update_type='semi_gradient',
            log_every=40000,
        )
        mean_l, med_l, suc_l = eval_barrier_V(
            V_full, M_learned, T_new, barrier_states, ROWS, COLS)
        print(f"  Planning: mean={mean_l:.1f}, median={med_l:.1f}, success={suc_l:.2f}")


if __name__ == '__main__':
    main()