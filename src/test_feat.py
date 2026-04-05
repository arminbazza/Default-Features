"""
Cognitive map update from local experience.
"""

import numpy as np
from numpy.linalg import inv, norm, eigvals

from utils import (create_transition_open, get_complete_DR, make_maze,
                   evaluate_planning, evaluate_planning_barrier_crossing,
                   plan_with_D)


ROWS, COLS = 20, 20
NUM_STATES = ROWS * COLS
BARRIER_TYPE = 'H'
d = 400


# --- Helper functions ---
def plan_with_absorbed_eigenbasis(U_os, C, R, T_new, goal_state, start_state, max_steps=500):
    """
    Plan using Woodbury update with eigenvalues absorbed into the basis.
    
    With U_os = V @ diag(sqrt(lambda)), we have D_os ≈ U_os @ U_os.T.
    
    The Woodbury correction (Eq. 2 from the paper) becomes:
    
        G = U_os + U_os @ phi_C @ A @ psi_R
        D_new = G @ U_os.T
        
    For planning we only need exp_v = D_new_nt @ t, which decomposes as:
    
        exp_v = U_os_nt @ (w_base + phi_C @ A @ z_barrier)
        
    where:
        w_base = U_os_nt.T @ t              (d-dim, base planning)
        z_barrier = psi_R @ w_base          (O-dim, barrier effect on goal paths)
        correction = phi_C @ A @ z_barrier  (d-dim, correction in feature space)
    
    No separate Lambda matrix anywhere.
    """
    N = U_os.shape[0]
    num_affected = C.shape[1]
    nonterminal = [s for s in range(N) if s != goal_state]
    nt_idx = np.array(nonterminal)

    # Project barrier matrices into absorbed eigenbasis
    phi_C = U_os.T @ C       # d x O
    psi_R = R @ U_os          # O x d

    # POR: O x O
    Z = psi_R @ phi_C
    A = np.linalg.inv(np.eye(num_affected) - Z)

    # Planning
    t = T_new[np.ix_(nonterminal, [goal_state])].flatten()
    U_os_nt = U_os[nt_idx]   # (N-1) x d

    w_base = U_os_nt.T @ t                        # d-dim
    z_barrier = psi_R @ w_base                     # O-dim
    w_total = w_base + phi_C @ A @ z_barrier       # d-dim

    exp_v = U_os_nt @ w_total
    exp_v = np.maximum(exp_v, 1e-10)
    v = np.log(exp_v)

    values = np.full(N, -np.inf)
    for i, s in enumerate(nonterminal):
        values[s] = v[i]
    values[goal_state] = 10.0

    state = start_state
    path = [state]
    for step in range(max_steps):
        if state == goal_state:
            return step, path
        neighbors = np.where(T_new[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = neighbors[np.argmax(values[neighbors])]
        path.append(state)
    return max_steps, path

def woodbury(T_old, D, T_new):
    """Performs woodbury update in full space (ground truth)."""
    diffs = T_old != T_new
    diff_rows, _ = np.where(diffs)
    delta_locs = np.unique(diff_rows)

    N = T_old.shape[0]
    R = (T_new - T_old)[delta_locs, :]
    C = np.eye(N)[:, delta_locs]

    Z = R @ D @ C
    A = np.linalg.inv(np.eye(Z.shape[0]) - Z)
    woodbury_update = D @ C @ A @ R @ D

    D_new = D + woodbury_update
    return D_new, R, C, delta_locs

def build_psi_R(w, delta_Phi, psi_R_barrier, barrier_idx, neighbor_idx, num_affected, d):
    """Build full H matrix: fixed barrier rows + parameterized neighbor rows."""
    psi_R = np.zeros((num_affected, d))
    psi_R[barrier_idx, :] = psi_R_barrier
    psi_R[neighbor_idx, :] = np.diag(w) @ delta_Phi.T
    return psi_R

def compute_M_from_H(H_hat, phi_C, d):
    """Compute M = I + phi_C @ A @ H via exact inverse (Eq. 9)."""
    B = H_hat.shape[0]
    Z = H_hat @ phi_C          # B x B
    A = np.linalg.inv(np.eye(B) - Z)
    M = np.eye(d) + phi_C @ A @ H_hat
    return M

def plan_with_M(U_os, M, T_new, goal, start, max_steps=500):
    """Plan using a given M operator in feature space."""
    N = U_os.shape[0]
    nonterminal = [s for s in range(N) if s != goal]
    nt_idx = np.array(nonterminal)
    
    t_vec = T_new[np.ix_(nt_idx, [goal])].flatten()
    U_nt = U_os[nt_idx]
    
    w_base = U_nt.T @ t_vec
    w_total = M @ w_base
    exp_v = U_nt @ w_total
    exp_v = np.maximum(exp_v, 1e-10)
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
    
    steps = len(path) - 1 if state == goal else max_steps
    return steps, path


def evaluate_barrier_crossing(U_os, M, T_new, barrier_states, 
                               rows, cols, n_trials=200, seed=42):
    """
    Evaluate planning on start-goal pairs that must cross the barrier.
    Returns mean steps, median steps, success rate.
    """
    N = rows * cols
    barrier_row = rows // 2
    valid_states = [s for s in range(N) if s not in barrier_states]
    rng = np.random.RandomState(seed)
    
    steps_list = []
    n_tested = 0
    
    while n_tested < n_trials:
        s = rng.choice(valid_states)
        g = rng.choice(valid_states)
        if s == g:
            continue
        s_row, g_row = s // cols, g // cols
        # Must cross barrier
        if not ((s_row < barrier_row and g_row >= barrier_row) or
                (s_row >= barrier_row and g_row < barrier_row)):
            continue
        
        st, _ = plan_with_M(U_os, M, T_new, g, s)
        steps_list.append(st)
        n_tested += 1
    
    steps_arr = np.array(steps_list)
    success = np.sum(steps_arr < 500) / n_trials
    return np.mean(steps_arr), np.median(steps_arr), success

def compute_loss_and_grad_fixed_M(theta, M_fixed, phi_C, delta_Phi, psi_R_barrier,
                                   barrier_idx, neighbor_idx, num_affected, d):
    """
    Loss with M held completely fixed (not derived from theta at all).
    L = ||I + G @ H(W(theta)) @ M_fixed - M_fixed||^2
    """
    w = np.tanh(theta)
    psi_R = build_psi_R(w, delta_Phi, psi_R_barrier, barrier_idx, neighbor_idx, num_affected, d)
    
    residual = np.eye(d) + phi_C @ psi_R @ M_fixed - M_fixed
    loss = np.sum(residual ** 2)
    
    # Gradient: same formula but M_bar replaced by M_fixed
    phi_C_neighbors = phi_C[:, neighbor_idx]
    M_fixed_delta = M_fixed.T @ delta_Phi
    temp1 = phi_C_neighbors.T @ residual
    grad_w = 2.0 * np.sum(temp1 * M_fixed_delta.T, axis=1)
    grad_theta = grad_w * (1.0 - np.tanh(theta) ** 2)
    
    return loss, grad_theta


# --- Combined Bellman + Eq 16 approach ---
def train_combined(U_os, T_new, ec, d, barrier_states,
                   phi_C, delta_Phi, psi_R_barrier,
                   barrier_idx, neighbor_idx, num_affected,
                   w_true, M_true,
                   n_outer=10, bellman_samples_per_round=10000,
                   bellman_lr=0.1, eq16_iters=200, eq16_lr=0.01,
                   seed=42):
    """
    Alternating:
    1. Run Bellman SGD on M for a batch of transitions
    2. Fix M, optimize W via Eq 16: ||I + GH(W)M - M||^2
    3. Rebuild M from W via Woodbury: M = I + G(I - HG)^{-1}H
    4. Repeat
    """
    N = U_os.shape[0]
    M = np.eye(d)
    theta = np.zeros(len(w_true))
    rng = np.random.RandomState(seed)
    total_samples = 0
    valid_states = np.array([s for s in range(NUM_STATES) if s not in barrier_states])
    
    history = {
        'w_error': [], 'M_error': [], 'planning_steps': [],
        'total_samples': [], 'stage': []
    }
    
    for outer in range(n_outer):
        ## Stage 1: Bellman updates on M
        for t in range(bellman_samples_per_round):
            s = rng.choice(valid_states)
            s_prime = rng.choice(N, p=T_new[s, :])
            
            f_s = U_os[s, :]
            f_sp = U_os[s_prime, :]
            
            residual = f_s + f_sp @ M - ec * f_s @ M
            grad_M = 2.0 * np.outer(f_sp - ec * f_s, residual)
            M -= bellman_lr * grad_M
        
        total_samples += bellman_samples_per_round
        
        # Log after Bellman
        m_err = np.linalg.norm(M - M_true) / np.linalg.norm(M_true)
        w_err = np.linalg.norm(np.tanh(theta) - w_true)
        history['M_error'].append(m_err)
        history['w_error'].append(w_err)
        history['total_samples'].append(total_samples)
        history['stage'].append('bellman')
        
        print(f"  Round {outer+1} after Bellman ({total_samples} samples): "
              f"M_err={m_err:.4f}, w_err={w_err:.4f}")
        
        ## Stage 2: Fix M, optimize W via Eq 16
        for it in range(eq16_iters):
            loss, grad = compute_loss_and_grad_fixed_M(
                theta, M, phi_C, delta_Phi, psi_R_barrier,
                barrier_idx, neighbor_idx, num_affected, d
            )
            theta -= eq16_lr * grad
        
        w_learned = np.tanh(theta)
        w_err = np.linalg.norm(w_learned - w_true)
        
        print(f"           after Eq 16: w_err={w_err:.4f}, "
              f"w_mean={np.mean(w_learned):.4f}, loss={loss:.2e}")
        
        ## Stage 3: Rebuild M from W via Woodbury
        psi_R_learned = build_psi_R(w_learned, delta_Phi, psi_R_barrier,
                                     barrier_idx, neighbor_idx, num_affected, d)
        M = compute_M_from_H(psi_R_learned, phi_C, d)
        
        m_err = np.linalg.norm(M - M_true) / np.linalg.norm(M_true)
        history['M_error'].append(m_err)
        history['w_error'].append(w_err)
        history['total_samples'].append(total_samples)
        history['stage'].append('rebuild')
        
        # Evaluate planning
        mean_s, med_s, suc_s = evaluate_barrier_crossing(
            U_os, M, T_new, barrier_states, ROWS, COLS)
        history['planning_steps'].append(mean_s)
        
        print(f"           after rebuild: M_err={m_err:.4f}, "
              f"planning: mean={mean_s:.1f}, median={med_s:.1f}, success={suc_s:.2f}")
        print()
    
    return M, theta, history


def main():
    # --- Build open-space environment ---
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)

    # --- Build absorbed eigenbasis ---
    eigenvalues, eigvecs = np.linalg.eigh(D_os)
    idx = np.argsort(eigenvalues)[::-1][:d]
    V = eigvecs[:, idx]
    lam = eigenvalues[idx]
    U_os = V * np.sqrt(lam)[np.newaxis, :]  # (N, d)

    # --- Add barrier ---
    T_new, barrier_states = make_maze(T_os, BARRIER_TYPE, ROWS, COLS)
    D_new, R, C, delta_locs = woodbury(T_os, D_os, T_new)
    num_affected = len(delta_locs)
    
    # Exact projections (model-based, for verification)
    phi_C = U_os.T @ C              # (d, num_affected) — G in the notes
    psi_R_exact = R @ U_os          # (num_affected, d) — H in the notes

    # Split affected states into barrier (fixed) vs neighbor (learnable)
    is_barrier = np.array([loc in barrier_states for loc in delta_locs])
    barrier_idx = np.where(is_barrier)[0]
    neighbor_idx = np.where(~is_barrier)[0]
    n_barrier = len(barrier_idx)
    n_neighbors = len(neighbor_idx)

    # --- Barrier states: H rows are fully known, no W to learn ---
    psi_R_barrier = psi_R_exact[barrier_idx, :] # (n_barrier, d)

    # --- Neighbor states: decompose R[b,:] = w_b * r_hat[b,:] ---
    R_neighbors = R[neighbor_idx, :]            # (n_neighbors, NUM_STATES)
    w_true = np.zeros(n_neighbors)
    delta_Phi = np.zeros((d, n_neighbors))

    for i in range(n_neighbors):
        row = R_neighbors[i, :]
        total_lost = np.sum(np.abs(row[row < 0]))  # total redirected mass
        w_true[i] = total_lost
        r_hat_i = row / total_lost              # normalized direction
        delta_Phi[:, i] = r_hat_i @ U_os        # feature-space direction

    # --- Verify: parameterized H matches exact H ---
    psi_R_reconstructed = np.zeros_like(psi_R_exact)
    psi_R_reconstructed[barrier_idx, :] = psi_R_barrier
    psi_R_reconstructed[neighbor_idx, :] = np.diag(w_true) @ delta_Phi.T

    print(f"Total affected states: {num_affected}")
    print(f"Barrier states (H fixed):      {n_barrier}")
    print(f"Neighbor states (W learnable):  {n_neighbors}")
    print(f"True weights (unique values):   {np.unique(np.round(w_true, 4))}")
    print(f"||H_reconstructed - H_exact||:  {np.linalg.norm(psi_R_reconstructed - psi_R_exact):.2e}")
    
    ec = 1.0 + 1.0 / lam[0]
    # Ground truth M from exact H
    M_true = compute_M_from_H(psi_R_exact, phi_C, d)
    
    # --- Run combined approach ---
    print("Combined Bellman + Eq 16 approach")
    print("=" * 70)

    M_combined, theta_combined, hist_combined = train_combined(
        U_os, T_new, ec, d, barrier_states,
        phi_C, delta_Phi, psi_R_barrier,
        barrier_idx, neighbor_idx, num_affected,
        w_true, M_true,
        n_outer=10, bellman_samples_per_round=10000,
        bellman_lr=0.1, eq16_iters=200, eq16_lr=0.01
    )

    w_combined = np.tanh(theta_combined)
    print(f"\nFinal W: {np.round(w_combined[:5], 4)} (true: {w_true[:5]})")
    print(f"\nComparison:")
    print(f"  Exact M:                       mean=18.3, median=18.0, success=1.00")
    print(f"  No repair:                     mean=184.5, median=23.0, success=0.66")
    print(f"  Prediction error (22 samples): mean=18.5, median=18.0, success=1.00")

if __name__ == '__main__':
    main()