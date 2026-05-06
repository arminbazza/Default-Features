"""
T-free cognitive map update from local experience.

The agent updates its cognitive map after a barrier appears, using ONLY:
1. U_os, eigenvalues (pre-learned from old environment)
2. Sampled transitions from T_new (the agent lives in this environment)
3. Barrier states discovered through exploration (hitting walls)
4. Grid topology (the agent knows its neighborhood structure)

T_old is NEVER used in the algorithm. It only appears in the verification
section to confirm correctness against ground truth.

Approach: Combined Bellman + Eq 16
- Barrier states: H rows computed from topology + U_os (fixed)
- Neighbor states: decomposed into (s, s_blocked, s_experienced) triples,
  one learnable weight per triple
- Bellman SGD gives rough M, Eq 16 extracts W, Woodbury rebuilds M
"""

import numpy as np
from numpy.linalg import inv, norm
from utils import create_transition_open, get_complete_DR, make_maze

ROWS, COLS = 20, 20
NUM_STATES = ROWS * COLS
BARRIER_TYPE = 'H'
d = NUM_STATES


# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================

def woodbury(T_old, D, T_new):
    """Ground truth Woodbury. ONLY for verification."""
    diffs = T_old != T_new
    diff_rows = np.unique(np.where(diffs)[0])
    N = T_old.shape[0]
    R = (T_new - T_old)[diff_rows, :]
    C = np.eye(N)[:, diff_rows]
    Z = R @ D @ C
    A = inv(np.eye(Z.shape[0]) - Z)
    D_new = D + D @ C @ A @ R @ D
    return D_new, R, C, diff_rows


def compute_M_from_H(H, G, d):
    """M = I_d + G @ (I_B - H @ G)^{-1} @ H"""
    B = H.shape[0]
    Z = H @ G
    A = inv(np.eye(B) - Z)
    return np.eye(d) + G @ A @ H


def plan_with_M(U_os, M, T_new, goal, start, max_steps=500):
    """Plan in feature space using corrected M."""
    N = U_os.shape[0]
    nonterminal = [s for s in range(N) if s != goal]
    nt_idx = np.array(nonterminal)
    t_vec = T_new[np.ix_(nt_idx, [goal])].flatten()
    U_nt = U_os[nt_idx]
    exp_v = np.maximum(U_nt @ M @ U_nt.T @ t_vec, 1e-10)
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


def evaluate_barrier_crossing(U_os, M, T_new, barrier_states,
                               rows, cols, n_trials=200, seed=42):
    """Evaluate planning on paths that must cross the barrier."""
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
        if not ((s_row < barrier_row and g_row >= barrier_row) or
                (s_row >= barrier_row and g_row < barrier_row)):
            continue
        st, _ = plan_with_M(U_os, M, T_new, g, s)
        steps_list.append(st)
        n_tested += 1
    steps_arr = np.array(steps_list)
    success = np.sum(steps_arr < 500) / n_trials
    return np.mean(steps_arr), np.median(steps_arr), success


# =========================================================================
# PHASE 1: EXPLORATION (T-free)
# =========================================================================

def explore_barrier(T_new, barrier_states, grid_shape,
                    n_samples_per_state=20, seed=42):
    """
    Agent explores the environment and discovers:
    - Which accessible states are adjacent to barrier states
    - Which successors are blocked vs reachable (from sampling)
    
    Returns list of (s, s_blocked, s_experienced) triples.
    
    Uses ONLY T_new for sampling and barrier_states from exploration.
    """
    rows, cols = grid_shape
    N = rows * cols
    barrier_set = set(barrier_states)
    rng = np.random.RandomState(seed)
    triples = []

    for s in range(N):
        if s in barrier_set:
            continue

        r, c = s // cols, s % cols
        blocked = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = nr * cols + nc
                if nb in barrier_set:
                    blocked.append(nb)

        if not blocked:
            continue

        experienced = set()
        for _ in range(n_samples_per_state):
            s_next = rng.choice(N, p=T_new[s, :])
            experienced.add(s_next)

        for b in blocked:
            for e in experienced:
                triples.append((s, b, e))

    return triples


# =========================================================================
# PHASE 2: BUILD FEATURE-SPACE PROJECTIONS (T-free)
# =========================================================================

def compute_barrier_H(barrier_states, U_os, grid_shape):
    """
    For completely inaccessible barrier states, compute H rows from
    grid topology and U_os. No T_old needed.
    
    In a uniform random walk: T_old[b, neighbor] = 1/n_neighbors.
    Since T_new[b,:] = 0 (blocked), H[b,:] = -T_old[b,:] @ U_os
    = -(1/n_neighbors) * sum(u(neighbor)).
    """
    rows, cols = grid_shape
    d = U_os.shape[1]
    n_barrier = len(barrier_states)
    H_barrier = np.zeros((n_barrier, d))
    phi_C_barrier = np.zeros((d, n_barrier))

    for i, b in enumerate(barrier_states):
        r, c = b // cols, b % cols
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append(nr * cols + nc)
        H_barrier[i, :] = -(1.0 / len(neighbors)) * np.sum(U_os[neighbors, :], axis=0)
        phi_C_barrier[:, i] = U_os[b, :]

    return H_barrier, phi_C_barrier


def build_triple_projections(triples, U_os):
    """
    From (s, s_blocked, s_experienced) triples, build:
    - phi_C_triples: (d, B_triples), column b = u(s)
    - delta_Phi:     (d, B_triples), column b = u(s_exp) - u(s_blocked)
    
    Uses ONLY U_os.
    """
    d = U_os.shape[1]
    B = len(triples)
    phi_C = np.zeros((d, B))
    delta_Phi = np.zeros((d, B))

    for b, (s, s_blocked, s_exp) in enumerate(triples):
        phi_C[:, b] = U_os[s, :]
        delta_Phi[:, b] = U_os[s_exp, :] - U_os[s_blocked, :]

    return phi_C, delta_Phi


# =========================================================================
# PHASE 3: COMBINED BELLMAN + EQ 16 (T-free)
# =========================================================================

def train_combined(U_os, T_new, ec, d, valid_states,
                   phi_C_full, H_barrier, delta_Phi,
                   barrier_idx, neighbor_idx, barrier_states,
                   M_true=None,
                   n_outer=10, bellman_samples_per_round=10000,
                   bellman_lr=0.1, eq16_iters=200, eq16_lr=0.01,
                   seed=42):
    """
    Combined Bellman + Eq 16, fully T-free.

    phi_C_full: (d, B_total) — barrier + triple columns
    H_barrier:  (n_barrier, d) — fixed rows for barrier states
    delta_Phi:  (d, B_triples) — feature contrasts for triples
    barrier_idx:  indices into full H for barrier rows
    neighbor_idx: indices into full H for triple rows
    """
    N = U_os.shape[0]
    B_total = phi_C_full.shape[1]
    B_triples = delta_Phi.shape[1]
    M = np.eye(d)
    theta = np.zeros(B_triples)
    rng = np.random.RandomState(seed)
    total_samples = 0

    def build_H_full(theta):
        w = np.tanh(theta)
        H = np.zeros((B_total, d))
        H[barrier_idx, :] = H_barrier
        H[neighbor_idx, :] = np.diag(w) @ delta_Phi.T
        return H

    for outer in range(n_outer):
        # === Stage 1: Bellman SGD ===
        for t in range(bellman_samples_per_round):
            s = rng.choice(valid_states)
            s_prime = rng.choice(N, p=T_new[s, :])
            f_s = U_os[s, :]
            f_sp = U_os[s_prime, :]
            residual = f_s + f_sp @ M - ec * f_s @ M
            grad_M = 2.0 * np.outer(f_sp - ec * f_s, residual)
            M -= bellman_lr * grad_M

        total_samples += bellman_samples_per_round
        m_str = ""
        if M_true is not None:
            m_str = f", M_err={norm(M - M_true) / norm(M_true):.4f}"
        print(f"  Round {outer+1} after Bellman ({total_samples} samples){m_str}")

        # === Stage 2: Fix M, optimize W (triple weights only) ===
        phi_C_nb = phi_C_full[:, neighbor_idx]  # (d, B_triples)

        for it in range(eq16_iters):
            w = np.tanh(theta)
            H = build_H_full(theta)
            residual = np.eye(d) + phi_C_full @ H @ M - M
            loss = np.sum(residual ** 2)

            temp1 = phi_C_nb.T @ residual           # (B_triples, d)
            temp2 = (M.T @ delta_Phi).T              # (B_triples, d)
            grad_w = 2.0 * np.sum(temp1 * temp2, axis=1)
            grad_theta = grad_w * (1.0 - w ** 2)
            theta -= eq16_lr * grad_theta

        w_learned = np.tanh(theta)
        print(f"           after Eq 16: w_mean={np.mean(w_learned):.4f}, "
              f"w_std={np.std(w_learned):.4f}, loss={loss:.2e}")

        # === Stage 3: Rebuild M via Woodbury ===
        H = build_H_full(theta)
        M = compute_M_from_H(H, phi_C_full, d)

        m_str = ""
        if M_true is not None:
            m_str = f"M_err={norm(M - M_true) / norm(M_true):.4f}, "

        mean_s, med_s, suc_s = evaluate_barrier_crossing(
            U_os, M, T_new, barrier_states, ROWS, COLS)
        print(f"           after rebuild: {m_str}"
              f"planning: mean={mean_s:.1f}, median={med_s:.1f}, success={suc_s:.2f}")
        print()

    return M, theta


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("T-FREE COGNITIVE MAP UPDATE FROM LOCAL EXPERIENCE")
    print("=" * 70)

    # === SETUP ===
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)

    eigenvalues, eigvecs = np.linalg.eigh(D_os)
    idx = np.argsort(eigenvalues)[::-1][:d]
    V = eigvecs[:, idx]
    lam = eigenvalues[idx]
    U_os = V * np.sqrt(lam)[np.newaxis, :]

    ec = 1.0 + 1.0 / lam[0]

    T_new, barrier_states = make_maze(T_os, BARRIER_TYPE, ROWS, COLS)
    valid_states = np.array([s for s in range(NUM_STATES) if s not in barrier_states])

    # Ground truth (ONLY for verification)
    D_new, R, C, delta_locs = woodbury(T_os, D_os, T_new)
    phi_C_gt = U_os.T @ C
    psi_R_gt = R @ U_os
    M_true = compute_M_from_H(psi_R_gt, phi_C_gt, d)

    print(f"\nGrid: {ROWS}x{COLS} = {NUM_STATES} states")
    print(f"Eigenbasis: d = {d}")
    print(f"e^c = {ec:.6f}")
    print(f"D reconstruction error: {norm(U_os @ U_os.T - D_os) / norm(D_os):.2e}")

    # === PHASE 1: EXPLORATION ===
    print("\n--- Phase 1: Exploration (T-free) ---")
    triples = explore_barrier(T_new, barrier_states, (ROWS, COLS), n_samples_per_state=20)

    print(f"Barrier states: {len(barrier_states)}")
    print(f"Triples discovered: {len(triples)}")
    print(f"  Unique neighbor states: {len(set(t[0] for t in triples))}")
    print(f"  Unique blocked:         {len(set(t[1] for t in triples))}")
    print(f"  Unique experienced:     {len(set(t[2] for t in triples))}")

    # === PHASE 2: FEATURE-SPACE PROJECTIONS ===
    print("\n--- Phase 2: Feature-space projections (T-free) ---")

    # Barrier rows: from topology + U_os
    H_barrier, phi_C_barrier = compute_barrier_H(barrier_states, U_os, (ROWS, COLS))
    n_barrier = len(barrier_states)

    # Triple rows: from exploration + U_os
    phi_C_triples, delta_Phi = build_triple_projections(triples, U_os)
    n_triples = len(triples)

    # Combine into full phi_C
    phi_C_full = np.hstack([phi_C_barrier, phi_C_triples])  # (d, n_barrier + n_triples)
    B_total = phi_C_full.shape[1]
    barrier_idx = np.arange(n_barrier)
    neighbor_idx = np.arange(n_barrier, B_total)

    print(f"Total components: {B_total} = {n_barrier} barrier (fixed) + {n_triples} triples (learnable)")

    # === VERIFICATION ===
    print("\n--- Verification (uses T_old, not part of algorithm) ---")

    # Check barrier H matches ground truth
    is_barrier = np.array([loc in barrier_states for loc in delta_locs])
    barrier_idx_gt = np.where(is_barrier)[0]
    print(f"  ||H_barrier_topo - H_barrier_exact||: "
          f"{norm(H_barrier - psi_R_gt[barrier_idx_gt]):.2e}")

    # Check full reconstruction with true weights
    w_true_triples = np.zeros(n_triples)
    for i, (s, s_blocked, s_exp) in enumerate(triples):
        w_true_triples[i] = T_new[s, s_exp] - T_os[s, s_exp]

    H_full_true = np.zeros((B_total, d))
    H_full_true[barrier_idx, :] = H_barrier
    H_full_true[neighbor_idx, :] = np.diag(w_true_triples) @ delta_Phi.T
    M_from_triples = compute_M_from_H(H_full_true, phi_C_full, d)

    mean_gt, med_gt, suc_gt = evaluate_barrier_crossing(
        U_os, M_true, T_new, barrier_states, ROWS, COLS)
    mean_tr, med_tr, suc_tr = evaluate_barrier_crossing(
        U_os, M_from_triples, T_new, barrier_states, ROWS, COLS)
    mean_nr, med_nr, suc_nr = evaluate_barrier_crossing(
        U_os, np.eye(d), T_new, barrier_states, ROWS, COLS)

    print(f"  Ground truth M:            mean={mean_gt:.1f}, median={med_gt:.1f}, success={suc_gt:.2f}")
    print(f"  M from triples (true W):   mean={mean_tr:.1f}, median={med_tr:.1f}, success={suc_tr:.2f}")
    print(f"  No repair (M=I):           mean={mean_nr:.1f}, median={med_nr:.1f}, success={suc_nr:.2f}")

    # === PHASE 3: COMBINED BELLMAN + EQ 16 ===
    print("\n--- Phase 3: Combined Bellman + Eq 16 (T-free) ---")
    print("=" * 70)

    M_learned, theta_learned = train_combined(
        U_os, T_new, ec, d, valid_states,
        phi_C_full, H_barrier, delta_Phi,
        barrier_idx, neighbor_idx, barrier_states,
        M_true=M_true,
        n_outer=10, bellman_samples_per_round=10000,
        bellman_lr=0.1, eq16_iters=200, eq16_lr=0.01
    )

    # === SUMMARY ===
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mean_l, med_l, suc_l = evaluate_barrier_crossing(
        U_os, M_learned, T_new, barrier_states, ROWS, COLS)

    print(f"  Ground truth (exact Woodbury):       mean={mean_gt:.1f}, median={med_gt:.1f}, success={suc_gt:.2f}")
    print(f"  Combined Bellman + Eq 16 (T-free):   mean={mean_l:.1f}, median={med_l:.1f}, success={suc_l:.2f}")
    print(f"  No repair:                           mean={mean_nr:.1f}, median={med_nr:.1f}, success={suc_nr:.2f}")

    
    # === Consolidate ===
    print("=" * 70)
    print("CONSOLIDATE")
    print("=" * 70)

    def consolidate_triples(triples, U_os):
        """
        Group triples by (s, s_blocked). Each group shares one weight.
        delta_Phi column = sum of [u(s_exp) - u(s_blocked)] over all experienced successors.
        phi_C column = u(s).
        """
        d = U_os.shape[1]
        groups = {}
        for s, s_blocked, s_exp in triples:
            key = (s, s_blocked)
            if key not in groups:
                groups[key] = []
            groups[key].append(s_exp)

        B_compact = len(groups)
        phi_C = np.zeros((d, B_compact))
        delta_Phi = np.zeros((d, B_compact))
        group_info = []

        for b, ((s, s_blocked), exp_list) in enumerate(groups.items()):
            phi_C[:, b] = U_os[s, :]
            for e in exp_list:
                delta_Phi[:, b] += U_os[e, :] - U_os[s_blocked, :]
            group_info.append((s, s_blocked, exp_list))

        return phi_C, delta_Phi, group_info


    # === Build consolidated projections ===
    phi_C_compact, delta_Phi_compact, group_info = consolidate_triples(triples, U_os)
    n_groups = phi_C_compact.shape[1]

    # Combine with barrier
    phi_C_full_compact = np.hstack([phi_C_barrier, phi_C_compact])
    B_total_compact = phi_C_full_compact.shape[1]
    barrier_idx_c = np.arange(n_barrier)
    neighbor_idx_c = np.arange(n_barrier, B_total_compact)

    print(f"Consolidated: {len(triples)} triples -> {n_groups} groups")
    print(f"Total components: {B_total_compact} = {n_barrier} barrier + {n_groups} groups")

    # === Verify ===
    w_true_compact = np.zeros(n_groups)
    for b, (s, s_blocked, exp_list) in enumerate(group_info):
        w_true_compact[b] = T_new[s, exp_list[0]] - T_os[s, exp_list[0]]

    H_full_compact_true = np.zeros((B_total_compact, d))
    H_full_compact_true[barrier_idx_c, :] = H_barrier
    H_full_compact_true[neighbor_idx_c, :] = np.diag(w_true_compact) @ delta_Phi_compact.T
    M_compact_true = compute_M_from_H(H_full_compact_true, phi_C_full_compact, d)

    mean_vc, med_vc, suc_vc = evaluate_barrier_crossing(
        U_os, M_compact_true, T_new, barrier_states, ROWS, COLS)
    print(f"Verification (true W, compact): mean={mean_vc:.1f}, median={med_vc:.1f}, success={suc_vc:.2f}")

    # === Train ===
    print("\nCombined Bellman + Eq 16 (compact + anchor, T-free):")
    print("=" * 65)

    M_compact, theta_compact = train_combined(
        U_os, T_new, ec, d, valid_states,
        phi_C_full_compact, H_barrier, delta_Phi_compact,
        barrier_idx_c, neighbor_idx_c, barrier_states,
        M_true=M_true,
        n_outer=10, bellman_samples_per_round=10000,
        bellman_lr=0.1, eq16_iters=200, eq16_lr=0.01
    )

    mean_c, med_c, suc_c = evaluate_barrier_crossing(
        U_os, M_compact, T_new, barrier_states, ROWS, COLS)
    print(f"\nCompact + anchor: mean={mean_c:.1f}, median={med_c:.1f}, success={suc_c:.2f}")


    w_learned = np.tanh(theta_learned)
    print(f"\n  Learned W: mean={np.mean(w_learned):.4f}, std={np.std(w_learned):.4f}")
    print(f"  True W:    mean={np.mean(w_true_triples):.4f}, std={np.std(w_true_triples):.4f}")

    print(f"\n  Information used by algorithm:")
    print(f"    U_os (pre-learned eigenbasis):       YES")
    print(f"    Eigenvalues of D (for e^c):          YES")
    print(f"    Grid topology (for barrier H rows):  YES")
    print(f"    T_new (sampled transitions):          YES")
    print(f"    Barrier states (from exploration):    YES")
    print(f"    T_old:                               NO")
    print(f"    R (transition differences):           NO")


if __name__ == '__main__':
    main()