import numpy as np
from numpy.linalg import inv, norm, eigh

from utils import create_transition_open, get_complete_DR, make_maze, make_wall_maze

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
    dd = F.shape[1]
    R = (T_new - T_old)[delta_locs, :]
    C = np.eye(N)[:, delta_locs]
    H = R @ F
    G = F.T @ C
    B = H.shape[0]
    Z = H @ G
    A = inv(np.eye(B) - Z)
    M = np.eye(dd) + G @ A @ H
    return M


def plan_with_M(F, M, T_new, goal, start, max_steps=500):
    """Plan in feature space: values = log(F @ M @ F^T @ t)."""
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


def evaluate_barrier_crossing(F, M, T_new, barrier_states,
                               rows, cols, n_trials=200, seed=42):
    """Evaluate on paths that must cross the barrier."""
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
        st, _ = plan_with_M(F, M, T_new, g, s)
        steps_list.append(st)
        n_tested += 1

    steps_arr = np.array(steps_list)
    success = np.sum(steps_arr < 500) / n_trials
    return np.mean(steps_arr), np.median(steps_arr), success


# =========================================================================
# BELLMAN TD LEARNING OF M
# =========================================================================

def train_bellman_td(F, T_new, ec, lam,
                     valid_states,
                     M_true=None,
                     n_samples=100000, lr=0.1, lr_decay=1.0,
                     update_type='semi_gradient',
                     log_every=10000, seed=42):
    """
    Learn M via TD updates from sampled transitions.
    """
    N = F.shape[0]
    dd = F.shape[1]
    M = np.eye(dd)
    rng = np.random.RandomState(seed)

    history = {'samples': [], 'M_error': [], 'td_loss': []}
    running_loss = 0.0
    current_lr = lr
    lam_inv = 1.0 / lam
    
    
    for t in range(n_samples):
        s = rng.choice(valid_states)
        sp = rng.choice(N, p=T_new[s, :])
        
        f_s = F[s, :]
        f_sp = F[sp, :]
        r_s  = f_s * lam_inv

        # TD error
        # delta = f_s + f_sp @ M - ec * f_s @ M
        emc = 1.0 / ec  # e^{-c}, the discount factor < 1
        emc * ec = 1.0
        ec 
        delta = emc * (r_s + f_sp @ M) - f_s @ M

        loss = np.sum(delta ** 2)
        running_loss = 0.99 * running_loss + 0.01 * loss

        # Update M
        if update_type == 'full_gradient':
            grad = 2.0 * np.outer(f_sp - ec * f_s, delta)
            M -= current_lr * grad
        elif update_type == 'semi_gradient':
            M += current_lr * np.outer(f_s, delta)

        current_lr *= lr_decay

        if (t + 1) % log_every == 0:
            m_err = norm(M - M_true) / norm(M_true) if M_true is not None else None
            m_str = f", M_err={m_err:.4f}" if m_err is not None else ""
            history['samples'].append(t + 1)
            history['M_error'].append(m_err)
            history['td_loss'].append(running_loss)
            print(f"  {t+1:>7d} samples: td_loss={running_loss:.4f}{m_str}, lr={current_lr:.6f}")

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
    D_os = (D_os + D_os.T) / 2
    T_new, barrier_states = make_maze(T_os, BARRIER_TYPE, ROWS, COLS)

    valid_states = np.array([s for s in range(NUM_STATES)
                             if s not in barrier_states])

    print(f"Grid: {ROWS}x{COLS} = {NUM_STATES} states")
    print(f"Barrier states: {len(barrier_states)}")
    print(f"Valid states: {len(valid_states)}")

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

    # Baselines
    mean_gt, med_gt, suc_gt = evaluate_barrier_crossing(
        F, M_true, T_new, barrier_states, ROWS, COLS)
    mean_nr, med_nr, suc_nr = evaluate_barrier_crossing(
        F, np.eye(d), T_new, barrier_states, ROWS, COLS)

    print(f"Ground truth M:  mean={mean_gt:.1f}, median={med_gt:.1f}, success={suc_gt:.2f}")
    print(f"No repair (M=I): mean={mean_nr:.1f}, median={med_nr:.1f}, success={suc_nr:.2f}")
    
    print("=" * 70)
    print("COMPARISON: Reduced feature space")
    print("=" * 70)
    # Recompute F at reduced d
    for test_d in [50, 100, 400]:
        dd = test_d
        lam_dd = lam[:dd]
        F = V[:, :test_d] * np.sqrt(lam[:test_d])[np.newaxis, :]
        recon = norm(F @ F.T - D_os) / norm(D_os)
        M_true = woodbury_ground_truth(T_os, D_os, T_new, F)

        print(f"\n{'='*70}")
        print(f"d = {test_d}, ||FF^T - D||/||D|| = {recon:.4f}")
        print(f"||M_true - I|| = {norm(M_true - np.eye(dd)):.4f}")
        print(f"{'='*70}")

        # Baselines
        mean_gt, med_gt, suc_gt = evaluate_barrier_crossing(
            F, M_true, T_new, barrier_states, ROWS, COLS)
        mean_nr, med_nr, suc_nr = evaluate_barrier_crossing(
            F, np.eye(dd), T_new, barrier_states, ROWS, COLS)
        print(f"True M:    mean={mean_gt:.1f}, median={med_gt:.1f}, success={suc_gt:.2f}")
        print(f"No repair: mean={mean_nr:.1f}, median={med_nr:.1f}, success={suc_nr:.2f}")
        
        print(f"\n--- Semi-gradient TD, d={test_d}, lr=0.01 ---")
        M_test, hist = train_bellman_td(
            F, T_new, ec, lam_dd,
            valid_states,
            M_true=M_true,
            n_samples=200000, lr=0.01, lr_decay=1.0,
            update_type='semi_gradient',
            log_every=50000,
        )
        mean_s, med_s, suc_s = evaluate_barrier_crossing(
            F, M_test, T_new, barrier_states, ROWS, COLS)
        print(f"  Planning: mean={mean_s:.1f}, median={med_s:.1f}, success={suc_s:.2f}")
        
        print(f"\n--- Full-gradient TD, d={test_d}, lr=0.01 ---")
        M_test, hist = train_bellman_td(
            F, T_new, ec, lam_dd,
            valid_states,
            M_true=M_true,
            n_samples=200000, lr=0.01, lr_decay=1.0,
            update_type='full_gradient',
            log_every=50000,
        )
        mean_s, med_s, suc_s = evaluate_barrier_crossing(
            F, M_test, T_new, barrier_states, ROWS, COLS)
        print(f"  Planning: mean={mean_s:.1f}, median={med_s:.1f}, success={suc_s:.2f}")


if __name__ == '__main__':
    main()