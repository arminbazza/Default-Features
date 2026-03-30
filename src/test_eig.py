import numpy as np
from utils import (create_transition_open, get_complete_DR, make_maze,
                   evaluate_planning, evaluate_planning_barrier_crossing,
                   plan_with_D)

ROWS, COLS = 20, 20
NUM_STATES = ROWS * COLS
BARRIER_TYPE = 'H'
d = 400


def woodbury(T_old, D, T_new):
    """
    Performs woodbury update in full space (ground truth).
    """
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


# === Setup ===
T_os = create_transition_open(shape=(ROWS, COLS))
D_os = get_complete_DR(T=T_os)
T_new, barrier_states = make_maze(T_os, BARRIER_TYPE, ROWS, COLS)
D_new, R, C, delta_locs = woodbury(T_os, D_os, T_new)
num_affected = len(delta_locs)

# === Build absorbed eigenbasis ===
eigenvalues, eigvecs = np.linalg.eigh(D_os)
idx = np.argsort(eigenvalues)[::-1][:d]
V = eigvecs[:, idx]
lam = np.maximum(eigenvalues[idx], 1e-10)  # clip negative numerics

# Absorb eigenvalues: U_os = V * sqrt(eigenvalues)
U_os = V * np.sqrt(lam)[np.newaxis, :]  # S x d

# Verify reconstruction quality
D_os_reconstructed = U_os @ U_os.T
recon_error = np.linalg.norm(D_os - D_os_reconstructed) / np.linalg.norm(D_os)
print(f"D_os reconstruction error with d={d}: {recon_error:.6f}")

# === Evaluate ===
start_state = 0
goal_state = NUM_STATES - 1

print("\n--- Single path: top-left to bottom-right ---")
steps_true, _ = plan_with_D(D_new, T_new, goal_state, start_state)
print(f"  Ground truth (full Woodbury):  {steps_true} steps")

steps_os, _ = plan_with_D(D_os, T_new, goal_state, start_state)
print(f"  No repair (D_os):              {steps_os} steps")

steps_eig, _ = plan_with_absorbed_eigenbasis(U_os, C, R, T_new, goal_state, start_state)
print(f"  Absorbed eigenbasis (d={d}):   {steps_eig} steps")

print("\n--- Averaged over 200 random trials ---")
evaluate_planning(D_new, T_new, barrier_states, "Ground truth")
evaluate_planning(D_os, T_new, barrier_states, "No repair (D_os)")

# Run 200 random trials for eigenbasis approach
valid_states = [s for s in range(NUM_STATES) if s not in barrier_states]
rng = np.random.RandomState(42)
n_trials = 200

steps_list = []
successes = 0
for _ in range(n_trials):
    s = rng.choice(valid_states)
    g = rng.choice(valid_states)
    while g == s:
        g = rng.choice(valid_states)
    st, _ = plan_with_absorbed_eigenbasis(U_os, C, R, T_new, g, s)
    steps_list.append(st)
    if st < 500:
        successes += 1

print(f"  Absorbed eigenbasis (d={d})    | avg: {np.mean(steps_list):7.1f} | "
      f"median: {np.median(steps_list):5.1f} | success: {successes/n_trials:.2f}")

print("\n--- Barrier-crossing paths only ---")
evaluate_planning_barrier_crossing(D_new, T_new, barrier_states, "Ground truth", ROWS, COLS)
evaluate_planning_barrier_crossing(D_os, T_new, barrier_states, "No repair (D_os)", ROWS, COLS)

# Barrier-crossing trials for eigenbasis
barrier_row = ROWS // 2
steps_bc = []
successes_bc = 0
n_bc = 0
for _ in range(1000):
    s = rng.choice(valid_states)
    g = rng.choice(valid_states)
    while g == s:
        g = rng.choice(valid_states)
    s_row, g_row = s // COLS, g // COLS
    if not ((s_row < barrier_row and g_row >= barrier_row) or 
            (s_row >= barrier_row and g_row < barrier_row)):
        continue
    n_bc += 1
    st, _ = plan_with_absorbed_eigenbasis(U_os, C, R, T_new, g, s)
    steps_bc.append(st)
    if st < 500:
        successes_bc += 1
    if n_bc >= 200:
        break

if n_bc > 0:
    print(f"  Absorbed eigenbasis (d={d})    | avg: {np.mean(steps_bc):7.1f} | "
          f"median: {np.median(steps_bc):5.1f} | success: {successes_bc/n_bc:.2f}")