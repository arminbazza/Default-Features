import os

import numpy as np
from scipy.stats import spearmanr
import torch


def woodbury_complete(T_old, D, T_new):
    """
    Performs woodbury update
    """
    # Use difference between new and old transition matrices to find location of change
    diffs = T_old != T_new
    diff_rows, _ = np.where(diffs)
    delta_locs = np.unique(diff_rows)

    # Get C and R which identify the rows and columns which changed
    N = T_old.shape[0]
    R = (T_new - T_old)[delta_locs, :]
    C = np.eye(N)[:,delta_locs]

    # Calculate the POR for the new terminal state
    Z = R @ D @ C
    A = np.linalg.inv(np.eye(Z.shape[0]) - Z)

    woodbury_update = D@C@A@R@D
    eig = R@D**2@C@A
    D_new = D + woodbury_update

    return D_new


def get_complete_DR(T):
    """
    Uses matrix inversion to calculate the complete DR
    """
    # Assume a reward of -0.1 across all states and lambda=1
    r = np.full(np.shape(T)[0], -0.1)
    terminals = np.diag(T) == 1
    r[terminals] = 10
    lambd = 1
    L = np.diag(np.exp(-r / lambd)) - T
    D = np.linalg.inv(L)
    
    return D


def create_transition_open(shape):
    """
    Creates an open field transition matrix for the environment with the specified shape
    """
    rows, cols = shape
    T = np.zeros((rows*cols, rows*cols))
    
    for r in range(rows):
        for c in range(cols):
            state = r * cols + c
            neighbors = []
            if r > 0:
                neighbors.append((r - 1) * cols + c)  # up
            if r < rows - 1:
                neighbors.append((r + 1) * cols + c)  # down
            if c > 0:
                neighbors.append(r * cols + (c - 1))  # left
            if c < cols - 1:
                neighbors.append(r * cols + (c + 1))  # right
            for neighbor in neighbors:
                T[state, neighbor] = 1 / len(neighbors)

    return T


def sample_transition(T, state):
    """Sample next state from transition matrix."""
    probs = T[state]
    if probs.sum() == 0:
        return state  # stuck (barrier state)
    return np.random.choice(len(probs), p=probs)


def save_model_weights(model, path):
    torch.save(model.state_dict(), path)


def load_model_weights(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"  Loaded model from {path}")
    return model


def state_to_coords(state, rows, cols):
    """Convert state index to normalized (x, y) in [0, 1]."""
    r = state // cols
    c = state % cols
    return np.array([r / (rows - 1), c / (cols - 1)], dtype=np.float32)


def all_coords(rows, cols):
    """Get normalized coords for all states. Returns (num_states, 2)."""
    coords = np.zeros((rows * cols, 2), dtype=np.float32)
    for s in range(rows * cols):
        coords[s] = state_to_coords(s, rows, cols)
    return coords


def place_barrier(T, barrier_states):
    """
    Place a barrier by blocking transitions into and out of barrier states.
    """
    T_new = T.copy()
    for bs in barrier_states:
        T_new[:, bs] = 0
        T_new[bs, :] = 0

    for s in range(T_new.shape[0]):
        row_sum = T_new[s].sum()
        if row_sum > 0:
            T_new[s] /= row_sum

    return T_new


def make_maze(T_os, barrier_type, rows=20, cols=20):
    """
    Build a maze by placing a barrier of the specified type.
    """
    if barrier_type not in ['H', 'V', 'L']:
        raise ValueError(f"Barrier type '{barrier_type}' not in available list: ['horizontal', 'vertical', 'L']")

    if barrier_type == 'H':
        barrier_row, barrier_col_start = rows//2, cols//4
        barrier_states = sorted([barrier_row * cols + c for c in range(barrier_col_start, barrier_col_start + 10)])
        filename = os.getcwd() + '/envs/baselines/HBarrier.npy'

    elif barrier_type == 'L':
        corner_row, corner_col = 11, 7
        arm_length = 8
        vertical_states = [r * cols + corner_col for r in range(corner_row - arm_length + 1, corner_row + 1)]
        horizontal_states = [corner_row * cols + c for c in range(corner_col, corner_col + arm_length)]
        barrier_states = sorted(set(vertical_states + horizontal_states))
        filename = os.getcwd() + '/envs/baselines/LBarrier.npy'

    T_new = place_barrier(T_os, barrier_states)

    if not os.path.exists(filename):
        np.save(filename, T_new)

    return T_new, barrier_states


def h_wall_edges(rows, cols, barrier_row, col_start, col_end):
    """
    Generate horizontal wall edges that block up/down transitions
    """
    return [
        ((barrier_row - 1) * cols + c, barrier_row * cols + c)
        for c in range(col_start, col_end)
    ]


def v_wall_edges(rows, cols, barrier_col, row_start, row_end):
    """
    Generate vertical wall edges that block left/right transitions
    """
    return [
        (r * cols + (barrier_col - 1), r * cols + barrier_col)
        for r in range(row_start, row_end)
    ]


def place_walls(T, wall_edges):
    """
    Block specific directed edges between pairs of states.
    """
    T_new = T.copy()
    for s1, s2 in wall_edges:
        T_new[s1, s2] = 0.0
        T_new[s2, s1] = 0.0

    for s in range(T_new.shape[0]):
        row_sum = T_new[s].sum()
        if row_sum > 0:
            T_new[s] /= row_sum

    return T_new


def make_wall_maze(T_os, barrier_type, rows=10, cols=10):
    """
    Build a maze by blocking transitions between states (wall edges) rather
    than making entire states inaccessible.
    """
    if barrier_type == 'H':
        barrier_row = rows // 2          # wall sits between row-1 and row
        col_start   = cols // 4          # e.g. col 2 for 10x10, col 5 for 20x20
        col_end     = 3 * cols // 4      # e.g. col 7 for 10x10, col 15 for 20x20

        wall_edges = [
            ((barrier_row - 1) * cols + c, barrier_row * cols + c)
            for c in range(col_start, col_end)
        ]
    else:
        raise ValueError(f"Barrier type '{barrier_type}' not supported by make_wall_maze")

    T_new = place_walls(T_os, wall_edges)
    return T_new, wall_edges


def make_four_room(rows=11, cols=11):
    """
    Standard four-room grid environment.

    Layout (each room is 5x5):
      TL: rows 0-4, cols 0-4
      TR: rows 0-4, cols 6-10
      BL: rows 6-10, cols 0-4
      BR: rows 6-10, cols 6-10

    Walls (inaccessible cells):
      Horizontal wall : row 5, all cols except doorways
      Vertical wall   : col 5, all rows except doorways

    Doorways:
      TL <-> TR : (row 2, col 5)   — middle of vertical wall
      TL <-> BL : (row 5, col 3)   — second from right within TL side
      BL <-> BR : (row 9, col 5)   — second from bottom within BL side
      TR <-> BR : (row 5, col 8)   — middle of TR side

    Returns T, wall_states.
    """
    T = create_transition_open(shape=(rows, cols))

    # Horizontal wall at row 5 — gaps at col 3 (TL<->BL) and col 8 (TR<->BR)
    h_wall = [5 * cols + c for c in range(cols) if c not in (3, 8)]
    # Vertical wall at col 5 — gaps at row 2 (TL<->TR) and row 9 (BL<->BR)
    # Exclude row 5 since it is already covered by h_wall
    v_wall = [r * cols + 5 for r in range(rows) if r not in (2, 5, 9)]

    wall_states = sorted(set(h_wall + v_wall))
    T = place_barrier(T, wall_states)
    return T, wall_states


def make_four_room_walls(rows=11, cols=11):
    """
    Four-room grid using wall EDGES instead of barrier states.

    All 121 states remain accessible.  Rooms are separated by blocking
    transitions across the wall lines:
      Horizontal wall  : between row 4 and row 5, gaps at col 3 and col 8
      Vertical wall    : between col 4 and col 5, gaps at row 2 and row 9

    Doorways (each is one edge, not a state):
      TL <-> TR : (2*cols+4, 2*cols+5)  = (26, 27)
      TL <-> BL : (4*cols+3, 5*cols+3)  = (47, 58)
      TR <-> BR : (4*cols+8, 5*cols+8)  = (52, 63)
      BL <-> BR : (9*cols+4, 9*cols+5)  = (103, 104)

    Returns T, wall_edges.
    """
    T = create_transition_open(shape=(rows, cols))

    wall_edges = []

    # Horizontal wall at row 5 — gaps at col 3 and col 8
    for c in range(cols):
        if c in (3, 8):
            continue
        s_above = 4 * cols + c
        s_below = 5 * cols + c
        if s_above < rows * cols and s_below < rows * cols:
            wall_edges.append((s_above, s_below))

    # Vertical wall at col 5 — gaps at row 2 and row 9
    for r in range(rows):
        if r in (2, 9):
            continue
        s_left  = r * cols + 4
        s_right = r * cols + 5
        wall_edges.append((s_left, s_right))

    T = place_walls(T, wall_edges)
    return T, wall_edges


def make_four_room_reduced(rows=11, cols=11):
    """
    Build the four-room environment using only the valid (non-wall) states.

    Computes make_four_room, removes the 17 inaccessible wall states, and
    returns a compact (N_valid x N_valid) transition matrix together with
    index-mapping helpers.

    Returns
    -------
    T_reduced       : (N_valid, N_valid) transition matrix
    valid_states    : list[int]  —  full-space state indices (sorted)
    full_to_reduced : dict[int, int]  —  full index -> reduced index
    wall_states     : list[int]  —  full-space wall-state indices
    """
    T_full, wall_states = make_four_room(rows, cols)
    N_full = rows * cols

    wall_set     = set(wall_states)
    valid_states = [s for s in range(N_full) if s not in wall_set]
    full_to_reduced = {s: i for i, s in enumerate(valid_states)}

    vs = np.array(valid_states)
    T_reduced = T_full[np.ix_(vs, vs)]

    return T_reduced, valid_states, full_to_reduced, wall_states


def make_absorbing(T, state):
    """
    Return a copy of T with `state` made absorbing: all outgoing transitions
    are zeroed. Incoming transitions from other states are unchanged, so the
    state remains reachable.

    Note: differs from place_barrier, which zeros both row AND column.
    """
    T_new = T.copy()
    T_new[state, :] = 0.0
    return T_new


def softmax(values, temp):
    """
    Numerically stable softmax over an array of values at a given temperature.
    Returns a probability distribution of the same shape.
    """
    v = values / temp
    v = v - v.max()          # shift for numerical stability
    probs = np.exp(v)
    return probs / probs.sum()


def _pick_next(neighbors, values, temp, rng=None):
    """
    Select the next state from neighbors.
    temp=None  ->  greedy argmax
    temp>0     ->  softmax sampling at that temperature
    """
    if temp is None:
        return neighbors[np.argmax(values[neighbors])]
    probs = softmax(values[neighbors], temp)
    return neighbors[(rng or np.random).choice(len(neighbors), p=probs)]


def plan_with_M(F, M, T, goal, start, r_goal=10, max_steps=500, temp=None):
    """
    Plan in feature space using the approximation D ≈ F @ M @ F^T.
    """
    N   = F.shape[0]
    nt  = [s for s in range(N) if s != goal]
    nt_idx = np.array(nt)

    t_vec = T[np.ix_(nt_idx, [goal])].flatten()
    F_nt  = F[nt_idx]

    exp_v = np.maximum(F_nt @ M @ F_nt.T @ t_vec, 1e-10)
    v     = np.log(exp_v)

    values = np.full(N, -np.inf)
    for i, s in enumerate(nt):
        values[s] = v[i]
    values[goal] = r_goal

    state = start
    path  = [state]
    for _ in range(max_steps):
        if state == goal:
            break
        neighbors = np.where(T[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = _pick_next(neighbors, values, temp)
        path.append(state)

    steps = len(path) - 1 if state == goal else max_steps
    return steps, path


def evaluate_wall_crossing(plan_fn, barrier_row, rows, cols,
                            n_trials=200, max_steps=500, seed=42):
    """
    Evaluate a planner on start/goal pairs that must cross the horizontal wall.
    """
    N     = rows * cols
    above = [s for s in range(N) if s // cols < barrier_row]
    below = [s for s in range(N) if s // cols >= barrier_row]
    rng   = np.random.RandomState(seed)

    steps_list = []
    for _ in range(n_trials):
        start = rng.choice(above)
        goal  = rng.choice(below)
        st, _ = plan_fn(goal, start)
        steps_list.append(st)

    arr     = np.array(steps_list)
    mean    = np.mean(arr)
    median  = np.median(arr)
    success = np.mean(arr < max_steps)
    return mean, median, success


def plan_with_D(D, T, goal_state, start_state, r_goal=10, max_steps=500, temp=None):
    """
    Plan a route from start to goal using linear RL with DR matrix D.
    """
    N = D.shape[0]
    nonterminal = [s for s in range(N) if s != goal_state]

    D_NN = D[np.ix_(nonterminal, nonterminal)]
    t = T[np.ix_(nonterminal, [goal_state])].flatten()

    exp_v = D_NN @ t
    exp_v = np.maximum(exp_v, 1e-10)
    v = np.log(exp_v)

    values = np.full(N, -np.inf)
    for i, s in enumerate(nonterminal):
        values[s] = v[i]
    values[goal_state] = r_goal
    
    state = start_state
    path = [state]
    for step in range(max_steps):
        if state == goal_state:
            return step, path
        neighbors = np.where(T[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = _pick_next(neighbors, values, temp)
        path.append(state)

    return max_steps, path


def plan_with_SR(SR, T, goal_state, start_state, r_goal=10, max_steps=500, temp=None):
    """
    Plan using a full state-space SR matrix D (output of sr_td).
    D[s, s'] = expected discounted occupancy at s' starting from s.
    """
    N = SR.shape[0]

    r = np.zeros(N)
    r[goal_state] = r_goal

    values = SR @ r

    state = start_state
    path  = [state]
    for step in range(max_steps):
        if state == goal_state:
            return step, path
        neighbors = np.where(T[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = _pick_next(neighbors, values, temp)
        path.append(state)

    return max_steps, path


def plan_with_dyna_q(V, T_true, goal_state, start_state, max_steps=500, temp=None):
    """
    Plan using goal-conditioned values V[s, goal] (output of dyna_q).
    """
    values = V[:, goal_state]

    state = start_state
    path  = [state]
    for step in range(max_steps):
        if state == goal_state:
            return step, path
        neighbors = np.where(T_true[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = _pick_next(neighbors, values, temp)
        path.append(state)

    return max_steps, path


def plan_with_V(V, T_true, goal_state, start_state, max_steps=500, temp=None):
    """
    Plan using a 1D value function V[s] learned for a fixed goal (output of dyna_q).
    Same call signature as plan_with_M / plan_with_D.
    """
    state = start_state
    path  = [state]
    for step in range(max_steps):
        if state == goal_state:
            return step, path
        neighbors = np.where(T_true[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = _pick_next(neighbors, V, temp)
        path.append(state)
    return max_steps, path


def plan_with_SF(Psi, F, lam, T, goal_state, start_state, r_goal=10, max_steps=500, temp=None):
    """
    Plan using successor features (output of sr_sf).

    sr_sf learns  Psi = M @ F  (M is the standard SR, F is the feature basis),
    so  Psi[s, i]  is the discounted sum of feature i along trajectories from s.
    The correct value function is  V(s) = (M @ r)[s] = (Psi @ F^{-1} @ r)[s].

    Since F = V * sqrt(lam) gives  F^T F = diag(lam),
    the pseudoinverse is  F^{-1} = diag(1/lam) @ F^T,  so:
        w = F^T @ r / lam
        V = Psi @ w
    """
    N = Psi.shape[0]

    r = np.zeros(N)
    r[goal_state] = r_goal

    w = (F.T @ r) / lam
    values = Psi @ w

    state = start_state
    path  = [state]
    for step in range(max_steps):
        if state == goal_state:
            return step, path
        neighbors = np.where(T[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = _pick_next(neighbors, values, temp)
        path.append(state)

    return max_steps, path


def plan_random(T, goal_state, start_state, seed):
    """
    Plans by randomly selecting among valid neighbors at each step.
    """
    state = start_state
    path  = [state]
    rng   = np.random.default_rng(seed)

    while state != goal_state:
        neighbors = np.where(T[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = rng.choice(neighbors)
        path.append(state)

    return len(path) - 1, path


def plan_with_woodbury_features(U_os, Lambda, phi_C, psi_R, A, T, 
                                 goal_state, start_state, max_steps=500):
    """
    Plan using Woodbury components in feature space WITHOUT reconstructing D.
    """
    N = U_os.shape[0]
    nonterminal = [s for s in range(N) if s != goal_state]
    nt_idx = np.array(nonterminal)
    
    # t vector: transition probs from nonterminal states to goal
    t = T[np.ix_(nonterminal, [goal_state])].flatten()
    
    # Features at nonterminal states only
    U_nt = U_os[nt_idx, :]  # (N-1) x d
    
    # Base planning vector (d-dimensional)
    w_base = Lambda @ (U_nt.T @ t)
    
    # Barrier correction (all in small spaces: O-dim and d-dim)
    z_barrier = psi_R @ w_base              # O-dimensional
    correction = Lambda @ phi_C @ A @ z_barrier  # d-dimensional
    
    # Combined planning vector
    w_total = w_base + correction  # d-dimensional
    
    # Compute exp(v*) for all nonterminal states at once
    exp_v = U_nt @ w_total  # (N-1)-dimensional
    
    exp_v = np.maximum(exp_v, 1e-10)
    v = np.log(exp_v)
    
    # Map back to state indices
    values = np.full(N, -np.inf)
    for i, s in enumerate(nonterminal):
        values[s] = v[i]
    values[goal_state] = 10.0
    
    # Greedy planning
    state = start_state
    path = [state]
    for step in range(max_steps):
        if state == goal_state:
            return step, path
        neighbors = np.where(T[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = neighbors[np.argmax(values[neighbors])]
        path.append(state)
    return max_steps, path


def evaluate_planning(D, T, barrier_states, name, num_trials=200, 
                      max_steps=500, seed=42):
    """
    Evaluate planning over random start/goal pairs.
    Returns dict with avg, median, success rate.
    """
    valid = [s for s in range(D.shape[0]) if s not in barrier_states]
    np.random.seed(seed)
    all_steps = []

    for _ in range(num_trials):
        start, goal = np.random.choice(valid, size=2, replace=False)
        steps, _ = plan_with_D(D, T, goal, start, max_steps)
        all_steps.append(steps)

    results = {
        'avg': np.mean(all_steps),
        'median': np.median(all_steps),
        'success': np.mean([s < max_steps for s in all_steps]),
    }
    print(f"  {name:30s} | avg: {results['avg']:7.1f} | "
          f"median: {results['median']:5.1f} | success: {results['success']:.2f}")
    return results


def evaluate_planning_barrier_crossing(D, T, barrier_states, name, 
                                        rows, cols, num_trials=200, 
                                        max_steps=500, seed=42):
    """Only test start/goal pairs on opposite sides of the barrier."""
    barrier_rows = set(bs // cols for bs in barrier_states)
    barrier_row = min(barrier_rows)  # e.g., row 10
    
    # States above and below the barrier
    above = [s for s in range(rows * cols) 
             if s // cols < barrier_row and s not in barrier_states]
    below = [s for s in range(rows * cols) 
             if s // cols > barrier_row and s not in barrier_states]
    
    np.random.seed(seed)
    all_steps = []
    for _ in range(num_trials):
        start = np.random.choice(above)
        goal = np.random.choice(below)
        steps, _ = plan_with_D(D, T, goal, start, max_steps)
        all_steps.append(steps)
    
    avg = np.mean(all_steps)
    median = np.median(all_steps)
    success = np.mean([s < max_steps for s in all_steps])
    print(f"  {name:30s} | avg: {avg:7.1f} | median: {median:5.1f} | success: {success:.2f}")
    return all_steps


def visualize_path(path, barrier_states, title, filename, rows, cols):
    """Plot a planned path on the grid."""
    import matplotlib.pyplot as plt
    grid = np.zeros((rows, cols))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='gray_r', vmin=-1, vmax=1, alpha=0.3)

    path_r = [s // cols for s in path]
    path_c = [s % cols for s in path]
    ax.plot(path_c, path_r, 'b.-', markersize=8, linewidth=1.5)
    ax.plot(path_c[0], path_r[0], 'go', markersize=12, label='Start')
    ax.plot(path_c[-1], path_r[-1], 'r*', markersize=15, label='Goal')

    for bs in barrier_states:
        br, bc = bs // cols, bs % cols
        ax.plot(bc, br, 'ks', markersize=8)

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)

def sliding_correlation(map_before, map_after, rows, cols, box_radius=3):
    """
    Local Spearman correlation between two spatial maps.
    For each location, correlate values within a box_radius window.
    Returns (rows, cols) correlation map.
    """
    corr_map = np.full((rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            r_lo = max(0, r - box_radius)
            r_hi = min(rows, r + box_radius + 1)
            c_lo = max(0, c - box_radius)
            c_hi = min(cols, c + box_radius + 1)

            patch_b = map_before[r_lo:r_hi, c_lo:c_hi].ravel()
            patch_a = map_after[r_lo:r_hi, c_lo:c_hi].ravel()

            if len(patch_b) < 4:
                continue
            if np.std(patch_b) < 1e-10 or np.std(patch_a) < 1e-10:
                continue

            corr_map[r, c] = spearmanr(patch_b, patch_a).correlation

    return corr_map


def analyze_wernle(F_before, F_after, rows, cols, split_row,
                    eig_indices=None, box_radius=3):
    """
    Full Wernle analysis: sliding correlation for each eigenvector,
    average across eigenvectors, row-by-row profile.
    """
    if eig_indices is None:
        eig_indices = list(range(min(20, F_before.shape[1])))

    all_corr_maps = []
    for k in eig_indices:
        map_b = F_before[:, k].reshape(rows, cols)
        map_a = F_after[:, k].reshape(rows, cols)
        cm = sliding_correlation(map_b, map_a, rows, cols, box_radius)
        all_corr_maps.append(cm)

    # Average across eigenvectors
    corr_stack = np.array(all_corr_maps)
    mean_corr = np.nanmean(corr_stack, axis=0)

    # Row profile
    row_profile = np.array([np.nanmean(mean_corr[r, :]) for r in range(rows)])

    return mean_corr, row_profile, all_corr_maps