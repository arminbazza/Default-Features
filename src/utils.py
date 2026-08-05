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


# =============================================================================
# PLANNING: value functions, rollouts, and shared evaluation machinery
# =============================================================================
#
# Every planner below is now a thin wrapper over two pieces:
#
#     values_*(...)  ->  the value vector the planner follows
#     rollout(...)   ->  the start -> goal walk under those values
#
# Two reasons for the split.  (a) The value vector can be computed once per
# goal and reused across many rollouts; the per-trial recomputation that used
# to happen inside each planner is wasted work when the same goal is evaluated
# 20 or 200 times.  (b) There is now a single source of truth for each
# planner's value computation, so a cached evaluation loop in an experiment
# script cannot silently drift from `plan_with_*`.
#
# The expressions are byte-identical to the ones that previously lived inline
# in each planner (same operand order, same associativity), so every existing
# result reproduces exactly.  The only signature change is an optional
# `rng` argument, which defaults to the legacy global-`np.random` behaviour.


def neighbour_table(T):
    """
    Precompute successor lists once.  `np.where(T[s] > 0)` inside a rollout
    loop is pure overhead when the same transition matrix is reused.
    """
    return [np.where(T[s] > 0)[0] for s in range(T.shape[0])]


def values_M(F, M, T, goal, r_goal=10):
    """Values for the feature-space model  D ~= F @ M @ F^T."""
    N = F.shape[0]
    nt_idx = np.array([s for s in range(N) if s != goal])
    t_vec = T[np.ix_(nt_idx, [goal])].flatten()
    F_nt = F[nt_idx]
    exp_v = np.maximum(F_nt @ M @ F_nt.T @ t_vec, 1e-10)
    values = np.full(N, -np.inf)
    values[nt_idx] = np.log(exp_v)
    values[goal] = r_goal
    return values


def values_D(D, T, goal, r_goal=10):
    """Values for a full DR matrix D (linear RL)."""
    N = D.shape[0]
    nt_idx = np.array([s for s in range(N) if s != goal])
    D_NN = D[np.ix_(nt_idx, nt_idx)]
    t = T[np.ix_(nt_idx, [goal])].flatten()
    exp_v = np.maximum(D_NN @ t, 1e-10)
    values = np.full(N, -np.inf)
    values[nt_idx] = np.log(exp_v)
    values[goal] = r_goal
    return values


def values_SR(SR, goal, r_goal=10):
    """Values for a full state-space SR matrix:  V = SR @ r."""
    r = np.zeros(SR.shape[0])
    r[goal] = r_goal
    return SR @ r


def values_SF(Psi, F, lam, goal, r_goal=10):
    """
    Values for successor features Psi = M @ F.

    F = V * sqrt(lam) gives F^T F = diag(lam), so the pseudoinverse is
    F^{-1} = diag(1/lam) @ F^T and therefore  V = Psi @ (F^T r / lam).
    """
    r = np.zeros(Psi.shape[0])
    r[goal] = r_goal
    return Psi @ ((F.T @ r) / lam)


def rollout(values, T, goal, start, max_steps=500, temp=None, rng=None, nbrs=None):
    """
    Walk from `start` under `values` until `goal` is reached or `max_steps`
    transitions have been taken.

    rng   : None -> the global `np.random` stream (legacy behaviour).  Pass an
            explicit RandomState to make evaluation noise reproducible and
            independent of execution order -- two planners evaluated with the
            same rng seed then see identical noise, which pairs the comparison.
    nbrs  : optional precomputed `neighbour_table(T)`.

    Returns (steps, path); `steps` is `max_steps` if the goal was not reached.
    """
    state = start
    path = [state]
    for _ in range(max_steps):
        if state == goal:
            break
        neighbors = nbrs[state] if nbrs is not None else np.where(T[state] > 0)[0]
        if len(neighbors) == 0:
            break
        state = _pick_next(neighbors, values, temp, rng)
        path.append(state)
    steps = len(path) - 1 if state == goal else max_steps
    return steps, path


def rollout_rng(base, run, trial, chunk=0,
                run_stride=100_000_000, chunk_stride=100_003):
    return np.random.RandomState(
        base + run * run_stride + chunk * chunk_stride + trial)


def sem(x, axis=0):
    """Standard error of the mean along `axis`; zeros if n < 2."""
    x = np.asarray(x)
    n = x.shape[axis]
    if n < 2:
        return np.zeros(np.delete(np.array(x.shape), axis))
    return x.std(axis=axis, ddof=1) / np.sqrt(n)


def plan_with_M(F, M, T, goal, start, r_goal=10, max_steps=500, temp=None,
                rng=None, nbrs=None):
    """
    Plan in feature space using the approximation D ≈ F @ M @ F^T.
    """
    values = values_M(F, M, T, goal, r_goal)
    return rollout(values, T, goal, start, max_steps, temp, rng, nbrs)



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


def plan_with_D(D, T, goal_state, start_state, r_goal=10, max_steps=500, temp=None,
                rng=None, nbrs=None):
    """
    Plan a route from start to goal using linear RL with DR matrix D.
    """
    values = values_D(D, T, goal_state, r_goal)
    return rollout(values, T, goal_state, start_state, max_steps, temp, rng, nbrs)


def plan_with_SR(SR, T, goal_state, start_state, r_goal=10, max_steps=500, temp=None,
                 rng=None, nbrs=None):
    """
    Plan using a full state-space SR matrix D (output of sr_td).
    D[s, s'] = expected discounted occupancy at s' starting from s.
    """
    values = values_SR(SR, goal_state, r_goal)
    return rollout(values, T, goal_state, start_state, max_steps, temp, rng, nbrs)


def plan_with_dyna_q(V, T_true, goal_state, start_state, max_steps=500, temp=None,
                     rng=None, nbrs=None):
    """
    Plan using goal-conditioned values V[s, goal] (output of dyna_q).
    """
    return rollout(V[:, goal_state], T_true, goal_state, start_state,
                   max_steps, temp, rng, nbrs)


def plan_with_V(V, T_true, goal_state, start_state, max_steps=500, temp=None,
                rng=None, nbrs=None):
    """
    Plan using a 1D value function V[s] learned for a fixed goal (output of dyna_q).
    Same call signature as plan_with_M / plan_with_D.
    """
    return rollout(V, T_true, goal_state, start_state, max_steps, temp, rng, nbrs)


def plan_with_SF(Psi, F, lam, T, goal_state, start_state, r_goal=10, max_steps=500,
                 temp=None, rng=None, nbrs=None):
    """
    Plan using successor features (output of sr_sf).
    """
    values = values_SF(Psi, F, lam, goal_state, r_goal)
    return rollout(values, T, goal_state, start_state, max_steps, temp, rng, nbrs)


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


# =============================================================================
# MULTI-SEED EXPERIMENT HARNESS
# =============================================================================
#
# Scaffolding for the multi-seed re-runs of the plan_* experiments, built on the
# planning primitives above: frozen evaluation sets, per-run training seeds, the
# three evaluation loops, resumable storage, and the report.
#
# The three conventions every multi-seed script follows
# -----------------------------------------------------
# 1. The *training seed is the replicate unit*: N_RUNS independent seeds spawned
#    from SeedSequence(SEED); results reported as mean over trials within a run
#    -> one number per run -> mean +/- SEM across the run-means.  The older
#    across-trials SEM measures dispersion within a single training run and
#    cannot speak to whether a result survives retraining.
# 2. The *evaluation set is frozen*: drawn once from a dedicated PAIR_SEED and
#    reused verbatim by every run, model and checkpoint.
# 3. *Rollout noise is keyed by (run, chunk, trial)*: shared across models at a
#    given checkpoint, so model contrasts are paired; resampled at every
#    checkpoint and every run, so curve jitter is honest evaluation noise and
#    the across-run SEM absorbs rollout variance as well as training variance.


def print_header(title, width=68):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


# =============================================================================
# EVALUATION SET
# =============================================================================

def draw_pairs(start_states, goal_states, n_trials, seed):
    """
    Draw (start, goal) pairs ONCE, with replacement, from a dedicated stream.

    Returns an (n_trials, 2) int array, identical for every run, model and
    checkpoint.  Draws start then goal in the same order as the original
    `_eval` helpers, so passing the script's original SEED reproduces the
    original pair sequence exactly.
    """
    starts = np.asarray(start_states)
    goals = np.asarray(goal_states)
    rng = np.random.RandomState(seed)
    pairs = np.empty((n_trials, 2), dtype=np.int64)
    for i in range(n_trials):
        pairs[i, 0] = rng.choice(starts)
        pairs[i, 1] = rng.choice(goals)
    return pairs


def draw_starts(start_states, n_trials, seed):
    """
    Draw start states ONCE for the fixed-goal (_mc) experiments, matching the
    original `start = eval_rng.choice(start_states)` sequence.
    """
    starts = np.asarray(start_states)
    rng = np.random.RandomState(seed)
    return np.array([rng.choice(starts) for _ in range(n_trials)], dtype=np.int64)


def spawn_train_seeds(master_seed, n_runs, n_checkpoints):
    """
    Independent per-run training seeds.  `SeedSequence.spawn` avoids the
    collisions that arithmetic offsets (SEED + chunk * 1009) can produce once a
    run index is added on top.  All values are < 2**32, as RandomState requires.
    """
    run_ss = np.random.SeedSequence(master_seed).spawn(n_runs)
    return np.stack([run_ss[r].generate_state(n_checkpoints).astype(np.int64)
                     for r in range(n_runs)])


# =============================================================================
# EVALUATION LOOPS
# =============================================================================

def evaluate_pairs(value_fn, pairs, run, chunk, nbrs, max_steps, temp,
                   rollout_base):
    """
    Sweep over (start, goal) pairs.  `value_fn(goal) -> values` is memoised over
    the distinct goals, so the value vector is built once per goal rather than
    once per trial.  `chunk` is the checkpoint index; rollout noise is resampled
    for each one.
    """
    cache = {}
    steps = np.empty(len(pairs), dtype=np.int32)
    for i, (start, goal) in enumerate(pairs):
        start, goal = int(start), int(goal)
        if goal not in cache:
            cache[goal] = value_fn(goal)
        steps[i], _ = rollout(cache[goal], None, goal, start, max_steps, temp,
                              rollout_rng(rollout_base, run, i, chunk), nbrs)
    return steps


def evaluate_starts(values, starts, goal, run, chunk, nbrs, max_steps, temp,
                    rollout_base):
    """Sweep over start states with a single fixed goal (the _mc experiments)."""
    steps = np.empty(len(starts), dtype=np.int32)
    for i, start in enumerate(starts):
        steps[i], _ = rollout(values, None, goal, int(start), max_steps, temp,
                              rollout_rng(rollout_base, run, i, chunk), nbrs)
    return steps


def evaluate_fixed(values, goal, start, run, chunk, n_trials, nbrs, max_steps,
                   temp, rollout_base):
    """Repeat rollouts from a single fixed (start, goal) pair (the four-room case)."""
    steps = np.empty(n_trials, dtype=np.int32)
    for i in range(n_trials):
        steps[i], _ = rollout(values, None, goal, start, max_steps, temp,
                              rollout_rng(rollout_base, run, i, chunk), nbrs)
    return steps


# =============================================================================
# RESUMABLE STORAGE
# =============================================================================

def init_records(keys, shape):
    return {k: np.zeros(shape, dtype=np.int32) for k in keys}


def load_resume(save_path, records, n_runs, fresh=False):
    """
    Resume at run granularity.  Per-run seeds are independent, so a stored file
    with fewer or more runs is still valid for the runs it contains.

    Returns the number of completed runs already present.
    """
    if fresh or not os.path.exists(save_path):
        return 0
    z = np.load(save_path, allow_pickle=False)
    probe = next(iter(records))
    ok = (probe in z.files
          and all(k in z.files for k in records)
          and z[probe].shape[1:] == records[probe].shape[1:])
    if not (ok and int(z["runs_done"]) >= 1):
        print("\n  existing file is incompatible - starting fresh")
        return 0
    take = min(int(z["runs_done"]), n_runs)
    for k in records:
        records[k][:take] = z[k][:take]
    print(f"\n  resuming: {take}/{n_runs} runs already complete")
    return take


def save_records(save_path, records, runs_done, **meta):
    np.savez(save_path, runs_done=runs_done, **meta, **records)


# =============================================================================
# REPORT
# =============================================================================

def _collapse(arr, runs_done):
    """
    Reduce a record array to (runs, checkpoints) by averaging every axis except
    run (0) and checkpoint (-2).  Handles both the plain
    (runs, checkpoints, trials) layout and the four-room-goal layout
    (runs, pairs, checkpoints, trials).
    """
    a = arr[:runs_done]
    keep = (0, a.ndim - 2)
    axes = tuple(i for i in range(a.ndim) if i not in keep)
    return a.mean(axis=axes)


def _within_run_sem(arr, last):
    """Spread across the replicate axis inside run 0 -- what the paper reports."""
    a = arr[0]
    sub = np.take(a, last, axis=a.ndim - 2).ravel()
    return sub.std(ddof=1) / np.sqrt(sub.size) if sub.size > 1 else 0.0


def report_multiseed(save_path, groups, contrasts=(), within_label="trials",
           curve_points=7):
    """
    Print the multi-seed summary.

    groups : list of (label, {display_name: record_key})
             one block per scenario / K value / same-vs-diff split.
    contrasts : list of (display_name_a, display_name_b) for paired t-tests.
    """
    z = np.load(save_path, allow_pickle=False)
    steps_axis = z["checkpoint_steps"]
    runs_done = int(z["runs_done"])
    max_steps = int(z["eval_max_steps"])
    last = -1

    print_header(f"REPORT  --  {runs_done} runs, {len(steps_axis)} checkpoints")

    for label, mapping in groups:
        present = {n: k for n, k in mapping.items() if k in z.files}
        if not present:
            continue
        run_means = {n: _collapse(z[k], runs_done) for n, k in present.items()}
        run_succ = {n: _collapse((z[k] < max_steps).astype(np.float64), runs_done)
                    for n, k in present.items()}

        print(f"\n{label}  --  final checkpoint ({steps_axis[last]:,} steps)")
        print(f"  {'model':<24s}  {'mean':>8s}  {'SEM(runs)':>10s}  "
              f"{'SEM(' + within_label + ')':>14s}  {'ratio':>6s}  {'success':>8s}")
        print(f"  {'-'*24}  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*6}  {'-'*8}")
        for n, k in present.items():
            rm = run_means[n][:, last]
            s_runs = sem(rm)
            s_within = _within_run_sem(z[k], last)
            ratio = s_runs / s_within if s_within > 0 else np.nan
            print(f"  {n:<24s}  {rm.mean():8.1f}  {s_runs:10.2f}  "
                  f"{s_within:14.2f}  {ratio:6.2f}  "
                  f"{run_succ[n][:, last].mean():8.2f}")

        print("\n  Per-run means: " + "  ".join(f"r{r}".rjust(7)
                                                for r in range(runs_done)))
        for n in present:
            print(f"  {n:<24s}  "
                  + "  ".join(f"{v:7.1f}" for v in run_means[n][:, last]))

        usable = [(a, b) for a, b in contrasts if a in present and b in present]
        if usable:
            print("\n  Paired contrasts at final checkpoint (per-run differences)")
            print(f"  {'contrast':<40s}  {'mean d':>9s}  {'SEM':>8s}  "
                  f"{'t':>7s}  {'df':>3s}")
            for a, b in usable:
                d = run_means[a][:, last] - run_means[b][:, last]
                s_d = sem(d)
                t = d.mean() / s_d if s_d > 0 else np.nan
                print(f"  {a + ' - ' + b:<40s}  {d.mean():9.2f}  {s_d:8.2f}  "
                      f"{t:7.2f}  {runs_done - 1:3d}")

        print("\n  Learning curve (mean over runs +/- SEM across runs)")
        idx = np.unique(np.linspace(0, len(steps_axis) - 1, curve_points).astype(int))
        print(f"  {'model':<24s}  "
              + "  ".join(f"{steps_axis[i]//1000:>13d}k" for i in idx))
        for n in present:
            cells = [f"{run_means[n][:, i].mean():6.1f}+/-{sem(run_means[n][:, i]):5.2f}"
                     for i in idx]
            print(f"  {n:<24s}  " + "  ".join(cells))