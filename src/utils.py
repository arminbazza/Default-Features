import os

import numpy as np
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
    Re-normalizes rows so remaining transitions sum to 1.
    Returns new transition matrix.
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
    Returns T_new, barrier_states.
    """
    if barrier_type not in ['H', 'V', 'L']:
        raise ValueError(f"Barrier type '{barrier_type}' not in available list: ['horizontal', 'vertical', 'L']")

    if barrier_type == 'H':
        barrier_row, barrier_col_start = 10, 5
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


def plan_with_D(D, T, goal_state, start_state, max_steps=500):
    """
    Plan a route from start to goal using linear RL with DR matrix D.
    Returns (num_steps, path).
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
    values[goal_state] = 0.0

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