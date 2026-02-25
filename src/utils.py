import numpy as np

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
