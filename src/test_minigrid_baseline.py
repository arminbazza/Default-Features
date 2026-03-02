# Supress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from envs.barrier_grid import create_open_env, create_baseline_env_with_barrier
from networks.repair import RepairNetworkLinear
from utils import get_complete_DR

# Make environments
env_open = create_open_env(size=22)  # 20x20 internal + 2 boundary walls
env_open.reset()
env_barrier = create_baseline_env_with_barrier()
env_barrier.reset()
T_open = env_open.get_transition_matrix()
T_barrier = env_barrier.get_transition_matrix()
D_os = get_complete_DR(T=T_open)
D_true = get_complete_DR(T=T_barrier)

# Fixed variables
ROWS, COLS = 20, 20
NUM_STATES = ROWS * COLS
COST = 0.1

# Hyperparameters
HIDDEN_DIM = 400
LR = 1e-3
NUM_STEPS = 5000
BATCH_SIZE = 256
GAMMA = np.exp(-COST)


def test_and_visualize():
    # Create environments
    print("Creating open environment (20x20)...")
    img_open = env_open.render()
    plt.imsave("figs/minigrid_env_open.png", img_open)

    print("Creating environment with barrier...")
    img_barrier = env_barrier.render()
    plt.imsave("figs/minigrid_env_1barrier.png", img_barrier)

    # Get transition matrices
    print("\nTesting transition matrices...")
    print(f"T_open shape: {T_open.shape}")
    print(f"T_barrier shape: {T_barrier.shape}")

    # Compute DRs using existing utils
    print("\nComputing D_os (open field DR)...")
    print(f"D_os shape: {D_os.shape}")

    print("\nComputing D_true (with barrier DR)...")
    print(f"D_true shape: {D_true.shape}")

    # Visualize the correction
    print("\nVisualizing correction ΔD...")
    delta_D = D_true - D_os

    # Grid dimensions (internal, excluding walls)
    ROWS, COLS = 20, 20

    # Probe state in middle of barrier
    probe_state = 9 * COLS + 10  # row 9, col 10
    probe_row = delta_D[probe_state].reshape(ROWS, COLS)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # D_os row for probe state
    im0 = axes[0].imshow(D_os[probe_state].reshape(ROWS, COLS), cmap='viridis')
    axes[0].set_title(f'D_os row (state {probe_state})')
    plt.colorbar(im0, ax=axes[0])

    # D_true row for probe state
    im1 = axes[1].imshow(D_true[probe_state].reshape(ROWS, COLS), cmap='viridis')
    axes[1].set_title(f'D_true row (state {probe_state})')
    plt.colorbar(im1, ax=axes[1])

    # ΔD row — this is what the repair network needs to learn
    im2 = axes[2].imshow(probe_row, cmap='RdBu_r',
                        vmin=-np.abs(probe_row).max(),
                        vmax=np.abs(probe_row).max())
    axes[2].set_title(f'ΔD row (state {probe_state})')
    plt.colorbar(im2, ax=axes[2])

    # Mark barrier (internal coords: row 10, cols 8-12)
    for ax in axes:
        for col in range(8, 13):
            ax.plot(col, 10, 'ks', markersize=4)

    plt.tight_layout()
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/minigrid_baseline_test.png', dpi=150)

    # Check correction locality
    delta_norms = np.linalg.norm(delta_D, axis=1).reshape(ROWS, COLS)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(delta_norms, cmap='hot')
    ax2.set_title('||ΔD(s,:)|| across all states\n(where does the correction matter?)')
    for col in range(8, 13):
        ax2.plot(col, 10, 'cs', markersize=4)
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig('figs/minigrid_correction_locality.png', dpi=150)

    # Compare with original baseline
    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60)
    print(f"Number of states: {T_open.shape[0]}")
    print(f"D_os range: [{D_os.min():.3f}, {D_os.max():.3f}]")
    print(f"D_true range: [{D_true.min():.3f}, {D_true.max():.3f}]")
    print(f"||ΔD|| (Frobenius): {np.linalg.norm(delta_D):.3f}")
    print(f"Relative change: {np.linalg.norm(delta_D) / np.linalg.norm(D_os):.3f}")
    print("\nEnvironment is ready for repair network training!")

def train_single_barrier():
    for step in range(NUM_STEPS):
        action = env_open.action_space.sample()
        print(env_open.step(action))
        break
    

if __name__ == '__main__':
    test_and_visualize()
    # train_single_barrier()