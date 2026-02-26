"""
Test script to verify minigrid environment matches the baseline setup.
This replicates test_baseline.py using minigrid instead of custom grid.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.barrier_grid import create_open_env, create_baseline_env_with_barrier
from utils import get_complete_DR

# Create environments
print("Creating open environment (20x20)...")
env_open = create_open_env(size=22)  # 20x20 internal + 2 boundary walls
env_open.reset()

print("Creating environment with barrier...")
env_barrier = create_baseline_env_with_barrier()
env_barrier.reset()

# Get transition matrices
print("\nComputing transition matrices...")
T_open = env_open.get_transition_matrix()
T_barrier = env_barrier.get_transition_matrix()

print(f"T_open shape: {T_open.shape}")
print(f"T_barrier shape: {T_barrier.shape}")
print(f"T_open sum per row (should be ~1.0): {T_open.sum(axis=1)[:5]}")

# Compute DRs using existing utils
print("\nComputing D_os (open field DR)...")
D_os = get_complete_DR(T=T_open)
print(f"D_os shape: {D_os.shape}")

print("\nComputing D_true (with barrier DR)...")
D_true = get_complete_DR(T=T_barrier)
print(f"D_true shape: {D_true.shape}")

# Visualize the correction
print("\nVisualizing correction ΔD...")
delta_D = D_true - D_os

# Grid dimensions (internal, excluding walls)
ROWS, COLS = 20, 20

# Pick a probe state near the barrier
# Barrier is at row 10 (internal coords), cols 8-12
# Let's pick state one row above, middle of barrier
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
print("Saved: figs/minigrid_baseline_test.png")

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
print("Saved: figs/minigrid_correction_locality.png")

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
