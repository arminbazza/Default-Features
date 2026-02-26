import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Add parent dir to path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import create_transition_open, get_complete_DR, woodbury_complete
from networks.repair import RepairNetworkLinear


# ============================================================
# Environment setup
# ============================================================

ROWS, COLS = 20, 20
NUM_STATES = ROWS * COLS
COST = 0.1


def state_to_coords(state, rows=ROWS, cols=COLS):
    """Convert state index to normalized (x, y) in [0, 1]."""
    r = state // cols
    c = state % cols
    return np.array([r / (rows - 1), c / (cols - 1)], dtype=np.float32)


def all_coords(rows=ROWS, cols=COLS):
    """Get normalized coords for all states. Returns (num_states, 2)."""
    coords = np.zeros((rows * cols, 2), dtype=np.float32)
    for s in range(rows * cols):
        coords[s] = state_to_coords(s, rows, cols)
    return coords


def place_barrier(T, barrier_states, rows=ROWS, cols=COLS):
    """
    Place a barrier by blocking transitions into barrier states.
    Returns new transition matrix.
    """
    T_new = T.copy()
    for bs in barrier_states:
        # Zero out transitions INTO the barrier state
        T_new[:, bs] = 0
        # Zero out transitions FROM the barrier state
        T_new[bs, :] = 0

    # Re-normalize rows (states adjacent to barrier now have fewer neighbors)
    for s in range(T_new.shape[0]):
        row_sum = T_new[s].sum()
        if row_sum > 0:
            T_new[s] /= row_sum

    return T_new


def sample_transition(T, state):
    """Sample next state from transition matrix."""
    probs = T[state]
    if probs.sum() == 0:
        return state  # stuck (barrier state)
    return np.random.choice(len(probs), p=probs)


# ============================================================
# Step 1-3: Compute ground truth and visualize
# ============================================================

print("Computing D_os for open 20x20 grid...")
T_os = create_transition_open(shape=(ROWS, COLS))
D_os = get_complete_DR(T=T_os)
print(f"  D_os shape: {D_os.shape}")

print("\nPlacing barrier and computing D_true...")
# Place a small horizontal barrier in the middle of the grid
barrier_row, barrier_col_start = 10, 8
barrier_states = [barrier_row * COLS + c for c in range(barrier_col_start, barrier_col_start + 5)]
print(f"  Barrier states: {barrier_states} (row {barrier_row}, cols {barrier_col_start}-{barrier_col_start+4})")

T_new = place_barrier(T_os, barrier_states)
D_true = get_complete_DR(T=T_new)

# print("\nVisualizing the correction ΔD...")
delta_D = D_true - D_os

# Pick a probe state near the barrier to visualize its row
probe_state = (barrier_row - 1) * COLS + (barrier_col_start + 2)  # one row above middle of barrier
probe_row = delta_D[probe_state].reshape(ROWS, COLS)

# fig, axes = plt.subplots(1, 3, figsize=(15, 4))

## D_os row for probe state
# im0 = axes[0].imshow(D_os[probe_state].reshape(ROWS, COLS), cmap='viridis')
# axes[0].set_title(f'D_os row (state {probe_state})')
# plt.colorbar(im0, ax=axes[0])

## D_true row for probe state
# im1 = axes[1].imshow(D_true[probe_state].reshape(ROWS, COLS), cmap='viridis')
# axes[1].set_title(f'D_true row (state {probe_state})')
# plt.colorbar(im1, ax=axes[1])

##  ΔD row — this is what the repair network needs to learn
# im2 = axes[2].imshow(probe_row, cmap='RdBu_r', vmin=-np.abs(probe_row).max(), vmax=np.abs(probe_row).max())
# axes[2].set_title(f'ΔD row (state {probe_state})')
# plt.colorbar(im2, ax=axes[2])

## Mark barrier
# for ax in axes:
#     for bs in barrier_states:
#         br, bc = bs // COLS, bs % COLS
#         ax.plot(bc, br, 'ks', markersize=4)

# plt.tight_layout()
# plt.savefig('figs/step3_delta_visualization.png', dpi=150)
# print("  Saved: step3_delta_visualization.png")

## Also check: how "local" is the correction?
# delta_norms = np.linalg.norm(delta_D, axis=1).reshape(ROWS, COLS)
# fig2, ax2 = plt.subplots(figsize=(5, 4))
# im = ax2.imshow(delta_norms, cmap='hot')
# ax2.set_title('||ΔD(s,:)|| across all states\n(where does the correction matter?)')
# for bs in barrier_states:
#     br, bc = bs // COLS, bs % COLS
#     ax2.plot(bc, br, 'cs', markersize=4)
# plt.colorbar(im, ax=ax2)
# plt.tight_layout()
# plt.savefig('figs/step3_correction_locality.png', dpi=150)
# print("  Saved: step3_correction_locality.png")


# ============================================================
# Step 4: Train repair network via TD
# ============================================================

print("\nTraining repair network...")

device = torch.device('mps')

# Hyperparameters
HIDDEN_DIM = 400
LR = 1e-3
NUM_STEPS = 5000
BATCH_SIZE = 256
GAMMA = np.exp(-COST)  # discount = exp(-c) since cost is uniform

# Precompute
coords_all = torch.tensor(all_coords(), device=device)              # (400, 2)
D_os_tensor = torch.tensor(D_os, dtype=torch.float32, device=device) # (400, 400)

# Model
model = RepairNetworkLinear(num_states=NUM_STATES, hidden_dim=HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# One-hot vectors for the "feature" in DR TD update
eye = torch.eye(NUM_STATES, device=device)

losses = []

for step in range(NUM_STEPS):
    # Sample batch of random states (not barrier states)
    valid_states = [s for s in range(NUM_STATES) if s not in barrier_states]
    batch_states = np.random.choice(valid_states, size=BATCH_SIZE)

    # Sample next states from T_new for each
    batch_next = np.array([sample_transition(T_new, s) for s in batch_states])

    # Frozen baseline rows
    D_os_s = D_os_tensor[batch_states]       # (B, S)
    D_os_sp = D_os_tensor[batch_next]        # (B, S)

    # Repair network predictions
    delta_s = model(D_os_s)                  # (B, S)
    with torch.no_grad():
        delta_sp = model(D_os_sp)            # (B, S)

    # Full predictions
    D_s = D_os_s + delta_s                 # (B, S)
    D_sp_target = D_os_sp + delta_sp       # (B, S)

    # TD target:
    onehot_s = eye[batch_states]               # (B, S)
    target = GAMMA * onehot_s + GAMMA * D_sp_target

    # Loss
    loss = nn.functional.mse_loss(D_s, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (step + 1) % 1000 == 0:
        print(f"  Step {step+1}/{NUM_STEPS}, Loss: {loss.item():.6f}")


# ============================================================
# Step 5: Compare against ground truth
# ============================================================

print("\nStep 5: Evaluating...")

model.eval()
with torch.no_grad():
    delta_learned = model(D_os_tensor).cpu().numpy()  # (400, 400)

D_learned = D_os + delta_learned

# Reconstruction error
recon_error = np.linalg.norm(D_learned - D_true) / np.linalg.norm(D_true)
print(f"  Relative reconstruction error: {recon_error:.4f}")

baseline_error = np.linalg.norm(D_os - D_true) / np.linalg.norm(D_true)
print(f"  Baseline error (no repair): {baseline_error:.4f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(D_true[probe_state].reshape(ROWS, COLS), cmap='viridis')
axes[0].set_title('Ground truth D_true')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(D_learned[probe_state].reshape(ROWS, COLS), cmap='viridis')
axes[1].set_title('Learned D_os + Δψ')
plt.colorbar(im1, ax=axes[1])

error_map = (D_learned[probe_state] - D_true[probe_state]).reshape(ROWS, COLS)
im2 = axes[2].imshow(error_map, cmap='RdBu_r', vmin=-np.abs(error_map).max(), vmax=np.abs(error_map).max())
axes[2].set_title('Error (learned - true)')
plt.colorbar(im2, ax=axes[2])

for ax in axes:
    for bs in barrier_states:
        br, bc = bs // COLS, bs % COLS
        ax.plot(bc, br, 'ks', markersize=4)

plt.tight_layout()
plt.savefig('figs/step5_comparison.png', dpi=150)
print("  Saved: step5_comparison.png")

# Loss curve
fig3, ax3 = plt.subplots(figsize=(8, 3))
ax3.plot(losses[::100])  # subsample for readability
ax3.set_xlabel('Step (x100)')
ax3.set_ylabel('TD Loss')
ax3.set_title('Training Loss')
plt.tight_layout()
plt.savefig('figs/step5_loss_curve.png', dpi=150)
print("  Saved: step5_loss_curve.png")

print("\nDone!")