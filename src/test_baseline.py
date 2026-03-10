import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (create_transition_open, get_complete_DR, save_model_weights,
                   sample_transition, make_maze)
from networks.repair import RepairNetworkLinear


ROWS, COLS = 20, 20
NUM_STATES = ROWS * COLS
COST = 0.1
# Hyperparameters
RANK = 20
LR = 1e-3
TAU = 0.005
NUM_STEPS = 25000
BATCH_SIZE = 256
GAMMA = np.exp(-COST)  # discount = exp(-c) since cost is uniform


def compute_rank(delta_D):
    U, S, Vt = np.linalg.svd(delta_D)

    print("Top 20 singular values:")
    for i in range(20):
        print(f"  Component {i+1}: {S[i]:.4f}")

    cumulative = np.cumsum(S**2) / np.sum(S**2)
    print(f"\nRank needed for 90% energy: {np.searchsorted(cumulative, 0.90) + 1}")
    print(f"Rank needed for 95% energy: {np.searchsorted(cumulative, 0.95) + 1}")
    print(f"Rank needed for 99% energy: {np.searchsorted(cumulative, 0.99) + 1}")


def visualize_delta(D_os, D_true, delta_D, probe_state, barrier_states, barrier_type):
    probe_row = delta_D[probe_state].reshape(ROWS, COLS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(D_os[probe_state].reshape(ROWS, COLS), cmap='viridis')
    axes[0].set_title(f'D_os row (state {probe_state})')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(D_true[probe_state].reshape(ROWS, COLS), cmap='viridis')
    axes[1].set_title(f'D_true row (state {probe_state})')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(probe_row, cmap='RdBu_r', vmin=-np.abs(probe_row).max(), vmax=np.abs(probe_row).max())
    axes[2].set_title(f'ΔD row (state {probe_state})')
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        for bs in barrier_states:
            br, bc = bs // COLS, bs % COLS
            ax.plot(bc, br, 'ks', markersize=4)

    plt.tight_layout()
    plt.savefig(f'../figs/baseline_{barrier_type}_delta_visualization.png', dpi=150)

    delta_norms = np.linalg.norm(delta_D, axis=1).reshape(ROWS, COLS)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(delta_norms, cmap='hot')
    ax2.set_title('||ΔD(s,:)|| across all states\n(where does the correction matter?)')
    for bs in barrier_states:
        br, bc = bs // COLS, bs % COLS
        ax2.plot(bc, br, 'cs', markersize=4)
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig(f'../figs/baseline_{barrier_type}_correction_locality.png', dpi=150)


def train_repair_network(model, target_model, optimizer, D_os_tensor, T_new, barrier_states, device):
    valid_states = [s for s in range(NUM_STATES) if s not in barrier_states]
    eye = torch.eye(NUM_STATES, device=device)
    losses = []

    for step in tqdm(range(NUM_STEPS)):
        batch_states = np.random.choice(valid_states, size=BATCH_SIZE)
        batch_next = np.array([sample_transition(T_new, s) for s in batch_states])

        D_os_s = D_os_tensor[batch_states]
        D_os_sp = D_os_tensor[batch_next]

        delta_s = model(D_os_s)
        with torch.no_grad():
            delta_sp = target_model(D_os_sp)

        D_s = D_os_s + delta_s
        D_sp_target = D_os_sp + delta_sp

        onehot_s = eye[batch_states]
        target = onehot_s + GAMMA * D_sp_target

        loss = nn.functional.mse_loss(D_s, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p, tp in zip(model.parameters(), target_model.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        losses.append(loss.item())

        if (step + 1) % 5000 == 0:
            print(f"  Step {step+1}/{NUM_STEPS}, Loss: {loss.item():.6f}")

    return losses


def evaluate_repair_network(model, D_os_tensor, D_os, D_true, probe_state, barrier_states, losses, barrier_type):
    model.eval()
    with torch.no_grad():
        delta_learned = model(D_os_tensor).cpu().numpy()

    D_learned = D_os + delta_learned

    recon_error = np.linalg.norm(D_learned - D_true) / np.linalg.norm(D_true)
    print(f"  Relative reconstruction error: {recon_error:.4f}")

    baseline_error = np.linalg.norm(D_os - D_true) / np.linalg.norm(D_true)
    print(f"  Baseline error (no repair): {baseline_error:.4f}")

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
    plt.savefig(f'../figs/baseline_{barrier_type}_comparison.png', dpi=150)
    print("  Saved comparison image")

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(losses[::100])
    ax3.set_xlabel('Step (x100)')
    ax3.set_ylabel('TD Loss')
    ax3.set_title('Training Loss')
    plt.tight_layout()
    plt.savefig(f'../figs/baseline_{barrier_type}_loss.png', dpi=150)
    print("  Saved loss curve image")

    print("\nDone!")


def main(get_rank=False, delta_vis=False, save_model=False,
         train_repair=True, eval_repair=False, barrier_type='horizontal'):
    """Main function"""
    print(f"Computing D_os for {barrier_type} barrier {ROWS}x{COLS} grid...")
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    print(f"  D_os shape: {D_os.shape}")

    print("\nPlacing barrier and computing D_true...")
    T_new, barrier_states = make_maze(T_os, barrier_type, ROWS, COLS)
    D_true = get_complete_DR(T=T_new)
    delta_D = D_true - D_os

    # Probe state for visualization: one row above the middle barrier cell
    mid_barrier = barrier_states[len(barrier_states) // 2]
    probe_state = mid_barrier - COLS

    if get_rank:
        compute_rank(delta_D)

    if delta_vis:
        visualize_delta(D_os, D_true, delta_D, probe_state, barrier_states, barrier_type)

    if train_repair:
        print("\nTraining repair network...")
        device = torch.device('mps')

        model = RepairNetworkLinear(num_states=NUM_STATES, hidden_dim=RANK).to(device)
        target_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        D_os_tensor = torch.tensor(D_os, dtype=torch.float32, device=device)

        losses = train_repair_network(model, target_model, optimizer, D_os_tensor, T_new, barrier_states, device)
        print("\nEvaluating repair network...")

    if eval_repair:
        evaluate_repair_network(model, D_os_tensor, D_os, D_true, probe_state, barrier_states, losses, barrier_type)

    if save_model:
        save_model_weights(model, path=f'src/networks/saved/{barrier_type}Barrier.pt')


if __name__ == '__main__':
    main(get_rank=False, delta_vis=False, save_model=True,
         train_repair=True, eval_repair=True, barrier_type='L')
