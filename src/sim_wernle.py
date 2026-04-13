"""
Simplified Wernle et al. (2018) simulation.

Setup: 20x20 grid divided by horizontal wall at row 10.
Agent learns DR for divided environment, computes eigenbasis.
Wall is removed. Agent learns correction M via Bellman + Eq 16.
Updated grid cells = U_os @ M_learned.
Analysis: sliding correlation shows grid fields change near wall,
preserved far from wall.

T_old is ONLY used for ground truth verification, never in the algorithm.
"""

import os
import numpy as np
from numpy.linalg import inv, norm, eigh
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import create_transition_open, get_complete_DR

ROWS, COLS = 20, 20
NUM_STATES = ROWS * COLS
SPLIT_ROW = 10
d = NUM_STATES

SAVE_DIR = 'figures/wernle'
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================================
# 1. ENVIRONMENT
# =========================================================================

def create_transition_divided(shape, split_row):
    """
    Transition matrix with horizontal wall between split_row-1 and split_row.
    All states remain accessible; only cross-wall transitions are removed
    and rows renormalized.
    """
    rows, cols = shape
    T = create_transition_open(shape)
    wall_adjacent = []

    for c in range(cols):
        top = (split_row - 1) * cols + c
        bot = split_row * cols + c
        T[top, bot] = 0.0
        T[bot, top] = 0.0
        wall_adjacent.append(top)
        wall_adjacent.append(bot)

    wall_adjacent = sorted(set(wall_adjacent))
    for s in wall_adjacent:
        row_sum = T[s].sum()
        if row_sum > 0:
            T[s] /= row_sum

    return T, wall_adjacent


# =========================================================================
# 2. EIGENBASIS
# =========================================================================

def compute_eigenbasis(D, num_eigvecs=None):
    """F = V * sqrt(lambda), so D ≈ F @ F^T."""
    if num_eigvecs is None:
        num_eigvecs = D.shape[0]
    eigenvalues, eigvecs = eigh(D)
    idx = np.argsort(eigenvalues)[::-1][:num_eigvecs]
    lam = eigenvalues[idx]
    V = eigvecs[:, idx]
    F = V * np.sqrt(np.maximum(lam, 1e-10))[np.newaxis, :]
    return lam, V, F


# =========================================================================
# 3. WOODBURY HELPERS (for ground truth only)
# =========================================================================

def compute_M_from_H(H, G, d):
    """M = I + G @ (I_B - H @ G)^{-1} @ H"""
    B = H.shape[0]
    Z = H @ G
    A = inv(np.eye(B) - Z)
    return np.eye(d) + G @ A @ H


def compute_ground_truth_M(T_divided, T_merged, F):
    """Exact M from known transition change. ONLY for verification."""
    diffs = T_divided != T_merged
    delta_locs = np.unique(np.where(diffs)[0])
    N = T_divided.shape[0]
    R = (T_merged - T_divided)[delta_locs, :]
    C = np.eye(N)[:, delta_locs]
    H_gt = R @ F
    G_gt = F.T @ C
    return compute_M_from_H(H_gt, G_gt, F.shape[1]), delta_locs


# =========================================================================
# 4. EXPLORATION (T-free)
# =========================================================================

def explore_wall_removal(T_merged, wall_adjacent, grid_shape,
                         split_row, n_samples=20, seed=42):
    """
    Agent discovers that the wall is gone by sampling transitions.
    
    The agent knows:
    - Which states were wall-adjacent (from prior experience)
    - The wall location (it used to be there)
    - That cross-wall transitions are now possible (it tries and succeeds)
    
    For each wall-adjacent state s, the agent knows which neighbor was
    previously across the wall (from grid topology + wall location).
    It samples transitions to discover which successors it now reaches.
    
    Returns (s, s_old_neighbor, s_new_neighbor) triples where:
    - s_old_neighbor: a neighbor that s could already reach (lost some mass)
    - s_new_neighbor: the neighbor across the former wall (gained mass)
    
    Does NOT use T_old.
    """
    rows, cols = grid_shape
    N = rows * cols
    rng = np.random.RandomState(seed)
    triples = []

    for s in wall_adjacent:
        r, c = s // cols, s % cols

        # Identify the cross-wall neighbor from topology + wall location
        new_neighbors = []
        old_neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = nr * cols + nc
                # The wall was between split_row-1 and split_row
                # Cross-wall = one side is < split_row, other is >= split_row
                s_side = r < split_row
                nb_side = nr < split_row
                if s_side != nb_side:
                    new_neighbors.append(nb)
                else:
                    old_neighbors.append(nb)

        if not new_neighbors or not old_neighbors:
            continue

        # Sample transitions to confirm new neighbors are reachable
        experienced = set()
        for _ in range(n_samples):
            s_next = rng.choice(N, p=T_merged[s, :])
            experienced.add(s_next)

        # Create triples: for each (old, new) pair
        for new_n in new_neighbors:
            if new_n in experienced:  # confirm it's reachable
                for old_n in old_neighbors:
                    triples.append((s, old_n, new_n))

    return triples


def consolidate_triples(triples, F):
    """Group triples by (s, s_old), sum feature contrasts."""
    dd = F.shape[1]
    groups = {}
    for s, s_old, s_new in triples:
        key = (s, s_old)
        if key not in groups:
            groups[key] = []
        groups[key].append(s_new)

    B = len(groups)
    phi_C = np.zeros((dd, B))
    delta_Phi = np.zeros((dd, B))
    group_info = []

    for b, ((s, s_old), new_list) in enumerate(groups.items()):
        phi_C[:, b] = F[s, :]
        for n in new_list:
            delta_Phi[:, b] += F[n, :] - F[s_old, :]
        group_info.append((s, s_old, new_list))

    return phi_C, delta_Phi, group_info


# =========================================================================
# 5. LEARNING: BELLMAN + EQ 16 + WOODBURY REBUILD
# =========================================================================

def train_combined(F, T_new, ec, dd,
                   phi_C, delta_Phi,
                   M_true=None,
                   n_outer=10, bellman_samples=10000,
                   bellman_lr=0.1, eq16_iters=200, eq16_lr=0.01,
                   seed=42):
    """
    Combined Bellman + Eq 16, T-free.
    All H rows are learnable (no barrier anchor for wall removal).
    """
    N = F.shape[0]
    B = phi_C.shape[1]
    M = np.eye(dd)
    theta = np.zeros(B)
    rng = np.random.RandomState(seed)
    total = 0

    for outer in range(n_outer):
        # Stage 1: Bellman SGD
        for _ in range(bellman_samples):
            s = rng.randint(N)
            sp = rng.choice(N, p=T_new[s, :])
            f_s = F[s, :]
            f_sp = F[sp, :]
            res = f_s + f_sp @ M - ec * f_s @ M
            M -= bellman_lr * 2.0 * np.outer(f_sp - ec * f_s, res)

        total += bellman_samples
        m_str = ""
        if M_true is not None:
            m_str = f", M_err={norm(M - M_true) / norm(M_true):.4f}"
        print(f"  Round {outer+1} Bellman ({total} samples){m_str}")

        # Stage 2: Eq 16 with fixed Bellman M
        for _ in range(eq16_iters):
            w = np.tanh(theta)
            H = np.diag(w) @ delta_Phi.T
            res = np.eye(dd) + phi_C @ H @ M - M
            loss = np.sum(res ** 2)
            temp1 = phi_C.T @ res
            temp2 = (M.T @ delta_Phi).T
            grad_w = 2.0 * np.sum(temp1 * temp2, axis=1)
            theta -= eq16_lr * grad_w * (1.0 - w ** 2)

        print(f"           Eq 16: w_mean={np.mean(np.tanh(theta)):.4f}, loss={loss:.2e}")

        # Stage 3: Woodbury rebuild
        w = np.tanh(theta)
        H = np.diag(w) @ delta_Phi.T
        M = compute_M_from_H(H, phi_C, dd)

        m_str = ""
        if M_true is not None:
            m_str = f", M_err={norm(M - M_true) / norm(M_true):.4f}"
        print(f"           Rebuild{m_str}")
        print()

    return M, theta


# =========================================================================
# 6. WERNLE ANALYSIS
# =========================================================================

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
    
    F_before: (N, d) eigenbasis before wall removal (= F = U_os)
    F_after:  (N, d) eigenbasis after wall removal (= F @ M)
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


# =========================================================================
# 7. VISUALIZATION
# =========================================================================

def plot_eigenvector_comparison(F_before, F_after, eig_indices, rows, cols,
                                split_row, save_path=None):
    """Side-by-side: eigenvectors before vs after wall removal."""
    n = len(eig_indices)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes[:, np.newaxis]

    for i, k in enumerate(eig_indices):
        mb = F_before[:, k].reshape(rows, cols)
        ma = F_after[:, k].reshape(rows, cols)
        vmax = max(np.abs(mb).max(), np.abs(ma).max())

        axes[0, i].imshow(mb, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          origin='lower')
        axes[0, i].axhline(split_row - 0.5, color='k', ls='--', lw=1.5)
        axes[0, i].set_title(f'Eig {k} (divided)')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        axes[1, i].imshow(ma, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          origin='lower')
        axes[1, i].axhline(split_row - 0.5, color='k', ls='--', lw=1.5,
                          alpha=0.5)
        axes[1, i].set_title(f'Eig {k} (merged)')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(corr_map, split_row, save_path=None):
    """2D heatmap of sliding correlation."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(corr_map, cmap='RdBu_r', vmin=-0.2, vmax=1.0,
                   origin='lower')
    ax.axhline(split_row - 0.5, color='k', ls='--', lw=1.5)
    ax.set_title('Sliding Correlation (divided vs merged)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_row_profile(row_profile, split_row, save_path=None):
    """1D profile: correlation vs row."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(row_profile)), row_profile, 'k.-', lw=1.5)
    ax.axvline(split_row - 0.5, color='r', ls='--', lw=1.5, label='Former wall')
    ax.set_xlabel('Row')
    ax.set_ylabel('Mean correlation')
    ax.set_title('Wernle Effect: Correlation Profile')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("WERNLE et al. (2018) — SIMPLIFIED SIMULATION")
    print("=" * 70)

    # === 1. Build environments ===
    print("\n--- 1. Environments ---")
    T_divided, wall_adjacent = create_transition_divided((ROWS, COLS), SPLIT_ROW)
    T_merged = create_transition_open(shape=(ROWS, COLS))
    print(f"Grid: {ROWS}x{COLS}, wall at row {SPLIT_ROW}")
    print(f"Wall-adjacent states: {len(wall_adjacent)}")

    # === 2. DR and eigenbasis from divided environment ===
    print("\n--- 2. DR + Eigenbasis ---")
    D_divided = get_complete_DR(T_divided)
    lam, V, F = compute_eigenbasis(D_divided, num_eigvecs=d)
    ec = 1.0 + 1.0 / lam[0]
    recon_err = norm(F @ F.T - D_divided) / norm(D_divided)
    print(f"F reconstruction error: {recon_err:.2e}")
    print(f"e^c = {ec:.6f}")

    # Ground truth M (for verification only)
    M_true, delta_locs = compute_ground_truth_M(T_divided, T_merged, F)
    print(f"Changed states: {len(delta_locs)}")

    # === 3. Exploration ===
    print("\n--- 3. Exploration (T-free) ---")
    triples = explore_wall_removal(T_merged, wall_adjacent, (ROWS, COLS),
                                    SPLIT_ROW, n_samples=20)
    phi_C, delta_Phi, groups = consolidate_triples(triples, F)
    print(f"Triples: {len(triples)} -> {len(groups)} consolidated groups")

    # === 4. Learn M ===
    print("\n--- 4. Learn M (Bellman + Eq 16) ---")
    print("=" * 70)
    M_learned, theta = train_combined(
        F, T_merged, ec, d,
        phi_C, delta_Phi,
        M_true=M_true,
        n_outer=10, bellman_samples=10000,
        bellman_lr=0.1, eq16_iters=200, eq16_lr=0.01
    )

    # === 5. Update eigenbasis using M_learned ===
    print("--- 5. Update eigenbasis ---")
    F_before = F.copy()             # Grid cells before wall removal
    F_after = F @ M_learned          # Grid cells after wall removal
    F_after_true = F @ M_true        # Ground truth for comparison

    # Quick check: how different are they?
    update_err = norm(F_after - F_after_true) / norm(F_after_true)
    print(f"||F_learned - F_true|| / ||F_true||: {update_err:.4f}")

    # === 6. Wernle analysis ===
    print("\n--- 6. Wernle analysis ---")
    eig_indices = list(range(20))  # first 20 eigenvectors

    # Learned M
    mean_corr, row_profile, _ = analyze_wernle(
        F_before, F_after, ROWS, COLS, SPLIT_ROW,
        eig_indices=eig_indices, box_radius=3)

    # Ground truth M
    mean_corr_gt, row_profile_gt, _ = analyze_wernle(
        F_before, F_after_true, ROWS, COLS, SPLIT_ROW,
        eig_indices=eig_indices, box_radius=3)

    center = np.mean(row_profile[SPLIT_ROW-2:SPLIT_ROW+2])
    distal = np.mean([np.mean(row_profile[:3]), np.mean(row_profile[-3:])])
    center_gt = np.mean(row_profile_gt[SPLIT_ROW-2:SPLIT_ROW+2])
    distal_gt = np.mean([np.mean(row_profile_gt[:3]), np.mean(row_profile_gt[-3:])])

    print(f"Learned M  — center corr: {center:.3f}, distal corr: {distal:.3f}, "
          f"effect: {distal - center:.3f}")
    print(f"True M     — center corr: {center_gt:.3f}, distal corr: {distal_gt:.3f}, "
          f"effect: {distal_gt - center_gt:.3f}")
    print(f"Expected: distal > center (Wernle effect)")

    # === 7. Plots ===
    print("\n--- 7. Plots ---")
    plot_eigs = [0, 1, 3, 5, 9]
    plot_eigenvector_comparison(F_before, F_after, plot_eigs, ROWS, COLS,
                                SPLIT_ROW, f'{SAVE_DIR}/eigvec_comparison.png')
    plot_eigenvector_comparison(F_before, F_after_true, plot_eigs, ROWS, COLS,
                                SPLIT_ROW, f'{SAVE_DIR}/eigvec_comparison_gt.png')
    plot_correlation_heatmap(mean_corr, SPLIT_ROW,
                             f'{SAVE_DIR}/corr_heatmap_learned.png')
    plot_correlation_heatmap(mean_corr_gt, SPLIT_ROW,
                             f'{SAVE_DIR}/corr_heatmap_gt.png')
    plot_row_profile(row_profile, SPLIT_ROW,
                     f'{SAVE_DIR}/row_profile_learned.png')
    plot_row_profile(row_profile_gt, SPLIT_ROW,
                     f'{SAVE_DIR}/row_profile_gt.png')

    # Both profiles on one plot for comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(ROWS), row_profile, 'b.-', lw=1.5, label='Learned M')
    ax.plot(range(ROWS), row_profile_gt, 'r.--', lw=1.5, label='True M')
    ax.axvline(SPLIT_ROW - 0.5, color='gray', ls='--', lw=1.5, label='Former wall')
    ax.set_xlabel('Row')
    ax.set_ylabel('Mean correlation')
    ax.set_title('Wernle Effect: Learned vs Ground Truth')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/row_profile_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Figures saved to {SAVE_DIR}/")

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  M error (learned vs true):    {norm(M_learned - M_true) / norm(M_true):.4f}")
    print(f"  F update error:               {update_err:.4f}")
    print(f"  Wernle effect (learned):      {distal - center:.3f}")
    print(f"  Wernle effect (true M):       {distal_gt - center_gt:.3f}")
    print(f"  No T_old used in algorithm:   YES")


if __name__ == '__main__':
    main()