"""
Wernle simulation: agent mainly walks near the changed states.

Compares three Bellman sampling strategies:
1. Uniform — sample all states equally
2. Local — 80% of samples from wall-adjacent states, 20% uniform
3. Very local — 100% from wall-adjacent states only

Uses same place-cell PCA grid cell construction as sim_wernle.py:
Gaussian place cells on the 20x20 grid, PCA of covariance, Woodbury on D0.
"""

import os
import numpy as np
from numpy.linalg import norm, eigh
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import create_transition_open, get_complete_DR
from comp_maps import compute_M_from_H

SAVE_DIR = 'figures/wernle_local'
ROWS, COLS = 20, 20
N = ROWS * COLS
SPLIT_ROW = 10
d = N


def create_transition_divided(shape, split_row):
    rows, cols = shape
    T = create_transition_open(shape)
    wall_adjacent = []
    for c in range(cols):
        top_s = (split_row - 1) * cols + c
        bot_s = split_row * cols + c
        T[top_s, bot_s] = 0.0
        T[bot_s, top_s] = 0.0
        wall_adjacent.append(top_s)
        wall_adjacent.append(bot_s)
    for s in wall_adjacent:
        rs = T[s].sum()
        if rs > 0:
            T[s] /= rs
    return T, sorted(set(wall_adjacent))


def explore_wall_removal(T_new, T_old, wall_adjacent, shape):
    rows, cols = shape
    triples = []
    for s in wall_adjacent:
        r, c = s // cols, s % cols
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = nr*cols+nc
                if T_old[s, nb] == 0 and T_new[s, nb] > 0:
                    new_n = nb
                    for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr2, nc2 = r+dr2, c+dc2
                        if 0 <= nr2 < rows and 0 <= nc2 < cols:
                            nb2 = nr2*cols+nc2
                            if T_old[s, nb2] > 0:
                                triples.append((s, nb2, new_n))
    return triples


def consolidate_triples(triples, U_os):
    d = U_os.shape[1]
    groups = {}
    for s, s_lost, s_gained in triples:
        key = (s, s_lost)
        if key not in groups:
            groups[key] = []
        groups[key].append(s_gained)
    B = len(groups)
    phi_C = np.zeros((d, B))
    delta_Phi = np.zeros((d, B))
    for b, ((s, s_lost), gained) in enumerate(groups.items()):
        phi_C[:, b] = U_os[s, :]
        for g in gained:
            delta_Phi[:, b] += U_os[g, :] - U_os[s_lost, :]
    return phi_C, delta_Phi


# =========================================================================
# PLACE CELL GRID CELL CONSTRUCTION (matching sim_wernle.py's working version)
# =========================================================================

def make_place_cells(rows, cols, n_pc, pc_size, seed=0):
    """
    Create Gaussian place cells on a grid.
    Returns activation matrix (N, n_pc) and covariance (n_pc, n_pc).
    """
    rng = np.random.RandomState(seed)
    # Place cell centers: random positions in grid space
    centers = rng.rand(n_pc, 2) * np.array([cols, rows])

    # Compute place cell activation at each grid state
    activation = np.zeros((rows * cols, n_pc))
    for s in range(rows * cols):
        r, c = s // cols, s % cols
        pos = np.array([c + 0.5, r + 0.5])  # center of grid cell
        dists2 = np.sum((centers - pos)**2, axis=1)
        activation[s, :] = np.exp(-dists2 / (2 * pc_size**2))

    # Covariance of place cell activities
    Cov = activation.T @ activation / (rows * cols)
    return activation, Cov, centers


def build_place_cell_grid_cells(rows, cols, split_row, n_pc=200,
                                 pc_size=2.0, n_components=50,
                                 seed_bottom=0, seed_top=1):
    """
    Build grid cells from place cell PCA, matching last working sim_wernle.py.
    - Independent place cell populations per half (different seeds)
    - PCA eigenvectors of covariance as grid cells
    - J = [J_bottom; J_top], D0 = blkdiag(Cov_bottom, Cov_top)
    """
    N = rows * cols

    # Bottom half place cells
    act_bot, Cov_bot, _ = make_place_cells(split_row, cols, n_pc, pc_size, seed_bottom)
    evals_b, evecs_b = eigh(Cov_bot)
    idx_b = np.argsort(evals_b)[::-1][:n_components]
    J_bot = act_bot @ evecs_b[:, idx_b]  # (split_row*cols, n_components)

    # Top half place cells
    top_rows = rows - split_row
    act_top, Cov_top, _ = make_place_cells(top_rows, cols, n_pc, pc_size, seed_top)
    evals_t, evecs_t = eigh(Cov_top)
    idx_t = np.argsort(evals_t)[::-1][:n_components]
    J_top = act_top @ evecs_t[:, idx_t]  # (top_rows*cols, n_components)

    # Combine: J is (N, n_components), D0 is (N, N)
    J = np.zeros((N, n_components))
    J[:split_row * cols, :] = J_bot
    J[split_row * cols:, :] = J_top

    D0 = np.zeros((N, N))
    n_bot = split_row * cols
    # D0 from place cell activations: act @ Cov @ act.T effectively
    # But we want the spatial covariance for Woodbury: D0 = act @ Cov_pc @ act.T / normalization
    D0_bot = act_bot @ Cov_bot @ act_bot.T
    D0_top = act_top @ Cov_top @ act_top.T
    D0[:n_bot, :n_bot] = D0_bot
    D0[n_bot:, n_bot:] = D0_top

    return J, D0


def train_local(U_os, T_new, ec, d, phi_C, delta_Phi, M_true,
                wall_adjacent, local_frac=0.8,
                n_outer=5, bellman_samples=5000,
                bellman_lr=0.1, eq16_iters=200, eq16_lr=0.01, seed=42):
    """
    Bellman + Eq16 + rebuild, with biased sampling toward wall-adjacent states.
    """
    B = phi_C.shape[1]
    M = np.eye(d)
    theta = np.zeros(B)
    rng = np.random.RandomState(seed)
    wall_arr = np.array(wall_adjacent)
    all_states = np.arange(N)
    total = 0
    history = []

    def build_H(th):
        return np.diag(np.tanh(th)) @ delta_Phi.T

    for outer in range(n_outer):
        for t in range(bellman_samples):
            if rng.rand() < local_frac:
                s = rng.choice(wall_arr)
            else:
                s = rng.choice(all_states)
            sp = rng.choice(N, p=T_new[s, :])
            f_s = U_os[s]
            f_sp = U_os[sp]
            res = f_s + f_sp @ M - ec * f_s @ M
            M -= bellman_lr * 2.0 * np.outer(f_sp - ec * f_s, res)

        total += bellman_samples
        m_err = norm(M - M_true) / norm(M_true)

        # Eq 16
        for it in range(eq16_iters):
            w = np.tanh(theta)
            H = build_H(theta)
            res = np.eye(d) + phi_C @ H @ M - M
            loss = np.sum(res**2)
            g_w = 2.0 * np.sum((phi_C.T @ res) * (M.T @ delta_Phi).T, axis=1)
            theta -= eq16_lr * g_w * (1 - w**2)

        # Rebuild
        H = build_H(theta)
        M = compute_M_from_H(H, phi_C, d)
        m_err_post = norm(M - M_true) / norm(M_true)
        history.append((total, m_err, m_err_post))
        print(f"  Round {outer+1}: M_err(bellman)={m_err:.4f} -> M_err(rebuild)={m_err_post:.4f}")

    return M, theta, history


def wernle_measures(J, U, rows, cols, box_size=6):
    xy = np.zeros((2, N))
    for s in range(N):
        xy[0, s] = s % cols
        xy[1, s] = s // cols
    K = J.shape[1]
    corr_map = np.full((N, K), np.nan)
    for i in range(N):
        dist = np.abs(xy - xy[:, i:i+1])
        mask = np.all(dist <= box_size, axis=0)
        if mask.sum() < 4:
            continue
        for k in range(K):
            j_col = J[mask, k]
            u_col = U[mask, k]
            if np.std(j_col) < 1e-10 or np.std(u_col) < 1e-10:
                continue
            corr_map[i, k] = spearmanr(j_col, u_col).correlation
    return corr_map


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 70)
    print("WERNLE — LOCAL SAMPLING COMPARISON")
    print("=" * 70)

    # Build environments
    T_merged = create_transition_open((ROWS, COLS))
    T_divided, wall_adjacent = create_transition_divided((ROWS, COLS), SPLIT_ROW)

    D_divided = get_complete_DR(T_divided)
    D_merged = get_complete_DR(T_merged)

    # Eigenbasis from divided
    lam, V = np.linalg.eigh(D_divided)
    idx = np.argsort(lam)[::-1]
    lam = lam[idx]; V = V[:, idx]
    U_os = V * np.sqrt(np.maximum(lam, 0))
    ec = 1.0 + 1.0 / lam[0]

    # Ground truth M
    diffs = T_divided != T_merged
    delta_locs = np.unique(np.where(diffs)[0])
    R_gt = (T_merged - T_divided)[delta_locs, :]
    C_gt = np.eye(N)[:, delta_locs]
    M_true = compute_M_from_H(R_gt @ U_os, U_os.T @ C_gt, d)

    # Exploration triples
    triples = explore_wall_removal(T_merged, T_divided, wall_adjacent, (ROWS, COLS))
    phi_C, delta_Phi = consolidate_triples(triples, U_os)
    print(f"Wall-adjacent states: {len(wall_adjacent)}, triples: {len(triples)}, "
          f"groups: {phi_C.shape[1]}")

    # === Grid cell construction (place cell PCA, same as sim_wernle.py) ===
    print("\n--- Building grid cells (place cell PCA) ---")
    n_components = 50
    J, D0 = build_place_cell_grid_cells(
        ROWS, COLS, SPLIT_ROW,
        n_pc=200, pc_size=2.0, n_components=n_components,
        seed_bottom=0, seed_top=1)
    print(f"J shape: {J.shape}, D0 shape: {D0.shape}")

    # Woodbury correction on D0 (same as sim_wernle.py)
    R_full = T_divided - T_merged
    barrier_index = np.where(np.sum(np.abs(R_full), axis=1) > 1e-12)[0]
    r_wb = R_full[barrier_index, :]
    c_wb = np.eye(N)[:, barrier_index]
    Y = -r_wb @ D0 @ c_wb
    A_wb = np.linalg.inv(np.eye(Y.shape[0]) - Y)
    U_exact = J - D0 @ c_wb @ A_wb @ r_wb @ J  # ground truth U from exact Woodbury
    print(f"Woodbury correction: {len(barrier_index)} changed states")

    # === Run three conditions ===
    conditions = [
        ("Uniform (0%)", 0.0),
        ("Local (80%)", 0.8),
        ("Very local (100%)", 1.0),
    ]

    results = {}
    for label, frac in conditions:
        print(f"\n{'='*50}")
        print(f"  {label} wall-adjacent sampling")
        print(f"{'='*50}")

        M_learned, theta, history = train_local(
            U_os, T_merged, ec, d, phi_C, delta_Phi, M_true,
            wall_adjacent, local_frac=frac,
            n_outer=5, bellman_samples=5000,
            bellman_lr=0.1, eq16_iters=200, eq16_lr=0.01
        )

        D_learned = U_os @ M_learned @ U_os.T
        d_err = norm(D_learned - D_merged) / norm(D_merged)

        # Apply learned M to get U_learned via Woodbury on D0
        # The learned M gives us D_learned. We can also compute updated
        # grid cells by applying the same Woodbury formula but using
        # the learned D instead of exact.
        # However, to be consistent, let's compute U from the learned D
        # the same way: U = J - D0 @ c @ A @ r @ J but using the learned
        # transition change implicitly through D_learned.
        #
        # Actually the cleanest approach: the learned M modifies D.
        # The grid cell update is U = J - D0 * c * A * r * J.
        # The Woodbury correction on D0 only depends on the transition
        # change R = T_divided - T_merged (which is known — the wall was removed).
        # So U_exact is the same regardless of how well M was learned.
        #
        # What changes with local sampling is M and D_learned.
        # The Wernle analysis should compare J (divided) vs U from the
        # LEARNED D, not from exact Woodbury on D0.
        #
        # Use eigenvectors of D_learned as the merged grid cells.
        from numpy.linalg import eigh as eigh2
        evals_l, evecs_l = eigh2(D_learned)
        idx_l = np.argsort(evals_l)[::-1][:n_components]
        U_learned_eig = evecs_l[:, idx_l]

        # Also get eigenvectors of D_divided for J
        evals_d, evecs_d = eigh2(D_divided)
        idx_d = np.argsort(evals_d)[::-1][:n_components]
        J_eig = evecs_d[:, idx_d]

        # Match U to J
        corr_mat = np.abs(J_eig.T @ U_learned_eig)
        U_matched = np.zeros_like(J_eig)
        used = set()
        for k in range(n_components):
            order = np.argsort(corr_mat[k, :])[::-1]
            for ii in order:
                if ii not in used:
                    used.add(ii)
                    sign = np.sign(J_eig[:, k] @ U_learned_eig[:, ii])
                    U_matched[:, k] = (sign if sign != 0 else 1) * U_learned_eig[:, ii]
                    break

        # Wernle correlation (using DR eigenvectors)
        corr_map_eig = wernle_measures(J_eig, U_matched, ROWS, COLS, box_size=6)
        mean_corr_eig = np.nanmean(corr_map_eig, axis=1).reshape(ROWS, COLS)
        row_corrs_eig = np.array([np.nanmean(mean_corr_eig[r, :]) for r in range(ROWS)])
        center_eig = np.mean(row_corrs_eig[SPLIT_ROW-1:SPLIT_ROW+1])
        distal_eig = np.mean([row_corrs_eig[0], row_corrs_eig[-1]])

        # Also do Wernle on place-cell grid cells (J vs U_exact is same for all;
        # J vs U from D0 Woodbury using learned M)
        # For the place cell version, apply Woodbury on D0 using the LEARNED D
        # to derive new transition structure... but that's circular.
        # Instead, just use the place cell J vs U_exact for comparison, and
        # the DR eigenvectors for the local-sampling comparison.

        # Wernle on place cell grid cells (same for all conditions — exact Woodbury)
        corr_map_pc = wernle_measures(J, U_exact, ROWS, COLS, box_size=6)
        mean_corr_pc = np.nanmean(corr_map_pc, axis=1).reshape(ROWS, COLS)
        row_corrs_pc = np.array([np.nanmean(mean_corr_pc[r, :]) for r in range(ROWS)])
        center_pc = np.mean(row_corrs_pc[SPLIT_ROW-1:SPLIT_ROW+1])
        distal_pc = np.mean([row_corrs_pc[0], row_corrs_pc[-1]])

        results[label] = {
            'history': history,
            'D_err': d_err,
            'M_err': history[-1][2],
            'mean_corr_eig': mean_corr_eig,
            'row_corrs_eig': row_corrs_eig,
            'center_eig': center_eig,
            'distal_eig': distal_eig,
            'mean_corr_pc': mean_corr_pc,
            'row_corrs_pc': row_corrs_pc,
            'center_pc': center_pc,
            'distal_pc': distal_pc,
            'U_matched': U_matched,
            'J_eig': J_eig,
        }
        print(f"  D_err={d_err:.4f}")
        print(f"  DR eigvecs:  center={center_eig:.3f}, distal={distal_eig:.3f}, "
              f"effect={distal_eig-center_eig:.3f}")
        print(f"  Place cells: center={center_pc:.3f}, distal={distal_pc:.3f}, "
              f"effect={distal_pc-center_pc:.3f}")

    # === Visualization ===
    print(f"\n--- Plotting ---")
    labels = list(results.keys())
    colors = ['tab:blue', 'tab:orange', 'tab:red']

    # Figure 1: DR eigenvector correlation heatmaps + profiles
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, lab in enumerate(labels):
        mc = results[lab]['mean_corr_eig']
        vmin = max(np.nanmin(mc) - 0.02, -0.3)
        im = axes[0, i].imshow(mc, cmap='jet', vmin=vmin, vmax=1.0,
                                origin='lower', aspect='equal')
        axes[0, i].axhline(SPLIT_ROW - 0.5, color='w', ls='--', lw=2, alpha=0.7)
        axes[0, i].set_title(f'{lab}\ncenter={results[lab]["center_eig"]:.3f}, '
                              f'distal={results[lab]["distal_eig"]:.3f}')
        plt.colorbar(im, ax=axes[0, i], shrink=0.8)

        rc = results[lab]['row_corrs_eig']
        axes[1, i].plot(np.arange(ROWS), rc, '.-', color=colors[i], lw=2, ms=6)
        axes[1, i].axvline(SPLIT_ROW, color='gray', ls='--', lw=1.5)
        axes[1, i].set_xlabel('Row')
        axes[1, i].set_ylabel('Correlation')
        axes[1, i].set_title(f'{lab} — profile')
        axes[1, i].set_ylim(min(0, np.nanmin(rc) - 0.05), 1.05)
        axes[1, i].spines['top'].set_visible(False)
        axes[1, i].spines['right'].set_visible(False)

    plt.suptitle('Wernle Effect (DR Eigenvectors): Local vs Uniform Sampling',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/local_comparison_eig.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Place cell grid cell analysis (same for all — exact Woodbury)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mc_pc = results[labels[0]]['mean_corr_pc']  # same for all
    vmin = max(np.nanmin(mc_pc) - 0.02, -0.3)
    im = axes[0].imshow(mc_pc, cmap='jet', vmin=vmin, vmax=1.0,
                         origin='lower', aspect='equal')
    axes[0].axhline(SPLIT_ROW - 0.5, color='w', ls='--', lw=2, alpha=0.7)
    axes[0].set_title(f'Place Cell Grid Cells (exact Woodbury)\n'
                       f'center={results[labels[0]]["center_pc"]:.3f}, '
                       f'distal={results[labels[0]]["distal_pc"]:.3f}')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    rc_pc = results[labels[0]]['row_corrs_pc']
    axes[1].plot(np.arange(ROWS), rc_pc, 'k.-', lw=2, ms=6)
    axes[1].axvline(SPLIT_ROW, color='gray', ls='--', lw=1.5)
    axes[1].set_xlabel('Row')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Correlation Profile (Place Cell PCA)')
    axes[1].set_ylim(min(0, np.nanmin(rc_pc) - 0.05), 1.05)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/place_cell_exact.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Learning curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, lab in enumerate(labels):
        hist = results[lab]['history']
        samples = [h[0] for h in hist]
        errs = [h[2] for h in hist]
        ax.plot(samples, errs, '.-', color=colors[i], lw=2, ms=8, label=lab)
    ax.set_xlabel('Total Bellman samples')
    ax.set_ylabel('M error (post-rebuild)')
    ax.set_title('Learning Convergence')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 4: Rate maps (DR eigenvectors)
    J_eig = results[labels[0]]['J_eig']
    eig_show = [0, 1, 2, 3, 4, 5]
    fig, axes = plt.subplots(4, len(eig_show), figsize=(3*len(eig_show), 11))
    row_labels = ['J (divided)', 'U (uniform)', 'U (80% local)', 'U (100% local)']
    u_keys = [None, labels[0], labels[1], labels[2]]

    for col, k in enumerate(eig_show):
        m = J_eig[:, k].reshape(ROWS, COLS)
        vmax = np.abs(m).max()
        axes[0, col].imshow(m, cmap='jet', vmin=-vmax, vmax=vmax,
                            origin='lower', aspect='equal')
        axes[0, col].axhline(SPLIT_ROW-0.5, color='w', ls='--', lw=1.5)
        axes[0, col].set_title(f'Eig {k}', fontsize=10)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

        for row_i, key in enumerate(u_keys[1:], 1):
            u = results[key]['U_matched'][:, k].reshape(ROWS, COLS)
            axes[row_i, col].imshow(u, cmap='jet', vmin=-vmax, vmax=vmax,
                                     origin='lower', aspect='equal')
            axes[row_i, col].axhline(SPLIT_ROW-0.5, color='w', ls='--', lw=1.5, alpha=0.5)
            axes[row_i, col].set_xticks([]); axes[row_i, col].set_yticks([])

    for row_i, lab in enumerate(row_labels):
        axes[row_i, 0].set_ylabel(lab, fontsize=10, fontweight='bold')

    plt.suptitle('Rate Maps: Divided vs Merged (3 sampling strategies)', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/rate_maps_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 5: Place cell rate maps (exact Woodbury — same for all)
    pc_show = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, axes = plt.subplots(2, len(pc_show), figsize=(2.5*len(pc_show), 5))
    for col, k in enumerate(pc_show):
        m_j = J[:, k].reshape(ROWS, COLS)
        m_u = U_exact[:, k].reshape(ROWS, COLS)
        vmax = max(np.abs(m_j).max(), np.abs(m_u).max())

        axes[0, col].imshow(m_j, cmap='jet', vmin=-vmax, vmax=vmax,
                            origin='lower', aspect='equal')
        axes[0, col].axhline(SPLIT_ROW-0.5, color='w', ls='--', lw=1.5)
        axes[0, col].set_title(f'PC {k}', fontsize=9)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

        axes[1, col].imshow(m_u, cmap='jet', vmin=-vmax, vmax=vmax,
                            origin='lower', aspect='equal')
        axes[1, col].axhline(SPLIT_ROW-0.5, color='w', ls='--', lw=1.5, alpha=0.5)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel('J (divided)', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('U (merged)', fontsize=10, fontweight='bold')
    plt.suptitle('Place Cell Grid Cells (exact Woodbury)', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/place_cell_rate_maps.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'M_err':>8} {'D_err':>8} {'Center':>8} {'Distal':>8} {'Effect':>8}")
    print("-" * 70)
    print("DR Eigenvectors:")
    for lab in labels:
        r = results[lab]
        print(f"  {lab:<23} {r['M_err']:>8.4f} {r['D_err']:>8.4f} "
              f"{r['center_eig']:>8.3f} {r['distal_eig']:>8.3f} "
              f"{r['distal_eig']-r['center_eig']:>8.3f}")
    print(f"\nPlace Cell Grid Cells (exact Woodbury, same for all):")
    r0 = results[labels[0]]
    print(f"  {'Exact':23} {'—':>8} {'—':>8} "
          f"{r0['center_pc']:>8.3f} {r0['distal_pc']:>8.3f} "
          f"{r0['distal_pc']-r0['center_pc']:>8.3f}")
    print(f"\nFigures saved to {SAVE_DIR}/")


if __name__ == '__main__':
    main()
