import os
import numpy as np
from numpy.linalg import norm, eigh
import matplotlib.pyplot as plt

from utils import create_transition_open, get_complete_DR, make_wall_maze
from models import woodbury_feat

# =============================================================================
# CONFIG
# =============================================================================

ROWS, COLS   = 10, 10
NUM_STATES   = ROWS * COLS
BARRIER_TYPE = 'H'

K_VALUES   = [10, 25, 50, 75, 100]    # 10x10
# K_VALUES   = [45, 90, 135, 180, 225]  # 15x15

N_SAMPLES  = 800_000
LOG_EVERY  = 1_000
LR         = 0.001
SEED       = 42
SAVE_DIR   = "../data/recon/"


# =============================================================================
# HELPERS
# =============================================================================

def recon_errors(M, F, D_true):
    """Element-wise reconstruction errors between F @ M @ F^T and D_true."""
    D_approx = F @ M @ F.T
    diff = np.abs(D_approx - D_true)
    norm_D   = norm(D_true, 'fro')
    return {
        'frob'  : norm(D_approx - D_true, 'fro') / norm_D,
        'mean'  : np.mean(diff),
        'median': np.median(diff),
        'max'   : np.max(diff),
    }


def run_td(k, D_os, T_os, T_new, n_samples, log_every, lr, seed):
    """
    Run semi-gradient TD for a single value of K.
    """
    rng = np.random.RandomState(seed)

    # Feature basis for this k
    eigenvalues, eigvecs = eigh(D_os)
    idx = np.argsort(eigenvalues)[::-1][:k]
    lam = eigenvalues[idx]
    F = eigvecs[:, idx] * np.sqrt(lam)[np.newaxis, :]
    lam_inv = 1.0 / lam
    ec = 1.0 + 1.0 / lam[0]
    emc = 1.0 / ec

    # Woodbury oracle target for this K
    M_wb = woodbury_feat(T_os, T_new, F)
    D_true = F @ M_wb @ F.T

    print(f"  K={k:>3d}  ||D_true||_F={norm(D_true,'fro'):.4f}  "
          f"emc={emc:.4f}")

    n_logs = n_samples // log_every
    steps = np.zeros(n_logs, dtype=np.int64)
    frob_arr = np.zeros(n_logs)
    mean_arr = np.zeros(n_logs)
    med_arr = np.zeros(n_logs)
    max_arr = np.zeros(n_logs)

    M = np.eye(k)
    running_loss = 0.0
    log_idx = 0

    for t in range(n_samples):
        s  = rng.choice(NUM_STATES)
        sp = rng.choice(NUM_STATES, p=T_new[s])

        f_s  = F[s]
        f_sp = F[sp]
        r_s  = f_s * lam_inv

        delta = emc * (r_s + f_sp @ M) - f_s @ M
        M += lr * np.outer(f_s, delta)
        running_loss = 0.99 * running_loss + 0.01 * np.sum(delta ** 2)

        if (t + 1) % log_every == 0:
            errs = recon_errors(M, F, D_true)
            steps[log_idx] = t + 1
            frob_arr[log_idx] = errs['frob']
            mean_arr[log_idx] = errs['mean']
            med_arr[log_idx] = errs['median']
            max_arr[log_idx] = errs['max']

            if (t + 1) % (log_every * 100) == 0:
                print(f"    step {t+1:>7,}  frob={errs['frob']:.4f}  "
                      f"td_loss={running_loss:.4f}")
            log_idx += 1

    return {'steps': steps, 'frob': frob_arr,
            'mean': mean_arr, 'median': med_arr, 'max': max_arr}


# =============================================================================
# SAVE / LOAD
# =============================================================================

def save_results(all_results, save_dir=SAVE_DIR):
    """
    Save all_results dict to an .npz file.

    Keys in the npz: 'k_values', and for each k: '{k}_steps', '{k}_frob',
    '{k}_mean', '{k}_median', '{k}_max'.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "recon_errors.npz")
    arrays = {'k_values': np.array(list(all_results.keys()))}
    for k, res in all_results.items():
        for metric in ('steps', 'frob', 'mean', 'median', 'max'):
            arrays[f'{k}_{metric}'] = res[metric]
    np.savez(save_path, **arrays)
    print(f"  Saved -> {save_path}")


def load_results(save_dir=SAVE_DIR):
    """
    Load results saved by save_results().
    Returns the same all_results dict structure: {k: {metric: array}}.
    """
    path = os.path.join(save_dir, "recon_errors.npz")
    data = np.load(path)
    k_values = data['k_values'].tolist()
    all_results = {}
    for k in k_values:
        all_results[k] = {m: data[f'{k}_{m}']
                          for m in ('steps', 'frob', 'mean', 'median', 'max')}
    return all_results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(all_results, save_path):
    """
    2x2 grid: one subplot per metric (frob, mean, median, max).
    One line per K value. Legend drawn once on the first subplot.
    Y-axis label ("Error") only on the left column.
    X-axis ticks divided by 1000 (e.g. 100000 -> 100).
    """
    metrics = ['frob', 'mean', 'median', 'max']
    titles  = ['Frobenius error', 'Mean abs. error',
               'Median abs. error', 'Max abs. error']

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_results)))

    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = ax_idx // 2, ax_idx % 2
        ax = axes[row, col]

        for (k, res), color in zip(all_results.items(), colors):
            ax.plot(res['steps'] / 1_000, res[metric],
                    label=f'K = {k}', color=color, linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel('TD steps (×10³)')
        ax.set_xlim(left=0)

        if col == 0:
            ax.set_ylabel('Error')

        if ax_idx == 0:
            ax.legend(framealpha=0.8)

    fig.suptitle(
        f'Reconstruction error vs. TD steps\n'
        f'({ROWS}×{COLS} grid, H wall, semi-gradient TD, lr={LR})',
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved figure -> {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("RECONSTRUCTION: TD learning for K = " + str(K_VALUES))
    print("=" * 65)

    # Shared environment (same for all K)
    print("\n[1] Building environment ...")
    T_os = create_transition_open(shape=(ROWS, COLS))
    D_os = get_complete_DR(T=T_os)
    D_os = (D_os + D_os.T) / 2          # symmetrise
    T_new, _ = make_wall_maze(T_os, BARRIER_TYPE, ROWS, COLS)
    print(f"    Grid: {ROWS}x{COLS}, H wall placed.")

    # Run TD for each K
    print(f"\n[2] Running TD  ({N_SAMPLES:,} steps, log every {LOG_EVERY:,}) ...")
    all_results = {}
    for k in K_VALUES:
        print(f"\n--- K = {k} ---")
        all_results[k] = run_td(
            k, D_os, T_os, T_new,
            n_samples=N_SAMPLES,
            log_every=LOG_EVERY,
            lr=LR,
            seed=SEED,
        )

    # Print final summary
    print("\n" + "=" * 65)
    print("SUMMARY  (final values)")
    print("=" * 65)
    print(f"  {'K':>4}  {'frob':>8}  {'mean':>10}  {'median':>10}  {'max':>10}")
    for k, res in all_results.items():
        print(f"  {k:>4}  {res['frob'][-1]:>8.4f}  {res['mean'][-1]:>10.6f}"
              f"  {res['median'][-1]:>10.6f}  {res['max'][-1]:>10.6f}")

    # Save
    print("\n[3] Saving results ...")
    save_results(all_results)

    # Plot
    print("\n[4] Plotting ...")
    plot_results(all_results, save_path='../figures/reconstruction_15x15.png')


if __name__ == '__main__':
    main()
