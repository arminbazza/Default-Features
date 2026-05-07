import os
import numpy as np
from numpy.linalg import inv, eigh
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import create_transition_open

ROWS, COLS, SPLIT = 20, 20, 10
N = ROWS*COLS
COST = 0.01
LR = 0.001
K = N  

SAVE_DIR = '../figures/wernle_reverse/'
DATA_DIR = '../data/wernle/'
os.makedirs(SAVE_DIR, exist_ok=True)

def create_transition_open(shape):
    rows, cols = shape
    N = rows * cols
    T = np.zeros((N, N))
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            nb = []
            if r > 0:        nb.append((r-1)*cols + c)
            if r < rows-1:   nb.append((r+1)*cols + c)
            if c > 0:        nb.append(r*cols + (c-1))
            if c < cols-1:   nb.append(r*cols + (c+1))
            for n in nb:
                T[s, n] = 1.0 / len(nb)
    return T


def create_transition_divided(shape, split_row):
    rows, cols = shape
    T = create_transition_open(shape)
    adj = []
    for c in range(cols):
        top = (split_row-1)*cols+c; bot = split_row*cols+c
        T[top,bot]=0; T[bot,top]=0; adj+=[top,bot]
    for s in sorted(set(adj)):
        rs=T[s].sum()
        if rs>0: T[s]/=rs
    return T, adj


def woodbury_td(F, T_new, emc, lam, grad_type='full',
                       n_total_samples=100000, lr=0.001,
                       log_every=10000, seed=42):
    N, k = F.shape
    M = np.eye(k)
    lam_inv = 1.0 / lam[:k]
    rng = np.random.RandomState(seed)

    for t in range(n_total_samples):
        s = rng.randint(N)
        sp = rng.choice(N, p=T_new[s, :])
        f_s = F[s, :]
        f_sp = F[sp, :]
        r_s = f_s * lam_inv
        
        delta = emc * (r_s + f_sp @ M) - f_s @ M
        if grad_type == "full":
            grad = 2.0 * np.outer(f_sp - 1/emc * f_s, delta)
            M -= lr * grad
        elif grad_type == "semi":
            M += lr * np.outer(f_s, delta)

        # logging
        if log_every and (t + 1) % log_every == 0:
            print(f"  woodbury_td step {t+1:>7,}")

    return M

def woodbury_td_uniform(F, T_new, emc, lam, grad_type='full',
                       n_total_samples=100000, lr=0.001,
                       log_every=10000, seed=42):
    N, k = F.shape
    M = np.eye(k)
    lam_inv = 1.0 / lam[:k]
    rng = np.random.RandomState(seed)

    # Calculate how many full, equal sweeps we can do
    n_epochs = n_total_samples // N
    
    t = 0  # Global step counter for logging
    
    for epoch in range(n_epochs):
        # Shuffle states to ensure equal sampling without cyclical update bias
        states = rng.permutation(N)
        
        for s in states:
            sp = rng.choice(N, p=T_new[s, :])
            f_s = F[s, :]
            f_sp = F[sp, :]
            r_s = f_s * lam_inv
            
            delta = emc * (r_s + f_sp @ M) - f_s @ M
            if grad_type == "full":
                grad = 2.0 * np.outer(f_sp - 1/emc * f_s, delta)
                M -= lr * grad
            elif grad_type == "semi":
                M += lr * np.outer(f_s, delta)

            t += 1
            
            # logging
            if log_every and t % log_every == 0:
                print(f"  woodbury_td step {t:>7,}")

    return M

print(f"Grid: {ROWS}x{COLS}, K={K}, cost={COST}, lr={LR}")

T_div, _ = create_transition_divided((ROWS,COLS), SPLIT)
T_os = create_transition_open((ROWS,COLS))
D_os = inv(np.exp(COST)*np.eye(N) - T_os)

eigenvalues, eigvecs = eigh(D_os)
idx = np.argsort(eigenvalues)[::-1][:K]
lam = eigenvalues[idx]
ec = 1.0 + 1.0 / lam[0]
emc = 1.0 / ec
F = eigvecs[:,idx] * np.sqrt(np.maximum(lam, 1e-10))

# Woodbury perturbation for wall
delta = np.unique(np.where(T_os != T_div)[0])
R_w = (T_div - T_os)[delta,:]
C_w = np.eye(N)[:,delta]
A_w = inv(np.eye(len(delta)) - R_w @ D_os @ C_w)
 
F_divided = F + D_os @ C_w @ A_w @ R_w @ F  # features WITH wall
np.save(DATA_DIR+'F_divided.npy', F_divided)

# M = woodbury_td(F_divided, T_os, emc, lam, grad_type='semi',
#                        n_total_samples=500_000, lr=LR,
#                        log_every=100_000)
M = woodbury_td_uniform(F_divided, T_os, emc, lam, grad_type='semi',
                       n_total_samples=500_000, lr=LR,
                       log_every=100_000)
F_merged = F_divided @ M
np.save(DATA_DIR+f"F_merged.npy", F_merged)

print(f"F_divided: {F_divided.shape}, F_merged: {F_merged.shape}")

# Wernle sliding correlation
coords = np.array([[r,c] for r in range(ROWS) for c in range(COLS)])
corr = np.full((N,K), np.nan)
for i in range(N):
    d = np.abs(coords - coords[i])
    succ = np.all(d <= 3, axis=1)
    
    same_comp = (coords[:, 0] < SPLIT) == (coords[i, 0] < SPLIT)
    succ = succ & same_comp
    
    if succ.sum() < 4: continue
    for k in range(K):
        b = F_divided[succ,k]; a = F_merged[succ,k]
        if np.std(b)<1e-10 or np.std(a)<1e-10: continue
        corr[i,k] = spearmanr(b,a).correlation

mm = np.nanmean(corr, axis=1).reshape(ROWS,COLS)
rp = np.nanmean(mm, axis=1)

wall_rows = slice(SPLIT-2, SPLIT+2)
distal_rows = list(range(3)) + list(range(ROWS-3, ROWS))
print(f"Wall-adjacent: {np.nanmean(rp[wall_rows]):.3f}")
print(f"Distal: {np.nanmean(rp[distal_rows]):.3f}")
print(f"Effect: {np.nanmean(rp[distal_rows]) - np.nanmean(rp[wall_rows]):.3f}")

# ---- Plots ----
# Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
vmin = mm.min()
vmax = mm.max()
im = axes[0].imshow(mm, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
axes[0].set_title(f'Correlation heatmap (cost={COST})')
axes[0].set_xlabel('Column'); axes[0].set_ylabel('Row')
plt.colorbar(im, ax=axes[0], shrink=0.8)
 
axes[1].plot(rp, 'b-o', markersize=4)
axes[1].axvline(x=SPLIT-0.5, color='red', linestyle='--', alpha=0.5, label='Wall')
axes[1].set_xlabel('Row'); axes[1].set_ylabel('Mean Spearman r')
axes[1].set_title('Row profile'); axes[1].legend()
axes[1].set_ylim([vmin, vmax])
 
plt.tight_layout()
plt.savefig(SAVE_DIR+f"wernle_semi_lr{LR}_c{COST}.png", dpi=150, bbox_inches='tight')
plt.close()

# Individual features
feat_indices = [5, 8, 10, 12, 15, 18, 25, 30, 35, 40, 45, 50]
n_feat = len(feat_indices)
fig, axes = plt.subplots(n_feat, 3, figsize=(10, 3*n_feat))
for row, k in enumerate(feat_indices):
    map_b = F_divided[:, k].reshape(ROWS, COLS)
    map_a = F_merged[:, k].reshape(ROWS, COLS)
    map_c = corr[:, k].reshape(ROWS, COLS)
    axes[row,0].imshow(map_b, origin='lower', cmap='jet')
    axes[row,0].set_title(f'F_divided (feat {k})')
    axes[row,1].imshow(map_a, origin='lower', cmap='jet')
    axes[row,1].set_title(f'F_merged (feat {k})')
    im = axes[row,2].imshow(map_c, origin='lower', cmap='jet', vmin=np.min(corr[:,k]), vmax=np.max(corr[:,k]))
    axes[row,2].set_title(f'Corr (feat {k})')
    plt.colorbar(im, ax=axes[row,2], shrink=0.8)
    for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
 
plt.tight_layout()
plt.savefig(SAVE_DIR+f"wernle_features_lr{LR}_c{COST}.png", dpi=150, bbox_inches='tight')
plt.close()
 
print("Plots saved.")