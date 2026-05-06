"""
peng_simulation.py — Simulating Peng et al. (2025) Figures 6g & 6b
Multi-seed version for box-and-whisker plots.
Input-based inclusion criteria (no reference to M_true for filtering).
"""
import os
import numpy as np
from numpy.linalg import inv, norm, eigh
from scipy.stats import wilcoxon, spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec



ROWS, COLS = 15, 15
N = ROWS * COLS
d = N
LEVER_SIZE = 5

SAVE_DIR = '../figures/peng'
DATA_DIR = '../data/peng'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MIN_SEARCH_STATES = 30
MIN_LEVER_NB_STATES = 14


# =========================================================================
# UTILITIES
# =========================================================================

def create_transition_open(rows, cols):
    N = rows * cols
    T = np.zeros((N, N))
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            nbs = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nbs.append(nr * cols + nc)
            for nb in nbs:
                T[s, nb] = 1.0 / len(nbs)
    return T


def get_complete_DR(T, gamma=0.95):
    return inv((1.0/gamma) * np.eye(T.shape[0]) - T)


def state_to_rc(s):
    return s // COLS, s % COLS


def rc_to_state(r, c):
    return r * COLS + c


def compute_eigenbasis(D):
    eigenvalues, eigvecs = eigh(D)
    idx = np.argsort(eigenvalues)[::-1][:d]
    lam = eigenvalues[idx]
    V = eigvecs[:, idx]
    F = V * np.sqrt(np.maximum(lam, 0))[np.newaxis, :]
    return lam, F


def random_lever_position(rng, size=4, margin=1):
    """Random top-left corner for a size x size lever block."""
    max_r = ROWS - size - margin
    max_c = COLS - size - margin
    r = rng.randint(margin, max_r + 1)
    c = rng.randint(margin, max_c + 1)
    return r, c


def setup_lever(F, T_open, lever_states):
    """All lever-dependent setup: T_lever, M_true, exploration, projections."""
    T_lever = create_lever_T(T_open, lever_states)
    M_true = compute_M_true(T_open, T_lever, F)

    H_barrier, phi_C_barrier = compute_barrier_H(lever_states, F)
    triples = explore_lever(T_lever, lever_states, n_samples=20)
    phi_C_triples, delta_Phi = consolidate_triples(triples, F)

    n_barrier = len(lever_states)
    n_triples = phi_C_triples.shape[1]
    phi_C_full = np.hstack([phi_C_barrier, phi_C_triples])
    barrier_idx = np.arange(n_barrier)
    neighbor_idx = np.arange(n_barrier, n_barrier + n_triples)

    search_states = np.array(get_far_states(lever_states, min_distance=3))
    lever_nb_states = np.array(get_neighbor_states(lever_states, distance=1))

    return (T_lever, M_true,
            phi_C_full, H_barrier, delta_Phi,
            barrier_idx, neighbor_idx,
            search_states, lever_nb_states)


def compute_rdm(F_matrix):
    """Compute representational dissimilarity matrix (squared Euclidean)."""
    # Using pdist for efficiency
    from scipy.spatial.distance import pdist
    return pdist(F_matrix, metric='sqeuclidean')


# =========================================================================
# LEVER ENVIRONMENT
# =========================================================================

def get_lever_states(center_r, center_c, size=4):
    states = []
    for dr in range(size):
        for dc in range(size):
            r, c = center_r + dr, center_c + dc
            if 0 <= r < ROWS and 0 <= c < COLS:
                states.append(rc_to_state(r, c))
    return states


def create_lever_T(T_open, lever_states):
    T = T_open.copy()
    lever_set = set(lever_states)
    for ls in lever_states:
        T[ls, :] = 0.0
    for s in range(N):
        if s in lever_set: continue
        changed = False
        for ls in lever_states:
            if T[s, ls] > 0:
                T[s, ls] = 0.0
                changed = True
        if changed:
            row_sum = T[s].sum()
            if row_sum > 0: T[s] /= row_sum
    return T


def get_neighbor_states(lever_states, distance=1):
    lever_set = set(lever_states)
    neighbors = set()
    for ls in lever_states:
        r, c = state_to_rc(ls)
        for dr in range(-distance, distance+1):
            for dc in range(-distance, distance+1):
                nr, nc = r+dr, c+dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    s = rc_to_state(nr, nc)
                    if s not in lever_set:
                        neighbors.add(s)
    return sorted(neighbors)


def get_far_states(lever_states, min_distance=3):
    lever_set = set(lever_states)
    far = []
    for s in range(N):
        if s in lever_set: continue
        r, c = state_to_rc(s)
        min_d = min(max(abs(r - state_to_rc(ls)[0]),
                        abs(c - state_to_rc(ls)[1]))
                    for ls in lever_states)
        if min_d >= min_distance:
            far.append(s)
    return far


# =========================================================================
# WOODBURY
# =========================================================================

def compute_M_from_H(H, G, d):
    B = H.shape[0]
    return np.eye(d) + G @ inv(np.eye(B) - H @ G) @ H


def compute_M_true(T_old, T_new, F):
    diffs = T_old != T_new
    delta_locs = np.unique(np.where(diffs)[0])
    R = (T_new - T_old)[delta_locs, :]
    C = np.eye(N)[:, delta_locs]
    return compute_M_from_H(R @ F, F.T @ C, F.shape[1])


# =========================================================================
# EXPLORATION
# =========================================================================

def compute_barrier_H(lever_states, F):
    n_barrier = len(lever_states)
    H_barrier = np.zeros((n_barrier, d))
    phi_C_barrier = np.zeros((d, n_barrier))
    for i, ls in enumerate(lever_states):
        r, c = state_to_rc(ls)
        nbs = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                nbs.append(rc_to_state(nr, nc))
        H_barrier[i] = -(1.0/len(nbs)) * np.sum(F[nbs], axis=0)
        phi_C_barrier[:, i] = F[ls]
    return H_barrier, phi_C_barrier


def explore_lever(T_lever, lever_states, n_samples=20, seed=42):
    rng = np.random.RandomState(seed)
    lever_set = set(lever_states)
    affected = set()
    for ls in lever_states:
        r, c = state_to_rc(ls)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                nb = rc_to_state(nr, nc)
                if nb not in lever_set:
                    affected.add(nb)
    triples = []
    for s in affected:
        experienced = set()
        for _ in range(n_samples):
            sp = rng.choice(N, p=T_lever[s])
            experienced.add(sp)
        r, c = state_to_rc(s)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                nb = rc_to_state(nr, nc)
                if nb in lever_set:
                    for e in experienced:
                        triples.append((s, nb, e))
    return triples


def consolidate_triples(triples, F):
    groups = {}
    for s, sb, se in triples:
        key = (s, sb)
        if key not in groups: groups[key] = []
        groups[key].append(se)
    B = len(groups)
    phi_C = np.zeros((d, B))
    delta_Phi = np.zeros((d, B))
    for b, ((s, sb), el) in enumerate(groups.items()):
        phi_C[:, b] = F[s]
        for e in el:
            delta_Phi[:, b] += F[e] - F[sb]
    return phi_C, delta_Phi


# =========================================================================
# SINGLE-SEED PHASED TRAINING
# =========================================================================

def train_one_seed(F, T_new, ec,
                    phi_C_full, H_barrier, delta_Phi,
                    barrier_idx, neighbor_idx,
                    search_states, lever_nb_states,
                    n_search_rounds=10, n_lever_rounds=10,
                    bellman_per_round=2000,
                    bellman_lr=0.01, eq16_iters=200, eq16_lr=0.01,
                    seed=42):
    B_total = phi_C_full.shape[1]
    B_triples = delta_Phi.shape[1]
    M = np.eye(d)
    theta = np.zeros(B_triples)
    rng = np.random.RandomState(seed)

    search_Ms = [M.copy()]
    lever_Ms = []

    def build_H(theta):
        w = np.tanh(theta)
        H = np.zeros((B_total, d))
        H[barrier_idx] = H_barrier
        H[neighbor_idx] = np.diag(w) @ delta_Phi.T
        return H

    def run_round(sample_states):
        nonlocal M, theta
        for _ in range(bellman_per_round):
            s = rng.choice(sample_states)
            sp = rng.choice(N, p=T_new[s])
            f_s, f_sp = F[s], F[sp]
            res = f_s + f_sp @ M - ec * f_s @ M
            M -= bellman_lr * 2.0 * np.outer(f_sp - ec * f_s, res)
        phi_C_nb = phi_C_full[:, neighbor_idx]
        for _ in range(eq16_iters):
            w = np.tanh(theta)
            H = build_H(theta)
            res = np.eye(d) + phi_C_full @ H @ M - M
            t1 = phi_C_nb.T @ res
            t2 = (M.T @ delta_Phi).T
            grad_w = 2.0 * np.sum(t1 * t2, axis=1)
            theta -= eq16_lr * grad_w * (1.0 - w**2)
        H = build_H(theta)
        M = compute_M_from_H(H, phi_C_full, d)

    for _ in range(n_search_rounds):
        run_round(search_states)
        search_Ms.append(M.copy())

    for _ in range(n_lever_rounds):
        run_round(lever_nb_states)
        lever_Ms.append(M.copy())

    return M, search_Ms, lever_Ms


# =========================================================================
# MULTI-SEED RUNNERS
# =========================================================================

# def run_6g(F, T_open, ec, n_seeds=50, load_data=True, 
#            save_data=True, plot=True):
    
#     data_path = DATA_DIR+'peng_6g.npz'
    
#     if load_data and os.path.exists(data_path):
#         print("  6g: Loading data")
#         data = np.load(data_path)
#         s_room_thirds = data["s_room_thirds"]
#         s_lever_thirds = data["s_lever_thirds"]
#         l_room_thirds = data["l_room_thirds"]
#         l_lever_thirds = data["l_lever_thirds"]
        
#     else:
#         print("  6g: Generating data")
#         n_sr, n_lr = 10, 10
#         search_room = np.zeros((n_seeds, n_sr + 1))
#         search_lever = np.zeros((n_seeds, n_sr + 1))
#         atlev_room = np.zeros((n_seeds, n_lr))
#         atlev_lever = np.zeros((n_seeds, n_lr))

#         # Room RDM — computed once, same for all seeds
#         rdm_room = compute_rdm(F)

#         rng = np.random.RandomState(0)

#         good = 0
#         tried = 0
#         while good < n_seeds:
#             tried += 1

#             lr, lc = random_lever_position(rng, size=LEVER_SIZE)
#             lever_states = get_lever_states(lr, lc, size=LEVER_SIZE)

#             search_states_check = get_far_states(lever_states, min_distance=3)
#             lever_nb_check = get_neighbor_states(lever_states, distance=1)
#             if len(search_states_check) < MIN_SEARCH_STATES or len(lever_nb_check) < MIN_LEVER_NB_STATES:
#                 continue

#             (T_lever, M_true,
#             phi_C_full, H_barrier, delta_Phi,
#             barrier_idx, neighbor_idx,
#             search_states, lever_nb_states) = setup_lever(F, T_open, lever_states)

#             # Lever RDM — specific to this lever position
#             F_lever = F @ M_true
#             rdm_lever = compute_rdm(F_lever)

#             _, s_Ms, l_Ms = train_one_seed(
#                 F, T_lever, ec,
#                 phi_C_full, H_barrier, delta_Phi,
#                 barrier_idx, neighbor_idx,
#                 search_states, lever_nb_states,
#                 n_search_rounds=n_sr, n_lever_rounds=n_lr,
#                 seed=tried)

#             for i, M_t in enumerate(s_Ms):
#                 # Pearson
#                 # rdm_t = compute_rdm(F @ M_t)
#                 # search_room[good, i] = np.corrcoef(rdm_t, rdm_room)[0, 1]
#                 # search_lever[good, i] = np.corrcoef(rdm_t, rdm_lever)[0, 1]
                
#                 # Spearman
#                 rdm_t = compute_rdm(F @ M_t)
#                 search_room[good, i] = spearmanr(rdm_t, rdm_room)[0]
#                 search_lever[good, i] = spearmanr(rdm_t, rdm_lever)[0]

#             for i, M_t in enumerate(l_Ms):
#                 # Pearson
#                 # rdm_t = compute_rdm(F @ M_t)
#                 # atlev_room[good, i] = np.corrcoef(rdm_t, rdm_room)[0, 1]
#                 # atlev_lever[good, i] = np.corrcoef(rdm_t, rdm_lever)[0, 1]
                
#                 # Spearman
#                 rdm_t = compute_rdm(F @ M_t)
#                 atlev_room[good, i] = spearmanr(rdm_t, rdm_room)[0]
#                 atlev_lever[good, i] = spearmanr(rdm_t, rdm_lever)[0]

#             good += 1
#             if good % 10 == 0:
#                 print(f"  6g: {good}/{n_seeds} good seeds (tried {tried})")

#         # Bin into thirds
#         def thirds(data):
#             n = data.shape[1]
#             t = max(1, n // 3)
#             return [data[:, :t].mean(axis=1),
#                     data[:, t:2*t].mean(axis=1),
#                     data[:, 2*t:].mean(axis=1)]

#         s_room_thirds = thirds(search_room)
#         s_lever_thirds = thirds(search_lever)
#         l_room_thirds = thirds(atlev_room)
#         l_lever_thirds = thirds(atlev_lever)

#         print(f"\n  Search (median across seeds):")
#         for i, label in enumerate(['Beg', 'Mid', 'End']):
#             print(f"    {label}: room={np.median(s_room_thirds[i]):.4f}, "
#                 f"lever={np.median(s_lever_thirds[i]):.4f}")
#         print(f"  At lever:")
#         for i, label in enumerate(['Beg', 'Mid', 'End']):
#             print(f"    {label}: room={np.median(l_room_thirds[i]):.4f}, "
#                 f"lever={np.median(l_lever_thirds[i]):.4f}")
            
#         if save_data:
#             print("  6g: Saving data")
#             data = {"s_room_thirds": s_room_thirds, "s_lever_thirds": s_lever_thirds, 
#                     "l_room_thirds": l_room_thirds, "l_lever_thirds": l_lever_thirds}
#             np.savez(data_path, **data)

#     # Plot
#     if plot:
#         print("  6g: Plotting data")
#         plot_6g(s_room_thirds, s_lever_thirds, l_room_thirds, l_lever_thirds)

def run_6g(F, T_open, ec, n_seeds=50, load_data=True, 
           save_data=True, plot=True):
    
    data_path = DATA_DIR+'peng_6g.npz'
    
    if load_data and os.path.exists(data_path):
        print("  6g: Loading data")
        data = np.load(data_path)
        s_room_thirds  = data["s_room_thirds"]
        s_lever_thirds = data["s_lever_thirds"]
        l_room_thirds  = data["l_room_thirds"]
        l_lever_thirds = data["l_lever_thirds"]
        s_pvals        = data["s_pvals"]
        l_pvals        = data["l_pvals"]
        
    else:
        print("  6g: Generating data")
        n_sr, n_lr = 10, 10
        search_room  = np.zeros((n_seeds, n_sr + 1))
        search_lever = np.zeros((n_seeds, n_sr + 1))
        atlev_room   = np.zeros((n_seeds, n_lr))
        atlev_lever  = np.zeros((n_seeds, n_lr))

        rdm_room = compute_rdm(F)
        rng = np.random.RandomState(0)

        good = 0
        tried = 0
        while good < n_seeds:
            tried += 1

            lr, lc = random_lever_position(rng, size=LEVER_SIZE)
            lever_states = get_lever_states(lr, lc, size=LEVER_SIZE)

            search_states_check = get_far_states(lever_states, min_distance=3)
            lever_nb_check = get_neighbor_states(lever_states, distance=1)
            if len(search_states_check) < MIN_SEARCH_STATES or len(lever_nb_check) < MIN_LEVER_NB_STATES:
                continue

            (T_lever, M_true,
            phi_C_full, H_barrier, delta_Phi,
            barrier_idx, neighbor_idx,
            search_states, lever_nb_states) = setup_lever(F, T_open, lever_states)

            F_lever  = F @ M_true
            rdm_lever = compute_rdm(F_lever)

            _, s_Ms, l_Ms = train_one_seed(
                F, T_lever, ec,
                phi_C_full, H_barrier, delta_Phi,
                barrier_idx, neighbor_idx,
                search_states, lever_nb_states,
                n_search_rounds=n_sr, n_lever_rounds=n_lr,
                seed=tried)

            for i, M_t in enumerate(s_Ms):
                rdm_t = compute_rdm(F @ M_t)
                search_room[good, i]  = spearmanr(rdm_t, rdm_room)[0]
                search_lever[good, i] = spearmanr(rdm_t, rdm_lever)[0]

            for i, M_t in enumerate(l_Ms):
                rdm_t = compute_rdm(F @ M_t)
                atlev_room[good, i]  = spearmanr(rdm_t, rdm_room)[0]
                atlev_lever[good, i] = spearmanr(rdm_t, rdm_lever)[0]

            good += 1
            if good % 10 == 0:
                print(f"  6g: {good}/{n_seeds} good seeds (tried {tried})")

        def thirds(data):
            n = data.shape[1]
            t = max(1, n // 3)
            return [data[:, :t].mean(axis=1),
                    data[:, t:2*t].mean(axis=1),
                    data[:, 2*t:].mean(axis=1)]

        s_room_thirds  = thirds(search_room)
        s_lever_thirds = thirds(search_lever)
        l_room_thirds  = thirds(atlev_room)
        l_lever_thirds = thirds(atlev_lever)

        s_pvals = np.array([wilcoxon(s_lever_thirds[i], s_room_thirds[i]).pvalue for i in range(3)])
        l_pvals = np.array([wilcoxon(l_lever_thirds[i], l_room_thirds[i]).pvalue for i in range(3)])

        print(f"\n  Search (median across seeds):")
        for i, label in enumerate(['Beg', 'Mid', 'End']):
            print(f"    {label}: room={np.median(s_room_thirds[i]):.4f}, "
                  f"lever={np.median(s_lever_thirds[i]):.4f}, p={s_pvals[i]:.4f}")
        print(f"  At lever:")
        for i, label in enumerate(['Beg', 'Mid', 'End']):
            print(f"    {label}: room={np.median(l_room_thirds[i]):.4f}, "
                  f"lever={np.median(l_lever_thirds[i]):.4f}, p={l_pvals[i]:.4f}")

        if save_data:
            print("  6g: Saving data")
            np.savez(data_path,
                     s_room_thirds=s_room_thirds, s_lever_thirds=s_lever_thirds,
                     l_room_thirds=l_room_thirds, l_lever_thirds=l_lever_thirds,
                     s_pvals=s_pvals, l_pvals=l_pvals)

    if plot:
        print("  6g: Plotting data")
        plot_6g(s_room_thirds, s_lever_thirds, l_room_thirds, l_lever_thirds,
                s_pvals, l_pvals)

def run_6b(F, T_open, ec, n_seeds=50, n_eigs=None,
           load_data=True, save_data=True, plot=True):
    
    data_path = DATA_DIR+'peng_6b.npz'
    if load_data and os.path.exists(data_path):
        print("  6b: Loading data")
        data = np.load(data_path)
        seed_mvl_null = data["seed_mvl_null"]
        seed_mvl_task = data["seed_mvl_task"]
        pval = data["pval"]
        
    else:
        print("  6b: Generating data")
        if n_eigs is None:
            n_eigs = d

        rng = np.random.RandomState(100)

        seed_mvl_null = np.zeros(n_seeds)
        seed_mvl_task = np.zeros(n_seeds)
        seed_n_increased = np.zeros(n_seeds)

        good = 0
        tried = 0
        while good < n_seeds:
            tried += 1

            lr, lc = random_lever_position(rng, size=LEVER_SIZE)
            lever_states = get_lever_states(lr, lc, size=LEVER_SIZE)

            # Input-based inclusion: enough states for learning?
            search_states_check = get_far_states(lever_states, min_distance=3)
            lever_nb_check = get_neighbor_states(lever_states, distance=1)
            if len(search_states_check) < MIN_SEARCH_STATES or len(lever_nb_check) < MIN_LEVER_NB_STATES:
                print(f"    Skipped position ({lr},{lc}): "
                    f"{len(search_states_check)} search, {len(lever_nb_check)} lever-nb")
                continue

            (T_lever, M_true,
            phi_C_full, H_barrier, delta_Phi,
            barrier_idx, neighbor_idx,
            search_states, lever_nb_states) = setup_lever(F, T_open, lever_states)

            # Ring states for this lever position
            rows_l = [state_to_rc(s)[0] for s in lever_states]
            cols_l = [state_to_rc(s)[1] for s in lever_states]
            cr, cc = np.mean(rows_l), np.mean(cols_l)

            ring_states, ring_dirs = [], []
            for s in get_neighbor_states(lever_states, distance=1):
                r, c = state_to_rc(s)
                ring_states.append(s)
                ring_dirs.append(np.arctan2(r - cr, c - cc))
            ring_states = np.array(ring_states)
            ring_dirs = np.array(ring_dirs)

            # Null MVL from F
            mvl_null_per_eig = np.zeros(n_eigs)
            for k in range(n_eigs):
                vals = F[ring_states, k]
                denom = np.sum(np.abs(vals))
                if denom > 0:
                    mvl_null_per_eig[k] = np.abs(
                        np.sum(vals * np.exp(1j * ring_dirs))) / denom

            # Train
            M_learned, _, _ = train_one_seed(
                F, T_lever, ec,
                phi_C_full, H_barrier, delta_Phi,
                barrier_idx, neighbor_idx,
                search_states, lever_nb_states,
                n_search_rounds=10, n_lever_rounds=10,
                seed=tried)

            # Task MVL from F @ M_learned
            F_lever = F @ M_learned
            mvl_task_per_eig = np.zeros(n_eigs)
            for k in range(n_eigs):
                vals = F_lever[ring_states, k]
                denom = np.sum(np.abs(vals))
                if denom > 0:
                    mvl_task_per_eig[k] = np.abs(
                        np.sum(vals * np.exp(1j * ring_dirs))) / denom

            seed_mvl_null[good] = np.median(mvl_null_per_eig)
            seed_mvl_task[good] = np.median(mvl_task_per_eig)
            seed_n_increased[good] = np.sum(mvl_task_per_eig > mvl_null_per_eig)

            good += 1
            if good % 10 == 0:
                print(f"  6b: {good}/{n_seeds} good seeds (tried {tried})")

        stat, pval = wilcoxon(seed_mvl_task, seed_mvl_null, alternative='greater')
        print(f"\n  6b results ({n_seeds} seeds):")
        print(f"    Median MVL null: {np.median(seed_mvl_null):.4f}")
        print(f"    Median MVL task: {np.median(seed_mvl_task):.4f}")
        print(f"    Mean fraction increased: {np.mean(seed_n_increased)/n_eigs:.2f}")
        print(f"    Wilcoxon (task > null): stat={stat:.1f}, p={pval:.2e}")
        
        if save_data:
            print("  6b: Saving data")
            data = {"seed_mvl_null": seed_mvl_null, "seed_mvl_task": seed_mvl_task, "pval": pval}
            np.savez(data_path, **data)

    if plot:
        print("  6b: Plotting data")
        plot_6b(seed_mvl_null, seed_mvl_task, pval)


def run_4c(F, T_open, ec, n_eigs=None, load_data=True,
           save_data=True, plot=True):
    
    data_path = DATA_DIR+'peng_4c.npz'
    
    if load_data and os.path.exists(data_path):
        print("  4c: Loading data")
        data = np.load(data_path)
        corr_room = data["corr_room"]
        corr_lever = data["corr_lever"]
        r_val = data["r_val"]
        n_pairs = data["n_pairs"]
        
    else:
        print("  4c: Generating data")
        # Place lever
        if n_eigs is None:
            n_eigs = d
        lever_states = get_lever_states(5, 5, size=5)
        (T_lever, M_true,
        phi_C_full, H_barrier, delta_Phi,
        barrier_idx, neighbor_idx,
        search_states, lever_nb_states) = setup_lever(F, T_open, lever_states)

        print(f"  Lever: {len(lever_states)} states at (5,5)")
        print(f"  Using first {n_eigs} eigenvectors")

        # Train
        print(f"  Training...")
        M_learned, _, _ = train_one_seed(
            F, T_lever, ec,
            phi_C_full, H_barrier, delta_Phi,
            barrier_idx, neighbor_idx,
            search_states, lever_nb_states,
            n_search_rounds=10, n_lever_rounds=10,
            seed=42)

        F_lever = F @ M_learned
        print(f"  M error: {norm(M_learned - M_true) / norm(M_true):.4f}")

        # Pairwise correlations for first n_eigs eigenvectors
        n_pairs = n_eigs * (n_eigs - 1) // 2
        corr_room = np.zeros(n_pairs)
        corr_lever = np.zeros(n_pairs)

        idx = 0
        for i in range(n_eigs):
            for j in range(i + 1, n_eigs):
                corr_room[idx] = np.corrcoef(F[:, i]**2, F[:, j]**2)[0, 1]
                corr_lever[idx] = np.corrcoef(F_lever[:, i]**2, F_lever[:, j]**2)[0, 1]
                idx += 1

        # Overall preservation
        r_val, p_val = pearsonr(corr_room, corr_lever)
        print(f"\n  Pairs: {n_pairs}")
        print(f"  Preservation r = {r_val:.4f}, p = {p_val:.2e}")
        
        if save_data:
            print("  4c: Saving data")
            data = {"corr_room": corr_room, "corr_lever": corr_lever, 
                    "r_val": r_val, "n_pairs": n_pairs}
            np.savez(data_path, **data)
    
    if plot:
        print("  4c: Plotting data")
        plot_4c(corr_room, corr_lever, r_val, n_pairs)

# =========================================================================
# PLOTTING
# =========================================================================

def plot_4c(corr_room, corr_lever, r_val, n_pairs):
    # Plot
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(corr_room, corr_lever, alpha=0.15, s=15, c='steelblue',
               edgecolors='none')
    # ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8)
    ax.set_xlabel('RF', fontsize=20)
    ax.set_ylabel('Task', fontsize=20)
    ax.set_title(f'N = {n_pairs}\nr = {r_val:.2f} pairs', fontsize=12, loc='left')
    ax.set_xlim(-0.5, 1)
    ax.set_ylim(-0.5, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=15)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(f'{SAVE_DIR}/fig4c_pairwise.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {SAVE_DIR}/fig4c_pairwise.png")
    

boxPlotColors = ['#ffecd1', '#ff7d00', '#15616d', '#78290f']  # Lever first
GLOBALFONTSIZE = 15  # adjust to match your preference

# Their colors: Lever/Task = cream, RF = dark brown
COLOR_TASK = '#ffecd1'
COLOR_RF   = '#78290f'

def plot_6b(seed_mvl_null, seed_mvl_task, pval):
    sns.set_theme(style='ticks')
    fig = plt.figure(figsize=(3, 3), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 1, figure=fig)
    ax  = fig.add_subplot(gs[0, 0])

    bp = ax.boxplot(
        [seed_mvl_task, seed_mvl_null],
        tick_labels=['Task', 'RF'],
        widths=0.5, patch_artist=True, showfliers=False,
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5))

    for cap in bp['caps']:
        x = cap.get_xdata()
        mid = (x[0] + x[1]) / 2
        half = (x[1] - x[0]) / 2 * 0.5
        cap.set_xdata([mid - half, mid + half])

    bp['boxes'][0].set_facecolor(COLOR_TASK)
    bp['boxes'][1].set_facecolor(COLOR_RF)

    for i, (data, color) in enumerate(zip([seed_mvl_task, seed_mvl_null], [COLOR_TASK, COLOR_RF])):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=50, color=color,
                   edgecolors='gray', linewidths=0.5, zorder=3)

    def get_sig_label(p):
        if p < 0.0001: return '****'
        elif p < 0.001: return '***'
        elif p < 0.01:  return '**'
        elif p < 0.05:  return '*'
        else:           return 'ns'

    y_max = max(np.max(seed_mvl_task), np.max(seed_mvl_null))
    y_bar = y_max * 1.06
    y_tip = y_max * 1.04
    ax.plot([1, 1, 2, 2], [y_tip, y_bar, y_bar, y_tip], color='black', linewidth=1.8)
    ax.text(1.5, y_bar * 1.01, get_sig_label(pval), ha='center', va='bottom',
            fontsize=GLOBALFONTSIZE + 10)

    ax.set_ylabel('Lever ref. dir. selectivity', fontsize=15)
    ax.tick_params(axis='both', which='both', labelsize=GLOBALFONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 0.3)
    ax.set_yticks([0, 0.1, 0.2, 0.3])

    fig.savefig(f'{SAVE_DIR}/fig6b_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {SAVE_DIR}/fig6b_boxplot.png")


def plot_6g(s_room_thirds, s_lever_thirds, l_room_thirds, l_lever_thirds,
            s_pvals, l_pvals):
    sns.set_theme(style='ticks')
    fig = plt.figure(figsize=(7, 3), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.5)

    positions_lever = [1,   4,   7  ]
    positions_room  = [1.8, 4.8, 7.8]
    tick_positions  = [1.4, 4.4, 7.4]
    tick_labels     = ['Beg.', 'Middle', 'End']

    def get_sig_label(p):
        if p < 0.0001: return '****'
        elif p < 0.001: return '***'
        elif p < 0.01:  return '**'
        elif p < 0.05:  return '*'
        else:           return 'ns'

    for idx, (room_thirds, lever_thirds, title, pvals) in enumerate([
            (s_room_thirds, s_lever_thirds, 'Search',   s_pvals),
            (l_room_thirds, l_lever_thirds, 'At lever', l_pvals)]):

        ax = fig.add_subplot(gs[0, idx])

        bp_lever = ax.boxplot(
            [lever_thirds[i] for i in range(3)],
            positions=positions_lever, widths=1.0, patch_artist=True, showfliers=False,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0))

        bp_room = ax.boxplot(
            [room_thirds[i] for i in range(3)],
            positions=positions_room, widths=1.0, patch_artist=True, showfliers=False,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0))

        for bp, color in [(bp_lever, COLOR_TASK), (bp_room, COLOR_RF)]:
            for patch in bp['boxes']:
                patch.set_facecolor(color)
            for cap in bp['caps']:
                x = cap.get_xdata()
                mid = (x[0] + x[1]) / 2
                half = (x[1] - x[0]) / 2 * 0.5
                cap.set_xdata([mid - half, mid + half])

        for pos, data in zip(positions_lever, [lever_thirds[i] for i in range(3)]):
            jitter = np.random.normal(pos, 0.08, size=len(data))
            ax.scatter(jitter, data, alpha=0.6, s=20, color=COLOR_TASK,
                       edgecolors='gray', linewidths=0.5, zorder=3)

        for pos, data in zip(positions_room, [room_thirds[i] for i in range(3)]):
            jitter = np.random.normal(pos, 0.08, size=len(data))
            ax.scatter(jitter, data, alpha=0.6, s=20, color=COLOR_RF,
                       edgecolors='gray', linewidths=0.5, zorder=3)

        for i, (pl, pr, pval) in enumerate(zip(positions_lever, positions_room, pvals)):
            all_data = np.concatenate([lever_thirds[i], room_thirds[i]])
            y_max = np.max(all_data)
            y_bar = y_max * 1.08
            y_tip = y_max * 1.04
            ax.plot([pl, pl, pr, pr], [y_tip, y_bar, y_bar, y_tip],
                    color='black', linewidth=1.2)
            ax.text((pl + pr) / 2, y_bar * 1.01, get_sig_label(pval),
                    ha='center', va='bottom', fontsize=GLOBALFONTSIZE + 4)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel('Anchoring strength (RSA)' if idx == 0 else '')
        ax.set_title(title)
        ax.tick_params(axis='both', which='both', labelsize=GLOBALFONTSIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.savefig(f'{SAVE_DIR}/fig6g_spearman.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {SAVE_DIR}/fig6g_spearman.png")


# def plot_6b(seed_mvl_null, seed_mvl_task, pval):
#     fig, ax = plt.subplots(figsize=(3, 3))
#     bp = ax.boxplot(
#         [seed_mvl_task, seed_mvl_null],
#         tick_labels=['Task', 'Random Foraging'],
#         widths=0.4, patch_artist=True, showfliers=False)
#     bp['boxes'][0].set_facecolor('lightblue')
#     bp['boxes'][1].set_facecolor('lightsalmon')

#     for i, data in enumerate([seed_mvl_task, seed_mvl_null]):
#         x = np.random.normal(i + 1, 0.04, size=len(data))
#         ax.scatter(x, data, alpha=0.4, s=10, c='gray', zorder=3)

#     ax.set_ylabel('Lever ref. dir. selectivity', fontsize=20)
#     ax.tick_params(axis='both', labelsize=20)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     fig.tight_layout()
#     fig.savefig(f'{SAVE_DIR}/fig6b_boxplot.png', dpi=200, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  Saved {SAVE_DIR}/fig6b_boxplot.png")


# def plot_6g(s_room_thirds, s_lever_thirds, l_room_thirds, l_lever_thirds):
#     fig, axes = plt.subplots(1, 2, figsize=(6, 3))

#     # Lever first (left), Room second (right), touching boxes
#     positions_lever = [1, 4, 7]
#     positions_room  = [1.8, 4.8, 7.8]
#     tick_positions  = [1.4, 4.4, 7.4]  # midpoint between each pair
#     tick_labels     = ['Beg.', 'Middle', 'End']

#     # Colors to match reference
#     color_lever = '#f0e6c8'   # light cream/tan
#     color_room  = '#d4691e'   # orange

#     for ax, room_thirds, lever_thirds, title in [
#             (axes[0], s_room_thirds, s_lever_thirds, 'Search'),
#             (axes[1], l_room_thirds, l_lever_thirds, 'At lever')]:

#         bp_lever = ax.boxplot(
#             [lever_thirds[i] for i in range(3)],
#             positions=positions_lever, widths=0.8, patch_artist=True,
#             showfliers=False)

#         bp_room = ax.boxplot(
#             [room_thirds[i] for i in range(3)],
#             positions=positions_room, widths=0.8, patch_artist=True,
#             showfliers=False)

#         for patch in bp_lever['boxes']:
#             patch.set_facecolor(color_lever)
#         for patch in bp_room['boxes']:
#             patch.set_facecolor(color_room)

#         for pos, data in zip(positions_lever, [lever_thirds[i] for i in range(3)]):
#             jitter = np.random.normal(pos, 0.08, size=len(data))
#             ax.scatter(jitter, data, alpha=0.4, s=10, c='gray', zorder=3)

#         for pos, data in zip(positions_room, [room_thirds[i] for i in range(3)]):
#             jitter = np.random.normal(pos, 0.08, size=len(data))
#             ax.scatter(jitter, data, alpha=0.4, s=10, c='gray', zorder=3)

#         ax.set_xticks(tick_positions)
#         ax.set_xticklabels(tick_labels)
#         ax.set_ylim(0, 1)
#         ax.set_ylabel('Anchoring strength (RSA)')
#         ax.set_title(title)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#     fig.tight_layout()
#     fig.savefig(f'{SAVE_DIR}/fig6g_spearman.png', dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print(f"\n  Saved {SAVE_DIR}/fig6g_spearman.png")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 60)
    print("PENG et al. (2025) — Multi-seed simulation")
    print("=" * 60)

    N_SEEDS = 50

    print("\n--- Setup ---")
    T_open = create_transition_open(ROWS, COLS)
    D_open = get_complete_DR(T_open)
    lam, F = compute_eigenbasis(D_open)
    ec = 1.0 + 1.0 / lam[0]
    print(f"Grid: {ROWS}x{COLS}, d={d}, ec={ec:.4f}")

    print(f"\n--- Fig 6g ({N_SEEDS}) seeds ---")
    run_6g(F, T_open, ec, n_seeds=N_SEEDS, load_data=True, save_data=True, plot=True)

    print(f"\n--- Fig 6b ({N_SEEDS}) seeds ---")
    # run_6b(F, T_open, ec, n_seeds=N_SEEDS, load_data=True, save_data=True, plot=True)
    
    print(f"\n--- Fig 4c ---")
    # run_4c(F, T_open, ec, n_eigs=d, load_data=True, save_data=True, plot=True)


if __name__ == '__main__':
    main()