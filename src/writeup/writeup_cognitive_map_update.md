# Updating Cognitive Maps from Local Experience: Implementation Write-Up

## Overview

This document walks through the complete implementation of the cognitive map update framework described in Piray (2026), "Some notes on updating cognitive maps from local experience." It traces each equation in the notes to its corresponding code, explains what we discovered during implementation, and describes the final working algorithm.

---

## 1. The Problem

An agent has learned a cognitive map — the Default Representation (DR) matrix D, of size N×N — encoding multi-step reachability between every pair of states in its environment. A barrier appears, changing some transitions. The agent must update its map efficiently.

**Environment:** 20×20 grid (N = 400 states). A horizontal barrier blocks 10 states, affecting transitions at 22 neighboring states.

**Pre-learned representations:**
- D_os: the N×N DR matrix for the open (pre-barrier) environment
- U_os: the absorbed eigenbasis, where D_os ≈ U_os @ U_os^T

**Code (Setup):**
```python
T_os = create_transition_open(shape=(ROWS, COLS))
D_os = get_complete_DR(T=T_os)

eigenvalues, eigvecs = np.linalg.eigh(D_os)
idx = np.argsort(eigenvalues)[::-1][:d]
V = eigvecs[:, idx]
lam = eigenvalues[idx]
U_os = V * np.sqrt(lam)[np.newaxis, :]  # F = U @ Lambda^0.5
```

**Notation mapping:** In the notes, F = U Λ^0.5 is the absorbed eigenbasis. In code, this is `U_os`. Since D is symmetric, F = U in the notes' terminology (page 2), so D = F F^T = U_os @ U_os^T. The raw eigenvectors V in code correspond to U in the notes' eigendecomposition D = U Λ U^{-1}.

---

## 2. The Woodbury Correction (Notes Eq. 1)

**Notes equation:**
```
D_new = D + D C A R D
```
where C ∈ R^{N×B} selects changed states, R ∈ R^{B×N} encodes transition changes, and A = (I_B - R D C)^{-1}.

**Code:**
```python
def woodbury(T_old, D, T_new):
    diffs = T_old != T_new
    diff_rows = np.unique(np.where(diffs)[0])
    R = (T_new - T_old)[diff_rows, :]         # B × N
    C = np.eye(N)[:, diff_rows]               # N × B
    Z = R @ D @ C                              # B × B
    A = np.linalg.inv(np.eye(Z.shape[0]) - Z) # B × B
    D_new = D + D @ C @ A @ R @ D
    return D_new, R, C, diff_rows
```

**What this does:** Exact correction to D using the known transition changes. Requires both T_old and T_new (model-based). This is our ground truth for verification — it is NOT used in the final T-free algorithm.

---

## 3. Feature-Space Projection (Notes Eqs. 8–9)

**Notes equations:**
```
G = U^T C ∈ R^{K×B}       (Eq. 8)
H = R F ∈ R^{B×K}         (Eq. 8)
A = (I_B - H G)^{-1}      (Eq. 9)
M = I_K + G A H            (Eq. 9)
```
where F_new = F M, meaning the entire N×N map update reduces to a K×K operator M applied uniformly to every feature vector.

**Code notation mapping:**

| Notes | Code | Shape | Description |
|-------|------|-------|-------------|
| G | phi_C | (d, B) | U_os^T @ C — feature projection of changed states |
| H | psi_R | (B, d) | R @ U_os — feature projection of transition changes |
| F | U_os | (N, d) | Absorbed eigenbasis |
| K | d | scalar | Feature dimension (= N = 400 at full rank) |
| M | M | (d, d) | Feature-space update operator |

**Code:**
```python
phi_C = U_os.T @ C              # G in the notes
psi_R = R @ U_os                # H in the notes

def compute_M_from_H(H, G, d):
    B = H.shape[0]
    Z = H @ G                           # B × B
    A = inv(np.eye(B) - Z)              # B × B
    M = np.eye(d) + G @ A @ H           # d × d
    return M
```

**Key insight:** The full N×N map update collapses to a d×d operator. Planning never reconstructs D_new; it computes exp_v = U_os @ M @ U_os^T @ t entirely in feature space.

---

## 4. Decomposed Parameterization (Notes Eq. 2)

**Notes equation:** For a state s where the transition to s'_1 is blocked, with mass redistributed to successors s'_2, ..., s'_J:

```
C = [e_s, ..., e_s] ∈ R^{N×(J-1)}
R = [ w_2(e_{s'_2} - e_{s'_1})^T ]
    [ ...                          ]    ∈ R^{(J-1)×N}
    [ w_J(e_{s'_J} - e_{s'_1})^T ]
```

**How this appears in our implementation:**

Each changed neighbor state s adjacent to barrier state b generates triples (s, b, s'_experienced) through exploration. Each triple is a component in the decomposed form:
- e_s is the indicator for the changed state → phi_C column = u(s)
- e_{s'_exp} - e_{s'_blocked} is the direction of change → delta_Phi column = u(s'_exp) - u(s'_blocked)
- w is the unknown weight (probability mass redirected)

**Code (exploration):**
```python
def explore_barrier(T_new, barrier_states, grid_shape, n_samples_per_state=20):
    # For each accessible state adjacent to a barrier:
    #   - Identify blocked successors (barrier states)
    #   - Sample transitions to find experienced successors
    #   - Create (s, s_blocked, s_experienced) triples
    ...
    return triples
```

**Code (feature projections from triples):**
```python
def build_triple_projections(triples, U_os):
    for b, (s, s_blocked, s_exp) in enumerate(triples):
        phi_C[:, b] = U_os[s, :]                              # G column = u(s)
        delta_Phi[:, b] = U_os[s_exp, :] - U_os[s_blocked, :] # feature contrast
    return phi_C, delta_Phi
```

**Consolidation:** Multiple triples from the same (s, s_blocked) pair share a weight. We consolidate by summing feature contrasts within each group, reducing B from 66 triples to 22 groups.

```python
def consolidate_triples(triples, U_os):
    # Group by (s, s_blocked)
    # Sum feature contrasts within each group
    # One weight per group
    ...
```

**This is exactly the equivalence proven in Notes Eqs. 5–7:** the decomposed form CR = C_compact R_compact. Summing the component contrasts within a group recovers the compact single-row form.

---

## 5. Experience-Based H (Notes Eqs. 10–11)

**Notes equations:**
```
H[b, :] = w_{s_b} (f(s'_experienced) - f(s'_blocked))^T    (Eq. 10)
H(W) = W ΔΦ^T,  W = diag(w_{s_1}, ..., w_{s_B})            (Eq. 11)
```

**Code:**
```python
def build_H_from_theta(theta, delta_Phi):
    w = np.tanh(theta)                      # W = diag(tanh(θ))
    return np.diag(w) @ delta_Phi.T         # H = W @ ΔΦ^T
```

**What the agent knows vs. learns:**

| Quantity | Known/Learned | Source |
|----------|--------------|--------|
| phi_C (= G) | Known | u(s) looked up from U_os |
| delta_Phi (= ΔΦ) | Known | u(s'_exp) - u(s'_blocked) from exploration |
| W | **Learned** | Scalar weights, learned via Eq 16 |

---

## 6. Barrier Anchor Rows (Not in the notes — our addition)

Completely inaccessible barrier states have T_new[b,:] = 0, so their full transition was removed. In a uniform random walk on a grid, the old transitions were uniform across neighbors:

```
H_barrier[b,:] = -(1/n_neighbors) * Σ u(neighbor)
```

This requires only grid topology and U_os — no T_old.

**Code:**
```python
def compute_barrier_H(barrier_states, U_os, grid_shape):
    for i, b in enumerate(barrier_states):
        neighbors = [grid neighbors of b]
        H_barrier[i, :] = -(1.0 / len(neighbors)) * np.sum(U_os[neighbors, :], axis=0)
        phi_C_barrier[:, i] = U_os[b, :]
    return H_barrier, phi_C_barrier
```

**Why this matters:** These fixed rows provide a strong anchor for the learning process. They encode "these states are completely gone" — the dominant effect of the barrier. Without them, the Eq 16 optimization struggles because the noisy Bellman M doesn't contain enough signal for the subtle neighbor-state redistribution effects alone.

---

## 7. The Self-Consistency Loss (Notes Eqs. 14 and 16)

**Notes Eq. 14:**
```
L(W) = || I_K + G H(W) M(W) - M(W) ||^2
```

**Notes Eq. 16 (target-separated):**
```
L(θ; θ̄) = || I_K + G H(W(θ)) M(W̄(θ̄)) - M(W̄(θ̄)) ||^2
```

**What we discovered:** By Proposition 1, M(W) = (I - GH(W))^{-1} satisfies M = I + GHM as a mathematical identity for ANY W. This means L(W) in Eq. 14 is exactly zero for every W. Target separation (Eq. 16) makes the loss temporarily nonzero, but the gradient always pulls θ back toward θ̄ (where the loss is zero for that M̄), not toward the true W.

**Empirical confirmation:**
- Zero initialization: weights stay at 0, loss = 0
- Random initialization: weights stay at initial values, loss = 0 after each target sync
- The loss measures self-consistency, not correctness

---

## 8. The Working Algorithm: Bellman + Eq 16 + Woodbury Rebuild

The fix: instead of computing M̄ from W̄ (which makes the loss trivially zero), compute M from an external source — the Bellman loss on sampled transitions.

**Bellman loss (suggested by advisor):**
```
L = || f(s) + f(s') M - e^c f(s) M ||^2
```
For a sampled transition s → s', this measures whether M satisfies the DR's defining equation D_new = (e^c I - T_new)^{-1} in feature space.

**The three-stage algorithm:**

**Stage 1 — Bellman SGD:** Sample transitions from T_new (the agent simply walks around). Update M via SGD on the Bellman loss. This gives a rough M (M_err ≈ 0.63) that captures some structure of the new environment.

```python
# Stage 1: Bellman
for t in range(bellman_samples_per_round):
    s = rng.choice(valid_states)
    s_prime = rng.choice(N, p=T_new[s, :])
    f_s = U_os[s, :]
    f_sp = U_os[s_prime, :]
    residual = f_s + f_sp @ M - ec * f_s @ M
    grad_M = 2.0 * np.outer(f_sp - ec * f_s, residual)
    M -= bellman_lr * grad_M
```

**Stage 2 — Eq 16 with fixed Bellman M:** Fix M from Stage 1 (NOT derived from W). Optimize θ to minimize ||I + G H(W(θ)) M - M||^2. Because M is external, the loss is genuinely informative — different W values produce different losses. The barrier anchor rows provide strong fixed signal.

```python
# Stage 2: Eq 16 with externally fixed M
for it in range(eq16_iters):
    w = np.tanh(theta)
    H = build_H_full(theta)  # barrier rows fixed, triple rows from W
    residual = np.eye(d) + phi_C_full @ H @ M - M
    loss = np.sum(residual ** 2)
    # Gradient only flows through learnable triple weights
    grad_w = 2.0 * np.sum((phi_C_nb.T @ residual) * (M.T @ delta_Phi).T, axis=1)
    grad_theta = grad_w * (1.0 - w ** 2)
    theta -= eq16_lr * grad_theta
```

**Stage 3 — Woodbury rebuild:** Compute the exact M from the learned W via the Woodbury formula (Eq. 9). This projects the noisy d×d Bellman estimate onto the low-rank correction manifold parameterized by B scalars — a massive denoising step.

```python
# Stage 3: Woodbury rebuild
H = build_H_full(theta)
M = compute_M_from_H(H, phi_C_full, d)  # Eq. 9
```

Repeat stages 1–3. Each round, the Bellman step provides a fresh rough M, Eq 16 extracts the structural correction, and the rebuild produces a clean M.

---

## 9. Planning in Feature Space

**How M is used:** The corrected DR is D_new ≈ U_os M U_os^T. For planning toward a goal state g, the agent computes expected values:

```
exp_v = U_os @ M @ U_os^T @ t
```

where t = T_new[nonterminal, goal]. This decomposes as:
1. w_base = U_os^T @ t (project goal transitions into feature space)
2. w_total = M @ w_base (apply correction)
3. exp_v = U_os @ w_total (project back to state space)

**Code:**
```python
def plan_with_M(U_os, M, T_new, goal, start):
    t_vec = T_new[np.ix_(nt_idx, [goal])].flatten()
    U_nt = U_os[nt_idx]
    exp_v = np.maximum(U_nt @ M @ U_nt.T @ t_vec, 1e-10)
    v = np.log(exp_v)
    # Greedy rollout using v as state values
    ...
```

No N×N matrix is ever reconstructed.

---

## 10. Stability (Notes Eq. 15) and Tanh Parameterization (Notes Page 5)

**Notes Eq. 15:** The correction is stable when:
```
min_b c_{s_b} > log(1 + 2 max_b |w_{s_b}|)
```

**Notes page 5:** To enforce |w_b| < 1, parameterize W via:
```
W(θ) = diag(tanh(θ_1), ..., tanh(θ_B))
```

**Code:** `w = np.tanh(theta)` enforces boundedness. The chain rule `(1 - tanh²(θ))` appears in the gradient.

---

## 11. Summary: What Information the Agent Uses

| Information | Used? | Source |
|-------------|-------|--------|
| U_os (absorbed eigenbasis) | Yes | Pre-learned from old environment |
| Eigenvalues of D (for e^c) | Yes | Stored from eigendecomposition |
| Grid topology (for barrier H) | Yes | Agent knows its neighborhood |
| T_new (sampled transitions) | Yes | Agent lives in this environment |
| Barrier states | Yes | Discovered through exploration |
| T_old | **No** | Not needed |
| R (transition differences) | **No** | Not needed |

---

## 12. Results

**Barrier-crossing paths (200 random trials, start and goal on opposite sides of barrier):**

| Method | Mean Steps | Median | Success Rate | Samples Used |
|--------|-----------|--------|-------------|--------------|
| Ground truth (exact Woodbury) | 18.3 | 18.0 | 100% | — |
| Combined Bellman + Eq 16 (T-free) | 18.3 | 18.0 | 100% | 10,000 |
| Prediction error (T_old-based) | 18.5 | 18.0 | 100% | 22 |
| Pure Bellman (best) | 23.2 | 19.0 | 99% | 500,000 |
| No repair | 184.5 | 23.0 | 66% | — |

The combined approach matches ground truth planning performance from round 1, using 10,000 Bellman samples plus the Woodbury structure. The barrier anchor rows and the Eq 16 → rebuild cycle are critical for this efficiency.
