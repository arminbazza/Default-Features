# Wernle et al. (2018) Simulation: Our Approach vs. Advisor's Approach

## 1. Background: What Wernle et al. (2018) Found

Rats trained in two separate compartments (A and B) develop independent grid maps in each. When the wall between compartments is removed, the grid maps gradually merge. The key finding: **grid fields near the former partition relocate significantly, while fields far from the wall (near distal walls) remain anchored**. This produces a spatial gradient of map similarity — high correlation at the edges, low correlation at the center.

---

## 2. The Advisor's Approach (MATLAB)

The advisor's simulation (`wernle_make.m` + `gen_base.m`) models this as follows:

### 2.1 Grid Cell Construction (Dordek et al., 2016)

Grid cells are the learned weights of a biologically inspired network, NOT raw PCA eigenvectors. The procedure (implemented in `gen_base.m` + `PCANetworkFunc.m`):

1. **Place cells**: Uniformly distributed 2D Gaussians on a regular grid.
   - 30 x 30 = 900 place cells per compartment
   - Centers: `(mod(0:N-1, N) + 0.2) * max / N` (evenly spaced, slightly offset)
   - Environment: 10 x 10 continuous space with periodic boundaries
   - Place cell width: sigma = 0.35

2. **Random walk**: An agent performs a random walk (500K steps) through the environment.
   - Velocity: `2.15 * maxx / 75`
   - Angular velocity: 0.2 rad/step (smooth, correlated walk)
   - Periodic boundary conditions

3. **Network learning (Oja's rule)**: 50 grid cell outputs learn from 900 place cell inputs.
   - Weight update: `deltaJ = psi' * r - diag(psi' * psi) * J`
   - **Non-negativity constraint**: `J(J < 0) = 0` applied after every step
   - Output with adaptation: `h_adpt = h - h_avg` (running average subtraction provides the zero-mean signal needed for PCA-like convergence)
   - Learning rate decays: `epsilon = 1 / (t * delta + epsilon_base)`

4. **Result**: J is an (N_place_cells x 50) matrix of non-negative weights. Each column is a "grid cell" — its spatial pattern resembles a hexagonal grid when plotted over place cell locations. The covariance matrix Cov = (1/L) * r_saved * r_saved' is also computed from the temporal activity.

### 2.2 Two-Compartment Setup

Each compartment gets an **independent** gen_base run (different random seed):
- Bottom half (seed=0): run gen_base on full 30x30, restrict to place cells with y < 5
- Top half (seed=1 or 50-seed): run gen_base on full 30x30, restrict to place cells with y > 5

Combined representations:
- `J = [J_bottom; J_top]` — stacked grid cell weights
- `D0 = blkdiag(Cov_bottom, Cov_top)` — block-diagonal covariance (no cross-half correlations, because each half was learned independently)

### 2.3 Transition Matrices in Place Cell Space

The advisor uses `get_xy2P.m` to build transition matrices directly from place cell positions:
- For each place cell, find its nearest neighbors (by Euclidean distance between centers)
- Assign equal transition probability to all nearest neighbors
- This gives a transition matrix P in **place cell space** (N_total x N_total), not grid-state space

Two transition matrices:
- `P0 = blkdiag(P_bottom, P_top)` — no cross-compartment transitions (wall present)
- `P = get_xy2P([xy_bottom, xy_top])` — includes cross-compartment transitions (wall removed)

### 2.4 Woodbury Correction

The core update uses the Woodbury matrix identity to compute how grid cell patterns change when the wall is removed:

```
R = P0 - P                          (transition change: old minus new)
barrier_index = rows where |R| > 0  (wall-adjacent place cells)
r = R(barrier_index, :)
c = I(:, barrier_index)

Y = -r * D0 * c
A = inv(I - Y)
U = J - D0 * c * A * r * J          (updated grid cell weights)
```

**J** = grid cells with wall (divided), **U** = grid cells after wall removal (merged).

### 2.5 Analysis: Sliding Correlation

The advisor's `wernle_measures` computes local **Spearman correlation** between J and U at each spatial location:
- For each place cell location i, find all place cells within a box of size 6
- Compute rank correlation between J(local, k) and U(local, k) for each grid cell k
- Average across grid cells to get a single correlation value per location

**Expected result**: High correlation at distal walls (fields unchanged), low correlation near the former partition (fields relocated).

---

## 3. Our Approach (Python — `sim_wernle.py`)

Our simulation has the same goal but takes a different route to the updated representation U. The key difference: **we learn the DR update using the compositional maps algorithm**, rather than computing it analytically from the full transition matrices.

### 3.1 Grid Cell Construction — Current Status

We have implemented `gen_base()` in Python to match the advisor's Dordek et al. procedure exactly:
- Same place cell grid layout, sigma, and environment size
- Same random walk with periodic boundaries
- Same Oja rule with non-negativity and adaptation
- Same 500K simulation steps

**Issue**: The pure-Python Oja loop is slow (~23 minutes per compartment at 500K steps). We are exploring:
- Vectorized/compiled alternatives (Numba, C extensions)
- Analytical covariance + non-negative eigenvector methods (faster but produces different-looking rate maps)

The `create_env()` and `build_place_cell_grid_cells()` functions mirror the advisor's two-compartment setup identically.

### 3.2 The DR and Eigenbasis — Where We Diverge

In addition to the place-cell-based grid cells (J, D0, U), our simulation also computes the Default Representation in **grid-state space** (a 20x20 discrete grid, N=400 states):

```python
T_merged = create_transition_open((20, 20))       # open field
T_divided = create_transition_divided((20, 20))    # with wall
D_divided = get_complete_DR(T_divided)             # DR with wall
D_merged = get_complete_DR(T_merged)               # DR without wall (ground truth)
```

The DR is D = inv(e^c * I - T), where T is a uniform random walk transition matrix on the grid. We compute an eigenbasis U_os from D_divided and define the feature-space update operator M such that D_merged ~ U_os @ M @ U_os^T.

### 3.3 Learning M — The Compositional Maps Algorithm

This is the core contribution that differs from the advisor. Instead of requiring both T_old and T_new to compute the Woodbury correction analytically, **the agent learns M from local experience**:

**Stage 1 — Bellman SGD**: The agent walks around the merged environment, sampling transitions s -> s'. It updates M via stochastic gradient descent on the Bellman residual:
```
L = || f(s) + f(s') M - e^c f(s) M ||^2
```
This gives a rough estimate of M (error ~ 0.63).

**Stage 2 — Eq 16 (self-consistency)**: Using the rough M from Stage 1, optimize scalar weights theta that parameterize the rank-B correction:
```
H(W) = diag(tanh(theta)) @ delta_Phi^T
L = || I + G @ H(W) @ M - M ||^2
```
The weights W encode how much transition probability mass shifted at each wall-adjacent state. G and delta_Phi come from exploration triples (s, s_lost, s_gained) discovered when the agent first crosses the former wall location.

**Stage 3 — Woodbury rebuild**: Compute the exact M from the learned weights via the Woodbury formula:
```
M = I + G @ inv(I - H @ G) @ H
```
This projects the noisy Bellman estimate onto the low-rank correction manifold — a denoising step.

Stages 1-3 repeat for 5 rounds. The final M captures the transition structure change to high accuracy.

### 3.4 Connecting Learned M to Grid Cells

Once M is learned, the updated DR in state space is:
```
D_learned = U_os @ M @ U_os.T
```

For the place-cell-based analysis (matching the advisor), we also apply the Woodbury correction directly on D0 (the place cell covariance):
```
R_full = T_divided - T_merged
r = R_full[barrier_idx, :]
c = I[:, barrier_idx]
Y = -r @ D0 @ c
A = inv(I - Y)
U_updated = J - D0 @ c @ A @ r @ J
```

The analysis then uses `wernle_measures()` to compute sliding Spearman correlations between J and U_updated, matching the advisor's MATLAB implementation.

---

## 4. Key Differences Summary

| Aspect | Advisor (MATLAB) | Ours (Python) |
|--------|------------------|---------------|
| **Grid cells** | Oja-learned network weights (J) from Dordek et al. | Same procedure (gen_base), currently slow |
| **Transition matrices** | In place cell space via get_xy2P (nearest-neighbor) | In grid-state space (uniform random walk on 20x20) + place cell space |
| **DR / covariance** | D0 = place cell covariance (empirical) | D = inv(e^c I - T) in state space; D0 = Cov in place cell space |
| **Update mechanism** | Analytical Woodbury (requires P0 and P) | **Learned** via Bellman + Eq16 + Woodbury rebuild (T-free for M) |
| **What agent needs** | Full knowledge of both transition matrices | Only sampled transitions from new environment + exploration |
| **Wernle analysis** | Local Spearman correlation (wernle_measures) | Same |
| **Wall direction** | Wall removal (P0 = divided -> P = merged) | Same |

### What we share:
- Same grid cell construction (Dordek et al. 2016, Oja + non-negativity)
- Same two-compartment setup (independent seeds, restrict by y-coordinate)
- Same Woodbury formula for the final U = J - D0 * c * A * r * J
- Same sliding Spearman correlation analysis
- Same expected result: correlation gradient with low at center, high at edges

### What differs:
- **We add a learning-based pathway**: the comp_maps algorithm learns the update operator M from experience, without requiring the old transition matrix T_old. This is the theoretical contribution — showing that local experience suffices to update the cognitive map.
- **We validate in state space**: the 20x20 discrete grid lets us compute exact ground truth (D_merged, M_true) for verification. The advisor works entirely in place cell space.
- **We use two representations in parallel**: (1) the abstract state-space DR for algorithmic validation, and (2) the biologically grounded place-cell-based representation for the Wernle analysis.

---

## 5. Current Results

### Comp Maps Algorithm (State Space)
- Woodbury reconstruction error: ~0 (exact)
- M learning converges over 5 rounds of Bellman + Eq16
- Final M error vs ground truth: decreasing across rounds
- D reconstruction error: low

### Wernle Analysis (Place Cell Space)
- Correlation at center (former wall): ~0.76
- Correlation at distal walls: ~0.94
- **Wernle effect (distal - center): ~0.18** — positive, matching the qualitative finding
- Rate maps show distinct patterns in divided vs merged conditions
- Correlation heatmap shows the expected spatial gradient (red at edges, cooler at center)

### Parameters
- Grid: 20 x 20 (state space) / 30 x 30 place cells (per compartment)
- Place cell sigma: 0.35 (matching paper)
- Box size for sliding correlation: 6 (matching paper)
- 50 grid cells per compartment
- Oja simulation: 500K steps per compartment

---

## 6. Open Issues

1. **Oja simulation speed**: Pure Python loop takes ~23 minutes per compartment. Need compiled acceleration or an analytical shortcut that preserves grid-like patterns.

2. **Rate map quality**: The non-negative Oja-learned weights should produce hexagonal grid patterns when properly converged. With the analytical covariance + non-negative power iteration shortcut, we get single-field bumps instead of grids. The full Oja simulation (matching the advisor exactly) is needed for proper rate maps.

3. **State-space to place-cell-space mapping**: The comp_maps algorithm learns M in state space (400x400). Connecting this learned M to the place-cell-space Woodbury (which operates on D0 = Cov) requires bridging the two representations. Currently these are run independently — the comp_maps validates the learning algorithm, while the place-cell Woodbury produces the Wernle analysis.
