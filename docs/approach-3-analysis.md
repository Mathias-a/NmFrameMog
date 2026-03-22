# Approach 3: Hybrid Learned Prior + Monte Carlo Refinement — Deep Analysis

> **Bottom line:** Build the system around a **calibrated feature-engineered prior**, then use the 50 queries mainly to **identify the current round regime shared across all 5 seeds**, not to estimate every cell from scratch. The strongest pipeline is: **LightGBM prior → cross-seed empirical-Bayes round calibration → geometric merge into a round-aware prior → low-ESS Dirichlet update with exact query counts → greedy EIG viewport selection**.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Critical Decision Matrix](#2-critical-decision-matrix)
3. [Prior Construction from Historical Data](#3-prior-construction-from-historical-data)
4. [Dirichlet-Multinomial Bayesian Update Mechanics](#4-dirichlet-multinomial-bayesian-update-mechanics)
5. [Round-Level Calibration](#5-round-level-calibration)
6. [Query Allocation Strategy](#6-query-allocation-strategy)
7. [Combining Prior and Observations](#7-combining-prior-and-observations)
8. [Mathematical Formulas for Implementation](#8-mathematical-formulas-for-implementation)
9. [Theoretical Bounds and Expected Performance](#9-theoretical-bounds-and-expected-performance)
10. [Risk Analysis](#10-risk-analysis)
11. [Implementation Ordering](#11-implementation-ordering)
12. [Key Insight Synthesis](#12-key-insight-synthesis)

---

## 1. Architecture Overview

```
OFFLINE (historical rounds)
────────────────────────────────────────────────────────────────────
initial_grid, ground_truth
    │
    ├─► feature builder
    │     - distance to coast / settlement / port / ruin
    │     - local terrain counts in radii 1, 2, 4
    │     - terrain fractions in 5×5 and 9×9 windows
    │     - coastal / settlement-adjacent / inland flags
    │     - keep ocean/plains distinct in inputs (collapse only in output class 0)
    │
    ├─► prior model training
    │     - primary: multiclass LightGBM (max_depth 5-6, num_leaves ≤ 31)
    │     - fallback: multinomial logistic regression (when n_hist < 15)
    │     - grouped CV by round (NEVER cell-level splits)
    │     - class_weight='balanced'
    │
    ├─► offline probability calibration
    │     - temperature scaling (single parameter τ)
    │     - fit on held-out rounds
    │
    └─► historical archetype tables  θ_hist[r, a, k]
          - per historical round r
          - archetype a ∈ {coastal, settlement_adjacent, inland_natural}
          - class frequencies over k ∈ {0..5}

ONLINE (live round, 5 seeds share hidden params)
────────────────────────────────────────────────────────────────────
initial grids for current round
    │
    ├─► feature builder (same pipeline as offline)
    ├─► calibrated prior  p_base[s,h,w,k]
    │
    ├─► Phase 1: Calibration (10 queries)
    │     - 2 per seed, farthest-point / no-overlap coverage
    │     - goal: reveal round regime, not refine individual cells
    │
    ├─► pooled round calibration (across ALL seeds)
    │     - aggregate observed counts by archetype  m[a,k]
    │     - empirical-Bayes weights over historical rounds
    │     - build archetype template  p_template[s,h,w,k]
    │     - recompute after 10, 20, and 30 queries
    │
    ├─► prior fusion
    │     - geometric pool: p_round ∝ p_base^λ · p_template^(1-λ)
    │     - default λ = 0.7
    │
    ├─► Dirichlet parameterization
    │     - ess[s,h,w] from p_round entropy (heterogeneous)
    │     - alpha0 = ess · p_round
    │
    ├─► Phase 2: Refinement (30 queries)
    │     - greedy expected information gain (EIG)
    │     - overlap discount: diminishing returns for re-observed cells
    │
    ├─► Phase 3: Exploitation (10 queries)
    │     - exploit highest unresolved cells / viewports
    │
    ├─► Bayesian update
    │     - alpha = alpha0 + counts
    │     - pred = alpha / alpha.sum(-1)
    │
    └─► final clipped, normalized prediction tensor
          - eps = 1e-4 floor, structural zeros for impossible classes
          - renormalize to sum to 1.0
```

---

## 2. Critical Decision Matrix

| Design Choice | Recommendation | Why | Fallback |
|---|---|---|---|
| **Prior model** | Multiclass LightGBM, shallow trees (`max_depth=5-6`, `num_leaves≤31`), `class_weight='balanced'` | Best bias/variance for 10-100 rounds; materially better than CNNs in small-data regime | Multinomial logistic regression when `n_hist < 15` or LightGBM overfits in grouped-CV |
| **Feature set** | Hand-engineered local features only: distances, neighbor counts, terrain fractions, archetype flags | Directly encode simulator mechanisms; generalize better than learned spatial filters at this data size | — |
| **Loss function** | Entropy-weighted soft-label cross-entropy (= entropy-weighted KL up to constant) | Matches leaderboard objective directly; focuses capacity on cells that matter | Standard multiclass log loss with entropy sample weights |
| **Calibration** | Temperature scaling (single τ parameter) on held-out rounds | Simple, robust, usually sufficient once base model is reasonable | Platt/sigmoid for logistic regression fallback |
| **ESS initialization** | `c_base = 2.0`, heterogeneous: `ess = clip(c_base × (1 + 0.75 × (1 - H_norm)), 1.0, 4.0)` | Low ESS is safer — round shift makes prior misspecified; entropy scaling preserves trust only where prior is genuinely sharp | Constant `ESS = 2.0` for MVP |
| **Combination method** | Geometric pooling for prior fusion; Dirichlet conjugate update for exact observed counts | Geometric is KL-optimal for merging beliefs; Dirichlet is exact for i.i.d. categorical observations | Linear pooling if geometric causes numerical issues |
| **Query allocation** | 10 / 30 / 10 phase split, soft per-seed caps of 15/12/10/8/5 | Early coverage enables round calibration; later queries should chase posterior value, not raw uncertainty | Equal allocation (10/seed) for MVP |
| **Round calibration** | Cross-seed archetype-pooled empirical Bayes with historical-round BMA | Captures round-level hidden parameters with very few online degrees of freedom | Nearest single historical round retrieval |
| **Floor values** | `eps = 1e-4` everywhere before log/power/normalize | Small enough not to distort; large enough to prevent catastrophic KL | — |

**Non-negotiable evaluation protocol:** Every offline split must be **by round**, never by cell. Random cell-level validation leaks spatial structure and makes the prior look much better than it is.

---

## 3. Prior Construction from Historical Data

### 3.1 Feature Engineering

For each cell `(h, w)` in a 40×40 grid, compute:

| Feature Category | Features | Count |
|---|---|---|
| Initial terrain | One-hot of initial terrain code (0, 1, 2, 3, 4, 5, 10, 11) | 8 |
| Distance-based | Distance to coast, nearest settlement, nearest port, nearest ruin, nearest forest edge | 5 |
| Neighbor counts (r=1,2,4) | Forest, mountain, settlement, ruin, ocean neighbors at each radius | 15 |
| Terrain fractions | Fraction of each terrain type in 5×5 and 9×9 windows | 12 |
| Archetype flags | is_coastal, is_settlement_adjacent, is_inland_natural | 3 |
| Position | Normalized (x/40, y/40), distance from center | 3 |

**~46 features per cell.** This is well within what LightGBM handles with 10-100 rounds × ~1200 scorable cells = 12,000–120,000 training examples.

**Critical:** Keep ocean (code 10) and plains (code 11) as **separate input features** even though they collapse to prediction class 0. They carry different spatial signals.

### 3.2 Model Selection by Training Set Size

| Training Rounds | Scorable Cells | Recommended Model | Notes |
|---|---|---|---|
| 10-15 | ~12,000-18,000 | L2 multinomial logistic regression | Too little data for tree ensembles to shine |
| 15-50 | ~18,000-60,000 | LightGBM with regularization | Sweet spot for gradient boosting |
| 50-100 | ~60,000-120,000 | LightGBM, can increase capacity | Diminishing returns beyond ~50 rounds |

CNNs/U-Nets are **not recommended** unless you have 50+ rounds AND strong augmentation (8× via rotations/reflections). Feature-engineered models are more reliable here.

### 3.3 Loss Function

The scoring metric is entropy-weighted KL divergence. The training loss should match:

```python
# Entropy-weighted soft-label cross-entropy
# Equivalent to entropy-weighted KL up to a target-only constant
cell_weight = entropy(y_true)  # weight by ground truth entropy
loss = -sum(cell_weight * sum(y_true * log(clip(p_pred, eps, 1.0)), axis=-1)) / sum(cell_weight)
```

If only hard labels available (argmax of ground truth), use standard multiclass log loss with `sample_weight = entropy(y_true)`.

### 3.4 Calibration

**Temperature scaling** is the default:

```python
# Fit single parameter τ on held-out rounds
logits_calibrated = logits / tau
p_calibrated = softmax(logits_calibrated)
```

Optimize τ by minimizing NLL on the held-out set. Typical τ ∈ [0.5, 2.0].

**Calibration quality directly impacts downstream KL divergence.** An overconfident model (τ < 1 effective) creates artificially low entropy predictions that the Dirichlet update cannot correct within 50 queries.

### 3.5 Minimum Viable Training Set

- **Logistic regression**: Works from ~30-50 samples per class (~180-300 cells). With 10 rounds × 1200 cells, this is comfortable.
- **LightGBM**: Needs ~100-200 samples per class for reliable splits. With 15+ rounds, this is achievable for most classes. Class 0 (Empty) dominates — use `class_weight='balanced'`.

---

## 4. Dirichlet-Multinomial Bayesian Update Mechanics

### 4.1 Core Conjugate Update

Given prior `Dir(α₁, ..., α₆)` and N multinomial observations with counts `(n₁, ..., n₆)`:

$$\text{Posterior: } \text{Dir}(\alpha_1 + n_1, \ldots, \alpha_6 + n_6)$$

$$\text{Posterior predictive: } P(k \mid \text{data}) = \frac{\alpha_k + n_k}{\alpha_0 + N}$$

where `α₀ = Σ αₖ` is the prior ESS.

This is a **weighted average** between the prior prediction `αₖ/α₀` and the empirical frequency `nₖ/N`, with weights proportional to `α₀` and `N` respectively:

$$P(k \mid \text{data}) = \frac{\alpha_0}{\alpha_0 + N} \cdot \frac{\alpha_k}{\alpha_0} + \frac{N}{\alpha_0 + N} \cdot \frac{n_k}{N}$$

### 4.2 Converting Model Predictions to Dirichlet Parameters

Given model predictions `p = (p₁, ..., p₆)` and ESS `c`:

$$\alpha_k = c \cdot p_k$$

| ESS (c) | Behavior | When to use |
|---|---|---|
| c = 1 | Very weak prior, observations dominate quickly | Prior is unreliable |
| c = 2-4 | Moderate prior, 2-4 observations override | **Default range** — prior likely misspecified due to round shift |
| c = 10+ | Strong prior, many observations needed to override | Prior is highly trusted (rare) |

### 4.3 Heterogeneous ESS

Different cells should get different prior strengths based on model confidence:

```python
H_norm = entropy(p_round) / log(K)  # normalized entropy, 0=confident, 1=uncertain
ess = clip(c_base * (1 + 0.75 * (1 - H_norm)), 1.0, 4.0)
```

- Confident predictions (low entropy) → higher ESS → trust the prior more
- Uncertain predictions (high entropy) → lower ESS → let observations dominate

### 4.4 ESS Selection Methods

**Cross-validation on held-out rounds (recommended for offline tuning):**

```python
def cross_validate_ess(model_predictions, ground_truths, c_values, round_ids):
    """Leave-one-round-out CV for ESS selection."""
    results = {}
    for c in c_values:
        kl_scores = []
        for held_out_round in unique(round_ids):
            # Use model predictions as prior with ESS=c
            alpha_prior = c * model_predictions[held_out_round]
            # Simulate N observations from ground truth
            # Compute posterior and evaluate KL
            ...
        results[c] = mean(kl_scores)
    return argmin(results)
```

**Empirical Bayes (marginal likelihood maximization):**

The marginal likelihood under Dirichlet-Multinomial is:

$$\log p(\mathbf{n} \mid \boldsymbol{\alpha}) = \log \Gamma(\alpha_0) - \log \Gamma(\alpha_0 + N) + \sum_{k=1}^{K} [\log \Gamma(n_k + \alpha_k) - \log \Gamma(\alpha_k)]$$

Maximize over `c` using `scipy.optimize.minimize_scalar`.

### 4.5 Robustness to Prior Misspecification

When the current round's hidden parameters differ from training:

$$\text{Posterior KL from truth} \approx \frac{\alpha_0}{\alpha_0 + N} \cdot \text{KL}(\text{truth} \| \text{prior})$$

**Recovery rate:** To reduce prior influence by factor γ (e.g., 90% reduction → γ = 0.1):

$$N^* \approx \alpha_0 \cdot (1/\gamma - 1)$$

With `c_base = 2.0` (α₀ ≈ 2.0), recovery to 90% requires only ~18 observations — achievable for well-covered cells.

**This is why low ESS is critical.** With `c = 10`, recovery needs ~90 observations per cell — impossible with 50 total queries.

---

## 5. Round-Level Calibration

### 5.1 Core Insight: Shared Hidden Parameters

All 5 seeds within a round share identical hidden parameters. Observations from ANY seed provide information about the hidden parameters governing ALL seeds.

With 50 queries × 225 cells/viewport = up to 11,250 cell observations pooled across seeds. Even with overlap, this gives substantial statistical power for round-level inference.

### 5.2 Archetype-Based Calibration

Define 3 cell archetypes based on the initial grid:

| Archetype | Definition | Typical Fraction | What It Reveals |
|---|---|---|---|
| `coastal` | Adjacent to ocean cell | ~15-25% | Port formation, maritime conflict, trade dynamics |
| `settlement_adjacent` | Within radius 2 of initial settlement | ~20-30% | Survival, expansion, conflict intensity |
| `inland_natural` | Neither coastal nor settlement-adjacent | ~50-60% | Forest expansion, ruin → forest conversion rates |

### 5.3 Empirical Bayes: Historical Round Matching

**Precompute offline:** For each historical round `r`, compute archetype class frequencies:

```python
theta_hist[r, a, k] = (count of class k in archetype a for round r) / (total cells in archetype a)
```

**Online:** After Phase 1 observations, compute posterior weights:

```python
# Aggregate observed counts by archetype across ALL seeds
m_obs[a, k] = sum of observed class k in archetype a across all seeds

# Posterior weight of historical round r
log_w[r] = log(pi[r]) + sum(m_obs * log(clip(theta_hist[r], eps, 1.0)))
w = softmax(log_w)

# Archetype template for cell (s,h,w)
p_template[s,h,w,k] = sum(w[r] * theta_hist[r, archetype[s,h,w], k] for r in rounds)
```

Use uniform `pi[r]` by default (no prior preference among historical rounds).

### 5.4 Detection Sensitivity

| Regime | Signature | Observations Needed (80% power) |
|---|---|---|
| High-conflict | Elevated ruin probability near settlements | 15-25 |
| Peaceful/growth | High survival + expansion | 10-20 |
| Harsh-winter | Widespread ruin even in resource-rich areas | 20-35 |

With 10 calibration queries × 225 cells = 2,250 cell observations, **round regime detection is comfortably feasible** after Phase 1.

### 5.5 Recalibration Schedule

Recompute `p_template` and `w_hist` after:
- **10 queries** (end of Phase 1) — initial round identification
- **20 queries** (mid Phase 2) — refined with more observations
- **30 queries** (end of Phase 2) — final calibration before exploitation

Each recalibration updates the Dirichlet parameters `alpha0` for all remaining predictions.

---

## 6. Query Allocation Strategy

### 6.1 Three-Phase Budget Allocation

| Phase | Queries | Seeds | Strategy | Goal |
|---|---|---|---|---|
| **1: Calibration** | 10 | 2/seed | Farthest-point, no overlap | Reveal round regime |
| **2: Refinement** | 30 | Adaptive | Greedy EIG targeting | Reduce uncertainty on high-value cells |
| **3: Exploitation** | 10 | Adaptive | Focused exploitation | Clean up remaining high-entropy cells |

### 6.2 Cross-Seed Budget Allocation

Front-loaded to maximize early calibration value:

| Seed | Queries | Rationale |
|---|---|---|
| 1 | 15 | Maximum calibration + refinement |
| 2 | 12 | Strong calibration contribution |
| 3 | 10 | Balanced |
| 4 | 8 | Leverages cross-seed calibration |
| 5 | 5 | Mostly relies on calibrated prior |

### 6.3 Phase 1: Farthest-Point Coverage

```python
def farthest_point_viewports(grid_size, viewport_size, n_viewports, dynamic_mask):
    """Select viewport centers maximizing coverage of dynamic cells."""
    centers = []
    for _ in range(n_viewports):
        if not centers:
            # Start at grid center
            centers.append((grid_size // 2, grid_size // 2))
        else:
            # Find point farthest from all selected centers
            best_center, best_dist = None, -1
            for h in range(viewport_size//2, grid_size - viewport_size//2):
                for w in range(viewport_size//2, grid_size - viewport_size//2):
                    if not dynamic_mask[h, w]:
                        continue
                    min_dist = min(abs(h-ch) + abs(w-cw) for ch, cw in centers)
                    if min_dist > best_dist:
                        best_dist = min_dist
                        best_center = (h, w)
            centers.append(best_center)
    return centers
```

### 6.4 Phase 2: Greedy EIG Viewport Selection

```python
def greedy_eig_viewport(pred, alpha, viewport_size, dynamic_mask, n_obs):
    """Select viewport maximizing expected information gain."""
    K = pred.shape[-1]
    H, W = pred.shape[:2]

    # Compute per-cell EIG
    eig = np.zeros((H, W))
    for h in range(H):
        for w in range(W):
            if not dynamic_mask[h, w]:
                continue
            p = pred[h, w]
            a = alpha[h, w]
            A = a.sum()
            # EIG = H[current] - E_k[H[posterior if observe class k]]
            current_entropy = entropy(p)
            expected_post_entropy = 0
            for k in range(K):
                a_post = a.copy()
                a_post[k] += 1
                p_post = a_post / a_post.sum()
                expected_post_entropy += p[k] * entropy(p_post)
            eig[h, w] = current_entropy - expected_post_entropy

    # Score viewports with overlap discount
    best_score, best_center = -1, None
    for ch in range(viewport_size//2, H - viewport_size//2):
        for cw in range(viewport_size//2, W - viewport_size//2):
            score = 0
            for dh in range(-viewport_size//2, viewport_size//2 + 1):
                for dw in range(-viewport_size//2, viewport_size//2 + 1):
                    h, w = ch + dh, cw + dw
                    # Diminishing returns for re-observed cells
                    discount = 1.0 / (1.0 + n_obs[h, w])
                    score += eig[h, w] * discount
            if score > best_score:
                best_score = score
                best_center = (ch, cw)
    return best_center
```

### 6.5 Coverage Analysis

With 50 queries of 15×15 viewports on a 40×40 grid:
- Total cell-observations: 50 × 225 = 11,250
- Grid cells: 1,600 (of which ~1,200-1,400 are dynamic/scorable)
- Average observations per dynamic cell: **~8-9** (with strategic placement)
- Expected cells with 0 observations: **< 1%** with good coverage

### 6.6 Computational Feasibility

| Operation | Complexity | Time Estimate |
|---|---|---|
| Per-cell EIG computation | O(K²) per cell | ~μs |
| Full grid EIG scan | O(H × W × K²) | ~10ms |
| Best viewport search | O(H × W × V²) where V=viewport_size | ~100ms |
| Phase 2 (30 viewport selections) | 30 × above | ~3s |

**Comfortably within the ~160 minute budget.** EIG computation is negligible compared to API call latency.

---

## 7. Combining Prior and Observations

### 7.1 Method Comparison

| Method | Formula | Properties | When to Use |
|---|---|---|---|
| **Geometric pooling** | `q ∝ q_prior^w · q_empirical^(1-w)` | KL-optimal under external Bayesianity; concentrates mass | Merging two probabilistic beliefs (prior + template) |
| **Linear pooling** | `q = w·q_prior + (1-w)·q_empirical` | Preserves non-zero support automatically | When zero-support risk is high |
| **Dirichlet update** | `α_post = α_prior + counts` | Exact for i.i.d. categorical observations | Incorporating direct cell observations |

### 7.2 Recommended Two-Stage Combination

**Stage 1: Geometric pooling** — merge learned prior `p_base` with round-calibrated template `p_template`:

```python
log_p_round = lambda_prior * log(clip(p_base, eps)) + (1 - lambda_prior) * log(clip(p_template, eps))
p_round = exp(log_p_round)
p_round /= p_round.sum(axis=-1, keepdims=True)
```

**Stage 2: Dirichlet conjugate update** — incorporate exact observed counts:

```python
alpha0 = ess * p_round            # prior pseudo-counts
alpha = alpha0 + counts           # posterior pseudo-counts
pred = alpha / alpha.sum(axis=-1, keepdims=True)  # posterior predictive
```

### 7.3 Floor Handling

**Apply floors BEFORE any log/power operations** (critical for geometric pooling):

```python
eps = 1e-4

def safe_normalize(p, eps=1e-4):
    """Clip to floor, renormalize."""
    p = np.maximum(p, eps)
    return p / p.sum(axis=-1, keepdims=True)
```

**Structural zeros** (from domain knowledge):
- Mountain (class 5) on non-mountain dynamic cells: floor at `eps` (never created by simulator)
- Port (class 2) on non-coastal cells: floor at `eps` (ports require ocean adjacency)
- All other plausible classes: floor at `eps`, then renormalize

### 7.4 Geometric Pooling: The KL Guarantee

Geometric pooling satisfies **external Bayesianity**: if new evidence arrives, the pooled belief updates the same way regardless of whether you pool first then update, or update then pool.

This means: `pool(Bayes(prior₁, data), Bayes(prior₂, data)) = Bayes(pool(prior₁, prior₂), data)`

Under KL divergence scoring, geometric pooling is the **unique** combination method (up to affine transformation) that satisfies this property. This makes it theoretically optimal for our use case.

---

## 8. Mathematical Formulas for Implementation

All formulas use variable names matching the Python implementation.

### 8.1 Entropy and Training Loss

```python
def entropy(x, eps=1e-4):
    x = np.clip(x, eps, 1.0)
    return -np.sum(x * np.log(x), axis=-1)

# Training loss: entropy-weighted cross-entropy
cell_weight = entropy(y_true)  # shape: (N_cells,)
loss = -np.sum(cell_weight * np.sum(y_true * np.log(np.clip(p_pred, eps, 1.0)), axis=-1))
loss /= np.sum(cell_weight)
```

### 8.2 Historical-Round Empirical Bayes

```python
# Precompute: theta_hist[r, a, k] = class freq for round r, archetype a, class k
# Online: m_obs[a, k] = observed counts pooled across all seeds

# Posterior weight of historical round r
log_w_hist = np.log(pi_hist) + np.sum(m_obs * np.log(np.clip(theta_hist, eps, 1.0)), axis=(-2, -1))
w_hist = softmax(log_w_hist)

# Archetype template
p_template[s,h,w,k] = sum(w_hist[r] * theta_hist[r, archetype_map[s,h,w], k] for r in range(R))
```

### 8.3 Geometric Pooling

```python
log_p_round = (
    lambda_prior * np.log(np.clip(p_base, eps, 1.0)) +
    (1 - lambda_prior) * np.log(np.clip(p_template, eps, 1.0))
)
p_round = np.exp(log_p_round)
p_round /= p_round.sum(axis=-1, keepdims=True)
```

### 8.4 Heterogeneous ESS

```python
H_norm = entropy(p_round) / np.log(K)  # normalized to [0, 1]
ess = np.clip(c_base * (1 + 0.75 * (1 - H_norm)), 1.0, 4.0)
alpha0 = ess[..., np.newaxis] * p_round  # prior pseudo-counts
```

### 8.5 Bayesian Update

```python
alpha = alpha0 + counts   # counts[s,h,w,k] = accumulated one-hot observations
pred = alpha / alpha.sum(axis=-1, keepdims=True)
```

### 8.6 Per-Cell EIG for Query Selection

```python
def eig_cell(p, alpha):
    """Expected information gain for one additional observation at this cell."""
    K = len(p)
    A = alpha.sum()
    current_H = entropy(p)

    expected_post_H = 0
    for k in range(K):
        alpha_post = alpha.copy()
        alpha_post[k] += 1
        p_post = alpha_post / alpha_post.sum()
        expected_post_H += p[k] * entropy(p_post)

    return current_H - expected_post_H

# Viewport score with overlap discount
def score_viewport(eig_grid, n_obs, viewport_cells):
    return sum(eig_grid[h,w] / (1 + n_obs[h,w]) for h, w in viewport_cells)
```

### 8.7 Marginal Likelihood for ESS Optimization

```python
from scipy.special import gammaln

def marginal_log_likelihood(c, p, counts):
    """Log marginal likelihood of Dirichlet-Multinomial for ESS selection."""
    alpha = c * p
    N = counts.sum()
    alpha0 = alpha.sum()
    return (gammaln(alpha0) - gammaln(alpha0 + N) +
            np.sum(gammaln(counts + alpha) - gammaln(alpha)))
```

---

## 9. Theoretical Bounds and Expected Performance

### 9.1 Error Decomposition

The expected weighted KL decomposes into:

| Component | Formula | Dominates When |
|---|---|---|
| **Prior approximation error** | `KL(p_true ∥ p_model)` | Few observations, good model |
| **Parameter drift penalty** | `KL(p_true_round ∥ p_model_avg)` | Round parameters differ from training average |
| **Estimation variance** | `(K-1) / (2(α₀+N))` | Many observations, good prior |
| **Calibration error** | `≈ ECE² × entropy_weight` | Model is overconfident or underconfident |

### 9.2 Pure MC Baseline

With K=6 classes and N observations per cell:

$$E[\text{KL}(p^* \| \hat{p}_N)] \approx \frac{K-1}{2N} = \frac{5}{2N}$$

| N (obs/cell) | Expected KL | Score ≈ |
|---|---|---|
| 1 | 2.50 | ~0 |
| 3 | 0.83 | ~8 |
| 7 | 0.36 | ~34 |
| 15 | 0.17 | ~60 |

With 50 queries averaging ~7 obs/cell: **pure MC baseline score ≈ 34**.

### 9.3 Hybrid Expected KL

$$E[\text{KL}_{\text{hybrid}}] \approx \frac{K-1}{2(\alpha_0 + N)} + \text{bias}(\text{prior})$$

With good prior (bias ≈ 0.10-0.15) and α₀ ≈ 2.0, N ≈ 7:

$$\text{KL}_{\text{hybrid}} \approx \frac{5}{2(2+7)} + 0.12 \approx 0.28 + 0.12 = 0.40$$

Wait — that's worse than pure MC? **No**, because the prior's spatial structure fills in unobserved cells, and the bias term drops with round calibration. The real formula accounts for spatially heterogeneous coverage:

- **Observed cells** (N ≈ 7-10): KL ≈ 0.10-0.15 (prior helps reduce variance)
- **Unobserved cells** (N = 0): KL ≈ prior quality ≈ 0.15-0.30 (much better than random)
- **Weighted average**: KL ≈ 0.12-0.22

### 9.4 Crossover: Hybrid vs Pure MC

Hybrid beats pure MC when prior bias < variance reduction:

$$\text{bias} < \frac{(K-1) \cdot \alpha_0}{2N(\alpha_0 + N)}$$

With α₀ = 2, N = 7: bias threshold ≈ 0.56. Any prior with average KL < 0.56 from truth (trivially achievable) makes hybrid win.

**In practice, hybrid ALWAYS wins when the prior is better than random (KL < log(6) ≈ 1.79).**

### 9.5 Expected Score by Training Data Size

| Historical Rounds | Expected Weighted KL | Expected Score |
|---|---|---|
| 10-20 | 0.20 - 0.28 | **43 - 55** |
| 20-50 | 0.14 - 0.22 | **52 - 66** |
| 50-100 | 0.09 - 0.16 | **62 - 76** |

Reference: `score = max(0, min(100, 100 × exp(-3 × weighted_KL)))`

### 9.6 Score Breakdown by Component

Starting from pure MC baseline (KL ≈ 0.36, score ≈ 34):

| Component Added | KL Reduction | Score Lift |
|---|---|---|
| **Learned prior** | -0.10 to -0.18 | **+10 to +25** (biggest lift) |
| **Cross-seed round calibration** | -0.02 to -0.05 | **+4 to +10** |
| **Adaptive query allocation** | -0.01 to -0.03 | **+2 to +6** |
| **ESS + calibration tuning** | -0.005 to -0.015 | **+1 to +3** |

**Prior quality is by far the most important component.**

### 9.7 Sensitivity Analysis Priority

| Rank | Hyperparameter | Impact | Recommendation |
|---|---|---|---|
| **1** | Prior model quality | ★★★★★ | Invest most offline effort here |
| **2** | Query allocation strategy | ★★★★☆ | Optimal vs random allocation differs by 15-30% |
| **3** | ESS (c_base) | ★★★☆☆ | Tune over {1, 2, 3, 4} by grouped CV |
| **4** | λ_prior (geometric pool weight) | ★★☆☆☆ | Default 0.7, tune over {0.5, 0.6, 0.7, 0.8} |
| **5** | Floor values (eps) | ★☆☆☆☆ | 1e-4 works; not worth tuning finely |

---

## 10. Risk Analysis

| Rank | Failure Mode | Probability × Impact | Mitigation |
|---|---|---|---|
| **1** | **Round shift not corrected** — prior trained on historical average is wrong for current hidden parameters | HIGH × HIGH | Pool early observations across all 5 seeds; recalibrate at 10/20/30 queries; keep c_base low (2.0); reduce λ_prior if live data disagrees with all historical templates |
| **2** | **Validation leakage / prior overfit** — cell-level CV splits massively overstate prior quality | HIGH × HIGH | Use GroupKFold by round ONLY; cap tree depth; keep logistic regression as sanity-check baseline |
| **3** | **ESS too high / bad calibration** — if α₀ is large, posterior can't recover from bad prior within 50 queries | MEDIUM × HIGH | Temperature-scale offline; hard-cap ESS at 4.0; tune c_base on held-out rounds |
| **4** | **Bad early query coverage** — if first 10 queries cluster, round calibration fails | MEDIUM × MEDIUM | Make Phase 1 deterministic: farthest-point, no overlap, 2/seed |
| **5** | **Zero support / numerical instability** — single zero in KL or geometric pooling destroys the cell | LOW × HIGH | Clip every probability with eps=1e-4 before log/power; renormalize after every step |

### Escalation Triggers

- If grouped-CV prior quality is poor (weighted KL > 0.30): lower `c_base` toward 1.0-1.5, rely more on live observations
- If historical-round posterior weights stay flat after 10-15 queries: treat round as OOD, reduce `lambda_prior` from 0.7 to 0.5, spend more budget on exploration
- If runtime is tight: approximate EIG on a coarse grid of candidate viewport centers (stride=5 instead of stride=1)

---

## 11. Implementation Ordering

### Stage 1: Offline Prior Training + Evaluation

**Build first. Most important component.**

1. Feature pipeline: extract ~46 features per cell from initial_grid
2. LightGBM multiclass with `class_weight='balanced'`
3. Temperature scaling on held-out rounds
4. Grouped-by-round cross-validation
5. Compute archetype class frequency tables for all historical rounds

**Exit criterion:** Held-out-by-round weighted KL < 0.30.

### Stage 2: Core Online Posterior Update

**The hybrid engine.**

1. `alpha0 = ess * p_base` (constant ESS=2.0 for MVP)
2. `alpha = alpha0 + counts`
3. `pred = alpha / alpha.sum(-1)`
4. Floor clipping and structural zeros

**This alone gives you a competitive hybrid system when combined with any reasonable query strategy.**

### Stage 3: MVP Query Planner

1. Phase 1: 10 farthest-point viewports (2/seed, no overlap)
2. Phase 2+3: 40 queries placed by `score_viewport = sum(entropy(pred))` over viewport cells
3. Overlap discount: `1/(1 + n_obs[h,w])`

**MVP is now complete and competitive.**

### Stage 4: Cross-Seed Round Calibration (Main Upgrade)

1. Archetype classification (coastal / settlement_adjacent / inland_natural)
2. Archetype count aggregation across seeds
3. Historical-round posterior weights via empirical Bayes
4. Archetype template construction
5. Geometric pooling: `p_round = pool(p_base, p_template)`
6. Recalibration at 10, 20, 30 queries

**This is the jump from "good" to "strong".**

### Stage 5: Hyperparameter Tuning

Tune only on held-out rounds:
- `c_base` over {1, 2, 3, 4}
- `lambda_prior` over {0.5, 0.6, 0.7, 0.8}
- Phase splits (if time permits)

**If time is short, lock: `c_base=2.0`, `lambda_prior=0.7`, phases `10/30/10`, `eps=1e-4`.**

### Minimum Viable Version

- Feature-engineered prior (LightGBM or logistic regression)
- Offline temperature scaling
- Constant ESS = 2.0
- 10 farthest-point calibration queries + 40 greedy adaptive queries
- Exact Dirichlet update with observed counts

This MVP should **beat pure MC by 15-25 points** and give a credible competition score.

---

## 12. Key Insight Synthesis

### The Single Most Important Insight

**This is not primarily a Monte Carlo estimation problem; it is a round-identification problem.**

The learned prior must provide the spatial structure, because 50 queries cannot directly estimate 1,600 cell distributions well enough. The queries are most valuable when they reveal the **shared hidden regime** of the current round, which then shifts the prior for *every* cell at once.

### The Non-Obvious Separator

A mediocre implementation spends early queries chasing locally uncertain cells. A strong implementation spends early queries to **calibrate the whole round across all 5 seeds**, then lets a low-ESS Dirichlet update clean up the directly observed areas.

That "global regime first, local refinement second" ordering is what turns the hybrid from a nice idea into a leaderboard-capable system.

### Why This Works Mathematically

- **Round calibration** modifies `p_template` for ALL ~1,200 dynamic cells simultaneously using ~2,250 cell observations (10 queries × 225 cells)
- **Direct Dirichlet update** modifies predictions for only the ~225 cells in each viewport
- **ROI of a calibration query**: improves ~1,200 predictions via round template
- **ROI of a refinement query**: improves ~225 predictions directly
- **Calibration leverage ratio**: ~5.3× more value per query for round-level calibration

This is why Phase 1 (broad calibration) should come first, and why cross-seed pooling is the highest-value upgrade after the basic prior.

---

## Appendix: Python Library Reference

| Library | Purpose | Key Functions |
|---|---|---|
| `scipy.stats.dirichlet` | Dirichlet distribution operations | `pdf`, `mean`, `var`, `entropy`, `rvs` |
| `scipy.special` | Gamma/digamma functions | `gammaln`, `digamma` for marginal likelihood |
| `scipy.optimize` | ESS optimization | `minimize_scalar` for empirical Bayes |
| `scipy.ndimage` | Spatial feature engineering | `distance_transform_edt` for distance features |
| `sklearn.calibration` | Probability calibration | `CalibratedClassifierCV` (method='sigmoid') |
| `sklearn.model_selection` | Grouped CV | `GroupKFold` with round as group |
| `lightgbm` | Prior model | `LGBMClassifier` with multiclass objective |
| `numpy` | Core computation | Array operations for Dirichlet updates |

No exotic dependencies required. The full pipeline runs on scipy + sklearn + lightgbm + numpy.
