# Approach 2: Supervised Learning from Historical Rounds

## Comprehensive Research Document

**Objective**: Train a model on all available historical `(initial_grid → ground_truth)` pairs to predict the full probability tensor for new unseen grids, using zero live queries.

**Dataset**: ~90 fixture files (16+ rounds × 5 seeds), each containing a 40×40 grid with 8 terrain codes mapped to 6 prediction classes, and a ground-truth probability tensor of shape `(40, 40, 6)`.

**Scoring**: `score = 100 × exp(-3 × weighted_KL)`, where `weighted_KL = Σ(entropy_i × KL_i) / Σ(entropy_i)`.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Feature Engineering for Spatial Grids](#2-feature-engineering-for-spatial-grids)
3. [Model Architecture Selection](#3-model-architecture-selection)
4. [Data Augmentation and Regularization](#4-data-augmentation-and-regularization)
5. [Handling Hidden Parameter Variation](#5-handling-hidden-parameter-variation)
6. [Loss Function and Calibration](#6-loss-function-and-calibration)
7. [Evaluation and Validation](#7-evaluation-and-validation)
8. [Implementation Blueprint](#8-implementation-blueprint)

---

## 1. Problem Formulation

### 1.1 Core Task

Given an initial grid `G ∈ {0,1,2,3,4,5,10,11}^{40×40}` (8 terrain codes), predict a probability tensor `P ∈ [0,1]^{40×40×6}` where `P[x,y,:]` is a valid probability distribution over 6 prediction classes (EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN) for each cell.

The terrain code mapping collapses 8 codes to 6 classes:
- OCEAN (10) → EMPTY (0)
- PLAINS (11) → EMPTY (0)
- All others map directly (0→0, 1→1, 2→2, 3→3, 4→4, 5→5)

### 1.2 Loss Function Choice

The competition scores via entropy-weighted KL divergence:

```
weighted_KL = Σ_{i} H(p_i) · KL(p_i || q_i) / Σ_{i} H(p_i)
```

where `p_i` is the ground truth distribution at cell `i`, `q_i` is our prediction, and `H(p_i)` is the entropy of the ground truth.

**Key insight**: High-entropy cells (uncertain outcomes) receive more weight. Cells where the ground truth is nearly deterministic (e.g., ocean stays empty with p≈1) contribute very little to the score regardless of our prediction quality. This means **our model's accuracy on ambiguous cells matters most**.

#### Which loss to train with?

| Loss | Formula | KL-optimal? | Notes |
|------|---------|-------------|-------|
| **Cross-entropy (soft)** | `-Σ_k p_k log q_k` | Yes* | Equivalent to KL up to constant `H(p)` |
| **KL divergence** | `Σ_k p_k log(p_k/q_k)` | Yes | Direct match to scoring metric |
| **Brier score** | `Σ_k (p_k - q_k)²` | No | L2 loss, different optimal |
| **Entropy-weighted KL** | `H(p) · KL(p\|\|q)` | Yes | Matches competition metric exactly |

**Recommendation**: Train with **soft cross-entropy** (equivalent to KL up to the constant `H(p)` which doesn't depend on `q`). For tree-based models that use MSE internally, post-hoc calibration is needed.

*Proof*: `CE(p, q) = -Σ_k p_k log q_k = H(p) + KL(p||q)`. Since `H(p)` is constant w.r.t. `q`, minimizing CE is equivalent to minimizing KL.

### 1.3 Soft-Label Training

Our targets are **not** hard class labels — they are full probability vectors. This is critical:

- Standard `nn.CrossEntropyLoss` expects integer class indices → **cannot use directly**
- Must use soft cross-entropy: `L(q, p) = -Σ_k p_k log q_k`
- For tree models: train K=6 separate regressors, one per class probability, then renormalize

### 1.4 Per-Cell vs. Whole-Grid Formulation

Two modeling approaches:

**Per-cell model**: Extract features for each cell → predict its 6-class distribution independently.
- Training examples: `N_grids × 40 × 40 = N × 1600` cells
- With 90 fixtures: **144,000 training cells**
- Pro: Large effective dataset, simple
- Con: Ignores spatial correlations between cells

**Whole-grid model (CNN/U-Net)**: Input the entire 40×40 grid → output the entire 40×40×6 tensor.
- Training examples: `N_grids` (just 90)
- Pro: Captures spatial patterns (e.g., settlements cluster, forests spread)
- Con: Very small dataset for a neural network

**Recommendation**: Start with per-cell models (large effective N), then try spatial models with heavy augmentation. The per-cell approach with rich spatial features can capture most spatial information while having 1600× more training examples.

### 1.5 Optimal Prediction Under the Scoring Metric

**Theorem**: The KL-optimal prediction is the arithmetic mean of the training distributions for cells with matching features.

Given training cells with feature vector `f` and ground-truth distributions `p_1, ..., p_m`:

```
q*(f) = (1/m) Σ_{j=1}^{m} p_j
```

This is the **arithmetic mean**, not the geometric mean. The arithmetic mean minimizes `E[KL(p || q)]` over the training distribution.

*Proof sketch*: For fixed `q`, `E[KL(p||q)] = E[-Σ_k p_k log q_k] - E[H(p)]`. The first term is minimized by `q_k = E[p_k]` (by Jensen's inequality on the concave `log`). The second term is constant w.r.t. `q`.

**Corollary**: For entropy-weighted KL, the optimal prediction is the **entropy-weighted arithmetic mean**:

```
q*(f) = Σ_j w_j p_j / Σ_j w_j,  where w_j = H(p_j)
```

However, since we don't know `H(p_j)` at test time (the ground truth is unknown), the unweighted arithmetic mean is the practical optimum.

---

## 2. Feature Engineering for Spatial Grids

### 2.1 Existing Features (from `strategies.py`)

The current codebase uses a 4-tuple context key per cell:

```python
context_key = (initial_class, has_forest_neighbor, has_settlement_neighbor, is_coastal)
```

This gives 22 distinct context keys from the data, capturing major variation:
- `initial_class`: The cell's own terrain type (6 values after mapping)
- `has_forest_neighbor`: Binary — any of 8 neighbors is FOREST
- `has_settlement_neighbor`: Binary — any of 8 neighbors is SETTLEMENT
- `is_coastal`: Binary — any of 8 neighbors is OCEAN

### 2.2 Enhanced Local Features

Richer features that maintain interpretability while capturing more spatial structure:

#### 2.2.1 Neighbor Counts (replacing binary indicators)

```python
import numpy as np
import scipy.ndimage as ndimage

def count_neighbors_by_class(grid: np.ndarray, num_classes: int = 6) -> np.ndarray:
    """Count neighbors of each class in 3×3 window (excluding center).
    
    Returns: (40, 40, 6) array where [x,y,k] = count of class-k neighbors.
    """
    H, W = grid.shape
    counts = np.zeros((H, W, num_classes), dtype=np.int32)
    
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Exclude center cell
    
    for k in range(num_classes):
        binary = (grid == k).astype(np.float32)
        counts[:, :, k] = ndimage.convolve(binary, kernel, mode='constant', cval=0).astype(np.int32)
    
    return counts
```

This replaces `has_forest_neighbor` (binary) with `forest_neighbor_count` (0–8), capturing more gradation. A cell with 5 forest neighbors behaves differently from one with 1.

#### 2.2.2 Distance Transforms

Distance to the nearest cell of each class, computed via Euclidean distance transform:

```python
from scipy.ndimage import distance_transform_edt

def compute_distance_features(grid: np.ndarray, num_classes: int = 6) -> np.ndarray:
    """Distance to nearest cell of each class.
    
    Returns: (40, 40, 6) array where [x,y,k] = Euclidean distance to nearest class-k cell.
    """
    H, W = grid.shape
    distances = np.zeros((H, W, num_classes))
    
    for k in range(num_classes):
        mask = (grid == k)
        if mask.any():
            distances[:, :, k] = distance_transform_edt(~mask)
        else:
            distances[:, :, k] = 999.0  # No cells of this class exist
    
    return distances
```

Distance features capture long-range spatial structure: "how far am I from the ocean?" matters for port placement. Max useful distance on a 40×40 grid is ~56.6 (diagonal), so normalize by dividing by 40.

#### 2.2.3 Local Entropy (Neighborhood Diversity)

```python
from scipy.stats import entropy

def local_class_entropy(grid: np.ndarray, window_size: int = 5, num_classes: int = 6) -> np.ndarray:
    """Shannon entropy of class distribution in local window.
    
    High entropy = diverse neighborhood. Low entropy = homogeneous area.
    Returns: (40, 40) array.
    """
    H, W = grid.shape
    one_hot = np.zeros((H, W, num_classes))
    for k in range(num_classes):
        one_hot[:, :, k] = (grid == k).astype(np.float32)
    
    # Average one-hot over window → local class probabilities
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_probs = np.zeros((H, W, num_classes))
    for k in range(num_classes):
        local_probs[:, :, k] = ndimage.convolve(one_hot[:, :, k], kernel, mode='constant', cval=0)
    
    # Compute entropy at each cell
    ent = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            p = local_probs[i, j]
            p = p[p > 0]  # Filter zeros for log
            ent[i, j] = -np.sum(p * np.log(p))
    
    return ent
```

#### 2.2.4 Positional Features

```python
def positional_features(H: int = 40, W: int = 40) -> np.ndarray:
    """Normalized (x, y) coordinates + distance from center/edges.
    
    Returns: (40, 40, 4) array with [norm_x, norm_y, dist_center, dist_edge].
    """
    xs = np.arange(H) / (H - 1)  # 0 to 1
    ys = np.arange(W) / (W - 1)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    
    dist_center = np.sqrt((xx - 0.5)**2 + (yy - 0.5)**2)
    dist_edge = np.minimum(np.minimum(xx, 1 - xx), np.minimum(yy, 1 - yy))
    
    return np.stack([xx, yy, dist_center, dist_edge], axis=-1)
```

Coastal cells concentrate near edges; terrain distribution may vary with position.

### 2.3 Global Features (Grid-Level)

Features computed once per grid and broadcast to all cells:

```python
def global_grid_features(grid: np.ndarray, num_classes: int = 6) -> np.ndarray:
    """Grid-level statistics broadcast to every cell.
    
    Returns: (num_classes + 2,) vector:
      - Class fractions (6 values)
      - Total ocean fraction
      - Grid entropy
    """
    total = grid.size
    class_fractions = np.array([(grid == k).sum() / total for k in range(num_classes)])
    
    ocean_fraction = class_fractions[0]  # EMPTY includes OCEAN+PLAINS
    grid_entropy = entropy(class_fractions[class_fractions > 0])
    
    return np.concatenate([class_fractions, [ocean_fraction, grid_entropy]])
```

Global features act as a crude proxy for hidden simulation parameters (see Section 5).

### 2.4 Complete Feature Vector

Combining all features per cell:

| Feature Group | Dimensions | Description |
|---------------|-----------|-------------|
| One-hot class | 6 | Cell's own class (one-hot encoded) |
| Neighbor counts | 6 | Count of each class in 3×3 neighborhood |
| Distance transforms | 6 | Distance to nearest cell of each class |
| Local entropy | 1 | Shannon entropy in 5×5 window |
| Position | 4 | (x, y, dist_center, dist_edge) |
| Global fractions | 8 | Class fractions + ocean fraction + grid entropy |
| **Total** | **31** | Per-cell feature vector |

For per-cell models, this gives a feature matrix of shape `(N × 1600, 31)` and target matrix `(N × 1600, 6)`.

### 2.5 Feature Importance Expectations

Based on the existing `CONTEXT_TRANSITIONS` table:

1. **Cell's own class** — dominant feature; empty cells, forests, mountains have very different transition distributions
2. **is_coastal (neighbor ocean count)** — gates port transitions; ports only appear on coastal cells
3. **has_settlement_neighbor (settlement count)** — influences settlement clustering/spread
4. **has_forest_neighbor (forest count)** — influences forest spread dynamics
5. **Distance to ocean** — long-range version of coastal detection
6. **Global class fractions** — proxy for hidden parameters (thriving rounds have more settlements)

---

## 3. Model Architecture Selection

### 3.1 Architecture Comparison for This Problem

| Architecture | Effective N | Spatial Context | Calibration | Training Time | Complexity |
|-------------|------------|-----------------|-------------|--------------|------------|
| Logistic Regression | 144k cells | Via features only | Good (inherent) | Seconds | Low |
| Random Forest | 144k cells | Via features only | Poor (needs calibration) | Seconds | Low |
| Gradient Boosted Trees | 144k cells | Via features only | Moderate | Minutes | Low |
| k-NN (distributional) | 144k cells | Via features only | Inherent | Seconds | Low |
| Small U-Net (2-3 layers) | 90 grids | Native (conv) | Moderate | Minutes | Medium |
| Per-cell MLP | 144k cells | Via features only | Good | Minutes | Low-Medium |

### 3.2 Per-Cell Models (Recommended Starting Point)

#### 3.2.1 Logistic Regression Baseline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

def train_logistic_baseline(X_train, y_train_soft):
    """Train K=6 logistic regressors for soft probability targets.
    
    X_train: (N*1600, 31) feature matrix
    y_train_soft: (N*1600, 6) soft probability targets
    """
    models = []
    for k in range(6):
        model = LogisticRegression(
            C=1.0,       # Regularization (tune via CV)
            max_iter=1000,
            solver='lbfgs'
        )
        # Convert soft targets to binary for logistic regression
        # Use probability as sample weight
        y_binary = (y_train_soft[:, k] > 0.5).astype(int)
        weights = np.abs(y_train_soft[:, k] - 0.5) + 0.1  # Higher weight for confident cells
        model.fit(X_train, y_binary, sample_weight=weights)
        models.append(model)
    
    return models

def predict_logistic(models, X_test):
    """Predict and renormalize."""
    probs = np.stack([m.predict_proba(X_test)[:, 1] for m in models], axis=1)
    probs = np.clip(probs, 1e-6, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs
```

**Better approach for soft targets — use regression directly:**

```python
from sklearn.linear_model import Ridge

def train_ridge_soft(X_train, y_train_soft, alpha=1.0):
    """Ridge regression for each class probability. Simple, fast, effective."""
    models = []
    for k in range(6):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train_soft[:, k])
        models.append(model)
    return models

def predict_ridge(models, X_test):
    probs = np.stack([m.predict(X_test) for m in models], axis=1)
    probs = np.clip(probs, 1e-6, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs
```

#### 3.2.2 Gradient Boosted Trees (Strong Baseline)

```python
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

def train_gbt_soft(X_train, y_train_soft, max_iter=200, max_depth=4):
    """Train K=6 GBT regressors on soft probability targets."""
    models = []
    for k in range(6):
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=0.05,
            min_samples_leaf=20,     # Regularize for stability
            l2_regularization=1.0,   # Prevent overfitting
            max_bins=64,             # Reduce complexity
        )
        model.fit(X_train, y_train_soft[:, k])
        models.append(model)
    return models

def predict_gbt(models, X_test):
    """Predict and renormalize to valid probabilities."""
    probs = np.stack([m.predict(X_test) for m in models], axis=1)
    probs = np.clip(probs, 1e-6, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs
```

**Hyperparameter guidance for 144k cells, 31 features:**
- `max_depth=4-6`: Deeper trees capture more interactions but risk overfitting cross-round variance
- `min_samples_leaf=20-50`: Prevents leaves with too few cells
- `learning_rate=0.01-0.1`: Lower = more trees needed but better generalization
- `l2_regularization=0.1-10.0`: Key regularizer for cross-round generalization

#### 3.2.3 Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

def train_rf_soft(X_train, y_train_soft, n_estimators=200, max_depth=8):
    """Random Forest for soft probability targets."""
    models = []
    for k in range(6):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=10,
            max_features='sqrt',
            n_jobs=-1,
        )
        model.fit(X_train, y_train_soft[:, k])
        models.append(model)
    return models
```

Random forests tend to produce **overconfident** probability estimates because they average indicator functions. Post-hoc calibration (Section 6) is essential.

#### 3.2.4 k-NN (Distributional)

k-NN naturally handles soft targets by averaging neighbor distributions:

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_knn_soft(X_train, y_train_soft, n_neighbors=15):
    """k-NN that averages probability vectors of neighbors.
    
    This is mathematically optimal under KL divergence (arithmetic mean).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    models = []
    for k in range(6):
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights='distance',   # Inverse-distance weighting
            algorithm='ball_tree',
            metric='minkowski',
            p=2,
        )
        model.fit(X_scaled, y_train_soft[:, k])
        models.append(model)
    
    return models, scaler

def predict_knn(models, scaler, X_test):
    X_scaled = scaler.transform(X_test)
    probs = np.stack([m.predict(X_scaled) for m in models], axis=1)
    probs = np.clip(probs, 1e-6, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs
```

**k-NN is theoretically ideal here**: the KL-optimal prediction is the arithmetic mean of matching training distributions, which is exactly what k-NN computes with uniform weights. Key tuning parameter is `k` (number of neighbors) — higher `k` smooths more, reducing variance but increasing bias.

**Recommended k range**: 10–50. With 144k training cells, `k=30` is a reasonable starting point.

### 3.3 Spatial Models (CNN/U-Net)

#### 3.3.1 When to Use Spatial Models

Spatial models are justified when:
- Per-cell models plateau and you suspect spatial correlations are being missed
- Feature engineering can't capture the relevant spatial patterns
- You have enough augmented training data (see Section 4)

With 90 grids × 8 (D4 augmentation) = 720 effective training examples, a small U-Net is feasible but risky.

#### 3.3.2 Small U-Net Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyUNet(nn.Module):
    """Minimal U-Net for 40×40 grids. ~50k parameters."""
    
    def __init__(self, in_channels: int = 6, out_channels: int = 6):
        super().__init__()
        # Encoder
        self.enc1 = self._block(in_channels, 16)  # 40×40 → 40×40
        self.enc2 = self._block(16, 32)            # 20×20 → 20×20
        self.enc3 = self._block(32, 64)            # 10×10 → 10×10
        
        # Bottleneck
        self.bottleneck = self._block(64, 64)      # 5×5 → 5×5
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec3 = self._block(128, 32)           # 10×10
        self.up2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec2 = self._block(64, 16)            # 20×20
        self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.dec1 = self._block(32, 16)            # 40×40
        
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),  # GroupNorm for small batches
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
        )
    
    def forward(self, x):
        # x: (B, C_in, 40, 40)
        e1 = self.enc1(x)                          # (B, 16, 40, 40)
        e2 = self.enc2(self.pool(e1))              # (B, 32, 20, 20)
        e3 = self.enc3(self.pool(e2))              # (B, 64, 10, 10)
        
        b = self.bottleneck(self.pool(e3))         # (B, 64, 5, 5)
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # (B, 32, 10, 10)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 16, 20, 20)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 16, 40, 40)
        
        return self.out_conv(d1)  # (B, 6, 40, 40) — raw logits
```

**Critical design choices:**
- **GroupNorm instead of BatchNorm**: BatchNorm fails with batch size 1–4 (which is common here). GroupNorm with 8 groups is stable regardless of batch size.
- **Dropout2d at 0.15**: Spatial dropout — drops entire feature channels. Higher rates (0.3+) caused underfitting in small-model experiments.
- **~50k parameters**: Deliberately tiny. With 90 training grids, even this may overfit without augmentation.
- **Input channels = 6**: One-hot encoding of the initial grid's class at each cell.

#### 3.3.3 Input Encoding for CNN

```python
def encode_grid_for_cnn(grid: np.ndarray, num_classes: int = 6) -> np.ndarray:
    """Convert integer grid to one-hot tensor for CNN input.
    
    grid: (40, 40) with integer class indices 0–5
    Returns: (6, 40, 40) float32 tensor
    """
    H, W = grid.shape
    one_hot = np.zeros((num_classes, H, W), dtype=np.float32)
    for k in range(num_classes):
        one_hot[k] = (grid == k).astype(np.float32)
    return one_hot
```

Can augment the input with additional feature channels (distance transforms, neighbor counts) to give the CNN a head start:

```python
def encode_grid_rich(grid: np.ndarray, num_classes: int = 6) -> np.ndarray:
    """Rich input encoding: one-hot + distance features.
    
    Returns: (12, 40, 40) — 6 one-hot + 6 distance channels.
    """
    one_hot = encode_grid_for_cnn(grid, num_classes)  # (6, 40, 40)
    
    distances = compute_distance_features(grid, num_classes)  # (40, 40, 6)
    distances = distances.transpose(2, 0, 1) / 40.0  # Normalize, (6, 40, 40)
    
    return np.concatenate([one_hot, distances], axis=0)  # (12, 40, 40)
```

### 3.4 Model Ranking (Effort-to-Impact)

| Rank | Model | Expected Score Improvement* | Implementation Effort |
|------|-------|---------------------------|----------------------|
| 1 | k-NN (31 features, k=30) | Strong baseline | 1 hour |
| 2 | GBT × 6 regressors | +2-5 points over k-NN | 2 hours |
| 3 | GBT + context-aware features | +1-3 points over plain GBT | 3 hours |
| 4 | Ensemble (GBT + k-NN + Ridge) | +1-2 points over best single | 4 hours |
| 5 | Small U-Net with augmentation | Uncertain; may not beat GBT | 8+ hours |

*Relative to the flat empirical baseline (`EMPIRICAL_TRANSITIONS` in `strategies.py` which scores ~60-70 on backtests).

### 3.5 Ensemble Strategy

Given the KL-optimal averaging result (Section 1.5), the optimal ensemble is the arithmetic mean:

```python
def ensemble_predict(model_predictions: list[np.ndarray]) -> np.ndarray:
    """Average predictions from multiple models.
    
    model_predictions: list of (N, 6) probability arrays
    Returns: (N, 6) averaged probabilities
    """
    stacked = np.stack(model_predictions, axis=0)  # (M, N, 6)
    avg = stacked.mean(axis=0)                      # (N, 6)
    avg = np.clip(avg, 1e-6, 1.0)
    avg /= avg.sum(axis=1, keepdims=True)
    return avg
```

Can also use **stacking**: train a meta-model on the per-model predictions using LOROCV splits. But with only 16 rounds, stacking risks overfitting the meta-model.

---

## 4. Data Augmentation and Regularization

### 4.1 D4 Symmetry Group (8× Data)

The terrain simulation has no preferred orientation — rotating or flipping a grid produces an equally valid scenario. The D4 symmetry group consists of 8 transformations:

| Transform | Matrix | Description |
|-----------|--------|-------------|
| e | I | Identity |
| r | rot90 | 90° rotation |
| r² | rot180 | 180° rotation |
| r³ | rot270 | 270° rotation |
| s | flipH | Horizontal flip |
| sr | flipH + rot90 | |
| sr² | flipH + rot180 | Equivalent to vertical flip |
| sr³ | flipH + rot270 | |

```python
def generate_d4_augmentations(grid: np.ndarray, gt: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate all 8 D4 transformations of a (grid, ground_truth) pair.
    
    grid: (40, 40) integer grid
    gt: (40, 40, 6) probability tensor
    
    Returns: list of 8 (grid, gt) pairs
    """
    augmented = []
    for k in range(4):
        g = np.rot90(grid, k=k)
        t = np.rot90(gt, k=k, axes=(0, 1))  # Rotate spatial dims only
        augmented.append((g.copy(), t.copy()))
        
        # Add horizontal flip
        g_flip = np.flip(g, axis=1)
        t_flip = np.flip(t, axis=1)
        augmented.append((g_flip.copy(), t_flip.copy()))
    
    return augmented
```

**Quantitative benefit**: 90 fixtures × 8 = 720 augmented training examples for CNN models. For per-cell models, this also multiplies the cell count to 1,152,000.

**Caveat**: D4-augmented cells from the same grid are highly correlated. For cross-validation, all 8 augmentations of a grid must go into the same fold (otherwise information leaks between train and validation).

### 4.2 Patch Extraction

For CNNs, extract overlapping patches from the 40×40 grid to further multiply data:

```python
def extract_patches(grid: np.ndarray, gt: np.ndarray, 
                    patch_size: int = 20, stride: int = 10) -> list[tuple]:
    """Extract overlapping patches from grid and ground truth.
    
    40×40 with patch_size=20, stride=10 → 9 patches per grid.
    """
    H, W = grid.shape
    patches = []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            g_patch = grid[i:i+patch_size, j:j+patch_size].copy()
            t_patch = gt[i:i+patch_size, j:j+patch_size, :].copy()
            patches.append((g_patch, t_patch))
    return patches
```

With D4 + patches: 90 × 8 × 9 = **5,184** training patches of size 20×20. Significant, but patches have heavy overlap (correlated).

**Recommendation**: Use patch extraction only if CNN performance plateaus with full-grid D4 augmentation.

### 4.3 Cross-Round Regularization

The biggest risk is overfitting to round-specific hidden parameters. Regularization strategies:

#### 4.3.1 Mixup (Inter-Grid Interpolation)

```python
def mixup_grids(gt1: np.ndarray, gt2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Mix two ground-truth tensors (probability interpolation).
    
    Conceptually: create a "virtual round" that's a blend of two real rounds.
    Only for ground truth — input grids can't be meaningfully interpolated (they're integers).
    """
    lam = np.random.beta(alpha, alpha)
    return lam * gt1 + (1 - lam) * gt2
```

**Warning**: Mixup works well for soft targets but input grids are discrete (integer terrain types). Options:
1. Mix only the soft targets, keeping one input grid → acts as label smoothing
2. Mix the one-hot encoded inputs → creates "fractional terrain" which is unphysical but may help regularize
3. Use CutMix instead (see below)

#### 4.3.2 CutMix (Spatial Mixing)

```python
def cutmix_grids(grid1, gt1, grid2, gt2, alpha=1.0):
    """Cut a rectangle from grid2 and paste into grid1.
    
    More natural than Mixup for spatial data — maintains spatial coherence.
    """
    H, W = grid1.shape
    lam = np.random.beta(alpha, alpha)
    
    # Random box
    cut_h = int(H * np.sqrt(1 - lam))
    cut_w = int(W * np.sqrt(1 - lam))
    cx, cy = np.random.randint(H), np.random.randint(W)
    x1, y1 = max(0, cx - cut_h // 2), max(0, cy - cut_w // 2)
    x2, y2 = min(H, cx + cut_h // 2), min(W, cy + cut_w // 2)
    
    grid_mixed = grid1.copy()
    gt_mixed = gt1.copy()
    grid_mixed[x1:x2, y1:y2] = grid2[x1:x2, y1:y2]
    gt_mixed[x1:x2, y1:y2] = gt2[x1:x2, y1:y2]
    
    return grid_mixed, gt_mixed
```

CutMix is more natural for spatial data: it creates grids with regions from different rounds, forcing the model to handle diverse local contexts.

### 4.4 Effective Dataset Size Summary

| Augmentation | Multiplier | Effective Grid Count | Effective Cell Count |
|-------------|-----------|---------------------|---------------------|
| None | 1× | 90 | 144,000 |
| D4 symmetry | 8× | 720 | 1,152,000 |
| D4 + patches (20×20, stride 10) | 72× | 6,480* | 2,592,000* |
| D4 + CutMix (pairwise) | variable | ~2,000+ | ~3,200,000+ |

*Patches are smaller (20×20 = 400 cells each), so total cell count formula differs.

**For per-cell models**: Even without augmentation, 144,000 cells is sufficient for all traditional ML models. D4 augmentation helps primarily by increasing diversity of neighborhood contexts.

**For CNNs**: D4 augmentation (720 grids) is the minimum viable dataset. Add CutMix for further regularization.

### 4.5 Regularization Summary

| Technique | Applies To | Effect | Recommended? |
|-----------|-----------|--------|-------------|
| L2 regularization (weight decay) | All models | Shrinks parameters toward zero | Yes, always |
| Dropout (0.1–0.3) | Neural networks | Prevents co-adaptation | Yes for CNN |
| Early stopping | Neural networks, GBT | Prevents overfitting | Yes, always |
| `min_samples_leaf` | Trees | Prevents tiny leaves | Yes (20–50) |
| `max_depth` limit | Trees | Limits complexity | Yes (4–6) |
| GroupNorm | CNN | Stable normalization for small batches | Yes (replaces BN) |
| D4 augmentation | All | 8× data | Yes, always |
| CutMix | CNN | Cross-round spatial mixing | Try after D4 |

---

## 5. Handling Hidden Parameter Variation

### 5.1 The Core Challenge

Each round has hidden simulation parameters (not directly observable) that dramatically affect transition probabilities. Evidence from `strategies.py`:

| Parameter Regime | Settlement Survival | Forest Spread | Description |
|-----------------|-------------------|---------------|-------------|
| COLLAPSE | ~6.5% | Low | Settlements die, forests thin |
| MIDDLE | ~22% | Moderate | Balanced dynamics |
| THRIVING | ~43.6% | High | Settlements persist, forests grow |

The settlement survival rate varies by **~7×** between collapse and thriving regimes. This is the **dominant source of prediction error** — the variance between rounds far exceeds the variance within a round for a given context.

### 5.2 Regime Detection from Initial Grid

The initial grid carries weak signals about hidden parameters:

```python
def estimate_regime_features(grid: np.ndarray) -> dict:
    """Extract features that correlate with hidden parameters.
    
    These are proxies — the initial grid doesn't directly reveal the parameters,
    but certain patterns are more likely under certain regimes.
    """
    H, W = grid.shape
    total = H * W
    
    # Class fractions
    settlement_frac = (grid == 1).sum() / total
    forest_frac = (grid == 4).sum() / total
    ocean_frac = (grid == 0).sum() / total  # After mapping OCEAN→EMPTY
    mountain_frac = (grid == 5).sum() / total
    
    # Settlement clustering
    settlement_mask = (grid == 1).astype(float)
    kernel = np.ones((3, 3)); kernel[1, 1] = 0
    settlement_neighbors = ndimage.convolve(settlement_mask, kernel, mode='constant')
    avg_settlement_clustering = settlement_neighbors[grid == 1].mean() if (grid == 1).any() else 0
    
    return {
        'settlement_frac': settlement_frac,
        'forest_frac': forest_frac,
        'ocean_frac': ocean_frac,
        'mountain_frac': mountain_frac,
        'settlement_clustering': avg_settlement_clustering,
    }
```

**Important caveat**: The initial grid is generated **before** the simulation runs, so it may not directly reflect the hidden parameters. The regime affects the *transition dynamics*, not the initial state. However, some correlation may exist if the game generates initial states that are consistent with the regime.

### 5.3 Cross-Round Variance Analysis

The existing codebase handles this with **Dirichlet shrinkage**:

```python
# From strategies.py — RegimePosterior
class RegimePosterior:
    """Beta posterior for settlement survival rate, used to detect regime."""
    def __init__(self):
        self.alpha = 1.0  # Prior: uniform
        self.beta = 1.0
    
    def update(self, survived: int, total: int):
        self.alpha += survived
        self.beta += (total - survived)
    
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
```

And with regime-specific transition tables that blend predictions:

```python
def blend_regime_predictions(predictions_by_regime: dict[str, np.ndarray],
                             regime_weights: dict[str, float]) -> np.ndarray:
    """Blend predictions from different regime-specific models.
    
    predictions_by_regime: {'collapse': (6,), 'middle': (6,), 'thriving': (6,)}
    regime_weights: {'collapse': 0.2, 'middle': 0.5, 'thriving': 0.3}
    """
    total_weight = sum(regime_weights.values())
    blended = sum(
        (w / total_weight) * predictions_by_regime[regime]
        for regime, w in regime_weights.items()
    )
    return blended
```

### 5.4 Strategies for Robust Prediction Without Regime Knowledge

At test time, we don't know the round's regime. Options:

#### Strategy A: Marginalize Over Regimes (Recommended)

Train one model across all rounds. The model implicitly averages over regimes, producing predictions that hedge across scenarios. This is **mathematically optimal under KL** when regime probabilities are unknown (Section 1.5 — arithmetic mean minimizes expected KL).

#### Strategy B: Train Per-Regime Models + Blend

1. Cluster historical rounds into regimes (e.g., k=3 clusters using settlement survival rate from ground truth)
2. Train separate models per regime
3. At test time, predict with uniform regime weights: `q = (1/3)(q_collapse + q_middle + q_thriving)`

**Advantage**: Each per-regime model sees more coherent data, potentially fitting better.
**Disadvantage**: Each model sees only ~30 rounds (90/3), and regime clustering itself introduces noise.

#### Strategy C: Include Round-Level Features

Add global grid features (class fractions, clustering metrics) as input features. The model can then learn to condition on regime proxies:

```python
# Feature vector per cell includes:
# [...local_features..., settlement_frac, forest_frac, ocean_frac, ...]
```

This lets the model learn regime-dependent behavior implicitly without explicit clustering.

**Recommendation**: Use Strategy A (marginalize) as the baseline, augment with Strategy C (round-level features). Strategy B is worth trying if Strategy C doesn't help, but the data split concern is significant.

### 5.5 Quantifying Cross-Round Variance

From the existing `CONTEXT_TRANSITIONS` and `REGIME_*_TRANSITIONS` tables, we can compute the variance of predictions across rounds for each context:

```python
def compute_cross_round_variance(fixtures: list[dict]) -> dict:
    """Compute per-context variance across rounds.
    
    This tells us how much predictions should vary due to hidden parameters.
    """
    from collections import defaultdict
    context_distributions = defaultdict(list)
    
    for fixture in fixtures:
        grid = np.array(fixture['initial_grid'])
        gt = np.array(fixture['ground_truth'])
        
        for x in range(40):
            for y in range(40):
                context = extract_context(grid, x, y)
                context_distributions[context].append(gt[x, y])
    
    variances = {}
    for context, dists in context_distributions.items():
        dists = np.array(dists)  # (N, 6)
        variances[context] = {
            'mean': dists.mean(axis=0),
            'std': dists.std(axis=0),
            'n': len(dists),
        }
    
    return variances
```

**Expected finding**: Contexts involving settlements will show the highest cross-round variance (due to regime-dependent survival rates). Empty and mountain cells will show low variance (stable across regimes).

---

## 6. Loss Function and Calibration

### 6.1 Matching the Scoring Metric

The competition scores via:

```
score = 100 × exp(-3 × weighted_KL)
weighted_KL = Σ_i H(p_i) × KL(p_i || q_i) / Σ_i H(p_i)
```

**Training loss options ranked by alignment with scoring:**

| Loss | Formula | Alignment | Notes |
|------|---------|-----------|-------|
| Entropy-weighted KL | `Σ H(p_i) KL(p_i\|\|q_i) / Σ H(p_i)` | Perfect | Exact match to scoring metric |
| Soft cross-entropy | `-Σ p_k log q_k` | Near-perfect | ≡ KL + constant |
| MSE per class | `Σ (p_k - q_k)²` | Good | Not KL-optimal but practical for trees |
| Brier score | `Σ (p_k - q_k)²` | Good | Same as MSE for probabilities |

**For neural networks**: Use soft cross-entropy directly:

```python
def soft_cross_entropy_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy for training with probability vector targets.
    
    logits: (B, K, H, W) raw model output before softmax
    soft_targets: (B, K, H, W) probability vectors summing to 1 along K
    """
    log_probs = F.log_softmax(logits, dim=1)  # Along class dimension
    loss = -(soft_targets * log_probs).sum(dim=1)  # Sum over classes → (B, H, W)
    return loss.mean()
```

**For tree models**: Train with MSE (default for `HistGradientBoostingRegressor`), then calibrate post-hoc. MSE-trained trees produce predictions that minimize L2 distance to target probabilities, which is close to but not identical to KL-optimal.

### 6.2 Entropy-Weighted Training Loss

Since the scoring metric weights by ground-truth entropy, we can weight our training loss similarly:

```python
def entropy_weighted_soft_ce(logits, soft_targets):
    """Weight cross-entropy by ground-truth entropy (matching scoring metric).
    
    Cells with higher ground-truth entropy get more weight in training.
    """
    log_probs = F.log_softmax(logits, dim=1)
    per_cell_ce = -(soft_targets * log_probs).sum(dim=1)  # (B, H, W)
    
    # Compute entropy weights
    eps = 1e-10
    gt_entropy = -(soft_targets * torch.log(soft_targets + eps)).sum(dim=1)  # (B, H, W)
    
    # Weight CE by entropy
    weighted_loss = (gt_entropy * per_cell_ce).sum() / (gt_entropy.sum() + eps)
    return weighted_loss
```

**Caveat**: At test time, we don't know `H(p_i)`. But training with entropy weights teaches the model to focus capacity on uncertain cells, which is where scoring impact is highest.

### 6.3 Probability Floors

The competition mandates minimum probability floors to prevent infinite KL divergence:

```python
# From prediction.py
PROBABILITY_FLOOR = 0.01
IMPOSSIBLE_FLOOR = 0.001

def apply_probability_floors(probs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Apply class-specific probability floors.
    
    - Mountain on non-mountain cell: floor 0.001 (nearly impossible)
    - Port on non-coastal cell: floor 0.001 (impossible)
    - All other classes: floor 0.01
    
    probs: (40, 40, 6) probability tensor
    grid: (40, 40) initial grid (for coastal detection)
    """
    H, W, K = probs.shape
    floored = probs.copy()
    
    coastal_mask = compute_coastal_mask(grid)
    mountain_mask = (grid == 5)
    
    for x in range(H):
        for y in range(W):
            for k in range(K):
                if k == 5 and not mountain_mask[x, y]:  # Mountain on non-mountain
                    floored[x, y, k] = max(floored[x, y, k], IMPOSSIBLE_FLOOR)
                elif k == 2 and not coastal_mask[x, y]:  # Port on non-coastal
                    floored[x, y, k] = max(floored[x, y, k], IMPOSSIBLE_FLOOR)
                else:
                    floored[x, y, k] = max(floored[x, y, k], PROBABILITY_FLOOR)
            
            # Renormalize
            floored[x, y] /= floored[x, y].sum()
    
    return floored
```

**Key insight**: Floors are applied **after** model prediction, so the model doesn't need to learn floors — they're enforced post-hoc. However, the floor-then-renormalize step slightly distorts well-calibrated predictions. Train the model to predict pre-floor probabilities, then apply floors.

### 6.4 Temperature Scaling

Temperature scaling adjusts prediction confidence without changing the ranking:

```
q_T[k] = softmax(z[k] / T) = exp(z[k]/T) / Σ_j exp(z[j]/T)
```

- `T > 1`: Softens predictions (more uniform) → helps if model is overconfident
- `T < 1`: Sharpens predictions → helps if model is underconfident
- `T = 1`: No change

**No closed-form solution for optimal T**. Must optimize via gradient descent on validation KL:

```python
from scipy.optimize import minimize_scalar

def find_optimal_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Find T that minimizes KL divergence on validation set.
    
    logits: (N, K) raw model outputs (before softmax)
    targets: (N, K) ground-truth probability vectors
    """
    def kl_at_temperature(T):
        # Apply temperature scaling
        scaled = logits / T
        # Softmax
        exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-10, 1.0)
        
        # KL divergence
        kl = np.sum(targets * np.log(targets / probs + 1e-10), axis=1)
        return kl.mean()
    
    result = minimize_scalar(kl_at_temperature, bounds=(0.1, 10.0), method='bounded')
    return result.x
```

**Empirical finding from existing codebase**: `T ≈ 1.16–1.20` was optimal in backtests, suggesting the current models are slightly overconfident. This is expected — small training sets typically produce overconfident predictions.

### 6.5 Calibration Post-Processing Pipeline

The full post-processing pipeline, applied after raw model prediction:

```
Raw prediction → Temperature scaling → Probability floors → Renormalization
```

```python
def calibrate_predictions(raw_probs: np.ndarray, grid: np.ndarray, 
                          temperature: float = 1.18) -> np.ndarray:
    """Full calibration pipeline.
    
    raw_probs: (40, 40, 6) model output probabilities
    grid: (40, 40) initial grid
    temperature: Learned on validation set
    """
    # Step 1: Temperature scaling (if model outputs logits, apply here)
    # For probability outputs, convert to logits first
    logits = np.log(np.clip(raw_probs, 1e-10, 1.0))
    scaled_logits = logits / temperature
    
    # Softmax
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=-1, keepdims=True))
    calibrated = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    
    # Step 2: Apply probability floors
    calibrated = apply_probability_floors(calibrated, grid)
    
    return calibrated
```

### 6.6 Calibration Quality Metrics

```python
def reliability_diagram(predictions: np.ndarray, targets: np.ndarray, n_bins: int = 10):
    """Compute reliability diagram data for assessing calibration.
    
    For each predicted probability bin, compute the actual frequency.
    Well-calibrated: predicted probability ≈ observed frequency.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracies.append(targets[mask].mean())
            bin_confidences.append(predictions[mask].mean())
            bin_counts.append(mask.sum())
    
    return bin_confidences, bin_accuracies, bin_counts

def expected_calibration_error(predictions: np.ndarray, targets: np.ndarray, 
                                n_bins: int = 10) -> float:
    """ECE: weighted average of |accuracy - confidence| across bins."""
    confs, accs, counts = reliability_diagram(predictions, targets, n_bins)
    total = sum(counts)
    ece = sum(c / total * abs(a - f) for f, a, c in zip(confs, accs, counts))
    return ece
```

---

## 7. Evaluation and Validation

### 7.1 Leave-One-Round-Out Cross-Validation (LOROCV)

The gold standard for this problem. Each round has ~5 seeds (fixtures), so we have ~16 rounds as folds:

```python
from collections import defaultdict

def lorocv(fixtures: list[dict], train_fn, predict_fn):
    """Leave-one-round-out cross-validation.
    
    fixtures: list of {'round_id': str, 'seed': int, 'initial_grid': ..., 'ground_truth': ...}
    train_fn: (X_train, y_train) → model
    predict_fn: (model, X_test) → predictions
    
    Returns: dict of round_id → score
    """
    # Group by round
    rounds = defaultdict(list)
    for f in fixtures:
        rounds[f['round_id']].append(f)
    
    round_scores = {}
    for held_out_round in rounds:
        # Train on all other rounds
        train_fixtures = [f for rid, fs in rounds.items() if rid != held_out_round for f in fs]
        test_fixtures = rounds[held_out_round]
        
        X_train, y_train = extract_features_and_targets(train_fixtures)
        model = train_fn(X_train, y_train)
        
        # Evaluate on held-out round
        scores = []
        for fixture in test_fixtures:
            X_test = extract_features(fixture)
            predictions = predict_fn(model, X_test)
            predictions = calibrate_predictions(predictions, fixture['initial_grid'])
            score = competition_score(fixture['ground_truth'], predictions)
            scores.append(score)
        
        round_scores[held_out_round] = np.mean(scores)
    
    return round_scores
```

**Why LOROCV, not random CV**: Cells within the same round share hidden parameters and are correlated. Random CV would leak information about the round's hidden parameters from training cells to test cells in the same round, giving an overly optimistic estimate.

**Expected behavior**: LOROCV scores will be **lower** than random CV scores. The gap between them measures how much the model overfits to round-specific patterns.

### 7.2 Per-Cell Error Analysis

Not all cells are equal — error analysis should focus on where mistakes hurt most:

```python
def per_class_error_analysis(predictions: np.ndarray, ground_truth: np.ndarray, 
                              grid: np.ndarray) -> dict:
    """Break down KL divergence by initial terrain class.
    
    Returns dict of initial_class → mean KL divergence for cells of that class.
    """
    H, W = grid.shape
    class_errors = defaultdict(list)
    
    for x in range(H):
        for y in range(W):
            initial_class = grid[x, y]
            p = ground_truth[x, y]
            q = predictions[x, y]
            
            # KL divergence for this cell
            kl = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
            entropy = -np.sum(p * np.log(p + 1e-10))
            
            class_errors[initial_class].append({
                'kl': kl,
                'entropy': entropy,
                'weighted_kl': entropy * kl,
            })
    
    summary = {}
    for cls, errors in class_errors.items():
        summary[cls] = {
            'count': len(errors),
            'mean_kl': np.mean([e['kl'] for e in errors]),
            'mean_entropy': np.mean([e['entropy'] for e in errors]),
            'mean_weighted_kl': np.mean([e['weighted_kl'] for e in errors]),
            'total_weighted_kl': np.sum([e['weighted_kl'] for e in errors]),
        }
    
    return summary
```

**Expected high-error classes** (from existing backtest results):
1. **Settlements** — highest entropy (most uncertain), most sensitive to regime
2. **Empty cells** — many of them; individually low KL but collectively significant
3. **Ports** — rare (81 in training data), high uncertainty, hard to predict

### 7.3 Baseline Comparisons

Every model should be compared against these baselines:

| Baseline | Description | Expected Score |
|----------|-------------|---------------|
| **Uniform** | `q = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]` | ~20-30 |
| **Marginal** | Global class frequency across all training data | ~40-50 |
| **Empirical transitions** | Per-class transition table (no context) | ~60-70 |
| **Context transitions** | 4-feature context key (current baseline) | ~70-80 |

A supervised model must beat "Context transitions" to justify the added complexity.

### 7.4 Overfitting Detection

```python
def detect_overfitting(train_scores: list[float], lorocv_scores: list[float]) -> dict:
    """Compare in-sample fit vs LOROCV generalization.
    
    Overfitting indicators:
    - Large gap between train and CV scores
    - High variance across CV folds
    - Score degrades with more complex models
    """
    train_mean = np.mean(train_scores)
    cv_mean = np.mean(lorocv_scores)
    cv_std = np.std(lorocv_scores)
    
    gap = train_mean - cv_mean
    
    return {
        'train_mean': train_mean,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'gap': gap,
        'gap_pct': gap / train_mean * 100,
        'is_overfitting': gap > 5.0,  # 5-point gap is concerning
        'high_variance': cv_std > 10.0,  # High variance across rounds
    }
```

**Rule of thumb**: If `train_score - cv_score > 5 points`, the model is likely overfitting to round-specific patterns. Increase regularization or reduce model complexity.

### 7.5 Statistical Significance

With only ~16 rounds as CV folds, statistical power is limited:

```python
from scipy import stats

def paired_comparison(scores_a: list[float], scores_b: list[float]) -> dict:
    """Paired t-test comparing two models across LOROCV folds.
    
    scores_a, scores_b: Score on each fold for model A and B.
    """
    differences = np.array(scores_a) - np.array(scores_b)
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    return {
        'mean_diff': differences.mean(),
        'std_diff': differences.std(),
        't_stat': t_stat,
        'p_value': p_value,
        'significant_at_05': p_value < 0.05,
        'significant_at_10': p_value < 0.10,
    }
```

With 16 folds, a paired t-test has limited power. Differences of <2 points are likely noise. Focus on **consistent directional improvement** across folds rather than aggregate score differences.

---

## 8. Implementation Blueprint

### 8.1 End-to-End Pipeline

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Load Fixtures   │────▶│  Preprocess  │────▶│ Feature Extract │
│  (90 JSON files) │     │  Grid → 6cls │     │ (31 features)   │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                       │
                         ┌──────────────┐              │
                         │  LOROCV      │◀─────────────┤
                         │  (16 folds)  │              │
                         └──────┬───────┘              │
                                │                      │
                    ┌───────────▼───────────┐          │
                    │  Train Models         │          │
                    │  (GBT/kNN/Ridge/...)  │◀─────────┘
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Calibrate            │
                    │  (Temperature + Floor)│
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Evaluate (Score)     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Select Best Model   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Retrain on All Data  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Save Model Artifact  │
                    │  (pickle/joblib)      │
                    └──────────────────────┘
```

### 8.2 Implementation Phases

#### Phase 1: Data Loading and Feature Extraction (Day 1)

```python
import json
import numpy as np
from pathlib import Path

# Constants
TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
GRID_SIZE = 40

def load_fixtures(fixture_dir: Path) -> list[dict]:
    """Load all fixture files."""
    fixtures = []
    for path in sorted(fixture_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        
        # Map terrain codes to prediction classes
        raw_grid = np.array(data['initial_grid'])
        grid = np.vectorize(TERRAIN_TO_CLASS.get)(raw_grid)
        gt = np.array(data['ground_truth'])
        
        # Parse round_id from filename (e.g., "round_123_seed0.json")
        parts = path.stem.split('_seed')
        round_id = parts[0]
        seed = int(parts[1]) if len(parts) > 1 else 0
        
        fixtures.append({
            'round_id': round_id,
            'seed': seed,
            'grid': grid,           # (40, 40) int, 0-5
            'ground_truth': gt,     # (40, 40, 6) float
            'path': str(path),
        })
    
    return fixtures

def extract_cell_features(grid: np.ndarray) -> np.ndarray:
    """Extract full feature matrix for all cells in a grid.
    
    Returns: (1600, 31) feature matrix
    """
    H, W = grid.shape
    
    # One-hot encoding of cell class
    one_hot = np.zeros((H, W, NUM_CLASSES))
    for k in range(NUM_CLASSES):
        one_hot[:, :, k] = (grid == k).astype(np.float32)
    
    # Neighbor counts
    nbr_counts = count_neighbors_by_class(grid, NUM_CLASSES)  # (40, 40, 6)
    
    # Distance transforms
    distances = compute_distance_features(grid, NUM_CLASSES)   # (40, 40, 6)
    distances /= GRID_SIZE  # Normalize
    
    # Local entropy
    local_ent = local_class_entropy(grid, window_size=5)       # (40, 40)
    
    # Position features
    pos = positional_features(H, W)                             # (40, 40, 4)
    
    # Global features (broadcast)
    global_feats = global_grid_features(grid)                   # (8,)
    global_broadcast = np.tile(global_feats, (H, W, 1))        # (40, 40, 8)
    
    # Concatenate all
    features = np.concatenate([
        one_hot,                           # 6
        nbr_counts / 8.0,                 # 6 (normalized)
        distances,                         # 6
        local_ent[:, :, np.newaxis],       # 1
        pos,                               # 4
        global_broadcast,                  # 8
    ], axis=-1)  # (40, 40, 31)
    
    return features.reshape(H * W, -1)  # (1600, 31)

def extract_cell_targets(gt: np.ndarray) -> np.ndarray:
    """Flatten ground truth to per-cell targets.
    
    Returns: (1600, 6)
    """
    return gt.reshape(-1, NUM_CLASSES)
```

#### Phase 2: Baseline Models (Day 1-2)

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

class PerCellModelEnsemble:
    """Ensemble of K=6 regressors for soft probability prediction."""
    
    def __init__(self, model_type='gbt'):
        self.model_type = model_type
        self.models = []
        self.scaler = StandardScaler()
        
    def _create_model(self):
        if self.model_type == 'gbt':
            return HistGradientBoostingRegressor(
                max_iter=200, max_depth=5, learning_rate=0.05,
                min_samples_leaf=30, l2_regularization=1.0, max_bins=64,
            )
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0)
        elif self.model_type == 'knn':
            return KNeighborsRegressor(
                n_neighbors=30, weights='distance', algorithm='ball_tree',
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y_soft: np.ndarray):
        """Train K=6 models on soft probability targets."""
        X_scaled = self.scaler.fit_transform(X)
        self.models = []
        for k in range(NUM_CLASSES):
            model = self._create_model()
            model.fit(X_scaled, y_soft[:, k])
            self.models.append(model)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probability vectors and renormalize."""
        X_scaled = self.scaler.transform(X)
        probs = np.stack([m.predict(X_scaled) for m in self.models], axis=1)
        probs = np.clip(probs, 1e-6, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs
```

#### Phase 3: LOROCV Evaluation (Day 2)

```python
def run_lorocv_evaluation(fixtures, model_type='gbt'):
    """Full LOROCV evaluation pipeline."""
    from collections import defaultdict
    
    # Group by round
    rounds = defaultdict(list)
    for f in fixtures:
        rounds[f['round_id']].append(f)
    
    all_round_ids = sorted(rounds.keys())
    results = []
    
    for held_out in all_round_ids:
        # Prepare training data
        train_X, train_y = [], []
        for rid in all_round_ids:
            if rid == held_out:
                continue
            for f in rounds[rid]:
                X = extract_cell_features(f['grid'])
                y = extract_cell_targets(f['ground_truth'])
                train_X.append(X)
                train_y.append(y)
        
        train_X = np.vstack(train_X)
        train_y = np.vstack(train_y)
        
        # Train model
        model = PerCellModelEnsemble(model_type=model_type)
        model.fit(train_X, train_y)
        
        # Find optimal temperature on training data (or use fixed T)
        temperature = 1.18  # From existing backtests
        
        # Evaluate on held-out round
        round_scores = []
        for f in rounds[held_out]:
            X_test = extract_cell_features(f['grid'])
            raw_probs = model.predict(X_test)
            raw_probs = raw_probs.reshape(GRID_SIZE, GRID_SIZE, NUM_CLASSES)
            
            calibrated = calibrate_predictions(raw_probs, f['grid'], temperature)
            score = competition_score(f['ground_truth'], calibrated)
            round_scores.append(score)
        
        avg_score = np.mean(round_scores)
        results.append({
            'round_id': held_out,
            'score': avg_score,
            'n_seeds': len(round_scores),
        })
        print(f"  Round {held_out}: score = {avg_score:.2f} ({len(round_scores)} seeds)")
    
    # Summary
    scores = [r['score'] for r in results]
    print(f"\nLOROCV Results ({model_type}):")
    print(f"  Mean:  {np.mean(scores):.2f}")
    print(f"  Std:   {np.std(scores):.2f}")
    print(f"  Min:   {np.min(scores):.2f}")
    print(f"  Max:   {np.max(scores):.2f}")
    
    return results
```

#### Phase 4: Hyperparameter Tuning (Day 3)

```python
def tune_hyperparameters(fixtures):
    """Grid search over key hyperparameters using LOROCV."""
    
    param_grid = {
        'gbt': {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_regularization': [0.1, 1.0, 10.0],
            'min_samples_leaf': [10, 20, 50],
        },
        'knn': {
            'n_neighbors': [5, 10, 20, 30, 50],
        },
        'ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    }
    
    # For each combination, run LOROCV
    # Track: mean score, std, and per-round scores
    # Select: combination with highest mean LOROCV score
    # Tiebreak: prefer lower variance (more consistent across rounds)
    pass  # Implementation follows standard grid search patterns
```

#### Phase 5: Final Model Training (Day 3)

```python
def train_final_model(fixtures, model_type='gbt', temperature=1.18):
    """Train on ALL data with best hyperparameters. Save for deployment."""
    import joblib
    
    # Extract features and targets from all fixtures
    all_X, all_y = [], []
    for f in fixtures:
        X = extract_cell_features(f['grid'])
        y = extract_cell_targets(f['ground_truth'])
        all_X.append(X)
        all_y.append(y)
    
    all_X = np.vstack(all_X)
    all_y = np.vstack(all_y)
    
    # Train final model
    model = PerCellModelEnsemble(model_type=model_type)
    model.fit(all_X, all_y)
    
    # Save
    artifact = {
        'model': model,
        'temperature': temperature,
        'model_type': model_type,
        'n_fixtures': len(fixtures),
        'n_cells': len(all_X),
    }
    
    joblib.dump(artifact, 'model_artifact.joblib')
    print(f"Saved model: {model_type}, trained on {len(all_X)} cells from {len(fixtures)} fixtures")
    
    return artifact
```

### 8.3 Prediction for New Grids

```python
def predict_new_grid(grid_raw: list[list[int]], artifact_path: str = 'model_artifact.joblib') -> np.ndarray:
    """Predict probability tensor for a new unseen grid.
    
    grid_raw: 40×40 list of terrain codes (0-5, 10, 11)
    Returns: (40, 40, 6) probability tensor
    """
    import joblib
    
    artifact = joblib.load(artifact_path)
    model = artifact['model']
    temperature = artifact['temperature']
    
    # Map terrain codes to classes
    grid = np.vectorize(TERRAIN_TO_CLASS.get)(np.array(grid_raw))
    
    # Extract features
    X = extract_cell_features(grid)  # (1600, 31)
    
    # Predict
    raw_probs = model.predict(X)  # (1600, 6)
    raw_probs = raw_probs.reshape(GRID_SIZE, GRID_SIZE, NUM_CLASSES)
    
    # Calibrate
    final_probs = calibrate_predictions(raw_probs, grid, temperature)
    
    return final_probs
```

### 8.4 Computational Budget

| Phase | Time (est.) | Compute |
|-------|-------------|---------|
| Load fixtures | <1s | Disk I/O |
| Feature extraction (all fixtures) | ~5s | CPU (scipy distance transforms) |
| GBT training (6 models) | ~30s | CPU |
| k-NN training (6 models) | ~10s | CPU |
| Ridge training (6 models) | <1s | CPU |
| LOROCV (16 folds × GBT) | ~8 min | CPU |
| Temperature calibration | ~5s | CPU |
| Prediction for new grid | ~1s | CPU |
| **Total pipeline (one run)** | **~10 min** | **CPU only** |

For CNN/U-Net: add ~30 min training with GPU, or ~2-3 hours CPU-only.

### 8.5 Dependencies

```toml
# pyproject.toml additions
[project]
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "joblib>=1.3",
]

[project.optional-dependencies]
deep = [
    "torch>=2.0",
]
```

No GPU required for the per-cell pipeline. PyTorch only needed if pursuing the U-Net approach (Phase 5+).

### 8.6 File Structure

```
supervised_learning_from_historical_3/
├── apps/astar-island/src/astar_island/
│   ├── solver.py           # Main prediction entry point
│   ├── features.py         # Feature extraction (Section 2)
│   ├── models.py           # Model definitions (Section 3)
│   ├── calibration.py      # Temperature scaling, floors (Section 6)
│   ├── evaluation.py       # LOROCV, scoring, analysis (Section 7)
│   └── pipeline.py         # End-to-end training pipeline (Section 8)
├── data/
│   └── fixtures/           # Symlink or copy from sibling project
├── models/                 # Saved model artifacts
├── docs/
│   ├── reference.md        # Problem specification
│   └── research.md         # This document
└── scripts/
    ├── train.py            # CLI for training
    ├── evaluate.py         # CLI for LOROCV evaluation
    └── predict.py          # CLI for prediction
```

### 8.7 Reusable Code from Sibling Project

The following can be directly imported or adapted from `astar_island_round_2_optimize_new_implementation/`:

| Module | What to Reuse | Adaptation Needed |
|--------|--------------|-------------------|
| `scoring.py` | `kl_divergence_per_cell()`, `competition_score()` | None — use as-is |
| `prediction.py` | `apply_probability_floors()`, `compute_coastal_mask()` | None — use as-is |
| `terrain.py` | `TerrainType`, `PredictionClass`, `TERRAIN_TO_CLASS` | None — use as-is |
| `backtesting/fixtures.py` | `load_fixture()`, `list_fixtures()` | Point to local data dir |
| `backtesting/runner.py` | Backtest framework | Adapt strategy interface |
| `scripts/cross_validate_context.py` | LOROCV skeleton | Replace context model with ML model |

---

## Appendix A: Mathematical Proofs

### A.1 Arithmetic Mean Minimizes Expected KL

**Claim**: Given distributions `p_1, ..., p_m`, the distribution `q*` minimizing `Σ_j KL(p_j || q)` is `q* = (1/m) Σ_j p_j`.

**Proof**:

```
Σ_j KL(p_j || q) = Σ_j Σ_k p_j[k] log(p_j[k] / q[k])
                  = Σ_j Σ_k p_j[k] log p_j[k] - Σ_j Σ_k p_j[k] log q[k]
                  = const - Σ_k (Σ_j p_j[k]) log q[k]
                  = const - m Σ_k q̄[k] log q[k]
```

where `q̄[k] = (1/m) Σ_j p_j[k]`. This is maximized (and hence the KL sum minimized) when `q = q̄` by Gibbs' inequality: for any distribution `r`, `Σ_k r[k] log r[k] ≥ Σ_k r[k] log q[k]`, with equality iff `q = r`. ∎

### A.2 Entropy-Weighted Variant

For weights `w_j = H(p_j)`:

```
Σ_j w_j KL(p_j || q) = const - Σ_k (Σ_j w_j p_j[k]) log q[k]
```

Optimal: `q*[k] = Σ_j w_j p_j[k] / Σ_j w_j` — the entropy-weighted arithmetic mean.

### A.3 Soft Cross-Entropy Equivalence

```
CE(p, q) = -Σ_k p_k log q_k = H(p) + KL(p || q)
```

Since `H(p)` doesn't depend on `q`:

```
argmin_q CE(p, q) = argmin_q KL(p || q)
```

Training with soft cross-entropy is equivalent to training with KL divergence. ∎

---

## Appendix B: Key Quantitative Baselines

From existing `strategies.py` and backtesting results:

### B.1 Empirical Transition Probabilities (Flat)

```
EMPTY    → [0.822, 0.028, 0.004, 0.013, 0.109, 0.024]  (n=53,547)
SETTLEMENT → [0.327, 0.258, 0.019, 0.133, 0.209, 0.054]  (n=2,042)
PORT     → [0.173, 0.062, 0.296, 0.074, 0.296, 0.099]  (n=81)
RUIN     → [0.258, 0.045, 0.011, 0.208, 0.372, 0.106]  (n=1,073)
FOREST   → [0.123, 0.025, 0.004, 0.023, 0.783, 0.042]  (n=15,517)
MOUNTAIN → [0.078, 0.007, 0.001, 0.020, 0.116, 0.778]  (n=1,420)
```

### B.2 Regime-Specific Settlement Survival

| Regime | P(Settlement → Settlement) | Defining Range |
|--------|---------------------------|----------------|
| Collapse | ~6.5% | survival < 15% |
| Middle | ~22% | 15% ≤ survival < 35% |
| Thriving | ~43.6% | survival ≥ 35% |

### B.3 Temperature Scaling

Empirically optimal `T ≈ 1.16–1.20` across backtests, indicating models are slightly overconfident (need softening).
