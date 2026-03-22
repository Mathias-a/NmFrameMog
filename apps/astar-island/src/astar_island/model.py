"""Models: ShrunkenArchetype baseline + PerCellGBDT primary model.

ShrunkenArchetype: KNN-based archetype average with shrinkage toward the global
mean. Simple but effective baseline.

PerCellGBDT: 6 separate HistGradientBoostingRegressors, one per class, trained
on soft probability targets. Uses D4 augmentation (8 transforms of the square
grid) for data amplification.

Both models produce (H, W, 6) probability tensors.  ``predict_grid`` applies
probability floors; ``predict_grid_raw`` returns renormalized but unfloored
predictions (useful for entropy ranking before blending).
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import NearestNeighbors

from astar_island.api import RoundData
from astar_island.features import extract_features
from astar_island.prob import (
    NUM_CLASSES,
    apply_floors,
    make_dynamic_mask,
    renormalize,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for sklearn interfaces (avoids no-any-unimported errors)
# ---------------------------------------------------------------------------


@runtime_checkable
class _NNModel(Protocol):
    def fit(self, x: NDArray[np.float64]) -> object: ...
    def kneighbors(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int_]]: ...


@runtime_checkable
class _GBDTModel(Protocol):
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> object: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


# ---------------------------------------------------------------------------
# D4 augmentation (dihedral group of the square)
# ---------------------------------------------------------------------------


def _d4_transforms(
    grid: NDArray[np.int_],
    probs: NDArray[np.float64],
) -> list[tuple[NDArray[np.int_], NDArray[np.float64]]]:
    """Apply all 8 D4 transformations to (grid, probs) pair.

    D4 = {rot90^k, rot90^k ∘ flip_h} for k in 0..3.
    Returns list of 8 (transformed_grid, transformed_probs) pairs.
    """
    results: list[tuple[NDArray[np.int_], NDArray[np.float64]]] = []
    for k in range(4):
        g = np.rot90(grid, k)
        p = np.rot90(probs, k, axes=(0, 1))
        results.append((g.copy(), p.copy()))

        gf = np.fliplr(g)
        pf = np.fliplr(p)
        results.append((gf.copy(), pf.copy()))

    return results


def _d4_inverse(
    probs: NDArray[np.float64],
    transform_idx: int,
) -> NDArray[np.float64]:
    """Inverse of the i-th D4 transform on a probability tensor.

    Transform i:
        k = i // 2, flip = i % 2
        forward = flip_h? then rot90(k)
        inverse = rot90(4-k) then flip_h?

    Actually forward is rot90(k) then flip_h (if odd index).
    Inverse: un-flip then un-rotate = flip_h (if odd) then rot90(4-k).
    """
    k = transform_idx // 2
    flip = transform_idx % 2

    p = probs
    if flip:
        p = np.fliplr(p)
    p = np.rot90(p, (4 - k) % 4, axes=(0, 1))
    return p.copy()


def tta_predict(
    initial_grid: NDArray[np.int_],
    predict_fn: _PredictFn,
) -> NDArray[np.float64]:
    """Test-time augmentation: average predictions over all 8 D4 transforms.

    Args:
        initial_grid: (H, W) terrain codes.
        predict_fn: Function that takes initial_grid and returns (H, W, 6).

    Returns:
        (H, W, 6) averaged prediction.
    """
    # Create a dummy probs to transform alongside grid
    h, w = initial_grid.shape
    dummy = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    transforms = _d4_transforms(initial_grid, dummy)

    accumulated = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    for i, (g_t, _) in enumerate(transforms):
        pred_t = predict_fn(g_t)
        pred_orig = _d4_inverse(pred_t, i)
        accumulated += pred_orig

    return renormalize(accumulated / 8.0)


class _PredictFn(Protocol):
    def __call__(self, initial_grid: NDArray[np.int_]) -> NDArray[np.float64]: ...


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------


def prepare_training_data(
    rounds: RoundData,
    exclude_round: int | None = None,
    augment: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Prepare flattened feature matrix and soft targets from round data.

    Args:
        rounds: RoundData dict (round_id -> seeds).
        exclude_round: If set, exclude this round (for LOROCV).
        augment: If True, apply D4 augmentation (8x data).

    Returns:
        (X, Y, mask) where:
        - X: (N, F) feature matrix (dynamic cells only)
        - Y: (N, 6) soft probability targets
        - mask: not returned separately — dynamic masking is built in
    """
    x_parts: list[NDArray[np.float64]] = []
    y_parts: list[NDArray[np.float64]] = []

    for rid, seeds in rounds.items():
        if rid == exclude_round:
            continue
        for ig, gt in seeds:
            pairs: list[tuple[NDArray[np.int_], NDArray[np.float64]]]
            if augment:
                pairs = _d4_transforms(ig, gt)
            else:
                pairs = [(ig, gt)]

            for g, p in pairs:
                feats = extract_features(g)
                dynamic = make_dynamic_mask(g)

                # Flatten dynamic cells
                flat_feats = feats[dynamic]  # (N_dyn, F)
                flat_probs = p[dynamic]  # (N_dyn, 6)

                x_parts.append(flat_feats)
                y_parts.append(flat_probs)

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    logger.info(
        "Training data: %d samples, %d features", x_all.shape[0], x_all.shape[1]
    )
    return x_all, y_all, np.ones(x_all.shape[0], dtype=np.bool_)


# ---------------------------------------------------------------------------
# Shrunken Archetype baseline
# ---------------------------------------------------------------------------


class ShrunkenArchetype:
    """KNN-based baseline with shrinkage toward the global mean.

    For each test cell, finds the k nearest training cells by feature
    distance, averages their soft targets, and shrinks toward the
    global mean to reduce overfitting.

    shrunk = alpha * knn_mean + (1 - alpha) * global_mean
    """

    def __init__(
        self,
        k: int = 10,
        alpha: float = 0.7,
    ) -> None:
        self.k = k
        self.alpha = alpha
        self._nn: _NNModel | None = None
        self._y_train: NDArray[np.float64] | None = None
        self._global_mean: NDArray[np.float64] | None = None

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
    ) -> None:
        """Fit the KNN model.

        Args:
            x_train: (N, F) feature matrix.
            y_train: (N, 6) soft probability targets.
        """
        raw_nn: object = NearestNeighbors(n_neighbors=self.k, algorithm="auto")
        assert isinstance(raw_nn, _NNModel)
        nn_obj: _NNModel = raw_nn
        nn_obj.fit(x_train)
        self._nn = nn_obj
        self._y_train = y_train.copy()
        self._global_mean = np.asarray(np.mean(y_train, axis=0), dtype=np.float64)

    def predict_raw(
        self,
        x_test: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict soft probabilities for test features.

        Args:
            x_test: (M, F) feature matrix.

        Returns:
            (M, 6) predicted probabilities.
        """
        if self._nn is None or self._y_train is None or self._global_mean is None:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        _, indices = self._nn.kneighbors(x_test)
        _knn_raw: object = self._y_train[indices].mean(axis=1)
        assert isinstance(_knn_raw, np.ndarray)
        knn_mean: NDArray[np.float64] = np.asarray(_knn_raw, dtype=np.float64)  # (M, 6)
        shrunk: NDArray[np.float64] = np.asarray(
            self.alpha * knn_mean + (1.0 - self.alpha) * self._global_mean,
            dtype=np.float64,
        )
        return renormalize(shrunk)

    def predict_grid(
        self,
        initial_grid: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Predict (H, W, 6) probability tensor for an initial grid.

        Args:
            initial_grid: (H, W) terrain codes.

        Returns:
            (H, W, 6) predicted probabilities with floors applied.
        """
        h, w = initial_grid.shape
        feats = extract_features(initial_grid)
        dynamic = make_dynamic_mask(initial_grid)

        flat_feats = feats[dynamic]
        flat_pred = self.predict_raw(flat_feats)

        # Reconstruct full grid
        out = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
        out[dynamic] = flat_pred
        # Static cells get one-hot from apply_floors
        return apply_floors(out, initial_grid)


# ---------------------------------------------------------------------------
# Per-cell GBDT model
# ---------------------------------------------------------------------------


class PerCellGBDT:
    """6 separate HistGradientBoostingRegressors, one per class.

    Each regressor is trained on the soft probability target for its class.
    Predictions are clipped to [0, 1] and renormalized.
    """

    def __init__(
        self,
        max_iter: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ) -> None:
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._models: list[_GBDTModel] = []

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
    ) -> None:
        """Train 6 regressors on soft targets.

        Args:
            x_train: (N, F) feature matrix.
            y_train: (N, 6) soft probability targets.
        """
        self._models = []
        for c in range(NUM_CLASSES):
            logger.info("Training GBDT for class %d/%d", c + 1, NUM_CLASSES)
            raw_reg: object = HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            assert isinstance(raw_reg, _GBDTModel)
            reg: _GBDTModel = raw_reg
            reg.fit(x_train, y_train[:, c])
            self._models.append(reg)

    def predict_raw(
        self,
        x_test: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict soft probabilities for test features.

        Args:
            x_test: (M, F) feature matrix.

        Returns:
            (M, 6) predicted probabilities, clipped and renormalized.
        """
        if not self._models:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        pred_list: list[NDArray[np.float64]] = [
            np.asarray(model.predict(x_test), dtype=np.float64)
            for model in self._models
        ]
        _cs_raw: object = np.column_stack(pred_list)
        assert isinstance(_cs_raw, np.ndarray)
        preds: NDArray[np.float64] = np.asarray(_cs_raw, dtype=np.float64)
        # Clip to valid range, then renormalize
        _clip_raw: object = np.clip(preds, 0.0, 1.0)
        assert isinstance(_clip_raw, np.ndarray)
        preds = np.asarray(_clip_raw, dtype=np.float64)
        return renormalize(preds)

    def predict_grid_raw(
        self,
        initial_grid: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Predict (H, W, 6) probability tensor *without* floors.

        Use this when unfloored predictions are needed (e.g. entropy
        ranking for query planning).  Floors should be applied later,
        after any blending step.

        Args:
            initial_grid: (H, W) terrain codes.

        Returns:
            (H, W, 6) predicted probabilities, renormalized but unfloored.
        """
        h, w = initial_grid.shape
        feats = extract_features(initial_grid)
        dynamic = make_dynamic_mask(initial_grid)

        flat_feats = feats[dynamic]
        flat_pred = self.predict_raw(flat_feats)

        out = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
        out[dynamic] = flat_pred
        return out

    def predict_grid(
        self,
        initial_grid: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Predict (H, W, 6) probability tensor for an initial grid.

        Args:
            initial_grid: (H, W) terrain codes.

        Returns:
            (H, W, 6) predicted probabilities with floors applied.
        """
        return apply_floors(self.predict_grid_raw(initial_grid), initial_grid)
