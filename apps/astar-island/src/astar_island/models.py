"""GBT-based soft probability predictor.

Trains K=6 independent HistGradientBoostingRegressor models,
one per prediction class, on soft probability targets.
Predictions are clipped and renormalized to valid distributions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from astar_island.features import (
    extract_cell_features,
    raw_grid_to_class_grid,
)
from astar_island.terrain import NUM_PREDICTION_CLASSES


@dataclass
class GBTHyperparams:
    max_iter: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    min_samples_leaf: int = 30
    l2_regularization: float = 1.0
    max_bins: int = 64


def _fit_single_regressor(
    x_scaled: NDArray[np.float64],
    y_col: NDArray[np.float64],
    params: GBTHyperparams,
) -> HistGradientBoostingRegressor:
    x_copy = x_scaled.copy()
    model = HistGradientBoostingRegressor(
        max_iter=params.max_iter,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        min_samples_leaf=params.min_samples_leaf,
        l2_regularization=params.l2_regularization,
        max_bins=params.max_bins,
    )
    model.fit(x_copy, y_col)
    return model


class GBTSoftClassifier:
    def __init__(self, params: GBTHyperparams | None = None) -> None:
        self.params = params or GBTHyperparams()
        self.models: list[HistGradientBoostingRegressor] = []
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, x: NDArray[np.float64], y_soft: NDArray[np.float64]) -> None:
        x_scaled = self.scaler.fit_transform(x)
        p = self.params

        with ThreadPoolExecutor(max_workers=NUM_PREDICTION_CLASSES) as pool:
            futures = [
                pool.submit(_fit_single_regressor, x_scaled, y_soft[:, k], p)
                for k in range(NUM_PREDICTION_CLASSES)
            ]
            self.models = [f.result() for f in futures]

        self._fitted = True

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        x_scaled = self.scaler.transform(x)
        probs = np.stack(
            [m.predict(x_scaled) for m in self.models],
            axis=1,
        )
        probs = np.clip(probs, 1e-6, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict_grid(self, raw_grid: list[list[int]]) -> NDArray[np.float64]:
        class_grid = raw_grid_to_class_grid(raw_grid)
        features = extract_cell_features(class_grid, raw_grid)
        flat_probs = self.predict(features)
        rows = len(raw_grid)
        cols = len(raw_grid[0]) if rows > 0 else 0
        return flat_probs.reshape(rows, cols, NUM_PREDICTION_CLASSES)
