from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.harness.budget import Budget
from astar_twin.strategies.calibrated_mc.strategy import (
    CalibratedMCStrategy,
    _build_coastal_mask,
    _build_static_mask,
)
from astar_twin.strategies.filter_baseline.strategy import FilterBaselineStrategy
from astar_twin.strategies.learned_calibrator.model import (
    blend_predictions,
    normalize_zone_weights,
)


class LearnedCalibratorStrategy:
    def __init__(
        self,
        zone_weights: Mapping[str, float] | None = None,
        n_runs: int = 25,
    ) -> None:
        self._zone_weights = normalize_zone_weights(zone_weights)
        self._base_strategy = CalibratedMCStrategy(n_runs=n_runs)
        self._fallback_strategy = FilterBaselineStrategy()
        self._zone_helper = CalibratedMCStrategy(n_runs=5)

    @property
    def name(self) -> str:
        return "learned_calibrator"

    @property
    def zone_weights(self) -> dict[str, float]:
        return dict(self._zone_weights)

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        base_prediction = self._base_strategy.predict(initial_state, budget, base_seed)
        fallback_prediction = self._fallback_strategy.predict(initial_state, budget, base_seed)

        grid = initial_state.grid
        height = len(grid)
        width = len(grid[0])
        is_static = _build_static_mask(grid, height, width)
        is_coastal = _build_coastal_mask(grid, height, width)
        zone_map = self._zone_helper._build_zone_map(
            initial_state,
            height,
            width,
            is_static,
            is_coastal,
        )
        return blend_predictions(
            base_prediction,
            fallback_prediction,
            zone_map,
            is_static,
            self._zone_weights,
        )
