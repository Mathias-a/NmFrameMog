from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES


class NaiveBaselineStrategy:
    """Uniform distribution strategy — every cell gets equal probability for all 6 classes.

    This is the theoretical minimum: it ignores terrain and history entirely.
    Score is expected to be ~1-5. Any real model should beat this.
    """

    STRATEGY_NAME = "naive_baseline"

    @property
    def name(self) -> str:
        return self.STRATEGY_NAME

    def predict(
        self,
        initial_state: InitialState,
        budget: int,
        base_seed: int,
    ) -> NDArray[np.float64]:
        height = len(initial_state.grid)
        width = len(initial_state.grid[0])
        tensor = np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
        return tensor
