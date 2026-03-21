from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState


class UniformStrategy:
    @property
    def name(self) -> str:
        return "uniform"

    def predict(
        self,
        initial_state: InitialState,
        budget: int,
        base_seed: int,
    ) -> NDArray[np.float64]:
        height = len(initial_state.grid)
        width = len(initial_state.grid[0])
        return np.full((height, width, 6), 1.0 / 6.0, dtype=np.float64)
