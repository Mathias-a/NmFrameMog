from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.harness.budget import Budget
from astar_twin.solver.predict.structural_anchor import build_structural_anchor


class FilterBaselineStrategy:
    @property
    def name(self) -> str:
        return "filter_baseline"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        del budget, base_seed

        grid = initial_state.grid
        height = len(grid)
        width = len(grid[0])
        return build_structural_anchor(initial_state, height, width)
