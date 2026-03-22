from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.harness.budget import Budget

_INLAND_TEMPLATE = np.array([0.5496, 0.1798, 0.0002, 0.0706, 0.1996, 0.0002], dtype=np.float64)
_COASTAL_TEMPLATE = np.array([0.4798, 0.1698, 0.1198, 0.0604, 0.1700, 0.0002], dtype=np.float64)


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
        tensor = np.zeros((height, width, NUM_CLASSES), dtype=np.float64)

        for y, row in enumerate(grid):
            for x, code in enumerate(row):
                if code == TerrainCode.OCEAN:
                    tensor[y, x, ClassIndex.EMPTY] = 1.0
                elif code == TerrainCode.MOUNTAIN:
                    tensor[y, x, ClassIndex.MOUNTAIN] = 1.0
                elif self._is_coastal(grid=grid, x=x, y=y):
                    tensor[y, x] = _COASTAL_TEMPLATE
                else:
                    tensor[y, x] = _INLAND_TEMPLATE

        return tensor

    def _is_coastal(self, grid: list[list[int]], x: int, y: int) -> bool:
        height = len(grid)
        width = len(grid[0])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == TerrainCode.OCEAN:
                return True
        return False
