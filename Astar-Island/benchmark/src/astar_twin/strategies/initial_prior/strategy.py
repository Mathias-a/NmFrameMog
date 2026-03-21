from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, TERRAIN_TO_CLASS

_HIGH_CONF = 0.9
_LOW_CONF = (1.0 - _HIGH_CONF) / (NUM_CLASSES - 1)


class InitialPriorStrategy:
    """Use the initial terrain grid as a strong prior.

    For each cell the terrain code is mapped to a submission class index via
    ``TERRAIN_TO_CLASS``.  That class gets probability ``_HIGH_CONF`` (0.9) and
    the remaining probability mass is spread uniformly across the other five
    classes.

    This strategy is intentionally simple — it captures the large signal from
    the static terrain (Ocean, Mountain) while acknowledging that dynamic cells
    (Plains, Forest) can change during simulation.
    """

    @property
    def name(self) -> str:
        return "initial_prior"

    def predict(
        self,
        initial_state: InitialState,
        budget: int,
        base_seed: int,
    ) -> NDArray[np.float64]:
        grid = initial_state.grid
        H = len(grid)
        W = len(grid[0])
        tensor = np.full((H, W, NUM_CLASSES), _LOW_CONF, dtype=np.float64)
        for y, row in enumerate(grid):
            for x, code in enumerate(row):
                cls = TERRAIN_TO_CLASS[code]
                tensor[y, x, cls] = _HIGH_CONF
        return tensor
