from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.types import NUM_CLASSES, TERRAIN_TO_CLASS
from astar_twin.state.round_state import RoundState


def aggregate_runs(runs: list[RoundState], height: int, width: int) -> NDArray[np.float64]:
    counts = np.zeros((height, width, NUM_CLASSES), dtype=np.float64)
    if not runs:
        return counts

    for run in runs:
        for y in range(height):
            for x in range(width):
                terrain_code = run.grid.get(y, x)
                class_index = TERRAIN_TO_CLASS[terrain_code]
                counts[y, x, class_index] += 1.0

    counts /= float(len(runs))
    return counts
