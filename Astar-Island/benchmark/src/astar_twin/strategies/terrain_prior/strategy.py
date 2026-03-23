from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import (
    NUM_CLASSES,
    TERRAIN_TO_CLASS,
    TerrainCode,
)

# Confidence placed on the initial terrain code for static cells (Ocean, Mountain).
_STATIC_CONFIDENCE = 0.97

# For dynamic cells, split probability mass: put this much on the initial class,
# distribute the rest among the other dynamic classes (Settlement, Port, Ruin, Forest).
_DYNAMIC_INITIAL_WEIGHT = 0.60
_DYNAMIC_OTHER_CLASSES = (
    int(TERRAIN_TO_CLASS[TerrainCode.SETTLEMENT]),
    int(TERRAIN_TO_CLASS[TerrainCode.PORT]),
    int(TERRAIN_TO_CLASS[TerrainCode.RUIN]),
    int(TERRAIN_TO_CLASS[TerrainCode.FOREST]),
)


class TerrainPriorStrategy:
    """Terrain-aware prior strategy.

    Uses the initial terrain grid as an informative prior:
    - Static cells (Ocean, Mountain) get near-certainty for their class.
    - Dynamic cells (Empty, Plains, Forest, Settlement, Port, Ruin) get a
      weighted prior that favours their initial class but keeps mass on other
      dynamically reachable classes.

    This beats naive uniform by respecting information already visible in the
    initial state, without burning any API queries.
    """

    STRATEGY_NAME = "terrain_prior"

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

        tensor = np.zeros((height, width, NUM_CLASSES), dtype=np.float64)

        static_codes = frozenset([int(TerrainCode.OCEAN), int(TerrainCode.MOUNTAIN)])

        for y, row in enumerate(initial_state.grid):
            for x, code in enumerate(row):
                class_idx = TERRAIN_TO_CLASS[code]
                probs = _cell_probs(code, class_idx, static_codes)
                tensor[y, x] = probs

        return tensor


def _cell_probs(code: int, class_idx: int, static_codes: frozenset[int]) -> NDArray[np.float64]:
    """Build a length-6 probability vector for one cell."""
    probs = np.zeros(NUM_CLASSES, dtype=np.float64)

    if code in static_codes:
        # Static terrain: very high confidence it stays the same.
        remainder = 1.0 - _STATIC_CONFIDENCE
        other_count = NUM_CLASSES - 1
        probs[class_idx] = _STATIC_CONFIDENCE
        for k in range(NUM_CLASSES):
            if k != class_idx:
                probs[k] = remainder / other_count
    else:
        # Dynamic terrain: put weight on initial class, spread remainder
        # across other dynamically reachable classes.
        probs[class_idx] = _DYNAMIC_INITIAL_WEIGHT
        remainder = 1.0 - _DYNAMIC_INITIAL_WEIGHT
        spread_classes = [k for k in _DYNAMIC_OTHER_CLASSES if k != class_idx]
        if spread_classes:
            per_class = remainder / len(spread_classes)
            for k in spread_classes:
                probs[k] = per_class
        else:
            # class_idx is the only dynamic class — distribute remainder uniformly
            other_count = NUM_CLASSES - 1
            for k in range(NUM_CLASSES):
                if k != class_idx:
                    probs[k] = remainder / other_count

    return probs
