"""Structural-anchor tensor builder.

Produces a heuristic prior probability tensor from the initial map layout
alone — no simulation or observation needed.  The same logic was originally
embedded inside the ``filter_baseline`` benchmark strategy; this module
makes it reusable by the live solver's hybrid hedge path.

Ocean and mountain cells receive deterministic one-hot predictions.
Coastal land cells (4-connected to ocean) receive a template that
includes port probability.  All other land cells receive an inland
template that excludes ports.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode

# Heuristic templates (same values used by the original filter_baseline).
INLAND_TEMPLATE: NDArray[np.float64] = np.array(
    (0.55, 0.18, 0.0, 0.07, 0.20, 0.0), dtype=np.float64
)
COASTAL_TEMPLATE: NDArray[np.float64] = np.array(
    (0.48, 0.17, 0.12, 0.06, 0.17, 0.0), dtype=np.float64
)


def _is_coastal(grid: list[list[int]], x: int, y: int) -> bool:
    """Return True if the cell at (x, y) is 4-connected to an ocean cell."""
    height = len(grid)
    width = len(grid[0])
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == TerrainCode.OCEAN:
            return True
    return False


def build_structural_anchor(
    initial_state: InitialState,
    height: int,
    width: int,
) -> NDArray[np.float64]:
    """Build a heuristic probability tensor from the initial map layout.

    Args:
        initial_state: The seed's initial state containing the terrain grid.
        height: Map height (rows).
        width: Map width (columns).

    Returns:
        An (H, W, 6) float64 tensor with probabilities summing to 1.0 per cell.
    """
    grid = initial_state.grid
    tensor = np.zeros((height, width, NUM_CLASSES), dtype=np.float64)

    for y, row in enumerate(grid):
        for x, code in enumerate(row):
            if code == TerrainCode.OCEAN:
                tensor[y, x, ClassIndex.EMPTY] = 1.0
            elif code == TerrainCode.MOUNTAIN:
                tensor[y, x, ClassIndex.MOUNTAIN] = 1.0
            elif _is_coastal(grid=grid, x=x, y=y):
                tensor[y, x] = COASTAL_TEMPLATE
            else:
                tensor[y, x] = INLAND_TEMPLATE

    return tensor


def build_structural_anchors(
    initial_states: list[InitialState],
    height: int,
    width: int,
) -> list[NDArray[np.float64]]:
    """Build structural anchors for multiple seeds.

    Args:
        initial_states: Per-seed initial states.
        height: Map height.
        width: Map width.

    Returns:
        One (H, W, 6) anchor tensor per seed.
    """
    return [build_structural_anchor(s, height, width) for s in initial_states]
