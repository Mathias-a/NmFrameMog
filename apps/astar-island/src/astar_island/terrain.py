"""Terrain types and prediction class mapping for Astar Island.

The simulator uses 8 internal terrain types that map to 6 prediction classes.

Mapping:
  Ocean (10), Plains (11), Empty (0) → class 0 (Empty)
  Settlement (1) → class 1
  Port (2)       → class 2
  Ruin (3)       → class 3
  Forest (4)     → class 4
  Mountain (5)   → class 5
"""

from __future__ import annotations

from enum import IntEnum


class TerrainType(IntEnum):
    """The 8 internal terrain types used by the simulator."""

    EMPTY = 0
    SETTLEMENT = 1
    PORT = 2
    RUIN = 3
    FOREST = 4
    MOUNTAIN = 5
    OCEAN = 10
    PLAINS = 11


class PredictionClass(IntEnum):
    """The 6 prediction classes for submission."""

    EMPTY = 0
    SETTLEMENT = 1
    PORT = 2
    RUIN = 3
    FOREST = 4
    MOUNTAIN = 5


NUM_PREDICTION_CLASSES: int = 6
GRID_SIZE: int = 40

TERRAIN_TO_CLASS: dict[TerrainType, PredictionClass] = {
    TerrainType.EMPTY: PredictionClass.EMPTY,
    TerrainType.SETTLEMENT: PredictionClass.SETTLEMENT,
    TerrainType.PORT: PredictionClass.PORT,
    TerrainType.RUIN: PredictionClass.RUIN,
    TerrainType.FOREST: PredictionClass.FOREST,
    TerrainType.MOUNTAIN: PredictionClass.MOUNTAIN,
    TerrainType.OCEAN: PredictionClass.EMPTY,
    TerrainType.PLAINS: PredictionClass.EMPTY,
}

# Integer lookup for vectorized mapping (covers codes 0-11).
TERRAIN_CODE_TO_CLASS: dict[int, int] = {
    int(k): int(v) for k, v in TERRAIN_TO_CLASS.items()
}


def map_grid_to_classes(raw_grid: list[list[int]]) -> list[list[int]]:
    """Map a raw terrain grid (8 codes) to prediction classes (6 codes).

    Args:
        raw_grid: Grid with grid[x][y] indexing, values in {0..5, 10, 11}.

    Returns:
        Grid with same shape, values in {0..5}.
    """
    return [[TERRAIN_CODE_TO_CLASS[cell] for cell in row] for row in raw_grid]
