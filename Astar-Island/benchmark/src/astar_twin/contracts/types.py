"""Terrain codes, class indices, and canonical type aliases.

Internal terrain codes (as returned by the simulator and stored in grids):
    0  = Empty
    1  = Settlement
    2  = Port
    3  = Ruin
    4  = Forest
    5  = Mountain
    10 = Ocean
    11 = Plains

Submission class indices (6 classes in prediction tensor):
    0 = Empty  (Ocean=10, Plains=11, Empty=0 all map here)
    1 = Settlement
    2 = Port
    3 = Ruin
    4 = Forest
    5 = Mountain
"""

from enum import IntEnum


class TerrainCode(IntEnum):
    """Internal terrain codes used inside the simulator and API grid responses."""

    EMPTY = 0
    SETTLEMENT = 1
    PORT = 2
    RUIN = 3
    FOREST = 4
    MOUNTAIN = 5
    OCEAN = 10
    PLAINS = 11


class ClassIndex(IntEnum):
    """Submission class indices used in the H×W×6 prediction tensor."""

    EMPTY = 0
    SETTLEMENT = 1
    PORT = 2
    RUIN = 3
    FOREST = 4
    MOUNTAIN = 5


# Codes that are static and must NEVER change during simulation.
STATIC_TERRAIN_CODES: frozenset[int] = frozenset([TerrainCode.OCEAN, TerrainCode.MOUNTAIN])

# Codes that are considered "land" and can host settlements.
BUILDABLE_TERRAIN_CODES: frozenset[int] = frozenset(
    [TerrainCode.EMPTY, TerrainCode.PLAINS, TerrainCode.FOREST]
)

# Valid terrain codes accepted by the API.
VALID_TERRAIN_CODES: frozenset[int] = frozenset(int(c) for c in TerrainCode)

# Map from internal terrain code → submission class index.
TERRAIN_TO_CLASS: dict[int, int] = {
    TerrainCode.OCEAN: ClassIndex.EMPTY,
    TerrainCode.PLAINS: ClassIndex.EMPTY,
    TerrainCode.EMPTY: ClassIndex.EMPTY,
    TerrainCode.SETTLEMENT: ClassIndex.SETTLEMENT,
    TerrainCode.PORT: ClassIndex.PORT,
    TerrainCode.RUIN: ClassIndex.RUIN,
    TerrainCode.FOREST: ClassIndex.FOREST,
    TerrainCode.MOUNTAIN: ClassIndex.MOUNTAIN,
}

NUM_CLASSES: int = 6
DEFAULT_MAP_WIDTH: int = 40
DEFAULT_MAP_HEIGHT: int = 40
SIM_YEARS: int = 50
MAX_SEEDS: int = 5
MIN_VIEWPORT: int = 5
MAX_VIEWPORT: int = 15
MAX_QUERIES: int = 50
