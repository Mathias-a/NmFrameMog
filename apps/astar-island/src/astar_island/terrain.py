"""Terrain types and prediction class mapping for Astar Island.

The simulator uses 8 internal terrain types that map to 6 prediction classes.
Source: challenge://astar-island/mechanics (validated terrain table).

Real mapping from the docs:
| Code | Terrain    | Class | Description                         |
|------|------------|-------|-------------------------------------|
| 10   | Ocean      | 0     | Impassable water, borders the map   |
| 11   | Plains     | 0     | Flat land, buildable                |
| 0    | Empty      | 0     | Generic empty cell                  |
| 1    | Settlement | 1     | Active Norse settlement             |
| 2    | Port       | 2     | Coastal settlement with harbour     |
| 3    | Ruin       | 3     | Collapsed settlement                |
| 4    | Forest     | 4     | Provides food to adjacent settle... |
| 5    | Mountain   | 5     | Impassable terrain                  |

Ocean, Plains, and Empty all merge into class 0 (Empty).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TerrainType(IntEnum):
    """The 8 internal terrain types used by the simulator.

    Values match the internal codes from the docs.
    """

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

    EMPTY = 0  # Ocean + Plains + Empty
    SETTLEMENT = 1
    PORT = 2
    RUIN = 3
    FOREST = 4
    MOUNTAIN = 5


NUM_PREDICTION_CLASSES: int = 6

# 8 terrain types -> 6 prediction classes.
# Ocean, Plains, Empty all merge into class 0 (Empty).
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


@dataclass(frozen=True, slots=True)
class TerrainInfo:
    """Metadata for a terrain type."""

    terrain: TerrainType
    prediction_class: PredictionClass
    description: str


TERRAIN_CATALOG: tuple[TerrainInfo, ...] = (
    TerrainInfo(
        TerrainType.OCEAN,
        PredictionClass.EMPTY,
        "Impassable water, borders the map",
    ),
    TerrainInfo(
        TerrainType.PLAINS,
        PredictionClass.EMPTY,
        "Flat land, buildable",
    ),
    TerrainInfo(
        TerrainType.EMPTY,
        PredictionClass.EMPTY,
        "Generic empty cell",
    ),
    TerrainInfo(
        TerrainType.SETTLEMENT,
        PredictionClass.SETTLEMENT,
        "Active Norse settlement",
    ),
    TerrainInfo(
        TerrainType.PORT,
        PredictionClass.PORT,
        "Coastal settlement with harbour",
    ),
    TerrainInfo(
        TerrainType.RUIN,
        PredictionClass.RUIN,
        "Collapsed settlement",
    ),
    TerrainInfo(
        TerrainType.FOREST,
        PredictionClass.FOREST,
        "Provides food to adjacent settlements",
    ),
    TerrainInfo(
        TerrainType.MOUNTAIN,
        PredictionClass.MOUNTAIN,
        "Impassable terrain",
    ),
)


def terrain_to_class(terrain: TerrainType) -> PredictionClass:
    """Map a terrain type to its prediction class."""
    return TERRAIN_TO_CLASS[terrain]


def class_index(terrain: TerrainType) -> int:
    """Return the integer class index (0-5) for a terrain type."""
    return int(TERRAIN_TO_CLASS[terrain])


def validate_class_index(index: int) -> bool:
    """Check that a class index is within the valid range [0, 5]."""
    return 0 <= index < NUM_PREDICTION_CLASSES


def all_terrain_types() -> list[TerrainType]:
    """Return all 8 terrain types."""
    return list(TerrainType)


def all_prediction_classes() -> list[PredictionClass]:
    """Return all 6 prediction classes."""
    return list(PredictionClass)
