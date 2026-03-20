from __future__ import annotations

from enum import IntEnum
from typing import Final

PROBABILITY_FLOOR: Final[float] = 0.01
CLASS_COUNT: Final[int] = 6
DEFAULT_MAP_WIDTH: Final[int] = 40
DEFAULT_MAP_HEIGHT: Final[int] = 40
DEFAULT_SEED_COUNT: Final[int] = 5
DEFAULT_QUERY_BUDGET: Final[int] = 50
MIN_VIEWPORT_SIZE: Final[int] = 5
MAX_VIEWPORT_SIZE: Final[int] = 15
INTERNAL_TERRAIN_CODE_COUNT: Final[int] = 8


class TerrainClass(IntEnum):
    EMPTY = 0
    SETTLEMENT = 1
    PORT = 2
    RUIN = 3
    FOREST = 4
    MOUNTAIN = 5


TERRAIN_CODE_TO_CLASS: Final[dict[int, TerrainClass]] = {
    0: TerrainClass.EMPTY,
    1: TerrainClass.SETTLEMENT,
    2: TerrainClass.PORT,
    3: TerrainClass.RUIN,
    4: TerrainClass.FOREST,
    5: TerrainClass.MOUNTAIN,
    10: TerrainClass.EMPTY,
    11: TerrainClass.EMPTY,
}

STATIC_TERRAIN_CODES: Final[frozenset[int]] = frozenset({5, 10})
EMPTYISH_TERRAIN_CODES: Final[frozenset[int]] = frozenset({0, 10, 11})
DYNAMIC_TERRAIN_CODES: Final[frozenset[int]] = frozenset(
    set(TERRAIN_CODE_TO_CLASS) - set(STATIC_TERRAIN_CODES)
)

CLASS_NAMES: Final[dict[int, str]] = {
    TerrainClass.EMPTY: "empty",
    TerrainClass.SETTLEMENT: "settlement",
    TerrainClass.PORT: "port",
    TerrainClass.RUIN: "ruin",
    TerrainClass.FOREST: "forest",
    TerrainClass.MOUNTAIN: "mountain",
}


def terrain_code_to_class_index(terrain_code: int) -> int:
    try:
        return int(TERRAIN_CODE_TO_CLASS[terrain_code])
    except KeyError as error:
        raise ValueError(f"Unsupported terrain code: {terrain_code}") from error


def terrain_code_is_static(terrain_code: int) -> bool:
    return terrain_code in STATIC_TERRAIN_CODES


def canonical_mapping_artifact() -> dict[str, object]:
    return {
        "internal_code_count": INTERNAL_TERRAIN_CODE_COUNT,
        "prediction_class_count": CLASS_COUNT,
        "mapping": {
            str(code): {
                "class_index": int(terrain_class),
                "class_name": CLASS_NAMES[int(terrain_class)],
            }
            for code, terrain_class in sorted(TERRAIN_CODE_TO_CLASS.items())
        },
    }
