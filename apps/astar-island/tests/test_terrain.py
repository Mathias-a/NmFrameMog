"""Tests for terrain type mapping and prediction class constants."""

from __future__ import annotations

import pytest

from astar_island.terrain import (
    TERRAIN_CODE_TO_CLASS,
    TERRAIN_TO_CLASS,
    NUM_PREDICTION_CLASSES,
    PredictionClass,
    TerrainType,
    map_grid_to_classes,
)


class TestTerrainCodeMapping:
    """Verify all 8 terrain codes map to the correct prediction class."""

    @pytest.mark.parametrize(
        ("code", "expected_class"),
        [
            (0, 0),  # Empty → class 0
            (1, 1),  # Settlement → class 1
            (2, 2),  # Port → class 2
            (3, 3),  # Ruin → class 3
            (4, 4),  # Forest → class 4
            (5, 5),  # Mountain → class 5
            (10, 0),  # Ocean → class 0 (Empty)
            (11, 0),  # Plains → class 0 (Empty)
        ],
    )
    def test_code_maps_correctly(self, code: int, expected_class: int) -> None:
        assert TERRAIN_CODE_TO_CLASS[code] == expected_class

    def test_all_terrain_types_have_mapping(self) -> None:
        """Every TerrainType enum member should be in the mapping."""
        for tt in TerrainType:
            assert int(tt) in TERRAIN_CODE_TO_CLASS, f"TerrainType.{tt.name} missing"

    def test_terrain_to_class_enum_mapping(self) -> None:
        """TERRAIN_TO_CLASS (enum keys) matches TERRAIN_CODE_TO_CLASS (int keys)."""
        for tt, pc in TERRAIN_TO_CLASS.items():
            assert TERRAIN_CODE_TO_CLASS[int(tt)] == int(pc)


class TestPredictionClassConstants:
    def test_num_classes_is_six(self) -> None:
        assert NUM_PREDICTION_CLASSES == 6

    def test_prediction_class_values(self) -> None:
        assert int(PredictionClass.EMPTY) == 0
        assert int(PredictionClass.SETTLEMENT) == 1
        assert int(PredictionClass.PORT) == 2
        assert int(PredictionClass.RUIN) == 3
        assert int(PredictionClass.FOREST) == 4
        assert int(PredictionClass.MOUNTAIN) == 5


class TestMapGridToClasses:
    def test_simple_mapping(self) -> None:
        raw = [[0, 10], [5, 11]]
        result = map_grid_to_classes(raw)
        assert result == [[0, 0], [5, 0]]

    def test_all_codes(self) -> None:
        raw = [[0, 1, 2, 3, 4, 5, 10, 11]]
        result = map_grid_to_classes(raw)
        assert result == [[0, 1, 2, 3, 4, 5, 0, 0]]

    def test_unknown_code_raises(self) -> None:
        """map_grid_to_classes should raise KeyError on unknown terrain code."""
        raw = [[0, 99]]
        with pytest.raises(KeyError):
            map_grid_to_classes(raw)
