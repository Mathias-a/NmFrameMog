"""Terrain type -> prediction class mapping tests.

Validated against challenge://astar-island/mechanics terrain table.
"""

from __future__ import annotations

from astar_island.terrain import (
    NUM_PREDICTION_CLASSES,
    TERRAIN_CATALOG,
    TERRAIN_TO_CLASS,
    PredictionClass,
    TerrainType,
    all_prediction_classes,
    all_terrain_types,
    class_index,
    terrain_to_class,
    validate_class_index,
)


class TestAllTerrainsMapped:
    """Every terrain type must have a mapping to a prediction class."""

    def test_all_8_terrains_in_map(self) -> None:
        terrains = all_terrain_types()
        assert len(terrains) == 8
        for t in terrains:
            assert t in TERRAIN_TO_CLASS, f"{t.name} is not in TERRAIN_TO_CLASS"

    def test_all_terrains_have_catalog_entry(self) -> None:
        catalog_terrains = {info.terrain for info in TERRAIN_CATALOG}
        for t in all_terrain_types():
            assert t in catalog_terrains, f"{t.name} missing from TERRAIN_CATALOG"


class TestMappedClassesValid:
    """All mapped classes must be in the valid range [0, 5]."""

    def test_class_indices_in_range(self) -> None:
        for terrain in all_terrain_types():
            idx = class_index(terrain)
            assert 0 <= idx < NUM_PREDICTION_CLASSES, (
                f"{terrain.name} maps to invalid class {idx}"
            )

    def test_validate_class_index_boundaries(self) -> None:
        assert validate_class_index(0) is True
        assert validate_class_index(5) is True
        assert validate_class_index(-1) is False
        assert validate_class_index(6) is False


class TestExpectedMappings:
    """Verify mapping matches the official docs terrain table.

    From challenge://astar-island/mechanics:
    | Code | Terrain    | Class |
    |------|------------|-------|
    | 10   | Ocean      | 0     |
    | 11   | Plains     | 0     |
    | 0    | Empty      | 0     |
    | 1    | Settlement | 1     |
    | 2    | Port       | 2     |
    | 3    | Ruin       | 3     |
    | 4    | Forest     | 4     |
    | 5    | Mountain   | 5     |
    """

    def test_class_0_empty_group(self) -> None:
        """Ocean, Plains, Empty all map to class 0 (Empty)."""
        assert terrain_to_class(TerrainType.OCEAN) == PredictionClass.EMPTY
        assert terrain_to_class(TerrainType.PLAINS) == PredictionClass.EMPTY
        assert terrain_to_class(TerrainType.EMPTY) == PredictionClass.EMPTY

    def test_settlement_class_1(self) -> None:
        assert terrain_to_class(TerrainType.SETTLEMENT) == (PredictionClass.SETTLEMENT)

    def test_port_class_2(self) -> None:
        assert terrain_to_class(TerrainType.PORT) == PredictionClass.PORT

    def test_ruin_class_3(self) -> None:
        assert terrain_to_class(TerrainType.RUIN) == PredictionClass.RUIN

    def test_forest_class_4(self) -> None:
        assert terrain_to_class(TerrainType.FOREST) == (PredictionClass.FOREST)

    def test_mountain_class_5(self) -> None:
        assert terrain_to_class(TerrainType.MOUNTAIN) == (PredictionClass.MOUNTAIN)

    def test_internal_codes_match_docs(self) -> None:
        """Verify enum values match the internal codes from the docs."""
        assert TerrainType.EMPTY == 0
        assert TerrainType.SETTLEMENT == 1
        assert TerrainType.PORT == 2
        assert TerrainType.RUIN == 3
        assert TerrainType.FOREST == 4
        assert TerrainType.MOUNTAIN == 5
        assert TerrainType.OCEAN == 10
        assert TerrainType.PLAINS == 11


class TestPredictionClasses:
    """Prediction class enumeration."""

    def test_exactly_6_classes(self) -> None:
        classes = all_prediction_classes()
        assert len(classes) == 6

    def test_class_values_0_to_5(self) -> None:
        for i, cls in enumerate(sorted(all_prediction_classes())):
            assert int(cls) == i

    def test_catalog_consistency(self) -> None:
        """Catalog entries must agree with the mapping dict."""
        for info in TERRAIN_CATALOG:
            assert TERRAIN_TO_CLASS[info.terrain] == info.prediction_class
