"""Tests for map generation."""

from __future__ import annotations

from astar_island.simulator.mapgen import _distance, generate_map
from astar_island.simulator.models import SimConfig
from astar_island.terrain import TerrainType


class TestMapgenDeterminism:
    def test_same_seed_same_map(self) -> None:
        config = SimConfig()
        map_a = generate_map(42, config, 20, 20)
        map_b = generate_map(42, config, 20, 20)

        for y in range(20):
            for x in range(20):
                assert map_a.grid[y][x] == map_b.grid[y][x]

        assert len(map_a.settlements) == len(map_b.settlements)
        for sa, sb in zip(map_a.settlements, map_b.settlements, strict=True):
            assert sa.x == sb.x
            assert sa.y == sb.y
            assert sa.faction_id == sb.faction_id


class TestOceanBorders:
    def test_ocean_borders_present(self) -> None:
        config = SimConfig()
        state = generate_map(99, config, 20, 20)

        # Top and bottom 2 rows should be ocean
        for y in range(2):
            for x in range(20):
                assert state.grid[y][x] == TerrainType.OCEAN
        for y in range(18, 20):
            for x in range(20):
                assert state.grid[y][x] == TerrainType.OCEAN

        # Left and right 2 columns should be ocean
        for y in range(20):
            for x in range(2):
                assert state.grid[y][x] == TerrainType.OCEAN
            for x in range(18, 20):
                assert state.grid[y][x] == TerrainType.OCEAN


class TestSettlementPlacement:
    def test_settlements_on_plains_not_ocean(self) -> None:
        config = SimConfig()
        state = generate_map(42, config, 30, 30)

        for settlement in state.settlements:
            # Settlement grid cell should be SETTLEMENT (placed there)
            assert state.grid[settlement.y][settlement.x] == TerrainType.SETTLEMENT
            # Not in ocean border zone
            assert settlement.x >= 2
            assert settlement.y >= 2
            assert settlement.x < state.width - 2
            assert settlement.y < state.height - 2

    def test_min_spacing_respected(self) -> None:
        config = SimConfig()
        state = generate_map(42, config, 40, 40)

        positions = [(s.x, s.y) for s in state.settlements]
        for idx in range(len(positions)):
            for jdx in range(idx + 1, len(positions)):
                x1, y1 = positions[idx]
                x2, y2 = positions[jdx]
                dist = _distance(x1, y1, x2, y2)
                assert dist >= config.min_settlement_spacing


class TestGridValidity:
    def test_all_cells_valid_terrain(self) -> None:
        config = SimConfig()
        state = generate_map(123, config, 25, 25)
        valid_types = set(TerrainType)

        for y in range(state.height):
            for x in range(state.width):
                assert state.grid[y][x] in valid_types

    def test_grid_dimensions(self) -> None:
        config = SimConfig()
        state = generate_map(7, config, 30, 25)

        assert state.width == 30
        assert state.height == 25
        assert len(state.grid) == 25
        for row in state.grid:
            assert len(row) == 30
