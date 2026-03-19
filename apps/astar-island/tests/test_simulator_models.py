"""Tests for simulator data models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from astar_island.simulator.models import Settlement, SimConfig, WorldState
from astar_island.terrain import TerrainType


class TestSettlement:
    def test_construction(self) -> None:
        settlement = Settlement(
            x=5,
            y=10,
            population=100,
            food=50,
            wealth=10,
            defense=5,
            tech_level=1,
            faction_id=1,
            has_port=False,
            has_longship=False,
            alive=True,
        )
        assert settlement.x == 5
        assert settlement.y == 10
        assert settlement.population == 100
        assert settlement.food == 50
        assert settlement.wealth == 10
        assert settlement.defense == 5
        assert settlement.tech_level == 1
        assert settlement.faction_id == 1
        assert settlement.has_port is False
        assert settlement.has_longship is False
        assert settlement.alive is True

    def test_mutable(self) -> None:
        settlement = Settlement(
            x=0,
            y=0,
            population=100,
            food=50,
            wealth=10,
            defense=5,
            tech_level=1,
            faction_id=1,
            has_port=False,
            has_longship=False,
            alive=True,
        )
        settlement.population = 200
        assert settlement.population == 200

        settlement.alive = False
        assert settlement.alive is False

        settlement.food = 0
        assert settlement.food == 0


class TestSimConfig:
    def test_frozen(self) -> None:
        config = SimConfig()
        with pytest.raises(FrozenInstanceError):
            config.food_per_forest = 99  # type: ignore[misc]

    def test_defaults_sane(self) -> None:
        config = SimConfig()
        assert config.food_per_forest > 0
        assert config.food_per_plains > 0
        assert config.expansion_threshold > 0
        assert 0.0 < config.port_development_prob <= 1.0
        assert 0.0 < config.longship_build_prob <= 1.0
        assert config.raid_range > 0
        assert 0.0 < config.raid_success_base <= 1.0
        assert config.trade_range > 0
        assert 0.0 <= config.winter_severity_mean <= 1.0
        assert config.winter_severity_std > 0
        assert config.initial_settlement_count > 0
        assert config.min_settlement_spacing > 0


class TestWorldState:
    def test_construction(self) -> None:
        grid: list[list[TerrainType]] = [
            [TerrainType.PLAINS, TerrainType.OCEAN],
            [TerrainType.FOREST, TerrainType.MOUNTAIN],
        ]
        state = WorldState(
            width=2,
            height=2,
            grid=grid,
            settlements=[],
            year=0,
        )
        assert state.width == 2
        assert state.height == 2
        assert state.grid[0][0] == TerrainType.PLAINS
        assert state.grid[1][1] == TerrainType.MOUNTAIN
        assert state.settlements == []
        assert state.year == 0
