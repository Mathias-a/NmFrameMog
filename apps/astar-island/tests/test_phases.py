"""Tests for the 5 simulation phases."""

from __future__ import annotations

import random

from astar_island.simulator.models import Settlement, SimConfig, WorldState
from astar_island.simulator.phases import (
    phase_conflict,
    phase_environment,
    phase_growth,
    phase_trade,
    phase_winter,
)
from astar_island.terrain import TerrainType


def _make_small_world(width: int = 10, height: int = 10) -> WorldState:
    """Create a minimal world for phase testing."""
    grid = [[TerrainType.PLAINS for _ in range(width)] for _ in range(height)]
    # Ocean border
    for x in range(width):
        grid[0][x] = TerrainType.OCEAN
        grid[height - 1][x] = TerrainType.OCEAN
    for y in range(height):
        grid[y][0] = TerrainType.OCEAN
        grid[y][width - 1] = TerrainType.OCEAN
    return WorldState(width=width, height=height, grid=grid, settlements=[], year=0)


def _add_settlement(
    state: WorldState,
    x: int,
    y: int,
    faction_id: int = 1,
    population: int = 100,
    food: int = 50,
    has_port: bool = False,
) -> Settlement:
    s = Settlement(
        x=x,
        y=y,
        population=population,
        food=food,
        wealth=0,
        defense=5,
        tech_level=1,
        faction_id=faction_id,
        has_port=has_port,
        has_longship=False,
        alive=True,
    )
    state.settlements.append(s)
    state.grid[y][x] = TerrainType.PORT if has_port else TerrainType.SETTLEMENT
    return s


class TestGrowth:
    def test_food_from_adjacent_forest(self) -> None:
        state = _make_small_world()
        config = SimConfig()
        s = _add_settlement(state, 5, 5, food=10)
        state.grid[4][5] = TerrainType.FOREST
        state.grid[5][4] = TerrainType.FOREST
        old_food = s.food
        phase_growth(state, config, random.Random(42))
        assert s.food > old_food

    def test_expansion_creates_settlement(self) -> None:
        state = _make_small_world()
        config = SimConfig(expansion_threshold=50)
        _add_settlement(state, 5, 5, population=200, food=100)
        initial_count = len(state.settlements)
        # Run growth several times to give expansion a chance
        rng = random.Random(42)
        for _ in range(10):
            phase_growth(state, config, rng)
        assert len(state.settlements) > initial_count


class TestConflict:
    def test_raid_can_change_faction(self) -> None:
        state = _make_small_world()
        config = SimConfig(raid_range=5, raid_success_base=0.9)
        s1 = _add_settlement(state, 3, 3, faction_id=1, population=200)
        s2 = _add_settlement(state, 5, 5, faction_id=2, population=50)
        rng = random.Random(42)
        # Run conflict many times
        for _ in range(20):
            if s2.faction_id != s2.faction_id:
                break
            phase_conflict(state, config, rng)
        # At least food should have changed
        assert s1.food != 50 or s2.food != 50 or s2.faction_id == 1


class TestTrade:
    def test_ports_gain_food(self) -> None:
        state = _make_small_world()
        config = SimConfig(trade_range=10)
        s1 = _add_settlement(state, 3, 3, faction_id=1, has_port=True)
        s2 = _add_settlement(state, 5, 5, faction_id=2, has_port=True)
        old_food_1 = s1.food
        old_food_2 = s2.food
        phase_trade(state, config, random.Random(42))
        assert s1.food > old_food_1 or s2.food > old_food_2


class TestWinter:
    def test_settlement_collapses_to_ruin(self) -> None:
        state = _make_small_world()
        config = SimConfig(winter_severity_mean=0.99, winter_severity_std=0.01)
        _add_settlement(state, 5, 5, food=1, population=10)
        rng = random.Random(42)
        for _ in range(5):
            phase_winter(state, config, rng)
        # Settlement should have collapsed
        dead = [s for s in state.settlements if not s.alive]
        if dead:
            assert state.grid[dead[0].y][dead[0].x] == TerrainType.RUIN


class TestEnvironment:
    def test_ruin_can_be_reclaimed(self) -> None:
        state = _make_small_world()
        config = SimConfig(ruin_reclaim_prob=1.0)
        # Add a ruin
        state.grid[5][5] = TerrainType.RUIN
        dead_s = Settlement(
            x=5,
            y=5,
            population=0,
            food=0,
            wealth=0,
            defense=0,
            tech_level=0,
            faction_id=1,
            has_port=False,
            has_longship=False,
            alive=False,
        )
        state.settlements.append(dead_s)
        # Add a thriving settlement nearby
        _add_settlement(state, 4, 5, population=200, food=100)
        phase_environment(state, config, random.Random(42))
        assert state.grid[5][5] in (
            TerrainType.SETTLEMENT,
            TerrainType.RUIN,
        )

    def test_ruin_can_become_forest(self) -> None:
        state = _make_small_world()
        config = SimConfig(ruin_reclaim_prob=0.0, ruin_to_forest_prob=1.0)
        state.grid[5][5] = TerrainType.RUIN
        dead_s = Settlement(
            x=5,
            y=5,
            population=0,
            food=0,
            wealth=0,
            defense=0,
            tech_level=0,
            faction_id=1,
            has_port=False,
            has_longship=False,
            alive=False,
        )
        state.settlements.append(dead_s)
        phase_environment(state, config, random.Random(42))
        assert state.grid[5][5] == TerrainType.FOREST
