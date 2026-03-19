"""Tests for the simulation engine."""

from __future__ import annotations

from astar_island.simulator.engine import run_simulation
from astar_island.simulator.models import SimConfig
from astar_island.terrain import TerrainType


class TestSimulationCompletion:
    def test_completes_50_years(self) -> None:
        state = run_simulation(seed=42, num_years=50, width=20, height=20)
        assert state.year == 50

    def test_valid_terrain_everywhere(self) -> None:
        state = run_simulation(seed=42, num_years=50, width=20, height=20)
        valid = set(TerrainType)
        for y in range(state.height):
            for x in range(state.width):
                assert state.grid[y][x] in valid


class TestDeterminism:
    def test_same_seeds_same_result(self) -> None:
        s1 = run_simulation(seed=42, stochastic_seed=99, width=20, height=20)
        s2 = run_simulation(seed=42, stochastic_seed=99, width=20, height=20)
        for y in range(s1.height):
            for x in range(s1.width):
                assert s1.grid[y][x] == s2.grid[y][x]

    def test_different_stochastic_seed_differs(self) -> None:
        """Different stochastic seeds should produce different internal state."""
        config = SimConfig(
            winter_severity_mean=0.8,
            winter_severity_std=0.3,
            expansion_threshold=80,
        )
        s1 = run_simulation(
            seed=42, config=config, stochastic_seed=1, width=20, height=20
        )
        s2 = run_simulation(
            seed=42, config=config, stochastic_seed=2, width=20, height=20
        )
        # Grid or settlement internal state should differ
        grid_diff = any(
            s1.grid[y][x] != s2.grid[y][x]
            for y in range(s1.height)
            for x in range(s1.width)
        )
        state_diff = any(
            a.food != b.food or a.population != b.population
            for a, b in zip(s1.settlements, s2.settlements, strict=False)
        )
        assert grid_diff or state_diff, (
            "Different stochastic seeds should produce different outcomes"
        )


class TestDynamics:
    def test_state_changes_from_initial(self) -> None:
        from astar_island.simulator.mapgen import generate_map

        config = SimConfig()
        initial = generate_map(42, config, 20, 20)
        final = run_simulation(seed=42, stochastic_seed=1, width=20, height=20)
        any_diff = False
        for y in range(20):
            for x in range(20):
                if initial.grid[y][x] != final.grid[y][x]:
                    any_diff = True
                    break
        assert any_diff, "Simulation should change the world"
