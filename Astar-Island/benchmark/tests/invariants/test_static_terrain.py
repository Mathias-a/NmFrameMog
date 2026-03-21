from __future__ import annotations

from astar_twin.contracts.types import TerrainCode
from astar_twin.data.models import RoundFixture
from astar_twin.engine import Simulator


def test_ocean_cells_never_change(fixture: RoundFixture) -> None:
    simulator = Simulator(params=fixture.simulation_params)
    initial_state = fixture.initial_states[0]
    initial_grid = initial_state.grid

    for seed in range(3):
        final_state = simulator.run(initial_state, sim_seed=seed)
        for y, row in enumerate(initial_grid):
            for x, cell in enumerate(row):
                if cell == TerrainCode.OCEAN:
                    assert final_state.grid.get(y, x) == TerrainCode.OCEAN


def test_mountain_cells_never_change(fixture: RoundFixture) -> None:
    simulator = Simulator(params=fixture.simulation_params)
    initial_state = fixture.initial_states[0]
    initial_grid = initial_state.grid

    for seed in range(3):
        final_state = simulator.run(initial_state, sim_seed=seed)
        for y, row in enumerate(initial_grid):
            for x, cell in enumerate(row):
                if cell == TerrainCode.MOUNTAIN:
                    assert final_state.grid.get(y, x) == TerrainCode.MOUNTAIN
