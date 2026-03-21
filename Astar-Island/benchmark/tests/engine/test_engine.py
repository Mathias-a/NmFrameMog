from __future__ import annotations

import numpy as np

from astar_twin.contracts.types import TerrainCode
from astar_twin.data.models import RoundFixture
from astar_twin.engine import Simulator
from astar_twin.state.round_state import RoundState


def test_simulator_run_returns_round_state_with_correct_shape(fixture: RoundFixture) -> None:
    result = Simulator().run(fixture.initial_states[0], sim_seed=0)
    assert isinstance(result, RoundState)
    assert result.grid.height == 10
    assert result.grid.width == 10


def test_two_runs_with_different_seeds_produce_different_grids(fixture: RoundFixture) -> None:
    simulator = Simulator()
    first = simulator.run(fixture.initial_states[0], sim_seed=1)
    second = simulator.run(fixture.initial_states[0], sim_seed=2)
    assert first.grid.to_list() != second.grid.to_list()


def test_two_runs_with_same_seed_are_identical(fixture: RoundFixture) -> None:
    simulator = Simulator()
    first = simulator.run(fixture.initial_states[0], sim_seed=7)
    second = simulator.run(fixture.initial_states[0], sim_seed=7)
    assert first.grid.to_list() == second.grid.to_list()


def test_initial_settlement_positions_start_as_settlement_or_port(fixture: RoundFixture) -> None:
    """Initial state grid cells for settlements must be Settlement (1) or Port (2).

    After 50 stochastic years a settlement can legitimately collapse → Ruin → Forest/Plains,
    so we only verify the *initial* state, not the final state.
    """
    initial_state = fixture.initial_states[0]
    grid = np.array(initial_state.grid)
    for settlement in initial_state.settlements:
        cell_code = int(grid[settlement.y, settlement.x])
        assert cell_code in (
            int(TerrainCode.SETTLEMENT),
            int(TerrainCode.PORT),
        ), f"Expected Settlement or Port at ({settlement.y},{settlement.x}), got {cell_code}"
