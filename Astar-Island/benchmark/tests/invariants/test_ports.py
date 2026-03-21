from __future__ import annotations

from astar_twin.contracts.types import TerrainCode
from astar_twin.data.models import RoundFixture
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner


def test_all_ports_are_coastal_in_mc_runs(fixture: RoundFixture) -> None:
    runner = MCRunner(Simulator(params=fixture.simulation_params))
    runs = runner.run_batch(fixture.initial_states[0], n_runs=10)
    for run in runs:
        for y in range(run.grid.height):
            for x in range(run.grid.width):
                if run.grid.get(y, x) == TerrainCode.PORT:
                    assert run.grid.is_coastal(y, x)
