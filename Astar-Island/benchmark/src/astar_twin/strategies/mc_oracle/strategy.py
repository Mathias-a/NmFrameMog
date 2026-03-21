from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams


class MCOracleStrategy:
    """Upper-bound oracle that runs many MC simulations per seed.

    This strategy cheats by using the simulator directly and running
    ``budget.remaining * 4`` simulations per seed.  It is intended as a score
    ceiling to validate that the harness and scoring pipeline work correctly.
    It is NOT a fair competitor and should not be compared directly to real
    strategies.
    """

    def __init__(self, params: SimulationParams | None = None) -> None:
        self._params = params or SimulationParams()

    @property
    def name(self) -> str:
        return "mc_oracle"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        simulator = Simulator(params=self._params)
        mc_runner = MCRunner(simulator)
        n_runs = max(budget.remaining * 4, 20)
        H = len(initial_state.grid)
        W = len(initial_state.grid[0])
        runs = mc_runner.run_batch(
            initial_state=initial_state,
            n_runs=n_runs,
            base_seed=base_seed,
        )
        return aggregate_runs(runs, H, W)
