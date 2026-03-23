from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams


class MCChallengerStrategy:
    """Fair Monte Carlo strategy with a realistic simulation budget.

    Unlike ``mc_oracle`` which uses ``budget.remaining * 4`` runs per seed (an
    unrealistically generous ceiling), this strategy uses a fixed, moderate
    number of Monte Carlo runs.  It answers the question: *how good is the
    digital twin with a practical compute budget?*

    The default of 100 runs per seed is chosen to balance quality against
    runtime — it is large enough to smooth out individual-run noise while
    remaining fast enough for iterative development.

    This is a **fair competitor** and should be compared directly against
    heuristic strategies.  It is NOT an oracle/ceiling.
    """

    def __init__(
        self,
        n_runs: int = 100,
        params: SimulationParams | None = None,
    ) -> None:
        self._n_runs = n_runs
        self._params = params or SimulationParams()

    @property
    def name(self) -> str:
        return "mc_challenger"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        simulator = Simulator(params=self._params)
        mc_runner = MCRunner(simulator)
        H = len(initial_state.grid)
        W = len(initial_state.grid[0])
        runs = mc_runner.run_batch(
            initial_state=initial_state,
            n_runs=self._n_runs,
            base_seed=base_seed,
        )
        return aggregate_runs(runs, H, W)
