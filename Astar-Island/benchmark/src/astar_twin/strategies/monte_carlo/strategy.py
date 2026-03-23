from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params.simulation_params import SimulationParams


class MonteCarlStrategy:
    """Monte Carlo strategy using the digital twin.

    Each call to ``Simulator.run()`` advances the world through all SIM_YEARS
    (50) in one shot — it is a full independent timeline, not a single year.
    This strategy runs ``min(budget, max_runs)`` such timelines and aggregates
    their final terrain-class distributions into a probability tensor.

    Using ``budget`` as the run cap ties local compute to the real API query
    budget, making offline benchmark scores comparable to live-round scores.
    In a live round, ``budget`` represents the number of viewport observations
    available; here it proxies for the number of full simulations we're allowed
    to run, keeping the benchmark honest.

    ``max_runs`` is an optional ceiling to limit wall-clock time regardless of
    budget. The actual number of runs is ``min(budget, max_runs)``.
    """

    STRATEGY_NAME = "monte_carlo"

    def __init__(self, max_runs: int = 200) -> None:
        self._max_runs = max_runs

    @property
    def name(self) -> str:
        return self.STRATEGY_NAME

    def predict(
        self,
        initial_state: InitialState,
        budget: int,
        base_seed: int,
    ) -> NDArray[np.float64]:
        height = len(initial_state.grid)
        width = len(initial_state.grid[0])

        n_runs = min(budget, self._max_runs)

        simulator = Simulator(SimulationParams())
        runner = MCRunner(simulator)
        runs = runner.run_batch(initial_state, n_runs=n_runs, base_seed=base_seed)
        tensor = aggregate_runs(runs, height=height, width=width)
        return tensor
