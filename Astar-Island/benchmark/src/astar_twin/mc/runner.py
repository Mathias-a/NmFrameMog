from __future__ import annotations

from astar_twin.contracts.api_models import InitialState
from astar_twin.engine import Simulator
from astar_twin.state.round_state import RoundState


class MCRunner:
    def __init__(self, simulator: Simulator) -> None:
        self.simulator = simulator

    def run_batch(
        self, initial_state: InitialState, n_runs: int, base_seed: int = 0
    ) -> list[RoundState]:
        return [
            self.simulator.run(initial_state=initial_state, sim_seed=base_seed + i)
            for i in range(n_runs)
        ]
