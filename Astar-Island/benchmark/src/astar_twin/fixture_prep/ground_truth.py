from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.data.models import RoundFixture
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner, aggregate_runs


def compute_and_attach_ground_truths(
    fixture: RoundFixture,
    n_runs: int,
    base_seed: int,
) -> RoundFixture:
    simulator = Simulator(params=fixture.simulation_params)
    mc_runner = MCRunner(simulator)
    ground_truths: list[list[list[list[float]]]] = []

    for seed_idx in range(fixture.seeds_count):
        initial_state = fixture.initial_states[seed_idx]
        runs = mc_runner.run_batch(
            initial_state=initial_state,
            n_runs=n_runs,
            base_seed=base_seed + seed_idx * n_runs,
        )
        tensor: NDArray[np.float64] = aggregate_runs(runs, fixture.map_height, fixture.map_width)
        ground_truths.append(tensor.tolist())  # type: ignore[any]

    return fixture.model_copy(update={"ground_truths": ground_truths})  # type: ignore[any]
