from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.data.models import ParamsSource, RoundFixture
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import sample_default_prior_params


def compute_and_attach_ground_truths(
    fixture: RoundFixture,
    n_runs: int,
    base_seed: int,
    prior_spread: float = 1.0,
) -> RoundFixture:
    simulation_params = fixture.simulation_params
    if fixture.params_source == ParamsSource.DEFAULT_PRIOR:
        simulation_params = sample_default_prior_params(
            seed=base_seed,
            defaults=fixture.simulation_params,
            spread=prior_spread,
        )

    simulator = Simulator(params=simulation_params)
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

    return fixture.model_copy(
        update={
            "ground_truths": ground_truths,
            "simulation_params": simulation_params,
        }
    )
