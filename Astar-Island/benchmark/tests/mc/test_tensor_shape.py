from __future__ import annotations

import numpy as np

from astar_twin.data.models import RoundFixture
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.scoring import safe_prediction


def test_aggregate_tensor_has_correct_shape(fixture: RoundFixture) -> None:
    runs = MCRunner(Simulator()).run_batch(fixture.initial_states[0], n_runs=5)
    tensor = aggregate_runs(runs, 10, 10)
    assert tensor.shape == (10, 10, 6)


def test_all_cell_probabilities_sum_to_one(fixture: RoundFixture) -> None:
    runs = MCRunner(Simulator()).run_batch(fixture.initial_states[0], n_runs=5)
    tensor = aggregate_runs(runs, 10, 10)
    sums = np.sum(tensor, axis=2)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_no_probability_exactly_zero_after_safe_prediction(fixture: RoundFixture) -> None:
    runs = MCRunner(Simulator()).run_batch(fixture.initial_states[0], n_runs=5)
    tensor = safe_prediction(aggregate_runs(runs, 10, 10))
    assert np.all(tensor > 0.0)


def test_safe_prediction_floors_and_renormalizes() -> None:
    # All-zero input: each class floors to 0.01, then renormalizes to 1/6 each.
    tensor = np.zeros((1, 1, 6), dtype=np.float64)
    result = safe_prediction(tensor)
    # After floor + renorm all values must be > 0 and cell must sum to 1.
    assert np.all(result > 0.0)
    assert np.isclose(float(np.sum(result[0, 0])), 1.0)
    # With uniform input each class gets exactly 1/6.
    assert np.allclose(result[0, 0], 1.0 / 6.0)
