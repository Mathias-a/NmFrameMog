from __future__ import annotations

import numpy as np

from astar_twin.data.models import RoundFixture
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner, aggregate_runs


def test_historical_fixture_mc_tensor_shape_and_normalization(fixture: RoundFixture) -> None:
    runs = MCRunner(Simulator()).run_batch(fixture.initial_states[0], n_runs=5)
    tensor = aggregate_runs(runs, fixture.map_height, fixture.map_width)
    assert tensor.shape == (fixture.map_height, fixture.map_width, 6)
    assert np.allclose(np.sum(tensor, axis=2), 1.0)
