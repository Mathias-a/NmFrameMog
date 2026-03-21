from __future__ import annotations

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialSettlement, InitialState
from astar_twin.strategies.uniform.strategy import UniformStrategy


@pytest.fixture
def initial_state() -> InitialState:
    grid = [[10, 11, 4], [11, 1, 11], [5, 11, 4]]
    return InitialState(
        grid=grid,
        settlements=[InitialSettlement(x=1, y=1, has_port=False, alive=True)],
    )


class TestUniformStrategy:
    def test_output_shape(self, initial_state: InitialState) -> None:
        strategy = UniformStrategy()
        result = strategy.predict(initial_state, budget=50, base_seed=0)
        assert result.shape == (3, 3, 6)

    def test_probabilities_sum_to_one(self, initial_state: InitialState) -> None:
        strategy = UniformStrategy()
        result = strategy.predict(initial_state, budget=50, base_seed=0)
        assert np.allclose(result.sum(axis=2), 1.0)

    def test_all_classes_equal(self, initial_state: InitialState) -> None:
        strategy = UniformStrategy()
        result = strategy.predict(initial_state, budget=50, base_seed=0)
        assert np.allclose(result, 1.0 / 6.0)

    def test_deterministic_across_seeds(self, initial_state: InitialState) -> None:
        strategy = UniformStrategy()
        r1 = strategy.predict(initial_state, budget=50, base_seed=0)
        r2 = strategy.predict(initial_state, budget=50, base_seed=99)
        assert np.array_equal(r1, r2)

    def test_name(self) -> None:
        assert UniformStrategy().name == "uniform"

    def test_dtype_is_float64(self, initial_state: InitialState) -> None:
        result = UniformStrategy().predict(initial_state, budget=50, base_seed=0)
        assert result.dtype == np.float64

    def test_respects_map_dimensions(self) -> None:
        grid = [[10] * 7 for _ in range(5)]
        state = InitialState(grid=grid, settlements=[])
        result = UniformStrategy().predict(state, budget=50, base_seed=0)
        assert result.shape == (5, 7, 6)
