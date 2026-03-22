from __future__ import annotations

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.harness.budget import Budget
from astar_twin.strategies import REGISTRY
from astar_twin.strategies.filter_baseline.strategy import FilterBaselineStrategy

_INLAND_TEMPLATE = np.array([0.5496, 0.1798, 0.0002, 0.0706, 0.1996, 0.0002], dtype=np.float64)
_COASTAL_TEMPLATE = np.array([0.4798, 0.1698, 0.1198, 0.0604, 0.1700, 0.0002], dtype=np.float64)


def _make_state(grid: list[list[int]]) -> InitialState:
    return InitialState(grid=grid, settlements=[])


@pytest.fixture
def strategy() -> FilterBaselineStrategy:
    return FilterBaselineStrategy()


def test_name(strategy: FilterBaselineStrategy) -> None:
    assert strategy.name == "filter_baseline"


def test_output_shape_and_dtype(strategy: FilterBaselineStrategy) -> None:
    state = _make_state([[10, 11, 4], [11, 1, 11]])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result.shape == (2, 3, NUM_CLASSES)
    assert result.dtype == np.float64


def test_probabilities_sum_to_one(strategy: FilterBaselineStrategy) -> None:
    state = _make_state([[10, 11, 5], [4, 0, 3]])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-9)


def test_water_and_mountains_remain_fixed(strategy: FilterBaselineStrategy) -> None:
    state = _make_state([[TerrainCode.OCEAN, TerrainCode.MOUNTAIN]])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    np.testing.assert_array_equal(
        result[0, 0], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.EMPTY]
    )
    np.testing.assert_array_equal(
        result[0, 1], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.MOUNTAIN]
    )


def test_ports_only_allowed_next_to_water(strategy: FilterBaselineStrategy) -> None:
    state = _make_state(
        [
            [TerrainCode.OCEAN, TerrainCode.PLAINS, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
        ]
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result[0, 1, ClassIndex.PORT] > 0.0
    assert result[1, 1, ClassIndex.PORT] < 0.001  # near-zero for inland
    np.testing.assert_allclose(result[0, 1], _COASTAL_TEMPLATE)
    np.testing.assert_allclose(result[1, 1], _INLAND_TEMPLATE)


def test_diagonal_water_does_not_make_cell_coastal(strategy: FilterBaselineStrategy) -> None:
    state = _make_state(
        [
            [TerrainCode.OCEAN, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.PLAINS],
        ]
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result[1, 1, ClassIndex.PORT] < 0.001  # near-zero for inland
    np.testing.assert_allclose(result[1, 1], _INLAND_TEMPLATE)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (TerrainCode.EMPTY, _INLAND_TEMPLATE),
        (TerrainCode.PLAINS, _INLAND_TEMPLATE),
        (TerrainCode.FOREST, _INLAND_TEMPLATE),
        (TerrainCode.SETTLEMENT, _INLAND_TEMPLATE),
        (TerrainCode.PORT, _INLAND_TEMPLATE),
        (TerrainCode.RUIN, _INLAND_TEMPLATE),
    ],
)
def test_non_static_inland_cells_use_stable_distribution(
    strategy: FilterBaselineStrategy,
    code: int,
    expected: np.ndarray,
) -> None:
    state = _make_state(
        [
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, code, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
        ]
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    np.testing.assert_allclose(result[1, 1], expected)


def test_budget_and_seed_do_not_change_output(strategy: FilterBaselineStrategy) -> None:
    state = _make_state([[10, 11, 4], [11, 5, 3]])
    r1 = strategy.predict(state, budget=Budget(total=10), base_seed=0)
    r2 = strategy.predict(state, budget=Budget(total=50), base_seed=999)
    np.testing.assert_array_equal(r1, r2)


def test_registry_contains_filter_baseline() -> None:
    assert REGISTRY["filter_baseline"] is FilterBaselineStrategy
