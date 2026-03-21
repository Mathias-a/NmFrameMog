from __future__ import annotations

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, TERRAIN_TO_CLASS, TerrainCode
from astar_twin.strategies.initial_prior.strategy import InitialPriorStrategy


def _make_state(grid: list[list[int]]) -> InitialState:
    return InitialState(grid=grid, settlements=[])


def _uniform_grid(code: int, h: int = 4, w: int = 4) -> list[list[int]]:
    return [[code] * w for _ in range(h)]


@pytest.fixture
def strategy() -> InitialPriorStrategy:
    return InitialPriorStrategy()


def test_name(strategy: InitialPriorStrategy) -> None:
    assert strategy.name == "initial_prior"


def test_output_shape(strategy: InitialPriorStrategy) -> None:
    state = _make_state(_uniform_grid(TerrainCode.OCEAN))
    result = strategy.predict(state, budget=50, base_seed=0)
    assert result.shape == (4, 4, NUM_CLASSES)


def test_probabilities_sum_to_one(strategy: InitialPriorStrategy) -> None:
    state = _make_state(_uniform_grid(TerrainCode.FOREST))
    result = strategy.predict(state, budget=50, base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_dominant_class_is_correct_for_ocean(strategy: InitialPriorStrategy) -> None:
    state = _make_state(_uniform_grid(TerrainCode.OCEAN))
    result = strategy.predict(state, budget=50, base_seed=0)
    dominant = result[0, 0].argmax()
    assert dominant == TERRAIN_TO_CLASS[TerrainCode.OCEAN]


def test_dominant_class_is_correct_for_mountain(strategy: InitialPriorStrategy) -> None:
    state = _make_state(_uniform_grid(TerrainCode.MOUNTAIN))
    result = strategy.predict(state, budget=50, base_seed=0)
    dominant = result[0, 0].argmax()
    assert dominant == TERRAIN_TO_CLASS[TerrainCode.MOUNTAIN]


def test_dominant_class_is_correct_for_forest(strategy: InitialPriorStrategy) -> None:
    state = _make_state(_uniform_grid(TerrainCode.FOREST))
    result = strategy.predict(state, budget=50, base_seed=0)
    dominant = result[0, 0].argmax()
    assert dominant == TERRAIN_TO_CLASS[TerrainCode.FOREST]


def test_dominant_probability_is_high_conf(strategy: InitialPriorStrategy) -> None:
    state = _make_state(_uniform_grid(TerrainCode.PLAINS))
    result = strategy.predict(state, budget=50, base_seed=0)
    dominant_cls = TERRAIN_TO_CLASS[TerrainCode.PLAINS]
    np.testing.assert_allclose(result[0, 0, dominant_cls], 0.9, atol=1e-9)


def test_mixed_grid_assigns_correct_classes(strategy: InitialPriorStrategy) -> None:
    grid = [
        [TerrainCode.OCEAN, TerrainCode.MOUNTAIN],
        [TerrainCode.FOREST, TerrainCode.PLAINS],
    ]
    state = _make_state(grid)
    result = strategy.predict(state, budget=50, base_seed=0)
    for row_idx, row in enumerate(grid):
        for col_idx, code in enumerate(row):
            expected_cls = TERRAIN_TO_CLASS[code]
            assert result[row_idx, col_idx].argmax() == expected_cls


def test_output_dtype_is_float64(strategy: InitialPriorStrategy) -> None:
    state = _make_state(_uniform_grid(TerrainCode.OCEAN))
    result = strategy.predict(state, budget=50, base_seed=0)
    assert result.dtype == np.float64


def test_budget_and_seed_do_not_affect_output(strategy: InitialPriorStrategy) -> None:
    state = _make_state(_uniform_grid(TerrainCode.OCEAN))
    r1 = strategy.predict(state, budget=10, base_seed=0)
    r2 = strategy.predict(state, budget=50, base_seed=999)
    np.testing.assert_array_equal(r1, r2)
