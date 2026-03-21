from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.data.loaders import load_fixture
from astar_twin.harness.budget import Budget
from astar_twin.strategies.terrain_aware_mc.strategy import TerrainAwareMCStrategy

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def first_initial_state() -> InitialState:
    fixture = load_fixture(FIXTURE_PATH)
    return fixture.initial_states[0]


@pytest.fixture
def strategy() -> TerrainAwareMCStrategy:
    return TerrainAwareMCStrategy()


def test_name(strategy: TerrainAwareMCStrategy) -> None:
    assert strategy.name == "terrain_aware_mc"


def test_output_shape(strategy: TerrainAwareMCStrategy, first_initial_state: InitialState) -> None:
    H = len(first_initial_state.grid)
    W = len(first_initial_state.grid[0])
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.shape == (H, W, NUM_CLASSES)


def test_probabilities_sum_to_one(
    strategy: TerrainAwareMCStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_output_dtype_is_float64(
    strategy: TerrainAwareMCStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.dtype == np.float64


def test_deterministic_with_same_seed(
    strategy: TerrainAwareMCStrategy, first_initial_state: InitialState
) -> None:
    r1 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    r2 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_different_seeds_produce_different_results(
    strategy: TerrainAwareMCStrategy, first_initial_state: InitialState
) -> None:
    r1 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    r2 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=12345)
    assert not np.allclose(r1, r2)


def test_all_probabilities_in_unit_interval(
    strategy: TerrainAwareMCStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_budget_affects_run_count(
    strategy: TerrainAwareMCStrategy, first_initial_state: InitialState
) -> None:
    """Higher remaining budget should produce different results (more MC runs)."""
    budget_low = Budget(total=50)
    budget_low.consume(48)
    r_low = strategy.predict(first_initial_state, budget=budget_low, base_seed=42)
    r_high = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=42)
    # With different MC run counts, results should differ (unless all cells are static)
    assert not np.allclose(r_low, r_high)
