from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.data.loaders import load_fixture
from astar_twin.harness.budget import Budget
from astar_twin.strategies.mc_oracle.strategy import MCOracleStrategy

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def first_initial_state() -> InitialState:
    fixture = load_fixture(FIXTURE_PATH)
    return fixture.initial_states[0]


@pytest.fixture
def strategy() -> MCOracleStrategy:
    return MCOracleStrategy()


def test_name(strategy: MCOracleStrategy) -> None:
    assert strategy.name == "mc_oracle"


def test_output_shape(strategy: MCOracleStrategy, first_initial_state: InitialState) -> None:
    H = len(first_initial_state.grid)
    W = len(first_initial_state.grid[0])
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.shape == (H, W, NUM_CLASSES)


def test_probabilities_sum_to_one(
    strategy: MCOracleStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_output_dtype_is_float64(
    strategy: MCOracleStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.dtype == np.float64


def test_deterministic_with_same_seed(
    strategy: MCOracleStrategy, first_initial_state: InitialState
) -> None:
    r1 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    r2 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_different_seeds_produce_different_results(
    strategy: MCOracleStrategy, first_initial_state: InitialState
) -> None:
    r1 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    r2 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=12345)
    assert not np.allclose(r1, r2)


def test_all_probabilities_in_unit_interval(
    strategy: MCOracleStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
