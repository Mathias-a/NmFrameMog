from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.data.loaders import load_fixture
from astar_twin.harness.budget import Budget
from astar_twin.strategies import REGISTRY
from astar_twin.strategies.mc_challenger.strategy import MCChallengerStrategy

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def first_initial_state() -> InitialState:
    fixture = load_fixture(FIXTURE_PATH)
    return fixture.initial_states[0]


@pytest.fixture
def strategy() -> MCChallengerStrategy:
    return MCChallengerStrategy(n_runs=10)  # small for fast tests


def test_name(strategy: MCChallengerStrategy) -> None:
    assert strategy.name == "mc_challenger"


def test_output_shape(strategy: MCChallengerStrategy, first_initial_state: InitialState) -> None:
    H = len(first_initial_state.grid)
    W = len(first_initial_state.grid[0])
    result = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=0)
    assert result.shape == (H, W, NUM_CLASSES)


def test_probabilities_sum_to_one(
    strategy: MCChallengerStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_output_dtype_is_float64(
    strategy: MCChallengerStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=0)
    assert result.dtype == np.float64


def test_deterministic_with_same_seed(
    strategy: MCChallengerStrategy, first_initial_state: InitialState
) -> None:
    r1 = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=42)
    r2 = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_different_seeds_produce_different_results(
    strategy: MCChallengerStrategy, first_initial_state: InitialState
) -> None:
    r1 = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=0)
    r2 = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=12345)
    assert not np.allclose(r1, r2)


def test_all_probabilities_in_unit_interval(
    strategy: MCChallengerStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=0)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_registry_contains_mc_challenger() -> None:
    assert REGISTRY["mc_challenger"] is MCChallengerStrategy


def test_does_not_consume_budget(
    strategy: MCChallengerStrategy, first_initial_state: InitialState
) -> None:
    budget = Budget(total=50)
    strategy.predict(first_initial_state, budget=budget, base_seed=0)
    assert budget.remaining == 50
