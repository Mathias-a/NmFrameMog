from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import RoundFixture
from astar_twin.harness.budget import Budget
from astar_twin.strategies.hybrid_solver.strategy import HybridSolverStrategy

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def test_fixture() -> RoundFixture:
    return load_fixture(FIXTURE_PATH)


@pytest.fixture
def first_initial_state(test_fixture: RoundFixture) -> InitialState:
    return test_fixture.initial_states[0]


@pytest.fixture
def strategy(test_fixture: RoundFixture) -> HybridSolverStrategy:
    return HybridSolverStrategy(
        test_fixture,
        n_particles=4,
        n_inner_runs=1,
        sims_per_seed=4,
    )


def test_name(strategy: HybridSolverStrategy) -> None:
    assert strategy.name == "hybrid_solver"


def test_output_shape(strategy: HybridSolverStrategy, first_initial_state: InitialState) -> None:
    h = len(first_initial_state.grid)
    w = len(first_initial_state.grid[0])
    result = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=42)
    assert result.shape == (h, w, NUM_CLASSES)


def test_probabilities_sum_to_one(
    strategy: HybridSolverStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=42)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_output_dtype_is_float64(
    strategy: HybridSolverStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=42)
    assert result.dtype == np.float64


def test_all_probabilities_in_unit_interval(
    strategy: HybridSolverStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=50), base_seed=42)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_deterministic_with_same_seed(
    test_fixture: RoundFixture, first_initial_state: InitialState
) -> None:
    s1 = HybridSolverStrategy(test_fixture, n_particles=4, n_inner_runs=1, sims_per_seed=4)
    s2 = HybridSolverStrategy(test_fixture, n_particles=4, n_inner_runs=1, sims_per_seed=4)
    r1 = s1.predict(first_initial_state, budget=Budget(total=50), base_seed=42)
    r2 = s2.predict(first_initial_state, budget=Budget(total=50), base_seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_caches_across_seeds(test_fixture: RoundFixture) -> None:
    s = HybridSolverStrategy(test_fixture, n_particles=4, n_inner_runs=1, sims_per_seed=4)
    budget = Budget(total=50)
    results = []
    for seed_idx in range(min(test_fixture.seeds_count, 3)):
        initial_state = test_fixture.initial_states[seed_idx]
        t = s.predict(initial_state, budget=budget, base_seed=42)
        results.append(t)

    assert s._solve_result is not None
    assert len(results) == min(test_fixture.seeds_count, 3)
    for t in results:
        np.testing.assert_allclose(t.sum(axis=-1), 1.0, atol=1e-6)


def test_budget_consumed(test_fixture: RoundFixture, first_initial_state: InitialState) -> None:
    s = HybridSolverStrategy(test_fixture, n_particles=4, n_inner_runs=1, sims_per_seed=4)
    budget = Budget(total=50)
    s.predict(first_initial_state, budget=budget, base_seed=42)
    assert budget.used > 0
