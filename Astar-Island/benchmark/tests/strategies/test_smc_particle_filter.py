from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.data.loaders import load_fixture
from astar_twin.harness.budget import Budget
from astar_twin.strategies import REGISTRY
from astar_twin.strategies.smc_particle_filter.strategy import SMCParticleFilterStrategy

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def first_initial_state() -> InitialState:
    fixture = load_fixture(FIXTURE_PATH)
    return fixture.initial_states[0]


@pytest.fixture
def strategy() -> SMCParticleFilterStrategy:
    """Strategy with small params for fast test execution."""
    return SMCParticleFilterStrategy(
        n_particles=4,
        sims_per_seed=8,
        top_k=2,
        n_inner_runs=2,
        n_bootstrap_viewports=1,
    )


def test_name(strategy: SMCParticleFilterStrategy) -> None:
    assert strategy.name == "smc_particle_filter"


def test_output_shape(
    strategy: SMCParticleFilterStrategy, first_initial_state: InitialState
) -> None:
    H = len(first_initial_state.grid)
    W = len(first_initial_state.grid[0])
    result = strategy.predict(first_initial_state, budget=Budget(total=10), base_seed=0)
    assert result.shape == (H, W, NUM_CLASSES)


def test_probabilities_sum_to_one(
    strategy: SMCParticleFilterStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=10), base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_output_dtype_is_float64(
    strategy: SMCParticleFilterStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=10), base_seed=0)
    assert result.dtype == np.float64


def test_deterministic_with_same_seed(first_initial_state: InitialState) -> None:
    """Two fresh strategy instances with same seed must produce identical output."""
    s1 = SMCParticleFilterStrategy(
        n_particles=4, sims_per_seed=8, top_k=2, n_inner_runs=2, n_bootstrap_viewports=1
    )
    s2 = SMCParticleFilterStrategy(
        n_particles=4, sims_per_seed=8, top_k=2, n_inner_runs=2, n_bootstrap_viewports=1
    )
    r1 = s1.predict(first_initial_state, budget=Budget(total=10), base_seed=42)
    r2 = s2.predict(first_initial_state, budget=Budget(total=10), base_seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_different_seeds_produce_different_results(
    first_initial_state: InitialState,
) -> None:
    s1 = SMCParticleFilterStrategy(
        n_particles=4, sims_per_seed=8, top_k=2, n_inner_runs=2, n_bootstrap_viewports=1
    )
    s2 = SMCParticleFilterStrategy(
        n_particles=4, sims_per_seed=8, top_k=2, n_inner_runs=2, n_bootstrap_viewports=1
    )
    r1 = s1.predict(first_initial_state, budget=Budget(total=10), base_seed=0)
    r2 = s2.predict(first_initial_state, budget=Budget(total=10), base_seed=12345)
    assert not np.allclose(r1, r2)


def test_all_probabilities_in_unit_interval(
    strategy: SMCParticleFilterStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=10), base_seed=0)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_registered_in_registry() -> None:
    assert "smc_particle_filter" in REGISTRY
    assert REGISTRY["smc_particle_filter"] is SMCParticleFilterStrategy


def test_works_with_zero_budget(
    first_initial_state: InitialState,
) -> None:
    """Strategy should still produce valid output when budget is exhausted."""
    s = SMCParticleFilterStrategy(
        n_particles=4, sims_per_seed=8, top_k=2, n_inner_runs=2, n_bootstrap_viewports=1
    )
    H = len(first_initial_state.grid)
    W = len(first_initial_state.grid[0])
    result = s.predict(first_initial_state, budget=Budget(total=0), base_seed=0)
    assert result.shape == (H, W, NUM_CLASSES)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)
