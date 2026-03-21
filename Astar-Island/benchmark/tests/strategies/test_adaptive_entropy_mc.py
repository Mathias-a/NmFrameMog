from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.data.loaders import load_fixture
from astar_twin.harness.budget import Budget
from astar_twin.strategies.adaptive_entropy_mc.strategy import AdaptiveEntropyMCStrategy

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def first_initial_state() -> InitialState:
    fixture = load_fixture(FIXTURE_PATH)
    return fixture.initial_states[0]


@pytest.fixture
def strategy() -> AdaptiveEntropyMCStrategy:
    return AdaptiveEntropyMCStrategy()


def test_name(strategy: AdaptiveEntropyMCStrategy) -> None:
    assert strategy.name == "adaptive_entropy_mc"


def test_output_shape(
    strategy: AdaptiveEntropyMCStrategy, first_initial_state: InitialState
) -> None:
    H = len(first_initial_state.grid)
    W = len(first_initial_state.grid[0])
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.shape == (H, W, NUM_CLASSES)


def test_probabilities_sum_to_one(
    strategy: AdaptiveEntropyMCStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_output_dtype_is_float64(
    strategy: AdaptiveEntropyMCStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.dtype == np.float64


def test_deterministic_with_same_seed(first_initial_state: InitialState) -> None:
    """Two fresh instances with same seed produce identical results.

    Note: we use fresh instances because AdaptiveEntropyMCStrategy is stateful —
    _seed_call_count changes between calls, altering run allocation.
    """
    s1 = AdaptiveEntropyMCStrategy()
    s2 = AdaptiveEntropyMCStrategy()
    r1 = s1.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    r2 = s2.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_different_seeds_produce_different_results(
    strategy: AdaptiveEntropyMCStrategy, first_initial_state: InitialState
) -> None:
    r1 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    r2 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=12345)
    assert not np.allclose(r1, r2)


def test_all_probabilities_in_unit_interval(
    strategy: AdaptiveEntropyMCStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_cross_seed_state_accumulates(first_initial_state: InitialState) -> None:
    """After first predict call, strategy should have learned entropy profile."""
    strategy = AdaptiveEntropyMCStrategy()
    assert strategy._learned_entropy_profile is None
    assert strategy._seed_call_count == 0

    # First call (seed 0 — learning seed)
    strategy.predict(first_initial_state, budget=Budget(total=10), base_seed=42)
    assert strategy._seed_call_count == 1
    assert strategy._learned_entropy_profile is not None

    # Second call (seed 1 — should use learned profile)
    strategy.predict(first_initial_state, budget=Budget(total=10), base_seed=99)
    assert strategy._seed_call_count == 2
    # Learned profile should still be from seed 0
    assert strategy._learned_entropy_profile is not None


def test_reset_clears_state(first_initial_state: InitialState) -> None:
    """reset() should clear cross-seed state."""
    strategy = AdaptiveEntropyMCStrategy()
    strategy.predict(first_initial_state, budget=Budget(total=10), base_seed=42)
    assert strategy._seed_call_count == 1
    assert strategy._learned_entropy_profile is not None

    strategy.reset()
    assert strategy._seed_call_count == 0
    assert strategy._learned_entropy_profile is None


def test_determinism_across_fresh_instances(first_initial_state: InitialState) -> None:
    """Two fresh strategy instances with same seed should produce identical results."""
    s1 = AdaptiveEntropyMCStrategy()
    s2 = AdaptiveEntropyMCStrategy()
    r1 = s1.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    r2 = s2.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_hard_limits_fix_ocean_and_mountain() -> None:
    state = InitialState(
        grid=[[TerrainCode.OCEAN, TerrainCode.MOUNTAIN]],
        settlements=[],
    )
    result = AdaptiveEntropyMCStrategy().predict(state, budget=Budget(total=5), base_seed=42)
    np.testing.assert_array_equal(
        result[0, 0], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.EMPTY]
    )
    np.testing.assert_array_equal(
        result[0, 1], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.MOUNTAIN]
    )


def test_hard_limits_disallow_inland_ports_and_land_mountains() -> None:
    state = InitialState(
        grid=[
            [TerrainCode.OCEAN, TerrainCode.PLAINS, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
        ],
        settlements=[],
    )
    result = AdaptiveEntropyMCStrategy().predict(state, budget=Budget(total=5), base_seed=42)
    assert result[1, 1, ClassIndex.PORT] == 0.0
    assert result[1, 1, ClassIndex.MOUNTAIN] == 0.0
    assert result[0, 1, ClassIndex.PORT] > 0.0
    assert result[0, 1, ClassIndex.MOUNTAIN] == 0.0
