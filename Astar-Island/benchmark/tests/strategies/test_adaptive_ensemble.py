from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.data.loaders import load_fixture
from astar_twin.harness.budget import Budget
from astar_twin.scoring import safe_prediction
from astar_twin.strategies.adaptive_ensemble.bundles import get_bundles
from astar_twin.strategies.adaptive_ensemble.strategy import AdaptiveEnsembleStrategy

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def first_initial_state() -> InitialState:
    fixture = load_fixture(FIXTURE_PATH)
    return fixture.initial_states[0]


@pytest.fixture
def strategy() -> AdaptiveEnsembleStrategy:
    return AdaptiveEnsembleStrategy(n_runs=18)


def test_output_shape(
    strategy: AdaptiveEnsembleStrategy, first_initial_state: InitialState
) -> None:
    height = len(first_initial_state.grid)
    width = len(first_initial_state.grid[0])
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.shape == (height, width, NUM_CLASSES)


def test_probabilities_sum_to_one(
    strategy: AdaptiveEnsembleStrategy,
    first_initial_state: InitialState,
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_output_dtype_is_float64(
    strategy: AdaptiveEnsembleStrategy,
    first_initial_state: InitialState,
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.dtype == np.float64


def test_deterministic_with_same_seed(
    strategy: AdaptiveEnsembleStrategy,
    first_initial_state: InitialState,
) -> None:
    first = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    second = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_produce_different_results(
    strategy: AdaptiveEnsembleStrategy,
    first_initial_state: InitialState,
) -> None:
    first = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    second = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=12345)
    assert not np.allclose(first, second)


def test_all_probabilities_in_unit_interval(
    strategy: AdaptiveEnsembleStrategy,
    first_initial_state: InitialState,
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_no_zero_probabilities_after_safe_prediction(
    strategy: AdaptiveEnsembleStrategy,
    first_initial_state: InitialState,
) -> None:
    raw = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    safe = safe_prediction(raw)
    assert float(np.min(safe)) >= 0.01 - 1e-9


def test_static_cells_have_high_confidence(first_initial_state: InitialState) -> None:
    strategy = AdaptiveEnsembleStrategy(n_runs=18, static_confidence=0.97)
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)

    for y, row in enumerate(first_initial_state.grid):
        for x, code in enumerate(row):
            if code == TerrainCode.OCEAN:
                assert result[y, x, ClassIndex.EMPTY] >= 0.90
            elif code == TerrainCode.MOUNTAIN:
                assert result[y, x, ClassIndex.MOUNTAIN] >= 0.90


def test_bundle_count() -> None:
    assert len(get_bundles()) == 9


def test_bundle_weights_sum_to_one() -> None:
    total_weight = sum(bundle.prior_weight for bundle in get_bundles())
    assert total_weight == pytest.approx(1.0)
