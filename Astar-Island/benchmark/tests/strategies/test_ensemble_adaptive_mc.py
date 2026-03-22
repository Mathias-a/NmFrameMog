from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialSettlement, InitialState
from astar_twin.contracts.types import ClassIndex, NUM_CLASSES, TerrainCode
from astar_twin.data.loaders import load_fixture
from astar_twin.harness.budget import Budget
from astar_twin.strategies import REGISTRY
from astar_twin.strategies.ensemble_adaptive_mc.strategy import EnsembleAdaptiveMCStrategy

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def first_initial_state() -> InitialState:
    fixture = load_fixture(FIXTURE_PATH)
    return fixture.initial_states[0]


def test_name_and_registry() -> None:
    strategy = EnsembleAdaptiveMCStrategy()
    assert strategy.name == "ensemble_adaptive_mc"
    assert REGISTRY["ensemble_adaptive_mc"] is EnsembleAdaptiveMCStrategy


def test_predict_returns_valid_tensor(first_initial_state: InitialState) -> None:
    strategy = EnsembleAdaptiveMCStrategy()
    height = len(first_initial_state.grid)
    width = len(first_initial_state.grid[0])

    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=42)

    assert result.shape == (height, width, NUM_CLASSES)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-9)


def test_predict_is_deterministic_across_fresh_instances(first_initial_state: InitialState) -> None:
    result_a = EnsembleAdaptiveMCStrategy().predict(
        first_initial_state, budget=Budget(total=5), base_seed=7
    )
    result_b = EnsembleAdaptiveMCStrategy().predict(
        first_initial_state, budget=Budget(total=5), base_seed=7
    )
    np.testing.assert_array_equal(result_a, result_b)


def test_reset_clears_state(first_initial_state: InitialState) -> None:
    strategy = EnsembleAdaptiveMCStrategy()
    strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=13)

    assert strategy._seed_call_count == 1
    assert strategy._remaining_run_bank < 1000

    strategy.reset()

    assert strategy._seed_call_count == 0
    assert strategy._remaining_run_bank == 1000


def test_new_budget_resets_round_state(first_initial_state: InitialState) -> None:
    strategy = EnsembleAdaptiveMCStrategy()
    strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=21)
    assert strategy._seed_call_count == 1

    reset_result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=21)
    fresh_result = EnsembleAdaptiveMCStrategy().predict(
        first_initial_state, budget=Budget(total=5), base_seed=21
    )

    np.testing.assert_array_equal(reset_result, fresh_result)


def test_apply_hard_mask_disallows_inland_ports_and_land_mountains() -> None:
    state = InitialState(
        grid=[
            [TerrainCode.OCEAN, TerrainCode.PLAINS, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.MOUNTAIN],
        ],
        settlements=[InitialSettlement(x=1, y=1, has_port=False, alive=True)],
    )
    raw = np.full((2, 3, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)

    constrained = EnsembleAdaptiveMCStrategy()._apply_hard_mask(raw, state, 2, 3)

    np.testing.assert_array_equal(
        constrained[0, 0], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.EMPTY]
    )
    np.testing.assert_array_equal(
        constrained[1, 2], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.MOUNTAIN]
    )
    assert constrained[1, 1, ClassIndex.PORT] == 0.0
    assert constrained[1, 1, ClassIndex.MOUNTAIN] == 0.0
    assert constrained[0, 1, ClassIndex.PORT] > 0.0
