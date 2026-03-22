from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialSettlement, InitialState
from astar_twin.contracts.types import ClassIndex, NUM_CLASSES, TerrainCode
from astar_twin.data.loaders import load_fixture
from astar_twin.harness.budget import Budget
from astar_twin.solver.policy.hotspots import ViewportCandidate
from astar_twin.strategies import REGISTRY
from astar_twin.strategies.reduced_rollout_planner.strategy import (
    _MAX_CATEGORY_CANDIDATES,
    _MAX_REDUCED_ACTIONS,
    ReducedRolloutPlannerStrategy,
)

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def first_initial_state() -> InitialState:
    fixture = load_fixture(FIXTURE_PATH)
    return fixture.initial_states[0]


@pytest.fixture
def strategy() -> ReducedRolloutPlannerStrategy:
    return ReducedRolloutPlannerStrategy()


def _make_hotspot_state() -> InitialState:
    grid = [
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [10, 11, 11, 11, 11, 11, 11, 11, 11, 10],
        [10, 11, 4, 4, 11, 11, 4, 4, 11, 10],
        [10, 11, 4, 1, 11, 11, 4, 1, 11, 10],
        [10, 11, 11, 11, 3, 11, 11, 11, 11, 10],
        [10, 11, 11, 11, 11, 11, 11, 11, 11, 10],
        [10, 11, 4, 4, 11, 11, 4, 4, 11, 10],
        [10, 11, 4, 1, 11, 11, 4, 11, 2, 10],
        [10, 11, 11, 11, 11, 11, 11, 11, 11, 10],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    ]
    settlements = [
        InitialSettlement(x=3, y=3, has_port=False, alive=True),
        InitialSettlement(x=7, y=3, has_port=False, alive=True),
        InitialSettlement(x=8, y=7, has_port=True, alive=True),
    ]
    return InitialState(grid=grid, settlements=settlements)


def test_name(strategy: ReducedRolloutPlannerStrategy) -> None:
    assert strategy.name == "reduced_rollout_planner"


def test_registry_contains_strategy() -> None:
    assert REGISTRY["reduced_rollout_planner"] is ReducedRolloutPlannerStrategy


def test_output_shape_dtype_sum_and_finiteness(
    strategy: ReducedRolloutPlannerStrategy,
    first_initial_state: InitialState,
) -> None:
    height = len(first_initial_state.grid)
    width = len(first_initial_state.grid[0])
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=7)
    assert result.shape == (height, width, NUM_CLASSES)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-9)
    assert np.all(np.isfinite(result))


def test_predict_is_deterministic(
    first_initial_state: InitialState,
) -> None:
    strategy = ReducedRolloutPlannerStrategy()
    result_a = strategy.predict(first_initial_state, budget=Budget(total=6), base_seed=42)
    result_b = strategy.predict(first_initial_state, budget=Budget(total=6), base_seed=42)
    np.testing.assert_array_equal(result_a, result_b)


def test_reset_clears_cross_seed_learning(first_initial_state: InitialState) -> None:
    strategy = ReducedRolloutPlannerStrategy()
    strategy.predict(first_initial_state, budget=Budget(total=6), base_seed=42)
    assert strategy._seed_call_count == 1
    assert strategy._learned_attention_profile is not None

    strategy.reset()

    assert strategy._seed_call_count == 0
    assert strategy._learned_attention_profile is None


def test_new_round_budget_resets_strategy_state(first_initial_state: InitialState) -> None:
    strategy = ReducedRolloutPlannerStrategy()
    first_budget = Budget(total=6)
    strategy.predict(first_initial_state, budget=first_budget, base_seed=42)
    assert strategy._seed_call_count == 1

    second_budget = Budget(total=6)
    result_a = strategy.predict(first_initial_state, budget=second_budget, base_seed=42)
    fresh_result = ReducedRolloutPlannerStrategy().predict(
        first_initial_state,
        budget=Budget(total=6),
        base_seed=42,
    )

    np.testing.assert_array_equal(result_a, fresh_result)


def test_budget_consumption_is_bounded_and_changes_output(
    strategy: ReducedRolloutPlannerStrategy,
    first_initial_state: InitialState,
) -> None:
    tight_budget = Budget(total=1)
    roomy_budget = Budget(total=6)

    tight_result = strategy.predict(first_initial_state, budget=tight_budget, base_seed=99)
    roomy_result = strategy.predict(first_initial_state, budget=roomy_budget, base_seed=99)

    assert tight_budget.used == 1
    assert 1 <= roomy_budget.used <= 3
    assert not np.allclose(tight_result, roomy_result)


def test_reduced_action_set_is_capped(strategy: ReducedRolloutPlannerStrategy) -> None:
    state = _make_hotspot_state()
    candidates = [
        ViewportCandidate(x=0, y=0, w=5, h=5, category="coastal"),
        ViewportCandidate(x=1, y=0, w=5, h=5, category="coastal"),
        ViewportCandidate(x=2, y=0, w=5, h=5, category="coastal"),
        ViewportCandidate(x=0, y=1, w=5, h=5, category="corridor"),
        ViewportCandidate(x=1, y=1, w=5, h=5, category="corridor"),
        ViewportCandidate(x=2, y=1, w=5, h=5, category="corridor"),
        ViewportCandidate(x=0, y=2, w=5, h=5, category="frontier"),
        ViewportCandidate(x=1, y=2, w=5, h=5, category="frontier"),
        ViewportCandidate(x=2, y=2, w=5, h=5, category="frontier"),
        ViewportCandidate(x=0, y=3, w=5, h=5, category="reclaim"),
    ]

    reduced = strategy._select_reduced_actions(candidates, state)

    assert len(reduced) <= _MAX_REDUCED_ACTIONS
    category_counts: dict[str, int] = {}
    for candidate in reduced:
        category_counts[candidate.category] = category_counts.get(candidate.category, 0) + 1
    assert all(count <= _MAX_CATEGORY_CANDIDATES for count in category_counts.values())


def test_depth_two_rollout_score_differs_from_immediate(
    strategy: ReducedRolloutPlannerStrategy,
) -> None:
    first = ViewportCandidate(x=0, y=0, w=5, h=5, category="coastal")
    overlapping = ViewportCandidate(x=1, y=0, w=5, h=5, category="corridor")
    followup = ViewportCandidate(x=5, y=0, w=5, h=5, category="frontier")
    immediate_scores = {
        (0, 0, 5, 5): 1.25,
        (1, 0, 5, 5): 9.0,
        (5, 0, 5, 5): 2.5,
    }

    immediate = immediate_scores[(0, 0, 5, 5)]
    rollout = strategy._score_first_action_with_rollout(
        first,
        [first, overlapping, followup],
        immediate_scores,
    )

    assert rollout > immediate
    assert rollout == pytest.approx(immediate + 0.65 * immediate_scores[(5, 0, 5, 5)])


def test_refinement_selection_prefers_high_entropy_candidate(
    strategy: ReducedRolloutPlannerStrategy,
) -> None:
    first = ViewportCandidate(x=0, y=0, w=5, h=5, category="coastal")
    second = ViewportCandidate(x=5, y=0, w=5, h=5, category="frontier")
    low_entropy = np.zeros((10, 10, NUM_CLASSES), dtype=np.float64)
    low_entropy[:, :, ClassIndex.EMPTY] = 1.0
    high_entropy = np.full((10, 10, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    rollout_cache = {
        (0, 0, 5, 5): low_entropy,
        (5, 0, 5, 5): high_entropy,
    }
    immediate_scores = {
        (0, 0, 5, 5): 5.0,
        (5, 0, 5, 5): 1.0,
    }

    selected = strategy._select_refinement_actions(
        [first, second],
        [first],
        immediate_scores,
        rollout_cache,
        Budget(total=5),
    )

    assert selected == [second]


def test_apply_hard_limits_disallows_inland_ports_and_land_mountains() -> None:
    state = InitialState(
        grid=[
            [TerrainCode.OCEAN, TerrainCode.PLAINS, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.MOUNTAIN],
        ],
        settlements=[],
    )
    raw = np.full((2, 3, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)

    constrained = ReducedRolloutPlannerStrategy()._apply_hard_limits(raw, state, 2, 3)

    np.testing.assert_array_equal(
        constrained[0, 0], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.EMPTY]
    )
    np.testing.assert_array_equal(
        constrained[1, 2], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.MOUNTAIN]
    )
    assert constrained[1, 1, ClassIndex.PORT] == 0.0
    assert constrained[1, 1, ClassIndex.MOUNTAIN] == 0.0


def test_hotspot_state_produces_valid_probabilities() -> None:
    state = _make_hotspot_state()
    result = ReducedRolloutPlannerStrategy().predict(state, budget=Budget(total=5), base_seed=3)
    assert result.shape == (10, 10, NUM_CLASSES)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-9)
    assert np.all(result >= 0.0)
    assert np.all(np.isfinite(result))
    assert result[0, 0].sum() == pytest.approx(1.0)
    assert state.grid[0][0] == TerrainCode.OCEAN
