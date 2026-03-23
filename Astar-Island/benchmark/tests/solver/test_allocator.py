"""Tests for viewport allocation policy.

Covers:
  - Budget accounting (never exceeds 50)
  - Bootstrap phase: 2 per seed, correct category selection
  - Adaptive phase: scoring, soft overlap, batch selection
  - Reserve phase: contradiction triggers, release logic
  - Phase transitions
"""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialSettlement, InitialState
from astar_twin.contracts.types import MAX_QUERIES
from astar_twin.solver.inference.posterior import create_posterior
from astar_twin.solver.policy import allocator
from astar_twin.solver.policy.allocator import (
    ADAPTIVE_BATCH_SIZE,
    ADAPTIVE_QUERIES,
    BOOTSTRAP_PER_SEED,
    BOOTSTRAP_QUERIES,
    RESERVE_QUERIES,
    AllocationState,
    check_contradiction_triggers,
    compute_entropy_map,
    compute_posterior_disagreement,
    initialize_allocation,
    plan_bootstrap_queries,
    plan_reserve_queries,
    record_query,
    score_candidate,
    select_adaptive_batch,
    transition_phase,
)
from astar_twin.solver.policy.hotspots import ViewportCandidate

# ---- Fixtures ----


def _make_initial_state(width: int = 10, height: int = 10, n_settlements: int = 3) -> InitialState:
    """Create a minimal initial state for testing."""
    from astar_twin.contracts.types import TerrainCode

    grid: list[list[int]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            if y == 0:
                row.append(TerrainCode.OCEAN)
            elif x < 3 and y < 3:
                row.append(TerrainCode.FOREST)
            else:
                row.append(TerrainCode.PLAINS)
        grid.append(row)

    settlements: list[InitialSettlement] = []
    for i in range(n_settlements):
        settlements.append(
            InitialSettlement(
                x=3 + i,
                y=2,
                has_port=False,
                alive=True,
            )
        )

    return InitialState(grid=grid, settlements=settlements)


def _make_5_initial_states() -> list[InitialState]:
    """Create 5 initial states (one per seed)."""
    return [_make_initial_state() for _ in range(5)]


# ---- Budget constants ----


def test_budget_constants() -> None:
    """Bootstrap + adaptive + reserve = 50."""
    assert BOOTSTRAP_QUERIES + ADAPTIVE_QUERIES + RESERVE_QUERIES == MAX_QUERIES
    assert BOOTSTRAP_QUERIES == 10
    assert ADAPTIVE_QUERIES == 30
    assert RESERVE_QUERIES == 10
    assert BOOTSTRAP_PER_SEED == 2


# ---- Initialization ----


def test_initialize_allocation_creates_candidates_per_seed() -> None:
    """Each seed gets a candidate pool from hotspot generation."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)

    assert alloc.phase == "bootstrap"
    assert alloc.queries_used == 0
    assert alloc.queries_remaining == MAX_QUERIES
    assert len(alloc.seed_candidates) == 5
    for seed_idx in range(5):
        assert len(alloc.seed_candidates[seed_idx]) >= 1


# ---- Bootstrap ----


def test_plan_bootstrap_queries_produces_2_per_seed() -> None:
    """Bootstrap plans exactly 2 queries per seed = 10 total."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    planned = plan_bootstrap_queries(alloc)

    assert len(planned) == BOOTSTRAP_QUERIES
    per_seed: dict[int, list[ViewportCandidate]] = {}
    for seed_idx, vp in planned:
        per_seed.setdefault(seed_idx, []).append(vp)
    for seed_idx in range(5):
        assert len(per_seed.get(seed_idx, [])) == BOOTSTRAP_PER_SEED


def test_bootstrap_two_queries_differ_per_seed() -> None:
    """The two bootstrap queries per seed should be different viewports."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    planned = plan_bootstrap_queries(alloc)

    per_seed: dict[int, list[ViewportCandidate]] = {}
    for seed_idx, vp in planned:
        per_seed.setdefault(seed_idx, []).append(vp)

    for _seed_idx, vps in per_seed.items():
        if len(vps) == 2:
            # Either different position or different category
            a, b = vps
            assert (a.x, a.y, a.w, a.h) != (b.x, b.y, b.w, b.h) or a.category != b.category


# ---- Scoring ----


def test_score_candidate_soft_penalizes_high_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """High-overlap candidates are heavily penalized instead of hard-rejected."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=4, seed=42)

    def zero_disagreement(
        candidate: ViewportCandidate,
        posterior_state: object,
        initial_state: object,
    ) -> float:
        del candidate, posterior_state, initial_state
        return 0.0

    monkeypatch.setattr(allocator, "compute_posterior_disagreement", zero_disagreement)

    queried = ViewportCandidate(x=0, y=0, w=10, h=10, category="coastal")
    candidate = ViewportCandidate(x=0, y=0, w=10, h=10, category="frontier")
    disjoint = ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback")

    overlap_score = score_candidate(
        candidate,
        0,
        posterior,
        state,
        [queried],
    )
    baseline_score = score_candidate(
        disjoint,
        0,
        posterior,
        state,
        [],
    )

    assert np.isfinite(overlap_score)
    assert overlap_score < baseline_score


def test_score_candidate_allows_high_overlap_when_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relaxed overlap mode scores the same candidate less harshly."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=4, seed=42)

    def zero_disagreement(
        candidate: ViewportCandidate,
        posterior_state: object,
        initial_state: object,
    ) -> float:
        del candidate, posterior_state, initial_state
        return 0.0

    monkeypatch.setattr(allocator, "compute_posterior_disagreement", zero_disagreement)

    queried = ViewportCandidate(x=0, y=0, w=10, h=10, category="coastal")
    candidate = ViewportCandidate(x=0, y=0, w=10, h=10, category="frontier")

    strict_score = score_candidate(
        candidate,
        0,
        posterior,
        state,
        [queried],
    )
    relaxed_score = score_candidate(
        candidate,
        0,
        posterior,
        state,
        [queried],
        allow_high_overlap=True,
    )

    assert np.isfinite(strict_score)
    assert relaxed_score > strict_score


def test_score_candidate_with_entropy_map() -> None:
    """Entropy map boosts score for high-entropy regions."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=4, seed=42)

    # High-entropy map
    entropy_map = np.full((10, 10), np.log(6), dtype=np.float64)
    candidate = ViewportCandidate(x=0, y=0, w=5, h=5, category="frontier")

    score_with_entropy = score_candidate(
        candidate,
        0,
        posterior,
        state,
        [],
        entropy_map=entropy_map,
    )

    # Low-entropy map
    entropy_map_low = np.zeros((10, 10), dtype=np.float64)
    score_without_entropy = score_candidate(
        candidate,
        0,
        posterior,
        state,
        [],
        entropy_map=entropy_map_low,
    )

    assert score_with_entropy > score_without_entropy


# ---- Entropy map ----


def test_compute_entropy_map_shape() -> None:
    """Entropy map has same H×W shape as input."""
    tensor = np.ones((10, 10, 6), dtype=np.float64) / 6.0
    entropy = compute_entropy_map(tensor)
    assert entropy.shape == (10, 10)
    # Uniform distribution → max entropy
    expected = np.log(6)
    np.testing.assert_allclose(entropy, expected, atol=1e-5)


def test_compute_entropy_map_deterministic() -> None:
    """Certain prediction → zero entropy."""
    tensor = np.zeros((10, 10, 6), dtype=np.float64)
    tensor[:, :, 0] = 1.0  # all class-0
    entropy = compute_entropy_map(tensor)
    assert entropy.shape == (10, 10)
    np.testing.assert_allclose(entropy, 0.0, atol=1e-5)


# ---- Adaptive selection ----


def test_select_adaptive_batch_respects_budget() -> None:
    """Adaptive batch never exceeds remaining budget."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    posterior = create_posterior(n_particles=4, seed=42)

    # Simulate that 48 queries already used
    for i in range(48):
        vp = ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback")
        record_query(alloc, i % 5, vp, "adaptive")

    batch = select_adaptive_batch(alloc, posterior, states, batch_size=5)
    assert len(batch) <= 2  # Only 2 queries remaining


def test_select_adaptive_batch_accepts_per_seed_predictions() -> None:
    """Adaptive selection accepts per-seed prediction tensors."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    posterior = create_posterior(n_particles=4, seed=42)
    seed_predictions = {
        seed_idx: np.ones((10, 10, 6), dtype=np.float64) / 6.0 for seed_idx in range(5)
    }

    batch = select_adaptive_batch(
        alloc,
        posterior,
        states,
        seed_predictions=seed_predictions,
        batch_size=ADAPTIVE_BATCH_SIZE,
    )

    assert len(batch) <= ADAPTIVE_BATCH_SIZE
    assert all(seed_idx in seed_predictions for seed_idx, _ in batch)


def test_select_adaptive_batch_backfills_nonredundant_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adaptive selection backfills later candidates when top scores are redundant."""
    states = [_make_initial_state(width=30, height=10)]
    alloc = AllocationState(
        seed_candidates={
            0: [
                ViewportCandidate(x=0, y=0, w=10, h=10, category="frontier"),
                ViewportCandidate(x=0, y=0, w=10, h=10, category="corridor"),
                ViewportCandidate(x=20, y=0, w=10, h=10, category="fallback"),
            ]
        }
    )
    posterior = create_posterior(n_particles=4, seed=42)
    score_map = {
        (0, 0, "frontier"): 1.0,
        (0, 0, "corridor"): 0.95,
        (20, 0, "fallback"): 0.75,
    }

    def fake_score_candidate(
        candidate: ViewportCandidate,
        *args: object,
        **kwargs: object,
    ) -> float:
        return score_map[(candidate.x, candidate.y, candidate.category)]

    monkeypatch.setattr(allocator, "score_candidate", fake_score_candidate)

    batch = select_adaptive_batch(alloc, posterior, states, batch_size=2)

    assert [(seed_idx, vp.x, vp.y, vp.category) for seed_idx, vp in batch] == [
        (0, 0, 0, "frontier"),
        (0, 20, 0, "fallback"),
    ]


def test_select_adaptive_batch_soft_balances_underserved_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Soft seed balancing can elevate an underserved seed into the batch."""
    states = [
        _make_initial_state(width=30, height=10),
        _make_initial_state(width=30, height=10),
    ]
    alloc = AllocationState(
        seed_candidates={
            0: [
                ViewportCandidate(x=0, y=0, w=10, h=10, category="frontier"),
                ViewportCandidate(x=10, y=0, w=10, h=10, category="fallback"),
            ],
            1: [ViewportCandidate(x=20, y=0, w=10, h=10, category="coastal")],
        }
    )
    posterior = create_posterior(n_particles=4, seed=42)
    previous = ViewportCandidate(x=0, y=0, w=5, h=5, category="existing")
    record_query(alloc, 0, previous, "adaptive")
    record_query(alloc, 0, previous, "adaptive")

    score_map = {
        (0, 0, "frontier"): 1.0,
        (10, 0, "fallback"): 0.9,
        (20, 0, "coastal"): 0.85,
    }

    def fake_score_candidate(
        candidate: ViewportCandidate,
        *args: object,
        **kwargs: object,
    ) -> float:
        return score_map[(candidate.x, candidate.y, candidate.category)]

    monkeypatch.setattr(allocator, "score_candidate", fake_score_candidate)

    batch = select_adaptive_batch(alloc, posterior, states, batch_size=2)

    assert [seed_idx for seed_idx, _ in batch] == [0, 1]


# ---- Phase transitions ----


def test_phase_transitions() -> None:
    """Phase transitions happen at correct query counts."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    assert alloc.phase == "bootstrap"

    # Record 10 bootstrap queries
    for i in range(BOOTSTRAP_QUERIES):
        vp = ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback")
        record_query(alloc, i % 5, vp, "bootstrap")
    transition_phase(alloc)
    assert alloc.phase == "adaptive"

    # Record 30 adaptive queries
    for i in range(ADAPTIVE_QUERIES):
        vp = ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback")
        record_query(alloc, i % 5, vp, "adaptive")
    transition_phase(alloc)
    assert alloc.phase == "reserve"

    # Record 10 reserve queries
    for i in range(RESERVE_QUERIES):
        vp = ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback")
        record_query(alloc, i % 5, vp, "reserve")
    transition_phase(alloc)
    assert alloc.phase == "done"

    assert alloc.queries_used == MAX_QUERIES
    assert alloc.queries_remaining == 0


# ---- Contradiction triggers ----


def test_contradiction_ess_trigger() -> None:
    """ESS < 6 fires contradiction trigger."""
    alloc = AllocationState()
    # Create a posterior with collapsed weights
    posterior = create_posterior(n_particles=8, seed=0)
    # Force one particle to dominate
    posterior.particles[0].log_weight = 100.0
    for p in posterior.particles[1:]:
        p.log_weight = -1000.0

    assert posterior.ess < 6.0
    assert check_contradiction_triggers(alloc, posterior) is True


def test_no_contradiction_with_healthy_posterior() -> None:
    """No trigger with healthy ESS and enough queries per seed."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    posterior = create_posterior(n_particles=8, seed=42)
    # All equal weights → ESS = n_particles
    assert posterior.ess >= 6.0
    assert check_contradiction_triggers(alloc, posterior) is False


def test_under_queried_seed_trigger() -> None:
    """After adaptive done, seed with < MIN_QUERIES_PER_SEED fires trigger."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)

    # Record 10 bootstrap
    for i in range(BOOTSTRAP_QUERIES):
        vp = ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback")
        record_query(alloc, i % 5, vp, "bootstrap")
    transition_phase(alloc)

    # Record 30 adaptive, all to seed 0
    for _i in range(ADAPTIVE_QUERIES):
        vp = ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback")
        record_query(alloc, 0, vp, "adaptive")
    transition_phase(alloc)

    # Seed 1-4 each have only 2 bootstrap queries < 8
    posterior = create_posterior(n_particles=8, seed=42)
    assert check_contradiction_triggers(alloc, posterior) is True


def test_disagreement_trigger_fires_contradiction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disagreement can trigger reserve release even when ESS is healthy."""
    states = [_make_initial_state(width=20, height=10)]
    alloc = AllocationState(
        seed_candidates={0: [ViewportCandidate(x=0, y=0, w=10, h=10, category="frontier")]}
    )
    posterior = create_posterior(n_particles=8, seed=42)

    def constant_score(
        candidate: ViewportCandidate,
        seed_index: int,
        posterior_state: object,
        initial_state: object,
        queried_viewports: list[ViewportCandidate],
        entropy_map: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
        allow_high_overlap: bool = False,
        observation_ledger: object | None = None,
    ) -> float:
        del (
            candidate,
            seed_index,
            posterior_state,
            initial_state,
            queried_viewports,
            entropy_map,
            allow_high_overlap,
            observation_ledger,
        )
        return 1.0

    def always_disagree(
        candidate: ViewportCandidate,
        posterior_state: object,
        initial_state: object,
    ) -> bool:
        del candidate, posterior_state, initial_state
        return True

    monkeypatch.setattr(allocator, "score_candidate", constant_score)
    monkeypatch.setattr(allocator, "check_argmax_disagreement", always_disagree)

    assert check_contradiction_triggers(alloc, posterior, initial_states=states) is True


# ---- Reserve planning ----


def test_reserve_queries_with_contradiction() -> None:
    """Reserve queries are selected with relaxed overlap when contradiction fires."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)

    # Fill bootstrap + adaptive
    for i in range(BOOTSTRAP_QUERIES + ADAPTIVE_QUERIES):
        vp = ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback")
        phase = "bootstrap" if i < BOOTSTRAP_QUERIES else "adaptive"
        record_query(alloc, i % 5, vp, phase)
    alloc.phase = "reserve"

    # Force ESS collapse
    posterior = create_posterior(n_particles=8, seed=0)
    posterior.particles[0].log_weight = 100.0
    for p in posterior.particles[1:]:
        p.log_weight = -1000.0

    reserve = plan_reserve_queries(alloc, posterior, states)
    assert len(reserve) <= RESERVE_QUERIES
    assert len(reserve) > 0  # Should produce queries


def test_reserve_queries_accepts_per_seed_predictions() -> None:
    """Reserve planning accepts per-seed prediction tensors."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    posterior = create_posterior(n_particles=8, seed=42)
    seed_predictions = {
        seed_idx: np.ones((10, 10, 6), dtype=np.float64) / 6.0 for seed_idx in range(5)
    }

    reserve = plan_reserve_queries(
        alloc,
        posterior,
        states,
        seed_predictions=seed_predictions,
    )

    assert len(reserve) <= RESERVE_QUERIES


@pytest.mark.parametrize(
    ("override", "expected_category"),
    [
        (False, "adaptive"),
        (True, "contradiction"),
    ],
)
def test_plan_reserve_queries_honors_contradiction_override(
    monkeypatch: pytest.MonkeyPatch,
    override: bool,
    expected_category: str,
) -> None:
    """Explicit contradiction override chooses the reserve branch directly."""
    states = [_make_initial_state(width=20, height=10)]
    alloc = AllocationState(
        phase="reserve",
        seed_candidates={0: [ViewportCandidate(x=0, y=0, w=10, h=10, category="frontier")]},
    )
    posterior = create_posterior(n_particles=8, seed=42)
    adaptive_result = [(0, ViewportCandidate(x=10, y=0, w=5, h=5, category="adaptive"))]
    contradiction_result = [(0, ViewportCandidate(x=0, y=0, w=5, h=5, category="contradiction"))]

    def fail_if_recomputed(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("check_contradiction_triggers should not be recomputed")

    def return_adaptive_batch(
        state: AllocationState,
        posterior_state: object,
        initial_states: list[InitialState],
        seed_predictions: dict[int, np.ndarray[tuple[int, int, int], np.dtype[np.float64]]]
        | None = None,
        batch_size: int = ADAPTIVE_BATCH_SIZE,
        observation_ledger: object | None = None,
    ) -> list[tuple[int, ViewportCandidate]]:
        del state, posterior_state, initial_states, seed_predictions, batch_size, observation_ledger
        return adaptive_result

    def return_contradiction_batch(
        state: AllocationState,
        posterior_state: object,
        initial_states: list[InitialState],
        seed_predictions: dict[int, np.ndarray[tuple[int, int, int], np.dtype[np.float64]]] | None,
        n_queries: int,
        observation_ledger: object | None = None,
    ) -> list[tuple[int, ViewportCandidate]]:
        del state, posterior_state, initial_states, seed_predictions, n_queries, observation_ledger
        return contradiction_result

    monkeypatch.setattr(allocator, "check_contradiction_triggers", fail_if_recomputed)
    monkeypatch.setattr(allocator, "select_adaptive_batch", return_adaptive_batch)
    monkeypatch.setattr(allocator, "_select_contradiction_queries", return_contradiction_batch)

    reserve = plan_reserve_queries(
        alloc,
        posterior,
        states,
        contradiction_triggered=override,
        n_queries=1,
    )

    assert len(reserve) == 1
    assert reserve[0][1].category == expected_category


# ---- Record + query tracking ----


def test_record_query_increments_count() -> None:
    """Recording queries increments budget counters."""
    alloc = AllocationState()
    vp = ViewportCandidate(x=0, y=0, w=10, h=10, category="coastal")
    record_query(alloc, 0, vp, "bootstrap")
    assert alloc.queries_used == 1
    assert alloc.queries_remaining == MAX_QUERIES - 1
    assert len(alloc.queries_for_seed(0)) == 1
    assert len(alloc.queries_for_seed(1)) == 0


# ---- Posterior disagreement ----


def test_posterior_disagreement_equal_weights() -> None:
    """Equal-weight particles still produce a valid disagreement fraction."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=10, seed=42)
    candidate = ViewportCandidate(x=0, y=0, w=5, h=5, category="frontier")

    disagreement = compute_posterior_disagreement(candidate, posterior, state)
    assert isinstance(disagreement, float)
    assert 0.0 <= disagreement <= 1.0


def test_posterior_disagreement_collapsed() -> None:
    """Collapsed posterior: disagreement is in [0, 1] and function is deterministic."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=10, seed=42)
    posterior.particles[0].log_weight = 100.0
    for p in posterior.particles[1:]:
        p.log_weight = -1000.0

    candidate = ViewportCandidate(x=0, y=0, w=5, h=5, category="frontier")
    d1 = compute_posterior_disagreement(candidate, posterior, state)
    d2 = compute_posterior_disagreement(candidate, posterior, state)
    assert 0.0 <= d1 <= 1.0
    assert d1 == d2


def test_disagreement_returns_valid_range() -> None:
    """Real disagreement is always clipped into [0, 1]."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=4, seed=7)
    candidate = ViewportCandidate(x=1, y=1, w=4, h=4, category="frontier")

    disagreement = compute_posterior_disagreement(candidate, posterior, state)

    assert 0.0 <= disagreement <= 1.0


def test_disagreement_single_particle_zero() -> None:
    """A single-particle posterior has no top-2 disagreement."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=1, seed=42)
    candidate = ViewportCandidate(x=0, y=0, w=5, h=5, category="frontier")

    disagreement = compute_posterior_disagreement(candidate, posterior, state)

    assert disagreement == 0.0
