"""Tests for viewport allocation policy.

Covers:
  - Budget accounting (never exceeds 50)
  - Bootstrap phase: 2 per seed, correct category selection
  - Adaptive phase: scoring, overlap rejection, batch selection
  - Reserve phase: contradiction triggers, release logic
  - Phase transitions
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialSettlement, InitialState
from astar_twin.contracts.types import ClassIndex, TerrainCode
from astar_twin.solver.inference.posterior import PosteriorState, create_posterior
from astar_twin.solver.policy.allocator import (
    ADAPTIVE_BATCH_SIZE,
    ADAPTIVE_QUERIES,
    BOOTSTRAP_PER_SEED,
    BOOTSTRAP_QUERIES,
    MAX_QUERIES,
    RERANK_EIG_WEIGHT,
    RERANK_HEURISTIC_WEIGHT,
    RESERVE_QUERIES,
    AllocationState,
    ViewportCandidate,
    approximate_expected_information_gain,
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
from astar_twin.solver.policy.legality import apply_legality_filter

# ---- Fixtures ----


def _make_initial_state(width: int = 10, height: int = 10, n_settlements: int = 3) -> InitialState:
    """Create a minimal initial state for testing."""

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
    per_seed = {}
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


def test_score_candidate_respects_overlap_rejection() -> None:
    """Candidate with >60% overlap is rejected (score = -1)."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=4, seed=42)

    # Pre-queried viewport
    queried = ViewportCandidate(x=0, y=0, w=10, h=10, category="coastal")
    # Candidate that fully overlaps
    candidate = ViewportCandidate(x=0, y=0, w=10, h=10, category="frontier")

    score = score_candidate(
        candidate,
        0,
        posterior,
        state,
        [queried],
        queries_used=15,
    )
    assert score == -1.0


def test_score_candidate_allows_high_overlap_when_flagged() -> None:
    """When allow_high_overlap=True, overlapping candidate is scored normally."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=4, seed=42)

    queried = ViewportCandidate(x=0, y=0, w=10, h=10, category="coastal")
    candidate = ViewportCandidate(x=0, y=0, w=10, h=10, category="frontier")

    score = score_candidate(
        candidate,
        0,
        posterior,
        state,
        [queried],
        queries_used=15,
        allow_high_overlap=True,
    )
    assert score != -1.0  # Scored normally despite overlap


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
        queries_used=15,
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
        queries_used=15,
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


def test_apply_legality_filter_removes_illegal_mass_and_preserves_statics() -> None:
    state = InitialState(
        grid=[
            [TerrainCode.OCEAN, TerrainCode.PLAINS, TerrainCode.MOUNTAIN],
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
            [TerrainCode.OCEAN, TerrainCode.PLAINS, TerrainCode.PLAINS],
        ],
        settlements=[],
    )
    tensor = np.full((3, 3, 6), 1.0 / 6.0, dtype=np.float64)

    filtered = apply_legality_filter(tensor, state)

    np.testing.assert_allclose(filtered[0, 0, ClassIndex.EMPTY], 1.0)
    np.testing.assert_allclose(filtered[0, 0, 1:], 0.0)
    np.testing.assert_allclose(filtered[0, 2, ClassIndex.MOUNTAIN], 1.0)
    np.testing.assert_allclose(
        filtered[1, 1, [ClassIndex.PORT, ClassIndex.MOUNTAIN]],
        0.0,
    )
    assert filtered[1, 1, ClassIndex.SETTLEMENT] > 0.0
    assert filtered[1, 1, ClassIndex.EMPTY] > 0.0


def test_apply_legality_filter_fallbacks_when_only_illegal_mass_present() -> None:
    state = InitialState(
        grid=[
            [TerrainCode.PLAINS, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.OCEAN],
        ],
        settlements=[],
    )
    tensor = np.zeros((2, 2, 6), dtype=np.float64)
    tensor[0, 0, ClassIndex.PORT] = 1.0
    tensor[0, 1, ClassIndex.MOUNTAIN] = 1.0
    tensor[1, 1, ClassIndex.MOUNTAIN] = 1.0

    filtered = apply_legality_filter(tensor, state)

    np.testing.assert_allclose(filtered[1, 1, ClassIndex.EMPTY], 1.0)
    np.testing.assert_allclose(filtered[0, 0, ClassIndex.PORT], 0.0)
    np.testing.assert_allclose(filtered[0, 0, ClassIndex.MOUNTAIN], 0.0)
    np.testing.assert_allclose(np.sum(filtered[0, 0]), 1.0)
    assert np.all(
        filtered[
            0, 0, [ClassIndex.EMPTY, ClassIndex.SETTLEMENT, ClassIndex.RUIN, ClassIndex.FOREST]
        ]
        > 0.0
    )


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


def test_select_adaptive_batch_shortlists_then_reranks(monkeypatch: pytest.MonkeyPatch) -> None:
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    posterior = create_posterior(n_particles=4, seed=42)

    candidates = [
        ViewportCandidate(x=0, y=0, w=5, h=5, category="frontier"),
        ViewportCandidate(x=5, y=0, w=5, h=5, category="fallback"),
        ViewportCandidate(x=0, y=5, w=5, h=5, category="fallback"),
        ViewportCandidate(x=5, y=5, w=5, h=5, category="fallback"),
    ]
    alloc.seed_candidates = {0: candidates}

    heuristic_scores = {
        (0, 0): 0.90,
        (5, 0): 0.80,
        (0, 5): 0.70,
        (5, 5): 0.10,
    }

    eig_scores = {
        (0, 0): 0.20,
        (5, 0): 0.10,
        (0, 5): 0.95,
        (5, 5): 0.99,
    }

    def fake_score_candidate(
        candidate: ViewportCandidate,
        seed_index: int,
        posterior_state: PosteriorState,
        initial_state: InitialState,
        queried_viewports: list[ViewportCandidate],
        queries_used: int,
        entropy_map: NDArray[np.float64] | None = None,
        allow_high_overlap: bool = False,
    ) -> float:
        del (
            seed_index,
            posterior_state,
            initial_state,
            queried_viewports,
            queries_used,
            entropy_map,
            allow_high_overlap,
        )
        return heuristic_scores[(candidate.x, candidate.y)]

    def fake_eig(
        candidate: ViewportCandidate,
        posterior_state: PosteriorState,
        initial_state: InitialState,
        seed_index: int,
    ) -> float:
        del posterior_state, initial_state, seed_index
        return eig_scores[(candidate.x, candidate.y)]

    monkeypatch.setattr(
        "astar_twin.solver.policy.allocator.score_candidate",
        fake_score_candidate,
    )
    monkeypatch.setattr(
        "astar_twin.solver.policy.allocator.approximate_expected_information_gain",
        fake_eig,
    )

    batch = select_adaptive_batch(
        alloc,
        posterior,
        states,
        batch_size=1,
    )

    assert len(batch) == 1
    seed_idx, viewport = batch[0]
    assert seed_idx == 0
    assert (viewport.x, viewport.y) == (0, 5)
    expected_score = (
        RERANK_HEURISTIC_WEIGHT * heuristic_scores[(0, 5)] + RERANK_EIG_WEIGHT * eig_scores[(0, 5)]
    )
    np.testing.assert_allclose(viewport.score, expected_score)


def test_rerank_keeps_high_heuristic_candidate_when_eig_gap_is_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    posterior = create_posterior(n_particles=4, seed=42)
    candidates = [
        ViewportCandidate(x=0, y=0, w=5, h=5, category="frontier"),
        ViewportCandidate(x=5, y=0, w=5, h=5, category="fallback"),
    ]
    alloc.seed_candidates = {0: candidates}

    heuristic_scores = {(0, 0): 0.90, (5, 0): 0.60}
    eig_scores = {(0, 0): 0.20, (5, 0): 0.45}

    monkeypatch.setattr(
        "astar_twin.solver.policy.allocator.score_candidate",
        lambda candidate,
        seed_index,
        posterior_state,
        initial_state,
        queried_viewports,
        queries_used,
        entropy_map=None,
        allow_high_overlap=False: heuristic_scores[(candidate.x, candidate.y)],
    )
    monkeypatch.setattr(
        "astar_twin.solver.policy.allocator.approximate_expected_information_gain",
        lambda candidate, posterior_state, initial_state, seed_index: eig_scores[
            (candidate.x, candidate.y)
        ],
    )

    batch = select_adaptive_batch(alloc, posterior, states, batch_size=1)

    assert len(batch) == 1
    assert (batch[0][1].x, batch[0][1].y) == (0, 0)


def test_select_adaptive_batch_rerank_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    posterior = create_posterior(n_particles=4, seed=42)
    alloc.seed_candidates = {
        0: [
            ViewportCandidate(x=0, y=0, w=5, h=5, category="frontier"),
            ViewportCandidate(x=5, y=0, w=5, h=5, category="fallback"),
            ViewportCandidate(x=0, y=5, w=5, h=5, category="fallback"),
        ]
    }

    def fake_eig(
        candidate: ViewportCandidate,
        posterior_state: PosteriorState,
        initial_state: InitialState,
        seed_index: int,
    ) -> float:
        del posterior_state, initial_state, seed_index
        return float(candidate.x + candidate.y) / 10.0

    monkeypatch.setattr(
        "astar_twin.solver.policy.allocator.approximate_expected_information_gain",
        fake_eig,
    )

    batch_1 = select_adaptive_batch(alloc, posterior, states, batch_size=2)
    batch_2 = select_adaptive_batch(alloc, posterior, states, batch_size=2)

    assert batch_1 == batch_2
    assert all(vp.score > 0.0 for _, vp in batch_1)


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
    assert check_contradiction_triggers(alloc, posterior, states) is False


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
    assert check_contradiction_triggers(alloc, posterior, states) is True


def test_contradiction_trigger_fires_on_argmax_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    posterior = create_posterior(n_particles=8, seed=42)
    alloc.seed_candidates = {
        0: [ViewportCandidate(x=0, y=0, w=5, h=5, category="coastal")],
    }

    monkeypatch.setattr(
        "astar_twin.solver.policy.allocator.check_argmax_disagreement",
        lambda candidate, posterior_state, initial_state: candidate.category == "coastal",
    )

    assert check_contradiction_triggers(alloc, posterior, states) is True


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


def test_reserve_queries_store_final_reranked_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    posterior = create_posterior(n_particles=8, seed=42)
    alloc.seed_candidates = {
        0: [
            ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback"),
            ViewportCandidate(x=5, y=5, w=5, h=5, category="fallback"),
        ]
    }

    posterior.particles[0].log_weight = 100.0
    for particle in posterior.particles[1:]:
        particle.log_weight = -1000.0

    monkeypatch.setattr(
        "astar_twin.solver.policy.allocator.approximate_expected_information_gain",
        lambda candidate, posterior_state, initial_state, seed_index: 0.8
        if candidate.x == 5
        else 0.2,
    )

    reserve = plan_reserve_queries(alloc, posterior, states, n_queries=1)

    assert reserve[0][0] == 0
    assert reserve[0][1].x == 5
    expected_score = (
        RERANK_HEURISTIC_WEIGHT
        * score_candidate(
            ViewportCandidate(x=5, y=5, w=5, h=5, category="fallback"),
            0,
            posterior,
            states[0],
            [],
            queries_used=15,
        )
        + RERANK_EIG_WEIGHT * 0.8
    )
    np.testing.assert_allclose(reserve[0][1].score, expected_score)


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


def test_approximate_expected_information_gain_is_deterministic() -> None:
    state = _make_initial_state()
    posterior = create_posterior(n_particles=4, seed=42)
    candidate = ViewportCandidate(x=0, y=0, w=5, h=5, category="frontier")

    eig_1 = approximate_expected_information_gain(candidate, posterior, state, seed_index=0)
    eig_2 = approximate_expected_information_gain(candidate, posterior, state, seed_index=0)

    assert 0.0 <= eig_1 <= 1.0
    assert eig_1 == eig_2

def test_get_adaptive_weights_early_phase() -> None:
    from astar_twin.solver.policy.allocator import get_adaptive_weights
    w_entropy, w_disagreement, w_stat_gain = get_adaptive_weights(10)
    assert w_entropy == 0.50
    assert w_disagreement == 0.10
    assert w_stat_gain == 0.40

def test_get_adaptive_weights_late_phase() -> None:
    from astar_twin.solver.policy.allocator import get_adaptive_weights
    w_entropy, w_disagreement, w_stat_gain = get_adaptive_weights(30)
    assert w_entropy == 0.20
    assert w_disagreement == 0.70
    assert w_stat_gain == 0.10

def test_get_adaptive_weights_transition() -> None:
    from astar_twin.solver.policy.allocator import get_adaptive_weights
    w_entropy, w_disagreement, w_stat_gain = get_adaptive_weights(20)
    # 20 is halfway between 18 and 22
    assert np.isclose(w_entropy, 0.35)
    assert np.isclose(w_disagreement, 0.40)
    assert np.isclose(w_stat_gain, 0.25)
