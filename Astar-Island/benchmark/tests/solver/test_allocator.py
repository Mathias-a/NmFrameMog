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

from astar_twin.contracts.api_models import InitialState, InitialSettlement
from astar_twin.solver.inference.particles import initialize_particles
from astar_twin.solver.inference.posterior import PosteriorState, create_posterior
from astar_twin.solver.policy.allocator import (
    ADAPTIVE_BATCH_SIZE,
    ADAPTIVE_QUERIES,
    BOOTSTRAP_PER_SEED,
    BOOTSTRAP_QUERIES,
    MAX_OVERLAP_FRACTION,
    MAX_QUERIES,
    MIN_QUERIES_PER_SEED,
    RESERVE_QUERIES,
    AllocationState,
    QueryRecord,
    ViewportCandidate,
    check_argmax_disagreement,
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
from astar_twin.solver.policy.hotspots import generate_hotspots


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
                has_port=(i == 0),
                alive=True,
            )
        )

    return InitialState(grid=grid, settlements=settlements)


def _make_5_initial_states() -> list[InitialState]:
    """Create 5 initial states (one per seed)."""
    return [_make_initial_state() for _ in range(5)]


# ---- Budget constants ----


def test_budget_constants():
    """Bootstrap + adaptive + reserve = 50."""
    assert BOOTSTRAP_QUERIES + ADAPTIVE_QUERIES + RESERVE_QUERIES == MAX_QUERIES
    assert BOOTSTRAP_QUERIES == 10
    assert ADAPTIVE_QUERIES == 30
    assert RESERVE_QUERIES == 10
    assert BOOTSTRAP_PER_SEED == 2


# ---- Initialization ----


def test_initialize_allocation_creates_candidates_per_seed():
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


def test_plan_bootstrap_queries_produces_2_per_seed():
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


def test_bootstrap_two_queries_differ_per_seed():
    """The two bootstrap queries per seed should be different viewports."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    planned = plan_bootstrap_queries(alloc)

    per_seed: dict[int, list[ViewportCandidate]] = {}
    for seed_idx, vp in planned:
        per_seed.setdefault(seed_idx, []).append(vp)

    for seed_idx, vps in per_seed.items():
        if len(vps) == 2:
            # Either different position or different category
            a, b = vps
            assert (a.x, a.y, a.w, a.h) != (b.x, b.y, b.w, b.h) or a.category != b.category


# ---- Scoring ----


def test_score_candidate_respects_overlap_rejection():
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
    )
    assert score == -1.0


def test_score_candidate_allows_high_overlap_when_flagged():
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
        allow_high_overlap=True,
    )
    assert score != -1.0  # Scored normally despite overlap


def test_score_candidate_with_entropy_map():
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


def test_compute_entropy_map_shape():
    """Entropy map has same H×W shape as input."""
    tensor = np.ones((10, 10, 6), dtype=np.float64) / 6.0
    entropy = compute_entropy_map(tensor)
    assert entropy.shape == (10, 10)
    # Uniform distribution → max entropy
    expected = np.log(6)
    np.testing.assert_allclose(entropy, expected, atol=1e-5)


def test_compute_entropy_map_deterministic():
    """Certain prediction → zero entropy."""
    tensor = np.zeros((10, 10, 6), dtype=np.float64)
    tensor[:, :, 0] = 1.0  # all class-0
    entropy = compute_entropy_map(tensor)
    assert entropy.shape == (10, 10)
    np.testing.assert_allclose(entropy, 0.0, atol=1e-5)


# ---- Adaptive selection ----


def test_select_adaptive_batch_respects_budget():
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


def test_select_adaptive_batch_accepts_per_seed_predictions():
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


# ---- Phase transitions ----


def test_phase_transitions():
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


def test_contradiction_ess_trigger():
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


def test_no_contradiction_with_healthy_posterior():
    """No trigger with healthy ESS and enough queries per seed."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)
    posterior = create_posterior(n_particles=8, seed=42)
    # All equal weights → ESS = n_particles
    assert posterior.ess >= 6.0
    assert check_contradiction_triggers(alloc, posterior) is False


def test_under_queried_seed_trigger():
    """After adaptive done, seed with < MIN_QUERIES_PER_SEED fires trigger."""
    states = _make_5_initial_states()
    alloc = initialize_allocation(states, map_height=10, map_width=10)

    # Record 10 bootstrap
    for i in range(BOOTSTRAP_QUERIES):
        vp = ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback")
        record_query(alloc, i % 5, vp, "bootstrap")
    transition_phase(alloc)

    # Record 30 adaptive, all to seed 0
    for i in range(ADAPTIVE_QUERIES):
        vp = ViewportCandidate(x=0, y=0, w=5, h=5, category="fallback")
        record_query(alloc, 0, vp, "adaptive")
    transition_phase(alloc)

    # Seed 1-4 each have only 2 bootstrap queries < 8
    posterior = create_posterior(n_particles=8, seed=42)
    assert check_contradiction_triggers(alloc, posterior) is True


# ---- Reserve planning ----


def test_reserve_queries_with_contradiction():
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


def test_reserve_queries_accepts_per_seed_predictions():
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


# ---- Record + query tracking ----


def test_record_query_increments_count():
    """Recording queries increments budget counters."""
    alloc = AllocationState()
    vp = ViewportCandidate(x=0, y=0, w=10, h=10, category="coastal")
    record_query(alloc, 0, vp, "bootstrap")
    assert alloc.queries_used == 1
    assert alloc.queries_remaining == MAX_QUERIES - 1
    assert len(alloc.queries_for_seed(0)) == 1
    assert len(alloc.queries_for_seed(1)) == 0


# ---- Posterior disagreement ----


def test_posterior_disagreement_equal_weights():
    """Equal-weight particles yield high disagreement proxy."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=10, seed=42)
    candidate = ViewportCandidate(x=0, y=0, w=5, h=5, category="frontier")

    disagreement = compute_posterior_disagreement(candidate, posterior, state)
    # 10 equal particles → top mass = 0.1, disagreement = 0.9
    assert 0.85 <= disagreement <= 0.95


def test_posterior_disagreement_collapsed():
    """Collapsed posterior yields low disagreement proxy."""
    state = _make_initial_state()
    posterior = create_posterior(n_particles=10, seed=42)
    posterior.particles[0].log_weight = 100.0
    for p in posterior.particles[1:]:
        p.log_weight = -1000.0

    candidate = ViewportCandidate(x=0, y=0, w=5, h=5, category="frontier")
    disagreement = compute_posterior_disagreement(candidate, posterior, state)
    # Top particle has ~100% mass, disagreement ≈ 0
    assert disagreement < 0.05
