"""Tests for end-to-end solver pipeline.

Covers:
  - Full pipeline runs on benchmark adapter
  - Budget accounting: exactly ≤50 queries
  - Returns 5 valid tensors
  - Transcript is complete and consistent
  - Deterministic output from same seed
  - Graceful degradation under near-exhausted budget
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.types import MAX_QUERIES, NUM_CLASSES
from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import RoundFixture
from astar_twin.solver.adapters.benchmark import BenchmarkAdapter
from astar_twin.solver.pipeline import SolveResult, solve


# ---- Fixtures ----


FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def fixture() -> RoundFixture:
    return load_fixture(FIXTURE_PATH)


@pytest.fixture
def adapter(fixture: RoundFixture) -> BenchmarkAdapter:
    return BenchmarkAdapter(fixture, n_mc_runs=5, sim_seed_offset=0)


# ---- Full pipeline ----


def test_pipeline_returns_5_tensors(adapter: BenchmarkAdapter):
    """Pipeline returns one tensor per seed."""
    result = solve(
        adapter, "test-round-001",
        n_particles=8, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    assert len(result.tensors) == 5


def test_pipeline_tensor_shapes(adapter: BenchmarkAdapter, fixture: RoundFixture):
    """All tensors have correct H×W×6 shape."""
    result = solve(
        adapter, "test-round-001",
        n_particles=8, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    for t in result.tensors:
        assert t.shape == (fixture.map_height, fixture.map_width, NUM_CLASSES)


def test_pipeline_tensors_are_valid(adapter: BenchmarkAdapter):
    """All tensors have positive probs that sum to ~1."""
    result = solve(
        adapter, "test-round-001",
        n_particles=8, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    for t in result.tensors:
        assert np.all(t > 0), "No zeros allowed"
        assert np.all(np.isfinite(t)), "All values must be finite"
        sums = np.sum(t, axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=0.02)


def test_pipeline_budget_not_exceeded(adapter: BenchmarkAdapter):
    """Pipeline uses at most 50 queries."""
    result = solve(
        adapter, "test-round-001",
        n_particles=8, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    assert result.total_queries_used <= MAX_QUERIES


def test_pipeline_transcript_consistent(adapter: BenchmarkAdapter):
    """Transcript length matches total queries used."""
    result = solve(
        adapter, "test-round-001",
        n_particles=8, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    assert len(result.transcript) == result.total_queries_used


def test_pipeline_transcript_phases(adapter: BenchmarkAdapter):
    """Transcript records phases in order: bootstrap, adaptive, reserve."""
    result = solve(
        adapter, "test-round-001",
        n_particles=8, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    phases = [q.phase for q in result.transcript]
    # All bootstrap queries should come first
    bootstrap_count = sum(1 for p in phases if p == "bootstrap")
    assert bootstrap_count >= 1  # At least some bootstrap queries
    # Bootstrap should be before adaptive
    if bootstrap_count > 0 and "adaptive" in phases:
        last_bootstrap = max(i for i, p in enumerate(phases) if p == "bootstrap")
        first_adaptive = min(i for i, p in enumerate(phases) if p == "adaptive")
        assert last_bootstrap < first_adaptive


def test_pipeline_all_seeds_queried(adapter: BenchmarkAdapter):
    """All 5 seeds should receive at least 1 query."""
    result = solve(
        adapter, "test-round-001",
        n_particles=8, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    queried_seeds = {q.seed_index for q in result.transcript}
    assert len(queried_seeds) == 5, f"Only queried seeds: {queried_seeds}"


def test_pipeline_ess_positive(adapter: BenchmarkAdapter):
    """Final ESS should be positive."""
    result = solve(
        adapter, "test-round-001",
        n_particles=8, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    assert result.final_ess > 0


def test_pipeline_deterministic(fixture: RoundFixture):
    """Same inputs produce identical results."""
    adapter1 = BenchmarkAdapter(fixture, n_mc_runs=5, sim_seed_offset=0)
    r1 = solve(
        adapter1, "test-round-001",
        n_particles=8, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    adapter2 = BenchmarkAdapter(fixture, n_mc_runs=5, sim_seed_offset=0)
    r2 = solve(
        adapter2, "test-round-001",
        n_particles=8, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    for t1, t2 in zip(r1.tensors, r2.tensors):
        np.testing.assert_array_equal(t1, t2)
    assert r1.total_queries_used == r2.total_queries_used


# ---- Degraded budget scenario ----


def test_pipeline_with_pre_exhausted_budget(fixture: RoundFixture):
    """Pipeline degrades gracefully if budget is nearly exhausted mid-run."""
    # Create adapter with offset that simulates queries already used
    adapter = BenchmarkAdapter(fixture, n_mc_runs=5, sim_seed_offset=0)
    # Manually burn through most of the budget
    for i in range(45):
        try:
            adapter.simulate(
                "test-round-001", 0,
                viewport_x=0, viewport_y=0, viewport_w=5, viewport_h=5,
            )
        except RuntimeError:
            break

    # Pipeline should still produce valid tensors with remaining budget
    result = solve(
        adapter, "test-round-001",
        n_particles=4, n_inner_runs=2, sims_per_seed=4, base_seed=42,
    )
    # Should have tensors even with limited queries
    assert len(result.tensors) == 5
    for t in result.tensors:
        assert np.all(t > 0)
        assert np.all(np.isfinite(t))
