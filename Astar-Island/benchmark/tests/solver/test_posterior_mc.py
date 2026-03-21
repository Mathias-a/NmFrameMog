"""Tests for posterior-predictive tensor generation.

Covers:
  - Run allocation proportional to weights with min guarantee
  - Single seed prediction produces valid tensors
  - All-seeds prediction produces 5 valid tensors
  - Runtime fallback triggers correctly
  - Deterministic output from same base_seed
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.data.loaders import load_fixture
from astar_twin.solver.baselines import uniform_baseline
from astar_twin.solver.inference.posterior import PosteriorState, create_posterior
from astar_twin.solver.predict.posterior_mc import (
    FALLBACK_SIMS_PER_SEED,
    allocate_runs_to_particles,
    predict_all_seeds,
    predict_seed,
)

# ---- Fixtures ----


@pytest.fixture
def fixture():
    return load_fixture(
        Path(__file__).parent.parent.parent
        / "data"
        / "rounds"
        / "test-round-001"
        / "round_detail.json"
    )


@pytest.fixture
def initial_states(fixture):
    return fixture.initial_states


@pytest.fixture
def map_dims(fixture):
    return fixture.map_height, fixture.map_width


@pytest.fixture
def posterior():
    return create_posterior(n_particles=8, seed=42)


# ---- Run allocation ----


def test_allocate_runs_sums_to_total():
    """Allocated runs sum to requested total."""
    weights = [0.4, 0.3, 0.2, 0.1]
    total = 64
    alloc = allocate_runs_to_particles(weights, total)
    assert sum(alloc) == total


def test_allocate_runs_respects_minimum():
    """Every particle gets at least MIN_RUNS_PER_PARTICLE."""
    weights = [0.9, 0.05, 0.03, 0.02]
    total = 64
    alloc = allocate_runs_to_particles(weights, total, min_per_particle=4)
    assert all(r >= 4 for r in alloc)


def test_allocate_runs_proportional():
    """Higher-weight particles get more runs."""
    weights = [0.6, 0.3, 0.1]
    total = 30
    alloc = allocate_runs_to_particles(weights, total)
    # First particle should get the most
    assert alloc[0] >= alloc[1] >= alloc[2]


def test_allocate_runs_equal_weights():
    """Equal weights produce roughly equal allocation."""
    weights = [0.25, 0.25, 0.25, 0.25]
    total = 40
    alloc = allocate_runs_to_particles(weights, total)
    assert all(r == 10 for r in alloc)


def test_allocate_runs_empty():
    """Empty weights returns empty list."""
    assert allocate_runs_to_particles([], 64) == []


def test_allocate_runs_tight_budget():
    """When total < n * min, still produces valid allocation."""
    weights = [0.5, 0.5]
    total = 4  # = 2 * min(4) so just barely fits
    alloc = allocate_runs_to_particles(weights, total, min_per_particle=4)
    assert sum(alloc) == total
    assert len(alloc) == 2


# ---- Single seed prediction ----


def test_predict_seed_shape(posterior, initial_states, map_dims):
    """Prediction tensor has correct H×W×6 shape."""
    height, width = map_dims
    tensor, metrics = predict_seed(
        posterior,
        initial_states[0],
        seed_index=0,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=6,
        base_seed=0,
    )
    assert tensor.shape == (height, width, NUM_CLASSES)


def test_predict_seed_normalized(posterior, initial_states, map_dims):
    """Prediction tensor sums to ~1 per cell."""
    height, width = map_dims
    tensor, _ = predict_seed(
        posterior,
        initial_states[0],
        seed_index=0,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=6,
        base_seed=0,
    )
    sums = np.sum(tensor, axis=2)
    np.testing.assert_allclose(sums, 1.0, atol=0.02)


def test_predict_seed_no_zeros(posterior, initial_states, map_dims):
    """No exact zeros in prediction (safe_prediction applied)."""
    height, width = map_dims
    tensor, _ = predict_seed(
        posterior,
        initial_states[0],
        seed_index=0,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=6,
        base_seed=0,
    )
    assert np.all(tensor > 0)


def test_predict_seed_metrics(posterior, initial_states, map_dims):
    """Metrics are populated correctly."""
    height, width = map_dims
    _, metrics = predict_seed(
        posterior,
        initial_states[0],
        seed_index=0,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=6,
        base_seed=0,
    )
    assert metrics.seed_index == 0
    assert metrics.n_particles_used == 3
    assert metrics.total_sims == 6
    assert not metrics.fallback_used
    assert len(metrics.runs_per_particle) == 3


def test_predict_seed_deterministic(posterior, initial_states, map_dims):
    """Same inputs produce identical output."""
    height, width = map_dims
    t1, _ = predict_seed(
        posterior,
        initial_states[0],
        seed_index=0,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=6,
        base_seed=42,
    )
    t2, _ = predict_seed(
        posterior,
        initial_states[0],
        seed_index=0,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=6,
        base_seed=42,
    )
    np.testing.assert_array_equal(t1, t2)


def test_predict_seed_empty_posterior_returns_uniform(initial_states, map_dims):
    """Empty posterior falls back to uniform baseline."""
    height, width = map_dims
    posterior = PosteriorState(particles=[])

    tensor, metrics = predict_seed(
        posterior,
        initial_states[0],
        seed_index=0,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=6,
        base_seed=0,
    )

    expected = uniform_baseline(height, width)
    assert tensor.shape == (height, width, NUM_CLASSES)
    np.testing.assert_allclose(tensor, expected)
    assert metrics.seed_index == 0
    assert metrics.n_particles_used == 0
    assert metrics.total_sims == 0
    assert metrics.fallback_used is True
    assert metrics.runs_per_particle == []


def test_predict_seed_top_k_zero_returns_uniform(posterior, initial_states, map_dims):
    """top_k=0 falls back to uniform baseline."""
    height, width = map_dims

    tensor, metrics = predict_seed(
        posterior,
        initial_states[0],
        seed_index=0,
        map_height=height,
        map_width=width,
        top_k=0,
        sims_per_seed=6,
        base_seed=0,
    )

    expected = uniform_baseline(height, width)
    np.testing.assert_allclose(tensor, expected)
    assert metrics.n_particles_used == 0
    assert metrics.total_sims == 0
    assert metrics.fallback_used is True
    assert metrics.runs_per_particle == []


# ---- All seeds prediction ----


def test_predict_all_seeds_count(posterior, initial_states, map_dims):
    """Produces one tensor per seed."""
    height, width = map_dims
    tensors, metrics = predict_all_seeds(
        posterior,
        initial_states,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=6,
        base_seed=0,
    )
    assert len(tensors) == len(initial_states)
    assert len(metrics) == len(initial_states)


def test_predict_all_seeds_shapes(posterior, initial_states, map_dims):
    """All tensors have correct shape."""
    height, width = map_dims
    tensors, _ = predict_all_seeds(
        posterior,
        initial_states,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=6,
        base_seed=0,
    )
    for t in tensors:
        assert t.shape == (height, width, NUM_CLASSES)
        assert np.all(t > 0)
        sums = np.sum(t, axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=0.02)


# ---- Runtime fallback ----


def test_runtime_fallback_triggers(posterior, initial_states, map_dims):
    """High runtime fraction triggers fallback to fewer sims."""
    height, width = map_dims
    _, metrics = predict_all_seeds(
        posterior,
        initial_states,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=64,
        base_seed=0,
        runtime_fraction=0.85,
    )
    for m in metrics:
        assert m.fallback_used is True
        assert m.total_sims <= FALLBACK_SIMS_PER_SEED


def test_runtime_no_fallback(posterior, initial_states, map_dims):
    """Low runtime fraction uses full sim count."""
    height, width = map_dims
    _, metrics = predict_all_seeds(
        posterior,
        initial_states,
        map_height=height,
        map_width=width,
        top_k=3,
        sims_per_seed=12,
        base_seed=0,
        runtime_fraction=0.5,
    )
    for m in metrics:
        assert m.fallback_used is False
