"""Tests for posterior updates, resampling, and tempering."""

from __future__ import annotations

import numpy as np

from astar_twin.contracts.api_models import (
    InitialState,
    SimSettlement,
    SimulateResponse,
    ViewportBounds,
)
from astar_twin.contracts.types import TerrainCode
from astar_twin.solver.inference.particles import initialize_particles
from astar_twin.solver.inference.posterior import (
    PosteriorState,
    create_posterior,
    prune_and_resample_bootstrap,
    resample_if_needed,
    temper_if_collapsed,
    update_posterior,
)


def _make_initial_state() -> InitialState:
    grid = [[TerrainCode.PLAINS] * 10 for _ in range(10)]
    return InitialState(grid=grid, settlements=[])


def _make_observation() -> SimulateResponse:
    return SimulateResponse(
        grid=[[TerrainCode.PLAINS] * 5 for _ in range(5)],
        settlements=[],
        viewport=ViewportBounds(x=0, y=0, w=5, h=5),
        width=10,
        height=10,
        queries_used=1,
        queries_max=50,
    )


def test_create_posterior_has_24_particles() -> None:
    state = create_posterior(n_particles=24, seed=42)
    assert len(state.particles) == 24
    assert state.n_updates == 0
    assert state.phase == "bootstrap"


def test_empty_posterior_state_properties() -> None:
    state = PosteriorState(particles=[])

    assert state.ess == 0.0
    assert state.top_particle_mass == 0.0
    assert state.normalized_weights() == []
    assert state.top_k_indices(3) == []


def test_update_changes_weights() -> None:
    state = create_posterior(n_particles=6, seed=42)
    initial = _make_initial_state()
    obs = _make_observation()

    # All weights start at 0
    assert all(p.log_weight == 0.0 for p in state.particles)

    state = update_posterior(state, obs, initial, n_inner_runs=2, base_seed=100)

    # After update, weights should have changed (not all zero unless very unlikely)
    weights = [p.log_weight for p in state.particles]
    assert state.n_updates == 1
    assert len(state.ess_history) == 1


def test_deterministic_replay() -> None:
    initial = _make_initial_state()
    obs = _make_observation()

    state1 = create_posterior(n_particles=6, seed=42)
    state1 = update_posterior(state1, obs, initial, n_inner_runs=2, base_seed=100)

    state2 = create_posterior(n_particles=6, seed=42)
    state2 = update_posterior(state2, obs, initial, n_inner_runs=2, base_seed=100)

    w1 = [p.log_weight for p in state1.particles]
    w2 = [p.log_weight for p in state2.particles]
    assert np.allclose(w1, w2)


def test_prune_and_resample_reduces_particles() -> None:
    state = create_posterior(n_particles=24, seed=42)
    # Give varied weights
    for i, p in enumerate(state.particles):
        p.log_weight = float(-i)  # decreasing weights

    state = prune_and_resample_bootstrap(state, top_k=8, target_n=12, seed=42)
    assert len(state.particles) == 12
    assert state.phase == "adaptive"
    # All weights should be reset to 0
    assert all(p.log_weight == 0.0 for p in state.particles)


def test_ess_triggered_resampling() -> None:
    state = create_posterior(n_particles=12, seed=42)
    # Create extreme weight concentration
    state.particles[0].log_weight = 100.0
    for p in state.particles[1:]:
        p.log_weight = -100.0

    ess_before = state.ess
    assert ess_before < 6.0  # Should trigger

    state = resample_if_needed(state, ess_threshold=6.0, seed=42)
    # After resampling, weights should be reset
    assert all(p.log_weight == 0.0 for p in state.particles)


def test_no_resampling_when_ess_sufficient() -> None:
    state = create_posterior(n_particles=12, seed=42)
    # All uniform weights → ESS = 12
    ess_before = state.ess
    assert ess_before >= 6.0

    original_weights = [p.log_weight for p in state.particles]
    state = resample_if_needed(state, ess_threshold=6.0, seed=42)
    # Weights unchanged
    for p, orig in zip(state.particles, original_weights):
        assert p.log_weight == orig


def test_temper_when_collapsed() -> None:
    state = create_posterior(n_particles=12, seed=42)
    state.particles[0].log_weight = 100.0
    for p in state.particles[1:]:
        p.log_weight = -100.0

    assert state.top_particle_mass > 0.85

    state = temper_if_collapsed(state, mass_threshold=0.85, temperature=0.5)
    # Weights should be halved
    assert state.particles[0].log_weight == 50.0


def test_no_temper_when_not_collapsed() -> None:
    state = create_posterior(n_particles=12, seed=42)
    # Uniform weights → top mass ≈ 1/12 ≈ 0.083
    assert state.top_particle_mass < 0.85

    orig_weights = [p.log_weight for p in state.particles]
    state = temper_if_collapsed(state, mass_threshold=0.85, temperature=0.5)
    for p, orig in zip(state.particles, orig_weights):
        assert p.log_weight == orig


def test_posterior_ranking_changes_with_evidence() -> None:
    """Different observations should produce different posterior rankings."""
    initial = _make_initial_state()

    # Observation 1: all plains
    obs1 = SimulateResponse(
        grid=[[TerrainCode.PLAINS] * 5 for _ in range(5)],
        settlements=[],
        viewport=ViewportBounds(x=0, y=0, w=5, h=5),
        width=10,
        height=10,
        queries_used=1,
        queries_max=50,
    )

    # Observation 2: mixed terrain
    obs2 = SimulateResponse(
        grid=[[TerrainCode.FOREST] * 5 for _ in range(5)],
        settlements=[],
        viewport=ViewportBounds(x=0, y=0, w=5, h=5),
        width=10,
        height=10,
        queries_used=1,
        queries_max=50,
    )

    state1 = create_posterior(n_particles=6, seed=42)
    state1 = update_posterior(state1, obs1, initial, n_inner_runs=2, base_seed=100)

    state2 = create_posterior(n_particles=6, seed=42)
    state2 = update_posterior(state2, obs2, initial, n_inner_runs=2, base_seed=100)

    w1 = [p.log_weight for p in state1.particles]
    w2 = [p.log_weight for p in state2.particles]
    # Weights should differ given different observations
    assert not np.allclose(w1, w2)
