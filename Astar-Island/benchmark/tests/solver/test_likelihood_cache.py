from __future__ import annotations

from astar_twin.contracts.api_models import InitialState, SimulateResponse, ViewportBounds
from astar_twin.contracts.types import TerrainCode
from astar_twin.solver.inference.likelihood import (
    LikelihoodCache,
    _build_cache_key,
    cached_compute_particle_loglik,
    compute_particle_loglik,
)
from astar_twin.solver.inference.particles import Particle, initialize_particles


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


def test_cache_hit_returns_same_value() -> None:
    initial = _make_initial_state()
    obs = _make_observation()
    particle = initialize_particles(n_particles=1, seed=42)[0]
    cache = LikelihoodCache()

    first = cached_compute_particle_loglik(
        particle,
        obs,
        initial,
        cache=cache,
        n_inner_runs=2,
        base_seed=100,
    )
    second = cached_compute_particle_loglik(
        particle,
        obs,
        initial,
        cache=cache,
        n_inner_runs=2,
        base_seed=100,
    )

    assert first == second


def test_cache_hit_increments_counter() -> None:
    initial = _make_initial_state()
    obs = _make_observation()
    particle = initialize_particles(n_particles=1, seed=42)[0]
    cache = LikelihoodCache()

    cached_compute_particle_loglik(
        particle,
        obs,
        initial,
        cache=cache,
        n_inner_runs=2,
        base_seed=100,
    )
    cached_compute_particle_loglik(
        particle,
        obs,
        initial,
        cache=cache,
        n_inner_runs=2,
        base_seed=100,
    )

    assert cache.misses == 1
    assert cache.hits == 1


def test_different_params_produce_different_keys() -> None:
    initial = _make_initial_state()
    particle_a = initialize_particles(n_particles=2, seed=42)[0]
    particle_b = initialize_particles(n_particles=2, seed=42)[1]

    key_a = _build_cache_key(
        particle_a,
        initial,
        viewport_x=0,
        viewport_y=0,
        viewport_w=5,
        viewport_h=5,
        n_inner_runs=2,
        base_seed=100,
    )
    key_b = _build_cache_key(
        particle_b,
        initial,
        viewport_x=0,
        viewport_y=0,
        viewport_w=5,
        viewport_h=5,
        n_inner_runs=2,
        base_seed=100,
    )

    assert key_a != key_b


def test_cache_clear_empties() -> None:
    initial = _make_initial_state()
    obs = _make_observation()
    particle = initialize_particles(n_particles=1, seed=42)[0]
    cache = LikelihoodCache()

    cached_compute_particle_loglik(
        particle,
        obs,
        initial,
        cache=cache,
        n_inner_runs=2,
        base_seed=100,
    )
    cache.clear()

    assert (
        cache.get(
            _build_cache_key(
                particle,
                initial,
                viewport_x=0,
                viewport_y=0,
                viewport_w=5,
                viewport_h=5,
                n_inner_runs=2,
                base_seed=100,
            )
        )
        is None
    )


def test_cache_equivalence() -> None:
    initial = _make_initial_state()
    obs = _make_observation()
    particle = initialize_particles(n_particles=1, seed=42)[0]
    cache = LikelihoodCache()

    uncached = compute_particle_loglik(
        particle,
        obs,
        initial,
        n_inner_runs=2,
        base_seed=100,
    )
    cached = cached_compute_particle_loglik(
        particle,
        obs,
        initial,
        cache=cache,
        n_inner_runs=2,
        base_seed=100,
    )

    assert uncached == cached


def test_new_cache_instance_is_empty() -> None:
    initial = _make_initial_state()
    obs = _make_observation()
    particle = initialize_particles(n_particles=1, seed=42)[0]
    cache_one = LikelihoodCache()
    cache_two = LikelihoodCache()

    cached_compute_particle_loglik(
        particle,
        obs,
        initial,
        cache=cache_one,
        n_inner_runs=2,
        base_seed=100,
    )

    key = _build_cache_key(
        particle,
        initial,
        viewport_x=0,
        viewport_y=0,
        viewport_w=5,
        viewport_h=5,
        n_inner_runs=2,
        base_seed=100,
    )
    assert cache_two.get(key) is None
    assert cache_two.hits == 0
    assert cache_two.misses == 0


def test_update_posterior_reuses_duplicate_particle_likelihoods() -> None:
    initial = _make_initial_state()
    obs = _make_observation()
    particle = initialize_particles(n_particles=1, seed=42)[0]
    duplicate = Particle(params=dict(particle.params), log_weight=0.0)
    different = initialize_particles(n_particles=2, seed=99)[1]

    cache = LikelihoodCache()
    first = cached_compute_particle_loglik(
        particle,
        obs,
        initial,
        cache=cache,
        n_inner_runs=2,
        base_seed=100,
    )
    second = cached_compute_particle_loglik(
        duplicate,
        obs,
        initial,
        cache=cache,
        n_inner_runs=2,
        base_seed=100,
    )
    third = cached_compute_particle_loglik(
        different,
        obs,
        initial,
        cache=cache,
        n_inner_runs=2,
        base_seed=100,
    )

    assert first == second
    assert cache.hits == 1
    assert cache.misses == 2
    assert third == cache.get(
        _build_cache_key(
            different,
            initial,
            viewport_x=0,
            viewport_y=0,
            viewport_w=5,
            viewport_h=5,
            n_inner_runs=2,
            base_seed=100,
        )
    )
