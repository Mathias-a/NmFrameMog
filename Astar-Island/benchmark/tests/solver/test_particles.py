"""Tests for particle schema, priors, and parameter subset."""

from __future__ import annotations

from astar_twin.params.simulation_params import SimulationParams
from astar_twin.solver.inference.particles import (
    INFERRED_PARAMS,
    Particle,
    initialize_particles,
    validate_particle,
)


def test_initialize_24_particles() -> None:
    particles = initialize_particles(n_particles=24, seed=42)
    assert len(particles) == 24


def test_all_inferred_fields_present() -> None:
    particles = initialize_particles(n_particles=24, seed=42)
    for p in particles:
        for name in INFERRED_PARAMS:
            assert name in p.params, f"Missing {name} in particle"


def test_deterministic_initialization() -> None:
    p1 = initialize_particles(n_particles=24, seed=42)
    p2 = initialize_particles(n_particles=24, seed=42)
    for a, b in zip(p1, p2):
        assert a.params == b.params
        assert a.log_weight == b.log_weight


def test_different_seeds_differ() -> None:
    p1 = initialize_particles(n_particles=24, seed=42)
    p2 = initialize_particles(n_particles=24, seed=99)
    # At least some particles should differ (not all — first is always default)
    differs = sum(1 for a, b in zip(p1[1:], p2[1:]) if a.params != b.params)
    assert differs > 0


def test_first_particle_is_defaults() -> None:
    particles = initialize_particles(n_particles=24, seed=42)
    defaults = SimulationParams()
    first = particles[0]
    for name in INFERRED_PARAMS:
        assert first.params[name] == getattr(defaults, name), f"First particle {name} != default"


def test_non_inferred_params_frozen_at_defaults() -> None:
    particles = initialize_particles(n_particles=24, seed=42)
    defaults = SimulationParams()
    for p in particles:
        sim_params = p.to_simulation_params()
        for f in defaults.__dataclass_fields__:
            if f not in INFERRED_PARAMS:
                assert getattr(sim_params, f) == getattr(defaults, f), (
                    f"Non-inferred param {f} differs from default"
                )


def test_to_simulation_params_returns_valid_object() -> None:
    particles = initialize_particles(n_particles=24, seed=42)
    for p in particles:
        sp = p.to_simulation_params()
        assert isinstance(sp, SimulationParams)


def test_validate_particle_passes_for_valid() -> None:
    particles = initialize_particles(n_particles=24, seed=42)
    for p in particles:
        errors = validate_particle(p)
        assert errors == [], f"Unexpected validation errors: {errors}"


def test_validate_particle_catches_missing_param() -> None:
    p = Particle(params={"adjacency_mode": "moore8"})
    errors = validate_particle(p)
    assert len(errors) > 0
    assert any("Missing" in e for e in errors)


def test_validate_particle_catches_out_of_range() -> None:
    particles = initialize_particles(n_particles=1, seed=42)
    p = particles[0]
    p.params["expansion_rate"] = 999.0  # way out of range
    errors = validate_particle(p)
    assert any("expansion_rate" in e for e in errors)


def test_validate_particle_catches_invalid_enum() -> None:
    particles = initialize_particles(n_particles=1, seed=42)
    p = particles[0]
    p.params["adjacency_mode"] = "invalid_mode"
    errors = validate_particle(p)
    assert any("adjacency_mode" in e for e in errors)


def test_uniform_initial_weights() -> None:
    particles = initialize_particles(n_particles=24, seed=42)
    for p in particles:
        assert p.log_weight == 0.0


def test_seventeen_inferred_params() -> None:
    assert len(INFERRED_PARAMS) == 17
