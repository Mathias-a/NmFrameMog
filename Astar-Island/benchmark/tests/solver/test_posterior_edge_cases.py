import numpy as np
import pytest
from astar_twin.solver.inference.posterior import PosteriorState, Particle, temper_if_collapsed, prune_and_resample_bootstrap

def test_posterior_all_neginf():
    particles = [Particle(params={}, log_weight=-np.inf) for _ in range(5)]
    state = PosteriorState(particles=particles)
    
    assert state.ess == pytest.approx(5.0)
    assert state.top_particle_mass == pytest.approx(0.2)
    assert state.normalized_weights() == pytest.approx([0.2, 0.2, 0.2, 0.2, 0.2])
    
    state = temper_if_collapsed(state)
    assert all(np.isneginf(p.log_weight) for p in state.particles)

def test_posterior_all_nan():
    particles = [Particle(params={}, log_weight=np.nan) for _ in range(5)]
    state = PosteriorState(particles=particles)
    
    assert state.ess == pytest.approx(5.0)
    assert state.top_particle_mass == pytest.approx(0.2)
    assert state.normalized_weights() == pytest.approx([0.2, 0.2, 0.2, 0.2, 0.2])
    
    state = temper_if_collapsed(state)
    assert all(np.isnan(p.log_weight) for p in state.particles)

def test_posterior_mixed_nan():
    particles = [Particle(params={}, log_weight=0.0)] + [Particle(params={}, log_weight=np.nan) for _ in range(4)]
    state = PosteriorState(particles=particles)
    
    assert state.ess == pytest.approx(5.0)
    assert state.top_particle_mass == pytest.approx(0.2)
    assert state.normalized_weights() == pytest.approx([0.2, 0.2, 0.2, 0.2, 0.2])
    
    state = temper_if_collapsed(state)
    assert state.particles[0].log_weight == 0.0
    assert all(np.isnan(p.log_weight) for p in state.particles[1:])

def test_prune_and_resample_neginf():
    particles = [Particle(params={}, log_weight=-np.inf) for _ in range(10)]
    state = PosteriorState(particles=particles)
    
    state = prune_and_resample_bootstrap(state, top_k=5, target_n=5)
    assert len(state.particles) == 5
    assert all(p.log_weight == 0.0 for p in state.particles)

