"""Posterior updates with resampling and tempering.

Maintains a particle set with:
  - Log-space weight updates from likelihood
  - ESS-triggered resampling
  - Post-bootstrap pruning (top 8 → resample to 12)
  - Tempering when top-particle mass > 0.85
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import default_rng

from astar_twin.contracts.api_models import InitialState, SimulateResponse
from astar_twin.solver.inference.likelihood import compute_particle_loglik
from astar_twin.solver.inference.particles import Particle, initialize_particles


@dataclass
class PosteriorState:
    """Current state of the particle posterior."""

    particles: list[Particle]
    ess_history: list[float] = field(default_factory=list)
    n_updates: int = 0
    phase: str = "bootstrap"  # "bootstrap" | "adaptive" | "reserve"

    @property
    def ess(self) -> float:
        """Effective sample size."""
        if not self.particles:
            return 0.0

        log_weights = np.array([p.log_weight for p in self.particles])
        max_w = np.max(log_weights)
        weights = np.exp(log_weights - max_w)
        weights /= np.sum(weights)
        return float(1.0 / np.sum(weights**2))

    @property
    def top_particle_mass(self) -> float:
        """Normalized weight of the highest-weight particle."""
        if not self.particles:
            return 0.0

        log_weights = np.array([p.log_weight for p in self.particles])
        max_w = np.max(log_weights)
        weights = np.exp(log_weights - max_w)
        weights /= np.sum(weights)
        return float(np.max(weights))

    def normalized_weights(self) -> list[float]:
        """Return normalized weights for all particles."""
        if not self.particles:
            return []

        log_weights = np.array([p.log_weight for p in self.particles])
        max_w = np.max(log_weights)
        weights = np.exp(log_weights - max_w)
        weights /= np.sum(weights)
        return weights.tolist()

    def top_k_indices(self, k: int) -> list[int]:
        """Return indices of top-k particles by weight."""
        if not self.particles:
            return []

        log_weights = [p.log_weight for p in self.particles]
        sorted_idx = sorted(range(len(log_weights)), key=lambda i: log_weights[i], reverse=True)
        return sorted_idx[:k]


def create_posterior(n_particles: int = 24, seed: int = 0) -> PosteriorState:
    """Create a fresh posterior with initialized particles."""
    particles = initialize_particles(n_particles=n_particles, seed=seed)
    return PosteriorState(particles=particles)


def update_posterior(
    state: PosteriorState,
    observed_response: SimulateResponse,
    initial_state: InitialState,
    n_inner_runs: int = 6,
    base_seed: int = 0,
) -> PosteriorState:
    """Update particle weights given a new observation."""
    for particle in state.particles:
        ll = compute_particle_loglik(
            particle,
            observed_response,
            initial_state,
            n_inner_runs=n_inner_runs,
            base_seed=base_seed,
        )
        particle.log_weight += ll

    state.n_updates += 1
    state.ess_history.append(state.ess)
    return state


def prune_and_resample_bootstrap(
    state: PosteriorState,
    top_k: int = 8,
    target_n: int = 12,
    seed: int = 0,
) -> PosteriorState:
    """After bootstrap phase: keep top-k particles, resample to target_n.

    Used after the 10 bootstrap queries are complete.
    """
    rng = default_rng(seed)

    # Sort by weight, keep top-k
    top_indices = state.top_k_indices(top_k)
    top_particles = [state.particles[i] for i in top_indices]

    # Normalize weights among survivors
    log_weights = np.array([p.log_weight for p in top_particles])
    max_w = np.max(log_weights)
    weights = np.exp(log_weights - max_w)
    weights /= np.sum(weights)

    # Systematic resample to target_n
    new_particles: list[Particle] = []
    cumsum = np.cumsum(weights)
    u = rng.random() / target_n

    idx = 0
    for j in range(target_n):
        threshold = u + j / target_n
        while idx < len(cumsum) - 1 and cumsum[idx] < threshold:
            idx += 1
        # Clone the selected particle with reset weight
        source = top_particles[idx]
        new_particles.append(
            Particle(
                params=dict(source.params),
                log_weight=0.0,
            )
        )

    state.particles = new_particles
    state.phase = "adaptive"
    return state


def resample_if_needed(
    state: PosteriorState,
    ess_threshold: float = 6.0,
    seed: int = 0,
) -> PosteriorState:
    """Resample during adaptive phase when ESS drops below threshold."""
    if state.ess >= ess_threshold:
        return state

    rng = default_rng(seed)
    n = len(state.particles)
    weights = np.array(state.normalized_weights())

    # Systematic resample
    new_particles: list[Particle] = []
    cumsum = np.cumsum(weights)
    u = rng.random() / n

    idx = 0
    for j in range(n):
        threshold = u + j / n
        while idx < len(cumsum) - 1 and cumsum[idx] < threshold:
            idx += 1
        source = state.particles[idx]
        new_particles.append(
            Particle(
                params=dict(source.params),
                log_weight=0.0,
            )
        )

    state.particles = new_particles
    return state


def temper_if_collapsed(
    state: PosteriorState,
    mass_threshold: float = 0.85,
    temperature: float = 0.5,
) -> PosteriorState:
    """Down-temper weights if top particle mass exceeds threshold."""
    if state.top_particle_mass <= mass_threshold:
        return state

    for p in state.particles:
        p.log_weight *= temperature

    return state
