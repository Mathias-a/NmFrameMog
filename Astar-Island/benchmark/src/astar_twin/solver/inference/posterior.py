"""Posterior updates with resampling, tempering, and MCMC rejuvenation.

Maintains a particle set with:
  - Log-space weight updates from likelihood
  - ESS-triggered resampling with post-resample mutation (MCMC rejuvenation)
  - Post-bootstrap pruning (top 8 → resample to 12)
  - Progressive tempering schedule (adaptive to collapse severity)
  - ESS-trend collapse detection via consecutive decline monitoring

Calibration improvements (worktree-6 avenue):
  - Particles are perturbed after resampling to break degeneracy.
  - ESS threshold scales with observation count (stricter early, relaxed late).
  - Tempering uses a progressive schedule instead of a single fixed factor.
  - Consecutive ESS decline triggers early intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import default_rng

from astar_twin.contracts.api_models import InitialState, SimulateResponse
from astar_twin.solver.inference.likelihood import compute_particle_loglik
from astar_twin.solver.inference.particles import Particle, initialize_particles, perturb_particle


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

    @property
    def ess_declining(self) -> bool:
        """True if ESS has been declining for 3+ consecutive updates.

        This is an early-warning signal for posterior collapse: the particle
        set is concentrating on fewer and fewer hypotheses, which may indicate
        over-commitment to a single parameter region.
        """
        if len(self.ess_history) < 3:
            return False
        recent = self.ess_history[-3:]
        return recent[0] > recent[1] > recent[2]


def create_posterior(
    n_particles: int = 24, seed: int = 0, use_experts: bool = False
) -> PosteriorState:
    """Create a fresh posterior with initialized particles."""
    if use_experts:
        from astar_twin.solver.inference.particles import initialize_expert_particles

        particles = initialize_expert_particles()
    else:
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


def _mutate_after_resample(
    particles: list[Particle],
    seed: int,
    spread: float = 0.10,
    skip_first: bool = True,
) -> list[Particle]:
    """Apply MCMC rejuvenation: perturb cloned particles after resampling.

    After systematic resampling, many particles are exact copies.  Without
    mutation, the posterior collapses to a small number of unique hypotheses
    and subsequent likelihood updates cannot discriminate between identical
    particles.

    This function applies small Gaussian perturbations to each particle's
    continuous parameters and occasionally re-rolls enum parameters, so that
    even cloned particles explore slightly different regions of parameter space.

    Args:
        particles: List of particles (typically freshly resampled).
        seed: RNG seed for deterministic mutation.
        spread: Perturbation scale as fraction of parameter range (default 0.10).
        skip_first: If True, leave the first particle unperturbed as an anchor.

    Returns:
        List of mutated particles (same length, new objects).
    """
    rng = default_rng(seed)
    result: list[Particle] = []

    for i, p in enumerate(particles):
        if skip_first and i == 0:
            # Keep one particle as-is for stability
            result.append(Particle(params=dict(p.params), log_weight=0.0))
        else:
            result.append(perturb_particle(p, rng, spread=spread))

    return result


def prune_and_resample_bootstrap(
    state: PosteriorState,
    top_k: int = 8,
    target_n: int = 12,
    seed: int = 0,
) -> PosteriorState:
    """After bootstrap phase: keep top-k particles, resample to target_n.

    Used after the 10 bootstrap queries are complete.
    Now includes MCMC rejuvenation to break degeneracy in cloned particles.
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

    # MCMC rejuvenation: perturb cloned particles to break degeneracy
    new_particles = _mutate_after_resample(new_particles, seed=seed + 7777, spread=0.12)

    state.particles = new_particles
    state.phase = "adaptive"
    return state


def _adaptive_ess_threshold(state: PosteriorState) -> float:
    """Compute an ESS threshold that adapts to how many updates have occurred.

    Early in the inference process (few observations), we want a stricter
    threshold to prevent premature collapse.  Later, when the posterior has
    seen many observations, we can afford a lower threshold because the
    remaining particles should already be in the right region.

    The threshold scales from ``n_particles / 2`` (strict) down to a floor
    of ``max(3.0, n_particles / 4)``.
    """
    n = len(state.particles)
    if n == 0:
        return 0.0

    upper = n / 2.0  # Strict threshold: half the particle count
    lower = max(3.0, n / 4.0)  # Floor: quarter of particles, min 3

    # Decay from upper to lower over ~20 updates
    decay = min(state.n_updates / 20.0, 1.0)
    return upper - decay * (upper - lower)


def resample_if_needed(
    state: PosteriorState,
    ess_threshold: float = 6.0,
    seed: int = 0,
) -> PosteriorState:
    """Resample during adaptive phase when ESS drops below threshold.

    Now uses adaptive ESS threshold (scales with observation count) and
    applies MCMC rejuvenation after resampling to prevent degeneracy.

    The ``ess_threshold`` argument is retained for backward compatibility
    and test overrides, but in production the adaptive threshold is used
    when it would be more aggressive (higher) than the provided value.
    """
    adaptive_thresh = _adaptive_ess_threshold(state)
    effective_thresh = max(ess_threshold, adaptive_thresh)

    if state.ess >= effective_thresh:
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

    # MCMC rejuvenation: perturb cloned particles
    new_particles = _mutate_after_resample(new_particles, seed=seed + 8888, spread=0.08)

    state.particles = new_particles
    return state


def _progressive_temperature(top_mass: float) -> float:
    """Compute tempering factor based on collapse severity.

    Instead of a single fixed temperature of 0.5, this uses a progressive
    schedule that applies gentler tempering for moderate concentration and
    stronger tempering for severe collapse:

    - top_mass in [0.85, 0.92): mild tempering (0.7)
    - top_mass in [0.92, 0.97): moderate tempering (0.5)
    - top_mass >= 0.97:         aggressive tempering (0.3)
    """
    if top_mass >= 0.97:
        return 0.3
    elif top_mass >= 0.92:
        return 0.5
    else:
        return 0.7


def temper_if_collapsed(
    state: PosteriorState,
    mass_threshold: float = 0.85,
    temperature: float = 0.5,
) -> PosteriorState:
    """Down-temper weights if top particle mass exceeds threshold.

    Uses progressive tempering schedule based on collapse severity.
    The ``temperature`` argument is retained for backward compatibility
    and tests, but the progressive schedule is used when collapse is
    detected.
    """
    top_mass = state.top_particle_mass
    if top_mass <= mass_threshold:
        return state

    # Use progressive schedule instead of fixed temperature
    temp = _progressive_temperature(top_mass)

    for p in state.particles:
        p.log_weight *= temp

    return state
