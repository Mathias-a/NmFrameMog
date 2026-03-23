"""Posterior-predictive tensor generation for all seeds.

Uses top-K particles weighted by posterior to generate full-map predictions:
  1. Select top-6 particles by weight.
  2. Allocate simulation runs proportional to weight (min 4 per particle).
  3. Run full-map MC per particle/seed pair.
  4. Combine tensors by normalized posterior weight.
  5. Finalize through the shared tensor finalizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import cast

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.engine import Simulator
from astar_twin.mc.aggregate import aggregate_runs
from astar_twin.mc.runner import MCRunner
from astar_twin.solver.inference.posterior import PosteriorState
from astar_twin.solver.observe.ledger import ObservationLedger
from astar_twin.solver.predict.finalize import finalize_tensor

# Defaults
DEFAULT_TOP_K = 6
DEFAULT_SIMS_PER_SEED = 64
FALLBACK_SIMS_PER_SEED = 32
MIN_RUNS_PER_PARTICLE = 4


@dataclass
class PredictionMetrics:
    """Metrics from the prediction generation process."""

    seed_index: int
    n_particles_used: int
    total_sims: int
    fallback_used: bool
    runs_per_particle: list[int] = field(default_factory=list)


def allocate_runs_to_particles(
    weights: list[float],
    total_runs: int,
    min_per_particle: int = MIN_RUNS_PER_PARTICLE,
) -> list[int]:
    """Allocate simulation runs to particles proportional to weight.

    Guarantees at least min_per_particle runs per particle.
    Total may be clipped to sum of allocation.

    Args:
        weights: Normalized particle weights (sum to ~1).
        total_runs: Total budget of runs.
        min_per_particle: Minimum runs per particle.

    Returns:
        List of run counts per particle.
    """
    n = len(weights)
    if n == 0:
        return []

    # Start with minimum allocation
    alloc = [min_per_particle] * n
    remaining = total_runs - sum(alloc)

    if remaining <= 0:
        # Not enough budget for even min; distribute proportionally
        raw = [max(1, int(w * total_runs)) for w in weights]
        # Ensure total matches
        diff = total_runs - sum(raw)
        if diff > 0:
            # Add remaining to highest-weight particle
            best = max(range(len(weights)), key=lambda idx: weights[idx])
            raw[best] += diff
        elif diff < 0:
            # Remove from lowest-weight particle
            worst = min(range(len(weights)), key=lambda idx: weights[idx])
            raw[worst] = max(1, raw[worst] + diff)
        return raw

    # Distribute remaining proportionally
    total_weight = sum(weights)
    normalized = [weight / total_weight for weight in weights]
    fractional = [weight * remaining for weight in normalized]
    floored = [int(floor(value)) for value in fractional]
    leftovers = [fractional[i] - float(floored[i]) for i in range(len(fractional))]
    # Distribute rounding remainder to particles with largest leftovers
    leftover_count = remaining - sum(floored)
    if leftover_count > 0:
        top_idx = sorted(range(len(leftovers)), key=lambda idx: leftovers[idx], reverse=True)[
            :leftover_count
        ]
        for idx in top_idx:
            floored[idx] += 1

    for i in range(n):
        alloc[i] += int(floored[i])

    return alloc


def predict_seed(
    posterior: PosteriorState,
    initial_state: InitialState,
    seed_index: int,
    map_height: int,
    map_width: int,
    top_k: int = DEFAULT_TOP_K,
    sims_per_seed: int = DEFAULT_SIMS_PER_SEED,
    base_seed: int = 0,
    observation_ledger: ObservationLedger | None = None,
) -> tuple[NDArray[np.float64], PredictionMetrics]:
    """Generate posterior-predictive tensor for a single seed.

    Args:
        posterior: Current posterior state with weighted particles.
        initial_state: Initial state for this seed.
        seed_index: Seed index (for metrics).
        map_height: Full map height.
        map_width: Full map width.
        top_k: Number of top particles to use.
        sims_per_seed: Total simulation runs to allocate.
        base_seed: Base RNG seed for reproducibility.

    Returns:
        Tuple of (H×W×6 finalized tensor, metrics).
    """
    # Select top-K particles
    k = min(top_k, len(posterior.particles))
    if k == 0:
        from astar_twin.solver.baselines import uniform_baseline

        uniform = uniform_baseline(map_height, map_width)
        metrics = PredictionMetrics(
            seed_index=seed_index,
            n_particles_used=0,
            total_sims=0,
            fallback_used=True,
            runs_per_particle=[],
        )
        return uniform, metrics

    top_indices = posterior.top_k_indices(k)
    top_particles = [posterior.particles[i] for i in top_indices]

    # Get normalized weights for selected particles
    all_weights = posterior.normalized_weights()
    selected_weights = [all_weights[i] for i in top_indices]
    # Re-normalize among selected
    w_sum = sum(selected_weights)
    selected_weights = [w / w_sum for w in selected_weights] if w_sum > 0 else [1.0 / k] * k

    # Allocate runs
    run_alloc = allocate_runs_to_particles(selected_weights, sims_per_seed)

    # Run MC for each particle
    combined = cast(
        NDArray[np.float64], np.zeros((map_height, map_width, NUM_CLASSES), dtype=np.float64)
    )

    for i in range(len(top_particles)):
        particle = top_particles[i]
        n_runs = run_alloc[i]
        weight = selected_weights[i]
        sim_params = particle.to_simulation_params()
        simulator = Simulator(params=sim_params)
        runner = MCRunner(simulator)

        # Offset seed by particle index to avoid correlated runs
        particle_seed = base_seed + i * sims_per_seed
        runs = runner.run_batch(initial_state, n_runs, base_seed=particle_seed)
        particle_tensor = aggregate_runs(runs, map_height, map_width)

        # Weight by posterior mass
        combined += weight * particle_tensor

    # Blend empirical cell evidence from ledger
    if observation_ledger is not None:
        seed_counts = observation_ledger.per_seed_class_counts.get(seed_index)
        seed_visits = observation_ledger.per_seed_visit_counts.get(seed_index)
        if seed_counts is not None and seed_visits is not None:
            height = int(seed_visits.shape[0])
            width = int(seed_visits.shape[1])
            for y in range(height):
                for x in range(width):
                    visits = float(cast(np.float64, seed_visits[y, x]))
                    if visits <= 0.0:
                        continue
                    visit_sum = 0.0
                    for cls_idx in range(NUM_CLASSES):
                        visit_sum += float(cast(np.float64, seed_counts[y, x, cls_idx]))
                    denom = max(visit_sum, 1e-10)
                    alpha = min(max(visits / 3.0, 0.0), 0.6)
                    for cls_idx in range(NUM_CLASSES):
                        empirical = float(cast(np.float64, seed_counts[y, x, cls_idx])) / denom
                        current = float(cast(np.float64, combined[y, x, cls_idx]))
                        combined[y, x, cls_idx] = (1.0 - alpha) * current + alpha * empirical

    # Finalize
    result = finalize_tensor(combined, map_height, map_width, initial_state)

    metrics = PredictionMetrics(
        seed_index=seed_index,
        n_particles_used=k,
        total_sims=sum(run_alloc),
        fallback_used=(sims_per_seed == FALLBACK_SIMS_PER_SEED),
        runs_per_particle=run_alloc,
    )

    return result, metrics


def predict_all_seeds(
    posterior: PosteriorState,
    initial_states: list[InitialState],
    map_height: int,
    map_width: int,
    top_k: int = DEFAULT_TOP_K,
    sims_per_seed: int = DEFAULT_SIMS_PER_SEED,
    base_seed: int = 0,
    observation_ledger: ObservationLedger | None = None,
    runtime_fraction: float = 0.0,
) -> tuple[list[NDArray[np.float64]], list[PredictionMetrics]]:
    """Generate posterior-predictive tensors for all seeds.

    Args:
        posterior: Current posterior state.
        initial_states: Initial states for all seeds.
        map_height: Full map height.
        map_width: Full map width.
        top_k: Number of top particles to use.
        sims_per_seed: Total sim runs per seed (default 64).
        base_seed: Base RNG seed.
        runtime_fraction: Fraction of runtime budget already used (0-1).
            If > 0.80, falls back to FALLBACK_SIMS_PER_SEED.

    Returns:
        Tuple of (list of H×W×6 tensors, list of metrics).
    """
    # Runtime budget check
    effective_sims = sims_per_seed
    if runtime_fraction > 0.80:
        effective_sims = FALLBACK_SIMS_PER_SEED

    tensors: list[NDArray[np.float64]] = []
    all_metrics: list[PredictionMetrics] = []

    for seed_idx, initial_state in enumerate(initial_states):
        seed_offset = base_seed + seed_idx * 10000
        tensor, metrics = predict_seed(
            posterior,
            initial_state,
            seed_idx,
            map_height,
            map_width,
            top_k=top_k,
            sims_per_seed=effective_sims,
            base_seed=seed_offset,
            observation_ledger=observation_ledger,
        )
        tensors.append(tensor)
        all_metrics.append(metrics)

    return tensors, all_metrics
