"""Two-sided likelihood computation for particle posterior updates.

For each observed viewport, computes particle likelihood using inner MC:
  - Run simulations per particle (ONCE — shared across grid and stats)
  - Map simulated terrain to classes
  - Estimate per-cell predictive probabilities
  - Compute grid log-likelihood and settlement-stat log-likelihood
  - Total = adaptive_grid_weight * loglik_grid + adaptive_stats_weight * loglik_stats

Calibration improvements (worktree-6 avenue):
  - Single inner MC loop shared across grid and stats terms (halves sim cost).
  - Adaptive grid/stats weighting based on settlement density in viewport.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import (
    InitialState,
    SimSettlement,
    SimulateResponse,
    ViewportBounds,
)
from astar_twin.contracts.types import NUM_CLASSES, TERRAIN_TO_CLASS
from astar_twin.engine import Simulator
from astar_twin.solver.inference.particles import Particle
from astar_twin.solver.observe.features import ObservationFeatures, extract_features


@dataclass(frozen=True)
class CombinedSimResult:
    """Result of a combined inner MC loop for one particle.

    Holds both the grid class probability tensor and the per-run settlement
    feature vectors, computed from the *same* simulator runs.
    """

    class_probs: NDArray[np.float64]  # shape (vh, vw, NUM_CLASSES)
    features_list: list[ObservationFeatures]


def _simulate_viewport_combined(
    particle: Particle,
    initial_state: InitialState,
    viewport_x: int,
    viewport_y: int,
    viewport_w: int,
    viewport_h: int,
    n_inner_runs: int,
    base_seed: int,
) -> CombinedSimResult:
    """Run inner MC simulations and extract BOTH grid classes and settlement features.

    This replaces the two separate functions ``_simulate_viewport_classes`` and
    ``_simulate_settlement_stats`` that previously duplicated all simulator
    calls.  Each simulation run now contributes to *both* likelihood terms.
    """
    sim_params = particle.to_simulation_params()
    simulator = Simulator(params=sim_params)

    counts = np.zeros((viewport_h, viewport_w, NUM_CLASSES), dtype=np.float64)
    features_list: list[ObservationFeatures] = []

    for i in range(n_inner_runs):
        state = simulator.run(initial_state=initial_state, sim_seed=base_seed + i)
        vp = state.grid.viewport(viewport_x, viewport_y, viewport_w, viewport_h)

        # --- Grid class counts (same logic as old _simulate_viewport_classes) ---
        for y in range(vp.height):
            for x in range(vp.width):
                code = vp.get(y, x)
                cls_idx = TERRAIN_TO_CLASS.get(code, 0)
                counts[y, x, cls_idx] += 1.0

        # --- Settlement features (same logic as old _simulate_settlement_stats) ---
        settlements_in_vp: list[SimSettlement] = []
        for s in state.settlements:
            if (
                viewport_x <= s.x < viewport_x + viewport_w
                and viewport_y <= s.y < viewport_y + viewport_h
            ):
                settlements_in_vp.append(
                    SimSettlement(
                        x=s.x,
                        y=s.y,
                        population=s.population,
                        food=s.food,
                        wealth=s.wealth,
                        defense=s.defense,
                        has_port=s.has_port,
                        alive=s.alive,
                        owner_id=s.owner_id,
                    )
                )

        mock_resp = SimulateResponse(
            grid=vp.to_list(),
            settlements=settlements_in_vp,
            viewport=ViewportBounds(x=viewport_x, y=viewport_y, w=viewport_w, h=viewport_h),
            width=40,
            height=40,
            queries_used=0,
            queries_max=50,
        )
        features_list.append(extract_features(mock_resp))

    # Normalize grid counts to probabilities with floor
    counts /= max(n_inner_runs, 1)
    counts = np.maximum(counts, 1e-6)
    sums = np.sum(counts, axis=2, keepdims=True)
    counts /= sums

    return CombinedSimResult(class_probs=counts, features_list=features_list)


def compute_grid_loglik(
    observed_response: SimulateResponse,
    predicted_probs: NDArray[np.float64],
) -> float:
    """Compute log-likelihood of observed grid given predicted class probabilities."""
    loglik = 0.0
    for y, row in enumerate(observed_response.grid):
        for x, code in enumerate(row):
            if y < predicted_probs.shape[0] and x < predicted_probs.shape[1]:
                cls_idx = TERRAIN_TO_CLASS.get(code, 0)
                prob = max(float(predicted_probs[y, x, cls_idx]), 1e-6)
                loglik += np.log(prob)
    return float(loglik)


def compute_stats_loglik(
    observed_features: ObservationFeatures,
    simulated_features_list: list[ObservationFeatures],
) -> float:
    """Compute settlement-stat log-likelihood using diagonal Gaussian penalty."""
    if not simulated_features_list or observed_features.alive_count == 0:
        return 0.0  # No info to compare

    # Compute mean and variance of simulated stats
    stat_names = [
        ("population_mean", "population_var"),
        ("food_mean", "food_var"),
        ("wealth_mean", "wealth_var"),
        ("defense_mean", "defense_var"),
        ("prosperity_proxy_mean", "prosperity_proxy_var"),
    ]

    loglik = 0.0
    for mean_attr, _var_attr in stat_names:
        obs_val = getattr(observed_features, mean_attr)
        sim_vals = [getattr(f, mean_attr) for f in simulated_features_list]
        sim_mean = float(np.mean(sim_vals))
        sim_var = float(np.var(sim_vals)) + 0.01  # prevent zero variance

        # Gaussian log-likelihood
        diff = obs_val - sim_mean
        loglik -= 0.5 * (diff**2) / sim_var

    # Also penalize alive count difference
    obs_alive = observed_features.alive_count
    sim_alive_vals = [f.alive_count for f in simulated_features_list]
    sim_alive_mean = float(np.mean(sim_alive_vals))
    sim_alive_var = float(np.var(sim_alive_vals)) + 0.5
    loglik -= 0.5 * ((obs_alive - sim_alive_mean) ** 2) / sim_alive_var

    # Port count
    obs_ports = observed_features.port_count
    sim_port_vals = [f.port_count for f in simulated_features_list]
    sim_port_mean = float(np.mean(sim_port_vals))
    sim_port_var = float(np.var(sim_port_vals)) + 0.5
    loglik -= 0.5 * ((obs_ports - sim_port_mean) ** 2) / sim_port_var

    return float(loglik)


def _adaptive_weights(observed_response: SimulateResponse) -> tuple[float, float]:
    """Compute adaptive grid/stats weights based on settlement density.

    When the viewport contains settlements, their stats carry meaningful
    information about hidden parameters (growth rates, collapse thresholds,
    trade ranges, etc.).  When there are no settlements, stats contribute
    nothing and would just add noise.

    Returns:
        (grid_weight, stats_weight) that sum to 1.0.
    """
    alive_in_vp = sum(1 for s in observed_response.settlements if s.alive)

    if alive_in_vp == 0:
        # No settlement data — grid only
        return 1.0, 0.0
    elif alive_in_vp <= 2:
        # Few settlements — stats are noisy, lean toward grid
        return 0.75, 0.25
    elif alive_in_vp <= 5:
        # Moderate settlement density — balanced weighting
        return 0.60, 0.40
    else:
        # Many settlements — stats are very informative
        return 0.50, 0.50


def compute_particle_loglik(
    particle: Particle,
    observed_response: SimulateResponse,
    initial_state: InitialState,
    n_inner_runs: int = 6,
    base_seed: int = 0,
) -> float:
    """Compute total log-likelihood for a particle given an observation.

    Uses a single shared MC loop for both grid and stats terms (previously
    duplicated), and adaptive weighting based on settlement density.
    """
    vp = observed_response.viewport

    # Single combined inner MC loop (was two separate loops)
    combined = _simulate_viewport_combined(
        particle,
        initial_state,
        vp.x,
        vp.y,
        vp.w,
        vp.h,
        n_inner_runs,
        base_seed,
    )

    grid_ll = compute_grid_loglik(observed_response, combined.class_probs)

    obs_features = extract_features(observed_response)
    stats_ll = compute_stats_loglik(obs_features, combined.features_list)

    grid_w, stats_w = _adaptive_weights(observed_response)
    return grid_w * grid_ll + stats_w * stats_ll
