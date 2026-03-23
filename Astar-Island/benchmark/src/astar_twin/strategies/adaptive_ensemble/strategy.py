"""Adaptive ensemble strategy — data-calibrated zone priors with MC adaptation.

Key improvements over calibrated_mc_zones:

1. **Data-derived zone templates** computed from ground truth averages across
   all 15 historical rounds, replacing the manually-tuned (and significantly
   miscalibrated) templates in calibrated_mc_zones.

2. **MC-signal template scaling** — uses the ratio of MC-observed settlement
   activity to expected activity as a signal for whether this round has
   high or low settlement dynamics, and scales zone templates accordingly.

3. **Temperature-scaled MC blending** — temperature scaling softens the MC
   predictions before blending with the prior, reducing overconfidence from
   mismatched parameters.

4. **Adaptive per-zone blend weights** — zones closer to settlements get
   higher prior weight (MC spatial signal is most reliable nearby), while
   remote zones rely more on the prior template.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams
from astar_twin.strategies.calibrated_mc.strategy import (
    CalibratedMCStrategy,
    Zone,
    _apply_temperature_scaling,
    _build_coastal_mask,
    _build_static_mask,
    _fill_static_cells,
)

# Data-derived optimal zone templates, computed as the mean ground truth
# probability distribution per zone across all 15 real historical rounds
# (75 seed-level examples).  These replace the hand-tuned templates in
# calibrated_mc_zones which were significantly miscalibrated (e.g. 35%
# settlement in CORE vs actual 23%, 18% settlement in REMOTE_INLAND vs 4%).
#
# Format: [Empty, Settlement, Port, Ruin, Forest, Mountain]
_CORE_TEMPLATE = np.array([0.535, 0.231, 0.008, 0.020, 0.206, 0.0], dtype=np.float64)
_RING_TEMPLATE = np.array([0.630, 0.123, 0.008, 0.012, 0.227, 0.0], dtype=np.float64)
_COASTAL_HUB_TEMPLATE = np.array([0.620, 0.074, 0.086, 0.013, 0.207, 0.0], dtype=np.float64)
_REMOTE_COASTAL_TEMPLATE = np.array([0.728, 0.020, 0.014, 0.003, 0.236, 0.0], dtype=np.float64)
_REMOTE_INLAND_TEMPLATE = np.array([0.697, 0.044, 0.000, 0.004, 0.255, 0.0], dtype=np.float64)

# Expected MC settlement fraction with default SimulationParams.
# Used as the reference for scaling: if observed MC settlement is higher
# than this, the round likely has more settlement activity than average.
_EXPECTED_MC_SETTLEMENT_FRAC = 0.002

# Per-zone blend weights: what fraction of the final prediction comes from
# the zone prior (vs the MC tensor).  Higher = trust prior more.
# Core/expansion zones get higher prior weight because MC systematically
# under-predicts settlement (0.2% vs 23% GT), so the prior is more valuable
# near settlements.  Remote zones get lower weight because the MC is more
# accurate there (both MC and GT are dominated by empty/forest).
_ZONE_PRIOR_WEIGHTS: dict[Zone, float] = {
    Zone.CORE: 0.55,
    Zone.EXPANSION_RING: 0.50,
    Zone.COASTAL_HUB: 0.50,
    Zone.REMOTE_COASTAL: 0.35,
    Zone.REMOTE_INLAND: 0.35,
}


class AdaptiveEnsembleStrategy:
    """MC simulation with data-calibrated zone priors and MC-based adaptation."""

    name = "adaptive_ensemble"

    def __init__(
        self,
        n_runs: int = 50,
        static_confidence: float = 0.97,
        temperature: float = 1.3,
        mc_scale_sensitivity: float = 0.40,
        n_subbatches: int = 5,
    ) -> None:
        self._params = SimulationParams()
        self._n_runs = n_runs
        self._static_confidence = static_confidence
        self._temperature = temperature
        self._mc_scale_sensitivity = mc_scale_sensitivity
        self._n_subbatches = n_subbatches
        self._zone_helper = CalibratedMCStrategy(n_runs=1)

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        del budget

        grid = initial_state.grid
        height = len(grid)
        width = len(grid[0])

        # Step 1: Run MC simulation with sub-batching for variance estimation
        mc_tensor, mc_variance = self._run_mc_with_variance(initial_state, height, width, base_seed)

        # Step 2: Build masks and zone map
        is_static = _build_static_mask(grid, height, width)
        is_coastal = _build_coastal_mask(grid, height, width)
        zone_map = self._zone_helper._build_zone_map(
            initial_state, height, width, is_static, is_coastal
        )

        # Step 3: Build data-derived zone prior and adapt to this round
        zone_prior = _build_zone_prior(zone_map, height, width, is_static)
        mc_settlement_signal = _compute_mc_settlement_signal(mc_tensor, is_static)
        adapted_prior = _adapt_prior_to_round(
            zone_prior,
            mc_settlement_signal,
            zone_map,
            is_static,
            self._mc_scale_sensitivity,
        )

        # Step 4: Temperature-scale MC on dynamic cells
        mc_scaled = _apply_temperature_scaling(mc_tensor, is_static, self._temperature)

        # Step 5: Adaptive per-zone blending
        pw_map = _compute_adaptive_blend_weights(zone_map, height, width, mc_variance)

        result = np.empty((height, width, NUM_CLASSES), dtype=np.float64)
        dynamic = ~is_static
        pw_expanded = pw_map[..., np.newaxis]
        result[dynamic] = (
            pw_expanded[dynamic] * adapted_prior[dynamic]
            + (1.0 - pw_expanded[dynamic]) * mc_scaled[dynamic]
        )

        # Step 6: Static cells override
        _fill_static_cells(result, grid, height, width, is_static, self._static_confidence)

        # Step 7: Normalize
        sums: NDArray[np.float64] = np.sum(result, axis=2, keepdims=True)
        sums = np.maximum(sums, 1e-10)
        normalized: NDArray[np.float64] = result / sums
        return normalized

    def _run_mc_with_variance(
        self,
        initial_state: InitialState,
        height: int,
        width: int,
        base_seed: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run MC in sub-batches to estimate both mean and variance."""
        if self._n_subbatches <= 0:
            raise ValueError("n_subbatches must be positive")

        simulator = Simulator(params=self._params)
        runner = MCRunner(simulator)
        runs_per_batch = self._n_runs // self._n_subbatches
        if runs_per_batch <= 0:
            raise ValueError("n_runs must be at least n_subbatches when variance is enabled")

        subbatch_tensors: list[NDArray[np.float64]] = []
        for batch_index in range(self._n_subbatches):
            runs = runner.run_batch(
                initial_state,
                runs_per_batch,
                base_seed=base_seed + batch_index * runs_per_batch,
            )
            subbatch_tensors.append(aggregate_runs(runs, height, width))

        stacked = np.stack(subbatch_tensors, axis=0)
        mc_tensor = np.mean(stacked, axis=0)
        variance = np.var(stacked, axis=0)
        return mc_tensor, variance


def _build_zone_prior(
    zone_map: NDArray[np.int8],
    height: int,
    width: int,
    is_static: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Build zone prior from data-derived optimal templates."""
    prior = np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    prior[zone_map == Zone.CORE] = _CORE_TEMPLATE
    prior[zone_map == Zone.EXPANSION_RING] = _RING_TEMPLATE
    prior[zone_map == Zone.COASTAL_HUB] = _COASTAL_HUB_TEMPLATE
    prior[zone_map == Zone.REMOTE_COASTAL] = _REMOTE_COASTAL_TEMPLATE
    prior[zone_map == Zone.REMOTE_INLAND] = _REMOTE_INLAND_TEMPLATE
    prior[is_static] = 1.0 / NUM_CLASSES
    return prior


def _compute_mc_settlement_signal(
    mc_tensor: NDArray[np.float64],
    is_static: NDArray[np.bool_],
) -> float:
    """Compute the MC settlement activity signal for this round.

    Returns the ratio of observed MC settlement fraction to the expected
    baseline.  Values > 1.0 suggest this round has higher settlement activity
    than average; values < 1.0 suggest lower.
    """
    dynamic = ~is_static
    if not np.any(dynamic):
        return 1.0
    mc_settlement_frac = float(np.mean(mc_tensor[dynamic, ClassIndex.SETTLEMENT]))
    if _EXPECTED_MC_SETTLEMENT_FRAC <= 0.0:
        return 1.0
    return mc_settlement_frac / _EXPECTED_MC_SETTLEMENT_FRAC


def _adapt_prior_to_round(
    zone_prior: NDArray[np.float64],
    mc_signal: float,
    zone_map: NDArray[np.int8],
    is_static: NDArray[np.bool_],
    sensitivity: float,
) -> NDArray[np.float64]:
    """Scale zone prior settlement probabilities based on MC signal.

    When MC observes more settlement activity than expected (mc_signal > 1),
    we scale settlement probabilities up (and empty down).  When MC observes
    less (mc_signal < 1), we scale settlement down (and empty up).

    The sensitivity parameter controls how aggressively we adapt.  At 0.0,
    no adaptation occurs; at 1.0, the prior is fully scaled by the signal.
    """
    if abs(sensitivity) < 1e-9:
        return zone_prior.copy()

    adapted = zone_prior.copy()

    # Compute scaling factor: log-scale to handle asymmetry
    # mc_signal=2 → scale_factor > 1 (more settlement)
    # mc_signal=0.5 → scale_factor < 1 (less settlement)
    # Clamp to avoid extreme scaling
    log_signal = np.clip(np.log(max(mc_signal, 0.01)), -2.0, 2.0)
    scale_factor = np.exp(sensitivity * log_signal)

    dynamic = ~is_static
    settlement_idx = ClassIndex.SETTLEMENT
    port_idx = ClassIndex.PORT
    ruin_idx = ClassIndex.RUIN
    empty_idx = ClassIndex.EMPTY

    # Scale settlement and port probabilities
    adapted[dynamic, settlement_idx] *= scale_factor
    adapted[dynamic, port_idx] *= scale_factor
    # Ruin scales with settlement activity (more settlements → more ruins)
    adapted[dynamic, ruin_idx] *= np.sqrt(scale_factor)

    # Redistribute probability mass: empty absorbs the change
    old_settlement_mass = zone_prior[dynamic, settlement_idx] + zone_prior[dynamic, port_idx]
    new_settlement_mass = adapted[dynamic, settlement_idx] + adapted[dynamic, port_idx]
    old_ruin_mass = zone_prior[dynamic, ruin_idx]
    new_ruin_mass = adapted[dynamic, ruin_idx]
    mass_change = (new_settlement_mass - old_settlement_mass) + (new_ruin_mass - old_ruin_mass)
    # Reduce empty to compensate, but never below a floor
    adapted[dynamic, empty_idx] = np.maximum(adapted[dynamic, empty_idx] - mass_change, 0.05)

    # Re-normalize
    sums = np.sum(adapted, axis=-1, keepdims=True)
    sums = np.maximum(sums, 1e-12)
    return adapted / sums


def _compute_adaptive_blend_weights(
    zone_map: NDArray[np.int8],
    height: int,
    width: int,
    variance: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Compute per-cell blend weights (prior weight) based on zone and MC variance."""
    weights = np.empty((height, width), dtype=np.float64)
    for zone, weight in _ZONE_PRIOR_WEIGHTS.items():
        weights[zone_map == zone] = weight

    if variance is not None:
        # High MC variance on rare classes → trust prior more
        rare_class_var = variance[..., 1:4].sum(axis=-1)
        variance_adjustment = np.clip(rare_class_var * 2.0, -0.15, 0.15)
        weights = weights + variance_adjustment

    return np.clip(weights, 0.1, 0.8)
