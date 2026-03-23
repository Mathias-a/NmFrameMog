"""Hotspot-constrained information-directed sampling / budgeted bandit strategy.

Combines:
- **Lightweight posterior surrogate**: ensemble of parameter hypotheses drawn from
  the prior, weighted with an ABC-style kernel favouring hypotheses near the prior
  mode and producing confident (low-entropy) predictions.
- **IDS viewport selector**: arms are candidate hotspot windows on the map; the
  reward proxy selects windows that would yield the most information.
- **Reward proxy**: predictive entropy, top-particle disagreement, settlement-stat
  sensitivity, and a cross-seed parameter-identification proxy.

The strategy runs multiple IDS → refine → re-weight iterations, allocating extra
Monte-Carlo budget to the particles that are most informative in the highest-scoring
spatial regions.
"""

from __future__ import annotations

import dataclasses
import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import ClassIndex
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams, sample_default_prior_params

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-10

_SETTLEMENT_CLASSES: tuple[int, ...] = (
    ClassIndex.SETTLEMENT,
    ClassIndex.PORT,
    ClassIndex.RUIN,
)

# IDS reward proxy coefficients
_WEIGHT_ENTROPY: float = 0.35
_WEIGHT_DISAGREEMENT: float = 0.30
_WEIGHT_SENSITIVITY: float = 0.20
_WEIGHT_CROSS_SEED: float = 0.15

# ABC kernel coefficients
_ABC_DISTANCE_SCALE: float = 0.5
_ABC_ENTROPY_PENALTY: float = 0.1


# ---------------------------------------------------------------------------
# Hotspot window dataclass (candidate bandit arm)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _HotspotWindow:
    """Rectangular map region representing a candidate bandit arm."""

    y: int
    x: int
    h: int
    w: int

    def cells(self, map_h: int, map_w: int) -> list[tuple[int, int]]:
        """Return (row, col) pairs for all cells in this window, clipped to map."""
        result: list[tuple[int, int]] = []
        for row in range(self.y, min(self.y + self.h, map_h)):
            for col in range(self.x, min(self.x + self.w, map_w)):
                result.append((row, col))
        return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _safe_log(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise log with floor to avoid -inf."""
    return cast(NDArray[np.float64], np.log(np.maximum(arr, _EPSILON)))


def _param_distance(params: SimulationParams, defaults: SimulationParams) -> float:
    """Normalised squared L2 distance between two parameter sets on numeric fields.

    Enum (StrEnum) fields are excluded.  Each numeric field is normalised by the
    absolute value of the default to make the distance scale-invariant.
    """
    dist_sq = 0.0
    n = 0
    param_fields = cast(
        tuple[dataclasses.Field[object], ...],
        dataclasses.fields(SimulationParams),
    )
    for field_info in param_fields:
        field_name: str = field_info.name
        val = getattr(params, field_name)
        default_val = getattr(defaults, field_name)
        # Skip non-numeric fields (StrEnum values are str instances)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if not isinstance(default_val, (int, float)) or isinstance(default_val, bool):
            continue
        scale = max(abs(float(default_val)), 0.01)
        dist_sq += ((float(val) - float(default_val)) / scale) ** 2
        n += 1
    return dist_sq / max(n, 1)


# ---------------------------------------------------------------------------
# IDS reward proxy components
# ---------------------------------------------------------------------------


def _predictive_entropy(
    stacked: NDArray[np.float64],
    weights: NDArray[np.float64],
    cells: list[tuple[int, int]],
) -> float:
    """Shannon entropy of the weighted ensemble mean prediction over a window.

    High entropy = high uncertainty = high information potential from observation.
    """
    w = weights[:, np.newaxis, np.newaxis, np.newaxis]
    mean_pred = cast(NDArray[np.float64], (stacked * w).sum(axis=0))
    total = 0.0
    for y, x in cells:
        p = cast(NDArray[np.float64], mean_pred[y, x])
        entropy_term = np.sum(p * _safe_log(p))
        total += float(-entropy_term)
    return total / max(len(cells), 1)


def _top_particle_disagreement(
    stacked: NDArray[np.float64],
    weights: NDArray[np.float64],
    cells: list[tuple[int, int]],
) -> float:
    """KL divergence of the top-2 weighted particles from the ensemble mean.

    When the most influential particles disagree with the consensus, the window
    is highly informative for resolving parameter uncertainty.
    """
    w = weights[:, np.newaxis, np.newaxis, np.newaxis]
    mean_pred = cast(NDArray[np.float64], (stacked * w).sum(axis=0))
    top2_indices_array = cast(NDArray[np.int64], np.argsort(weights)[-2:])
    top2_indices = [
        int(cast(np.int64, top2_indices_array[i])) for i in range(len(top2_indices_array))
    ]

    total = 0.0
    count = 0
    for idx in top2_indices:
        for y, x in cells:
            p = cast(NDArray[np.float64], stacked[idx, y, x])
            q = cast(NDArray[np.float64], mean_pred[y, x])
            kl_term = np.sum(p * (_safe_log(p) - _safe_log(q)))
            kl = float(kl_term)
            total += max(kl, 0.0)
            count += 1
    return total / max(count, 1)


def _settlement_stat_sensitivity(
    stacked: NDArray[np.float64],
    cells: list[tuple[int, int]],
) -> float:
    """Variance of settlement-related class probabilities across particles.

    High variance in settlement/port/ruin classes signals that the window
    discriminates between parameter hypotheses on the most score-critical cells.
    """
    total_var = 0.0
    for y, x in cells:
        for cls in _SETTLEMENT_CLASSES:
            vals = cast(NDArray[np.float64], stacked[:, y, x, cls])
            total_var += float(cast(np.float64, np.var(vals)))
    return total_var / max(len(cells), 1)


def _cross_seed_identification_proxy(
    stacked: NDArray[np.float64],
    cells: list[tuple[int, int]],
) -> float:
    """Inter-particle variance proxy for cross-seed parameter identification.

    Windows where particles disagree across *all* classes help identify the true
    hidden parameters, and that knowledge transfers to every seed (since all seeds
    share the same hidden parameters).
    """
    cell_vars: list[float] = []
    for y, x in cells:
        cell_predictions = cast(NDArray[np.float64], stacked[:, y, x, :])
        class_variances = cast(NDArray[np.float64], np.var(cell_predictions, axis=0))
        cell_vars.append(float(cast(np.float64, class_variances.sum())))
    return sum(cell_vars) / max(len(cell_vars), 1)


def _ids_reward_proxy(
    stacked: NDArray[np.float64],
    weights: NDArray[np.float64],
    window: _HotspotWindow,
    map_h: int,
    map_w: int,
) -> float:
    """Combined IDS reward for a candidate hotspot window."""
    cells = window.cells(map_h, map_w)
    if not cells:
        return 0.0

    entropy = _predictive_entropy(stacked, weights, cells)
    disagreement = _top_particle_disagreement(stacked, weights, cells)
    sensitivity = _settlement_stat_sensitivity(stacked, cells)
    cross_seed = _cross_seed_identification_proxy(stacked, cells)

    return (
        _WEIGHT_ENTROPY * entropy
        + _WEIGHT_DISAGREEMENT * disagreement
        + _WEIGHT_SENSITIVITY * sensitivity
        + _WEIGHT_CROSS_SEED * cross_seed
    )


# ---------------------------------------------------------------------------
# Hotspot window generation and selection
# ---------------------------------------------------------------------------


def _generate_candidate_windows(
    initial_state: InitialState,
    map_h: int,
    map_w: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> list[_HotspotWindow]:
    """Generate candidate hotspot windows centred on settlements and random regions."""
    candidates: list[_HotspotWindow] = []
    window_size = min(10, map_h, map_w)

    # Settlement-centred windows (natural hotspots for dynamics)
    for s in initial_state.settlements:
        if s.alive:
            y0 = max(0, s.y - window_size // 2)
            x0 = max(0, s.x - window_size // 2)
            h = min(window_size, map_h - y0)
            w = min(window_size, map_w - x0)
            candidates.append(_HotspotWindow(y=y0, x=x0, h=h, w=w))

    # Fill remaining slots with random windows (constrained to valid viewport dims)
    min_wh = min(5, map_h)
    max_wh = min(15, map_h)
    min_ww = min(5, map_w)
    max_ww = min(15, map_w)

    while len(candidates) < n_candidates:
        wh = int(rng.integers(min_wh, max_wh + 1))
        ww = int(rng.integers(min_ww, max_ww + 1))
        wy = int(rng.integers(0, max(1, map_h - wh + 1)))
        wx = int(rng.integers(0, max(1, map_w - ww + 1)))
        candidates.append(_HotspotWindow(y=wy, x=wx, h=wh, w=ww))

    return candidates[:n_candidates]


def _select_top_windows(
    stacked: NDArray[np.float64],
    weights: NDArray[np.float64],
    candidates: list[_HotspotWindow],
    map_h: int,
    map_w: int,
    n_select: int,
) -> list[_HotspotWindow]:
    """Select the windows with highest IDS reward proxy."""
    scores = [_ids_reward_proxy(stacked, weights, w, map_h, map_w) for w in candidates]
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in sorted_indices[:n_select]]


# ---------------------------------------------------------------------------
# ABC-style particle weighting
# ---------------------------------------------------------------------------


def _compute_particle_weights(
    particles: list[SimulationParams],
    predictions: list[NDArray[np.float64]],
    defaults: SimulationParams,
) -> NDArray[np.float64]:
    """ABC-style weighting: particles near the prior mode with confident predictions.

    Two factors multiplied together:
      1. **Proximity kernel**: Gaussian-like ABC kernel on normalised parameter
         distance — particles closer to the default (prior mode) get more weight.
      2. **Confidence penalty**: particles producing diffuse (high-entropy) predictions
         are down-weighted, favouring internally consistent hypotheses.
    """
    n = len(particles)
    weights = np.ones(n, dtype=np.float64)

    for i, params in enumerate(particles):
        dist = _param_distance(params, defaults)
        weights[i] = math.exp(-_ABC_DISTANCE_SCALE * dist)

    for i, pred in enumerate(predictions):
        pred_log_product = cast(NDArray[np.float64], pred * _safe_log(pred))
        entropy_sums = cast(NDArray[np.float64], np.sum(pred_log_product, axis=-1))
        cell_entropies = cast(NDArray[np.float64], -entropy_sums)
        mean_entropy = float(cast(np.float64, cell_entropies.mean()))
        current_weight = float(cast(np.float64, weights[i]))
        weights[i] = math.exp(-_ABC_ENTROPY_PENALTY * mean_entropy) * current_weight

    total = float(cast(np.float64, weights.sum()))
    if total > 0:
        weights /= total
    else:
        weights[:] = 1.0 / n
    return weights


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------


class IDSBanditStrategy:
    """Hotspot-constrained information-directed sampling with budgeted bandit.

    Combines a lightweight posterior surrogate (ensemble of parameter hypotheses
    with ABC-style weighting) with an IDS viewport selector that allocates MC
    budget adaptively across particles and map regions.

    **Algorithm overview:**

    1. Sample ``n_particles`` parameter hypotheses from the prior
       (``sample_default_prior_params`` with controlled ``prior_spread``).
    2. Run ``mc_runs_per_particle`` Monte-Carlo simulations per particle to
       build per-particle ``(H, W, 6)`` probability tensors.
    3. Generate candidate hotspot windows (settlement-centred + random).
    4. For each refinement round:

       a. Score each window via the IDS reward proxy (entropy + disagreement
          + settlement sensitivity + cross-seed parameter identification).
       b. Identify which particles are most informative in the top windows
          (highest KL divergence from ensemble mean).
       c. Run extra MC simulations for those particles and blend predictions.
       d. Re-compute ABC-style particle weights.

    5. Return the weighted ensemble mean prediction.

    Args:
        n_particles: Number of parameter hypotheses in the ensemble.
        mc_runs_per_particle: Base MC runs per particle per seed.
        n_hotspot_candidates: Number of candidate windows for IDS scoring.
        n_refinement_rounds: IDS → refine → re-weight iterations.
        prior_spread: Controls how far particles deviate from defaults (0–1).
    """

    def __init__(
        self,
        n_particles: int = 8,
        mc_runs_per_particle: int = 12,
        n_hotspot_candidates: int = 20,
        n_refinement_rounds: int = 2,
        prior_spread: float = 0.4,
    ) -> None:
        self._n_particles = n_particles
        self._mc_runs_per_particle = mc_runs_per_particle
        self._n_hotspot_candidates = n_hotspot_candidates
        self._n_refinement_rounds = n_refinement_rounds
        self._prior_spread = prior_spread

    @property
    def name(self) -> str:
        return "ids_bandit"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        """Return an ``(H, W, 6)`` probability tensor via IDS ensemble prediction."""
        H = len(initial_state.grid)
        W = len(initial_state.grid[0])
        rng = np.random.default_rng(base_seed)
        defaults = SimulationParams()

        # ------------------------------------------------------------------
        # 1. Sample parameter particle ensemble from the prior.
        # ------------------------------------------------------------------
        particles = [
            sample_default_prior_params(
                seed=base_seed + 1_000_000 * i,
                spread=self._prior_spread,
            )
            for i in range(self._n_particles)
        ]

        # ------------------------------------------------------------------
        # 2. Run base MC for each particle.
        # ------------------------------------------------------------------
        predictions: list[NDArray[np.float64]] = []
        for i, params in enumerate(particles):
            sim = Simulator(params=params)
            runner = MCRunner(sim)
            runs = runner.run_batch(
                initial_state=initial_state,
                n_runs=self._mc_runs_per_particle,
                base_seed=base_seed + 100_000 * i,
            )
            predictions.append(aggregate_runs(runs, H, W))

        # ------------------------------------------------------------------
        # 3. Initial ABC weighting.
        # ------------------------------------------------------------------
        weights = _compute_particle_weights(particles, predictions, defaults)

        # ------------------------------------------------------------------
        # 4. Generate candidate hotspot windows.
        # ------------------------------------------------------------------
        candidates = _generate_candidate_windows(
            initial_state,
            H,
            W,
            self._n_hotspot_candidates,
            rng,
        )

        # ------------------------------------------------------------------
        # 5. IDS refinement loop: score windows → refine informative particles.
        # ------------------------------------------------------------------
        for round_idx in range(self._n_refinement_rounds):
            stacked = cast(NDArray[np.float64], np.stack(predictions))  # (N, H, W, 6)

            # Select the most informative windows via IDS reward proxy.
            top_windows = _select_top_windows(
                stacked,
                weights,
                candidates,
                H,
                W,
                n_select=max(1, self._n_particles // 2),
            )

            # Collect all cells covered by the top windows.
            all_cells: set[tuple[int, int]] = set()
            for w in top_windows:
                all_cells.update(w.cells(H, W))

            # Score each particle by divergence from ensemble mean in hotspot cells.
            w_4d = weights[:, np.newaxis, np.newaxis, np.newaxis]
            mean_pred = cast(NDArray[np.float64], (stacked * w_4d).sum(axis=0))
            particle_info = np.zeros(self._n_particles, dtype=np.float64)

            for y, x in all_cells:
                for i in range(self._n_particles):
                    p = cast(NDArray[np.float64], stacked[i, y, x])
                    q = cast(NDArray[np.float64], mean_pred[y, x])
                    kl_term = np.sum(p * (_safe_log(p) - _safe_log(q)))
                    kl = float(kl_term)
                    particle_info[i] = float(cast(np.float64, particle_info[i])) + max(kl, 0.0)

            # Refine the most divergent particles — they carry the most
            # information about the true parameters in the hotspot regions.
            n_refine = max(1, self._n_particles // 4)
            refine_indices_array = cast(NDArray[np.int64], np.argsort(particle_info)[-n_refine:])
            refine_indices = [
                int(cast(np.int64, refine_indices_array[i]))
                for i in range(len(refine_indices_array))
            ]

            for idx in refine_indices:
                sim = Simulator(params=particles[idx])
                runner = MCRunner(sim)
                extra_runs = runner.run_batch(
                    initial_state=initial_state,
                    n_runs=self._mc_runs_per_particle,
                    base_seed=base_seed + 200_000 * idx + 50_000 * (round_idx + 1),
                )
                extra_pred = aggregate_runs(extra_runs, H, W)
                # Blend: effectively doubles the sample size for this particle.
                predictions[idx] = 0.5 * predictions[idx] + 0.5 * extra_pred

            # Re-compute weights after refinement.
            weights = _compute_particle_weights(particles, predictions, defaults)

        # ------------------------------------------------------------------
        # 6. Return weighted ensemble mean.
        # ------------------------------------------------------------------
        final_stacked = cast(NDArray[np.float64], np.stack(predictions))
        final_w = weights[:, np.newaxis, np.newaxis, np.newaxis]
        return cast(NDArray[np.float64], (final_stacked * final_w).sum(axis=0))
