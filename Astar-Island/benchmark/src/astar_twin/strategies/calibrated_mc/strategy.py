"""Calibrated MC strategy — improved final tensor construction.

Implements the Posterior Predictive / Final Tensor Calibration avenue:

1. **MC simulation** captures spatial variation in terrain outcomes.
2. **Filter-baseline prior** provides well-calibrated marginal class
   distributions derived from structural terrain templates.
3. **Temperature scaling** on MC output reduces overconfidence from
   potentially mismatched simulation parameters.
4. **Linear blend** combines MC spatial signal with the prior's better
   marginal calibration.  The blend weight is the primary tuning lever.
5. **Static terrain override** assigns near-certainty to ocean/mountain.

The key insight is that MC with default parameters captures *where* terrain
changes happen (spatial structure) but gets the *marginal class proportions*
wrong when hidden parameters don't match.  The filter-baseline templates have
good marginal proportions but no spatial variation.  Blending the two yields
predictions that are both spatially aware and well-calibrated.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import (
    NUM_CLASSES,
    ClassIndex,
    TerrainCode,
)
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams

# Prior templates derived from filter_baseline analysis.
# These encode the average class distribution for dynamic cells.
# Inland dynamic cells: Empty-heavy, moderate Settlement/Forest.
_INLAND_TEMPLATE = np.array([0.55, 0.18, 0.0, 0.07, 0.20, 0.0], dtype=np.float64)
# Coastal dynamic cells: adds Port probability, slightly less Empty.
_COASTAL_TEMPLATE = np.array([0.48, 0.17, 0.12, 0.06, 0.17, 0.0], dtype=np.float64)


class Zone(IntEnum):
    CORE = 0
    EXPANSION_RING = 1
    COASTAL_HUB = 2
    REMOTE_COASTAL = 3
    REMOTE_INLAND = 4


# Settlement-aware zone templates [Empty, Settlement, Port, Ruin, Forest, Mountain]
_CORE_TEMPLATE = np.array([0.25, 0.35, 0.0, 0.15, 0.25, 0.0], dtype=np.float64)
_RING_TEMPLATE = np.array([0.40, 0.25, 0.0, 0.10, 0.25, 0.0], dtype=np.float64)
_COASTAL_HUB_TEMPLATE = np.array([0.30, 0.20, 0.20, 0.10, 0.20, 0.0], dtype=np.float64)

# Per-zone blend weights (higher = trust prior more)
_ZONE_PRIOR_WEIGHTS: dict[Zone, float] = {
    Zone.CORE: 0.55,
    Zone.EXPANSION_RING: 0.50,
    Zone.COASTAL_HUB: 0.50,
    Zone.REMOTE_COASTAL: 0.40,
    Zone.REMOTE_INLAND: 0.40,
}


class CalibratedMCStrategy:
    """MC simulation blended with structural prior for calibrated predictions.

    The strategy blends two complementary signals:

    - **MC tensor**: spatial variation from Monte Carlo simulation, softened
      by temperature scaling to reduce overconfidence.
    - **Structural prior**: inland/coastal templates that provide well-
      calibrated marginal class proportions (from filter_baseline analysis).

    Static cells (ocean, mountain) receive near-certainty overrides.
    Dynamic cells get: ``prior_weight * prior + (1 - prior_weight) * mc_scaled``.

    Args:
        params: Simulation parameters (defaults to ``SimulationParams()``).
        n_runs: Number of MC simulation runs per seed.
        static_confidence: Probability assigned to the correct class for
            static cells (ocean, mountain).  Remainder is spread uniformly.
        prior_weight: Blend weight for the structural prior on dynamic cells.
            Higher values trust the prior more (safer when MC params are wrong).
            Empirically tuned to ~0.45.
        temperature: Temperature for scaling MC probabilities before blending.
            ``T > 1`` softens (safer), ``T < 1`` sharpens (riskier).
    """

    def __init__(
        self,
        params: SimulationParams | None = None,
        n_runs: int = 50,
        static_confidence: float = 0.97,
        prior_weight: float = 0.45,
        temperature: float = 1.3,
        use_settlement_zones: bool = True,
        use_adaptive_blend: bool = True,
        use_mc_variance: bool = True,
        n_subbatches: int = 5,
    ) -> None:
        self._params = params or SimulationParams()
        self._n_runs = n_runs
        self._static_confidence = static_confidence
        self._prior_weight = prior_weight
        self._temperature = temperature
        self._use_settlement_zones = use_settlement_zones
        self._use_adaptive_blend = use_adaptive_blend
        self._use_mc_variance = use_mc_variance
        self._n_subbatches = n_subbatches

    @property
    def name(self) -> str:
        if not any(
            [
                self._use_settlement_zones,
                self._use_adaptive_blend,
                self._use_mc_variance,
            ]
        ):
            return "calibrated_mc_v1"
        parts = ["calibrated_mc"]
        if self._use_settlement_zones:
            parts.append("zones")
        if self._use_adaptive_blend:
            parts.append("adaptive")
        if self._use_mc_variance:
            parts.append("variance")
        if len(parts) == 4:
            return "calibrated_mc"
        return "_".join(parts)

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        grid = initial_state.grid
        H = len(grid)
        W = len(grid[0])

        # Step 1: Run MC simulations
        if self._use_mc_variance:
            mc_tensor, variance = self._run_mc_with_variance(initial_state, H, W, base_seed)
        else:
            simulator = Simulator(params=self._params)
            runner = MCRunner(simulator)
            runs = runner.run_batch(initial_state, self._n_runs, base_seed=base_seed)
            mc_tensor = aggregate_runs(runs, H, W)
            variance = None

        # Step 2: Build masks
        is_static = _build_static_mask(grid, H, W)
        is_coastal = _build_coastal_mask(grid, H, W)

        # Step 3: Temperature-scale MC on dynamic cells
        mc_scaled = _apply_temperature_scaling(mc_tensor, is_static, self._temperature)

        # Step 4: Build structural prior from templates
        if self._use_settlement_zones:
            zone_map = self._build_zone_map(initial_state, H, W, is_static, is_coastal)
            prior = self._build_zone_prior(zone_map, H, W, is_static)
        else:
            zone_map = None
            prior = _build_template_prior(grid, H, W, is_static, is_coastal)

        # Step 5: Blend prior and MC for dynamic cells
        if self._use_adaptive_blend:
            if zone_map is None:
                zone_map = self._build_zone_map(initial_state, H, W, is_static, is_coastal)
            pw_map = self._compute_adaptive_weights(
                zone_map,
                H,
                W,
                variance if self._use_mc_variance else None,
            )
        else:
            pw_map = np.full((H, W), self._prior_weight, dtype=np.float64)

        result = np.empty((H, W, NUM_CLASSES), dtype=np.float64)
        dynamic = ~is_static
        pw_expanded = pw_map[..., np.newaxis]
        result[dynamic] = (
            pw_expanded[dynamic] * prior[dynamic]
            + (1.0 - pw_expanded[dynamic]) * mc_scaled[dynamic]
        )

        # Static cells: near-certainty override
        _fill_static_cells(result, grid, H, W, is_static, self._static_confidence)

        # Step 6: Normalize (harness applies safe_prediction for flooring)
        sums: NDArray[np.float64] = np.sum(result, axis=2, keepdims=True)
        sums = np.maximum(sums, 1e-10)
        normalized: NDArray[np.float64] = result / sums

        return normalized

    def _build_zone_map(
        self,
        initial_state: InitialState,
        H: int,
        W: int,
        is_static: NDArray[np.bool_],
        is_coastal: NDArray[np.bool_],
    ) -> NDArray[np.int8]:
        zone_map = np.full((H, W), Zone.REMOTE_INLAND, dtype=np.int8)
        alive_settlements = [s for s in initial_state.settlements if s.alive]

        if not alive_settlements:
            dynamic = ~is_static
            remote_coastal = dynamic & is_coastal
            zone_map[remote_coastal] = Zone.REMOTE_COASTAL
            zone_map[dynamic & ~is_coastal] = Zone.REMOTE_INLAND
            return zone_map

        for y in range(H):
            for x in range(W):
                if bool(is_static[y, x]):
                    zone_map[y, x] = Zone.REMOTE_INLAND
                    continue

                min_distance = min(
                    max(abs(settlement.x - x), abs(settlement.y - y))
                    for settlement in alive_settlements
                )
                has_nearby_port = any(
                    settlement.has_port and max(abs(settlement.x - x), abs(settlement.y - y)) <= 4
                    for settlement in alive_settlements
                )

                if bool(is_coastal[y, x]) and min_distance <= 4 and has_nearby_port:
                    zone_map[y, x] = Zone.COASTAL_HUB
                elif min_distance <= 1:
                    zone_map[y, x] = Zone.CORE
                elif 2 <= min_distance <= 4:
                    zone_map[y, x] = Zone.EXPANSION_RING
                elif bool(is_coastal[y, x]):
                    zone_map[y, x] = Zone.REMOTE_COASTAL
                else:
                    zone_map[y, x] = Zone.REMOTE_INLAND

        return zone_map

    def _build_zone_prior(
        self,
        zone_map: NDArray[np.int8],
        H: int,
        W: int,
        is_static: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        prior = np.full((H, W, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
        prior[zone_map == Zone.CORE] = _CORE_TEMPLATE
        prior[zone_map == Zone.EXPANSION_RING] = _RING_TEMPLATE
        prior[zone_map == Zone.COASTAL_HUB] = _COASTAL_HUB_TEMPLATE
        prior[zone_map == Zone.REMOTE_COASTAL] = _COASTAL_TEMPLATE
        prior[zone_map == Zone.REMOTE_INLAND] = _INLAND_TEMPLATE
        prior[is_static] = 1.0 / NUM_CLASSES
        return prior

    def _run_mc_with_variance(
        self,
        initial_state: InitialState,
        H: int,
        W: int,
        base_seed: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
            subbatch_tensors.append(aggregate_runs(runs, H, W))

        stacked = np.stack(subbatch_tensors, axis=0)
        mc_tensor = np.mean(stacked, axis=0)
        variance = np.var(stacked, axis=0)
        return mc_tensor, variance

    def _compute_adaptive_weights(
        self,
        zone_map: NDArray[np.int8],
        H: int,
        W: int,
        variance: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        weights = np.empty((H, W), dtype=np.float64)
        for zone, weight in _ZONE_PRIOR_WEIGHTS.items():
            weights[zone_map == zone] = weight

        if variance is not None:
            rare_class_var = variance[..., 1:4].sum(axis=-1)
            variance_adjustment = np.clip(rare_class_var * 2.0, -0.15, 0.15)
            weights = weights + variance_adjustment

        return np.clip(weights, 0.1, 0.8)


def _build_static_mask(grid: list[list[int]], H: int, W: int) -> NDArray[np.bool_]:
    """Build boolean mask: True for cells that never change (ocean, mountain)."""
    mask = np.zeros((H, W), dtype=np.bool_)
    for y in range(H):
        for x in range(W):
            code = grid[y][x]
            if code == TerrainCode.OCEAN or code == TerrainCode.MOUNTAIN:
                mask[y, x] = True
    return mask


def _build_coastal_mask(grid: list[list[int]], H: int, W: int) -> NDArray[np.bool_]:
    """Build boolean mask: True for land cells adjacent to ocean."""
    mask = np.zeros((H, W), dtype=np.bool_)
    for y in range(H):
        for x in range(W):
            code = grid[y][x]
            if code == TerrainCode.OCEAN or code == TerrainCode.MOUNTAIN:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == TerrainCode.OCEAN:
                    mask[y, x] = True
                    break
    return mask


def _fill_static_cells(
    result: NDArray[np.float64],
    grid: list[list[int]],
    H: int,
    W: int,
    is_static: NDArray[np.bool_],
    confidence: float,
) -> None:
    """Fill static cells in-place with near-certainty for their correct class."""
    residual = (1.0 - confidence) / (NUM_CLASSES - 1)
    for y in range(H):
        for x in range(W):
            if not bool(is_static[y, x]):
                continue
            code = grid[y][x]
            result[y, x, :] = residual
            if code == TerrainCode.OCEAN:
                result[y, x, ClassIndex.EMPTY] = confidence
            elif code == TerrainCode.MOUNTAIN:
                result[y, x, ClassIndex.MOUNTAIN] = confidence


def _apply_temperature_scaling(
    tensor: NDArray[np.float64],
    is_static: NDArray[np.bool_],
    temperature: float,
) -> NDArray[np.float64]:
    """Apply temperature scaling to dynamic cells only.

    For probability vector p, temperature-scaled version is:
        p_scaled[i] = p[i]^(1/T) / sum(p[j]^(1/T))

    T > 1 softens (more uniform, safer against overconfidence).
    T < 1 sharpens (more peaked, riskier).
    T = 1 is identity.
    """
    if abs(temperature - 1.0) < 1e-9:
        return tensor.copy()

    result = tensor.copy()
    dynamic = ~is_static
    inv_t = 1.0 / temperature

    # Vectorized over all dynamic cells at once
    dynamic_probs: NDArray[np.float64] = result[dynamic]  # shape: (N_dynamic, 6)
    clipped: NDArray[np.float64] = np.maximum(dynamic_probs, 1e-12)
    scaled: NDArray[np.float64] = np.power(clipped, inv_t)
    row_sums: NDArray[np.float64] = np.sum(scaled, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    result[dynamic] = scaled / row_sums

    return result


def _build_template_prior(
    grid: list[list[int]],
    H: int,
    W: int,
    is_static: NDArray[np.bool_],
    is_coastal: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Build structural prior from inland/coastal templates.

    Uses the same template approach as filter_baseline but produces a full
    probability tensor suitable for blending.  Static cells get uniform
    distributions (they'll be overridden anyway).  Dynamic cells get the
    coastal or inland template based on ocean adjacency.
    """
    prior = np.full((H, W, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)

    # Assign templates to dynamic cells
    dynamic = ~is_static
    inland = dynamic & ~is_coastal
    coastal = dynamic & is_coastal

    prior[inland] = _INLAND_TEMPLATE
    prior[coastal] = _COASTAL_TEMPLATE

    return prior
