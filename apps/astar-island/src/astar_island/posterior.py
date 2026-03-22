"""Bayesian posterior update module.

Implements the Dirichlet-Multinomial conjugate update pipeline:
  geometric pool → ESS → Dirichlet alpha → conjugate update → posterior mean → floors.

All operations are fully vectorized numpy — no Python loops over cells.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_island.config import Config
from astar_island.types import (
    CODE_TO_CLASS,
    AlphaTensor,
    BoolMask,
    CountTensor,
    ESSMap,
    Grid,
    K,
    ProbTensor,
)

# ---------------------------------------------------------------------------
# Terrain constants
# ---------------------------------------------------------------------------

_OCEAN_CODE: int = 10
_MOUNTAIN_CLASS: int = 5
_PORT_CLASS: int = 2

# Build a vectorized lookup from terrain code → prediction class.
# Maximum terrain code observed in CODE_TO_CLASS is 11.
_MAX_CODE: int = max(CODE_TO_CLASS) + 1
_CODE_TO_CLASS_LUT: NDArray[np.int32] = np.zeros(_MAX_CODE, dtype=np.int32)
for _code, _cls in CODE_TO_CLASS.items():
    _CODE_TO_CLASS_LUT[_code] = _cls


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def entropy(p: ProbTensor, eps: float = 1e-4) -> NDArray[np.float64]:
    """Compute entropy along last axis.

    Returns array of shape (...,) — one scalar per cell.
    """
    p_safe: NDArray[np.float64] = np.clip(p, eps, 1.0)
    result: NDArray[np.float64] = np.asarray(
        -np.sum(p_safe * np.log(p_safe), axis=-1),
        dtype=np.float64,
    )
    return result


# ---------------------------------------------------------------------------
# Heterogeneous ESS
# ---------------------------------------------------------------------------


def compute_ess(p_round: ProbTensor, config: Config) -> ESSMap:
    """Compute per-cell effective sample size from prior confidence.

    ess = clip(c_base × (1 + ess_confidence_weight × (1 − H_norm)), ess_min, ess_max)

    H_norm = entropy(p_round) / log(K), normalized to [0, 1].

    Returns: shape (H, W)
    """
    h_raw: NDArray[np.float64] = entropy(p_round, eps=config.eps)
    log_k: float = float(np.log(K))
    h_norm: NDArray[np.float64] = np.clip(h_raw / log_k, 0.0, 1.0)
    ess: NDArray[np.float64] = config.c_base * (
        1.0 + config.ess_confidence_weight * (1.0 - h_norm)
    )
    return np.clip(ess, config.ess_min, config.ess_max)


# ---------------------------------------------------------------------------
# Dirichlet parameterization
# ---------------------------------------------------------------------------


def init_alpha(p_round: ProbTensor, ess: ESSMap) -> AlphaTensor:
    """Convert probabilities + ESS to Dirichlet parameters.

    alpha[h, w, k] = ess[h, w] * p_round[h, w, k]

    Returns: shape (H, W, K)
    """
    # ess has shape (H, W); p_round has shape (H, W, K)
    return ess[..., np.newaxis] * p_round


# ---------------------------------------------------------------------------
# Conjugate update
# ---------------------------------------------------------------------------


def bayesian_update(alpha: AlphaTensor, counts: CountTensor) -> AlphaTensor:
    """Apply Dirichlet conjugate update: alpha_post = alpha + counts.

    Returns: shape (H, W, K)
    """
    return alpha + counts.astype(np.float64)


# ---------------------------------------------------------------------------
# Posterior predictive
# ---------------------------------------------------------------------------


def posterior_predictive(alpha: AlphaTensor) -> ProbTensor:
    """Compute posterior mean: pred[h, w, k] = alpha[h, w, k] / Σ_k alpha[h, w, :].

    Returns: shape (H, W, K)
    """
    alpha_sum: NDArray[np.float64] = np.sum(alpha, axis=-1, keepdims=True)
    return alpha / np.maximum(alpha_sum, 1e-10)


# ---------------------------------------------------------------------------
# Geometric pooling
# ---------------------------------------------------------------------------


def geometric_pool(
    p_base: ProbTensor,
    p_template: ProbTensor,
    lambda_prior: float,
    eps: float = 1e-4,
) -> ProbTensor:
    """Geometric pooling: p_round ∝ p_base^λ · p_template^(1−λ).

    log_p = λ·log(clip(p_base)) + (1−λ)·log(clip(p_template))
    p_round = exp(log_p) / Σ exp(log_p)

    Returns: shape matching inputs, normalized along last axis.
    """
    log_p: NDArray[np.float64] = lambda_prior * np.log(np.clip(p_base, eps, 1.0)) + (
        1.0 - lambda_prior
    ) * np.log(np.clip(p_template, eps, 1.0))
    # Subtract max for numerical stability before exp
    log_p_shifted: NDArray[np.float64] = log_p - np.max(log_p, axis=-1, keepdims=True)
    p_unnorm: NDArray[np.float64] = np.exp(log_p_shifted)
    p_sum: NDArray[np.float64] = np.sum(p_unnorm, axis=-1, keepdims=True)
    return p_unnorm / np.maximum(p_sum, 1e-10)


# ---------------------------------------------------------------------------
# Floor application
# ---------------------------------------------------------------------------


def apply_floors(pred: ProbTensor, initial_grid: Grid, config: Config) -> ProbTensor:
    """Apply probability floors and structural zeros.

    Rules applied in order (floors are lower-bounds; structural zeros are hard):
    - Static cells (ocean=10, mountain=5): one-hot on their class
    - Mountain class (5) on non-mountain dynamic cells: floor_impossible
    - Port class (2) on non-coastal dynamic cells: floor_impossible
    - All other classes: max(pred, floor_standard)
    - Renormalize to sum to 1.0

    A cell is coastal if it is not ocean but within Chebyshev distance 1
    of an ocean cell (8-connected neighbourhood, including diagonals).

    Returns: shape (H, W, K)
    """
    from scipy.ndimage import distance_transform_cdt

    result: NDArray[np.float64] = pred.copy()

    # -----------------------------------------------------------------------
    # Identify cell categories
    # -----------------------------------------------------------------------
    is_ocean: BoolMask = initial_grid == _OCEAN_CODE  # (H, W)
    is_mountain: BoolMask = initial_grid == 5  # terrain code 5 = Mountain
    is_static: BoolMask = is_ocean | is_mountain  # (H, W)
    is_dynamic: BoolMask = ~is_static  # (H, W)

    # Coastal: non-ocean cell within Chebyshev distance ≤ 1 of ocean
    dist_coast: NDArray[np.float64] = np.asarray(
        distance_transform_cdt(~is_ocean),
        dtype=np.float64,
    )
    adjacent_to_ocean: BoolMask = np.asarray(dist_coast <= 1.0, dtype=np.bool_)
    is_non_coastal_dynamic: BoolMask = is_dynamic & ~adjacent_to_ocean

    # -----------------------------------------------------------------------
    # Apply floor_standard to all dynamic cells first
    # -----------------------------------------------------------------------
    result[is_dynamic] = np.maximum(result[is_dynamic], config.floor_standard)

    # -----------------------------------------------------------------------
    # Mountain class on non-mountain dynamic cells: floor_impossible
    # -----------------------------------------------------------------------
    non_mountain_dynamic: BoolMask = is_dynamic & ~is_mountain
    result[non_mountain_dynamic, _MOUNTAIN_CLASS] = np.minimum(
        result[non_mountain_dynamic, _MOUNTAIN_CLASS], config.floor_impossible
    )

    # -----------------------------------------------------------------------
    # Port class on non-coastal dynamic cells: floor_impossible
    # -----------------------------------------------------------------------
    result[is_non_coastal_dynamic, _PORT_CLASS] = np.minimum(
        result[is_non_coastal_dynamic, _PORT_CLASS], config.floor_impossible
    )

    # -----------------------------------------------------------------------
    # Static cells: one-hot (applied after floor_standard to override)
    # -----------------------------------------------------------------------
    # Ocean cells → one-hot class 0
    ocean_onehot: NDArray[np.float64] = np.zeros(K, dtype=np.float64)
    ocean_onehot[0] = 1.0
    result[is_ocean] = ocean_onehot

    # Mountain cells → one-hot class 5
    mountain_onehot: NDArray[np.float64] = np.zeros(K, dtype=np.float64)
    mountain_onehot[_MOUNTAIN_CLASS] = 1.0
    result[is_mountain] = mountain_onehot

    # -----------------------------------------------------------------------
    # Renormalize dynamic cells
    # -----------------------------------------------------------------------
    dyn_h, dyn_w = np.where(is_dynamic)
    if dyn_h.size > 0:
        dyn_probs: NDArray[np.float64] = result[dyn_h, dyn_w, :]  # (N, K)
        dyn_sum: NDArray[np.float64] = np.sum(dyn_probs, axis=-1, keepdims=True)
        result[dyn_h, dyn_w, :] = dyn_probs / np.maximum(dyn_sum, 1e-10)

    return result


# ---------------------------------------------------------------------------
# Observation counting
# ---------------------------------------------------------------------------


def accumulate_counts(
    counts: CountTensor,
    grid_observation: NDArray[np.int32],
    vx: int,
    vy: int,
    vw: int,
    vh: int,
) -> CountTensor:
    """Add observation counts from a viewport query result.

    grid_observation: (vh, vw) terrain codes from simulate response.
    Maps terrain codes to prediction classes via CODE_TO_CLASS, then
    increments counts[vy:vy+vh, vx:vx+vw, class] += 1.

    Returns: updated counts tensor (mutates in-place and returns).
    """
    # Clip codes to valid range before LUT lookup
    obs_clipped: NDArray[np.int32] = np.clip(grid_observation, 0, _MAX_CODE - 1).astype(
        np.int32
    )
    classes: NDArray[np.int32] = _CODE_TO_CLASS_LUT[obs_clipped]  # (vh, vw)

    # Build one-hot increments and accumulate — shape: (vh, vw, K)
    one_hot: NDArray[np.int32] = np.zeros((vh, vw, K), dtype=np.int32)
    rows: NDArray[np.intp] = np.arange(vh, dtype=np.intp)[:, np.newaxis]
    cols: NDArray[np.intp] = np.arange(vw, dtype=np.intp)[np.newaxis, :]
    one_hot[rows, cols, classes] = 1

    counts[vy : vy + vh, vx : vx + vw, :] += one_hot
    return counts


# ---------------------------------------------------------------------------
# Full posterior pipeline
# ---------------------------------------------------------------------------


def compute_posterior(
    p_base: ProbTensor,
    p_template: ProbTensor | None,
    counts: CountTensor,
    initial_grid: Grid,
    config: Config,
) -> ProbTensor:
    """Full pipeline: geometric pool → ESS → alpha → update → predict → floor.

    If p_template is None, skip geometric pooling (use p_base directly).
    """
    # Step 1: geometric pool (or pass-through)
    if p_template is not None:
        p_round: ProbTensor = geometric_pool(
            p_base, p_template, config.lambda_prior, config.eps
        )
    else:
        p_round = p_base

    # Step 2: compute per-cell ESS
    ess: ESSMap = compute_ess(p_round, config)

    # Step 3: build Dirichlet alpha prior
    alpha_prior: AlphaTensor = init_alpha(p_round, ess)

    # Step 4: conjugate update with observations
    alpha_post: AlphaTensor = bayesian_update(alpha_prior, counts)

    # Step 5: posterior predictive mean
    pred: ProbTensor = posterior_predictive(alpha_post)

    # Step 6: apply floors and structural zeros
    return apply_floors(pred, initial_grid, config)


# ---------------------------------------------------------------------------
# Observation count tracking
# ---------------------------------------------------------------------------


def observation_count_map(counts: CountTensor) -> NDArray[np.int32]:
    """Return per-cell total observation count. Shape: (H, W)."""
    result: NDArray[np.int32] = np.asarray(np.sum(counts, axis=-1), dtype=np.int32)
    return result
