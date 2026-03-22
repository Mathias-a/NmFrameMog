"""Probability utilities: floors, entropy, KL divergence, renormalization, masks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 6
GRID_H, GRID_W = 40, 40

# API terrain codes -> prediction class mapping
TERRAIN_TO_CLASS: dict[int, int] = {
    0: 0,  # Empty -> class 0
    10: 0,  # Ocean -> class 0
    11: 0,  # Plains -> class 0
    1: 1,  # Settlement
    2: 2,  # Port
    3: 3,  # Ruin
    4: 4,  # Forest
    5: 5,  # Mountain
}

# Static terrain codes (excluded from scoring)
STATIC_CODES = {10, 5}  # Ocean, Mountain

FLOOR_IMPOSSIBLE = 0.001  # Mountain on non-mountain, Port on non-coastal
FLOOR_STANDARD = 0.01  # All other uncertain classes


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def make_static_mask(initial_grid: NDArray[np.int_]) -> NDArray[np.bool_]:
    """Boolean mask: True where cell is static (ocean=10 or mountain=5)."""
    return np.isin(initial_grid, [10, 5])


def make_dynamic_mask(initial_grid: NDArray[np.int_]) -> NDArray[np.bool_]:
    """Boolean mask: True where cell is dynamic (not static)."""
    return ~make_static_mask(initial_grid)


def make_ocean_mask(initial_grid: NDArray[np.int_]) -> NDArray[np.bool_]:
    """Boolean mask: True where cell is ocean (code 10)."""
    return np.equal(initial_grid, 10)


def make_mountain_mask(initial_grid: NDArray[np.int_]) -> NDArray[np.bool_]:
    """Boolean mask: True where cell is mountain (code 5)."""
    return np.equal(initial_grid, 5)


def make_coastal_mask(initial_grid: NDArray[np.int_]) -> NDArray[np.bool_]:
    """Boolean mask: True where non-ocean cell is adjacent to ocean (4-connected)."""
    ocean = make_ocean_mask(initial_grid)
    adj_ocean = np.zeros_like(ocean)
    adj_ocean[1:, :] |= ocean[:-1, :]
    adj_ocean[:-1, :] |= ocean[1:, :]
    adj_ocean[:, 1:] |= ocean[:, :-1]
    adj_ocean[:, :-1] |= ocean[:, 1:]
    return adj_ocean & ~ocean


def initial_grid_to_classes(initial_grid: NDArray[np.int_]) -> NDArray[np.int_]:
    """Convert initial_grid terrain codes to prediction class indices [0..5]."""
    result = np.zeros_like(initial_grid)
    for code, cls in TERRAIN_TO_CLASS.items():
        mask: NDArray[np.bool_] = np.equal(initial_grid, code)
        result[mask] = cls
    return result


# ---------------------------------------------------------------------------
# Floor and normalization
# ---------------------------------------------------------------------------


def renormalize(probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Renormalize probability vectors to sum to 1 along last axis."""
    sums_val: np.float64 | NDArray[np.float64] = np.sum(
        probs, axis=-1, dtype=np.float64
    )
    sums = np.expand_dims(sums_val, axis=-1)
    cond: NDArray[np.bool_] = np.equal(sums, 0.0)
    sums_nz = np.where(cond, 1.0, sums)
    return probs / sums_nz


def apply_floors(
    probs: NDArray[np.float64],
    initial_grid: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Apply class-specific probability floors and renormalize.

    Rules from reference.md:
        1. Static cells (ocean=10, mountain=5): one-hot on their class.
        2. Mountain class (5) on non-mountain dynamic cells: floor 0.001.
        3. Port class (2) on non-coastal dynamic cells: floor 0.001.
        4. All other dynamic class-cell combos: floor 0.01.
        5. Renormalize to sum=1.
    """
    out = probs.copy()

    static = make_static_mask(initial_grid)
    ocean = make_ocean_mask(initial_grid)
    mountain = make_mountain_mask(initial_grid)
    coastal = make_coastal_mask(initial_grid)
    dynamic = ~static

    # Static cells: one-hot
    out[ocean] = 0.0
    out[ocean, 0] = 1.0
    out[mountain] = 0.0
    out[mountain, 5] = 1.0

    # Dynamic cells: apply standard floor first
    out[dynamic] = np.maximum(out[dynamic], FLOOR_STANDARD)

    # Mountain class (5) on dynamic cells -> impossible floor
    out[dynamic, 5] = FLOOR_IMPOSSIBLE

    # Port class (2) on non-coastal dynamic cells -> impossible floor
    non_coastal_dynamic = dynamic & ~coastal
    out[non_coastal_dynamic, 2] = FLOOR_IMPOSSIBLE

    # Renormalize dynamic cells
    out[dynamic] = renormalize(out[dynamic])

    return out


# ---------------------------------------------------------------------------
# Information-theoretic functions
# ---------------------------------------------------------------------------


def entropy(probs: NDArray[np.float64], eps: float = 1e-12) -> NDArray[np.float64]:
    """Shannon entropy: H = -sum(p * log(p)) along last axis."""
    p = np.clip(probs, eps, 1.0)
    log_p = np.log(p)
    product = probs * log_p
    result: np.float64 | NDArray[np.float64] = np.sum(
        product, axis=-1, dtype=np.float64
    )
    assert isinstance(result, np.ndarray)
    negated: NDArray[np.float64] = -result
    return negated


def kl_divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """KL(p || q) per vector along last axis."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    log_ratio = np.log(p_safe / q_safe)
    product = p * log_ratio
    result: np.float64 | NDArray[np.float64] = np.sum(
        product, axis=-1, dtype=np.float64
    )
    assert isinstance(result, np.ndarray)
    return result


def weighted_kl(
    ground_truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
) -> float:
    """Entropy-weighted KL divergence (the official scoring metric).

    weighted_KL = sum(H(cell) * KL(gt || pred)) / sum(H(cell))

    Only cells with entropy > 0 contribute.
    """
    h_cells = entropy(ground_truth)
    kl_cells = kl_divergence(ground_truth, prediction)

    mask = h_cells > 0
    if not np.any(mask):
        return 0.0

    numerator = float(np.sum(h_cells[mask] * kl_cells[mask]))
    denominator = float(np.sum(h_cells[mask]))
    return numerator / denominator


def score_prediction(
    ground_truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
) -> float:
    """Compute prediction score: 100 * exp(-3 * weighted_KL), clipped to [0, 100]."""
    wkl = weighted_kl(ground_truth, prediction)
    raw: float = 100.0 * float(np.exp(-3.0 * wkl))
    return min(max(raw, 0.0), 100.0)


def bayesian_blend(
    prior: NDArray[np.float64],
    counts: NDArray[np.float64],
    n_obs: NDArray[np.int_],
    alpha: float = 5.0,
) -> NDArray[np.float64]:
    """Blend GBDT prior with empirical query observations using Bayesian update.

    p_final = (alpha * prior + counts) / (alpha + n)

    For cells with n_obs == 0, the prior is returned unchanged.

    Args:
        prior: (H, W, 6) GBDT predicted probabilities.
        counts: (H, W, 6) accumulated one-hot counts from queries.
        n_obs: (H, W) number of observations per cell.
        alpha: Prior strength (effective pseudo-count for GBDT).

    Returns:
        (H, W, 6) blended probabilities, renormalized.
    """
    n_float: NDArray[np.float64] = np.asarray(n_obs, dtype=np.float64)
    n_expanded: NDArray[np.float64] = n_float.reshape(*n_float.shape, 1)
    observed: NDArray[np.bool_] = n_expanded > 0

    numerator: NDArray[np.float64] = alpha * prior + counts
    denominator: NDArray[np.float64] = alpha + n_expanded
    safe_denom: NDArray[np.float64] = np.maximum(denominator, 1e-12)
    update: NDArray[np.float64] = numerator / safe_denom

    result: NDArray[np.float64] = prior.copy()
    mask_3d: NDArray[np.bool_] = np.broadcast_to(observed, prior.shape).copy()
    result[mask_3d] = update[mask_3d]

    return renormalize(result)
