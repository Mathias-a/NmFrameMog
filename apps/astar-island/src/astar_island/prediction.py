"""Prediction tensor utilities — probability flooring and validation.

Applies class-specific probability floors to prevent infinite KL divergence:
  - Mountain class on non-mountain cells: floor 0.001 (impossible transition)
  - Port class on non-coastal cells: floor 0.001 (impossible transition)
  - All other class-cell pairs: floor 0.01
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_island.terrain import (
    GRID_SIZE,
    NUM_PREDICTION_CLASSES,
    PredictionClass,
    TerrainType,
)

PredictionTensor = NDArray[np.float64]

DEFAULT_FLOOR: float = 0.01
IMPOSSIBLE_FLOOR: float = 0.001


def compute_masks(
    initial_grid: list[list[int]],
    width: int = GRID_SIZE,
    height: int = GRID_SIZE,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Compute mountain mask and coastal mask from the initial grid.

    Args:
        initial_grid: Grid with grid[x][y] indexing, raw terrain codes.
        width: Grid width (x dimension).
        height: Grid height (y dimension).

    Returns:
        (mountain_mask, coastal_mask) each of shape (W, H).
    """
    ocean_code = int(TerrainType.OCEAN)
    mountain_code = int(TerrainType.MOUNTAIN)

    grid_arr = np.zeros((width, height), dtype=np.int32)
    for x in range(width):
        for y in range(height):
            if x < len(initial_grid) and y < len(initial_grid[x]):
                grid_arr[x, y] = initial_grid[x][y]

    mountain_mask: NDArray[np.bool_] = grid_arr == mountain_code
    ocean_mask: NDArray[np.bool_] = grid_arr == ocean_code

    coastal_mask = np.zeros((width, height), dtype=np.bool_)
    if width > 1:
        coastal_mask[1:, :] |= ocean_mask[:-1, :]
        coastal_mask[:-1, :] |= ocean_mask[1:, :]
    if height > 1:
        coastal_mask[:, 1:] |= ocean_mask[:, :-1]
        coastal_mask[:, :-1] |= ocean_mask[:, 1:]
    coastal_mask &= ~ocean_mask

    return mountain_mask, coastal_mask


def apply_probability_floor(
    pred: PredictionTensor,
    initial_grid: list[list[int]],
    floor: float = DEFAULT_FLOOR,
) -> PredictionTensor:
    """Apply class-specific probability floors and renormalize.

    Args:
        pred: Prediction tensor of shape (W, H, C).
        initial_grid: Raw initial terrain grid (grid[x][y] indexing).
        floor: Base minimum probability.

    Returns:
        Safe prediction tensor with all values >= floor, summing to 1.
    """
    width, height = pred.shape[0], pred.shape[1]
    mountain_mask, coastal_mask = compute_masks(initial_grid, width, height)

    floor_tensor = np.full_like(pred, floor)

    mountain_idx = int(PredictionClass.MOUNTAIN)
    port_idx = int(PredictionClass.PORT)

    non_mountain = ~mountain_mask
    floor_tensor[non_mountain, mountain_idx] = IMPOSSIBLE_FLOOR

    non_coastal = ~coastal_mask
    floor_tensor[non_coastal, port_idx] = IMPOSSIBLE_FLOOR

    # Iterative clamp-and-renormalize: after renormalization some values may
    # drop below the floor again.  Repeat until convergence (typically 2-3
    # iterations for 6 classes).
    result: PredictionTensor = pred.copy()
    max_iterations = 10
    for _ in range(max_iterations):
        clamped = np.maximum(result, floor_tensor)
        denom = clamped.sum(axis=-1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        result = clamped / denom
        if bool(np.all(result >= floor_tensor - 1e-15)):
            break
    return result


def validate_prediction(
    pred: PredictionTensor,
    expected_width: int = GRID_SIZE,
    expected_height: int = GRID_SIZE,
    atol: float = 1e-6,
) -> list[str]:
    """Validate a prediction tensor. Returns empty list if valid."""
    errors: list[str] = []

    expected_shape = (expected_width, expected_height, NUM_PREDICTION_CLASSES)
    if pred.shape != expected_shape:
        errors.append(f"Expected shape {expected_shape}, got {pred.shape}")
        return errors

    if bool(np.any(np.isnan(pred))):
        errors.append("Prediction contains NaN values")
    if bool(np.any(np.isinf(pred))):
        errors.append("Prediction contains infinite values")
    if bool(np.any(pred < 0)):
        errors.append("Prediction contains negative probabilities")
    if bool(np.any(pred == 0.0)):
        errors.append("Prediction contains zero probabilities — KL will diverge")

    sums = pred.sum(axis=-1)
    if not bool(np.allclose(sums, 1.0, atol=atol)):
        max_dev = float(np.max(np.abs(sums - 1.0)))
        errors.append(f"Probabilities don't sum to 1 (max deviation: {max_dev:.2e})")

    return errors
