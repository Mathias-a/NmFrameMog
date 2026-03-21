"""Tensor finalization and safety enforcement.

Every prediction path MUST call finalize_tensor before returning.
This centralizes shape checks, finite-value checks, normalization,
static terrain overrides, and safe_prediction flooring.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import (
    NUM_CLASSES,
    ClassIndex,
    TerrainCode,
)
from astar_twin.scoring import safe_prediction


class TensorValidationError(Exception):
    """Raised when a tensor fails shape or finite-value validation."""


def _apply_static_overrides(
    tensor: NDArray[np.float64],
    initial_state: InitialState,
) -> NDArray[np.float64]:
    """Override probabilities for cells that are guaranteed static (ocean, mountain).

    Ocean/mountain cells cannot change during simulation, so we push probability
    mass toward the correct class before flooring.
    """
    result = tensor.copy()
    grid = initial_state.grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    for y in range(min(height, result.shape[0])):
        for x in range(min(width, result.shape[1])):
            code = grid[y][x]
            if code == TerrainCode.OCEAN:
                # Ocean → class Empty(0) with high confidence
                result[y, x, :] = 0.01
                result[y, x, ClassIndex.EMPTY] = 0.94
            elif code == TerrainCode.MOUNTAIN:
                # Mountain → class Mountain(5) with high confidence
                result[y, x, :] = 0.01
                result[y, x, ClassIndex.MOUNTAIN] = 0.94

    return result


def finalize_tensor(
    tensor: NDArray[np.float64],
    height: int,
    width: int,
    initial_state: InitialState | None = None,
) -> NDArray[np.float64]:
    """Validate, override statics, normalize, and floor a prediction tensor.

    Args:
        tensor: Raw H×W×6 prediction tensor.
        height: Expected height.
        width: Expected width.
        initial_state: If provided, apply static terrain overrides.

    Returns:
        Safe, normalized H×W×6 tensor with all probs >= 0.01.

    Raises:
        TensorValidationError: On wrong shape or non-finite values.
    """
    # Shape check
    expected_shape = (height, width, NUM_CLASSES)
    if tensor.shape != expected_shape:
        raise TensorValidationError(
            f"Expected shape {expected_shape}, got {tensor.shape}"
        )

    # Finite check
    if not np.all(np.isfinite(tensor)):
        raise TensorValidationError("Tensor contains non-finite values (NaN or Inf)")

    # Apply static overrides before normalization
    if initial_state is not None:
        tensor = _apply_static_overrides(tensor, initial_state)

    # Normalize per cell
    sums = np.sum(tensor, axis=2, keepdims=True)
    # Avoid division by zero for cells that are all zeros
    sums = np.maximum(sums, 1e-10)
    tensor = tensor / sums

    # Safe prediction floor
    return safe_prediction(tensor)
