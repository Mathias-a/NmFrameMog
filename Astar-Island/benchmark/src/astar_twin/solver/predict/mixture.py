"""Soft expert mixture blending for prediction tensors.

Blends multiple expert predictions using soft weights (e.g., softmax over confidence).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.solver.predict.finalize import finalize_tensor


def apply_soft_mixture(
    expert_tensors: list[NDArray[np.float64]],
    expert_weights: list[float] | NDArray[np.float64],
    height: int,
    width: int,
    initial_state: InitialState | None = None,
) -> NDArray[np.float64]:
    """Apply soft expert mixture blend and finalize.

    Args:
        expert_tensors: List of HxWx6 prediction tensors from different experts.
        expert_weights: Weights for each expert (will be normalized to sum to 1).
        height: Expected height.
        width: Expected width.
        initial_state: Optional initial state for static overrides during finalization.

    Returns:
        Safe, normalized HxWx6 tensor blended from experts.
    """
    if not expert_tensors:
        raise ValueError("At least one expert tensor must be provided.")
    if len(expert_tensors) != len(expert_weights):
        raise ValueError("Number of expert tensors must match number of weights.")

    # Normalize weights
    weights = np.array(expert_weights, dtype=np.float64)
    weight_sum = np.sum(weights)
    if weight_sum <= 0:
        # Fallback to uniform if weights are invalid
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weight_sum

    # Blend tensors
    blended = np.zeros_like(expert_tensors[0])
    for tensor, weight in zip(expert_tensors, weights):
        blended += weight * tensor

    # Finalize
    return finalize_tensor(blended, height, width, initial_state)
