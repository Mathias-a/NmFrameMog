"""Prediction tensor utilities for Astar Island.

Handles validation, probability flooring, and safe tensor construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astar_island.terrain import NUM_PREDICTION_CLASSES

PredictionTensor = NDArray[np.float64]

DEFAULT_FLOOR: float = 0.01
DEFAULT_WIDTH: int = 40
DEFAULT_HEIGHT: int = 40


@dataclass(frozen=True, slots=True)
class ValidationError:
    """A specific validation failure."""

    field: str
    message: str


def validate_prediction(
    pred: PredictionTensor,
    expected_width: int = DEFAULT_WIDTH,
    expected_height: int = DEFAULT_HEIGHT,
    atol: float = 1e-6,
) -> list[ValidationError]:
    """Validate a prediction tensor. Returns empty list if valid."""
    errors: list[ValidationError] = []

    expected_shape = (
        expected_width,
        expected_height,
        NUM_PREDICTION_CLASSES,
    )
    if pred.ndim != 3:
        errors.append(
            ValidationError(
                "shape",
                f"Expected 3 dimensions, got {pred.ndim}",
            )
        )
        return errors

    if pred.shape != expected_shape:
        errors.append(
            ValidationError(
                "shape",
                f"Expected {expected_shape}, got {pred.shape}",
            )
        )
        return errors

    if bool(np.any(np.isnan(pred))):
        errors.append(ValidationError("nan", "Prediction contains NaN values"))
    if bool(np.any(np.isinf(pred))):
        errors.append(ValidationError("inf", "Prediction contains infinite values"))

    if bool(np.any(pred < 0)):
        errors.append(
            ValidationError(
                "negative",
                "Prediction contains negative probabilities",
            )
        )

    zero_mask: PredictionTensor = pred == 0.0
    if bool(zero_mask.any()):
        errors.append(
            ValidationError(
                "zero",
                "CRITICAL: Zero probabilities — KL divergence infinite",
            )
        )

    sums: PredictionTensor = pred.sum(axis=-1)
    if not bool(np.allclose(sums, 1.0, atol=atol)):
        deviation: PredictionTensor = np.abs(sums - 1.0)
        max_dev = float(np.max(deviation))
        errors.append(
            ValidationError(
                "sum",
                f"Probabilities must sum to 1 per cell (max deviation: {max_dev:.2e})",
            )
        )

    return errors


def apply_probability_floor(
    pred: PredictionTensor,
    floor: float = DEFAULT_FLOOR,
) -> PredictionTensor:
    """Clamp minimum probability to `floor` and renormalize.

    Critical for avoiding KL divergence -> infinity when ground truth
    has mass on a class with zero predicted probability.

    Args:
        pred: Raw prediction tensor of shape (W, H, C).
        floor: Minimum probability per class. C * floor must be < 1.

    Returns:
        Safe prediction tensor with all values >= floor, summing to 1.
    """
    num_classes: int = int(pred.shape[-1])
    if num_classes * floor >= 1.0:
        msg = f"floor={floor} too large: {num_classes} classes * {floor} >= 1.0"
        raise ValueError(msg)

    result: PredictionTensor = np.maximum(pred, floor)
    denom: PredictionTensor = result.sum(axis=-1, keepdims=True)
    result = result / denom
    return result


def build_safe_prediction(
    raw: PredictionTensor,
    floor: float = DEFAULT_FLOOR,
    expected_width: int = DEFAULT_WIDTH,
    expected_height: int = DEFAULT_HEIGHT,
) -> PredictionTensor:
    """Convert raw model output into a safe, validated prediction tensor.

    Pipeline: clip negatives -> apply floor -> renormalize -> validate.

    Raises:
        ValueError: If the tensor has wrong shape or contains NaN/Inf.
    """
    expected_shape = (
        expected_width,
        expected_height,
        NUM_PREDICTION_CLASSES,
    )
    if raw.shape != expected_shape:
        msg = f"Expected shape {expected_shape}, got {raw.shape}"
        raise ValueError(msg)

    if bool(np.any(np.isnan(raw))) or bool(np.any(np.isinf(raw))):
        msg = "Raw prediction contains NaN or Inf values"
        raise ValueError(msg)

    clipped: PredictionTensor = np.maximum(raw, 0.0)
    safe = apply_probability_floor(clipped, floor=floor)

    errors = validate_prediction(safe, expected_width, expected_height)
    if errors:
        details = "; ".join(e.message for e in errors)
        msg = f"Post-processing validation failed: {details}"
        raise ValueError(msg)

    return safe


def make_uniform_prediction(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> PredictionTensor:
    """Create a uniform baseline prediction."""
    pred: PredictionTensor = np.full(
        (width, height, NUM_PREDICTION_CLASSES),
        1.0 / NUM_PREDICTION_CLASSES,
        dtype=np.float64,
    )
    return pred
