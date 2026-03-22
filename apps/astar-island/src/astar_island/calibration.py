"""Temperature scaling and prediction calibration.

Adjusts model confidence via a single temperature parameter T:
  q_T[k] = softmax(log(q[k]) / T)

T > 1 softens predictions (reduces overconfidence).
T < 1 sharpens predictions.

Full pipeline: raw prediction → temperature scaling → probability floors → renormalize.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from astar_island.prediction import apply_probability_floor
from astar_island.scoring import weighted_kl


def _apply_temperature(
    probs: NDArray[np.float64], temperature: float
) -> NDArray[np.float64]:
    logits = np.log(np.clip(probs, 1e-15, 1.0))
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp_scaled = np.exp(shifted)
    return exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)


def find_optimal_temperature(
    predictions: NDArray[np.float64],
    ground_truths: NDArray[np.float64],
    bounds: tuple[float, float] = (0.5, 3.0),
) -> float:
    """Find temperature T minimizing weighted KL on validation data.

    Args:
        predictions: (W, H, 6) or list of (W, H, 6) raw model predictions.
        ground_truths: Matching ground truth tensors, same shape.
        bounds: Search range for T.

    Returns:
        Optimal temperature value.
    """

    def objective(t: float) -> float:
        scaled = _apply_temperature(predictions, t)
        return weighted_kl(ground_truths, scaled)

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x)


def calibrate_prediction(
    raw_probs: NDArray[np.float64],
    initial_grid: list[list[int]],
    temperature: float = 1.0,
) -> NDArray[np.float64]:
    """Full calibration pipeline: temperature scale → floor → renormalize.

    Args:
        raw_probs: (W, H, 6) model output probabilities.
        initial_grid: Raw terrain grid for class-specific floors.
        temperature: Learned temperature parameter.

    Returns:
        (W, H, 6) calibrated prediction tensor.
    """
    if temperature != 1.0:
        raw_probs = _apply_temperature(raw_probs, temperature)

    return apply_probability_floor(raw_probs, initial_grid)
