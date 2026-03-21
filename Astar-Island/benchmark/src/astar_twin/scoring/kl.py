from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def compute_score(ground_truth: NDArray[np.float64], prediction: NDArray[np.float64]) -> float:
    safe_gt = np.where(ground_truth > 0, ground_truth, 1.0)
    entropy = -np.sum(np.where(ground_truth > 0, ground_truth * np.log(safe_gt), 0.0), axis=2)
    mask = entropy >= 1e-10
    if not np.any(mask):
        return 100.0

    kl = np.sum(
        np.where(ground_truth > 0, ground_truth * np.log(safe_gt / prediction), 0.0), axis=2
    )
    weighted_kl = float(np.sum(entropy[mask] * kl[mask]) / np.sum(entropy[mask]))
    score = 100.0 * math.exp(-3.0 * weighted_kl)
    return max(0.0, min(100.0, score))
