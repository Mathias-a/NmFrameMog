"""Local scoring engine for Astar Island predictions.

Implements the competition's entropy-weighted KL divergence formula.

Competition formula:
  weighted_kl = sum(entropy(cell) * KL(gt, pred)) / sum(entropy(cell))
  score = max(0, min(100, 100 * exp(-3 * weighted_kl)))

Key details:
- Only dynamic cells (entropy > 0) contribute to the score.
- Static cells (one-hot GT, entropy=0) are excluded.
- Score 100 = perfect, 0 = terrible. Higher is better.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Tensor3D = NDArray[np.float64]
Tensor2D = NDArray[np.float64]


def kl_divergence_per_cell(gt: Tensor3D, pred: Tensor3D) -> Tensor2D:
    """KL(ground_truth || prediction) per cell.

    Args:
        gt: Ground truth tensor, shape (W, H, C).
        pred: Prediction tensor, shape (W, H, C).

    Returns:
        KL divergence per cell, shape (W, H).
    """
    safe_gt: Tensor3D = np.where(gt > 0, gt, 1.0)
    safe_pred: Tensor3D = np.clip(pred, 1e-15, None)
    log_ratio: Tensor3D = np.log(safe_gt / safe_pred)
    terms: Tensor3D = np.where(gt > 0, gt * log_ratio, 0.0)
    kl: Tensor2D = terms.sum(axis=-1)
    return kl


def entropy_per_cell(gt: Tensor3D) -> Tensor2D:
    """Shannon entropy of ground truth per cell.

    Args:
        gt: Ground truth tensor, shape (W, H, C).

    Returns:
        Entropy per cell, shape (W, H).
    """
    safe_gt: Tensor3D = np.where(gt > 0, gt, 1.0)
    terms: Tensor3D = np.where(gt > 0, -gt * np.log(safe_gt), 0.0)
    ent: Tensor2D = terms.sum(axis=-1)
    return ent


def weighted_kl(gt: Tensor3D, pred: Tensor3D) -> float:
    """Entropy-weighted mean KL divergence (the raw metric).

    Formula: sum(entropy(cell) * KL(cell)) / sum(entropy(cell))

    Returns:
        Weighted KL divergence. Lower is better. 0.0 = perfect.
    """
    kl = kl_divergence_per_cell(gt, pred)
    ent = entropy_per_cell(gt)
    total_entropy = float(np.sum(ent))
    if total_entropy <= 1e-12:
        return 0.0
    return float(np.sum(ent * kl)) / total_entropy


def competition_score(gt: Tensor3D, pred: Tensor3D) -> float:
    """Compute the official competition score (0-100, higher = better).

    Formula: max(0, min(100, 100 * exp(-3 * weighted_kl)))

    Args:
        gt: Ground truth tensor, shape (W, H, C).
        pred: Prediction tensor, shape (W, H, C).

    Returns:
        Score in [0, 100].
    """
    wkl = weighted_kl(gt, pred)
    raw = 100.0 * math.exp(-3.0 * wkl)
    return max(0.0, min(100.0, raw))
