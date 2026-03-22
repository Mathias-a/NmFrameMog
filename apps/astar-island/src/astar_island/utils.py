"""Shared math utilities: entropy, KL divergence, scoring, normalization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_island.types import K

_F = np.float64


def clip_normalize(
    p: NDArray[np.float64],
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """Clip to [eps, 1] and renormalize to sum to 1 along last axis."""
    clipped: NDArray[np.float64] = np.clip(p, eps, 1.0)
    sums: NDArray[np.float64] = np.asarray(
        clipped.sum(axis=-1, keepdims=True), dtype=_F,
    )
    return np.asarray(clipped / sums, dtype=_F)


def entropy(
    p: NDArray[np.float64],
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """Shannon entropy along last axis. Shape: (...) from input (... , K)."""
    safe: NDArray[np.float64] = np.clip(p, eps, 1.0)
    return np.asarray(-np.sum(safe * np.log(safe), axis=-1), dtype=_F)


def kl_divergence(
    p_true: NDArray[np.float64],
    q_pred: NDArray[np.float64],
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """KL(p || q) per cell along last axis. Shape: (...) from input (..., K)."""
    p: NDArray[np.float64] = np.clip(p_true, eps, 1.0)
    q: NDArray[np.float64] = np.clip(q_pred, eps, 1.0)
    return np.asarray(np.sum(p * np.log(p / q), axis=-1), dtype=_F)


def weighted_kl_score(
    ground_truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
    eps: float = 1e-4,
) -> float:
    """Compute the official competition score (0-100)."""
    ent = entropy(ground_truth, eps)
    kl = kl_divergence(ground_truth, prediction, eps)

    total_weight = float(np.asarray(np.sum(ent), dtype=_F))
    if total_weight < 1e-12:
        return 100.0

    weighted_kl = float(np.asarray(np.sum(ent * kl), dtype=_F)) / total_weight
    score = 100.0 * float(np.exp(-3.0 * weighted_kl))
    return max(0.0, min(100.0, score))


def normalized_entropy(
    p: NDArray[np.float64],
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """Entropy normalized to [0, 1] by dividing by log(K). Shape: (...)."""
    return np.asarray(entropy(p, eps) / np.log(K), dtype=_F)


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax over 1D array."""
    shifted: NDArray[np.float64] = np.asarray(
        logits - np.max(logits), dtype=_F,
    )
    exp_vals: NDArray[np.float64] = np.asarray(np.exp(shifted), dtype=_F)
    return np.asarray(exp_vals / np.sum(exp_vals), dtype=_F)
