"""Shared fixtures for Astar Island test suite."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from astar_island.prediction import (
    DEFAULT_FLOOR,
    PredictionTensor,
    apply_probability_floor,
    make_uniform_prediction,
)
from astar_island.terrain import NUM_PREDICTION_CLASSES
from numpy.typing import NDArray

C = NUM_PREDICTION_CLASSES


def _make_one_hot_ground_truth(
    width: int, height: int, seed: int = 42
) -> NDArray[np.float64]:
    """One-hot ground truth: exactly one class per cell.

    Represents static cells (entropy=0). The competition excludes
    these from scoring, but they're useful for format validation.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, C, size=(width, height))
    gt: NDArray[np.float64] = np.zeros((width, height, C), dtype=np.float64)
    rows, cols = np.indices((width, height))
    gt[rows, cols, labels] = 1.0
    return gt


def _make_soft_ground_truth(
    width: int, height: int, seed: int = 42
) -> NDArray[np.float64]:
    """Soft ground truth: Dirichlet-sampled per cell.

    The real competition GT comes from running the simulation
    hundreds of times, producing a probability distribution per cell.
    This fixture approximates that with random soft distributions.
    """
    rng = np.random.default_rng(seed)
    # Use concentrated Dirichlet (alpha=0.5) to get realistic
    # distributions with a dominant class + small tails
    gt: NDArray[np.float64] = rng.dirichlet(np.full(C, 0.5), size=(width, height))
    return gt


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def uniform_10x10() -> PredictionTensor:
    return make_uniform_prediction(10, 10)


@pytest.fixture()
def uniform_40x40() -> PredictionTensor:
    return make_uniform_prediction(40, 40)


@pytest.fixture()
def uniform_100x100() -> PredictionTensor:
    return make_uniform_prediction(100, 100)


@pytest.fixture()
def gt_onehot_40x40() -> NDArray[np.float64]:
    """One-hot GT (static cells, entropy=0)."""
    return _make_one_hot_ground_truth(40, 40, seed=40)


@pytest.fixture()
def gt_soft_10x10() -> NDArray[np.float64]:
    """Soft GT, 10x10."""
    return _make_soft_ground_truth(10, 10, seed=10)


@pytest.fixture()
def gt_soft_40x40() -> NDArray[np.float64]:
    """Soft GT, 40x40 (competition-like)."""
    return _make_soft_ground_truth(40, 40, seed=40)


@pytest.fixture()
def gt_soft_100x100() -> NDArray[np.float64]:
    """Soft GT, 100x100."""
    return _make_soft_ground_truth(100, 100, seed=100)


@pytest.fixture()
def near_perfect_40x40(
    gt_soft_40x40: NDArray[np.float64],
) -> PredictionTensor:
    """GT + small noise, floored. Should score near 100."""
    rng = np.random.default_rng(123)
    noisy: PredictionTensor = (
        gt_soft_40x40 * 0.9 + rng.random(gt_soft_40x40.shape) * 0.1
    )
    return apply_probability_floor(noisy, floor=DEFAULT_FLOOR)


@pytest.fixture()
def adversarial_40x40(
    gt_soft_40x40: NDArray[np.float64],
) -> PredictionTensor:
    """All mass on the wrong class. Floored to stay finite."""
    width, height = gt_soft_40x40.shape[:2]
    dominant = np.argmax(gt_soft_40x40, axis=-1)
    wrong = (dominant + 1) % C
    pred: PredictionTensor = np.zeros_like(gt_soft_40x40)
    rows, cols = np.indices((width, height))
    pred[rows, cols, wrong] = 1.0
    return apply_probability_floor(pred, floor=DEFAULT_FLOOR)


@pytest.fixture()
def ground_truth_factory() -> Callable[[int, int, int], NDArray[np.float64]]:
    """Factory for generating soft ground truths of any size."""
    return _make_soft_ground_truth
