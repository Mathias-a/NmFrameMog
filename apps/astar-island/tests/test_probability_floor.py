"""Probability floor optimization tests.

Tests different floor values to help pick the optimal trade-off between
preventing infinite KL and not distorting good predictions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from astar_island.prediction import (
    PredictionTensor,
    apply_probability_floor,
    make_uniform_prediction,
)
from astar_island.scoring import competition_score, weighted_kl
from astar_island.terrain import NUM_PREDICTION_CLASSES
from numpy.typing import NDArray

C = NUM_PREDICTION_CLASSES
FLOOR_VALUES: list[float] = [0.001, 0.005, 0.01, 0.02, 0.05]


class TestFloorOnAdversarial:
    """Apply each floor value to an adversarial prediction and score it."""

    @pytest.mark.parametrize("floor", FLOOR_VALUES)
    def test_adversarial_finite(
        self,
        floor: float,
        gt_soft_40x40: NDArray[np.float64],
    ) -> None:
        """Adversarial prediction with floor must produce finite score."""
        dominant = np.argmax(gt_soft_40x40, axis=-1)
        wrong = (dominant + 1) % C
        pred: PredictionTensor = np.zeros_like(gt_soft_40x40)
        rows, cols = np.indices(gt_soft_40x40.shape[:2])
        pred[rows, cols, wrong] = 1.0

        floored = apply_probability_floor(pred, floor=floor)
        wkl = weighted_kl(gt_soft_40x40, floored)
        assert math.isfinite(wkl), f"floor={floor} -> non-finite wkl"
        score = competition_score(gt_soft_40x40, floored)
        assert 0.0 <= score <= 100.0

    @pytest.mark.parametrize("floor", FLOOR_VALUES)
    def test_adversarial_prevents_huge_kl(
        self,
        floor: float,
        gt_soft_40x40: NDArray[np.float64],
    ) -> None:
        """Floor must keep weighted KL bounded."""
        dominant = np.argmax(gt_soft_40x40, axis=-1)
        wrong = (dominant + 1) % C
        pred: PredictionTensor = np.zeros_like(gt_soft_40x40)
        rows, cols = np.indices(gt_soft_40x40.shape[:2])
        pred[rows, cols, wrong] = 1.0

        floored = apply_probability_floor(pred, floor=floor)
        wkl = weighted_kl(gt_soft_40x40, floored)
        assert wkl < 100, f"floor={floor} -> unreasonably large wkl {wkl}"


class TestFloorOnNearPerfect:
    """Floor should not significantly hurt near-perfect predictions."""

    @pytest.mark.parametrize("floor", FLOOR_VALUES)
    def test_near_perfect_not_hurt_much(
        self,
        floor: float,
        gt_soft_40x40: NDArray[np.float64],
    ) -> None:
        rng = np.random.default_rng(55)
        noisy: PredictionTensor = (
            gt_soft_40x40 * 0.95 + rng.random(gt_soft_40x40.shape) * 0.05
        )
        noisy = noisy / noisy.sum(axis=-1, keepdims=True)

        score_before = competition_score(gt_soft_40x40, noisy)
        floored = apply_probability_floor(noisy, floor=floor)
        score_after = competition_score(gt_soft_40x40, floored)

        # Floor should not drop score by more than 20 points
        drop = score_before - score_after
        assert drop < 20.0, (
            f"floor={floor} dropped score too much: "
            f"{score_before:.2f} -> {score_after:.2f}"
        )


class TestFloorComparisonTable:
    """Print a comparison table across floor values for analysis."""

    def test_print_comparison(
        self,
        gt_soft_40x40: NDArray[np.float64],
    ) -> None:
        rng = np.random.default_rng(55)

        near_perfect: PredictionTensor = (
            gt_soft_40x40 * 0.95 + rng.random(gt_soft_40x40.shape) * 0.05
        )
        near_perfect = near_perfect / near_perfect.sum(axis=-1, keepdims=True)

        dominant = np.argmax(gt_soft_40x40, axis=-1)
        wrong = (dominant + 1) % C
        adversarial: PredictionTensor = np.zeros_like(gt_soft_40x40)
        rows, cols = np.indices(gt_soft_40x40.shape[:2])
        adversarial[rows, cols, wrong] = 1.0

        uniform = make_uniform_prediction(40, 40)

        header = (
            f"{'Floor':>8} | {'Near-perf':>12} | {'Uniform':>12} | {'Adversarial':>12}"
        )
        sep = "-" * len(header)
        print(f"\n{sep}")
        print(f"{'':>8}   {'(score 0-100, higher=better)':>40}")
        print(header)
        print(sep)

        for floor in FLOOR_VALUES:
            np_f = apply_probability_floor(near_perfect, floor=floor)
            u_f = apply_probability_floor(uniform, floor=floor)
            a_f = apply_probability_floor(adversarial, floor=floor)

            s_np = competition_score(gt_soft_40x40, np_f)
            s_u = competition_score(gt_soft_40x40, u_f)
            s_a = competition_score(gt_soft_40x40, a_f)

            print(f"{floor:>8.3f} | {s_np:>12.2f} | {s_u:>12.2f} | {s_a:>12.2f}")

        print(sep)

        # Sanity check
        f_pred = apply_probability_floor(near_perfect, floor=0.01)
        score = competition_score(gt_soft_40x40, f_pred)
        assert math.isfinite(score)
