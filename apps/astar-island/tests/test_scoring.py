"""Scoring correctness tests for the local Astar Island scoring engine."""

from __future__ import annotations

import math
import time

import numpy as np
from astar_island.prediction import PredictionTensor, make_uniform_prediction
from astar_island.scoring import (
    competition_score,
    entropy_per_cell,
    kl_divergence_per_cell,
    weighted_kl,
)
from astar_island.terrain import NUM_PREDICTION_CLASSES
from numpy.typing import NDArray
from scipy.special import rel_entr  # type: ignore[import-untyped]

C = NUM_PREDICTION_CLASSES


class TestPerfectPrediction:
    """Perfect prediction (pred == gt) must score 100."""

    def test_perfect_soft_gt(self, gt_soft_40x40: NDArray[np.float64]) -> None:
        wkl = weighted_kl(gt_soft_40x40, gt_soft_40x40)
        assert math.isclose(wkl, 0.0, abs_tol=1e-10)
        score = competition_score(gt_soft_40x40, gt_soft_40x40)
        assert math.isclose(score, 100.0, abs_tol=1e-6)

    def test_perfect_onehot_gt(self, gt_onehot_40x40: NDArray[np.float64]) -> None:
        """One-hot GT has zero entropy everywhere, so weighted_kl = 0."""
        wkl = weighted_kl(gt_onehot_40x40, gt_onehot_40x40)
        assert math.isclose(wkl, 0.0, abs_tol=1e-10)
        score = competition_score(gt_onehot_40x40, gt_onehot_40x40)
        assert math.isclose(score, 100.0, abs_tol=1e-6)


class TestScoreOrdering:
    """Higher is better: adversarial < uniform < near-perfect < perfect."""

    def test_ordering(
        self,
        gt_soft_40x40: NDArray[np.float64],
        near_perfect_40x40: PredictionTensor,
        uniform_40x40: PredictionTensor,
        adversarial_40x40: PredictionTensor,
    ) -> None:
        perfect = competition_score(gt_soft_40x40, gt_soft_40x40)
        near_perf = competition_score(gt_soft_40x40, near_perfect_40x40)
        uniform = competition_score(gt_soft_40x40, uniform_40x40)
        adversarial = competition_score(gt_soft_40x40, adversarial_40x40)

        assert adversarial < uniform, "adversarial < uniform"
        assert uniform < near_perf, "uniform < near-perfect"
        assert near_perf <= perfect, "near-perfect <= perfect"

        print(
            f"\nScore ordering (higher=better): "
            f"{adversarial:.2f} < {uniform:.2f} "
            f"< {near_perf:.2f} <= {perfect:.2f}"
        )


class TestCompetitionScoreBounds:
    """Score must be in [0, 100]."""

    def test_perfect_is_100(self, gt_soft_40x40: NDArray[np.float64]) -> None:
        assert competition_score(gt_soft_40x40, gt_soft_40x40) == 100.0

    def test_bad_prediction_is_low(
        self,
        gt_soft_40x40: NDArray[np.float64],
        adversarial_40x40: PredictionTensor,
    ) -> None:
        score = competition_score(gt_soft_40x40, adversarial_40x40)
        assert 0.0 <= score <= 100.0

    def test_exp_decay_formula(self) -> None:
        """Verify the formula: 100 * exp(-3 * wkl)."""
        gt = np.zeros((2, 2, C), dtype=np.float64)
        gt[:, :] = 1.0 / C  # uniform GT (all cells have entropy)
        pred = make_uniform_prediction(2, 2)
        # Perfect match → wkl=0 → score=100
        assert math.isclose(competition_score(gt, pred), 100.0, abs_tol=1e-6)


class TestCrossValidateScipy:
    """Cross-validate our KL against scipy.special.rel_entr."""

    @staticmethod
    def _scipy_kl_per_cell(
        gt: NDArray[np.float64], pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        kl: NDArray[np.float64] = np.array(rel_entr(gt, pred)).sum(axis=-1)
        return kl

    def test_hand_calculated_uniform(self) -> None:
        """KL(one-hot || uniform) should be log(C)."""
        gt = np.zeros((1, 1, C), dtype=np.float64)
        gt[0, 0, 2] = 1.0
        pred = np.full((1, 1, C), 1.0 / C, dtype=np.float64)

        our_kl = kl_divergence_per_cell(gt, pred)
        expected = math.log(C)
        assert math.isclose(float(our_kl[0, 0]), expected, rel_tol=1e-10)

        scipy_kl = self._scipy_kl_per_cell(gt, pred)
        assert math.isclose(
            float(our_kl[0, 0]),
            float(scipy_kl[0, 0]),
            rel_tol=1e-10,
        )

    def test_hand_calculated_biased(self) -> None:
        """KL with known soft GT: gt=[0.5, 0.5, 0, ...], pred biased."""
        gt = np.zeros((1, 1, C), dtype=np.float64)
        gt[0, 0, 0] = 0.5
        gt[0, 0, 1] = 0.5
        pred = np.zeros((1, 1, C), dtype=np.float64)
        pred[0, 0, 0] = 0.25
        pred[0, 0, 1] = 0.75
        pred[0, 0, 2:] = 0.001
        pred = pred / pred.sum(axis=-1, keepdims=True)

        our_kl = float(kl_divergence_per_cell(gt, pred)[0, 0])
        scipy_kl = float(self._scipy_kl_per_cell(gt, pred)[0, 0])
        # Hand-calculated: 0.5*log(0.5/p0) + 0.5*log(0.5/p1)
        p0 = float(pred[0, 0, 0])
        p1 = float(pred[0, 0, 1])
        hand = 0.5 * math.log(0.5 / p0) + 0.5 * math.log(0.5 / p1)
        assert math.isclose(our_kl, hand, rel_tol=1e-10)
        assert math.isclose(our_kl, scipy_kl, rel_tol=1e-10)

    def test_random_soft_tensor(self) -> None:
        """Compare on a random 10x10 soft tensor (competition-like)."""
        rng = np.random.default_rng(77)
        gt: NDArray[np.float64] = rng.dirichlet(np.ones(C), size=(10, 10))
        pred: NDArray[np.float64] = rng.dirichlet(np.ones(C), size=(10, 10))
        our_kl = kl_divergence_per_cell(gt, pred)
        scipy_kl = self._scipy_kl_per_cell(gt, pred)
        np.testing.assert_allclose(our_kl, scipy_kl, rtol=1e-10)

    def test_one_hot_gt_vs_uniform(self, gt_onehot_40x40: NDArray[np.float64]) -> None:
        """One-hot GT vs uniform pred."""
        pred = make_uniform_prediction(40, 40)
        our_kl = kl_divergence_per_cell(gt_onehot_40x40, pred)
        scipy_kl = self._scipy_kl_per_cell(gt_onehot_40x40, pred)
        np.testing.assert_allclose(our_kl, scipy_kl, rtol=1e-10)


class TestStaticCellsExcluded:
    """Static cells (entropy=0) must not affect the score."""

    def test_all_static_scores_100(self) -> None:
        """All one-hot GT → all entropy=0 → score=100 regardless."""
        gt = np.zeros((10, 10, C), dtype=np.float64)
        gt[:, :, 3] = 1.0
        pred = make_uniform_prediction(10, 10)
        # Even though pred != gt, entropy=0 everywhere so wkl=0
        score = competition_score(gt, pred)
        assert math.isclose(score, 100.0, abs_tol=1e-6)

    def test_entropy_zero_for_onehot(self) -> None:
        gt = np.zeros((10, 10, C), dtype=np.float64)
        gt[:, :, 3] = 1.0
        ent = entropy_per_cell(gt)
        assert np.allclose(ent, 0.0)


class TestPerformance:
    """Scoring on the competition grid size must be fast."""

    def test_40x40_under_100ms(
        self,
        gt_soft_40x40: NDArray[np.float64],
        uniform_40x40: PredictionTensor,
    ) -> None:
        competition_score(gt_soft_40x40, uniform_40x40)

        start = time.perf_counter()
        for _ in range(100):
            competition_score(gt_soft_40x40, uniform_40x40)
        elapsed_ms = (time.perf_counter() - start) / 100 * 1000

        print(f"\n40x40 scoring: {elapsed_ms:.2f}ms per call")
        assert elapsed_ms < 100, f"Too slow: {elapsed_ms:.2f}ms"
