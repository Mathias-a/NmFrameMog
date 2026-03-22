"""Tests for scoring functions — KL divergence, entropy, competition score."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from astar_island.scoring import (
    competition_score,
    entropy_per_cell,
    kl_divergence_per_cell,
    weighted_kl,
)

K = 6


class TestKLDivergence:
    """KL(gt || pred) per cell."""

    def test_perfect_prediction(self) -> None:
        """KL = 0 when prediction == ground truth."""
        gt = np.full((5, 5, K), 1.0 / K, dtype=np.float64)
        pred = gt.copy()
        kl = kl_divergence_per_cell(gt, pred)
        np.testing.assert_allclose(kl, 0.0, atol=1e-12)

    def test_kl_nonnegative(self) -> None:
        """KL divergence is always >= 0."""
        rng = np.random.default_rng(42)
        gt = rng.dirichlet(np.ones(K), size=(5, 5))
        pred = rng.dirichlet(np.ones(K), size=(5, 5))
        kl = kl_divergence_per_cell(gt, pred)
        assert np.all(kl >= -1e-12)  # allow tiny float errors

    def test_one_hot_gt_clipped_pred(self) -> None:
        """One-hot GT with near-zero pred should produce finite KL."""
        gt = np.zeros((2, 2, K), dtype=np.float64)
        gt[:, :, 0] = 1.0
        pred = np.full((2, 2, K), 1e-10, dtype=np.float64)
        pred[:, :, 0] = 1.0 - (K - 1) * 1e-10
        kl = kl_divergence_per_cell(gt, pred)
        assert np.all(np.isfinite(kl))


class TestEntropyPerCell:
    def test_one_hot_zero_entropy(self) -> None:
        """Deterministic cells should have zero entropy."""
        gt = np.zeros((3, 3, K), dtype=np.float64)
        gt[:, :, 0] = 1.0
        ent = entropy_per_cell(gt)
        np.testing.assert_allclose(ent, 0.0, atol=1e-12)

    def test_uniform_max_entropy(self) -> None:
        """Uniform distribution should have maximum entropy."""
        gt = np.full((3, 3, K), 1.0 / K, dtype=np.float64)
        ent = entropy_per_cell(gt)
        expected = -K * (1.0 / K) * math.log(1.0 / K)
        np.testing.assert_allclose(ent, expected, atol=1e-12)

    def test_entropy_nonnegative(self) -> None:
        rng = np.random.default_rng(99)
        gt = rng.dirichlet(np.ones(K), size=(4, 4))
        ent = entropy_per_cell(gt)
        assert np.all(ent >= -1e-12)


class TestWeightedKL:
    def test_perfect_prediction_zero(self) -> None:
        gt = np.full((5, 5, K), 1.0 / K, dtype=np.float64)
        pred = gt.copy()
        assert weighted_kl(gt, pred) == pytest.approx(0.0, abs=1e-12)

    def test_all_deterministic_zero(self) -> None:
        """When all cells are one-hot (zero entropy), weighted KL = 0."""
        gt = np.zeros((3, 3, K), dtype=np.float64)
        gt[:, :, 0] = 1.0
        pred = np.full((3, 3, K), 1.0 / K, dtype=np.float64)
        # All entropy weights are 0, so weighted KL = 0
        assert weighted_kl(gt, pred) == pytest.approx(0.0, abs=1e-12)


class TestCompetitionScore:
    def test_perfect_score_100(self) -> None:
        """Perfect prediction → score 100."""
        gt = np.full((5, 5, K), 1.0 / K, dtype=np.float64)
        pred = gt.copy()
        assert competition_score(gt, pred) == pytest.approx(100.0, abs=1e-6)

    def test_score_in_range(self) -> None:
        """Score should always be in [0, 100]."""
        rng = np.random.default_rng(7)
        gt = rng.dirichlet(np.ones(K), size=(5, 5))
        pred = rng.dirichlet(np.ones(K), size=(5, 5))
        score = competition_score(gt, pred)
        assert 0.0 <= score <= 100.0

    def test_worse_prediction_lower_score(self) -> None:
        """A prediction further from GT should score lower."""
        rng = np.random.default_rng(123)
        gt = rng.dirichlet(np.ones(K) * 2, size=(5, 5))

        # Good prediction: close to GT
        good_pred = gt * 0.9 + 0.1 / K
        good_pred /= good_pred.sum(axis=-1, keepdims=True)

        # Bad prediction: far from GT (uniform)
        bad_pred = np.full_like(gt, 1.0 / K)

        good_score = competition_score(gt, good_pred)
        bad_score = competition_score(gt, bad_pred)
        assert good_score > bad_score

    def test_formula_matches(self) -> None:
        """Verify score = 100 * exp(-3 * wkl)."""
        rng = np.random.default_rng(42)
        gt = rng.dirichlet(np.ones(K), size=(5, 5))
        pred = rng.dirichlet(np.ones(K), size=(5, 5))
        wkl = weighted_kl(gt, pred)
        expected = max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))
        score = competition_score(gt, pred)
        assert score == pytest.approx(expected, abs=1e-10)
