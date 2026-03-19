"""End-to-end integration tests for the Astar Island local harness."""

from __future__ import annotations

import math

import numpy as np
from astar_island.prediction import (
    PredictionTensor,
    apply_probability_floor,
    build_safe_prediction,
    make_uniform_prediction,
    validate_prediction,
)
from astar_island.scoring import competition_score
from numpy.typing import NDArray


class TestUniformPipeline:
    """GT -> uniform prediction -> floor -> validate -> score -> finite."""

    def test_uniform_pipeline(self, gt_soft_40x40: NDArray[np.float64]) -> None:
        pred = make_uniform_prediction(40, 40)
        pred = apply_probability_floor(pred, floor=0.01)

        errors = validate_prediction(pred, expected_width=40, expected_height=40)
        assert errors == []

        score = competition_score(gt_soft_40x40, pred)
        assert 0.0 <= score <= 100.0
        assert score < 100.0, "Uniform shouldn't be perfect"
        print(f"\nUniform pipeline score: {score:.2f}/100")


class TestNearPerfectPipeline:
    """GT -> near-perfect prediction -> floor -> score -> near 100."""

    def test_near_perfect_pipeline(self, gt_soft_40x40: NDArray[np.float64]) -> None:
        rng = np.random.default_rng(42)
        raw: PredictionTensor = (
            gt_soft_40x40 * 0.95 + rng.random(gt_soft_40x40.shape) * 0.05
        )

        safe = build_safe_prediction(
            raw, floor=0.01, expected_width=40, expected_height=40
        )

        errors = validate_prediction(safe, expected_width=40, expected_height=40)
        assert errors == []

        score = competition_score(gt_soft_40x40, safe)
        assert score > 50.0, f"Near-perfect should score well, got {score:.2f}"
        print(f"\nNear-perfect pipeline score: {score:.2f}/100")


class TestBuildSafePrediction:
    """The build_safe_prediction helper handles the full pipeline."""

    def test_from_random_noise(self) -> None:
        rng = np.random.default_rng(99)
        raw: PredictionTensor = rng.random((40, 40, 6))
        safe = build_safe_prediction(raw, floor=0.01)
        assert validate_prediction(safe) == []

    def test_from_negative_values(self) -> None:
        rng = np.random.default_rng(77)
        raw: PredictionTensor = rng.random((40, 40, 6)) - 0.5
        safe = build_safe_prediction(raw, floor=0.01)
        assert validate_prediction(safe) == []
        assert np.all(safe >= 0)


class TestMultipleGridSizes:
    """Integration tests across different grid sizes."""

    def test_10x10(self, gt_soft_10x10: NDArray[np.float64]) -> None:
        pred = make_uniform_prediction(10, 10)
        pred = apply_probability_floor(pred)
        score = competition_score(gt_soft_10x10, pred)
        assert 0.0 <= score <= 100.0

    def test_40x40(self, gt_soft_40x40: NDArray[np.float64]) -> None:
        pred = make_uniform_prediction(40, 40)
        pred = apply_probability_floor(pred)
        score = competition_score(gt_soft_40x40, pred)
        assert 0.0 <= score <= 100.0

    def test_100x100(self, gt_soft_100x100: NDArray[np.float64]) -> None:
        pred = make_uniform_prediction(100, 100)
        pred = apply_probability_floor(pred)
        score = competition_score(gt_soft_100x100, pred)
        assert 0.0 <= score <= 100.0


class TestSoftVsOneHotGT:
    """Verify scoring behavior differs for soft vs one-hot GT."""

    def test_onehot_gt_all_static(self, gt_onehot_40x40: NDArray[np.float64]) -> None:
        """One-hot GT has zero entropy -> all cells excluded -> score=100."""
        pred = make_uniform_prediction(40, 40)
        score = competition_score(gt_onehot_40x40, pred)
        assert math.isclose(score, 100.0, abs_tol=1e-6)

    def test_soft_gt_penalizes_uniform(
        self, gt_soft_40x40: NDArray[np.float64]
    ) -> None:
        """Soft GT has entropy -> uniform pred gets penalized."""
        pred = make_uniform_prediction(40, 40)
        score = competition_score(gt_soft_40x40, pred)
        assert score < 100.0
