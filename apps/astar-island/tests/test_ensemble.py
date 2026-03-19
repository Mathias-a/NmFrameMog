"""Tests for ensemble GT generation."""

from __future__ import annotations

import numpy as np
from astar_island.prediction import validate_prediction
from astar_island.scoring import competition_score
from astar_island.simulator.ensemble import run_ensemble
from astar_island.terrain import NUM_PREDICTION_CLASSES


class TestEnsembleShape:
    def test_output_shape(self) -> None:
        pred = run_ensemble(seed=42, n_runs=5, width=10, height=10)
        assert pred.shape == (10, 10, NUM_PREDICTION_CLASSES)

    def test_sums_to_one(self) -> None:
        pred = run_ensemble(seed=42, n_runs=5, width=10, height=10)
        sums = pred.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_no_zeros(self) -> None:
        pred = run_ensemble(seed=42, n_runs=5, width=10, height=10)
        assert np.all(pred > 0)


class TestEnsembleValidation:
    def test_passes_validation(self) -> None:
        pred = run_ensemble(seed=42, n_runs=5, width=10, height=10)
        errors = validate_prediction(pred, expected_width=10, expected_height=10)
        assert errors == []


class TestEnsembleScoring:
    def test_self_score_near_100(self) -> None:
        """Scoring ensemble GT against itself should be near 100."""
        pred = run_ensemble(seed=42, n_runs=10, width=10, height=10)
        score = competition_score(pred, pred)
        assert score > 95.0, f"Self-score should be near 100, got {score}"

    def test_uniform_worse_than_ensemble(self) -> None:
        """Uniform prediction should score lower than ensemble."""
        gt = run_ensemble(seed=42, n_runs=10, width=10, height=10)
        uniform = np.full(
            (10, 10, NUM_PREDICTION_CLASSES),
            1.0 / NUM_PREDICTION_CLASSES,
            dtype=np.float64,
        )
        uniform_score = competition_score(gt, uniform)
        self_score = competition_score(gt, gt)
        assert uniform_score < self_score, (
            f"Uniform ({uniform_score:.2f}) should be worse "
            f"than ensemble ({self_score:.2f})"
        )
