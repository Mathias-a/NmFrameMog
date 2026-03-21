from __future__ import annotations

import math

import numpy as np

from astar_twin.scoring import compute_score, safe_prediction


def test_perfect_prediction_scores_exactly_100() -> None:
    tensor = safe_prediction(np.array([[[0.7, 0.1, 0.1, 0.05, 0.03, 0.02]]], dtype=np.float64))
    assert compute_score(tensor, tensor) == 100.0


def test_uniform_prediction_scores_between_zero_and_hundred() -> None:
    truth = safe_prediction(np.array([[[0.8, 0.1, 0.05, 0.03, 0.01, 0.01]]], dtype=np.float64))
    prediction = np.full((1, 1, 6), 1.0 / 6.0, dtype=np.float64)
    score = compute_score(truth, prediction)
    assert 0.0 < score < 100.0


def test_safe_prediction_output_never_has_zeros() -> None:
    result = safe_prediction(np.zeros((2, 2, 6), dtype=np.float64))
    assert np.all(result > 0.0)


def test_score_formula_matches_manual_calculation() -> None:
    truth = safe_prediction(np.array([[[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]]], dtype=np.float64))
    prediction = safe_prediction(np.array([[[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]]], dtype=np.float64))
    entropy = -float(np.sum(truth[0, 0] * np.log(truth[0, 0])))
    kl = float(np.sum(truth[0, 0] * np.log(truth[0, 0] / prediction[0, 0])))
    weighted_kl = entropy * kl / entropy
    expected = 100.0 * math.exp(-3.0 * weighted_kl)
    assert np.isclose(compute_score(truth, prediction), expected)
