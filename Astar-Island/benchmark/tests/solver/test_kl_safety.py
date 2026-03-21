from __future__ import annotations

import numpy as np

from astar_twin.scoring import compute_score


def test_no_warnings_zero_gt() -> None:
    ground_truth = np.array(
        [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    prediction = np.array(
        [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )

    score = compute_score(ground_truth, prediction)

    assert 0.0 <= score <= 100.0


def test_no_warnings_near_zero_pred() -> None:
    ground_truth = np.array(
        [[[0.4, 0.3, 0.2, 0.1, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]]],
        dtype=np.float64,
    )
    prediction = np.array(
        [[[1e-300, 0.999999999999, 0.0, 1e-320, 0.0, 0.0], [1e-200, 0.2, 0.3, 0.1, 0.2, 0.2]]],
        dtype=np.float64,
    )

    score = compute_score(ground_truth, prediction)

    assert 0.0 <= score <= 100.0


def test_perfect_prediction() -> None:
    perfect_gt = np.array(
        [[[0.7, 0.1, 0.1, 0.05, 0.03, 0.02], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    perfect_pred = perfect_gt.copy()

    assert compute_score(perfect_gt, perfect_pred) == 100.0


def test_uniform_vs_deterministic() -> None:
    ground_truth = np.array([[[0.7, 0.3, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float64)
    prediction = np.full((1, 1, 6), 1.0 / 6.0, dtype=np.float64)

    score = compute_score(ground_truth, prediction)

    assert 0.0 < score < 100.0
