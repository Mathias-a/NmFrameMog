from __future__ import annotations

import numpy as np

from astar_twin.scoring.safe_prediction import safe_prediction


def test_adversarial_floor() -> None:
    tensor = np.array([[[0.99, 0.002, 0.002, 0.002, 0.002, 0.002]]], dtype=np.float64)

    result = safe_prediction(tensor)

    assert result.min() >= 0.0001 - 1e-12


def test_sums_to_one() -> None:
    tensor = np.array(
        [
            [[0.2, 0.2, 0.2, 0.2, 0.1, 0.1], [3.0, 1.0, 0.0, 5.0, 2.0, 4.0]],
            [[0.99, 0.002, 0.002, 0.002, 0.002, 0.002], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    result = safe_prediction(tensor)

    assert np.allclose(result.sum(axis=2), 1.0)


def test_already_safe_unchanged() -> None:
    tensor = np.array(
        [[[0.2, 0.19, 0.18, 0.17, 0.14, 0.12], [0.15, 0.16, 0.17, 0.18, 0.16, 0.18]]],
        dtype=np.float64,
    )

    result = safe_prediction(tensor)

    assert np.allclose(result, tensor)


def test_all_zero_input() -> None:
    tensor = np.zeros((2, 3, 6), dtype=np.float64)

    result = safe_prediction(tensor)

    assert np.allclose(result, 1.0 / 6.0)
