"""Prediction format validation tests."""

from __future__ import annotations

import numpy as np
import pytest
from astar_island.prediction import (
    PredictionTensor,
    validate_prediction,
)
from astar_island.terrain import NUM_PREDICTION_CLASSES

C = NUM_PREDICTION_CLASSES


class TestCorrectShapeAccepted:
    """Valid tensors must pass validation."""

    @pytest.mark.parametrize("w,h", [(10, 10), (40, 40), (100, 100)])
    def test_uniform_various_sizes(self, w: int, h: int) -> None:
        pred: PredictionTensor = np.full((w, h, C), 1.0 / C)
        errors = validate_prediction(pred, expected_width=w, expected_height=h)
        assert errors == [], f"Valid tensor rejected: {[e.message for e in errors]}"


class TestWrongShapeRejected:
    """Tensors with wrong shapes must be rejected."""

    def test_missing_dimension(self) -> None:
        pred: PredictionTensor = np.full((40, 40), 1.0)  # 2D instead of 3D
        errors = validate_prediction(pred)
        assert any(e.field == "shape" for e in errors)

    def test_wrong_class_count(self) -> None:
        pred: PredictionTensor = np.full((40, 40, 5), 1.0 / 5)  # 5 classes instead of 6
        errors = validate_prediction(pred)
        assert any(e.field == "shape" for e in errors)

    def test_wrong_grid_size(self) -> None:
        pred: PredictionTensor = np.full((30, 30, C), 1.0 / C)
        errors = validate_prediction(pred, expected_width=40, expected_height=40)
        assert any(e.field == "shape" for e in errors)

    def test_4d_tensor(self) -> None:
        pred: PredictionTensor = np.full((1, 40, 40, C), 1.0 / C)
        errors = validate_prediction(pred)
        assert any(e.field == "shape" for e in errors)


class TestSumToOne:
    """Probabilities must sum to 1 per cell."""

    def test_sums_not_one_rejected(self) -> None:
        pred: PredictionTensor = np.full((40, 40, C), 0.5)  # Sum = 3.0
        errors = validate_prediction(pred)
        assert any(e.field == "sum" for e in errors)

    def test_near_one_accepted(self) -> None:
        pred: PredictionTensor = np.full((10, 10, C), 1.0 / C)
        # Tiny floating point noise — should still pass
        pred[0, 0, 0] += 1e-8
        errors = validate_prediction(pred, expected_width=10, expected_height=10)
        sum_errors = [e for e in errors if e.field == "sum"]
        assert sum_errors == [], "Tiny FP noise should be tolerated"


class TestNoNegatives:
    """Negative probabilities must be rejected."""

    def test_negative_rejected(self) -> None:
        pred: PredictionTensor = np.full((40, 40, C), 1.0 / C)
        pred[5, 5, 0] = -0.1
        errors = validate_prediction(pred)
        assert any(e.field == "negative" for e in errors)


class TestNoNaNInf:
    """NaN and Inf values must be rejected."""

    def test_nan_rejected(self) -> None:
        pred: PredictionTensor = np.full((40, 40, C), 1.0 / C)
        pred[0, 0, 0] = float("nan")
        errors = validate_prediction(pred)
        assert any(e.field == "nan" for e in errors)

    def test_inf_rejected(self) -> None:
        pred: PredictionTensor = np.full((40, 40, C), 1.0 / C)
        pred[0, 0, 0] = float("inf")
        errors = validate_prediction(pred)
        assert any(e.field == "inf" for e in errors)


class TestZeroProbabilities:
    """Zero probabilities must be caught with a clear error."""

    def test_zero_rejected(self) -> None:
        pred: PredictionTensor = np.zeros((40, 40, C))
        pred[:, :, 0] = 1.0  # All mass on class 0 → zeros on classes 1-5
        errors = validate_prediction(pred)
        zero_errors = [e for e in errors if e.field == "zero"]
        assert len(zero_errors) == 1
        assert "CRITICAL" in zero_errors[0].message
        assert "infinite" in zero_errors[0].message.lower()
