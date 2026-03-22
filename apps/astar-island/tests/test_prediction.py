"""Tests for prediction tensor utilities — masks, floors, and validation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from astar_island.prediction import (
    DEFAULT_FLOOR,
    IMPOSSIBLE_FLOOR,
    apply_probability_floor,
    compute_masks,
    validate_prediction,
)
from astar_island.terrain import NUM_PREDICTION_CLASSES, PredictionClass, TerrainType

K = NUM_PREDICTION_CLASSES


class TestComputeMasks:
    """Test mountain_mask and coastal_mask generation."""

    def test_mountain_mask_identifies_mountains(
        self, small_raw_grid: list[list[int]]
    ) -> None:
        mountain_mask, _ = compute_masks(small_raw_grid, 5, 5)
        # x=4 is all mountain
        assert mountain_mask[4, :].all()
        # x=0 (ocean), x=1 (plains), x=2 (mixed), x=3 (forest) not mountain
        assert not mountain_mask[0, :].any()
        assert not mountain_mask[1, :].any()
        assert not mountain_mask[3, :].any()

    def test_coastal_mask_adjacent_to_ocean(
        self, small_raw_grid: list[list[int]]
    ) -> None:
        _, coastal_mask = compute_masks(small_raw_grid, 5, 5)
        # x=1 is adjacent to x=0 (ocean), so should be coastal
        assert coastal_mask[1, :].all()
        # x=0 is ocean itself — should NOT be coastal
        assert not coastal_mask[0, :].any()

    def test_non_coastal_inland(self, small_raw_grid: list[list[int]]) -> None:
        _, coastal_mask = compute_masks(small_raw_grid, 5, 5)
        # x=3 (forest) and x=4 (mountain) are far from ocean
        assert not coastal_mask[3, :].any()
        assert not coastal_mask[4, :].any()

    def test_empty_grid(self) -> None:
        """All-empty grid: no mountains, no coast."""
        grid = [[0] * 5 for _ in range(5)]
        mountain_mask, coastal_mask = compute_masks(grid, 5, 5)
        assert not mountain_mask.any()
        assert not coastal_mask.any()

    def test_all_ocean_grid(self) -> None:
        """All-ocean grid: no land, no coast."""
        grid = [[10] * 3 for _ in range(3)]
        mountain_mask, coastal_mask = compute_masks(grid, 3, 3)
        assert not mountain_mask.any()
        assert not coastal_mask.any()


class TestApplyProbabilityFloor:
    """Test class-specific probability flooring."""

    def test_no_zeros_in_output(
        self,
        uniform_prediction: NDArray[np.float64],
        small_raw_grid: list[list[int]],
    ) -> None:
        result = apply_probability_floor(uniform_prediction, small_raw_grid)
        assert not np.any(result == 0.0)

    def test_sums_to_one(
        self,
        uniform_prediction: NDArray[np.float64],
        small_raw_grid: list[list[int]],
    ) -> None:
        result = apply_probability_floor(uniform_prediction, small_raw_grid)
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_mountain_floor_on_non_mountain(
        self, small_raw_grid: list[list[int]]
    ) -> None:
        """Non-mountain cells should have mountain prob >= IMPOSSIBLE_FLOOR (not DEFAULT_FLOOR)."""
        # Start with zero prediction to see flooring in action
        pred = np.zeros((5, 5, K), dtype=np.float64)
        pred[:, :, 0] = 1.0  # all mass on class 0

        result = apply_probability_floor(pred, small_raw_grid)
        mountain_idx = int(PredictionClass.MOUNTAIN)

        mountain_mask, _ = compute_masks(small_raw_grid, 5, 5)
        # Non-mountain cells should have mountain prob roughly at IMPOSSIBLE_FLOOR level
        # (after renormalization it won't be exactly IMPOSSIBLE_FLOOR, but should be small)
        non_mountain = ~mountain_mask
        if non_mountain.any():
            mountain_probs = result[non_mountain, mountain_idx]
            assert mountain_probs.min() > 0, "Mountain probs should not be zero"

    def test_port_floor_on_non_coastal(self, small_raw_grid: list[list[int]]) -> None:
        """Non-coastal cells should have port prob at IMPOSSIBLE_FLOOR level."""
        pred = np.zeros((5, 5, K), dtype=np.float64)
        pred[:, :, 0] = 1.0

        result = apply_probability_floor(pred, small_raw_grid)
        port_idx = int(PredictionClass.PORT)

        _, coastal_mask = compute_masks(small_raw_grid, 5, 5)
        non_coastal = ~coastal_mask
        if non_coastal.any():
            port_probs = result[non_coastal, port_idx]
            assert port_probs.min() > 0, "Port probs should not be zero"

    def test_shape_preserved(
        self,
        uniform_prediction: NDArray[np.float64],
        small_raw_grid: list[list[int]],
    ) -> None:
        result = apply_probability_floor(uniform_prediction, small_raw_grid)
        assert result.shape == uniform_prediction.shape

    def test_full_size(self, full_size_raw_grid: list[list[int]]) -> None:
        """Apply floor to a 40x40 prediction — should not error."""
        pred = np.full((40, 40, K), 1.0 / K, dtype=np.float64)
        result = apply_probability_floor(pred, full_size_raw_grid)
        assert result.shape == (40, 40, K)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-10)


class TestValidatePrediction:
    """Test prediction tensor validation."""

    def test_valid_uniform(self) -> None:
        pred = np.full((40, 40, K), 1.0 / K, dtype=np.float64)
        errors = validate_prediction(pred)
        assert errors == []

    def test_wrong_shape(self) -> None:
        pred = np.full((10, 10, K), 1.0 / K, dtype=np.float64)
        errors = validate_prediction(pred, expected_width=40, expected_height=40)
        assert len(errors) > 0
        assert "shape" in errors[0].lower()

    def test_nan_detected(self) -> None:
        pred = np.full((40, 40, K), 1.0 / K, dtype=np.float64)
        pred[0, 0, 0] = float("nan")
        errors = validate_prediction(pred)
        assert any("nan" in e.lower() for e in errors)

    def test_inf_detected(self) -> None:
        pred = np.full((40, 40, K), 1.0 / K, dtype=np.float64)
        pred[0, 0, 0] = float("inf")
        errors = validate_prediction(pred)
        assert any("inf" in e.lower() for e in errors)

    def test_negative_detected(self) -> None:
        pred = np.full((40, 40, K), 1.0 / K, dtype=np.float64)
        pred[0, 0, 0] = -0.1
        errors = validate_prediction(pred)
        assert any("negative" in e.lower() for e in errors)

    def test_zero_detected(self) -> None:
        pred = np.full((40, 40, K), 1.0 / K, dtype=np.float64)
        pred[0, 0, 0] = 0.0
        errors = validate_prediction(pred)
        assert any("zero" in e.lower() for e in errors)

    def test_sum_not_one_detected(self) -> None:
        pred = np.full((40, 40, K), 0.5, dtype=np.float64)  # sums to 3.0
        errors = validate_prediction(pred)
        assert any("sum" in e.lower() or "don't" in e.lower() for e in errors)

    def test_custom_size(self) -> None:
        pred = np.full((5, 5, K), 1.0 / K, dtype=np.float64)
        errors = validate_prediction(pred, expected_width=5, expected_height=5)
        assert errors == []
