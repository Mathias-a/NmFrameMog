"""Tests for temperature calibration."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from astar_island.calibration import _apply_temperature, calibrate_prediction
from astar_island.terrain import NUM_PREDICTION_CLASSES

K = NUM_PREDICTION_CLASSES


class TestApplyTemperature:
    def test_temperature_one_is_identity(self) -> None:
        """T=1 should not change the distribution."""
        probs = np.array([[[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]]], dtype=np.float64)
        result = _apply_temperature(probs, 1.0)
        np.testing.assert_allclose(result, probs, atol=1e-10)

    def test_high_temperature_softens(self) -> None:
        """T>1 should make the distribution more uniform (higher entropy)."""
        probs = np.array([[[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]]], dtype=np.float64)
        softened = _apply_temperature(probs, 2.0)
        # Max prob should decrease, min prob should increase
        assert softened[0, 0, 0] < probs[0, 0, 0]
        assert softened[0, 0, 1] > probs[0, 0, 1]

    def test_low_temperature_sharpens(self) -> None:
        """T<1 should make the distribution more peaked."""
        probs = np.array([[[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]]], dtype=np.float64)
        sharpened = _apply_temperature(probs, 0.5)
        # Max prob should increase
        assert sharpened[0, 0, 0] > probs[0, 0, 0]

    def test_output_sums_to_one(self) -> None:
        """Output should be valid probability distribution."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(K), size=(5, 5))
        result = _apply_temperature(probs, 1.5)
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_no_nans(self) -> None:
        """Should not produce NaN even with very peaked inputs."""
        probs = np.zeros((1, 1, K), dtype=np.float64)
        probs[0, 0, 0] = 0.9999
        probs[0, 0, 1:] = 0.0001 / (K - 1)
        result = _apply_temperature(probs, 2.0)
        assert np.all(np.isfinite(result))


class TestCalibrateRediction:
    def test_output_shape(self, small_raw_grid: list[list[int]]) -> None:
        """Output shape should match input shape."""
        probs = np.full((5, 5, K), 1.0 / K, dtype=np.float64)
        result = calibrate_prediction(probs, small_raw_grid, temperature=1.0)
        assert result.shape == (5, 5, K)

    def test_output_valid(self, small_raw_grid: list[list[int]]) -> None:
        """Calibrated prediction should have all positive values summing to 1."""
        probs = np.full((5, 5, K), 1.0 / K, dtype=np.float64)
        result = calibrate_prediction(probs, small_raw_grid, temperature=1.12)
        assert np.all(result > 0)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-10)

    def test_temperature_one_only_floors(self, small_raw_grid: list[list[int]]) -> None:
        """T=1 should only apply probability floors, not temperature scaling."""
        from astar_island.prediction import apply_probability_floor

        probs = np.full((5, 5, K), 1.0 / K, dtype=np.float64)
        calibrated = calibrate_prediction(probs, small_raw_grid, temperature=1.0)
        floored = apply_probability_floor(probs, small_raw_grid)
        np.testing.assert_allclose(calibrated, floored, atol=1e-10)
