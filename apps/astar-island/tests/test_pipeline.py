"""End-to-end pipeline test: predict_grid → floor → validate → tolist().

Tests that the full submission pipeline produces server-acceptable payloads.
Requires trained model artifact at the default path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from astar_island.prediction import apply_probability_floor, validate_prediction
from astar_island.solver import DEFAULT_ARTIFACT, predict_grid
from astar_island.terrain import NUM_PREDICTION_CLASSES

K = NUM_PREDICTION_CLASSES

# Skip all tests in this module if the model artifact isn't available
pytestmark = pytest.mark.skipif(
    not DEFAULT_ARTIFACT.exists(),
    reason=f"Model artifact not found at {DEFAULT_ARTIFACT}",
)


class TestEndToEndPipeline:
    """Integration tests for the full prediction → submission pipeline."""

    def test_predict_grid_returns_valid_tensor(
        self, full_size_raw_grid: list[list[int]]
    ) -> None:
        """predict_grid should return a valid (40,40,6) probability tensor."""
        pred = predict_grid(full_size_raw_grid)
        assert pred.shape == (40, 40, K)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)
        np.testing.assert_allclose(pred.sum(axis=-1), 1.0, atol=1e-6)

    def test_floored_prediction_valid(
        self, full_size_raw_grid: list[list[int]]
    ) -> None:
        """After apply_probability_floor, should pass validate_prediction."""
        pred = predict_grid(full_size_raw_grid)
        safe = apply_probability_floor(pred, full_size_raw_grid)
        errors = validate_prediction(safe)
        assert errors == [], f"Validation errors: {errors}"

    def test_tolist_payload_format(self, full_size_raw_grid: list[list[int]]) -> None:
        """pred.tolist() should produce valid nested list for API submission."""
        pred = predict_grid(full_size_raw_grid)
        safe = apply_probability_floor(pred, full_size_raw_grid)
        payload = safe.tolist()

        # Shape checks
        assert isinstance(payload, list)
        assert len(payload) == 40  # W
        assert len(payload[0]) == 40  # H
        assert len(payload[0][0]) == K  # 6 classes

        # Value checks on every cell
        for x in range(40):
            for y in range(40):
                cell = payload[x][y]
                assert len(cell) == K
                assert all(isinstance(v, float) for v in cell)
                assert all(v > 0 for v in cell), f"Zero at ({x},{y}): {cell}"
                assert abs(sum(cell) - 1.0) < 0.01, (
                    f"Sum != 1 at ({x},{y}): {sum(cell)}"
                )

    def test_payload_json_serializable(
        self, full_size_raw_grid: list[list[int]]
    ) -> None:
        """The payload should be JSON-serializable (no numpy types)."""
        pred = predict_grid(full_size_raw_grid)
        safe = apply_probability_floor(pred, full_size_raw_grid)
        payload = safe.tolist()
        # This would raise if any element is not JSON-serializable
        json_str = json.dumps({"prediction": payload})
        assert len(json_str) > 0

    def test_multiple_grids_consistent(
        self, full_size_raw_grid: list[list[int]]
    ) -> None:
        """Predicting the same grid twice should give identical results."""
        pred1 = predict_grid(full_size_raw_grid)
        pred2 = predict_grid(full_size_raw_grid)
        np.testing.assert_array_equal(pred1, pred2)
