"""Tests for tensor finalization and safety enforcement."""

from __future__ import annotations

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.solver.predict.finalize import TensorValidationError, finalize_tensor


def test_finalize_produces_valid_shape() -> None:
    raw = np.full((10, 10, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    result = finalize_tensor(raw, 10, 10)
    assert result.shape == (10, 10, NUM_CLASSES)


def test_finalize_normalizes_cells() -> None:
    raw = np.random.default_rng(42).random((10, 10, NUM_CLASSES))
    result = finalize_tensor(raw, 10, 10)
    sums = np.sum(result, axis=2)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_finalize_floors_to_minimum() -> None:
    raw = np.zeros((5, 5, NUM_CLASSES), dtype=np.float64)
    raw[:, :, 0] = 1.0  # all mass on class 0
    result = finalize_tensor(raw, 5, 5)
    # safe_prediction floors to 0.01 then renormalizes, so minimum can be
    # slightly below 0.01 after normalization — the invariant is all > 0
    assert np.all(result > 0), "All probabilities must be positive after flooring"
    sums = np.sum(result, axis=2)
    assert np.allclose(sums, 1.0, atol=1e-6), "All cells must sum to 1.0"


def test_finalize_all_values_positive() -> None:
    raw = np.zeros((3, 3, NUM_CLASSES), dtype=np.float64)
    result = finalize_tensor(raw, 3, 3)
    assert np.all(result > 0)


def test_finalize_rejects_wrong_shape() -> None:
    raw = np.zeros((10, 10, 3), dtype=np.float64)
    with pytest.raises(TensorValidationError, match="Expected shape"):
        finalize_tensor(raw, 10, 10)


def test_finalize_rejects_nan() -> None:
    raw = np.full((5, 5, NUM_CLASSES), float("nan"), dtype=np.float64)
    with pytest.raises(TensorValidationError, match="non-finite"):
        finalize_tensor(raw, 5, 5)


def test_finalize_rejects_inf() -> None:
    raw = np.full((5, 5, NUM_CLASSES), float("inf"), dtype=np.float64)
    with pytest.raises(TensorValidationError, match="non-finite"):
        finalize_tensor(raw, 5, 5)


def test_finalize_with_static_overrides() -> None:
    """Ocean cells should have high Empty probability, mountain cells high Mountain."""
    # Create a 3x3 grid: ocean at (0,0), mountain at (0,2), plains elsewhere
    grid = [
        [TerrainCode.OCEAN, TerrainCode.PLAINS, TerrainCode.MOUNTAIN],
        [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
        [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
    ]
    initial_state = InitialState(grid=grid, settlements=[])

    raw = np.full((3, 3, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    result = finalize_tensor(raw, 3, 3, initial_state=initial_state)

    # Ocean cell should have highest prob on Empty class
    assert result[0, 0, ClassIndex.EMPTY] > result[0, 0, ClassIndex.SETTLEMENT]
    assert result[0, 0, ClassIndex.EMPTY] > result[0, 0, ClassIndex.FOREST]

    # Mountain cell should have highest prob on Mountain class
    assert result[0, 2, ClassIndex.MOUNTAIN] > result[0, 2, ClassIndex.EMPTY]
    assert result[0, 2, ClassIndex.MOUNTAIN] > result[0, 2, ClassIndex.SETTLEMENT]

    # All cells still valid
    sums = np.sum(result, axis=2)
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert np.all(result > 0)
