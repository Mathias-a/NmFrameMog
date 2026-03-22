"""Tests for solver helpers — viewport planning, observation accumulation, Dirichlet blending, adaptive gate."""

from __future__ import annotations

import numpy as np
import pytest

from astar_island.solver import (
    BLEND_ENTROPY_THRESHOLD,
    VIEWPORT_SIZE,
    _accumulate_observation,
    _dirichlet_blend,
    _mean_dynamic_entropy,
    _plan_viewports,
)
from astar_island.terrain import NUM_PREDICTION_CLASSES

K = NUM_PREDICTION_CLASSES


class TestPlanViewports:
    """Test concentrated-repeat viewport tiling strategy."""

    def test_ten_queries_gives_repeats(self) -> None:
        """10 queries should produce 4 unique positions × 2-3 repeats."""
        positions = _plan_viewports(40, 40, 10)
        assert len(positions) == 10
        unique = set(positions)
        # Should be 4 unique positions (4 corners, 10 // 2 = 5, capped at 4)
        assert len(unique) == 4
        # Each position should appear at least 2 times
        from collections import Counter

        counts = Counter(positions)
        for pos, count in counts.items():
            assert count >= 2, f"Position {pos} only has {count} repeats"

    def test_concentrated_coverage(self) -> None:
        """Concentrated viewports should cover a good portion of the 40×40 grid."""
        positions = _plan_viewports(40, 40, 10)
        covered = set()
        for vx, vy in positions:
            for dx in range(VIEWPORT_SIZE):
                for dy in range(VIEWPORT_SIZE):
                    covered.add((vx + dx, vy + dy))
        # 4 corner viewports (15×15) on 40×40 → ~56% coverage (with 10×10 center gap)
        coverage_frac = len(covered) / (40 * 40)
        assert coverage_frac > 0.4, f"Coverage {coverage_frac:.2%} too low"

    def test_fewer_than_three_queries(self) -> None:
        """With 1-2 queries, should still return valid positions with repeats."""
        positions = _plan_viewports(40, 40, 2)
        assert len(positions) == 2
        for vx, vy in positions:
            assert 0 <= vx <= 40 - VIEWPORT_SIZE
            assert 0 <= vy <= 40 - VIEWPORT_SIZE

    def test_valid_bounds(self) -> None:
        """All viewport positions should be within valid grid bounds."""
        positions = _plan_viewports(40, 40, 10)
        for vx, vy in positions:
            assert 0 <= vx <= 40 - VIEWPORT_SIZE
            assert 0 <= vy <= 40 - VIEWPORT_SIZE

    def test_zero_queries(self) -> None:
        """Zero queries should return empty list."""
        positions = _plan_viewports(40, 40, 0)
        assert positions == []

    def test_many_queries_distributes_evenly(self) -> None:
        """20 queries should use all 4 candidate positions with 5 each."""
        positions = _plan_viewports(40, 40, 20)
        assert len(positions) == 20
        from collections import Counter

        counts = Counter(positions)
        assert len(counts) == 4
        assert all(c == 5 for c in counts.values())


class TestAccumulateObservation:
    """Test observation count accumulation from viewport data."""

    def test_single_viewport_counts(self) -> None:
        """One observation should produce exactly 1 count per cell in the viewport."""
        counts = np.zeros((40, 40, K), dtype=np.int32)
        # 3x3 viewport at (0,0) with all-empty terrain (code 0 → class 0)
        grid_rows = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        _accumulate_observation(counts, grid_rows, 0, 0)

        # Viewport cells should have count=1 for class 0
        for y in range(3):
            for x in range(3):
                assert counts[y, x, 0] == 1
                assert counts[y, x, 1:].sum() == 0

    def test_offset_placement(self) -> None:
        """Viewport at (5,10) should accumulate at the correct grid offset."""
        counts = np.zeros((40, 40, K), dtype=np.int32)
        grid_rows = [[1]]  # 1x1 viewport: settlement (code 1 → class 1)
        _accumulate_observation(counts, grid_rows, 5, 10)

        assert counts[10, 5, 1] == 1  # (vy=10, vx=5)
        assert counts.sum() == 1

    def test_multiple_observations_accumulate(self) -> None:
        """Multiple observations at the same position should sum counts."""
        counts = np.zeros((5, 5, K), dtype=np.int32)
        grid_rows = [[0]]  # code 0 → class 0
        _accumulate_observation(counts, grid_rows, 0, 0)
        _accumulate_observation(counts, grid_rows, 0, 0)
        _accumulate_observation(counts, grid_rows, 0, 0)

        assert counts[0, 0, 0] == 3

    def test_ocean_maps_to_class_zero(self) -> None:
        """Ocean (code 10) should map to class 0."""
        counts = np.zeros((20, 20, K), dtype=np.int32)
        grid_rows = [[10]]
        _accumulate_observation(counts, grid_rows, 0, 0)
        assert counts[0, 0, 0] == 1

    def test_all_terrain_codes(self) -> None:
        """Test all valid terrain codes map correctly."""
        counts = np.zeros((1, 8, K), dtype=np.int32)
        grid_rows = [[0, 1, 2, 3, 4, 5, 10, 11]]
        _accumulate_observation(counts, grid_rows, 0, 0)

        # code 0 → class 0, code 1 → class 1, ..., code 5 → class 5
        assert counts[0, 0, 0] == 1  # Empty
        assert counts[0, 1, 1] == 1  # Settlement
        assert counts[0, 2, 2] == 1  # Port
        assert counts[0, 3, 3] == 1  # Ruin
        assert counts[0, 4, 4] == 1  # Forest
        assert counts[0, 5, 5] == 1  # Mountain
        assert counts[0, 6, 0] == 1  # Ocean → class 0
        assert counts[0, 7, 0] == 1  # Plains → class 0


class TestDirichletBlend:
    """Test Dirichlet posterior blending."""

    def test_zero_counts_returns_prior(self) -> None:
        """With no observations, posterior equals prior."""
        ml_pred = np.full((5, 5, K), 1.0 / K, dtype=np.float64)
        counts = np.zeros((5, 5, K), dtype=np.int32)
        result = _dirichlet_blend(ml_pred, counts, ess=5.0)
        np.testing.assert_allclose(result, ml_pred, atol=1e-10)

    def test_high_counts_dominate(self) -> None:
        """With many observations, posterior should be close to observed frequencies."""
        ml_pred = np.full((1, 1, K), 1.0 / K, dtype=np.float64)
        counts = np.zeros((1, 1, K), dtype=np.int32)
        counts[0, 0, 0] = 100  # 100 observations of class 0

        result = _dirichlet_blend(ml_pred, counts, ess=5.0)
        # With ess=5 and 100 observations, posterior ≈ 100/105 = 0.952 for class 0
        assert result[0, 0, 0] > 0.9
        assert result[0, 0, 0] < 1.0

    def test_sums_to_one(self) -> None:
        """Blended prediction should sum to 1 per cell."""
        rng = np.random.default_rng(42)
        ml_pred = rng.dirichlet(np.ones(K), size=(5, 5))
        counts = rng.poisson(3, size=(5, 5, K)).astype(np.int32)
        result = _dirichlet_blend(ml_pred, counts, ess=5.0)
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_ess_controls_blend_weight(self) -> None:
        """Higher ESS = more weight on ML prior."""
        ml_pred = np.zeros((1, 1, K), dtype=np.float64)
        ml_pred[0, 0, 0] = 0.8
        ml_pred[0, 0, 1:] = 0.2 / (K - 1)

        counts = np.zeros((1, 1, K), dtype=np.int32)
        counts[0, 0, 1] = 10  # observations disagree with prior

        low_ess = _dirichlet_blend(ml_pred, counts, ess=1.0)
        high_ess = _dirichlet_blend(ml_pred, counts, ess=50.0)

        # Low ESS: observations dominate → class 1 should be higher
        # High ESS: prior dominates → class 0 should be higher
        assert low_ess[0, 0, 1] > high_ess[0, 0, 1]
        assert high_ess[0, 0, 0] > low_ess[0, 0, 0]

    def test_no_nans_or_infs(self) -> None:
        """Output should never contain NaN or Inf."""
        ml_pred = np.full((3, 3, K), 1.0 / K, dtype=np.float64)
        counts = np.zeros((3, 3, K), dtype=np.int32)
        result = _dirichlet_blend(ml_pred, counts, ess=5.0)
        assert np.all(np.isfinite(result))


class TestMeanDynamicEntropy:
    """Test adaptive blending gate entropy computation."""

    def test_confident_prediction_low_entropy(self) -> None:
        """One-hot predictions (very confident) should have near-zero entropy."""
        # 5×5 grid, all empty terrain (code 0 — dynamic cell)
        raw_grid = [[0] * 5 for _ in range(5)]
        pred = np.zeros((5, 5, K), dtype=np.float64)
        pred[:, :, 0] = 0.99
        pred[:, :, 1:] = 0.01 / (K - 1)
        entropy = _mean_dynamic_entropy(pred, raw_grid)
        assert entropy < 0.1  # Very confident → low entropy

    def test_uniform_prediction_high_entropy(self) -> None:
        """Uniform predictions should have high entropy."""
        raw_grid = [[0] * 5 for _ in range(5)]
        pred = np.full((5, 5, K), 1.0 / K, dtype=np.float64)
        entropy = _mean_dynamic_entropy(pred, raw_grid)
        max_entropy = np.log(K)
        assert entropy > max_entropy * 0.9  # Near-maximum entropy

    def test_static_cells_excluded(self) -> None:
        """Ocean (10) and mountain (5) cells should be excluded from entropy."""
        # Grid with mix of static and dynamic cells
        raw_grid = [
            [10, 10, 10, 10, 10],  # all ocean (static)
            [5, 5, 5, 5, 5],  # all mountain (static)
            [0, 0, 0, 0, 0],  # all empty (dynamic)
            [10, 5, 10, 5, 10],  # all static
            [0, 1, 2, 3, 4],  # all dynamic
        ]
        # Set confident predictions everywhere
        pred = np.zeros((5, 5, K), dtype=np.float64)
        pred[:, :, 0] = 0.99
        pred[:, :, 1:] = 0.01 / (K - 1)
        entropy = _mean_dynamic_entropy(pred, raw_grid)
        # Should only reflect dynamic cells (rows 2 and 4) which are confident
        assert entropy < 0.1

    def test_all_static_returns_inf(self) -> None:
        """Grid with only ocean/mountain should return inf."""
        raw_grid = [[10, 5, 10], [5, 10, 5]]
        pred = np.full((2, 3, K), 1.0 / K, dtype=np.float64)
        entropy = _mean_dynamic_entropy(pred, raw_grid)
        assert entropy == float("inf")

    def test_threshold_constant_is_reasonable(self) -> None:
        """BLEND_ENTROPY_THRESHOLD should be between 0 and max entropy."""
        max_entropy = np.log(K)
        assert 0 < BLEND_ENTROPY_THRESHOLD < max_entropy
