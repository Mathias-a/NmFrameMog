"""Comprehensive tests for the A* Island live submission pipeline.

Tests cover: plan_viewports, _accumulate_viewport, _score_viewport,
bayesian_blend, apply_floors, entropy, query_all_seeds, submit_prediction,
and end-to-end cmd_submit.

All tests use synthetic data — no real API calls, no model training.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import TracebackType
from unittest.mock import MagicMock, patch

import numpy as np
from astar_island.api import (
    VP_SIZE,
    _accumulate_viewport,
    _score_viewport,
    plan_viewports,
    query_all_seeds,
    submit_prediction,
)
from astar_island.features import extract_features, feature_count
from astar_island.prob import (
    NUM_CLASSES,
    TERRAIN_TO_CLASS,
    apply_floors,
    bayesian_blend,
    entropy,
)
from astar_island.solver import _SubmitArgs, cmd_submit
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

GRID_H = 40
GRID_W = 40


def make_grid(h: int = 40, w: int = 40) -> NDArray[np.int_]:
    """Return a synthetic initial grid.

    Layout:
    - Border row/col (index 0 and h-1 / w-1) = 10 (ocean)
    - A few interior cells = 5 (mountain) at (5,5), (6,5), (7,5)
    - Row 1 is adjacent to ocean border → coastal
    - Interior = 0 (empty/dynamic)
    """
    grid = np.zeros((h, w), dtype=np.int_)

    # Ocean border
    grid[0, :] = 10
    grid[h - 1, :] = 10
    grid[:, 0] = 10
    grid[:, w - 1] = 10

    # A few mountain cells in the interior
    grid[5, 5] = 5
    grid[6, 5] = 5
    grid[7, 5] = 5

    return grid


def make_prediction(h: int = 40, w: int = 40) -> NDArray[np.float64]:
    """Return a valid (H, W, 6) probability tensor (sums to 1 per cell)."""
    rng = np.random.default_rng(seed=42)
    raw = rng.dirichlet(alpha=np.ones(NUM_CLASSES), size=(h, w))
    return np.asarray(raw, dtype=np.float64)


def make_uniform_prediction(h: int = 40, w: int = 40) -> NDArray[np.float64]:
    """Return a (H, W, 6) tensor with uniform 1/6 across all classes."""
    return np.full((h, w, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)


# ---------------------------------------------------------------------------
# Typed stubs (replace MagicMock to satisfy basedmypy)
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Typed fake httpx response."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict[str, object] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data: dict[str, object] = json_data if json_data is not None else {}
        self.text = text

    def json(self) -> dict[str, object]:
        return self._json_data


class _PostSpy:
    """Typed spy capturing (url, json_body) for each client.post call."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.calls.append((url, json if json is not None else {}))
        return self._response


class _FakeClient:
    """Typed fake httpx.Client context manager."""

    def __init__(self, response: _FakeResponse) -> None:
        self.post = _PostSpy(response)

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


class _FakeClientFactory:
    """Replaces httpx.Client constructor; always returns the same _FakeClient."""

    def __init__(self, client: _FakeClient) -> None:
        self._client = client

    def __call__(
        self,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        transport: object = None,
    ) -> _FakeClient:
        return self._client

    @property
    def client(self) -> _FakeClient:
        return self._client


class _FakeGBDT:
    """Typed stub for PerCellGBDT.  Returns a fixed prediction tensor."""

    def __init__(self, prediction: NDArray[np.float64]) -> None:
        self._prediction = prediction

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        pass

    def predict_grid(self, grid: NDArray[np.int_]) -> NDArray[np.float64]:
        return self._prediction

    def predict_grid_raw(self, grid: NDArray[np.int_]) -> NDArray[np.float64]:
        return self._prediction


class _FakeGBDTFactory:
    """When called, returns a _FakeGBDT (replaces PerCellGBDT constructor).

    Accepts the same keyword arguments as PerCellGBDT.__init__ and ignores them.
    """

    def __init__(self, prediction: NDArray[np.float64]) -> None:
        self._prediction = prediction
        self.instance: _FakeGBDT | None = None

    def __call__(
        self,
        max_iter: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ) -> _FakeGBDT:
        self.instance = _FakeGBDT(self._prediction)
        return self.instance


class _SubmitSpy:
    """Typed spy that records calls to submit_prediction."""

    def __init__(self, return_val: float | None = None) -> None:
        self._return_val = return_val
        self.calls: list[tuple[str, int, NDArray[np.float64]]] = []

    def __call__(
        self,
        round_uuid: str,
        seed_index: int,
        prediction: NDArray[np.float64],
    ) -> float | None:
        self.calls.append((round_uuid, seed_index, prediction))
        return self._return_val


# ---------------------------------------------------------------------------
# Utility: typed scalar extraction from NDArrays
# ---------------------------------------------------------------------------


def _f(arr: NDArray[np.float64], *indices: int) -> float:
    """Extract a scalar float from an NDArray using flat-index arithmetic."""
    flat_idx: int = 0
    stride: int = 1
    dims: list[int] = [int(arr.shape[n]) for n in range(arr.ndim)]
    for dim_size, idx in zip(reversed(dims), reversed(list(indices)), strict=True):
        flat_idx += idx * stride
        stride *= dim_size
    v: np.float64 = arr.flat[flat_idx]
    return float(v)


def _i(arr: NDArray[np.int_], *indices: int) -> int:
    """Extract a scalar int from an NDArray using flat-index arithmetic."""
    flat_idx: int = 0
    stride: int = 1
    dims: list[int] = [int(arr.shape[n]) for n in range(arr.ndim)]
    for dim_size, idx in zip(reversed(dims), reversed(list(indices)), strict=True):
        flat_idx += idx * stride
        stride *= dim_size
    v: np.int_ = arr.flat[flat_idx]
    return int(v)


def _b(arr: NDArray[np.bool_], row: int, col: int, width: int) -> bool:
    """Extract a scalar bool from a 2D NDArray via flat index."""
    v: np.bool_ = arr.flat[row * width + col]
    return bool(v)


def _row_sums(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum last axis of a 3D (H, W, C) array — typed via @ operator."""
    ones: NDArray[np.float64] = np.ones(arr.shape[-1], dtype=np.float64)
    return arr @ ones


def _vec_sum(vec: NDArray[np.float64]) -> float:
    """Sum a 1D vector without using .sum() (which returns Any)."""
    return float(sum(float(x) for x in vec.flat))


# ---------------------------------------------------------------------------
# 2. Tests for plan_viewports()
# ---------------------------------------------------------------------------


def test_plan_viewports_count_40x40() -> None:
    """40×40 grid with VP_SIZE=15 should produce exactly 9 positions (3×3 tiling)."""
    positions = plan_viewports(grid_h=40, grid_w=40, vp_size=15)
    assert len(positions) == 9, f"Expected 9, got {len(positions)}: {positions}"


def test_plan_viewports_unique() -> None:
    """All viewport positions should be unique."""
    positions = plan_viewports(grid_h=40, grid_w=40)
    assert len(positions) == len(set(positions))


def test_plan_viewports_start_at_least_1() -> None:
    """All positions must start >= 1 (skip ocean border)."""
    positions = plan_viewports(grid_h=40, grid_w=40)
    for vx, vy in positions:
        assert vx >= 1, f"vx={vx} is < 1"
        assert vy >= 1, f"vy={vy} is < 1"


def test_plan_viewports_fit_within_grid() -> None:
    """All viewports must fit within grid: vx+VP_SIZE<=W and vy+VP_SIZE<=H."""
    positions = plan_viewports(grid_h=40, grid_w=40, vp_size=15)
    for vx, vy in positions:
        assert vx + 15 <= 40, f"Viewport at ({vx},{vy}) extends beyond width"
        assert vy + 15 <= 40, f"Viewport at ({vx},{vy}) extends beyond height"


def test_plan_viewports_cover_interior() -> None:
    """Every interior cell (1..38 × 1..38) is covered by at least one viewport."""
    positions = plan_viewports(grid_h=40, grid_w=40, vp_size=15)
    covered: NDArray[np.bool_] = np.zeros((40, 40), dtype=np.bool_)
    for vx, vy in positions:
        covered[vy : vy + 15, vx : vx + 15] = True

    for r in range(1, 39):
        for c in range(1, 39):
            assert _b(covered, r, c, GRID_W), (
                f"Interior cell ({r},{c}) not covered by any viewport"
            )


# ---------------------------------------------------------------------------
# 3. Tests for _accumulate_viewport()
# ---------------------------------------------------------------------------


def test_accumulate_viewport_basic() -> None:
    """A 3×3 viewport at (2,2) should set correct one-hot counts in accum/counts."""
    # Build a 3×3 viewport grid with known terrain codes
    # codes: 0→class0, 1→class1, 2→class2, 3→class3, 4→class4, 5→class5
    vp_raw: list[list[int]] = [[0, 1, 2], [3, 4, 5], [0, 1, 2]]
    vp_grid: NDArray[np.int_] = np.array(vp_raw, dtype=np.int_)
    vx, vy = 2, 2
    accum: NDArray[np.float64] = np.zeros(
        (GRID_H, GRID_W, NUM_CLASSES), dtype=np.float64
    )
    counts: NDArray[np.int_] = np.zeros((GRID_H, GRID_W), dtype=np.int_)

    _accumulate_viewport(vp_grid, vx, vy, accum, counts, GRID_H, GRID_W)

    expected_codes: list[list[int]] = [[0, 1, 2], [3, 4, 5], [0, 1, 2]]
    for dy in range(3):
        for dx in range(3):
            gy, gx = vy + dy, vx + dx
            code = expected_codes[dy][dx]
            cls = TERRAIN_TO_CLASS[code]
            accum_val = _f(accum, gy, gx, cls)
            count_val = _i(counts, gy, gx)
            assert accum_val == 1.0, f"accum[{gy},{gx},{cls}] expected 1.0"
            assert count_val == 1, f"counts[{gy},{gx}] expected 1"

    # Cells outside the viewport should remain 0
    assert _f(accum, 0, 0, 0) == 0.0
    assert _i(counts, 0, 0) == 0


def test_accumulate_viewport_cells_outside_remain_zero() -> None:
    """Cells outside the small viewport must not be touched."""
    vp_raw: list[list[int]] = [[1, 1], [1, 1]]
    vp_grid: NDArray[np.int_] = np.array(vp_raw, dtype=np.int_)
    accum: NDArray[np.float64] = np.zeros(
        (GRID_H, GRID_W, NUM_CLASSES), dtype=np.float64
    )
    counts: NDArray[np.int_] = np.zeros((GRID_H, GRID_W), dtype=np.int_)

    _accumulate_viewport(vp_grid, 10, 10, accum, counts, GRID_H, GRID_W)

    # Cells at (9,9), (12,12) must be zero
    assert _i(counts, 9, 9) == 0
    assert _i(counts, 12, 12) == 0


def test_accumulate_viewport_double_counts() -> None:
    """Calling twice on the same viewport should double counts and accumulate accum."""
    vp_raw: list[list[int]] = [[1, 2]]
    vp_grid: NDArray[np.int_] = np.array(vp_raw, dtype=np.int_)
    vx, vy = 3, 3
    accum: NDArray[np.float64] = np.zeros(
        (GRID_H, GRID_W, NUM_CLASSES), dtype=np.float64
    )
    counts: NDArray[np.int_] = np.zeros((GRID_H, GRID_W), dtype=np.int_)

    _accumulate_viewport(vp_grid, vx, vy, accum, counts, GRID_H, GRID_W)
    _accumulate_viewport(vp_grid, vx, vy, accum, counts, GRID_H, GRID_W)

    # class for code 1 → class 1, code 2 → class 2
    assert _i(counts, vy, vx) == 2
    assert _i(counts, vy, vx + 1) == 2
    assert abs(_f(accum, vy, vx, 1) - 2.0) < 1e-9
    assert abs(_f(accum, vy, vx + 1, 2) - 2.0) < 1e-9


def test_accumulate_viewport_terrain_to_class_mapping() -> None:
    """Verify TERRAIN_TO_CLASS maps ocean (10) and plains (11) to class 0."""
    for code in [0, 10, 11]:
        vp_raw: list[list[int]] = [[code]]
        vp_grid: NDArray[np.int_] = np.array(vp_raw, dtype=np.int_)
        accum: NDArray[np.float64] = np.zeros(
            (GRID_H, GRID_W, NUM_CLASSES), dtype=np.float64
        )
        counts: NDArray[np.int_] = np.zeros((GRID_H, GRID_W), dtype=np.int_)
        _accumulate_viewport(vp_grid, 5, 5, accum, counts, GRID_H, GRID_W)
        val = _f(accum, 5, 5, 0)
        assert abs(val - 1.0) < 1e-9, f"code {code} should map to class 0"


# ---------------------------------------------------------------------------
# 4. Tests for _score_viewport()
# ---------------------------------------------------------------------------


def test_score_viewport_sum_matches() -> None:
    """Score should equal sum of entropy in the viewport rectangle."""
    entropy_arr: NDArray[np.float64] = np.zeros((40, 40), dtype=np.float64)
    # Set specific region
    entropy_arr[5:20, 5:20] = 0.5
    vx, vy = 5, 5
    score = _score_viewport(vx, vy, entropy_arr, vp_size=15)
    patch_arr: NDArray[np.float64] = entropy_arr[vy : vy + 15, vx : vx + 15]
    ones15x15: NDArray[np.float64] = np.ones(patch_arr.shape[1], dtype=np.float64)
    # sum 2D patch: row sums then sum over rows
    row_sums_patch: NDArray[np.float64] = patch_arr @ ones15x15
    ones_h: NDArray[np.float64] = np.ones(patch_arr.shape[0], dtype=np.float64)
    expected: float = float(np.dot(row_sums_patch, ones_h))
    assert abs(score - expected) < 1e-9


def test_score_viewport_clips_at_boundary() -> None:
    """Viewport near grid boundary should clip correctly (no index out of bounds)."""
    entropy_arr: NDArray[np.float64] = np.ones((40, 40), dtype=np.float64)
    # Viewport starting at (30, 30), vp_size=15 → clips to [30:40, 30:40] = 10×10
    score = _score_viewport(30, 30, entropy_arr, vp_size=15)
    expected = 100.0  # 10×10 = 100 cells, each with value 1.0
    assert abs(score - expected) < 1e-9


def test_score_viewport_full_interior() -> None:
    """Score at (1,1) with vp_size=15 on entropy=1 should be exactly 225."""
    entropy_arr: NDArray[np.float64] = np.ones((40, 40), dtype=np.float64)
    score = _score_viewport(1, 1, entropy_arr, vp_size=15)
    assert abs(score - 225.0) < 1e-9


# ---------------------------------------------------------------------------
# 5. Tests for bayesian_blend()
# ---------------------------------------------------------------------------


def test_bayesian_blend_unobserved_returns_prior() -> None:
    """Where n_obs == 0, blended result should equal prior."""
    prior = make_prediction()
    counts: NDArray[np.float64] = np.zeros(
        (GRID_H, GRID_W, NUM_CLASSES), dtype=np.float64
    )
    n_obs: NDArray[np.int_] = np.zeros((GRID_H, GRID_W), dtype=np.int_)

    result = bayesian_blend(prior, counts, n_obs, alpha=5.0)
    np.testing.assert_allclose(result, prior, atol=1e-9)


def test_bayesian_blend_observed_formula() -> None:
    """Test blend formula: (alpha*prior + counts) / (alpha + n), renormalized."""
    prior_vals: list[float] = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]
    prior_vec: NDArray[np.float64] = np.fromiter(prior_vals, dtype=np.float64, count=6)
    prior: NDArray[np.float64] = np.zeros((1, 1, NUM_CLASSES), dtype=np.float64)
    prior[0, 0, :] = prior_vec

    counts: NDArray[np.float64] = np.zeros((1, 1, NUM_CLASSES), dtype=np.float64)
    counts[0, 0, 0] = 1.0  # one observation of class 0

    n_obs: NDArray[np.int_] = np.ones((1, 1), dtype=np.int_)

    alpha = 5.0
    result = bayesian_blend(prior, counts, n_obs, alpha=alpha)

    # Raw formula: (alpha * prior + counts) / (alpha + 1)
    obs_vals: list[float] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    obs_vec: NDArray[np.float64] = np.fromiter(obs_vals, dtype=np.float64, count=6)
    raw: NDArray[np.float64] = (alpha * prior_vec + obs_vec) / (alpha + 1.0)
    s: float = _vec_sum(raw)
    renorm: NDArray[np.float64] = raw / s

    result_row: NDArray[np.float64] = result[0, 0, :]
    np.testing.assert_allclose(result_row, renorm, atol=1e-9)


def test_bayesian_blend_sums_to_one() -> None:
    """Blended result must always sum to 1 along last axis."""
    prior = make_prediction()
    counts: NDArray[np.float64] = (
        np.random.default_rng(99)
        .dirichlet(np.ones(NUM_CLASSES), size=(GRID_H, GRID_W))
        .astype(np.float64)
    )
    n_obs: NDArray[np.int_] = np.random.default_rng(99).integers(
        0, 10, size=(GRID_H, GRID_W), dtype=np.int_
    )

    result = bayesian_blend(prior, counts, n_obs, alpha=5.0)
    row_sums: NDArray[np.float64] = _row_sums(result)
    expected: NDArray[np.float64] = np.ones((GRID_H, GRID_W), dtype=np.float64)
    np.testing.assert_allclose(row_sums, expected, atol=1e-9)


def test_bayesian_blend_high_n_obs_overwhelms_prior() -> None:
    """With high n_obs, result should be close to empirical counts/n."""
    prior: NDArray[np.float64] = np.zeros((1, 1, NUM_CLASSES), dtype=np.float64)
    prior_vals: list[float] = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]
    prior[0, 0, :] = np.fromiter(prior_vals, dtype=np.float64, count=6)

    # 100 observations all class 0
    n = 100
    counts: NDArray[np.float64] = np.zeros((1, 1, NUM_CLASSES), dtype=np.float64)
    counts[0, 0, 0] = float(n)

    n_obs: NDArray[np.int_] = np.full((1, 1), n, dtype=np.int_)

    result = bayesian_blend(prior, counts, n_obs, alpha=5.0)

    # With alpha=5 and n=100, weight on prior is 5/(5+100) ≈ 0.048 << 1
    # The first class should dominate
    assert _f(result, 0, 0, 0) > 0.9


def test_bayesian_blend_alpha_zero_equals_empirical() -> None:
    """With alpha=0, result equals empirical counts/n (renormalized)."""
    prior: NDArray[np.float64] = np.zeros((1, 1, NUM_CLASSES), dtype=np.float64)
    prior_vals: list[float] = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]
    prior[0, 0, :] = np.fromiter(prior_vals, dtype=np.float64, count=6)

    # 3 observations: 2 class-0, 1 class-1
    counts: NDArray[np.float64] = np.zeros((1, 1, NUM_CLASSES), dtype=np.float64)
    counts[0, 0, 0] = 2.0
    counts[0, 0, 1] = 1.0

    n_obs: NDArray[np.int_] = np.full((1, 1), 3, dtype=np.int_)

    result = bayesian_blend(prior, counts, n_obs, alpha=0.0)

    # alpha=0 → result = counts / 3, renormalized → [2/3, 1/3, 0, 0, 0, 0]
    raw_counts: NDArray[np.float64] = counts[0, 0, :]
    raw: NDArray[np.float64] = raw_counts / 3.0
    s: float = _vec_sum(raw)
    renorm: NDArray[np.float64] = raw / s

    result_row: NDArray[np.float64] = result[0, 0, :]
    np.testing.assert_allclose(result_row, renorm, atol=1e-9)


# ---------------------------------------------------------------------------
# 6. Tests for apply_floors()
# ---------------------------------------------------------------------------


def test_apply_floors_ocean_onehot() -> None:
    """Ocean cells (code 10) should become one-hot [1, 0, 0, 0, 0, 0]."""
    grid = make_grid()
    probs = make_prediction()

    result = apply_floors(probs, grid)

    # Top border is ocean
    expected_vals: list[float] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected: NDArray[np.float64] = np.fromiter(
        expected_vals, dtype=np.float64, count=6
    )
    for c in range(GRID_W):
        cell: NDArray[np.float64] = result[0, c, :]
        np.testing.assert_allclose(cell, expected, atol=1e-9)


def test_apply_floors_mountain_onehot() -> None:
    """Mountain cells (code 5) should become one-hot [0, 0, 0, 0, 0, 1]."""
    grid = make_grid()
    probs = make_prediction()

    result = apply_floors(probs, grid)

    # Mountains at (5,5), (6,5), (7,5)
    expected_vals: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    expected: NDArray[np.float64] = np.fromiter(
        expected_vals, dtype=np.float64, count=6
    )
    for row in [5, 6, 7]:
        cell: NDArray[np.float64] = result[row, 5, :]
        np.testing.assert_allclose(cell, expected, atol=1e-9)


def test_apply_floors_dynamic_mountain_class_is_floor_impossible() -> None:
    """Mountain class (5) on dynamic cells must be lower than FLOOR_STANDARD.

    After renormalization, FLOOR_IMPOSSIBLE cells don't stay exactly 0.001, but
    they must be strictly less than FLOOR_STANDARD (0.01) since other classes
    have at least FLOOR_STANDARD before normalization.
    """
    grid = make_grid()
    probs = make_uniform_prediction()
    result = apply_floors(probs, grid)

    # Dynamic non-coastal interior cell e.g. (10, 10)
    # Mountain class should be less than FLOOR_STANDARD after renorm
    mountain_val = _f(result, 10, 10, 5)
    assert mountain_val < 0.01, (
        f"Mountain class should be < FLOOR_STANDARD, got {mountain_val}"
    )
    # But should be positive (it got renormalized from FLOOR_IMPOSSIBLE=0.001)
    assert mountain_val > 0.0


def test_apply_floors_non_coastal_port_class_is_floor_impossible() -> None:
    """Port class (2) on non-coastal dynamic cells must be lower than FLOOR_STANDARD.

    After renormalization, FLOOR_IMPOSSIBLE cells don't stay exactly 0.001,
    but must be strictly less than FLOOR_STANDARD (0.01).
    """
    grid = make_grid()
    probs = make_uniform_prediction()
    result = apply_floors(probs, grid)

    # Interior non-coastal dynamic cell (10, 10) — far from ocean border
    port_val = _f(result, 10, 10, 2)
    assert port_val < 0.01, (
        f"Port class on non-coastal cell should be < FLOOR_STANDARD, got {port_val}"
    )
    assert port_val > 0.0


def test_apply_floors_dynamic_standard_floor() -> None:
    """Non-impossible classes on dynamic cells must be >= FLOOR_STANDARD=0.01."""
    grid = make_grid()
    probs = make_uniform_prediction()
    result = apply_floors(probs, grid)

    # Dynamic cell at (10, 10) — check non-impossible classes
    cell: NDArray[np.float64] = result[10, 10, :]
    for cls in [0, 1, 3, 4]:  # not mountain(5) or port(2) which are FLOOR_IMPOSSIBLE
        val: np.float64 = cell.flat[cls]
        assert float(val) >= 0.01 - 1e-9, f"class {cls} should be >= FLOOR_STANDARD"


def test_apply_floors_dynamic_cells_sum_to_one() -> None:
    """All dynamic cells must sum to 1.0 after apply_floors."""
    grid = make_grid()
    probs = make_prediction()
    result = apply_floors(probs, grid)

    mask1: NDArray[np.bool_] = np.not_equal(grid, 10)
    mask2: NDArray[np.bool_] = np.not_equal(grid, 5)
    dynamic_mask: NDArray[np.bool_] = np.logical_and(mask1, mask2)
    subset: NDArray[np.float64] = result[dynamic_mask]
    row_sums: NDArray[np.float64] = _row_sums(subset)
    n_dyn = int(row_sums.shape[0])
    expected: NDArray[np.float64] = np.ones(n_dyn, dtype=np.float64)
    np.testing.assert_allclose(row_sums, expected, atol=1e-9)


def test_apply_floors_coastal_port_class_at_standard_floor() -> None:
    """Coastal dynamic cells should have port class >= FLOOR_STANDARD=0.01."""
    grid = make_grid()
    # Row 1 is adjacent to ocean border (row 0), so cells in row 1 are coastal
    probs = make_uniform_prediction()
    result = apply_floors(probs, grid)

    # Find a coastal dynamic cell: row 1, col 5 (not ocean, adjacent to ocean at row 0)
    assert _f(result, 1, 5, 2) >= 0.01 - 1e-9


# ---------------------------------------------------------------------------
# 7. Tests for entropy()
# ---------------------------------------------------------------------------


def test_entropy_uniform_is_max() -> None:
    """Uniform [1/6]*6 should give maximum entropy = ln(6)."""
    uniform: NDArray[np.float64] = np.full(
        (1, 1, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64
    )
    result = entropy(uniform)
    expected = math.log(6)
    val: np.float64 = result.flat[0]
    assert abs(float(val) - expected) < 1e-6


def test_entropy_onehot_is_zero() -> None:
    """One-hot distribution should have entropy ≈ 0."""
    onehot: NDArray[np.float64] = np.zeros((1, 1, NUM_CLASSES), dtype=np.float64)
    onehot[0, 0, 0] = 1.0
    result = entropy(onehot)
    val: np.float64 = result.flat[0]
    assert abs(float(val)) < 1e-9


def test_entropy_known_distribution() -> None:
    """Manual calculation for a known distribution."""
    prob_vals: list[float] = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125]
    probs_vec: NDArray[np.float64] = np.fromiter(prob_vals, dtype=np.float64, count=6)
    probs: NDArray[np.float64] = probs_vec.reshape(1, 1, NUM_CLASSES)

    result = entropy(probs)
    clipped: NDArray[np.float64] = np.clip(probs_vec, 1e-12, 1.0)
    log_vals: NDArray[np.float64] = np.log(clipped)
    product: NDArray[np.float64] = probs_vec * log_vals
    expected = -_vec_sum(product)
    val: np.float64 = result.flat[0]
    assert abs(float(val) - expected) < 1e-6


def test_entropy_shape_preserved() -> None:
    """Entropy output shape should be input shape minus last dimension."""
    probs = make_uniform_prediction(h=10, w=8)
    result = entropy(probs)
    assert result.shape == (10, 8)


# ---------------------------------------------------------------------------
# 8. Tests for query_all_seeds() with mocked HTTP
# ---------------------------------------------------------------------------


def _make_simulate_response(vp_h: int = VP_SIZE, vp_w: int = VP_SIZE) -> _FakeResponse:
    """Create a typed fake simulate response returning a grid of zeros."""
    grid_data: list[list[int]] = [[0] * vp_w for _ in range(vp_h)]
    return _FakeResponse(
        status_code=200,
        json_data={
            "grid": grid_data,
            "queries_used": 1,
            "queries_max": 50,
        },
    )


def _make_client_factory(
    response: _FakeResponse,
) -> tuple[_FakeClient, _FakeClientFactory]:
    """Return (fake_client, factory) for patching httpx.Client."""
    client = _FakeClient(response)
    factory = _FakeClientFactory(client)
    return client, factory


def test_query_all_seeds_returns_five_results() -> None:
    """query_all_seeds should return a list of 5 (accum, counts) tuples."""
    grids = [make_grid() for _ in range(5)]
    predictions = [make_uniform_prediction() for _ in range(5)]

    fake_client, factory = _make_client_factory(_make_simulate_response())

    with (
        patch("astar_island.api._get_token", return_value="fake-token"),
        patch("astar_island.api.httpx.Client", new=factory),
        patch("astar_island.api.httpx.HTTPTransport", return_value=MagicMock()),
    ):
        results = query_all_seeds("test-uuid", grids, predictions, total_budget=9)

    assert len(results) == 5
    for accum, counts in results:
        assert accum.shape == (40, 40, 6)
        assert counts.shape == (40, 40)


def test_query_all_seeds_respects_budget() -> None:
    """Total queries made must not exceed total_budget."""
    grids = [make_grid() for _ in range(5)]
    predictions = [make_uniform_prediction() for _ in range(5)]

    fake_client, factory = _make_client_factory(_make_simulate_response())

    budget = 5
    with (
        patch("astar_island.api._get_token", return_value="fake-token"),
        patch("astar_island.api.httpx.Client", new=factory),
        patch("astar_island.api.httpx.HTTPTransport", return_value=MagicMock()),
    ):
        query_all_seeds("test-uuid", grids, predictions, total_budget=budget)

    actual_calls = len(fake_client.post.calls)
    assert actual_calls <= budget, f"Made {actual_calls} calls but budget was {budget}"


def test_query_all_seeds_all_seeds_get_results() -> None:
    """All 5 seeds must appear in the results even if some get 0 viewports."""
    grids = [make_grid() for _ in range(5)]
    predictions = [make_uniform_prediction() for _ in range(5)]

    fake_client, factory = _make_client_factory(_make_simulate_response())

    with (
        patch("astar_island.api._get_token", return_value="fake-token"),
        patch("astar_island.api.httpx.Client", new=factory),
        patch("astar_island.api.httpx.HTTPTransport", return_value=MagicMock()),
    ):
        results = query_all_seeds("test-uuid", grids, predictions, total_budget=50)

    assert len(results) == 5


def test_query_all_seeds_correct_shapes() -> None:
    """Each result tuple must have arrays with correct shapes."""
    grids = [make_grid() for _ in range(5)]
    predictions = [make_prediction() for _ in range(5)]

    fake_client, factory = _make_client_factory(_make_simulate_response())

    with (
        patch("astar_island.api._get_token", return_value="fake-token"),
        patch("astar_island.api.httpx.Client", new=factory),
        patch("astar_island.api.httpx.HTTPTransport", return_value=MagicMock()),
    ):
        results = query_all_seeds("test-uuid", grids, predictions, total_budget=9)

    for accum, counts in results:
        assert accum.dtype == np.float64
        assert counts.dtype == np.int_
        assert accum.shape == (40, 40, 6)
        assert counts.shape == (40, 40)


# ---------------------------------------------------------------------------
# 9. Tests for submit_prediction() serialization
# ---------------------------------------------------------------------------


def _make_submit_client() -> tuple[_FakeClient, _FakeClientFactory]:
    """Return (client, factory) pre-configured for submit tests."""
    resp = _FakeResponse(status_code=200, json_data={"score": 42.0})
    return _make_client_factory(resp)


def test_submit_prediction_json_structure() -> None:
    """Verify JSON body contains round_id, seed_index, prediction as nested list."""
    prediction = make_prediction()
    round_uuid = "test-round-uuid-1234"
    seed_idx = 2

    fake_client, factory = _make_submit_client()

    with (
        patch("astar_island.api._get_token", return_value="fake-token"),
        patch("astar_island.api.httpx.Client", new=factory),
        patch("astar_island.api.httpx.HTTPTransport", return_value=MagicMock()),
    ):
        score = submit_prediction(round_uuid, seed_idx, prediction)

    # Score should be 42.0
    assert score is not None
    assert abs(score - 42.0) < 1e-6

    # Inspect the call args
    assert len(fake_client.post.calls) == 1
    _url, body = fake_client.post.calls[0]

    round_id_val: object = body.get("round_id")
    assert isinstance(round_id_val, str)
    assert round_id_val == round_uuid

    seed_val: object = body.get("seed_index")
    assert isinstance(seed_val, int)
    assert seed_val == seed_idx

    pred_val: object = body.get("prediction")
    assert isinstance(pred_val, list), "prediction should be a nested list"
    assert isinstance(pred_val[0], list)
    assert isinstance(pred_val[0][0], list)
    assert len(pred_val) == 40
    assert len(pred_val[0]) == 40
    assert len(pred_val[0][0]) == 6


def test_submit_prediction_values_sum_to_one() -> None:
    """Prediction values in JSON body should sum to ~1.0 per cell."""
    prediction = make_prediction()
    round_uuid = "test-uuid-xyz"

    resp = _FakeResponse(status_code=200, json_data={})
    fake_client, factory = _make_client_factory(resp)

    with (
        patch("astar_island.api._get_token", return_value="fake-token"),
        patch("astar_island.api.httpx.Client", new=factory),
        patch("astar_island.api.httpx.HTTPTransport", return_value=MagicMock()),
    ):
        submit_prediction(round_uuid, 0, prediction)

    assert len(fake_client.post.calls) == 1
    _url, body = fake_client.post.calls[0]

    pred_val: object = body.get("prediction")
    assert isinstance(pred_val, list)

    for row in pred_val:
        assert isinstance(row, list)
        for cell in row:
            assert isinstance(cell, list)
            cell_sum = sum(float(v) for v in cell)
            assert abs(cell_sum - 1.0) < 1e-6, f"cell sums to {cell_sum}"


def test_submit_prediction_no_numpy_in_json() -> None:
    """Prediction body must be JSON-serializable (no numpy arrays)."""
    prediction = make_prediction()

    resp = _FakeResponse(status_code=200, json_data={})
    fake_client, factory = _make_client_factory(resp)

    with (
        patch("astar_island.api._get_token", return_value="fake-token"),
        patch("astar_island.api.httpx.Client", new=factory),
        patch("astar_island.api.httpx.HTTPTransport", return_value=MagicMock()),
    ):
        submit_prediction("uuid", 0, prediction)

    assert len(fake_client.post.calls) == 1
    _url, body = fake_client.post.calls[0]
    # Should serialize without errors
    json_str = json.dumps(body)
    assert len(json_str) > 0


# ---------------------------------------------------------------------------
# 10. End-to-end cmd_submit with mocks
# ---------------------------------------------------------------------------


def _make_submit_args(dry_run: bool = False) -> _SubmitArgs:
    """Create a _SubmitArgs-like object."""
    args = _SubmitArgs()
    args.dry_run = dry_run
    args.verbose = False
    args.command = "submit"
    return args


def _make_training_data() -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]
]:
    """Return synthetic (x_train, y_train, mask) for mocking prepare_training_data."""
    x_train: NDArray[np.float64] = (
        np.random.default_rng(0).random((100, 10)).astype(np.float64)
    )
    y_train: NDArray[np.float64] = (
        np.random.default_rng(0).dirichlet(np.ones(6), size=100).astype(np.float64)
    )
    mask: NDArray[np.bool_] = np.ones(100, dtype=np.bool_)
    return x_train, y_train, mask


def test_cmd_submit_calls_all_steps() -> None:
    """All 6 steps of cmd_submit should be called in the correct order."""
    grids = [make_grid() for _ in range(5)]
    synthetic_rounds = {1: [(make_grid(), make_prediction())] * 5}
    x_train, y_train, mask = _make_training_data()

    mock_pred_grid = make_prediction()
    query_results = [
        (
            np.zeros((40, 40, 6), dtype=np.float64),
            np.zeros((40, 40), dtype=np.int_),
        )
        for _ in range(5)
    ]

    gbdt_factory = _FakeGBDTFactory(mock_pred_grid)
    submit_spy = _SubmitSpy(return_val=99.0)

    with (
        patch("astar_island.solver.download_all") as mock_dl,
        patch(
            "astar_island.solver.load_all_rounds",
            return_value=synthetic_rounds,
        ) as mock_lar,
        patch(
            "astar_island.solver.fetch_active_round",
            return_value=("uuid-123", 42, grids),
        ) as mock_far,
        patch(
            "astar_island.solver.prepare_training_data",
            return_value=(x_train, y_train, mask),
        ) as mock_ptd,
        patch("astar_island.solver.PerCellGBDT", new=gbdt_factory),
        patch("astar_island.solver.tta_predict", return_value=mock_pred_grid),
        patch(
            "astar_island.solver.query_all_seeds",
            return_value=query_results,
        ) as mock_qas,
        patch("astar_island.solver.submit_prediction", new=submit_spy),
    ):
        cmd_submit(_make_submit_args(dry_run=False))

    mock_dl.assert_called_once()
    mock_lar.assert_called_once()
    mock_far.assert_called_once()
    mock_ptd.assert_called_once()
    mock_qas.assert_called_once()
    # submit_prediction called 5 times (once per seed)
    assert len(submit_spy.calls) == 5


def test_cmd_submit_submits_five_seeds() -> None:
    """submit_prediction should be called exactly 5 times (once per seed)."""
    grids = [make_grid() for _ in range(5)]
    synthetic_rounds = {1: [(make_grid(), make_prediction())] * 5}
    x_train, y_train, mask = _make_training_data()

    mock_pred_grid = make_prediction()
    query_results = [
        (
            np.zeros((40, 40, 6), dtype=np.float64),
            np.zeros((40, 40), dtype=np.int_),
        )
        for _ in range(5)
    ]

    gbdt_factory = _FakeGBDTFactory(mock_pred_grid)
    submit_spy = _SubmitSpy(return_val=None)

    with (
        patch("astar_island.solver.download_all"),
        patch(
            "astar_island.solver.load_all_rounds",
            return_value=synthetic_rounds,
        ),
        patch(
            "astar_island.solver.fetch_active_round",
            return_value=("uuid-abc", 7, grids),
        ),
        patch(
            "astar_island.solver.prepare_training_data",
            return_value=(x_train, y_train, mask),
        ),
        patch("astar_island.solver.PerCellGBDT", new=gbdt_factory),
        patch("astar_island.solver.tta_predict", return_value=mock_pred_grid),
        patch(
            "astar_island.solver.query_all_seeds",
            return_value=query_results,
        ),
        patch("astar_island.solver.submit_prediction", new=submit_spy),
    ):
        cmd_submit(_make_submit_args(dry_run=False))

    assert len(submit_spy.calls) == 5
    # Each call should have seed_index 0..4
    for i, (_, seed_idx, _arr) in enumerate(submit_spy.calls):
        assert seed_idx == i


def test_cmd_submit_dry_run_does_not_call_submit(tmp_path: Path) -> None:
    """With --dry-run, submit_prediction must NOT be called; .npy files are saved."""
    grids = [make_grid() for _ in range(5)]
    synthetic_rounds = {1: [(make_grid(), make_prediction())] * 5}
    x_train, y_train, mask = _make_training_data()

    mock_pred_grid = make_prediction()
    query_results = [
        (
            np.zeros((40, 40, 6), dtype=np.float64),
            np.zeros((40, 40), dtype=np.int_),
        )
        for _ in range(5)
    ]

    gbdt_factory = _FakeGBDTFactory(mock_pred_grid)
    submit_spy = _SubmitSpy(return_val=None)

    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        with (
            patch("astar_island.solver.download_all"),
            patch(
                "astar_island.solver.load_all_rounds",
                return_value=synthetic_rounds,
            ),
            patch(
                "astar_island.solver.fetch_active_round",
                return_value=("uuid-dry", 99, grids),
            ),
            patch(
                "astar_island.solver.prepare_training_data",
                return_value=(x_train, y_train, mask),
            ),
            patch("astar_island.solver.PerCellGBDT", new=gbdt_factory),
            patch("astar_island.solver.tta_predict", return_value=mock_pred_grid),
            patch(
                "astar_island.solver.query_all_seeds",
                return_value=query_results,
            ),
            patch("astar_island.solver.submit_prediction", new=submit_spy),
        ):
            cmd_submit(_make_submit_args(dry_run=True))

        # submit_prediction must NOT have been called
        assert len(submit_spy.calls) == 0

        # .npy files should have been saved
        saved = list(tmp_path.glob("prediction_r99_s*.npy"))
        assert len(saved) == 5, f"Expected 5 .npy files, got {saved}"
    finally:
        os.chdir(original_dir)


def test_cmd_submit_submit_prediction_args_correct() -> None:
    """submit_prediction should be called with (round_uuid, seed_idx, prediction)."""
    expected_uuid = "round-uuid-test-xyz"
    grids = [make_grid() for _ in range(5)]
    synthetic_rounds = {1: [(make_grid(), make_prediction())] * 5}
    x_train, y_train, mask = _make_training_data()

    mock_pred_grid = make_prediction()
    query_results = [
        (
            np.zeros((40, 40, 6), dtype=np.float64),
            np.zeros((40, 40), dtype=np.int_),
        )
        for _ in range(5)
    ]

    gbdt_factory = _FakeGBDTFactory(mock_pred_grid)
    submit_spy = _SubmitSpy(return_val=None)

    with (
        patch("astar_island.solver.download_all"),
        patch(
            "astar_island.solver.load_all_rounds",
            return_value=synthetic_rounds,
        ),
        patch(
            "astar_island.solver.fetch_active_round",
            return_value=(expected_uuid, 5, grids),
        ),
        patch(
            "astar_island.solver.prepare_training_data",
            return_value=(x_train, y_train, mask),
        ),
        patch("astar_island.solver.PerCellGBDT", new=gbdt_factory),
        patch("astar_island.solver.tta_predict", return_value=mock_pred_grid),
        patch(
            "astar_island.solver.query_all_seeds",
            return_value=query_results,
        ),
        patch("astar_island.solver.submit_prediction", new=submit_spy),
    ):
        cmd_submit(_make_submit_args(dry_run=False))

    assert len(submit_spy.calls) == 5
    for uuid_arg, _seed_idx, pred_arg in submit_spy.calls:
        assert uuid_arg == expected_uuid, "round_uuid mismatch"
        assert pred_arg.shape == (40, 40, 6)
        row_sums: NDArray[np.float64] = _row_sums(pred_arg)
        expected_ones: NDArray[np.float64] = np.ones((40, 40), dtype=np.float64)
        np.testing.assert_allclose(row_sums, expected_ones, atol=1e-6)


# ---------------------------------------------------------------------------
# 11. Tests for extract_features() — plains features
# ---------------------------------------------------------------------------


def test_extract_features_count_matches() -> None:
    """extract_features output dim must equal feature_count()."""
    grid = make_grid()
    feats = extract_features(grid)
    assert feats.shape == (40, 40, feature_count())


def test_extract_features_plains_indicator() -> None:
    """Plains cells (code 11) should have is_plains=1, others 0."""
    grid = make_grid()
    grid[10, 10] = 11
    grid[15, 20] = 11
    feats = extract_features(grid)

    plains_feat_idx = 30
    assert abs(_f(feats, 10, 10, plains_feat_idx) - 1.0) < 1e-9
    assert abs(_f(feats, 15, 20, plains_feat_idx) - 1.0) < 1e-9
    assert abs(_f(feats, 20, 20, plains_feat_idx) - 0.0) < 1e-9


def test_extract_features_plains_distance() -> None:
    """Distance-to-plains should be 0 at plains cells and > 0 elsewhere."""
    grid = make_grid()
    grid[20, 20] = 11
    feats = extract_features(grid)

    dist_plains_idx = 31
    assert abs(_f(feats, 20, 20, dist_plains_idx)) < 1e-9
    assert _f(feats, 10, 10, dist_plains_idx) > 0.0


def test_extract_features_no_plains_fills_ones() -> None:
    """When no plains exist, distance-to-plains should be 1.0 everywhere."""
    grid = make_grid()
    feats = extract_features(grid)

    dist_plains_idx = 31
    for r in range(40):
        for c in range(40):
            assert abs(_f(feats, r, c, dist_plains_idx) - 1.0) < 1e-9


def test_extract_features_plains_freq_nonzero_near_plains() -> None:
    """Plains neighbor frequency at 3x3 should be > 0 near a plains cell."""
    grid = make_grid()
    grid[20, 20] = 11
    feats = extract_features(grid)

    plains_freq_3x3_idx = 32
    assert _f(feats, 20, 20, plains_freq_3x3_idx) > 0.0
    assert _f(feats, 20, 21, plains_freq_3x3_idx) > 0.0
    assert _f(feats, 5, 10, plains_freq_3x3_idx) < 1e-9
