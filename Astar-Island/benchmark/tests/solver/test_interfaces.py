"""Tests for solver adapter contracts and pipeline boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import (
    AnalysisResponse,
    InitialState,
    RoundDetail,
    SimulateResponse,
    SimSettlement,
    SubmitResponse,
    ViewportBounds,
)
from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.solver.interfaces import SolverAdapter
from astar_twin.solver.pipeline import SolveResult, solve


class StubAdapter:
    """Minimal adapter that satisfies SolverAdapter protocol without benchmark stores."""

    def __init__(self, round_detail: RoundDetail) -> None:
        self._detail = round_detail
        self._queries_used = 0

    def get_round_detail(self, round_id: str) -> RoundDetail:
        return self._detail

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> SimulateResponse:
        self._queries_used += 1
        grid = [[0] * viewport_w for _ in range(viewport_h)]
        return SimulateResponse(
            grid=grid,
            settlements=[],
            viewport=ViewportBounds(x=viewport_x, y=viewport_y, w=viewport_w, h=viewport_h),
            width=self._detail.map_width,
            height=self._detail.map_height,
            queries_used=self._queries_used,
            queries_max=50,
        )

    def submit(
        self,
        round_id: str,
        seed_index: int,
        prediction: NDArray[np.float64],
    ) -> SubmitResponse:
        return SubmitResponse(status="accepted", round_id=round_id, seed_index=seed_index)

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResponse:
        h, w = self._detail.map_height, self._detail.map_width
        gt = np.full((h, w, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64).tolist()
        return AnalysisResponse(
            prediction=None,
            ground_truth=gt,
            score=None,
            width=w,
            height=h,
            initial_grid=None,
        )

    def get_budget(self, round_id: str) -> tuple[int, int]:
        return (self._queries_used, 50)


def _make_round_detail() -> RoundDetail:
    grid = [[0] * 10 for _ in range(10)]
    initial_state = InitialState(
        grid=grid,
        settlements=[],
    )
    return RoundDetail(
        id="test-round",
        round_number=1,
        status="active",
        map_width=10,
        map_height=10,
        seeds_count=5,
        initial_states=[initial_state] * 5,
    )


def test_stub_adapter_satisfies_protocol() -> None:
    """StubAdapter should be usable where SolverAdapter is expected."""
    detail = _make_round_detail()
    adapter: SolverAdapter = StubAdapter(detail)
    result = adapter.get_round_detail("test-round")
    assert result.id == "test-round"
    assert result.seeds_count == 5


def test_pipeline_returns_five_tensors_via_stub() -> None:
    """Pipeline returns one tensor per seed through a stub adapter."""
    detail = _make_round_detail()
    adapter = StubAdapter(detail)
    result = solve(adapter, "test-round")
    assert isinstance(result, SolveResult)
    assert len(result.tensors) == 5
    for tensor in result.tensors:
        assert tensor.shape == (10, 10, NUM_CLASSES)
        sums = np.sum(tensor, axis=2)
        assert np.allclose(sums, 1.0, atol=1e-6)


def test_solver_does_not_import_benchmark_stores() -> None:
    """Solver core modules must not depend on benchmark store internals."""
    import importlib
    import sys

    # Temporarily block store imports
    blocked = "astar_twin.api.store"
    original = sys.modules.get(blocked)
    saved_mods = {
        name: module for name, module in sys.modules.items() if name.startswith("astar_twin.solver")
    }
    sys.modules[blocked] = None  # type: ignore[assignment]
    try:
        # Force reimport of solver modules
        for mod_name in list(sys.modules):
            if mod_name.startswith("astar_twin.solver"):
                del sys.modules[mod_name]
        importlib.import_module("astar_twin.solver.interfaces")
        importlib.import_module("astar_twin.solver.pipeline")
    finally:
        for mod_name in list(sys.modules):
            if mod_name.startswith("astar_twin.solver"):
                del sys.modules[mod_name]
        sys.modules.update(saved_mods)
        for mod_name, module in saved_mods.items():
            parent_name, _, attr_name = mod_name.rpartition(".")
            if not parent_name:
                continue
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, attr_name, module)
        if original is not None:
            sys.modules[blocked] = original
        elif blocked in sys.modules:
            del sys.modules[blocked]


def test_simulate_returns_valid_response() -> None:
    detail = _make_round_detail()
    adapter = StubAdapter(detail)
    resp = adapter.simulate("test-round", 0, 0, 0, 10, 10)
    assert resp.queries_used == 1
    assert resp.queries_max == 50
    assert len(resp.grid) == 10
    assert len(resp.grid[0]) == 10


def test_budget_tracking() -> None:
    detail = _make_round_detail()
    adapter = StubAdapter(detail)
    assert adapter.get_budget("test-round") == (0, 50)
    adapter.simulate("test-round", 0, 0, 0, 5, 5)
    assert adapter.get_budget("test-round") == (1, 50)
