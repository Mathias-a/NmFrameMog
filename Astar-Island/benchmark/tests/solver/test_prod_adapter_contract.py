"""Tests for prod adapter contract seam.

Validates that the SolverAdapter protocol is sufficient for prod usage,
and that any implementation must provide exactly the required methods.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pytest
from numpy.typing import NDArray

from astar_twin.contracts.api_models import (
    AnalysisResponse,
    RoundDetail,
    SimulateResponse,
    SubmitResponse,
)
from astar_twin.solver.interfaces import SolverAdapter


# ---- Contract completeness ----


def test_solver_adapter_has_required_methods():
    """SolverAdapter defines all 5 required methods."""
    required = ["get_round_detail", "simulate", "submit", "get_analysis", "get_budget"]
    for method in required:
        assert hasattr(SolverAdapter, method), f"Missing: {method}"


def test_solver_adapter_is_protocol():
    """SolverAdapter is a Protocol, not a concrete class."""
    assert issubclass(type(SolverAdapter), type(Protocol))


# ---- Stub compliance ----


class CompliantStub:
    """A minimal compliant stub adapter for testing."""

    def get_round_detail(self, round_id: str) -> RoundDetail:
        raise NotImplementedError

    def simulate(
        self, round_id: str, seed_index: int,
        viewport_x: int, viewport_y: int,
        viewport_w: int, viewport_h: int,
    ) -> SimulateResponse:
        raise NotImplementedError

    def submit(
        self, round_id: str, seed_index: int,
        prediction: NDArray[np.float64],
    ) -> SubmitResponse:
        raise NotImplementedError

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResponse:
        raise NotImplementedError

    def get_budget(self, round_id: str) -> tuple[int, int]:
        raise NotImplementedError


def test_compliant_stub_is_solver_adapter():
    """A stub with all methods satisfies the protocol."""
    stub = CompliantStub()
    # Protocol structural check — if this doesn't raise, the stub is compliant
    assert isinstance(stub, SolverAdapter)


# ---- Non-compliant stubs ----


class MissingSimulate:
    """Stub missing the simulate method."""

    def get_round_detail(self, round_id: str) -> RoundDetail:
        raise NotImplementedError

    def submit(
        self, round_id: str, seed_index: int,
        prediction: NDArray[np.float64],
    ) -> SubmitResponse:
        raise NotImplementedError

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResponse:
        raise NotImplementedError

    def get_budget(self, round_id: str) -> tuple[int, int]:
        raise NotImplementedError


def test_missing_simulate_fails_contract():
    """Stub missing simulate does NOT satisfy the protocol."""
    stub = MissingSimulate()
    assert not isinstance(stub, SolverAdapter)


class MissingSubmit:
    """Stub missing the submit method."""

    def get_round_detail(self, round_id: str) -> RoundDetail:
        raise NotImplementedError

    def simulate(
        self, round_id: str, seed_index: int,
        viewport_x: int, viewport_y: int,
        viewport_w: int, viewport_h: int,
    ) -> SimulateResponse:
        raise NotImplementedError

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResponse:
        raise NotImplementedError

    def get_budget(self, round_id: str) -> tuple[int, int]:
        raise NotImplementedError


def test_missing_submit_fails_contract():
    """Stub missing submit does NOT satisfy the protocol."""
    stub = MissingSubmit()
    assert not isinstance(stub, SolverAdapter)


class MissingGetBudget:
    """Stub missing the get_budget method."""

    def get_round_detail(self, round_id: str) -> RoundDetail:
        raise NotImplementedError

    def simulate(
        self, round_id: str, seed_index: int,
        viewport_x: int, viewport_y: int,
        viewport_w: int, viewport_h: int,
    ) -> SimulateResponse:
        raise NotImplementedError

    def submit(
        self, round_id: str, seed_index: int,
        prediction: NDArray[np.float64],
    ) -> SubmitResponse:
        raise NotImplementedError

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResponse:
        raise NotImplementedError


def test_missing_get_budget_fails_contract():
    """Stub missing get_budget does NOT satisfy the protocol."""
    stub = MissingGetBudget()
    assert not isinstance(stub, SolverAdapter)


# ---- Prod switch seam ----


def test_prod_adapter_only_needs_five_methods():
    """A prod adapter needs ONLY these 5 methods. No extra requirements."""
    # The solver imports no benchmark-specific code;
    # switching to prod only needs a new SolverAdapter implementation.
    import inspect
    members = [
        name for name, _ in inspect.getmembers(SolverAdapter, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    assert set(members) == {"get_round_detail", "simulate", "submit", "get_analysis", "get_budget"}
