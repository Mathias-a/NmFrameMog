"""Solver-facing adapter protocols.

The solver core depends ONLY on these protocols, never on benchmark stores,
FastAPI internals, or fixture data directly.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from astar_twin.contracts.api_models import (
    AnalysisResponse,
    InitialState,
    RoundDetail,
    SimulateResponse,
    SubmitResponse,
)

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class SolverAdapter(Protocol):
    """Adapter that the solver pipeline uses to interact with any backend.

    Implementations:
      - BenchmarkAdapter: drives the local digital twin
      - (future) ProdAdapter: drives the real competition API
    """

    def get_round_detail(self, round_id: str) -> RoundDetail:
        """Fetch round metadata and initial states for all seeds."""
        ...

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> SimulateResponse:
        """Execute one viewport query (costs 1 query from budget)."""
        ...

    def submit(
        self,
        round_id: str,
        seed_index: int,
        prediction: NDArray[np.float64],
    ) -> SubmitResponse:
        """Submit a H×W×6 prediction tensor for one seed."""
        ...

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResponse:
        """Fetch post-round ground truth and score (only after round completes)."""
        ...

    def get_budget(self, round_id: str) -> tuple[int, int]:
        """Return (queries_used, queries_max)."""
        ...
