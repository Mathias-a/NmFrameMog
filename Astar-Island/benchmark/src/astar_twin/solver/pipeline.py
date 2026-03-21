"""End-to-end solver pipeline.

Accepts an adapter and returns 5 prediction tensors (one per seed).
This module is the single entrypoint that orchestrates:
  1. load round detail
  2. generate bootstrap candidates
  3. issue bootstrap queries
  4. update posterior
  5. iterate adaptive + reserve
  6. generate final tensors
  7. return metrics + transcripts
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.types import MAX_SEEDS
from astar_twin.solver.interfaces import SolverAdapter


@dataclass
class QueryRecord:
    """Single query issued during a solve."""

    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    phase: str  # "bootstrap" | "adaptive" | "reserve"
    utility_score: float = 0.0


@dataclass
class SolveResult:
    """Complete result from one solver run."""

    tensors: list[NDArray[np.float64]]
    transcript: list[QueryRecord] = field(default_factory=list)
    total_queries_used: int = 0
    runtime_seconds: float = 0.0


def solve(adapter: SolverAdapter, round_id: str) -> SolveResult:
    """Run the full solver pipeline against the given round.

    This is a placeholder that will be wired in Task 10.
    Currently returns uniform predictions for all seeds.
    """
    detail = adapter.get_round_detail(round_id)
    height = detail.map_height
    width = detail.map_width

    # Placeholder: uniform predictions (will be replaced in Task 10)
    from astar_twin.solver.baselines import uniform_baseline

    tensors = [uniform_baseline(height, width) for _ in range(len(detail.initial_states))]

    return SolveResult(
        tensors=tensors,
        transcript=[],
        total_queries_used=0,
        runtime_seconds=0.0,
    )
