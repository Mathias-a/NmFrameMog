from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import MAX_VIEWPORT, NUM_CLASSES
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams
from astar_twin.scoring import compute_score, safe_prediction


def uniform_baseline(height: int, width: int) -> NDArray[np.float64]:
    """Uniform 1/6 probability over all classes — simplest possible baseline."""
    return safe_prediction(
        np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    )


def _generate_grid_viewports(
    height: int,
    width: int,
    n_viewports: int,
) -> list[tuple[int, int, int, int]]:
    """Generate a grid of viewport positions for coverage.

    Returns list of (x, y, w, h) tuples.
    """
    if n_viewports <= 0:
        return []

    viewport_h = min(MAX_VIEWPORT, height)
    viewport_w = min(MAX_VIEWPORT, width)
    aspect_ratio = width / height if height > 0 else 1.0
    n_cols = max(1, ceil(sqrt(n_viewports * aspect_ratio)))
    n_rows = max(1, ceil(n_viewports / n_cols))

    max_x = max(0, width - viewport_w)
    max_y = max(0, height - viewport_h)
    x_positions = np.linspace(0, max_x, num=n_cols, dtype=int)
    y_positions = np.linspace(0, max_y, num=n_rows, dtype=int)

    viewports: list[tuple[int, int, int, int]] = []
    for y in y_positions:
        for x in x_positions:
            viewports.append((int(x), int(y), viewport_w, viewport_h))
            if len(viewports) == n_viewports:
                return viewports

    return viewports


def fixed_coverage_baseline(
    initial_states: list[InitialState],
    height: int,
    width: int,
    n_mc_runs: int = 200,
    base_seed: int = 42,
    queries_per_seed: int = 10,
) -> list[NDArray[np.float64]]:
    """Return fixed-viewport sweep predictions for each seed."""
    simulator = Simulator(SimulationParams())
    mc_runner = MCRunner(simulator)
    tensors: list[NDArray[np.float64]] = []
    for initial_state in initial_states:
        runs = mc_runner.run_batch(initial_state, n_runs=n_mc_runs, base_seed=base_seed)
        raw = aggregate_runs(runs, height, width)
        combined = np.full(
            (height, width, NUM_CLASSES),
            1.0 / NUM_CLASSES,
            dtype=np.float64,
        )
        for x, y, viewport_w, viewport_h in _generate_grid_viewports(
            height,
            width,
            queries_per_seed,
        ):
            combined[y : y + viewport_h, x : x + viewport_w, :] = raw[
                y : y + viewport_h,
                x : x + viewport_w,
                :,
            ]
        tensors.append(safe_prediction(combined))
    return tensors


@dataclass
class BaselineSummary:
    """Machine-readable summary of baseline scores for downstream comparison."""

    uniform_scores: list[float]
    fixed_coverage_scores: list[float]
    uniform_mean: float
    fixed_coverage_mean: float


def compute_baseline_summary(
    initial_states: list[InitialState],
    ground_truths: list[NDArray[np.float64]],
    height: int,
    width: int,
    n_mc_runs: int = 200,
    base_seed: int = 42,
) -> BaselineSummary:
    """Score both baselines against ground truths and return summary."""
    uniform = uniform_baseline(height, width)
    fc_tensors = fixed_coverage_baseline(initial_states, height, width, n_mc_runs, base_seed)

    uniform_scores = [float(compute_score(gt, uniform)) for gt in ground_truths]
    fc_scores = [float(compute_score(gt, t)) for gt, t in zip(ground_truths, fc_tensors)]

    return BaselineSummary(
        uniform_scores=uniform_scores,
        fixed_coverage_scores=fc_scores,
        uniform_mean=float(np.mean(uniform_scores)),
        fixed_coverage_mean=float(np.mean(fc_scores)),
    )
