from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams
from astar_twin.scoring import compute_score, safe_prediction


def uniform_baseline(height: int, width: int) -> NDArray[np.float64]:
    """Uniform 1/6 probability over all classes — simplest possible baseline."""
    return safe_prediction(
        np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    )


def fixed_coverage_baseline(
    initial_states: list[InitialState],
    height: int,
    width: int,
    n_mc_runs: int = 200,
    base_seed: int = 42,
) -> list[NDArray[np.float64]]:
    """Default-parameter full-map MC prediction for each seed.

    Uses SimulationParams() defaults — no hidden parameter knowledge.
    No adaptive viewport queries — just full-map prediction.
    """
    simulator = Simulator(SimulationParams())
    mc_runner = MCRunner(simulator)
    tensors: list[NDArray[np.float64]] = []
    for initial_state in initial_states:
        runs = mc_runner.run_batch(initial_state, n_runs=n_mc_runs, base_seed=base_seed)
        raw = aggregate_runs(runs, height, width)
        tensors.append(safe_prediction(raw))
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
