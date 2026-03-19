"""Ensemble prediction via Monte Carlo simulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_island.prediction import (
    DEFAULT_FLOOR,
    PredictionTensor,
    apply_probability_floor,
)
from astar_island.terrain import NUM_PREDICTION_CLASSES, TERRAIN_TO_CLASS

from .engine import run_simulation
from .models import SimConfig, WorldState


def world_state_to_class_grid(state: WorldState) -> NDArray[np.int32]:
    """Convert WorldState grid to prediction class indices (0-5)."""
    result: NDArray[np.int32] = np.zeros((state.width, state.height), dtype=np.int32)
    for y in range(state.height):
        for x in range(state.width):
            result[x, y] = int(TERRAIN_TO_CLASS[state.grid[y][x]])
    return result


def run_ensemble(
    seed: int,
    config: SimConfig | None = None,
    n_runs: int = 100,
    floor: float = DEFAULT_FLOOR,
    width: int = 40,
    height: int = 40,
) -> PredictionTensor:
    """Run an ensemble of simulations and produce a prediction tensor.

    Args:
        seed: Base seed for map generation (same map every run).
        config: Simulation parameters. Uses defaults if None.
        n_runs: Number of Monte Carlo runs.
        floor: Minimum probability per class (avoids KL divergence blowup).
        width: Map width.
        height: Map height.

    Returns:
        Prediction tensor of shape (width, height, NUM_PREDICTION_CLASSES).
    """
    cfg = config or SimConfig()
    counts: NDArray[np.int64] = np.zeros(
        (width, height, NUM_PREDICTION_CLASSES), dtype=np.int64
    )

    for run_idx in range(n_runs):
        state = run_simulation(
            seed,
            cfg,
            stochastic_seed=seed * 10000 + run_idx,
            width=width,
            height=height,
        )
        class_grid = world_state_to_class_grid(state)
        rows, cols = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
        np.add.at(counts, (rows, cols, class_grid), 1)

    pred: PredictionTensor = counts.astype(np.float64) / n_runs
    return apply_probability_floor(pred, floor=floor)
