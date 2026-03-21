"""Terrain-aware MC strategy with dynamism-weighted prior blending.

Translates the "hotspot redesign" avenue idea: instead of treating all cells
equally, classify each cell by how likely it is to change during simulation
and allocate prediction effort accordingly.

Static cells (ocean, mountain) get near-deterministic priors.
Semi-static cells (forest far from settlements) get blended prior + MC.
Dynamic cells (near settlements, coastal, frontiers) get heavy MC weighting.

This mirrors the real query-allocation idea: spend more observation budget
on regions with high expected information gain.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import (
    NUM_CLASSES,
    ClassIndex,
    TerrainCode,
)
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams

# --- Cell dynamism classification ---
# Weight = how much to trust MC vs initial-state prior.
# 1.0 = pure MC, 0.0 = pure prior.
_W_STATIC = 0.0  # ocean, mountain: never change
_W_FOREST_ISOLATED = 0.15  # forest cells far from any settlement
_W_FOREST_FRONTIER = 0.55  # forest cells near settlements (can be cleared for expansion)
_W_PLAINS_ISOLATED = 0.20  # plains far from settlements (low expansion chance)
_W_PLAINS_FRONTIER = 0.65  # plains near settlements (expansion target)
_W_SETTLEMENT = 0.85  # active settlements (can die → ruin, grow → port)
_W_PORT = 0.80  # ports (can die → ruin)
_W_RUIN = 0.70  # ruins (can be reclaimed or become forest/plains)

# Settlement proximity radius for "frontier" classification
_FRONTIER_RADIUS = 5

# MC run scaling: base + per-remaining-budget
_MC_BASE_RUNS = 20
_MC_PER_BUDGET = 3


class TerrainAwareMCStrategy:
    """MC strategy that weights predictions by per-cell terrain dynamism.

    Combines Monte Carlo simulation output with an initial-state prior,
    where the blending weight per cell depends on how likely that cell
    is to change during the 50-year simulation.

    This strategy implements the hotspot targeting idea from the query
    allocation avenue: static terrain gets cheap deterministic priors,
    while dynamic regions near settlements get the full MC treatment.
    """

    def __init__(self, params: SimulationParams | None = None) -> None:
        self._params = params or SimulationParams()

    @property
    def name(self) -> str:
        return "terrain_aware_mc"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        grid = initial_state.grid
        H = len(grid)
        W = len(grid[0])

        # 1. Build dynamism weight map
        dynamism = self._compute_dynamism_map(initial_state, H, W)

        # 2. Build initial-state prior tensor
        prior = self._build_prior(initial_state, H, W)

        # 3. Run MC simulations
        n_runs = _MC_BASE_RUNS + budget.remaining * _MC_PER_BUDGET
        simulator = Simulator(params=self._params)
        mc_runner = MCRunner(simulator)
        runs = mc_runner.run_batch(
            initial_state=initial_state,
            n_runs=n_runs,
            base_seed=base_seed,
        )
        mc_tensor = aggregate_runs(runs, H, W)

        # 4. Blend: result = dynamism * MC + (1 - dynamism) * prior
        dynamism_3d = dynamism[:, :, np.newaxis]  # (H, W, 1) for broadcasting
        result = dynamism_3d * mc_tensor + (1.0 - dynamism_3d) * prior

        return result

    def _compute_dynamism_map(
        self,
        initial_state: InitialState,
        H: int,
        W: int,
    ) -> NDArray[np.float64]:
        """Classify each cell's expected dynamism level.

        Returns H x W array of blend weights in [0, 1].
        """
        grid = initial_state.grid
        dynamism = np.zeros((H, W), dtype=np.float64)

        # Find alive settlement positions for proximity calculation
        settlement_positions: list[tuple[int, int]] = []
        for s in initial_state.settlements:
            if s.alive:
                settlement_positions.append((s.x, s.y))

        # Build distance-to-nearest-settlement map using Chebyshev distance
        if settlement_positions:
            dist_map = self._settlement_distance_map(settlement_positions, H, W)
        else:
            dist_map = np.full((H, W), float(_FRONTIER_RADIUS + 1), dtype=np.float64)

        for y in range(H):
            for x in range(W):
                code = grid[y][x]
                near_settlement = dist_map[y, x] <= _FRONTIER_RADIUS

                if code == TerrainCode.OCEAN or code == TerrainCode.MOUNTAIN:
                    dynamism[y, x] = _W_STATIC
                elif code == TerrainCode.SETTLEMENT:
                    dynamism[y, x] = _W_SETTLEMENT
                elif code == TerrainCode.PORT:
                    dynamism[y, x] = _W_PORT
                elif code == TerrainCode.RUIN:
                    dynamism[y, x] = _W_RUIN
                elif code == TerrainCode.FOREST:
                    dynamism[y, x] = _W_FOREST_FRONTIER if near_settlement else _W_FOREST_ISOLATED
                else:
                    # Plains or Empty
                    dynamism[y, x] = _W_PLAINS_FRONTIER if near_settlement else _W_PLAINS_ISOLATED

        return dynamism

    def _settlement_distance_map(
        self,
        positions: list[tuple[int, int]],
        H: int,
        W: int,
    ) -> NDArray[np.float64]:
        """Compute Chebyshev distance from each cell to nearest settlement."""
        dist = np.full((H, W), float(H + W), dtype=np.float64)
        for sx, sy in positions:
            for y in range(H):
                for x in range(W):
                    d = max(abs(x - sx), abs(y - sy))
                    if d < dist[y, x]:
                        dist[y, x] = d
        return dist

    def _build_prior(
        self,
        initial_state: InitialState,
        H: int,
        W: int,
    ) -> NDArray[np.float64]:
        """Build a strong prior from initial terrain state.

        Ocean → Empty (class 0) at 0.96
        Mountain → Mountain (class 5) at 0.96
        Settlement → spread: Settlement 0.50, Ruin 0.20, Empty 0.15, Port 0.10
        Port → spread: Port 0.45, Settlement 0.15, Ruin 0.20, Empty 0.15
        Ruin → spread: Ruin 0.30, Forest 0.25, Empty 0.25, Settlement 0.10
        Forest → spread: Forest 0.70, Empty 0.15, Settlement 0.08
        Plains/Empty → spread: Empty 0.65, Settlement 0.12, Forest 0.10
        """
        grid = initial_state.grid
        tensor = np.full((H, W, NUM_CLASSES), 0.01, dtype=np.float64)

        # Check which settlements have ports for coastal detection
        coastal_cells = self._find_coastal_cells(grid, H, W)

        for y in range(H):
            for x in range(W):
                code = grid[y][x]
                is_coastal = (x, y) in coastal_cells

                if code == TerrainCode.OCEAN:
                    tensor[y, x, ClassIndex.EMPTY] = 0.96
                elif code == TerrainCode.MOUNTAIN:
                    tensor[y, x, ClassIndex.MOUNTAIN] = 0.96
                elif code == TerrainCode.SETTLEMENT:
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.50
                    tensor[y, x, ClassIndex.RUIN] = 0.20
                    tensor[y, x, ClassIndex.EMPTY] = 0.12
                    tensor[y, x, ClassIndex.PORT] = 0.12 if is_coastal else 0.03
                    tensor[y, x, ClassIndex.FOREST] = 0.03
                elif code == TerrainCode.PORT:
                    tensor[y, x, ClassIndex.PORT] = 0.45
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.15
                    tensor[y, x, ClassIndex.RUIN] = 0.20
                    tensor[y, x, ClassIndex.EMPTY] = 0.12
                    tensor[y, x, ClassIndex.FOREST] = 0.03
                elif code == TerrainCode.RUIN:
                    tensor[y, x, ClassIndex.RUIN] = 0.30
                    tensor[y, x, ClassIndex.FOREST] = 0.25
                    tensor[y, x, ClassIndex.EMPTY] = 0.25
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.10
                    tensor[y, x, ClassIndex.PORT] = 0.05 if is_coastal else 0.01
                elif code == TerrainCode.FOREST:
                    tensor[y, x, ClassIndex.FOREST] = 0.70
                    tensor[y, x, ClassIndex.EMPTY] = 0.15
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.08
                    tensor[y, x, ClassIndex.PORT] = 0.03 if is_coastal else 0.01
                else:
                    # Plains or Empty
                    tensor[y, x, ClassIndex.EMPTY] = 0.65
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.12
                    tensor[y, x, ClassIndex.FOREST] = 0.10
                    tensor[y, x, ClassIndex.PORT] = 0.05 if is_coastal else 0.01
                    tensor[y, x, ClassIndex.RUIN] = 0.04

        # Normalize each cell's prior to sum to 1
        sums = tensor.sum(axis=2, keepdims=True)
        tensor = tensor / np.maximum(sums, 1e-10)

        return tensor

    def _find_coastal_cells(
        self,
        grid: list[list[int]],
        H: int,
        W: int,
    ) -> set[tuple[int, int]]:
        """Find all land cells adjacent to ocean."""
        coastal: set[tuple[int, int]] = set()
        for y in range(H):
            for x in range(W):
                if grid[y][x] == TerrainCode.OCEAN:
                    continue
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == TerrainCode.OCEAN:
                        coastal.add((x, y))
                        break
        return coastal
