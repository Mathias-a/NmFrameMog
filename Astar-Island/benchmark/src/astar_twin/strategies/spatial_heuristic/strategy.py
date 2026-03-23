from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.harness.budget import Budget

# ---------------------------------------------------------------------------
# Tuning knobs — derived from simulation mechanics and default params.
# All values are heuristic priors, NOT fitted to any specific round.
# ---------------------------------------------------------------------------

# How far from an existing settlement can expansion reach?
_EXPANSION_RADIUS: int = 3

# Settlement survival parameters
_BASE_SURVIVAL: float = 0.50
_FOREST_SURVIVAL_BONUS: float = 0.07  # per adjacent forest
_MAX_SURVIVAL: float = 0.88
_COASTAL_PORT_SHARE: float = 0.40  # fraction of survival prob allocated to Port class
_COLLAPSE_RUIN_SHARE: float = 0.65  # of (1 - survival), how much becomes Ruin vs Empty
_SETTLEMENT_FOREST_PROB: float = 0.02  # tiny chance settlement cell ends as Forest
_COMPETITOR_PENALTY: float = 0.04  # per nearby settlement (within radius 3)

# Expansion parameters for empty/plains cells
_EXPANSION_PEAK: float = 0.22  # max expansion probability (at distance 1)
_EXPANSION_PORT_BONUS: float = 0.08  # additional port probability if coastal
_EXPANSION_RUIN_FRAC: float = 0.12  # fraction of settlement prob that becomes ruin instead

# Forest cell parameters
_FOREST_BASE_STABILITY: float = 0.88
_FOREST_CLEARING_NEAR_SETTLEMENT: float = 0.08  # chance of clearing within 2 cells
_FOREST_EMPTY_RESIDUAL: float = 0.04

# Ruin cell parameters
_RUIN_RECLAIM_PROB: float = 0.25  # chance ruin becomes settlement/port again
_RUIN_FOREST_PROB: float = 0.20  # chance ruin becomes forest
_RUIN_EMPTY_PROB: float = 0.35  # chance ruin reverts to empty
_RUIN_STAYS_PROB: float = 0.20  # chance ruin is still a ruin after 50 years


class SpatialHeuristicStrategy:
    """Topology-aware heuristic using spatial context for per-cell templates.

    Improves on ``filter_baseline`` by treating each cell individually based on
    its initial terrain type **and** its spatial neighbourhood:

    - **Settlement cells**: survival probability depends on adjacent forests
      (food quality), coastal access (trade/port), and nearby settlement density
      (competition and raiding pressure).
    - **Empty/Plains cells**: expansion probability depends on distance to the
      nearest settlement and coastal adjacency.
    - **Forest cells**: mostly stable, with a small clearing probability near
      expanding settlements.
    - **Ruin cells**: modelled with reclaim, overgrowth, and reversion
      probabilities.
    - **Static cells** (Ocean, Mountain): deterministic, as in all strategies.

    This strategy uses **no simulation** — it is a pure analysis of the initial
    grid topology.  It is deterministic regardless of ``base_seed``.
    """

    @property
    def name(self) -> str:
        return "spatial_heuristic"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        del budget, base_seed  # deterministic heuristic — not used

        grid = initial_state.grid
        H = len(grid)
        W = len(grid[0])
        tensor = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)

        # Pre-compute settlement positions for distance lookups.
        settlement_positions: list[tuple[int, int]] = []
        for s in initial_state.settlements:
            if s.alive:
                settlement_positions.append((s.x, s.y))

        for y in range(H):
            for x in range(W):
                code = grid[y][x]
                tensor[y, x] = self._cell_distribution(
                    grid,
                    x,
                    y,
                    code,
                    settlement_positions,
                    H,
                    W,
                )

        return tensor

    # ------------------------------------------------------------------
    # Per-cell distribution logic
    # ------------------------------------------------------------------

    def _cell_distribution(
        self,
        grid: list[list[int]],
        x: int,
        y: int,
        code: int,
        settlement_positions: list[tuple[int, int]],
        H: int,
        W: int,
    ) -> NDArray[np.float64]:
        """Return a length-6 probability vector for this cell."""

        # --- static cells ---
        if code == TerrainCode.OCEAN:
            return _one_hot(ClassIndex.EMPTY)
        if code == TerrainCode.MOUNTAIN:
            return _one_hot(ClassIndex.MOUNTAIN)

        # --- spatial features ---
        n_forests = self._count_adjacent(grid, x, y, H, W, TerrainCode.FOREST)
        coastal = self._is_coastal(grid, x, y, H, W)
        dist_nearest = self._dist_nearest_settlement(x, y, settlement_positions)
        n_nearby = self._count_nearby_settlements(x, y, settlement_positions)

        # --- settlement / port cell ---
        if code in (TerrainCode.SETTLEMENT, TerrainCode.PORT):
            return self._settlement_template(n_forests, coastal, n_nearby)

        # --- ruin cell ---
        if code == TerrainCode.RUIN:
            return self._ruin_template(dist_nearest, coastal)

        # --- forest cell ---
        if code == TerrainCode.FOREST:
            return self._forest_template(dist_nearest)

        # --- empty / plains (buildable land) ---
        return self._empty_template(dist_nearest, coastal)

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def _settlement_template(
        self,
        n_forests: int,
        coastal: bool,
        n_nearby: int,
    ) -> NDArray[np.float64]:
        survival = min(
            _MAX_SURVIVAL,
            _BASE_SURVIVAL
            + _FOREST_SURVIVAL_BONUS * n_forests
            - _COMPETITOR_PENALTY * max(0, n_nearby - 1),
        )
        survival = max(0.15, survival)  # floor — settlements rarely all die
        collapse = 1.0 - survival

        dist = np.zeros(NUM_CLASSES, dtype=np.float64)
        if coastal:
            dist[ClassIndex.PORT] = survival * _COASTAL_PORT_SHARE
            dist[ClassIndex.SETTLEMENT] = survival * (1.0 - _COASTAL_PORT_SHARE)
        else:
            dist[ClassIndex.SETTLEMENT] = survival
            dist[ClassIndex.PORT] = 0.0

        dist[ClassIndex.RUIN] = collapse * _COLLAPSE_RUIN_SHARE
        dist[ClassIndex.FOREST] = _SETTLEMENT_FOREST_PROB
        dist[ClassIndex.EMPTY] = collapse * (1.0 - _COLLAPSE_RUIN_SHARE)
        dist[ClassIndex.MOUNTAIN] = 0.0
        return _normalise(dist)

    def _empty_template(
        self,
        dist_nearest: float,
        coastal: bool,
    ) -> NDArray[np.float64]:
        dist = np.zeros(NUM_CLASSES, dtype=np.float64)

        if dist_nearest <= _EXPANSION_RADIUS:
            # Probability of becoming a settlement decays with distance.
            t = max(0.0, (_EXPANSION_RADIUS + 1.0 - dist_nearest) / (_EXPANSION_RADIUS + 1.0))
            p_settle = _EXPANSION_PEAK * t
            p_port = (_EXPANSION_PORT_BONUS * t) if coastal else 0.0
            p_ruin = p_settle * _EXPANSION_RUIN_FRAC

            dist[ClassIndex.SETTLEMENT] = p_settle
            dist[ClassIndex.PORT] = p_port
            dist[ClassIndex.RUIN] = p_ruin
            dist[ClassIndex.FOREST] = 0.02
            dist[ClassIndex.EMPTY] = 1.0 - p_settle - p_port - p_ruin - 0.02
        else:
            # Far from any settlement — very likely stays empty.
            dist[ClassIndex.EMPTY] = 0.92
            dist[ClassIndex.SETTLEMENT] = 0.03
            dist[ClassIndex.PORT] = 0.01 if coastal else 0.0
            dist[ClassIndex.RUIN] = 0.01
            dist[ClassIndex.FOREST] = 0.03

        dist[ClassIndex.MOUNTAIN] = 0.0
        return _normalise(dist)

    def _forest_template(self, dist_nearest: float) -> NDArray[np.float64]:
        dist = np.zeros(NUM_CLASSES, dtype=np.float64)

        if dist_nearest <= 2.0:
            # Near settlement — small chance of clearing for expansion.
            p_clear = _FOREST_CLEARING_NEAR_SETTLEMENT
            dist[ClassIndex.FOREST] = _FOREST_BASE_STABILITY - p_clear
            dist[ClassIndex.SETTLEMENT] = p_clear * 0.7
            dist[ClassIndex.PORT] = p_clear * 0.1
            dist[ClassIndex.RUIN] = p_clear * 0.05
            dist[ClassIndex.EMPTY] = _FOREST_EMPTY_RESIDUAL + p_clear * 0.15
        else:
            dist[ClassIndex.FOREST] = _FOREST_BASE_STABILITY
            dist[ClassIndex.EMPTY] = _FOREST_EMPTY_RESIDUAL
            dist[ClassIndex.SETTLEMENT] = 0.03
            dist[ClassIndex.PORT] = 0.01
            dist[ClassIndex.RUIN] = 0.01

        dist[ClassIndex.MOUNTAIN] = 0.0
        return _normalise(dist)

    def _ruin_template(
        self,
        dist_nearest: float,
        coastal: bool,
    ) -> NDArray[np.float64]:
        dist = np.zeros(NUM_CLASSES, dtype=np.float64)

        if dist_nearest <= 4.0:
            # Near a thriving settlement — higher reclaim chance.
            t = max(0.0, (5.0 - dist_nearest) / 5.0)
            reclaim = _RUIN_RECLAIM_PROB * t
            port_share = 0.35 if coastal else 0.0
            dist[ClassIndex.SETTLEMENT] = reclaim * (1.0 - port_share)
            dist[ClassIndex.PORT] = reclaim * port_share
        else:
            dist[ClassIndex.SETTLEMENT] = 0.02
            dist[ClassIndex.PORT] = 0.01 if coastal else 0.0

        dist[ClassIndex.RUIN] = _RUIN_STAYS_PROB
        dist[ClassIndex.FOREST] = _RUIN_FOREST_PROB
        dist[ClassIndex.EMPTY] = _RUIN_EMPTY_PROB
        dist[ClassIndex.MOUNTAIN] = 0.0
        return _normalise(dist)

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_coastal(grid: list[list[int]], x: int, y: int, H: int, W: int) -> bool:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == TerrainCode.OCEAN:
                return True
        return False

    @staticmethod
    def _count_adjacent(
        grid: list[list[int]],
        x: int,
        y: int,
        H: int,
        W: int,
        target: int,
    ) -> int:
        """Count 8-neighbours matching *target* terrain code."""
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == target:
                    count += 1
        return count

    @staticmethod
    def _dist_nearest_settlement(
        x: int,
        y: int,
        positions: list[tuple[int, int]],
    ) -> float:
        if not positions:
            return float("inf")
        return min(math.sqrt((sx - x) ** 2 + (sy - y) ** 2) for sx, sy in positions)

    @staticmethod
    def _count_nearby_settlements(
        x: int,
        y: int,
        positions: list[tuple[int, int]],
        radius: int = 3,
    ) -> int:
        return sum(1 for sx, sy in positions if max(abs(sx - x), abs(sy - y)) <= radius)


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def _one_hot(class_index: int) -> NDArray[np.float64]:
    v = np.zeros(NUM_CLASSES, dtype=np.float64)
    v[class_index] = 1.0
    return v


def _normalise(dist: NDArray[np.float64]) -> NDArray[np.float64]:
    total = dist.sum()
    if total > 0:
        dist /= total
    else:
        dist[:] = 1.0 / NUM_CLASSES
    return dist
