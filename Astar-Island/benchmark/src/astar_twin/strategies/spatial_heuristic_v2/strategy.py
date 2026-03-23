from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.harness.budget import Budget
from astar_twin.strategies.filter_baseline.strategy import FilterBaselineStrategy

# ---------------------------------------------------------------------------
# Tuned knobs for the high-value settlement halo.
# ---------------------------------------------------------------------------

_EXPANSION_RADIUS: int = 4

_BASE_SURVIVAL: float = 0.58
_FOREST_SURVIVAL_BONUS: float = 0.07
_MAX_SURVIVAL: float = 0.90
_COASTAL_PORT_SHARE: float = 0.40
_HAS_PORT_BONUS: float = 0.15
_COLLAPSE_RUIN_SHARE: float = 0.50
_SETTLEMENT_FOREST_PROB: float = 0.02
_COMPETITOR_PENALTY: float = 0.02

_EXPANSION_PEAK: float = 0.28
_EXPANSION_PORT_BONUS: float = 0.08
_EXPANSION_RUIN_FRAC: float = 0.08
_NEAR_FOREST_MASS: float = 0.03

_FOREST_BASE_STABILITY: float = 0.88
_FOREST_CLEARING_NEAR_SETTLEMENT: float = 0.12
_FOREST_CLEARING_MID_SETTLEMENT: float = 0.06
_FOREST_EMPTY_RESIDUAL: float = 0.04

_RUIN_RECLAIM_PROB: float = 0.25
_RUIN_FOREST_PROB: float = 0.20
_RUIN_EMPTY_PROB: float = 0.35
_RUIN_STAYS_PROB: float = 0.20


class SpatialHeuristicV2Strategy:
    """Improved spatial heuristic with a tuned settlement halo and local hedge.

    v2 keeps the topology-first structure of ``spatial_heuristic`` but adjusts the
    high-value regions identified by benchmarking:

    - stronger settlement survival on origin cells
    - wider settlement influence on nearby plains/forest cells
    - less ruin-heavy collapse
    - selective hedging with ``filter_baseline`` on ambiguous land cells near
      settlements to reduce catastrophic round-level miscalibration
    """

    def __init__(self, hedge_weight: float = 0.40) -> None:
        self._hedge_weight = hedge_weight
        self._filter_baseline = FilterBaselineStrategy()

    @property
    def name(self) -> str:
        return "spatial_heuristic_v2"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        del base_seed

        grid = initial_state.grid
        height = len(grid)
        width = len(grid[0])
        tensor = np.zeros((height, width, NUM_CLASSES), dtype=np.float64)
        baseline_tensor = self._filter_baseline.predict(initial_state, budget=budget, base_seed=0)

        settlement_positions: list[tuple[int, int]] = []
        settlement_has_port: dict[tuple[int, int], bool] = {}
        for settlement in initial_state.settlements:
            if settlement.alive:
                pos = (settlement.x, settlement.y)
                settlement_positions.append(pos)
                settlement_has_port[pos] = settlement.has_port

        for y in range(height):
            for x in range(width):
                code = grid[y][x]
                tuned = self._cell_distribution(
                    grid=grid,
                    x=x,
                    y=y,
                    code=code,
                    settlement_positions=settlement_positions,
                    settlement_has_port=settlement_has_port,
                    height=height,
                    width=width,
                )
                alpha = self._tuned_weight(
                    code=code,
                    dist_nearest=self._dist_nearest_settlement(x, y, settlement_positions),
                )
                if alpha >= 1.0:
                    tensor[y, x] = tuned
                else:
                    baseline_cell = cast(NDArray[np.float64], baseline_tensor[y, x])
                    tensor[y, x] = _normalise(
                        cast(
                            NDArray[np.float64],
                            alpha * tuned + (1.0 - alpha) * baseline_cell,
                        )
                    )

        return tensor

    def _cell_distribution(
        self,
        grid: list[list[int]],
        x: int,
        y: int,
        code: int,
        settlement_positions: list[tuple[int, int]],
        settlement_has_port: dict[tuple[int, int], bool],
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        if code == TerrainCode.OCEAN:
            return _one_hot(ClassIndex.EMPTY)
        if code == TerrainCode.MOUNTAIN:
            return _one_hot(ClassIndex.MOUNTAIN)

        n_forests = self._count_adjacent(grid, x, y, height, width, TerrainCode.FOREST)
        coastal = self._is_coastal(grid, x, y, height, width)
        dist_nearest = self._dist_nearest_settlement(x, y, settlement_positions)
        n_nearby = self._count_nearby_settlements(x, y, settlement_positions)
        has_port_origin = settlement_has_port.get((x, y), code == TerrainCode.PORT)

        if code in (TerrainCode.SETTLEMENT, TerrainCode.PORT):
            return self._settlement_template(
                n_forests=n_forests,
                coastal=coastal,
                n_nearby=n_nearby,
                has_port_origin=has_port_origin,
            )
        if code == TerrainCode.RUIN:
            return self._ruin_template(dist_nearest=dist_nearest, coastal=coastal)
        if code == TerrainCode.FOREST:
            return self._forest_template(dist_nearest=dist_nearest, coastal=coastal)
        return self._empty_template(dist_nearest=dist_nearest, coastal=coastal)

    def _settlement_template(
        self,
        n_forests: int,
        coastal: bool,
        n_nearby: int,
        has_port_origin: bool,
    ) -> NDArray[np.float64]:
        survival = min(
            _MAX_SURVIVAL,
            _BASE_SURVIVAL
            + _FOREST_SURVIVAL_BONUS * n_forests
            - _COMPETITOR_PENALTY * max(0, n_nearby - 1),
        )
        survival = max(0.20, survival)
        collapse = 1.0 - survival

        dist = np.zeros(NUM_CLASSES, dtype=np.float64)
        if coastal:
            port_share = _COASTAL_PORT_SHARE + (_HAS_PORT_BONUS if has_port_origin else 0.0)
            port_share = min(port_share, 0.75)
            dist[ClassIndex.PORT] = survival * port_share
            dist[ClassIndex.SETTLEMENT] = survival * (1.0 - port_share)
        else:
            dist[ClassIndex.SETTLEMENT] = survival

        dist[ClassIndex.RUIN] = collapse * _COLLAPSE_RUIN_SHARE
        dist[ClassIndex.FOREST] = _SETTLEMENT_FOREST_PROB
        dist[ClassIndex.EMPTY] = collapse * (1.0 - _COLLAPSE_RUIN_SHARE)
        return _normalise(dist)

    def _empty_template(self, dist_nearest: float, coastal: bool) -> NDArray[np.float64]:
        dist = np.zeros(NUM_CLASSES, dtype=np.float64)

        if dist_nearest <= _EXPANSION_RADIUS:
            t = max(0.0, (_EXPANSION_RADIUS + 1.0 - dist_nearest) / (_EXPANSION_RADIUS + 1.0))
            p_settle = _EXPANSION_PEAK * t
            p_port = (_EXPANSION_PORT_BONUS * t) if coastal else 0.0
            p_ruin = p_settle * _EXPANSION_RUIN_FRAC
            p_forest = _NEAR_FOREST_MASS

            dist[ClassIndex.SETTLEMENT] = p_settle
            dist[ClassIndex.PORT] = p_port
            dist[ClassIndex.RUIN] = p_ruin
            dist[ClassIndex.FOREST] = p_forest
            dist[ClassIndex.EMPTY] = 1.0 - p_settle - p_port - p_ruin - p_forest
        else:
            dist[ClassIndex.EMPTY] = 0.92
            dist[ClassIndex.SETTLEMENT] = 0.03
            dist[ClassIndex.PORT] = 0.01 if coastal else 0.0
            dist[ClassIndex.RUIN] = 0.01
            dist[ClassIndex.FOREST] = 0.03

        return _normalise(dist)

    def _forest_template(self, dist_nearest: float, coastal: bool) -> NDArray[np.float64]:
        dist = np.zeros(NUM_CLASSES, dtype=np.float64)

        if dist_nearest <= 2.0:
            p_clear = _FOREST_CLEARING_NEAR_SETTLEMENT
        elif dist_nearest <= 5.0:
            p_clear = _FOREST_CLEARING_MID_SETTLEMENT
        else:
            p_clear = 0.0

        if p_clear > 0.0:
            dist[ClassIndex.FOREST] = _FOREST_BASE_STABILITY - p_clear
            dist[ClassIndex.SETTLEMENT] = p_clear * 0.72
            dist[ClassIndex.PORT] = p_clear * (0.12 if coastal else 0.05)
            dist[ClassIndex.RUIN] = p_clear * 0.05
            dist[ClassIndex.EMPTY] = _FOREST_EMPTY_RESIDUAL + p_clear * (0.11 if coastal else 0.18)
        else:
            dist[ClassIndex.FOREST] = _FOREST_BASE_STABILITY
            dist[ClassIndex.EMPTY] = _FOREST_EMPTY_RESIDUAL
            dist[ClassIndex.SETTLEMENT] = 0.03
            dist[ClassIndex.PORT] = 0.01 if coastal else 0.0
            dist[ClassIndex.RUIN] = 0.01

        return _normalise(dist)

    def _ruin_template(self, dist_nearest: float, coastal: bool) -> NDArray[np.float64]:
        dist = np.zeros(NUM_CLASSES, dtype=np.float64)

        if dist_nearest <= 4.0:
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
        return _normalise(dist)

    def _tuned_weight(self, code: int, dist_nearest: float) -> float:
        if code in (
            TerrainCode.OCEAN,
            TerrainCode.MOUNTAIN,
            TerrainCode.SETTLEMENT,
            TerrainCode.PORT,
        ):
            return 1.0
        if dist_nearest <= 2.0:
            return 1.0 - self._hedge_weight
        if dist_nearest <= 5.0:
            return 1.0 - self._hedge_weight * 0.75
        return 1.0

    @staticmethod
    def _is_coastal(grid: list[list[int]], x: int, y: int, height: int, width: int) -> bool:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == TerrainCode.OCEAN:
                return True
        return False

    @staticmethod
    def _count_adjacent(
        grid: list[list[int]],
        x: int,
        y: int,
        height: int,
        width: int,
        target: int,
    ) -> int:
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == target:
                    count += 1
        return count

    @staticmethod
    def _dist_nearest_settlement(x: int, y: int, positions: list[tuple[int, int]]) -> float:
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


def _one_hot(class_index: int) -> NDArray[np.float64]:
    vector = np.zeros(NUM_CLASSES, dtype=np.float64)
    vector[class_index] = 1.0
    return vector


def _normalise(dist: NDArray[np.float64]) -> NDArray[np.float64]:
    total = float(cast(np.float64, dist.sum()))
    if total > 0.0:
        dist /= total
    else:
        dist[:] = 1.0 / NUM_CLASSES
    return dist
