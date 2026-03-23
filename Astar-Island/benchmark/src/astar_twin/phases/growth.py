from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

from astar_twin.contracts.types import TerrainCode
from astar_twin.params.simulation_params import AdjacencyMode, SimulationParams
from astar_twin.state.grid import Grid
from astar_twin.state.round_state import RoundState
from astar_twin.state.settlement import Settlement


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _adjacent_forest_count(grid: Grid, y: int, x: int, mode: AdjacencyMode) -> int:
    neighbors = (
        grid.moore_neighbors(y, x)
        if mode == AdjacencyMode.MOORE8
        else grid.von_neumann_neighbors(y, x)
    )
    return sum(1 for ny, nx in neighbors if grid.get(ny, nx) == TerrainCode.FOREST)


def _adjacent_settlement_count(
    grid: Grid, settlements: list[Settlement], y: int, x: int, mode: AdjacencyMode
) -> int:
    neighbors = (
        grid.moore_neighbors(y, x)
        if mode == AdjacencyMode.MOORE8
        else grid.von_neumann_neighbors(y, x)
    )
    live_positions = {(s.y, s.x) for s in settlements if s.alive}
    return sum(1 for ny, nx in neighbors if (ny, nx) in live_positions)


def _chebyshev(y1: int, x1: int, y2: int, x2: int) -> float:
    return float(max(abs(y1 - y2), abs(x1 - x2)))


def _euclidean(y1: int, x1: int, y2: int, x2: int) -> float:
    return math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)


def _find_expansion_candidates(
    grid: Grid,
    settlements: list[Settlement],
    parent: Settlement,
    p: SimulationParams,
    rng: Generator,
) -> list[tuple[int, int]]:
    live = [(s.y, s.x) for s in settlements if s.alive]
    candidates: list[tuple[int, int]] = []
    r = p.expansion_radius

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ny, nx = parent.y + dy, parent.x + dx
            if ny < 0 or ny >= grid.height or nx < 0 or nx >= grid.width:
                continue
            code = grid.get(ny, nx)
            if code not in (TerrainCode.EMPTY, TerrainCode.PLAINS, TerrainCode.FOREST):
                continue
            too_close = any(
                max(abs(ny - sy), abs(nx - sx)) < p.expansion_min_spacing for sy, sx in live
            )
            if too_close:
                continue
            candidates.append((ny, nx))

    return candidates


def apply_growth(state: RoundState, p: SimulationParams, rng: Generator) -> RoundState:
    grid = state.grid
    settlements = [s.copy() for s in state.settlements]
    live = [s for s in settlements if s.alive]

    order = list(range(len(live)))
    rng.shuffle(order)

    new_settlements: list[Settlement] = []
    next_owner_id = max((s.owner_id for s in settlements), default=0) + 1

    for idx in order:
        s = live[idx]
        adj_forest = _adjacent_forest_count(grid, s.y, s.x, p.adjacency_mode)
        adj_settlements = _adjacent_settlement_count(grid, live, s.y, s.x, p.adjacency_mode)

        production = (
            p.food_base_yield
            + p.food_per_adjacent_forest * adj_forest
            - p.food_crowding_penalty * adj_settlements
        ) * s.population
        s.food += production - p.population_food_upkeep * s.population

        capacity = (
            p.carrying_capacity_base
            + p.carrying_capacity_per_adjacent_forest * adj_forest
            + (p.carrying_capacity_port_bonus if s.has_port else 0.0)
        )
        if s.food > p.growth_food_buffer:
            food_ratio = min((s.food - p.growth_food_buffer) / (s.population + 1.0), 1.0)
            delta_p = (
                p.population_growth_rate
                * s.population
                * max(0.0, 1.0 - s.population / max(capacity, 1e-6))
                * food_ratio
            )
            s.population = max(0.0, s.population + delta_p)

        prosperity = s.prosperity(p.tech_economic_bonus)

        if not s.has_port and grid.is_coastal(s.y, s.x):
            p_port = _sigmoid((prosperity - p.prosperity_threshold_port) / p.prosperity_logit_scale)
            if rng.random() < p_port:
                s.has_port = True
                grid.set(s.y, s.x, TerrainCode.PORT)

        if s.has_port and not s.has_longship:
            p_longship = _sigmoid(
                (prosperity - p.prosperity_threshold_longship) / p.prosperity_logit_scale
            )
            if rng.random() < p_longship:
                s.has_longship = True

        p_expand = p.expansion_rate * _sigmoid(
            (prosperity - p.prosperity_threshold_expand) / p.prosperity_logit_scale
        )
        if rng.random() < p_expand:
            candidates = _find_expansion_candidates(grid, settlements + new_settlements, s, p, rng)
            if candidates:
                scores = np.array(
                    [
                        -p.expansion_site_distance_decay * _chebyshev(s.y, s.x, cy, cx)
                        + p.expansion_site_coastal_bonus * float(grid.is_coastal(cy, cx))
                        - p.expansion_site_forest_penalty
                        * float(grid.get(cy, cx) == TerrainCode.FOREST)
                        + p.expansion_site_distance_decay
                        * float(_adjacent_forest_count(grid, cy, cx, p.adjacency_mode))
                        * 0.35
                        for cy, cx in candidates
                    ],
                    dtype=np.float64,
                )
                scores = scores - scores.max()
                weights = np.exp(scores / p.expansion_site_temperature)
                weights /= weights.sum()
                chosen_idx = int(rng.choice(len(candidates), p=weights))
                cy, cx = candidates[chosen_idx]

                child = Settlement(
                    x=cx,
                    y=cy,
                    owner_id=s.owner_id,
                    alive=True,
                    has_port=grid.is_coastal(cy, cx) and s.has_port,
                    population=s.population * p.expansion_population_transfer_fraction,
                    food=s.food * p.expansion_food_transfer_fraction,
                    wealth=s.wealth * p.expansion_wealth_transfer_fraction,
                    defense=s.defense * p.expansion_population_transfer_fraction,
                    tech=s.tech,
                )
                s.population *= 1.0 - p.expansion_population_transfer_fraction
                s.food *= 1.0 - p.expansion_food_transfer_fraction
                s.wealth *= 1.0 - p.expansion_wealth_transfer_fraction

                grid.set(cy, cx, TerrainCode.PORT if child.has_port else TerrainCode.SETTLEMENT)
                new_settlements.append(child)
                next_owner_id += 1

    result = state.copy()
    result.grid = grid
    result.settlements = settlements + new_settlements
    return result
