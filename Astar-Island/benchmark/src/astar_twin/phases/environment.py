from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

from astar_twin.contracts.types import TerrainCode
from astar_twin.params.simulation_params import SimulationParams
from astar_twin.state.round_state import RoundState


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def apply_environment(
    state: RoundState,
    p: SimulationParams,
    rng: Generator,
) -> RoundState:
    grid = state.grid.copy()
    settlements = [s.copy() for s in state.settlements]
    live = [s for s in settlements if s.alive]
    dead = [s for s in settlements if not s.alive]

    for ruin in dead:
        ruin.ruin_age += 1

        thriving_patrons = [
            s for s in live if max(abs(s.y - ruin.y), abs(s.x - ruin.x)) <= p.reclaim_radius
        ]

        if thriving_patrons:
            hazard = sum(
                _sigmoid(
                    (s.prosperity(p.tech_economic_bonus) - p.reclaim_threshold)
                    / p.prosperity_logit_scale
                )
                * math.exp(-max(abs(s.y - ruin.y), abs(s.x - ruin.x)) / max(p.reclaim_radius, 1))
                for s in thriving_patrons
            )
            p_reclaim = 1.0 - math.exp(-p.reclaim_rate * hazard)

            if rng.random() < p_reclaim:
                weights = np.array(
                    [
                        _sigmoid(
                            (s.prosperity(p.tech_economic_bonus) - p.reclaim_threshold)
                            / p.prosperity_logit_scale
                        )
                        * math.exp(
                            -max(abs(s.y - ruin.y), abs(s.x - ruin.x)) / max(p.reclaim_radius, 1)
                        )
                        for s in thriving_patrons
                    ],
                    dtype=np.float64,
                )
                weights = np.maximum(weights, 1e-12)
                weights /= weights.sum()
                patron = thriving_patrons[int(rng.choice(len(thriving_patrons), p=weights))]

                ruin.alive = True
                ruin.ruin_age = 0
                ruin.population = patron.population * p.reclaim_inheritance_frac
                ruin.food = patron.food * p.reclaim_inheritance_frac
                ruin.wealth = patron.wealth * p.reclaim_inheritance_frac
                ruin.defense = patron.defense * p.reclaim_inheritance_frac
                ruin.tech = patron.tech * p.reclaim_inheritance_frac
                ruin.owner_id = patron.owner_id
                ruin.has_port = grid.is_coastal(ruin.y, ruin.x) and patron.has_port
                grid.set(
                    ruin.y, ruin.x, TerrainCode.PORT if ruin.has_port else TerrainCode.SETTLEMENT
                )
                continue

        if ruin.ruin_age >= p.ruin_decay_delay:
            adj_forest = sum(
                1
                for dy in range(-1, 2)
                for dx in range(-1, 2)
                if (dy != 0 or dx != 0)
                and 0 <= ruin.y + dy < grid.height
                and 0 <= ruin.x + dx < grid.width
                and grid.get(ruin.y + dy, ruin.x + dx) == TerrainCode.FOREST
            )
            h_forest = p.ruin_forest_rate * (1.0 + adj_forest / 8.0)
            h_plain = p.ruin_plain_rate
            p_any = 1.0 - math.exp(-(h_forest + h_plain))
            if rng.random() < p_any:
                if rng.random() < h_forest / (h_forest + h_plain):
                    grid.set(ruin.y, ruin.x, TerrainCode.FOREST)
                else:
                    grid.set(ruin.y, ruin.x, TerrainCode.PLAINS)
                settlements.remove(ruin)

    result = state.copy()
    result.grid = grid
    result.settlements = settlements
    return result
