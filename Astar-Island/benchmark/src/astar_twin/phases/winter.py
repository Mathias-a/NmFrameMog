from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

from astar_twin.contracts.types import TerrainCode
from astar_twin.params.simulation_params import SimulationParams
from astar_twin.state.round_state import RoundState
from astar_twin.state.settlement import Settlement


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def apply_winter(
    state: RoundState,
    p: SimulationParams,
    rng: Generator,
    prev_severity: float,
) -> tuple[RoundState, float]:
    alpha = p.winter_severity_mean * p.winter_severity_concentration
    beta_param = (1.0 - p.winter_severity_mean) * p.winter_severity_concentration
    alpha = max(alpha, 1e-6)
    beta_param = max(beta_param, 1e-6)
    raw_severity = float(rng.beta(alpha, beta_param))
    severity = (
        p.winter_severity_autocorr * prev_severity
        + (1.0 - p.winter_severity_autocorr) * raw_severity
    )

    grid = state.grid.copy()
    settlements = [s.copy() for s in state.settlements]
    live = [s for s in settlements if s.alive]

    order = list(range(len(live)))
    rng.shuffle(order)

    collapsing: list[Settlement] = []

    for idx in order:
        s = live[idx]
        s.raid_stress *= 1.0 - p.raid_stress_decay

        food_loss = (p.winter_food_loss_flat + p.winter_food_loss_per_population * s.population) * (
            1.0 + p.winter_food_loss_severity_multiplier * severity
        )
        s.food -= food_loss

        starvation_stress = max(0.0, p.collapse_food_floor - s.food)
        pressure = (
            starvation_stress
            + p.collapse_raid_stress_weight * s.raid_stress
            + p.collapse_winter_severity_weight * severity
            - p.collapse_defense_buffer_weight * s.defense
        )
        p_collapse = _sigmoid((pressure - p.collapse_threshold) / p.collapse_softness)

        if rng.random() < p_collapse:
            collapsing.append(s)

    surviving_live = [s for s in live if s not in collapsing]

    for s in collapsing:
        s.alive = False
        s.ruin_age = 0
        grid.set(s.y, s.x, TerrainCode.RUIN)

        dispersal_pop = s.population * p.collapse_dispersal_fraction
        recipients = [
            r
            for r in surviving_live
            if r is not s and max(abs(r.y - s.y), abs(r.x - s.x)) <= p.collapse_dispersal_radius
        ]
        if recipients:
            weights = np.array(
                [
                    math.exp(
                        -p.collapse_dispersal_distance_decay * max(abs(r.y - s.y), abs(r.x - s.x))
                    )
                    for r in recipients
                ],
                dtype=np.float64,
            )
            weights /= weights.sum()
            for r, w in zip(recipients, weights, strict=True):
                r.population += dispersal_pop * float(w)

    result = state.copy()
    result.grid = grid
    result.settlements = settlements
    return result, severity
