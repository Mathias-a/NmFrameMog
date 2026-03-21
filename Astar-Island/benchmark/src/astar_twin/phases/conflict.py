from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

from astar_twin.params.simulation_params import DistanceMetric, SimulationParams
from astar_twin.state.round_state import RoundState
from astar_twin.state.settlement import Settlement


def _dist(s1: Settlement, s2: Settlement, metric: DistanceMetric) -> float:
    dy = abs(s1.y - s2.y)
    dx = abs(s1.x - s2.x)
    if metric == DistanceMetric.CHEBYSHEV:
        return float(max(dy, dx))
    if metric == DistanceMetric.MANHATTAN:
        return float(dy + dx)
    return math.sqrt(dy * dy + dx * dx)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def apply_conflict(
    state: RoundState,
    p: SimulationParams,
    rng: Generator,
    war_registry: dict[tuple[int, int], int],
    current_year: int,
) -> RoundState:
    settlements = [s.copy() for s in state.settlements]
    live = [s for s in settlements if s.alive]

    order = list(range(len(live)))
    rng.shuffle(order)

    for i in order:
        raider = live[i]
        food_pc = raider.food / max(raider.population, 0.5)
        desperation = _sigmoid((p.raid_desperation_food_pc - food_pc) / p.raid_desperation_scale)
        p_raid = p.raid_base_prob + (1.0 - p.raid_base_prob) * desperation

        if rng.random() > p_raid:
            continue

        raid_range = p.raid_range_base + (p.raid_range_longship_bonus if raider.has_longship else 0)

        targets = [
            t
            for t in live
            if t is not raider
            and t.owner_id != raider.owner_id
            and _dist(raider, t, p.distance_metric) <= raid_range
        ]
        if not targets:
            continue

        weights = np.array(
            [
                math.exp(-_dist(raider, t, p.distance_metric) / max(raid_range, 1))
                * (1.0 + t.food + t.wealth)
                / (1.0 + t.defense)
                for t in targets
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        target = targets[int(rng.choice(len(targets), p=weights))]

        tech_mil = p.tech_military_bonus
        attack = raider.population * (1.0 + tech_mil * raider.tech)
        guard = target.defense * (1.0 + tech_mil * target.tech)
        p_success = _sigmoid((attack - guard) / p.raid_success_scale)

        if rng.random() < p_success:
            loot_food = p.raid_loot_frac * target.food
            loot_wealth = p.raid_loot_frac * target.wealth
            raider.food += loot_food
            raider.wealth += loot_wealth
            target.food -= loot_food
            target.wealth -= loot_wealth
            target.population *= 1.0 - p.raid_damage_frac
            target.defense *= 1.0 - 1.5 * p.raid_damage_frac
            target.raid_stress += p.raid_damage_frac

            capture_ratio = attack / max(guard, 0.1)
            p_capture = _sigmoid((capture_ratio - p.raid_capture_threshold) / 0.25)
            if rng.random() < p_capture:
                target.owner_id = raider.owner_id

            key: tuple[int, int] = (
                min(raider.owner_id, target.owner_id),
                max(raider.owner_id, target.owner_id),
            )
            war_registry[key] = current_year

    result = state.copy()
    result.settlements = settlements
    return result
