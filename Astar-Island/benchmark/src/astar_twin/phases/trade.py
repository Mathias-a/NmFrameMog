from __future__ import annotations

import math
from typing import TYPE_CHECKING

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


def apply_trade(
    state: RoundState,
    p: SimulationParams,
    rng: Generator,
    war_registry: dict[tuple[int, int], int],
    current_year: int,
) -> RoundState:
    settlements = [s.copy() for s in state.settlements]
    ports = [s for s in settlements if s.alive and s.has_port]

    for i in range(len(ports)):
        for j in range(i + 1, len(ports)):
            a = ports[i]
            b = ports[j]

            key: tuple[int, int] = (min(a.owner_id, b.owner_id), max(a.owner_id, b.owner_id))
            last_war = war_registry.get(key, -999)
            if current_year - last_war <= p.war_cooldown_years:
                continue

            d = _dist(a, b, p.distance_metric)
            if d > p.trade_range:
                continue

            avg_tech = (a.tech + b.tech) / 2.0
            trade_val = (
                p.trade_value_scale
                * math.exp(-d / max(p.trade_range, 1))
                * math.sqrt((1.0 + a.wealth) * (1.0 + b.wealth))
                * (1.0 + p.tech_economic_bonus * avg_tech)
            )

            a.wealth += trade_val
            b.wealth += trade_val
            a.food += 0.5 * trade_val
            b.food += 0.5 * trade_val

            delta_tech_a = (
                p.tech_diffusion_rate * trade_val * (1.0 + b.tech) / (2.0 + a.tech + b.tech)
            )
            delta_tech_b = (
                p.tech_diffusion_rate * trade_val * (1.0 + a.tech) / (2.0 + a.tech + b.tech)
            )
            a.tech += delta_tech_a
            b.tech += delta_tech_b

    result = state.copy()
    result.settlements = settlements
    return result
