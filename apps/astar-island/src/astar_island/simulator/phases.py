"""Simulation phases for the Norse world simulator.

Each phase mutates a WorldState in place.
"""

from __future__ import annotations

import random

from astar_island.terrain import TerrainType

from .mapgen import _distance, _is_coastal, _neighbors
from .models import Settlement, SimConfig, WorldState


def phase_growth(state: WorldState, config: SimConfig, rng: random.Random) -> None:
    """Growth phase: settlements gather food, grow population, expand."""
    new_settlements: list[Settlement] = []

    for settlement in state.settlements:
        if not settlement.alive:
            continue

        # Count adjacent terrain
        forest_count = 0
        plains_count = 0
        for nx, ny in _neighbors(settlement.x, settlement.y, state.width, state.height):
            terrain = state.grid[ny][nx]
            if terrain == TerrainType.FOREST:
                forest_count += 1
            elif terrain == TerrainType.PLAINS:
                plains_count += 1

        # Add food from surrounding terrain
        food_gain = (
            forest_count * config.food_per_forest
            + plains_count * config.food_per_plains
        )
        settlement.food += food_gain

        # Population grows proportional to food
        if settlement.food > 0:
            growth = max(1, settlement.food // 10)
            settlement.population += growth

        # Expansion: found new settlement if large enough
        if settlement.population >= config.expansion_threshold:
            # Find nearest empty PLAINS within distance 4
            best_spot: tuple[int, int] | None = None
            best_dist = float("inf")
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    nx, ny = settlement.x + dx, settlement.y + dy
                    if (
                        0 <= nx < state.width
                        and 0 <= ny < state.height
                        and state.grid[ny][nx] == TerrainType.PLAINS
                        and _distance(settlement.x, settlement.y, nx, ny) <= 4.0
                    ):
                        dist = _distance(settlement.x, settlement.y, nx, ny)
                        if dist < best_dist and dist > 0:
                            best_dist = dist
                            best_spot = (nx, ny)

            if best_spot is not None:
                bx, by = best_spot
                state.grid[by][bx] = TerrainType.SETTLEMENT
                new_settlement = Settlement(
                    x=bx,
                    y=by,
                    population=settlement.population // 3,
                    food=settlement.food // 3,
                    wealth=settlement.wealth // 4,
                    defense=settlement.defense,
                    tech_level=settlement.tech_level,
                    faction_id=settlement.faction_id,
                    has_port=False,
                    has_longship=False,
                    alive=True,
                )
                new_settlements.append(new_settlement)
                settlement.population -= settlement.population // 3
                settlement.food -= settlement.food // 3

        # Port development
        is_coastal = _is_coastal(
            state.grid, settlement.x, settlement.y, state.width, state.height
        )
        if is_coastal and not settlement.has_port:
            if rng.random() < config.port_development_prob:
                settlement.has_port = True
                state.grid[settlement.y][settlement.x] = TerrainType.PORT

        # Longship building
        if settlement.has_port and not settlement.has_longship:
            if rng.random() < config.longship_build_prob:
                settlement.has_longship = True

    state.settlements.extend(new_settlements)


def phase_conflict(state: WorldState, config: SimConfig, rng: random.Random) -> None:
    """Conflict phase: settlements may raid enemies within range."""
    for attacker in state.settlements:
        if not attacker.alive:
            continue

        # Decide whether to raid
        if attacker.food < config.raid_aggression_threshold:
            raid_chance = 0.7
        else:
            raid_chance = 0.3

        if rng.random() > raid_chance:
            continue

        # Find enemy targets within range
        effective_range = config.raid_range
        if attacker.has_longship:
            effective_range += config.longship_range_bonus

        targets: list[Settlement] = []
        for other in state.settlements:
            if (
                other.alive
                and other.faction_id != attacker.faction_id
                and _distance(attacker.x, attacker.y, other.x, other.y)
                <= effective_range
            ):
                targets.append(other)

        if not targets:
            continue

        defender = rng.choice(targets)

        # Combat resolution
        attacker_str = attacker.population * (1.0 + attacker.tech_level * 0.1)
        defender_str = defender.population * (1.0 + defender.defense * 0.1)

        total_str = attacker_str + defender_str
        if total_str == 0:
            continue

        success_prob = config.raid_success_base * attacker_str / total_str

        if rng.random() < success_prob:
            # Raid succeeds: take half food, may change faction
            loot = defender.food // 2
            attacker.food += loot
            defender.food -= loot
            defender.faction_id = attacker.faction_id


def phase_trade(state: WorldState, config: SimConfig, rng: random.Random) -> None:
    """Trade phase: ports within range exchange resources."""
    ports = [s for s in state.settlements if s.alive and s.has_port]

    for idx in range(len(ports)):
        for jdx in range(idx + 1, len(ports)):
            port_a = ports[idx]
            port_b = ports[jdx]
            if _distance(port_a.x, port_a.y, port_b.x, port_b.y) <= config.trade_range:
                # Exchange food and wealth
                port_a.food += config.trade_food_bonus
                port_b.food += config.trade_food_bonus
                port_a.wealth += config.trade_wealth_bonus
                port_b.wealth += config.trade_wealth_bonus

                # Tech diffusion: close the gap
                tech_gap = port_a.tech_level - port_b.tech_level
                if tech_gap > 0:
                    transfer = max(1, int(tech_gap * config.tech_diffusion_rate))
                    port_b.tech_level += transfer
                elif tech_gap < 0:
                    transfer = max(1, int(-tech_gap * config.tech_diffusion_rate))
                    port_a.tech_level += transfer


def phase_winter(state: WorldState, config: SimConfig, rng: random.Random) -> None:
    """Winter phase: settlements lose food and population, may collapse."""
    severity = rng.gauss(config.winter_severity_mean, config.winter_severity_std)
    severity = max(0.0, min(1.0, severity))

    for settlement in state.settlements:
        if not settlement.alive:
            continue

        # Food loss
        food_loss = int(settlement.food * severity)
        settlement.food -= food_loss

        # Population loss
        pop_loss = int(settlement.population * severity * 0.3)
        settlement.population -= pop_loss

        # Collapse check
        if (
            settlement.food <= config.collapse_food_threshold
            or settlement.population <= 0
        ):
            settlement.alive = False
            state.grid[settlement.y][settlement.x] = TerrainType.RUIN

            # Disperse population to nearest friendly alive settlement
            if settlement.population > 0:
                nearest_friendly: Settlement | None = None
                nearest_dist = float("inf")
                for other in state.settlements:
                    if (
                        other.alive
                        and other.faction_id == settlement.faction_id
                        and other is not settlement
                    ):
                        dist = _distance(settlement.x, settlement.y, other.x, other.y)
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_friendly = other

                if nearest_friendly is not None:
                    nearest_friendly.population += settlement.population // 2

            settlement.population = 0


def phase_environment(state: WorldState, config: SimConfig, rng: random.Random) -> None:
    """Environment phase: ruins may be reclaimed, overgrown, or cleared."""
    for settlement in state.settlements:
        if settlement.alive:
            continue
        if state.grid[settlement.y][settlement.x] != TerrainType.RUIN:
            continue

        # Check for nearby thriving settlement
        has_thriving_neighbor = False
        for other in state.settlements:
            if (
                other.alive
                and other.population > 80
                and _distance(settlement.x, settlement.y, other.x, other.y) <= 3.0
            ):
                has_thriving_neighbor = True
                break

        if has_thriving_neighbor and rng.random() < config.ruin_reclaim_prob:
            # Reclaim the ruin
            settlement.alive = True
            settlement.population = 30
            settlement.food = 20
            settlement.wealth = 0
            settlement.defense = 1
            settlement.tech_level = 1
            state.grid[settlement.y][settlement.x] = TerrainType.SETTLEMENT
        elif rng.random() < config.ruin_to_forest_prob:
            state.grid[settlement.y][settlement.x] = TerrainType.FOREST
        elif rng.random() < config.ruin_to_plains_prob:
            state.grid[settlement.y][settlement.x] = TerrainType.PLAINS
