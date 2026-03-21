from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .contract import CLASS_COUNT, terrain_code_to_class_index

DEFAULT_SIMULATION_YEARS = 50


@dataclass(frozen=True)
class InitialSettlement:
    x: int
    y: int
    has_port: bool
    alive: bool
    owner_id: int


@dataclass
class SettlementState:
    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    tech_level: float
    has_port: bool
    has_longship: bool
    owner_id: int
    alive: bool = True


@dataclass(frozen=True)
class SimulationRunResult:
    grid: list[list[int]]
    settlements: tuple[SettlementState, ...]


def run_proxy_simulation(
    initial_grid: list[list[int]],
    initial_settlements: tuple[InitialSettlement, ...],
    *,
    rng: Random,
    years: int = DEFAULT_SIMULATION_YEARS,
) -> SimulationRunResult:
    if years <= 0:
        raise ValueError("Simulation years must be positive.")

    grid = [row[:] for row in initial_grid]
    settlements = _initialize_settlements(grid, initial_settlements, rng=rng)
    ruins = _initialize_ruins(grid)

    for _ in range(years):
        _growth_phase(grid, settlements, ruins, rng=rng)
        _conflict_phase(grid, settlements, ruins, rng=rng)
        _trade_phase(grid, settlements, rng=rng)
        _winter_phase(grid, settlements, ruins, rng=rng)
        _environment_phase(grid, settlements, ruins, rng=rng)
        _normalize_ports(grid, settlements)

    final_grid = [row[:] for row in grid]
    final_settlements = tuple(
        sorted(
            (
                settlement
                for settlement in settlements.values()
                if settlement.alive and final_grid[settlement.y][settlement.x] in {1, 2}
            ),
            key=lambda settlement: (settlement.y, settlement.x, settlement.owner_id),
        )
    )
    return SimulationRunResult(grid=final_grid, settlements=final_settlements)


def build_ground_truth_tensor(
    initial_grid: list[list[int]],
    initial_settlements: tuple[InitialSettlement, ...],
    *,
    rollout_count: int,
    base_seed: int,
    years: int = DEFAULT_SIMULATION_YEARS,
) -> list[list[list[float]]]:
    if rollout_count <= 0:
        raise ValueError("Rollout count must be positive.")

    height = len(initial_grid)
    width = len(initial_grid[0])
    counts = [
        [[0 for _ in range(CLASS_COUNT)] for _ in range(width)] for _ in range(height)
    ]

    for rollout_index in range(rollout_count):
        result = run_proxy_simulation(
            initial_grid,
            initial_settlements,
            rng=Random(base_seed + rollout_index * 97_531),
            years=years,
        )
        for y, row in enumerate(result.grid):
            for x, terrain_code in enumerate(row):
                counts[y][x][terrain_code_to_class_index(terrain_code)] += 1

    return [
        [[count / rollout_count for count in counts[y][x]] for x in range(width)]
        for y in range(height)
    ]


def _initialize_settlements(
    grid: list[list[int]],
    initial_settlements: tuple[InitialSettlement, ...],
    *,
    rng: Random,
) -> dict[tuple[int, int], SettlementState]:
    settlements: dict[tuple[int, int], SettlementState] = {}
    for settlement in initial_settlements:
        if not settlement.alive:
            continue
        coastal = _is_coastal(grid, settlement.x, settlement.y)
        has_port = settlement.has_port and coastal
        terrain_code = 2 if has_port else 1
        grid[settlement.y][settlement.x] = terrain_code
        settlements[(settlement.x, settlement.y)] = SettlementState(
            x=settlement.x,
            y=settlement.y,
            population=1.2 + 0.25 * rng.random(),
            food=0.9 + 0.2 * rng.random(),
            wealth=0.7 + 0.2 * rng.random(),
            defense=0.6 + 0.15 * rng.random(),
            tech_level=0.35 + 0.15 * rng.random(),
            has_port=has_port,
            has_longship=False,
            owner_id=settlement.owner_id,
        )
    return settlements


def _initialize_ruins(grid: list[list[int]]) -> dict[tuple[int, int], int]:
    ruins: dict[tuple[int, int], int] = {}
    for y, row in enumerate(grid):
        for x, terrain_code in enumerate(row):
            if terrain_code == 3:
                ruins[(x, y)] = 0
    return ruins


def _growth_phase(
    grid: list[list[int]],
    settlements: dict[tuple[int, int], SettlementState],
    ruins: dict[tuple[int, int], int],
    *,
    rng: Random,
) -> None:
    for settlement in list(_alive_settlements(settlements)):
        if not settlement.alive:
            continue
        forest_neighbors = _count_neighbors(grid, settlement.x, settlement.y, {4})
        plains_neighbors = _count_neighbors(grid, settlement.x, settlement.y, {0, 11})
        mountain_neighbors = _count_neighbors(grid, settlement.x, settlement.y, {5})
        active_neighbors = _count_neighbors(grid, settlement.x, settlement.y, {1, 2})
        coastal = _is_coastal(grid, settlement.x, settlement.y)

        food_gain = (
            0.42
            + 0.17 * forest_neighbors
            + 0.05 * plains_neighbors
            + (0.16 if settlement.has_port else 0.0)
            + 0.05 * rng.random()
        )
        wealth_gain = (
            0.18
            + 0.09 * active_neighbors
            + 0.18 * settlement.tech_level
            + (0.24 if settlement.has_port else 0.04)
            + 0.05 * rng.random()
        )
        defense_gain = (
            0.04
            + 0.08 * mountain_neighbors
            + 0.03 * settlement.wealth
            + 0.04 * rng.random()
        )
        upkeep = 0.28 + 0.16 * settlement.population

        settlement.food = max(-1.5, settlement.food + food_gain - upkeep)
        settlement.wealth = max(-1.0, settlement.wealth + wealth_gain)
        settlement.defense = max(0.1, settlement.defense + defense_gain)
        settlement.tech_level = min(
            4.0,
            settlement.tech_level
            + 0.02
            + 0.02 * max(0.0, settlement.wealth)
            + (0.02 if settlement.has_port else 0.0),
        )
        settlement.population = max(
            0.2,
            settlement.population
            + 0.18 * food_gain
            + 0.08 * settlement.tech_level
            - 0.04 * rng.random(),
        )

        if coastal and not settlement.has_port:
            if settlement.population > 1.8 and settlement.wealth > 1.1:
                if rng.random() < 0.35 + 0.08 * settlement.tech_level:
                    settlement.has_port = True
                    grid[settlement.y][settlement.x] = 2

        if (
            settlement.has_port
            and settlement.wealth > 1.5
            and settlement.tech_level > 0.8
        ):
            if rng.random() < 0.45 + 0.08 * settlement.tech_level:
                settlement.has_longship = True

        if (
            settlement.population > 2.6
            and settlement.food > 0.9
            and settlement.wealth > 1.0
            and rng.random() < 0.18 + 0.04 * settlement.tech_level
        ):
            candidate = _choose_expansion_target(grid, ruins, settlement, rng=rng)
            if candidate is not None:
                new_x, new_y = candidate
                was_ruin = (new_x, new_y) in ruins
                new_has_port = _is_coastal(grid, new_x, new_y) and (
                    was_ruin or rng.random() < 0.25
                )
                settlements[(new_x, new_y)] = SettlementState(
                    x=new_x,
                    y=new_y,
                    population=max(0.7, settlement.population * 0.42),
                    food=max(0.5, settlement.food * 0.35),
                    wealth=max(0.35, settlement.wealth * 0.28),
                    defense=max(0.3, settlement.defense * 0.55),
                    tech_level=max(0.2, settlement.tech_level * 0.75),
                    has_port=new_has_port,
                    has_longship=False,
                    owner_id=settlement.owner_id,
                )
                settlement.population = max(0.9, settlement.population * 0.72)
                settlement.food = max(0.2, settlement.food * 0.74)
                settlement.wealth = max(0.1, settlement.wealth * 0.8)
                ruins.pop((new_x, new_y), None)
                grid[new_y][new_x] = 2 if new_has_port else 1


def _conflict_phase(
    grid: list[list[int]],
    settlements: dict[tuple[int, int], SettlementState],
    ruins: dict[tuple[int, int], int],
    *,
    rng: Random,
) -> None:
    for attacker in list(_alive_settlements(settlements)):
        targets = _find_targets(settlements, attacker)
        if not targets:
            continue
        raid_bias = 0.1 if attacker.food < 0.6 else 0.0
        if rng.random() >= 0.2 + raid_bias + (0.08 if attacker.has_longship else 0.0):
            continue

        target = min(
            targets,
            key=lambda settlement: (
                abs(settlement.x - attacker.x) + abs(settlement.y - attacker.y),
                settlement.population,
            ),
        )
        attack_strength = (
            0.42 * attacker.population
            + 0.26 * attacker.wealth
            + 0.24 * attacker.defense
            + 0.18 * attacker.tech_level
            + (0.55 if attacker.has_longship else 0.0)
            + 0.5 * rng.random()
        )
        defense_strength = (
            0.38 * target.population
            + 0.35 * target.defense
            + 0.12 * target.wealth
            + 0.15 * target.tech_level
            + 0.45 * rng.random()
        )
        if attack_strength > defense_strength:
            loot = 0.18 + 0.15 * rng.random()
            attacker.food += 0.22 + 0.5 * loot
            attacker.wealth += 0.18 + 0.7 * loot
            target.food -= 0.28 + 0.25 * rng.random()
            target.wealth -= 0.22 + 0.2 * rng.random()
            target.defense -= 0.15 + 0.2 * rng.random()
            target.population -= 0.16 + 0.25 * rng.random()
            if target.population < 0.65 or target.defense < 0.2:
                if rng.random() < 0.45:
                    target.owner_id = attacker.owner_id
                    target.population = max(0.75, target.population + 0.15)
                    target.defense = max(0.25, target.defense + 0.12)
                else:
                    _collapse_settlement(grid, settlements, ruins, target)
        else:
            attacker.food -= 0.12 + 0.1 * rng.random()
            attacker.wealth -= 0.08 + 0.08 * rng.random()
            attacker.population -= 0.08 + 0.12 * rng.random()


def _trade_phase(
    grid: list[list[int]],
    settlements: dict[tuple[int, int], SettlementState],
    *,
    rng: Random,
) -> None:
    ports = [
        settlement
        for settlement in _alive_settlements(settlements)
        if settlement.has_port and _is_coastal(grid, settlement.x, settlement.y)
    ]
    for index, left in enumerate(ports):
        for right in ports[index + 1 :]:
            distance = abs(left.x - right.x) + abs(left.y - right.y)
            if distance > 7:
                continue
            trade_gain = max(0.0, 0.42 - 0.03 * distance) + 0.05 * rng.random()
            tech_gain = 0.02 + 0.01 * min(left.tech_level, right.tech_level)
            for settlement in (left, right):
                settlement.food += 0.08 + 0.25 * trade_gain
                settlement.wealth += 0.14 + 0.35 * trade_gain
                settlement.tech_level = min(4.0, settlement.tech_level + tech_gain)


def _winter_phase(
    grid: list[list[int]],
    settlements: dict[tuple[int, int], SettlementState],
    ruins: dict[tuple[int, int], int],
    *,
    rng: Random,
) -> None:
    severity = 0.85 + 0.8 * rng.random()
    for settlement in list(_alive_settlements(settlements)):
        settlement.food -= 0.32 * severity + 0.14 * settlement.population
        if settlement.food < 0.0:
            settlement.population += settlement.food * 0.6
            settlement.wealth += settlement.food * 0.15
        settlement.defense = max(0.05, settlement.defense - 0.04 * severity)
        if (
            settlement.population <= 0.35
            or settlement.food <= -0.8
            or settlement.wealth <= -0.45
        ):
            _collapse_settlement(grid, settlements, ruins, settlement)


def _environment_phase(
    grid: list[list[int]],
    settlements: dict[tuple[int, int], SettlementState],
    ruins: dict[tuple[int, int], int],
    *,
    rng: Random,
) -> None:
    for position in list(ruins):
        ruins[position] += 1

    for position, ruin_age in list(ruins.items()):
        x, y = position
        reclaiming_settlement = _find_reclaiming_settlement(settlements, x=x, y=y)
        if reclaiming_settlement is not None and rng.random() < 0.22 + 0.04 * min(
            reclaiming_settlement.tech_level, 3.0
        ):
            has_port = _is_coastal(grid, x, y) and rng.random() < 0.4
            settlements[position] = SettlementState(
                x=x,
                y=y,
                population=max(0.7, reclaiming_settlement.population * 0.33),
                food=max(0.45, reclaiming_settlement.food * 0.32),
                wealth=max(0.35, reclaiming_settlement.wealth * 0.28),
                defense=max(0.3, reclaiming_settlement.defense * 0.45),
                tech_level=max(0.2, reclaiming_settlement.tech_level * 0.7),
                has_port=has_port,
                has_longship=False,
                owner_id=reclaiming_settlement.owner_id,
            )
            reclaiming_settlement.population = max(
                0.8, reclaiming_settlement.population * 0.78
            )
            reclaiming_settlement.food = max(0.1, reclaiming_settlement.food * 0.84)
            grid[y][x] = 2 if has_port else 1
            ruins.pop(position, None)
            continue

        forest_neighbors = _count_neighbors(grid, x, y, {4})
        if ruin_age >= 2 and forest_neighbors >= 2 and rng.random() < 0.35:
            grid[y][x] = 4
            ruins.pop(position, None)
            continue
        if ruin_age >= 4 and rng.random() < 0.45:
            grid[y][x] = 11
            ruins.pop(position, None)


def _normalize_ports(
    grid: list[list[int]],
    settlements: dict[tuple[int, int], SettlementState],
) -> None:
    for settlement in _alive_settlements(settlements):
        if settlement.has_port and not _is_coastal(grid, settlement.x, settlement.y):
            settlement.has_port = False
        grid[settlement.y][settlement.x] = 2 if settlement.has_port else 1


def _find_targets(
    settlements: dict[tuple[int, int], SettlementState],
    attacker: SettlementState,
) -> list[SettlementState]:
    max_distance = (
        2 + (2 if attacker.has_longship else 0) + (1 if attacker.has_port else 0)
    )
    targets: list[SettlementState] = []
    for settlement in _alive_settlements(settlements):
        if settlement.owner_id == attacker.owner_id:
            continue
        distance = abs(settlement.x - attacker.x) + abs(settlement.y - attacker.y)
        if distance <= max_distance:
            targets.append(settlement)
    return targets


def _choose_expansion_target(
    grid: list[list[int]],
    ruins: dict[tuple[int, int], int],
    settlement: SettlementState,
    *,
    rng: Random,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int, int]] = []
    for y in range(max(0, settlement.y - 2), min(len(grid), settlement.y + 3)):
        for x in range(max(0, settlement.x - 2), min(len(grid[0]), settlement.x + 3)):
            if (x, y) == (settlement.x, settlement.y):
                continue
            terrain_code = grid[y][x]
            if terrain_code in {1, 2, 5, 10}:
                continue
            priority = 0
            if (x, y) in ruins:
                priority += 3
            if _is_coastal(grid, x, y):
                priority += 1
            if terrain_code in {0, 11}:
                priority += 1
            candidates.append((priority, x, y))
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -item[0],
            abs(item[1] - settlement.x) + abs(item[2] - settlement.y),
            item[2],
            item[1],
        )
    )
    top_priority = candidates[0][0]
    top_candidates = [
        (x, y) for priority, x, y in candidates if priority == top_priority
    ]
    return top_candidates[int(rng.random() * len(top_candidates))]


def _collapse_settlement(
    grid: list[list[int]],
    settlements: dict[tuple[int, int], SettlementState],
    ruins: dict[tuple[int, int], int],
    settlement: SettlementState,
) -> None:
    settlement.alive = False
    settlements.pop((settlement.x, settlement.y), None)
    grid[settlement.y][settlement.x] = 3
    ruins[(settlement.x, settlement.y)] = 0


def _find_reclaiming_settlement(
    settlements: dict[tuple[int, int], SettlementState],
    *,
    x: int,
    y: int,
) -> SettlementState | None:
    candidates = [
        settlement
        for settlement in _alive_settlements(settlements)
        if abs(settlement.x - x) + abs(settlement.y - y) <= 2
        and settlement.population > 1.7
        and settlement.wealth > 0.9
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda settlement: (
            settlement.population + settlement.wealth + settlement.tech_level,
            -abs(settlement.x - x) - abs(settlement.y - y),
        ),
    )


def _count_neighbors(
    grid: list[list[int]],
    x: int,
    y: int,
    terrain_codes: set[int],
) -> int:
    return sum(
        1
        for neighbor_code in _neighbor_codes(grid, x=x, y=y)
        if neighbor_code in terrain_codes
    )


def _neighbor_codes(grid: list[list[int]], *, x: int, y: int) -> list[int]:
    height = len(grid)
    width = len(grid[0])
    neighbors: list[int] = []
    for delta_y in (-1, 0, 1):
        for delta_x in (-1, 0, 1):
            if delta_x == 0 and delta_y == 0:
                continue
            next_x = x + delta_x
            next_y = y + delta_y
            if 0 <= next_x < width and 0 <= next_y < height:
                neighbors.append(grid[next_y][next_x])
    return neighbors


def _is_coastal(grid: list[list[int]], x: int, y: int) -> bool:
    return any(neighbor_code == 10 for neighbor_code in _neighbor_codes(grid, x=x, y=y))


def _alive_settlements(
    settlements: dict[tuple[int, int], SettlementState],
) -> list[SettlementState]:
    return [
        settlement
        for settlement in sorted(
            settlements.values(), key=lambda value: (value.y, value.x, value.owner_id)
        )
        if settlement.alive
    ]


def sampled_world_to_tensor(grid: list[list[int]]) -> list[list[list[float]]]:
    return [
        [_terrain_distribution(terrain_code) for terrain_code in row] for row in grid
    ]


def _terrain_distribution(terrain_code: int) -> list[float]:
    distribution = [0.0] * CLASS_COUNT
    distribution[terrain_code_to_class_index(terrain_code)] = 1.0
    return distribution
