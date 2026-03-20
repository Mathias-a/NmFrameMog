from __future__ import annotations

from .contract import (
    CLASS_COUNT,
    EMPTYISH_TERRAIN_CODES,
    PROBABILITY_FLOOR,
    TerrainClass,
    terrain_code_is_static,
)


def build_baseline_tensor(grid: list[list[int]]) -> list[list[list[float]]]:
    height = len(grid)
    width = len(grid[0])
    tensor: list[list[list[float]]] = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(_distribution_for_cell(grid=grid, x=x, y=y))
        tensor.append(row)
    return tensor


def blend_observation_into_tensor(
    tensor: list[list[list[float]]],
    *,
    x: int,
    y: int,
    observed_terrain_code: int,
) -> None:
    observed_distribution = _observation_distribution(observed_terrain_code)
    tensor[y][x] = combine_distributions(
        tensor[y][x], observed_distribution, prior_weight=0.35, evidence_weight=0.65
    )


def combine_distributions(
    prior: list[float],
    update: list[float],
    *,
    prior_weight: float,
    evidence_weight: float,
) -> list[float]:
    combined = [
        prior[index] * prior_weight + update[index] * evidence_weight
        for index in range(CLASS_COUNT)
    ]
    return floor_and_normalize(combined)


def floor_and_normalize(
    probabilities: list[float], *, probability_floor: float = PROBABILITY_FLOOR
) -> list[float]:
    if len(probabilities) != CLASS_COUNT:
        raise ValueError(
            f"Expected {CLASS_COUNT} class probabilities, got {len(probabilities)}"
        )
    floor_total = probability_floor * CLASS_COUNT
    if floor_total >= 1.0:
        raise ValueError("Probability floor leaves no room for renormalization.")
    safe_probabilities = [max(0.0, probability) for probability in probabilities]
    base_total = sum(safe_probabilities)
    remaining_mass = 1.0 - floor_total
    if base_total <= 0.0:
        scaled = [remaining_mass / CLASS_COUNT for _ in range(CLASS_COUNT)]
    else:
        scaled = [
            probability / base_total * remaining_mass
            for probability in safe_probabilities
        ]
    return [probability_floor + probability for probability in scaled]


def _distribution_for_cell(*, grid: list[list[int]], x: int, y: int) -> list[float]:
    terrain_code = grid[y][x]
    if terrain_code_is_static(terrain_code):
        if terrain_code == 5:
            return _one_hot_distribution(TerrainClass.MOUNTAIN)
        return _one_hot_distribution(TerrainClass.EMPTY)

    neighbor_codes = _neighbor_codes(grid=grid, x=x, y=y)
    touching_water = any(code == 10 for code in neighbor_codes)
    touching_forest = any(code == 4 for code in neighbor_codes)
    touching_settlement = any(code in {1, 2, 3} for code in neighbor_codes)

    if terrain_code in EMPTYISH_TERRAIN_CODES:
        distribution = [0.78, 0.10, 0.03, 0.03, 0.04, 0.02]
        if touching_settlement:
            distribution[TerrainClass.SETTLEMENT] += 0.07
            distribution[TerrainClass.RUIN] += 0.02
            distribution[TerrainClass.EMPTY] -= 0.06
        if touching_forest:
            distribution[TerrainClass.FOREST] += 0.04
            distribution[TerrainClass.EMPTY] -= 0.03
        if touching_water:
            distribution[TerrainClass.PORT] += 0.03
            distribution[TerrainClass.EMPTY] -= 0.02
        return floor_and_normalize(distribution)

    if terrain_code == 1:
        distribution = [0.08, 0.56, 0.16 if touching_water else 0.07, 0.14, 0.03, 0.03]
        return floor_and_normalize(distribution)

    if terrain_code == 2:
        distribution = [0.06, 0.20, 0.56, 0.12, 0.03, 0.03]
        return floor_and_normalize(distribution)

    if terrain_code == 3:
        distribution = [0.16, 0.22, 0.06 if touching_water else 0.03, 0.34, 0.16, 0.03]
        return floor_and_normalize(distribution)

    if terrain_code == 4:
        distribution = [0.10, 0.06, 0.02, 0.05, 0.74, 0.03]
        if touching_settlement:
            distribution[TerrainClass.EMPTY] += 0.03
            distribution[TerrainClass.SETTLEMENT] += 0.02
            distribution[TerrainClass.FOREST] -= 0.04
        return floor_and_normalize(distribution)

    raise ValueError(f"Unsupported terrain code in baseline model: {terrain_code}")


def _observation_distribution(terrain_code: int) -> list[float]:
    if terrain_code == 5:
        return _one_hot_distribution(TerrainClass.MOUNTAIN)
    if terrain_code == 10:
        return _one_hot_distribution(TerrainClass.EMPTY)
    if terrain_code in EMPTYISH_TERRAIN_CODES:
        return floor_and_normalize([0.84, 0.05, 0.02, 0.03, 0.04, 0.02])
    if terrain_code == 1:
        return floor_and_normalize([0.04, 0.74, 0.09, 0.08, 0.03, 0.02])
    if terrain_code == 2:
        return floor_and_normalize([0.04, 0.10, 0.75, 0.06, 0.03, 0.02])
    if terrain_code == 3:
        return floor_and_normalize([0.10, 0.12, 0.04, 0.60, 0.11, 0.03])
    if terrain_code == 4:
        return floor_and_normalize([0.06, 0.03, 0.02, 0.05, 0.81, 0.03])
    raise ValueError(f"Unsupported observed terrain code: {terrain_code}")


def _neighbor_codes(*, grid: list[list[int]], x: int, y: int) -> list[int]:
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


def _one_hot_distribution(terrain_class: TerrainClass) -> list[float]:
    probabilities = [PROBABILITY_FLOOR] * CLASS_COUNT
    probabilities[int(terrain_class)] = 1.0
    return floor_and_normalize(probabilities)
