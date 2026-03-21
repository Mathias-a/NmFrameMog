from __future__ import annotations

from idk_1.prediction import build_probability_grid as build_idk1_probability_grid
from round_8_implementation.solver.baseline import (
    build_baseline_tensor,
    floor_and_normalize,
)
from round_8_implementation.solver.contract import (
    EMPTYISH_TERRAIN_CODES,
    TerrainClass,
    terrain_code_to_class_index,
)


def uniform_floor_model(grid: list[list[int]]) -> list[list[list[float]]]:
    height = len(grid)
    width = len(grid[0])
    cell = floor_and_normalize([1.0 for _ in range(6)])
    return [[cell[:] for _ in range(width)] for _ in range(height)]


def initial_state_projection_model(grid: list[list[int]]) -> list[list[list[float]]]:
    return [[_project_cell(terrain_code) for terrain_code in row] for row in grid]


def idk1_wrapped_model(grid: list[list[int]]) -> list[list[list[float]]]:
    height = len(grid)
    width = len(grid[0])
    tensor = build_idk1_probability_grid(width, height)
    return [[floor_and_normalize(cell) for cell in row] for row in tensor]


def masked_baseline_model(grid: list[list[int]]) -> list[list[list[float]]]:
    baseline = build_baseline_tensor(grid)
    return [
        [
            _apply_feasibility_mask(
                baseline[y][x],
                terrain_code=grid[y][x],
                coastal=_is_coastal(grid, x=x, y=y),
            )
            for x in range(len(grid[0]))
        ]
        for y in range(len(grid))
    ]


def phase1_rules_model(grid: list[list[int]]) -> list[list[list[float]]]:
    baseline = build_baseline_tensor(grid)
    tensor: list[list[list[float]]] = []
    for y, row in enumerate(grid):
        tensor_row: list[list[float]] = []
        for x, terrain_code in enumerate(row):
            coastal = _is_coastal(grid, x=x, y=y)
            distribution = _apply_feasibility_mask(
                baseline[y][x],
                terrain_code=terrain_code,
                coastal=coastal,
            )
            adjusted = _motif_adjusted_distribution(
                distribution,
                grid=grid,
                x=x,
                y=y,
                terrain_code=terrain_code,
                coastal=coastal,
            )
            tensor_row.append(
                _apply_feasibility_mask(
                    adjusted,
                    terrain_code=terrain_code,
                    coastal=coastal,
                )
            )
        tensor.append(tensor_row)
    return tensor


def _project_cell(terrain_code: int) -> list[float]:
    distribution = [0.0 for _ in range(6)]
    distribution[terrain_code_to_class_index(terrain_code)] = 1.0

    if terrain_code == 10:
        distribution[int(TerrainClass.EMPTY)] = 1.0
    return floor_and_normalize(distribution)


def _apply_feasibility_mask(
    distribution: list[float], *, terrain_code: int, coastal: bool
) -> list[float]:
    allowed_classes = _allowed_classes(terrain_code=terrain_code, coastal=coastal)
    masked = [
        probability if index in allowed_classes else 0.0
        for index, probability in enumerate(distribution)
    ]
    return floor_and_normalize(masked)


def _allowed_classes(*, terrain_code: int, coastal: bool) -> set[int]:
    if terrain_code == 10:
        return {int(TerrainClass.EMPTY)}
    if terrain_code == 5:
        return {int(TerrainClass.MOUNTAIN)}

    allowed = {
        int(TerrainClass.EMPTY),
        int(TerrainClass.SETTLEMENT),
        int(TerrainClass.RUIN),
        int(TerrainClass.FOREST),
    }
    if coastal:
        allowed.add(int(TerrainClass.PORT))
    return allowed


def _motif_adjusted_distribution(
    distribution: list[float],
    *,
    grid: list[list[int]],
    x: int,
    y: int,
    terrain_code: int,
    coastal: bool,
) -> list[float]:
    adjusted = distribution[:]
    forest_neighbors = _count_neighbors(grid, x=x, y=y, terrain_codes={4})
    mountain_neighbors = _count_neighbors(grid, x=x, y=y, terrain_codes={5})
    active_neighbors = _count_neighbors(grid, x=x, y=y, terrain_codes={1, 2})
    ruin_neighbors = _count_neighbors(grid, x=x, y=y, terrain_codes={3})
    empty_neighbors = _count_neighbors(
        grid,
        x=x,
        y=y,
        terrain_codes=set(EMPTYISH_TERRAIN_CODES),
    )

    if terrain_code in EMPTYISH_TERRAIN_CODES:
        adjusted[int(TerrainClass.EMPTY)] += 0.05 + 0.01 * empty_neighbors
        adjusted[int(TerrainClass.SETTLEMENT)] += 0.04 * active_neighbors
        adjusted[int(TerrainClass.RUIN)] += 0.02 * ruin_neighbors
        adjusted[int(TerrainClass.FOREST)] += 0.02 * forest_neighbors
        if coastal:
            adjusted[int(TerrainClass.PORT)] += 0.03 + 0.02 * min(active_neighbors, 2)
        return floor_and_normalize(adjusted)

    if terrain_code == 4:
        adjusted[int(TerrainClass.FOREST)] += 0.08 + 0.01 * forest_neighbors
        adjusted[int(TerrainClass.SETTLEMENT)] += 0.02 * active_neighbors
        adjusted[int(TerrainClass.RUIN)] += 0.01 * ruin_neighbors
        if coastal and active_neighbors > 0:
            adjusted[int(TerrainClass.PORT)] += 0.02
        return floor_and_normalize(adjusted)

    if terrain_code == 1:
        support = forest_neighbors + mountain_neighbors + active_neighbors
        collapse_pressure = max(0, 2 - forest_neighbors) + ruin_neighbors
        adjusted[int(TerrainClass.SETTLEMENT)] += 0.06 + 0.015 * support
        adjusted[int(TerrainClass.RUIN)] += 0.03 * collapse_pressure
        adjusted[int(TerrainClass.FOREST)] += 0.01 * ruin_neighbors
        adjusted[int(TerrainClass.EMPTY)] += 0.01 * max(0, ruin_neighbors - 1)
        if coastal:
            adjusted[int(TerrainClass.PORT)] += 0.06 + 0.01 * support
        return floor_and_normalize(adjusted)

    if terrain_code == 2:
        support = forest_neighbors + active_neighbors
        collapse_pressure = max(0, 2 - forest_neighbors) + ruin_neighbors
        adjusted[int(TerrainClass.PORT)] += 0.08 + 0.015 * support
        adjusted[int(TerrainClass.SETTLEMENT)] += 0.015 * ruin_neighbors
        adjusted[int(TerrainClass.RUIN)] += 0.03 * collapse_pressure
        adjusted[int(TerrainClass.EMPTY)] += 0.01 * max(0, ruin_neighbors - 1)
        adjusted[int(TerrainClass.FOREST)] += 0.01 * ruin_neighbors
        return floor_and_normalize(adjusted)

    if terrain_code == 3:
        adjusted[int(TerrainClass.RUIN)] += 0.03 + 0.01 * ruin_neighbors
        adjusted[int(TerrainClass.SETTLEMENT)] += 0.04 * active_neighbors
        adjusted[int(TerrainClass.FOREST)] += 0.04 * forest_neighbors
        adjusted[int(TerrainClass.EMPTY)] += 0.02 * max(
            0, ruin_neighbors - active_neighbors
        )
        if coastal and active_neighbors > 0:
            adjusted[int(TerrainClass.PORT)] += 0.03
        return floor_and_normalize(adjusted)

    return floor_and_normalize(adjusted)


def _count_neighbors(
    grid: list[list[int]], *, x: int, y: int, terrain_codes: set[int]
) -> int:
    return sum(
        1
        for terrain_code in _neighbor_codes(grid, x=x, y=y)
        if terrain_code in terrain_codes
    )


def _is_coastal(grid: list[list[int]], *, x: int, y: int) -> bool:
    return 10 in _neighbor_codes(grid, x=x, y=y)


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
