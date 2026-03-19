"""Deterministic map generation for the Norse world simulator."""

from __future__ import annotations

import math
import random

from astar_island.terrain import TerrainType

from .models import Settlement, SimConfig, WorldState


def _distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Euclidean distance between two grid cells."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _neighbors(x: int, y: int, width: int, height: int) -> list[tuple[int, int]]:
    """Return 4-directional neighbors within bounds."""
    result: list[tuple[int, int]] = []
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            result.append((nx, ny))
    return result


def _is_coastal(
    grid: list[list[TerrainType]], x: int, y: int, width: int, height: int
) -> bool:
    """Check if a cell is adjacent to OCEAN."""
    for nx, ny in _neighbors(x, y, width, height):
        if grid[ny][nx] == TerrainType.OCEAN:
            return True
    return False


def _is_land(terrain: TerrainType) -> bool:
    """Check if a terrain type is walkable land (not ocean)."""
    return terrain != TerrainType.OCEAN


def generate_map(
    seed: int,
    config: SimConfig,
    width: int = 40,
    height: int = 40,
) -> WorldState:
    """Generate a deterministic Norse world map from a seed."""
    rng = random.Random(seed)

    # 1. Fill with PLAINS
    grid: list[list[TerrainType]] = [
        [TerrainType.PLAINS for _ in range(width)] for _ in range(height)
    ]

    # 2. Ocean borders (2 cells thick)
    for y in range(height):
        for x in range(width):
            if x < 2 or x >= width - 2 or y < 2 or y >= height - 2:
                grid[y][x] = TerrainType.OCEAN

    # 3. Fjords — random walks from ocean border inward
    for _ in range(config.fjord_count):
        # Pick a random ocean border cell
        border_cells: list[tuple[int, int]] = []
        for y in range(height):
            for x in range(width):
                if grid[y][x] == TerrainType.OCEAN and (
                    x == 2 or x == width - 3 or y == 2 or y == height - 3
                ):
                    # Inner edge of the ocean border
                    for nx, ny in _neighbors(x, y, width, height):
                        if grid[ny][nx] != TerrainType.OCEAN:
                            border_cells.append((x, y))
                            break

        if not border_cells:
            continue

        fx, fy = rng.choice(border_cells)
        for _ in range(config.fjord_max_length):
            grid[fy][fx] = TerrainType.OCEAN
            land_neighbors = [
                (nx, ny)
                for nx, ny in _neighbors(fx, fy, width, height)
                if grid[ny][nx] != TerrainType.OCEAN
            ]
            if not land_neighbors:
                break
            fx, fy = rng.choice(land_neighbors)

    # 4. Mountain chains
    for _ in range(config.mountain_chains):
        # Pick random interior land cell
        interior_land: list[tuple[int, int]] = [
            (x, y)
            for y in range(3, height - 3)
            for x in range(3, width - 3)
            if grid[y][x] == TerrainType.PLAINS
        ]
        if not interior_land:
            continue

        mx, my = rng.choice(interior_land)
        for _ in range(config.mountain_walk_length):
            grid[my][mx] = TerrainType.MOUNTAIN
            land_neighbors = [
                (nx, ny)
                for nx, ny in _neighbors(mx, my, width, height)
                if grid[ny][nx] != TerrainType.OCEAN
                and grid[ny][nx] != TerrainType.MOUNTAIN
            ]
            if not land_neighbors:
                break
            mx, my = rng.choice(land_neighbors)

    # 5. Forest patches
    for _ in range(config.forest_patch_count):
        land_cells: list[tuple[int, int]] = [
            (x, y)
            for y in range(2, height - 2)
            for x in range(2, width - 2)
            if grid[y][x] == TerrainType.PLAINS
        ]
        if not land_cells:
            continue

        cx, cy = rng.choice(land_cells)
        radius = config.forest_patch_radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                fx, fy = cx + dx, cy + dy
                if (
                    0 <= fx < width
                    and 0 <= fy < height
                    and _distance(cx, cy, fx, fy) <= radius
                    and grid[fy][fx] == TerrainType.PLAINS
                ):
                    grid[fy][fx] = TerrainType.FOREST

    # 6. Initial settlements — greedy placement on PLAINS
    settlements: list[Settlement] = []
    placed_positions: list[tuple[int, int]] = []

    plains_cells: list[tuple[int, int]] = [
        (x, y)
        for y in range(2, height - 2)
        for x in range(2, width - 2)
        if grid[y][x] == TerrainType.PLAINS
    ]
    rng.shuffle(plains_cells)

    faction_counter = 0
    for sx, sy in plains_cells:
        if len(settlements) >= config.initial_settlement_count:
            break

        # Check min spacing
        too_close = False
        for px, py in placed_positions:
            if _distance(sx, sy, px, py) < config.min_settlement_spacing:
                too_close = True
                break
        if too_close:
            continue

        faction_counter += 1
        grid[sy][sx] = TerrainType.SETTLEMENT
        settlement = Settlement(
            x=sx,
            y=sy,
            population=100,
            food=50,
            wealth=10,
            defense=5,
            tech_level=1,
            faction_id=faction_counter,
            has_port=False,
            has_longship=False,
            alive=True,
        )
        settlements.append(settlement)
        placed_positions.append((sx, sy))

    return WorldState(
        width=width,
        height=height,
        grid=grid,
        settlements=settlements,
        year=0,
    )
