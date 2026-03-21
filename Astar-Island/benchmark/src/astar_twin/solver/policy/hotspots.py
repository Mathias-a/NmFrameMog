"""Mechanism-aware hotspot generation from initial states.

Generates viewport candidate windows based on map analysis rules:
  - coastal settlement clusters (trade/port mechanism)
  - forest-frontier growth zones (expansion mechanism)
  - multi-settlement conflict corridors (raid mechanism)
  - reclaim-sensitive ruin/forest edges (environment mechanism)

Default viewport size: 15x15.
Shrink to 10x10 only when hotspot bbox is < 8x8.
Shrink to 5x5 only for contradiction-resolution probes.
"""

from __future__ import annotations

from dataclasses import dataclass

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import (
    MAX_VIEWPORT,
    MIN_VIEWPORT,
    TerrainCode,
)


@dataclass
class ViewportCandidate:
    """A candidate viewport window for querying."""

    x: int
    y: int
    w: int
    h: int
    category: str  # "coastal", "frontier", "corridor", "reclaim", "fallback"
    score: float = 0.0  # utility score (set later by allocator)

    def overlap_fraction(self, other: ViewportCandidate) -> float:
        """Compute fraction of this viewport that overlaps with another."""
        x_overlap = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        y_overlap = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        overlap_area = x_overlap * y_overlap
        self_area = self.w * self.h
        return overlap_area / self_area if self_area > 0 else 0.0


def _clamp_viewport(
    cx: int, cy: int, size: int, map_height: int, map_width: int
) -> tuple[int, int, int, int]:
    """Clamp a viewport centered at (cx, cy) to map bounds."""
    # Shrink to fit map if map is smaller than requested size
    w = max(MIN_VIEWPORT, min(MAX_VIEWPORT, size, map_width))
    h = max(MIN_VIEWPORT, min(MAX_VIEWPORT, size, map_height))
    half_w = w // 2
    half_h = h // 2
    x = max(0, min(map_width - w, cx - half_w))
    y = max(0, min(map_height - h, cy - half_h))
    return x, y, w, h


def _find_settlement_positions(initial_state: InitialState) -> list[tuple[int, int, bool, bool]]:
    """Extract (x, y, has_port, alive) from initial state settlements."""
    return [(s.x, s.y, s.has_port, s.alive) for s in initial_state.settlements]


def _find_coastal_cells(initial_state: InitialState) -> set[tuple[int, int]]:
    """Find all land cells adjacent to ocean."""
    grid = initial_state.grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    coastal: set[tuple[int, int]] = set()

    for y in range(height):
        for x in range(width):
            if grid[y][x] == TerrainCode.OCEAN:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and grid[ny][nx] == TerrainCode.OCEAN:
                    coastal.add((x, y))
                    break

    return coastal


def _find_forest_frontier(initial_state: InitialState) -> list[tuple[int, int]]:
    """Find forest cells adjacent to non-forest land (expansion zones)."""
    grid = initial_state.grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    frontier: list[tuple[int, int]] = []

    for y in range(height):
        for x in range(width):
            if grid[y][x] != TerrainCode.FOREST:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    ncode = grid[ny][nx]
                    if ncode in (
                        TerrainCode.SETTLEMENT,
                        TerrainCode.PORT,
                        TerrainCode.PLAINS,
                        TerrainCode.EMPTY,
                    ):
                        frontier.append((x, y))
                        break

    return frontier


def _select_viewport_size(bbox_w: int, bbox_h: int) -> int:
    """Choose hotspot viewport size from the contributing feature bbox."""
    if bbox_w < 8 and bbox_h < 8:
        return 10
    return MAX_VIEWPORT


def generate_hotspots(
    initial_state: InitialState,
    map_height: int,
    map_width: int,
    contradiction_probe: bool = False,
) -> list[ViewportCandidate]:
    """Generate mechanism-aware viewport candidates from one seed's initial state.

    Returns candidates sorted by category priority: coastal > corridor > frontier > reclaim > fallback.
    """
    candidates: list[ViewportCandidate] = []
    settlements = _find_settlement_positions(initial_state)
    alive_settlements = [(x, y, hp) for x, y, hp, alive in settlements if alive]

    # --- Coastal settlement clusters ---
    coastal_cells = _find_coastal_cells(initial_state)
    coastal_settlements = [
        (x, y) for x, y, hp in alive_settlements if (x, y) in coastal_cells or hp
    ]
    if coastal_settlements:
        # Cluster center of coastal settlements
        cx = sum(x for x, y in coastal_settlements) // len(coastal_settlements)
        cy = sum(y for x, y in coastal_settlements) // len(coastal_settlements)
        min_x = min(x for x, y in coastal_settlements)
        max_x = max(x for x, y in coastal_settlements)
        min_y = min(y for x, y in coastal_settlements)
        max_y = max(y for x, y in coastal_settlements)
        size = (
            MIN_VIEWPORT
            if contradiction_probe
            else _select_viewport_size(max_x - min_x, max_y - min_y)
        )
        x, y, w, h = _clamp_viewport(cx, cy, size, map_height, map_width)
        candidates.append(ViewportCandidate(x=x, y=y, w=w, h=h, category="coastal"))

    # --- Multi-settlement conflict corridors ---
    if len(alive_settlements) >= 2:
        # Find pairs of settlements close enough for raids (within ~10 cells)
        for i in range(len(alive_settlements)):
            for j in range(i + 1, len(alive_settlements)):
                x1, y1, _ = alive_settlements[i]
                x2, y2, _ = alive_settlements[j]
                dist = max(abs(x2 - x1), abs(y2 - y1))
                if dist <= 10:
                    mx = (x1 + x2) // 2
                    my = (y1 + y2) // 2
                    size = (
                        MIN_VIEWPORT
                        if contradiction_probe
                        else _select_viewport_size(abs(x2 - x1), abs(y2 - y1))
                    )
                    x, y, w, h = _clamp_viewport(mx, my, size, map_height, map_width)
                    candidates.append(ViewportCandidate(x=x, y=y, w=w, h=h, category="corridor"))

    # --- Forest-frontier growth zones ---
    frontier = _find_forest_frontier(initial_state)
    if frontier:
        # Cluster center of frontier cells
        fx = sum(x for x, y in frontier) // len(frontier)
        fy = sum(y for x, y in frontier) // len(frontier)
        min_x = min(x for x, y in frontier)
        max_x = max(x for x, y in frontier)
        min_y = min(y for x, y in frontier)
        max_y = max(y for x, y in frontier)
        size = (
            MIN_VIEWPORT
            if contradiction_probe
            else _select_viewport_size(max_x - min_x, max_y - min_y)
        )
        x, y, w, h = _clamp_viewport(fx, fy, size, map_height, map_width)
        candidates.append(ViewportCandidate(x=x, y=y, w=w, h=h, category="frontier"))

    # --- Ruin/forest reclaim edges ---
    grid = initial_state.grid
    grid_h = len(grid)
    grid_w = len(grid[0]) if grid_h > 0 else 0
    ruin_positions: list[tuple[int, int]] = []
    for y_pos in range(grid_h):
        for x_pos in range(grid_w):
            if grid[y_pos][x_pos] == TerrainCode.RUIN:
                ruin_positions.append((x_pos, y_pos))

    if ruin_positions:
        rx = sum(x for x, y in ruin_positions) // len(ruin_positions)
        ry = sum(y for x, y in ruin_positions) // len(ruin_positions)
        min_x = min(x for x, y in ruin_positions)
        max_x = max(x for x, y in ruin_positions)
        min_y = min(y for x, y in ruin_positions)
        max_y = max(y for x, y in ruin_positions)
        size = (
            MIN_VIEWPORT
            if contradiction_probe
            else _select_viewport_size(max_x - min_x, max_y - min_y)
        )
        x, y, w, h = _clamp_viewport(rx, ry, size, map_height, map_width)
        candidates.append(ViewportCandidate(x=x, y=y, w=w, h=h, category="reclaim"))

    # Deduplicate exact duplicates first
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[ViewportCandidate] = []
    for c in candidates:
        key = (c.x, c.y, c.w, c.h)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique

    # --- Fallback: grid-based coverage if too few unique candidates ---
    if len(candidates) < 2:
        step = max(MIN_VIEWPORT, min(MAX_VIEWPORT, map_width) // 2)
        for gy in range(0, map_height, step):
            for gx in range(0, map_width, step):
                viewport_size = MIN_VIEWPORT if contradiction_probe else MAX_VIEWPORT
                w = min(viewport_size, map_width - gx)
                h = min(viewport_size, map_height - gy)
                key = (gx, gy, w, h)
                if w >= MIN_VIEWPORT and h >= MIN_VIEWPORT and key not in seen:
                    seen.add(key)
                    candidates.append(ViewportCandidate(x=gx, y=gy, w=w, h=h, category="fallback"))
        # If still < 2, add a smaller centered viewport
        if len(candidates) < 2:
            if contradiction_probe:
                small_w = min(MIN_VIEWPORT, map_width)
                small_h = min(MIN_VIEWPORT, map_height)
            else:
                cw = min(MAX_VIEWPORT, map_width)
                ch = min(MAX_VIEWPORT, map_height)
                small_w = max(MIN_VIEWPORT, cw // 2)
                small_h = max(MIN_VIEWPORT, ch // 2)
            cx = (map_width - small_w) // 2
            cy = (map_height - small_h) // 2
            key = (cx, cy, small_w, small_h)
            if key not in seen:
                candidates.append(
                    ViewportCandidate(x=cx, y=cy, w=small_w, h=small_h, category="fallback")
                )

    return candidates
