from __future__ import annotations

import math
from dataclasses import dataclass

from .models import SeedState, Viewport


@dataclass(frozen=True)
class PlannedViewport:
    viewport: Viewport
    score: float
    average_entropy: float
    unresolved_fraction: float


def rank_candidate_viewports(
    state: SeedState,
    *,
    viewport_width: int,
    viewport_height: int,
    limit: int,
) -> list[PlannedViewport]:
    height = len(state.initial_grid)
    width = len(state.initial_grid[0])
    candidates: list[PlannedViewport] = []
    seen_cells = set(state.observed_cells)
    y_positions = _grid_positions(width=height, viewport_size=viewport_height)
    x_positions = _grid_positions(width=width, viewport_size=viewport_width)

    for y in y_positions:
        for x in x_positions:
            viewport = Viewport(x=x, y=y, width=viewport_width, height=viewport_height)
            score, average_entropy, unresolved_fraction = _score_viewport(
                tensor=state.current_tensor,
                viewport=viewport,
                seen_cells=seen_cells,
            )
            candidates.append(
                PlannedViewport(
                    viewport=viewport,
                    score=score,
                    average_entropy=average_entropy,
                    unresolved_fraction=unresolved_fraction,
                )
            )
    candidates.sort(
        key=lambda candidate: (
            candidate.score,
            candidate.average_entropy,
            candidate.unresolved_fraction,
        ),
        reverse=True,
    )
    return candidates[:limit]


def _score_viewport(
    *,
    tensor: list[list[list[float]]],
    viewport: Viewport,
    seen_cells: set[tuple[int, int]],
) -> tuple[float, float, float]:
    entropy_sum = 0.0
    unresolved_count = 0
    cell_count = viewport.width * viewport.height
    frontier_bonus = 0.0
    for y in range(viewport.y, viewport.y + viewport.height):
        for x in range(viewport.x, viewport.x + viewport.width):
            cell = tensor[y][x]
            entropy = -sum(probability * math.log(probability) for probability in cell)
            entropy_sum += entropy
            if (x, y) not in seen_cells:
                unresolved_count += 1
            else:
                frontier_bonus += 0.02
    average_entropy = entropy_sum / cell_count
    unresolved_fraction = unresolved_count / cell_count
    score = entropy_sum + unresolved_fraction * 2.0 + frontier_bonus
    return score, average_entropy, unresolved_fraction


def _grid_positions(*, width: int, viewport_size: int) -> list[int]:
    if viewport_size >= width:
        return [0]
    positions = list(range(0, width - viewport_size + 1, max(1, viewport_size // 2)))
    last_position = width - viewport_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions
