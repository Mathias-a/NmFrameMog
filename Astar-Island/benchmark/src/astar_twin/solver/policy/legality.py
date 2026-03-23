from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode


def _is_coastal(grid: list[list[int]], x: int, y: int) -> bool:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == TerrainCode.OCEAN:
            return True
    return False


def build_legal_class_mask(initial_state: InitialState) -> NDArray[np.bool_]:
    grid = initial_state.grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    mask = np.ones((height, width, NUM_CLASSES), dtype=np.bool_)

    for y, row in enumerate(grid):
        for x, code in enumerate(row):
            if code == TerrainCode.OCEAN:
                mask[y, x, :] = False
                mask[y, x, ClassIndex.EMPTY] = True
                continue

            if code == TerrainCode.MOUNTAIN:
                mask[y, x, :] = False
                mask[y, x, ClassIndex.MOUNTAIN] = True
                continue

            mask[y, x, ClassIndex.MOUNTAIN] = False
            if not _is_coastal(grid=grid, x=x, y=y):
                mask[y, x, ClassIndex.PORT] = False

    return mask


def legal_class_mask_for_viewport(
    initial_state: InitialState,
    viewport_x: int,
    viewport_y: int,
    viewport_w: int,
    viewport_h: int,
) -> NDArray[np.bool_]:
    return build_legal_class_mask(initial_state)[
        viewport_y : viewport_y + viewport_h,
        viewport_x : viewport_x + viewport_w,
        :,
    ]


def apply_legality_filter(
    tensor: NDArray[np.float64],
    initial_state: InitialState,
) -> NDArray[np.float64]:
    legal_mask = build_legal_class_mask(initial_state)
    return _apply_mask(tensor, legal_mask)


def apply_legality_filter_to_viewport(
    tensor: NDArray[np.float64],
    initial_state: InitialState,
    viewport_x: int,
    viewport_y: int,
) -> NDArray[np.float64]:
    viewport_h, viewport_w = tensor.shape[:2]
    legal_mask = legal_class_mask_for_viewport(
        initial_state,
        viewport_x=viewport_x,
        viewport_y=viewport_y,
        viewport_w=viewport_w,
        viewport_h=viewport_h,
    )
    return _apply_mask(tensor, legal_mask)


def _apply_mask(tensor: NDArray[np.float64], legal_mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    filtered = np.where(legal_mask, tensor, 0.0)

    sums = np.sum(filtered, axis=2, keepdims=True)
    zero_mass = sums <= 1e-12
    if np.any(zero_mass):
        legal_counts = np.sum(legal_mask, axis=2, keepdims=True).astype(np.float64)
        fallback = legal_mask.astype(np.float64) / np.maximum(legal_counts, 1.0)
        filtered = np.where(zero_mass, fallback, filtered)
        sums = np.sum(filtered, axis=2, keepdims=True)

    return filtered / np.maximum(sums, 1e-12)
