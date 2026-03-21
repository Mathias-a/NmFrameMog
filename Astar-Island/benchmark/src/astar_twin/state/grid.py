from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.types import (
    STATIC_TERRAIN_CODES,
    TERRAIN_TO_CLASS,
    TerrainCode,
)


@dataclass
class Grid:
    cells: NDArray[np.int32]

    @classmethod
    def from_list(cls, rows: list[list[int]]) -> Grid:
        return cls(cells=np.array(rows, dtype=np.int32))

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    def get(self, y: int, x: int) -> int:
        return int(self.cells[y, x])

    def set(self, y: int, x: int, code: int) -> None:
        self.cells[y, x] = code

    def to_list(self) -> list[list[int]]:
        return self.cells.tolist()  # type: ignore[no-any-return]

    def is_static(self, y: int, x: int) -> bool:
        return int(self.cells[y, x]) in STATIC_TERRAIN_CODES

    def is_ocean(self, y: int, x: int) -> bool:
        return int(self.cells[y, x]) == TerrainCode.OCEAN

    def is_mountain(self, y: int, x: int) -> bool:
        return int(self.cells[y, x]) == TerrainCode.MOUNTAIN

    def is_coastal(self, y: int, x: int) -> bool:
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width and self.is_ocean(ny, nx):
                return True
        return False

    def moore_neighbors(self, y: int, x: int) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    result.append((ny, nx))
        return result

    def von_neumann_neighbors(self, y: int, x: int) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                result.append((ny, nx))
        return result

    def class_tensor(self) -> NDArray[np.int32]:
        mapped = np.zeros_like(self.cells)
        for code, cls_idx in TERRAIN_TO_CLASS.items():
            mapped[self.cells == code] = cls_idx
        return mapped

    def viewport(self, vx: int, vy: int, vw: int, vh: int) -> Grid:
        x0 = max(0, vx)
        y0 = max(0, vy)
        x1 = min(self.width, vx + vw)
        y1 = min(self.height, vy + vh)
        return Grid(cells=self.cells[y0:y1, x0:x1].copy())

    def copy(self) -> Grid:
        return Grid(cells=self.cells.copy())
