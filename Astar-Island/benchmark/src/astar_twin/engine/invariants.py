from __future__ import annotations

from astar_twin.contracts.types import TerrainCode
from astar_twin.state.grid import Grid
from astar_twin.state.round_state import RoundState


class InvariantViolation(ValueError):
    pass


def check_invariants(state: RoundState, initial_grid: Grid) -> None:
    for y in range(initial_grid.height):
        for x in range(initial_grid.width):
            initial_code = initial_grid.get(y, x)
            current_code = state.grid.get(y, x)

            if (
                initial_code in (TerrainCode.OCEAN, TerrainCode.MOUNTAIN)
                and current_code != initial_code
            ):
                raise InvariantViolation(
                    f"Static terrain changed at ({y},{x}): {initial_code} -> {current_code}"
                )

            if initial_code != TerrainCode.MOUNTAIN and current_code == TerrainCode.MOUNTAIN:
                raise InvariantViolation(f"Mountain created at ({y},{x})")

            if initial_code != TerrainCode.OCEAN and current_code == TerrainCode.OCEAN:
                raise InvariantViolation(f"Ocean created at ({y},{x})")

            if current_code == TerrainCode.PORT and not state.grid.is_coastal(y, x):
                raise InvariantViolation(f"Non-coastal port at ({y},{x})")
