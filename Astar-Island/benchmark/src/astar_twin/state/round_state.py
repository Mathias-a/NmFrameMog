from __future__ import annotations

import copy
from dataclasses import dataclass

from astar_twin.state.grid import Grid
from astar_twin.state.settlement import Settlement


@dataclass
class RoundState:
    grid: Grid
    settlements: list[Settlement]
    year: int = 0
    rng_state: object = None

    def copy(self) -> RoundState:
        return RoundState(
            grid=self.grid.copy(),
            settlements=[s.copy() for s in self.settlements],
            year=self.year,
            rng_state=copy.deepcopy(self.rng_state),
        )

    def live_settlements(self) -> list[Settlement]:
        return [s for s in self.settlements if s.alive]

    def dead_settlements(self) -> list[Settlement]:
        return [s for s in self.settlements if not s.alive]
