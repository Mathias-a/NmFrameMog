from __future__ import annotations

from typing import Protocol

from astar_twin.contracts.api_models import InitialState


class InitialStateSource(Protocol):
    def generate(self, seed: int, width: int, height: int) -> InitialState: ...
