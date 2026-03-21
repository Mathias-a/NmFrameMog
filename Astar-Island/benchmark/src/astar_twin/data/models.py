from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from astar_twin.contracts.api_models import InitialState


class RoundFixture(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[InitialState]
    ground_truths: list[list[list[list[float]]]] | None = None
    event_date: str = ""
    round_weight: float = 1.0
