from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class RoundStatus(str):
    PENDING = "pending"
    ACTIVE = "active"
    SCORING = "scoring"
    COMPLETED = "completed"


class InitialSettlement(_StrictModel):
    x: int
    y: int
    has_port: bool
    alive: bool


class InitialState(_StrictModel):
    grid: list[list[int]]
    settlements: list[InitialSettlement]


class RoundSummary(_StrictModel):
    id: str
    round_number: int
    event_date: str
    status: str
    map_width: int
    map_height: int
    prediction_window_minutes: int
    started_at: str | None
    closes_at: str | None
    round_weight: float
    created_at: str


class RoundDetail(_StrictModel):
    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[InitialState]


class BudgetResponse(_StrictModel):
    round_id: str
    queries_used: int
    queries_max: int
    active: bool


class SimulateRequest(_StrictModel):
    round_id: str
    seed_index: int = Field(ge=0, le=4)
    viewport_x: int = Field(default=0, ge=0)
    viewport_y: int = Field(default=0, ge=0)
    viewport_w: int = Field(default=15, ge=5, le=15)
    viewport_h: int = Field(default=15, ge=5, le=15)


class SimSettlement(_StrictModel):
    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    has_port: bool
    alive: bool
    owner_id: int


class ViewportBounds(_StrictModel):
    x: int
    y: int
    w: int
    h: int


class SimulateResponse(_StrictModel):
    grid: list[list[int]]
    settlements: list[SimSettlement]
    viewport: ViewportBounds
    width: int
    height: int
    queries_used: int
    queries_max: int


class SubmitRequest(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid")

    round_id: str
    seed_index: int
    prediction: list[list[list[float]]]

    @field_validator("seed_index")
    @classmethod
    def validate_seed_index(cls, v: int) -> int:
        if v < 0 or v > 4:
            raise ValueError(f"seed_index must be 0–4, got {v}")
        return v


class SubmitResponse(_StrictModel):
    status: Literal["accepted"]
    round_id: str
    seed_index: int


class AnalysisResponse(_StrictModel):
    prediction: list[list[list[float]]] | None
    ground_truth: list[list[list[float]]]
    score: float | None
    width: int
    height: int
    initial_grid: list[list[int]] | None


def new_round_id() -> str:
    return str(uuid.uuid4())


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"
