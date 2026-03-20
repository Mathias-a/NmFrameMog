from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Viewport:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class SettlementObservation:
    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    has_port: bool
    alive: bool
    owner_id: int


@dataclass(frozen=True)
class QueryResponse:
    viewport: Viewport
    grid: list[list[int]]
    settlements: tuple[SettlementObservation, ...] = ()
    queries_used: int | None = None
    queries_max: int | None = None


@dataclass(frozen=True)
class ObservationRecord:
    step: int
    response: QueryResponse
    source: str


@dataclass
class SeedState:
    seed_index: int
    initial_grid: list[list[int]]
    current_tensor: list[list[list[float]]]
    observation_history: list[ObservationRecord] = field(default_factory=list)
    observed_cells: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    query_plan: list[Viewport] = field(default_factory=list)


@dataclass(frozen=True)
class RoundDetail:
    round_id: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: tuple[list[list[int]], ...]
    status: str | None = None
    prediction_window_minutes: int | None = None


@dataclass(frozen=True)
class SeedSolveResult:
    seed_index: int
    prediction: list[list[list[float]]]
    planned_viewports: tuple[Viewport, ...]
    observations_used: tuple[ObservationRecord, ...]
    prediction_path: str
    debug_output_dir: str


@dataclass(frozen=True)
class RunSummary:
    round_id: str
    run_id: str
    output_root: str
    seed_results: tuple[SeedSolveResult, ...]
