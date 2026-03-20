from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import cast

from .baseline import floor_and_normalize
from .contract import (
    DEFAULT_QUERY_BUDGET,
    MAX_VIEWPORT_SIZE,
    MIN_VIEWPORT_SIZE,
    terrain_code_is_static,
)
from .models import RoundDetail, Viewport
from .pipeline import parse_round_detail_payload
from .proxy_simulator import (
    DEFAULT_SIMULATION_YEARS,
    SettlementState,
    build_ground_truth_tensor,
    run_proxy_simulation,
)
from .proxy_simulator import (
    InitialSettlement as ProxyInitialSettlement,
)
from .validator import entropy_weighted_kl_score, validate_prediction_tensor

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FIXTURE_PATHS: tuple[Path, ...] = (
    _PROJECT_ROOT
    / Path(
        ".artifacts/astar-island-live/rounds/c5cdf100-a876-4fb7-b5d8-757162c97989.json"
    ),
    _PROJECT_ROOT
    / Path(
        ".artifacts/astar-island-prev-round/rounds/36e581f1-73f8-453f-ab98-cbe3052b701b.json"
    ),
)


class EmulatorError(Exception):
    pass


class RoundNotFoundError(EmulatorError):
    pass


class RoundInactiveError(EmulatorError):
    pass


class BudgetExceededError(EmulatorError):
    pass


@dataclass(frozen=True)
class InitialSettlement:
    x: int
    y: int
    has_port: bool
    alive: bool
    owner_id: int


@dataclass(frozen=True)
class LoadedRound:
    round_detail: RoundDetail
    source_path: Path
    round_number: int
    initial_settlements: tuple[tuple[InitialSettlement, ...], ...]


@dataclass
class StoredSubmission:
    prediction: list[list[list[float]]]
    submitted_at: str


@dataclass
class RoundRuntimeState:
    round_data: LoadedRound
    queries_used: int = 0
    queries_max: int = DEFAULT_QUERY_BUDGET
    submissions: dict[int, StoredSubmission] = field(default_factory=dict)
    analysis_cache: dict[int, dict[str, object]] = field(default_factory=dict)


class AstarIslandEmulator:
    def __init__(
        self,
        rounds: Sequence[LoadedRound],
        *,
        active_round_id: str | None = None,
        random_seed: int = 20260320,
        analysis_rollout_count: int = 128,
        simulation_years: int = DEFAULT_SIMULATION_YEARS,
    ) -> None:
        if not rounds:
            raise ValueError("At least one round fixture is required.")
        round_map = {item.round_detail.round_id: item for item in rounds}
        if len(round_map) != len(rounds):
            raise ValueError("Duplicate round IDs are not supported.")
        self._rounds = round_map
        self._round_state = {
            round_id: RoundRuntimeState(round_data=item)
            for round_id, item in round_map.items()
        }
        self._random_seed = random_seed
        self._analysis_rollout_count = analysis_rollout_count
        self._simulation_years = simulation_years
        self._simulate_counter = 0
        self._active_round_id = self._resolve_active_round_id(active_round_id)

    @classmethod
    def from_fixture_paths(
        cls,
        fixture_paths: Sequence[Path] | None = None,
        *,
        active_round_id: str | None = None,
        random_seed: int = 20260320,
        analysis_rollout_count: int = 128,
        simulation_years: int = DEFAULT_SIMULATION_YEARS,
    ) -> AstarIslandEmulator:
        paths = tuple(fixture_paths or DEFAULT_FIXTURE_PATHS)
        loaded_rounds = tuple(
            _load_round_fixture(_resolve_fixture_path(path), round_number=index + 1)
            for index, path in enumerate(paths)
        )
        return cls(
            loaded_rounds,
            active_round_id=active_round_id,
            random_seed=random_seed,
            analysis_rollout_count=analysis_rollout_count,
            simulation_years=simulation_years,
        )

    @property
    def active_round_id(self) -> str:
        return self._active_round_id

    def list_rounds(self) -> list[dict[str, object]]:
        return [
            _round_summary_payload(
                state.round_data, is_active=round_id == self._active_round_id
            )
            for round_id, state in sorted(
                self._round_state.items(),
                key=lambda item: item[1].round_data.round_number,
            )
        ]

    def get_round_detail(self, round_id: str) -> dict[str, object]:
        state = self._round_state.get(round_id)
        if state is None:
            raise RoundNotFoundError(f"Round not found: {round_id}")
        return _round_detail_payload(
            state.round_data, is_active=round_id == self._active_round_id
        )

    def get_budget(self) -> dict[str, object]:
        state = self._round_state[self._active_round_id]
        remaining = state.queries_max - state.queries_used
        return {
            "round_id": state.round_data.round_detail.round_id,
            "roundId": state.round_data.round_detail.round_id,
            "queries_used": state.queries_used,
            "queries_max": state.queries_max,
            "remaining": remaining,
            "budget": remaining,
            "active": True,
        }

    def simulate(self, payload: object) -> dict[str, object]:
        request = _require_mapping(payload, context="simulate payload")
        round_id = _read_optional_str(request, "round_id") or _read_optional_str(
            request, "roundId"
        )
        if round_id is None:
            round_id = self._active_round_id
        state = self._require_active_round(round_id)
        seed_index = _read_optional_int(request, "seed_index")
        if seed_index is None:
            seed_index = _read_optional_int(request, "seedIndex")
        if seed_index is None:
            seed_index = 0
        self._validate_seed_index(state.round_data.round_detail, seed_index)

        viewport = _parse_viewport_request(
            request,
            map_width=state.round_data.round_detail.map_width,
            map_height=state.round_data.round_detail.map_height,
        )
        if state.queries_used >= state.queries_max:
            raise BudgetExceededError(
                f"Query budget exhausted ({state.queries_used}/{state.queries_max})"
            )

        self._simulate_counter += 1
        rng = Random(self._random_seed + self._simulate_counter + seed_index * 10_000)
        simulation = run_proxy_simulation(
            state.round_data.round_detail.initial_states[seed_index],
            _proxy_initial_settlements(
                state.round_data.initial_settlements[seed_index]
            ),
            rng=rng,
            years=self._simulation_years,
        )

        state.queries_used += 1
        viewport_grid = _crop_grid(
            simulation.grid,
            x=viewport.x,
            y=viewport.y,
            width=viewport.width,
            height=viewport.height,
        )
        settlements = _viewport_settlements(
            simulation.settlements,
            viewport=viewport,
        )
        return {
            "round_id": state.round_data.round_detail.round_id,
            "roundId": state.round_data.round_detail.round_id,
            "seed_index": seed_index,
            "grid": viewport_grid,
            "settlements": settlements,
            "viewport": {
                "x": viewport.x,
                "y": viewport.y,
                "w": viewport.width,
                "h": viewport.height,
                "width": viewport.width,
                "height": viewport.height,
            },
            "width": state.round_data.round_detail.map_width,
            "height": state.round_data.round_detail.map_height,
            "queries_used": state.queries_used,
            "queries_max": state.queries_max,
        }

    def submit(self, payload: object) -> dict[str, object]:
        request = _require_mapping(payload, context="submit payload")
        round_id = _read_optional_str(request, "round_id") or _read_optional_str(
            request, "roundId"
        )
        if round_id is None:
            raise ValueError("Field 'round_id' is required.")
        state = self._require_active_round(round_id)
        seed_index = _read_optional_int(request, "seed_index")
        if seed_index is None:
            seed_index = _read_optional_int(request, "seedIndex")
        if seed_index is None:
            raise ValueError("Field 'seed_index' is required.")
        self._validate_seed_index(state.round_data.round_detail, seed_index)

        tensor = _parse_prediction_tensor(request.get("prediction"))
        validate_prediction_tensor(
            tensor,
            width=state.round_data.round_detail.map_width,
            height=state.round_data.round_detail.map_height,
        )
        state.submissions[seed_index] = StoredSubmission(
            prediction=tensor,
            submitted_at=dt.datetime.now(dt.UTC).isoformat(),
        )
        state.analysis_cache.pop(seed_index, None)
        return {
            "status": "accepted",
            "round_id": state.round_data.round_detail.round_id,
            "roundId": state.round_data.round_detail.round_id,
            "seed_index": seed_index,
        }

    def get_analysis(self, round_id: str, seed_index: int) -> dict[str, object]:
        state = self._require_round(round_id)
        self._validate_seed_index(state.round_data.round_detail, seed_index)
        cached = state.analysis_cache.get(seed_index)
        if cached is not None:
            return cached

        round_detail = state.round_data.round_detail
        prediction = state.submissions.get(seed_index)
        ground_truth = build_ground_truth_tensor(
            round_detail.initial_states[seed_index],
            _proxy_initial_settlements(
                state.round_data.initial_settlements[seed_index]
            ),
            rollout_count=self._analysis_rollout_count,
            base_seed=self._analysis_seed(round_id, seed_index),
            years=self._simulation_years,
        )
        normalized_ground_truth = [
            [floor_and_normalize(cell, probability_floor=0.0) for cell in row]
            for row in ground_truth
        ]
        score: float | None = None
        prediction_payload: list[list[list[float]]] | None = None
        if prediction is not None:
            prediction_payload = prediction.prediction
            score = entropy_weighted_kl_score(
                prediction.prediction, normalized_ground_truth
            )
        payload = {
            "prediction": prediction_payload,
            "ground_truth": normalized_ground_truth,
            "score": score,
            "width": round_detail.map_width,
            "height": round_detail.map_height,
            "initial_grid": round_detail.initial_states[seed_index],
        }
        state.analysis_cache[seed_index] = payload
        return payload

    def _resolve_active_round_id(self, active_round_id: str | None) -> str:
        if active_round_id is not None:
            if active_round_id not in self._rounds:
                raise RoundNotFoundError(f"Active round not found: {active_round_id}")
            return active_round_id
        for round_id, round_data in sorted(
            self._rounds.items(), key=lambda item: item[1].round_number
        ):
            if round_data.round_detail.status == "active":
                return round_id
        return next(iter(self._rounds))

    def _require_active_round(self, round_id: str) -> RoundRuntimeState:
        state = self._require_round(round_id)
        if round_id != self._active_round_id:
            raise RoundInactiveError(f"Round is not active: {round_id}")
        return state

    def _require_round(self, round_id: str) -> RoundRuntimeState:
        state = self._round_state.get(round_id)
        if state is None:
            raise RoundNotFoundError(f"Round not found: {round_id}")
        return state

    def _validate_seed_index(self, round_detail: RoundDetail, seed_index: int) -> None:
        if not 0 <= seed_index < round_detail.seeds_count:
            raise ValueError(
                "Invalid seed_index "
                f"{seed_index}; expected 0 <= seed_index < {round_detail.seeds_count}"
            )

    def _analysis_seed(self, round_id: str, seed_index: int) -> int:
        round_component = sum(ord(character) for character in round_id)
        return self._random_seed + round_component * 101 + seed_index * 10_000


def _load_round_fixture(path: Path, *, round_number: int) -> LoadedRound:
    payload = json.loads(path.read_text(encoding="utf-8"))
    round_detail = parse_round_detail_payload(payload)
    initial_settlements = tuple(
        _extract_initial_settlements(grid) for grid in round_detail.initial_states
    )
    return LoadedRound(
        round_detail=round_detail,
        source_path=path,
        round_number=round_number,
        initial_settlements=initial_settlements,
    )


def _resolve_fixture_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def _round_summary_payload(
    loaded_round: LoadedRound, *, is_active: bool
) -> dict[str, object]:
    round_detail = loaded_round.round_detail
    status = "active" if is_active else (round_detail.status or "completed")
    return {
        "id": round_detail.round_id,
        "round_id": round_detail.round_id,
        "roundId": round_detail.round_id,
        "round_number": loaded_round.round_number,
        "status": status,
        "width": round_detail.map_width,
        "height": round_detail.map_height,
        "map_width": round_detail.map_width,
        "map_height": round_detail.map_height,
        "seeds_count": round_detail.seeds_count,
        "prediction_window_minutes": round_detail.prediction_window_minutes,
    }


def _round_detail_payload(
    loaded_round: LoadedRound, *, is_active: bool
) -> dict[str, object]:
    round_detail = loaded_round.round_detail
    status = "active" if is_active else (round_detail.status or "completed")
    return {
        "id": round_detail.round_id,
        "round_id": round_detail.round_id,
        "roundId": round_detail.round_id,
        "round_number": loaded_round.round_number,
        "status": status,
        "width": round_detail.map_width,
        "height": round_detail.map_height,
        "map_width": round_detail.map_width,
        "map_height": round_detail.map_height,
        "seeds_count": round_detail.seeds_count,
        "prediction_window_minutes": round_detail.prediction_window_minutes,
        "initial_states": [
            {
                "grid": round_detail.initial_states[seed_index],
                "settlements": [
                    {
                        "x": settlement.x,
                        "y": settlement.y,
                        "has_port": settlement.has_port,
                        "alive": settlement.alive,
                    }
                    for settlement in loaded_round.initial_settlements[seed_index]
                ],
            }
            for seed_index in range(round_detail.seeds_count)
        ],
    }


def _extract_initial_settlements(
    grid: list[list[int]],
) -> tuple[InitialSettlement, ...]:
    settlements: list[InitialSettlement] = []
    next_owner_id = 1
    for y, row in enumerate(grid):
        for x, terrain_code in enumerate(row):
            if terrain_code not in {1, 2}:
                continue
            settlements.append(
                InitialSettlement(
                    x=x,
                    y=y,
                    has_port=terrain_code == 2,
                    alive=True,
                    owner_id=next_owner_id,
                )
            )
            next_owner_id += 1
    return tuple(settlements)


def _parse_viewport_request(
    payload: Mapping[str, object], *, map_width: int, map_height: int
) -> Viewport:
    nested_viewport = payload.get("viewport")
    viewport_mapping = (
        _require_mapping(nested_viewport, context="viewport")
        if nested_viewport is not None
        else None
    )
    x = _first_int(payload, viewport_mapping, keys=("viewport_x", "x"), default=0)
    y = _first_int(payload, viewport_mapping, keys=("viewport_y", "y"), default=0)
    width = _first_int(
        payload,
        viewport_mapping,
        keys=("viewport_w", "w", "width"),
        default=MAX_VIEWPORT_SIZE,
    )
    height = _first_int(
        payload,
        viewport_mapping,
        keys=("viewport_h", "h", "height"),
        default=MAX_VIEWPORT_SIZE,
    )
    if x < 0 or y < 0:
        raise ValueError("Viewport origin must be non-negative.")
    if not MIN_VIEWPORT_SIZE <= width <= MAX_VIEWPORT_SIZE:
        raise ValueError(
            "Viewport width must be between "
            f"{MIN_VIEWPORT_SIZE} and {MAX_VIEWPORT_SIZE}."
        )
    if not MIN_VIEWPORT_SIZE <= height <= MAX_VIEWPORT_SIZE:
        raise ValueError(
            "Viewport height must be between "
            f"{MIN_VIEWPORT_SIZE} and {MAX_VIEWPORT_SIZE}."
        )
    clamped_x = min(x, map_width - width)
    clamped_y = min(y, map_height - height)
    return Viewport(x=clamped_x, y=clamped_y, width=width, height=height)


def _first_int(
    root: Mapping[str, object],
    nested: Mapping[str, object] | None,
    *,
    keys: Sequence[str],
    default: int,
) -> int:
    for key in keys:
        value = root.get(key)
        if isinstance(value, int):
            return value
    if nested is not None:
        for key in keys:
            value = nested.get(key)
            if isinstance(value, int):
                return value
    return default


def _require_mapping(payload: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context.capitalize()} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _read_optional_str(payload: Mapping[str, object], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    return str(value)


def _read_optional_int(payload: Mapping[str, object], field_name: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Field '{field_name}' must be an integer.")
    return value


def _parse_prediction_tensor(payload: object) -> list[list[list[float]]]:
    if not isinstance(payload, list):
        raise ValueError("Field 'prediction' must be a nested list.")
    tensor: list[list[list[float]]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ValueError("Prediction rows must be lists.")
        parsed_row: list[list[float]] = []
        for cell in row:
            if not isinstance(cell, list):
                raise ValueError("Prediction cells must be lists.")
            parsed_cell: list[float] = []
            for probability in cell:
                if isinstance(probability, bool) or not isinstance(
                    probability, (int, float)
                ):
                    raise ValueError("Prediction probabilities must be numeric.")
                parsed_cell.append(float(probability))
            parsed_row.append(parsed_cell)
        tensor.append(parsed_row)
    return tensor


def _sample_world(initial_grid: list[list[int]], *, rng: Random) -> list[list[int]]:
    height = len(initial_grid)
    width = len(initial_grid[0])
    sampled: list[list[int | None]] = [
        [
            terrain_code if terrain_code_is_static(terrain_code) else None
            for terrain_code in row
        ]
        for row in initial_grid
    ]
    coordinates = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if not terrain_code_is_static(initial_grid[y][x])
    ]
    rng.shuffle(coordinates)
    for x, y in coordinates:
        sampled[y][x] = _sample_cell(initial_grid, sampled, x=x, y=y, rng=rng)
    finalized: list[list[int]] = []
    for row in sampled:
        finalized_row: list[int] = []
        for terrain_code in row:
            if terrain_code is None:
                raise AssertionError("Sampler left a cell unresolved.")
            finalized_row.append(terrain_code)
        finalized.append(finalized_row)
    return finalized


def _sample_cell(
    initial_grid: list[list[int]],
    sampled: list[list[int | None]],
    *,
    x: int,
    y: int,
    rng: Random,
) -> int:
    initial_code = initial_grid[y][x]
    neighbor_codes = _neighbor_codes(initial_grid, sampled, x=x, y=y)
    local_counts = {
        code: neighbor_codes.count(code) for code in {0, 1, 2, 3, 4, 5, 10, 11}
    }
    coastal = local_counts[10] > 0
    active_neighbors = local_counts[1] + local_counts[2]
    ruin_neighbors = local_counts[3]
    forest_neighbors = local_counts[4]
    mountain_neighbors = local_counts[5]

    weights: dict[int, float]
    if initial_code in {0, 11}:
        weights = {
            11: 5.2 if initial_code == 11 else 2.1,
            0: 1.8 if initial_code == 0 else 0.9,
            4: 0.8 + 0.75 * forest_neighbors,
            1: 0.12 + 0.65 * active_neighbors + 0.08 * forest_neighbors,
            3: 0.03 + 0.18 * ruin_neighbors + 0.07 * active_neighbors,
        }
        if coastal:
            weights[2] = 0.04 + 0.26 * active_neighbors
    elif initial_code == 1:
        weights = {
            1: 5.4 + 0.35 * active_neighbors,
            2: 2.8 + 0.15 * active_neighbors if coastal else 0.01,
            3: 1.2 + 0.25 * ruin_neighbors + 0.15 * mountain_neighbors,
            11: 0.55,
            0: 0.35,
            4: 0.22 + 0.12 * forest_neighbors,
        }
    elif initial_code == 2:
        weights = {
            2: 5.8 if coastal else 0.02,
            1: 1.9,
            3: 1.15 + 0.2 * ruin_neighbors,
            11: 0.35,
            0: 0.25,
            4: 0.2,
        }
    elif initial_code == 3:
        weights = {
            3: 3.0 + 0.3 * ruin_neighbors,
            4: 1.9 + 0.65 * forest_neighbors,
            11: 1.35,
            0: 1.0,
            1: 0.4 + 0.55 * active_neighbors,
        }
        if coastal:
            weights[2] = 0.2 + 0.28 * active_neighbors
    elif initial_code == 4:
        weights = {
            4: 6.1 + 0.35 * forest_neighbors,
            11: 0.95,
            0: 0.22,
            1: 0.2 + 0.28 * active_neighbors,
            3: 0.04,
        }
        if coastal:
            weights[2] = 0.03 + 0.1 * active_neighbors
    else:
        raise ValueError(f"Unsupported dynamic terrain code: {initial_code}")

    if not coastal:
        weights.pop(2, None)
    for terrain_code, weight in list(weights.items()):
        weights[terrain_code] = max(0.001, weight * (0.82 + 0.36 * rng.random()))
    if initial_code in {1, 2} and rng.random() < 0.03 + 0.015 * ruin_neighbors:
        weights[3] = weights.get(3, 0.1) + 2.2
    if initial_code == 3 and rng.random() < 0.08 + 0.03 * forest_neighbors:
        weights[4] = weights.get(4, 0.1) + 1.5
    population_codes = tuple(weights)
    selected = rng.choices(
        population_codes,
        weights=[weights[terrain_code] for terrain_code in population_codes],
        k=1,
    )[0]
    return int(selected)


def _neighbor_codes(
    initial_grid: list[list[int]],
    sampled: list[list[int | None]],
    *,
    x: int,
    y: int,
) -> list[int]:
    height = len(initial_grid)
    width = len(initial_grid[0])
    neighbors: list[int] = []
    for delta_y in (-1, 0, 1):
        for delta_x in (-1, 0, 1):
            if delta_x == 0 and delta_y == 0:
                continue
            next_x = x + delta_x
            next_y = y + delta_y
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            sampled_code = sampled[next_y][next_x]
            neighbors.append(
                initial_grid[next_y][next_x] if sampled_code is None else sampled_code
            )
    return neighbors


def _crop_grid(
    grid: list[list[int]], *, x: int, y: int, width: int, height: int
) -> list[list[int]]:
    return [row[x : x + width] for row in grid[y : y + height]]


def _viewport_settlements(
    settlements: tuple[SettlementState, ...], *, viewport: Viewport
) -> list[dict[str, object]]:
    viewport_settlements: list[dict[str, object]] = []
    for settlement in settlements:
        if not settlement.alive:
            continue
        within_x = viewport.x <= settlement.x < viewport.x + viewport.width
        within_y = viewport.y <= settlement.y < viewport.y + viewport.height
        if not within_x or not within_y:
            continue
        viewport_settlements.append(
            {
                "x": settlement.x,
                "y": settlement.y,
                "population": round(settlement.population, 3),
                "food": round(settlement.food, 3),
                "wealth": round(settlement.wealth, 3),
                "defense": round(settlement.defense, 3),
                "has_port": settlement.has_port,
                "alive": settlement.alive,
                "owner_id": settlement.owner_id,
            }
        )
    return viewport_settlements


def _proxy_initial_settlements(
    settlements: tuple[InitialSettlement, ...],
) -> tuple[ProxyInitialSettlement, ...]:
    return tuple(
        ProxyInitialSettlement(
            x=settlement.x,
            y=settlement.y,
            has_port=settlement.has_port,
            alive=settlement.alive,
            owner_id=settlement.owner_id,
        )
        for settlement in settlements
    )


def _nearest_owner_id(
    initial_settlements: tuple[InitialSettlement, ...], *, x: int, y: int
) -> int:
    if not initial_settlements:
        return 1
    best = min(
        initial_settlements,
        key=lambda settlement: (
            abs(settlement.x - x) + abs(settlement.y - y),
            settlement.owner_id,
        ),
    )
    return best.owner_id


def _settlement_stats(
    grid: list[list[int]], *, x: int, y: int, has_port: bool, rng: Random
) -> dict[str, float]:
    neighbor_codes = _neighbor_codes(
        grid,
        [[cell for cell in row] for row in grid],
        x=x,
        y=y,
    )
    forest_neighbors = neighbor_codes.count(4)
    active_neighbors = neighbor_codes.count(1) + neighbor_codes.count(2)
    mountain_neighbors = neighbor_codes.count(5)
    coastal = neighbor_codes.count(10) > 0
    food = (
        0.55
        + 0.22 * forest_neighbors
        + (0.18 if coastal else 0.0)
        + 0.08 * rng.random()
    )
    wealth = (
        0.45
        + 0.15 * active_neighbors
        + (0.4 if has_port else 0.05)
        + 0.12 * rng.random()
    )
    defense = (
        0.35 + 0.12 * mountain_neighbors + 0.09 * active_neighbors + 0.08 * rng.random()
    )
    population = 1.1 + 0.5 * food + 0.35 * wealth + 0.15 * defense + 0.2 * rng.random()
    return {
        "population": round(population, 3),
        "food": round(food, 3),
        "wealth": round(wealth, 3),
        "defense": round(defense, 3),
    }
