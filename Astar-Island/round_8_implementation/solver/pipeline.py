from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import cast

from .api import AstarIslandClient, query_response_to_payload
from .baseline import blend_observation_into_tensor, build_baseline_tensor
from .cache import LocalCache
from .contract import (
    DEFAULT_MAP_HEIGHT,
    DEFAULT_MAP_WIDTH,
    DEFAULT_QUERY_BUDGET,
    DEFAULT_SEED_COUNT,
    MAX_VIEWPORT_SIZE,
    MIN_VIEWPORT_SIZE,
    canonical_mapping_artifact,
)
from .debug_visualization import DebugTrace, ViewportQuery, render_debug_bundle
from .models import (
    ObservationRecord,
    QueryResponse,
    RoundDetail,
    RunSummary,
    SeedSolveResult,
    SeedState,
    SettlementObservation,
    Viewport,
)
from .planner import rank_candidate_viewports
from .rollouts import aggregate_rollouts
from .validator import validate_grid, validate_mapping, validate_prediction_tensor


def parse_round_detail_payload(payload: object) -> RoundDetail:
    if not isinstance(payload, dict):
        raise ValueError("Round detail payload must be an object.")
    payload_mapping = cast(Mapping[str, object], payload)
    round_id_value = (
        payload_mapping.get("id") or payload_mapping.get("round_id") or "unknown-round"
    )
    round_id = str(round_id_value)
    map_width = _read_int(payload_mapping.get("map_width"), default=DEFAULT_MAP_WIDTH)
    map_height = _read_int(
        payload_mapping.get("map_height"), default=DEFAULT_MAP_HEIGHT
    )
    seeds_count = _read_int(
        payload_mapping.get("seeds_count"), default=DEFAULT_SEED_COUNT
    )
    raw_initial_states = payload_mapping.get("initial_states")
    if not isinstance(raw_initial_states, list):
        raise ValueError("Round detail field 'initial_states' must be a list.")

    initial_states = tuple(
        _parse_initial_grid(item, width=map_width, height=map_height)
        for item in raw_initial_states
    )
    if len(initial_states) != seeds_count:
        raise ValueError(
            f"Seeds count mismatch: expected {seeds_count}, got {len(initial_states)}"
        )

    return RoundDetail(
        round_id=round_id,
        map_width=map_width,
        map_height=map_height,
        seeds_count=seeds_count,
        initial_states=initial_states,
        status=_optional_str(payload_mapping.get("status")),
        prediction_window_minutes=_optional_int(
            payload_mapping.get("prediction_window_minutes")
        ),
    )


def create_seed_states(round_detail: RoundDetail) -> list[SeedState]:
    validate_mapping()
    states: list[SeedState] = []
    for seed_index, grid in enumerate(round_detail.initial_states):
        validate_grid(
            grid, width=round_detail.map_width, height=round_detail.map_height
        )
        states.append(
            SeedState(
                seed_index=seed_index,
                initial_grid=grid,
                current_tensor=build_baseline_tensor(grid),
            )
        )
    return states


def solve_round(
    *,
    round_detail: RoundDetail,
    cache: LocalCache,
    viewport_width: int,
    viewport_height: int,
    planned_queries_per_seed: int,
    rollout_count: int,
    random_seed: int,
    live_client: AstarIslandClient | None,
    execute_live_queries: bool,
    submit_predictions: bool,
) -> RunSummary:
    if not MIN_VIEWPORT_SIZE <= viewport_width <= MAX_VIEWPORT_SIZE:
        raise ValueError("Viewport width must be between 5 and 15.")
    if not MIN_VIEWPORT_SIZE <= viewport_height <= MAX_VIEWPORT_SIZE:
        raise ValueError("Viewport height must be between 5 and 15.")

    cache.ensure()
    cache.save_json(cache.mapping_artifact_path(), canonical_mapping_artifact())
    cache.save_json(
        cache.round_detail_path(round_detail.round_id),
        round_detail_to_payload(round_detail),
    )

    run_id = _build_run_id(round_detail.round_id)
    cache.save_json(
        cache.run_config_path(run_id),
        {
            "round_id": round_detail.round_id,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "planned_queries_per_seed": planned_queries_per_seed,
            "rollout_count": rollout_count,
            "random_seed": random_seed,
            "execute_live_queries": execute_live_queries,
            "submit_predictions": submit_predictions,
            "base_url": live_client.base_url if live_client is not None else None,
        },
    )

    seed_results: list[SeedSolveResult] = []
    states = create_seed_states(round_detail)
    query_budget_remaining = DEFAULT_QUERY_BUDGET

    for state in states:
        limit = min(planned_queries_per_seed, query_budget_remaining)
        planned = rank_candidate_viewports(
            state,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            limit=limit,
        )
        state.query_plan = [candidate.viewport for candidate in planned]
        query_budget_remaining -= len(state.query_plan)

        if execute_live_queries:
            if live_client is None:
                raise ValueError("Live queries requested without a live client.")
            for step, viewport in enumerate(state.query_plan):
                response = _load_or_fetch_query(
                    cache=cache,
                    client=live_client,
                    round_id=round_detail.round_id,
                    seed_index=state.seed_index,
                    viewport=viewport,
                )
                assimilate_query_response(state=state, step=step, response=response)

        prediction = aggregate_rollouts(
            state,
            rollout_count=rollout_count,
            random_seed=random_seed + state.seed_index,
        )
        validate_prediction_tensor(
            prediction,
            width=round_detail.map_width,
            height=round_detail.map_height,
        )

        prediction_path = cache.prediction_path(run_id, state.seed_index)
        cache.save_json(
            prediction_path,
            {
                "round_id": round_detail.round_id,
                "seed_index": state.seed_index,
                "prediction": prediction,
            },
        )

        debug_trace = _build_debug_trace(state)
        debug_output_dir = cache.debug_output_dir(run_id, state.seed_index)
        render_debug_bundle(debug_trace, debug_output_dir)

        if submit_predictions:
            if live_client is None:
                raise ValueError("Submission requested without a live client.")
            response_payload = live_client.submit_prediction(
                round_id=round_detail.round_id,
                seed_index=state.seed_index,
                prediction=prediction,
            )
            cache.save_json(
                prediction_path.with_name(
                    f"seed-{state.seed_index:02d}-submit-response.json"
                ),
                response_payload,
            )

        seed_results.append(
            SeedSolveResult(
                seed_index=state.seed_index,
                prediction=prediction,
                planned_viewports=tuple(state.query_plan),
                observations_used=tuple(state.observation_history),
                prediction_path=str(prediction_path),
                debug_output_dir=str(debug_output_dir),
            )
        )

    summary = RunSummary(
        round_id=round_detail.round_id,
        run_id=run_id,
        output_root=str(cache.root),
        seed_results=tuple(seed_results),
    )
    cache.save_json(cache.run_summary_path(run_id), run_summary_to_payload(summary))
    return summary


def assimilate_query_response(
    *, state: SeedState, step: int, response: QueryResponse, source: str = "live"
) -> None:
    for row_offset, row in enumerate(response.grid):
        for column_offset, terrain_code in enumerate(row):
            x = response.viewport.x + column_offset
            y = response.viewport.y + row_offset
            state.observed_cells.setdefault((x, y), []).append(terrain_code)
            blend_observation_into_tensor(
                state.current_tensor,
                x=x,
                y=y,
                observed_terrain_code=terrain_code,
            )
    state.observation_history.append(
        ObservationRecord(step=step, response=response, source=source)
    )


def build_live_client_from_environment(
    *, token_env_var: str, base_url: str
) -> AstarIslandClient | None:
    token = os.environ.get(token_env_var)
    if token is None:
        return None
    return AstarIslandClient(base_url=base_url, token=token)


def round_detail_to_payload(round_detail: RoundDetail) -> dict[str, object]:
    return {
        "round_id": round_detail.round_id,
        "map_width": round_detail.map_width,
        "map_height": round_detail.map_height,
        "seeds_count": round_detail.seeds_count,
        "initial_states": round_detail.initial_states,
        "status": round_detail.status,
        "prediction_window_minutes": round_detail.prediction_window_minutes,
    }


def run_summary_to_payload(summary: RunSummary) -> dict[str, object]:
    return {
        "round_id": summary.round_id,
        "run_id": summary.run_id,
        "output_root": summary.output_root,
        "seed_results": [
            {
                "seed_index": result.seed_index,
                "planned_viewports": [
                    {
                        "x": viewport.x,
                        "y": viewport.y,
                        "width": viewport.width,
                        "height": viewport.height,
                    }
                    for viewport in result.planned_viewports
                ],
                "observations_used": [
                    {
                        "step": record.step,
                        "source": record.source,
                        "viewport": {
                            "x": record.response.viewport.x,
                            "y": record.response.viewport.y,
                            "width": record.response.viewport.width,
                            "height": record.response.viewport.height,
                        },
                    }
                    for record in result.observations_used
                ],
                "prediction_path": result.prediction_path,
                "debug_output_dir": result.debug_output_dir,
            }
            for result in summary.seed_results
        ],
    }


def _build_debug_trace(state: SeedState) -> DebugTrace:
    queries: list[ViewportQuery] = []
    if state.observation_history:
        for record in state.observation_history:
            queries.append(
                ViewportQuery(
                    step=record.step,
                    x=record.response.viewport.x,
                    y=record.response.viewport.y,
                    width=record.response.viewport.width,
                    height=record.response.viewport.height,
                    screenshot=_normalize_screenshot_grid(record.response.grid),
                    note=f"{record.source} observation",
                )
            )
    else:
        for step, viewport in enumerate(state.query_plan):
            queries.append(
                ViewportQuery(
                    step=step,
                    x=viewport.x,
                    y=viewport.y,
                    width=viewport.width,
                    height=viewport.height,
                    screenshot=_normalize_screenshot_grid(
                        _crop_grid(
                            state.initial_grid,
                            x=viewport.x,
                            y=viewport.y,
                            width=viewport.width,
                            height=viewport.height,
                        )
                    ),
                    note="planned viewport preview",
                )
            )
    return DebugTrace(
        title=f"Astar Island seed {state.seed_index} debug trace",
        start_grid=_normalize_screenshot_grid(state.initial_grid),
        queries=tuple(queries),
    )


def _load_or_fetch_query(
    *,
    cache: LocalCache,
    client: AstarIslandClient,
    round_id: str,
    seed_index: int,
    viewport: Viewport,
) -> QueryResponse:
    viewport_key = f"x-{viewport.x:03d}_y-{viewport.y:03d}_w-{viewport.width:03d}_h-{viewport.height:03d}"
    cache_path = cache.query_response_path(round_id, seed_index, viewport_key)
    if cache_path.exists():
        payload = cache.load_json(cache_path)
        return _parse_query_response_payload(payload)
    response = client.simulate(
        round_id=round_id, seed_index=seed_index, viewport=viewport
    )
    cache.save_json(cache_path, query_response_to_payload(response))
    return response


def _parse_query_response_payload(payload: object) -> QueryResponse:
    if not isinstance(payload, dict):
        raise ValueError("Cached query response must be a JSON object.")
    payload_mapping = cast(Mapping[str, object], payload)
    viewport_payload = payload_mapping.get("viewport")
    if not isinstance(viewport_payload, dict):
        raise ValueError("Cached query response is missing 'viewport'.")
    viewport_mapping = cast(Mapping[str, object], viewport_payload)

    viewport = Viewport(
        x=_require_int(viewport_mapping.get("x"), field_name="viewport.x"),
        y=_require_int(viewport_mapping.get("y"), field_name="viewport.y"),
        width=_require_int(viewport_mapping.get("width"), field_name="viewport.width"),
        height=_require_int(
            viewport_mapping.get("height"), field_name="viewport.height"
        ),
    )
    settlements_payload_obj = payload_mapping.get("settlements")
    settlements_payload = (
        [] if settlements_payload_obj is None else settlements_payload_obj
    )
    if not isinstance(settlements_payload, list):
        raise ValueError("Cached query response field 'settlements' must be a list.")

    return QueryResponse(
        viewport=viewport,
        grid=_parse_grid_payload(payload_mapping.get("grid")),
        settlements=tuple(
            _parse_settlement_payload(item) for item in settlements_payload
        ),
        queries_used=_optional_int(payload_mapping.get("queries_used")),
        queries_max=_optional_int(payload_mapping.get("queries_max")),
    )


def _parse_initial_grid(item: object, *, width: int, height: int) -> list[list[int]]:
    raw_grid = item
    if isinstance(item, dict):
        item_mapping = cast(Mapping[str, object], item)
        if "grid" in item_mapping:
            raw_grid = item_mapping["grid"]
        elif "initial_grid" in item_mapping:
            raw_grid = item_mapping["initial_grid"]
    grid = _parse_grid_payload(raw_grid)
    validate_grid(grid, width=width, height=height)
    return grid


def _crop_grid(
    grid: list[list[int]], *, x: int, y: int, width: int, height: int
) -> list[list[int]]:
    return [row[x : x + width] for row in grid[y : y + height]]


def _normalize_screenshot_grid(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


def _build_run_id(round_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{round_id}-{timestamp}"


def _read_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError(f"Expected integer value, got {value!r}")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Expected integer value, got {value!r}")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _parse_grid_payload(raw_grid: object) -> list[list[int]]:
    if not isinstance(raw_grid, list):
        raise ValueError("Grid payload must be a list.")
    grid: list[list[int]] = []
    for row in raw_grid:
        if not isinstance(row, list):
            raise ValueError("Each grid row must be a list.")
        grid.append([_require_int(cell, field_name="grid cell") for cell in row])
    return grid


def _parse_settlement_payload(item: object) -> SettlementObservation:
    if not isinstance(item, dict):
        raise ValueError("Settlement payload must be an object.")
    item_mapping = cast(Mapping[str, object], item)
    return SettlementObservation(
        x=_require_int(item_mapping.get("x"), field_name="settlement.x"),
        y=_require_int(item_mapping.get("y"), field_name="settlement.y"),
        population=_require_float(
            item_mapping.get("population"), field_name="settlement.population"
        ),
        food=_require_float(item_mapping.get("food"), field_name="settlement.food"),
        wealth=_require_float(
            item_mapping.get("wealth"), field_name="settlement.wealth"
        ),
        defense=_require_float(
            item_mapping.get("defense"), field_name="settlement.defense"
        ),
        has_port=_require_bool(
            item_mapping.get("has_port"), field_name="settlement.has_port"
        ),
        alive=_require_bool(item_mapping.get("alive"), field_name="settlement.alive"),
        owner_id=_require_int(
            item_mapping.get("owner_id"), field_name="settlement.owner_id"
        ),
    )


def _require_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Field '{field_name}' must be an integer.")
    return value


def _require_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Field '{field_name}' must be numeric.")
    return float(value)


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Field '{field_name}' must be a boolean.")
    return value
