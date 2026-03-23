from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState, SimulateResponse
from astar_twin.contracts.types import MAX_QUERIES, MAX_SEEDS
from astar_twin.solver.inference.posterior import (
    PosteriorState,
    create_posterior,
    prune_and_resample_bootstrap,
    resample_if_needed,
    temper_if_collapsed,
    update_posterior,
)
from astar_twin.solver.interfaces import SolverAdapter
from astar_twin.solver.policy.allocator import (
    RESERVE_QUERIES,
    AllocationState,
    check_contradiction_triggers,
    initialize_allocation,
    plan_bootstrap_queries,
    plan_reserve_queries,
    record_query,
    select_adaptive_batch,
    transition_phase,
)
from astar_twin.solver.policy.hotspots import ViewportCandidate
from astar_twin.solver.predict.posterior_mc import (
    DEFAULT_SIMS_PER_SEED,
    predict_all_seeds,
    predict_seed,
)


@dataclass
class QueryRecord:
    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    phase: str
    utility_score: float = 0.0
    ess_after: float = 0.0


@dataclass
class SolveResult:
    tensors: list[NDArray[np.float64]]
    transcript: list[QueryRecord] = field(default_factory=list)
    total_queries_used: int = 0
    runtime_seconds: float = 0.0
    final_ess: float = 0.0
    contradiction_triggered: bool = False


_INTERMEDIATE_TOP_K = 3
_INTERMEDIATE_SIMS_PER_SEED = 6
_GLOBAL_REFRESH_INTERVAL = 5


def _issue_query(
    adapter: SolverAdapter,
    round_id: str,
    seed_index: int,
    viewport_x: int,
    viewport_y: int,
    viewport_w: int,
    viewport_h: int,
) -> SimulateResponse:
    return adapter.simulate(
        round_id,
        seed_index,
        viewport_x,
        viewport_y,
        viewport_w,
        viewport_h,
    )


def _predict_seed_map(
    posterior: PosteriorState,
    initial_states: list[InitialState],
    height: int,
    width: int,
    base_seed: int,
) -> dict[int, NDArray[np.float64]]:
    seed_predictions: dict[int, NDArray[np.float64]] = {}
    for seed_index, initial_state in enumerate(initial_states):
        tensor, _ = predict_seed(
            posterior,
            initial_state,
            seed_index=seed_index,
            map_height=height,
            map_width=width,
            top_k=min(_INTERMEDIATE_TOP_K, len(posterior.particles)),
            sims_per_seed=_INTERMEDIATE_SIMS_PER_SEED,
            base_seed=base_seed + seed_index * 1000,
        )
        seed_predictions[seed_index] = tensor
    return seed_predictions


def _refresh_single_seed_prediction(
    *,
    seed_predictions: dict[int, NDArray[np.float64]],
    posterior: PosteriorState,
    initial_states: list[InitialState],
    height: int,
    width: int,
    seed_index: int,
    base_seed: int,
) -> dict[int, NDArray[np.float64]]:
    tensor, _ = predict_seed(
        posterior,
        initial_states[seed_index],
        seed_index=seed_index,
        map_height=height,
        map_width=width,
        top_k=min(_INTERMEDIATE_TOP_K, len(posterior.particles)),
        sims_per_seed=_INTERMEDIATE_SIMS_PER_SEED,
        base_seed=base_seed,
    )
    updated_predictions = dict(seed_predictions)
    updated_predictions[seed_index] = tensor
    return updated_predictions


def _execute_plan_step(
    *,
    adapter: SolverAdapter,
    round_id: str,
    initial_states: list[InitialState],
    transcript: list[QueryRecord],
    posterior: PosteriorState,
    alloc: AllocationState,
    query_counter: int,
    planned_query: tuple[int, ViewportCandidate] | None,
    phase: str,
    n_inner_runs: int,
    base_seed: int,
) -> tuple[PosteriorState, int, bool]:
    if planned_query is None or alloc.queries_remaining <= 0 or query_counter >= MAX_QUERIES:
        return posterior, query_counter, False

    seed_idx, viewport = planned_query
    try:
        response = _issue_query(
            adapter,
            round_id,
            seed_idx,
            viewport.x,
            viewport.y,
            viewport.w,
            viewport.h,
        )
    except RuntimeError:
        return posterior, query_counter, False

    posterior = update_posterior(
        posterior,
        response,
        initial_states[seed_idx],
        n_inner_runs=n_inner_runs,
        base_seed=base_seed + query_counter * 100,
    )
    record_query(alloc, seed_idx, viewport, phase)
    transcript.append(
        QueryRecord(
            seed_index=seed_idx,
            viewport_x=viewport.x,
            viewport_y=viewport.y,
            viewport_w=viewport.w,
            viewport_h=viewport.h,
            phase=phase,
            utility_score=viewport.score,
            ess_after=posterior.ess,
        )
    )
    return posterior, query_counter + 1, True


def solve_high_value_bidirectional(
    adapter: SolverAdapter,
    round_id: str,
    n_particles: int = 24,
    n_inner_runs: int = 6,
    sims_per_seed: int = DEFAULT_SIMS_PER_SEED,
    base_seed: int = 0,
) -> SolveResult:
    t_start = time.monotonic()
    transcript: list[QueryRecord] = []

    detail = adapter.get_round_detail(round_id)
    height = detail.map_height
    width = detail.map_width
    initial_states = detail.initial_states
    n_seeds = min(len(initial_states), MAX_SEEDS)
    active_states = initial_states[:n_seeds]

    posterior = create_posterior(n_particles=n_particles, seed=base_seed)
    alloc = initialize_allocation(active_states, height, width)
    bootstrap_plan = plan_bootstrap_queries(alloc)
    query_counter = 0

    for planned_query in bootstrap_plan:
        posterior, query_counter, executed = _execute_plan_step(
            adapter=adapter,
            round_id=round_id,
            initial_states=active_states,
            transcript=transcript,
            posterior=posterior,
            alloc=alloc,
            query_counter=query_counter,
            planned_query=planned_query,
            phase="bootstrap",
            n_inner_runs=n_inner_runs,
            base_seed=base_seed,
        )
        if not executed:
            break

    transition_phase(alloc)
    posterior = prune_and_resample_bootstrap(
        posterior,
        top_k=8,
        target_n=12,
        seed=base_seed + 1000,
    )

    seed_predictions = _predict_seed_map(
        posterior,
        active_states,
        height,
        width,
        base_seed=base_seed + 3000,
    )

    adaptive_steps = min(30, alloc.queries_remaining)
    for step in range(adaptive_steps):
        batch = select_adaptive_batch(
            alloc,
            posterior,
            active_states,
            seed_predictions=seed_predictions,
            batch_size=1,
        )
        planned_query = batch[0] if batch else None
        selected_seed_index = planned_query[0] if planned_query is not None else None
        posterior, query_counter, executed = _execute_plan_step(
            adapter=adapter,
            round_id=round_id,
            initial_states=active_states,
            transcript=transcript,
            posterior=posterior,
            alloc=alloc,
            query_counter=query_counter,
            planned_query=planned_query,
            phase="adaptive",
            n_inner_runs=n_inner_runs,
            base_seed=base_seed,
        )
        if not executed:
            break

        posterior = resample_if_needed(
            posterior,
            ess_threshold=6.0,
            seed=base_seed + 2000 + step,
        )
        posterior = temper_if_collapsed(posterior)
        if selected_seed_index is not None:
            seed_predictions = _refresh_single_seed_prediction(
                seed_predictions=seed_predictions,
                posterior=posterior,
                initial_states=active_states,
                height=height,
                width=width,
                seed_index=selected_seed_index,
                base_seed=base_seed + 4000 + step * 37,
            )
        if step % _GLOBAL_REFRESH_INTERVAL == _GLOBAL_REFRESH_INTERVAL - 1:
            seed_predictions = _predict_seed_map(
                posterior,
                active_states,
                height,
                width,
                base_seed=base_seed + 5000 + step * 53,
            )

    transition_phase(alloc)
    contradiction = check_contradiction_triggers(alloc, posterior)

    reserve_steps = min(RESERVE_QUERIES, alloc.queries_remaining)
    for step in range(reserve_steps):
        reserve_batch = plan_reserve_queries(
            alloc,
            posterior,
            active_states,
            seed_predictions=seed_predictions,
            n_queries=1,
        )
        planned_query = reserve_batch[0] if reserve_batch else None
        selected_seed_index = planned_query[0] if planned_query is not None else None
        posterior, query_counter, executed = _execute_plan_step(
            adapter=adapter,
            round_id=round_id,
            initial_states=active_states,
            transcript=transcript,
            posterior=posterior,
            alloc=alloc,
            query_counter=query_counter,
            planned_query=planned_query,
            phase="reserve",
            n_inner_runs=n_inner_runs,
            base_seed=base_seed,
        )
        if not executed:
            break

        posterior = resample_if_needed(
            posterior,
            ess_threshold=6.0,
            seed=base_seed + 6000 + step,
        )
        posterior = temper_if_collapsed(posterior)
        if selected_seed_index is not None:
            seed_predictions = _refresh_single_seed_prediction(
                seed_predictions=seed_predictions,
                posterior=posterior,
                initial_states=active_states,
                height=height,
                width=width,
                seed_index=selected_seed_index,
                base_seed=base_seed + 7000 + step * 37,
            )
        if step % _GLOBAL_REFRESH_INTERVAL == _GLOBAL_REFRESH_INTERVAL - 1:
            seed_predictions = _predict_seed_map(
                posterior,
                active_states,
                height,
                width,
                base_seed=base_seed + 8000 + step * 53,
            )

    transition_phase(alloc)

    elapsed = time.monotonic() - t_start
    runtime_fraction = elapsed / 9000.0
    tensors, _ = predict_all_seeds(
        posterior,
        active_states,
        map_height=height,
        map_width=width,
        top_k=6,
        sims_per_seed=sims_per_seed,
        base_seed=base_seed + 9000,
        runtime_fraction=runtime_fraction,
    )

    return SolveResult(
        tensors=tensors,
        transcript=transcript,
        total_queries_used=alloc.queries_used,
        runtime_seconds=time.monotonic() - t_start,
        final_ess=posterior.ess,
        contradiction_triggered=contradiction,
    )
