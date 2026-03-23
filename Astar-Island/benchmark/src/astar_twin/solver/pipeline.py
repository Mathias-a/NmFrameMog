"""End-to-end solver pipeline.

Accepts an adapter and returns 5 prediction tensors (one per seed).
This module is the single entrypoint that orchestrates:
  1. load round detail
  2. generate bootstrap candidates
  3. issue bootstrap queries
  4. update posterior
  5. iterate adaptive + reserve
  6. generate final tensors
  7. return metrics + transcripts
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.types import MAX_QUERIES, MAX_SEEDS
from astar_twin.solver.inference.posterior import (
    create_posterior,
    prune_and_resample_bootstrap,
    resample_if_needed,
    temper_if_collapsed,
    update_posterior,
)
from astar_twin.solver.interfaces import SolverAdapter
from astar_twin.solver.policy.allocator import (
    ADAPTIVE_BATCH_SIZE,
    ADAPTIVE_BATCHES,
    RESERVE_QUERIES,
    check_contradiction_triggers,
    initialize_allocation,
    plan_bootstrap_queries,
    plan_reserve_queries,
    record_query,
    select_adaptive_batch,
    transition_phase,
)
from astar_twin.solver.predict.posterior_mc import (
    DEFAULT_SIMS_PER_SEED,
    predict_all_seeds,
)


@dataclass
class QueryRecord:
    """Single query issued during a solve."""

    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    phase: str  # "bootstrap" | "adaptive" | "reserve"
    utility_score: float = 0.0
    ess_after: float = 0.0


@dataclass
class SolveResult:
    """Complete result from one solver run."""

    tensors: list[NDArray[np.float64]]
    transcript: list[QueryRecord] = field(default_factory=list)
    total_queries_used: int = 0
    runtime_seconds: float = 0.0
    final_ess: float = 0.0
    contradiction_triggered: bool = False


def _issue_query(
    adapter: SolverAdapter,
    round_id: str,
    seed_index: int,
    viewport_x: int,
    viewport_y: int,
    viewport_w: int,
    viewport_h: int,
):
    """Issue a single simulate query through the adapter."""
    return adapter.simulate(
        round_id,
        seed_index,
        viewport_x,
        viewport_y,
        viewport_w,
        viewport_h,
    )


def solve(
    adapter: SolverAdapter,
    round_id: str,
    n_particles: int = 24,
    n_inner_runs: int = 6,
    sims_per_seed: int = DEFAULT_SIMS_PER_SEED,
    base_seed: int = 0,
    use_experts: bool = True,
) -> SolveResult:
    """Run the full solver pipeline against the given round.

    Steps:
      1. Load round detail for all seeds
      2. Generate bootstrap candidates (hotspots)
      3. Issue 10 bootstrap queries (2 per seed)
      4. Update posterior after each observation
      5. Prune and resample after bootstrap
      6. Run 6 adaptive batches of 5 globally-selected queries
      7. Check contradiction triggers → release reserve if needed
      8. Generate posterior-predictive tensors for all 5 seeds
      9. Return structured SolveResult with transcript

    Args:
        adapter: SolverAdapter (benchmark or prod).
        round_id: Round identifier.
        n_particles: Initial particle count (default 24).
        n_inner_runs: Inner MC runs per likelihood update (default 6).
        sims_per_seed: Final prediction MC runs per seed (default 64).
        base_seed: Base seed for deterministic execution.

    Returns:
        SolveResult with tensors, transcript, and metrics.
    """
    t_start = time.monotonic()
    transcript: list[QueryRecord] = []

    # 1. Load round detail
    detail = adapter.get_round_detail(round_id)
    height = detail.map_height
    width = detail.map_width
    initial_states = detail.initial_states
    n_seeds = min(len(initial_states), MAX_SEEDS)

    # 2. Initialize posterior and allocation
    posterior = create_posterior(n_particles=n_particles, seed=base_seed, use_experts=use_experts)
    alloc = initialize_allocation(initial_states[:n_seeds], height, width)

    # 3. Bootstrap phase: 2 queries per seed
    bootstrap_plan = plan_bootstrap_queries(alloc)
    query_counter = 0

    for seed_idx, vp in bootstrap_plan:
        if query_counter >= MAX_QUERIES:
            break
        try:
            response = _issue_query(
                adapter,
                round_id,
                seed_idx,
                vp.x,
                vp.y,
                vp.w,
                vp.h,
            )
        except RuntimeError:
            # Budget exhausted
            break

        # Update posterior with observation
        posterior = update_posterior(
            posterior,
            response,
            initial_states[seed_idx],
            n_inner_runs=n_inner_runs,
            base_seed=base_seed + query_counter * 100,
        )
        record_query(alloc, seed_idx, vp, "bootstrap")
        transcript.append(
            QueryRecord(
                seed_index=seed_idx,
                viewport_x=vp.x,
                viewport_y=vp.y,
                viewport_w=vp.w,
                viewport_h=vp.h,
                phase="bootstrap",
                utility_score=vp.score,
                ess_after=posterior.ess,
            )
        )
        query_counter += 1

    # 4. Post-bootstrap: prune and resample
    transition_phase(alloc)
    if not use_experts:
        posterior = prune_and_resample_bootstrap(
            posterior,
            top_k=8,
            target_n=12,
            seed=base_seed + 1000,
        )
    else:
        posterior.phase = "adaptive"

    bootstrap_tensors, _ = predict_all_seeds(
        posterior,
        initial_states[:n_seeds],
        map_height=height,
        map_width=width,
        top_k=min(4, len(posterior.particles)),
        sims_per_seed=16,
        base_seed=base_seed + 3000,
    )
    seed_predictions: dict[int, NDArray[np.float64]] = {
        i: tensor for i, tensor in enumerate(bootstrap_tensors)
    }

    # 5. Adaptive phase: 6 batches of 5

    for batch_num in range(ADAPTIVE_BATCHES):
        if alloc.queries_remaining <= 0:
            break

        batch = select_adaptive_batch(
            alloc,
            posterior,
            initial_states[:n_seeds],
            seed_predictions=seed_predictions,
            batch_size=ADAPTIVE_BATCH_SIZE,
        )

        for seed_idx, vp in batch:
            if alloc.queries_remaining <= 0:
                break
            try:
                response = _issue_query(
                    adapter,
                    round_id,
                    seed_idx,
                    vp.x,
                    vp.y,
                    vp.w,
                    vp.h,
                )
            except RuntimeError:
                break

            posterior = update_posterior(
                posterior,
                response,
                initial_states[seed_idx],
                n_inner_runs=n_inner_runs,
                base_seed=base_seed + query_counter * 100,
            )
            record_query(alloc, seed_idx, vp, "adaptive")
            transcript.append(
                QueryRecord(
                    seed_index=seed_idx,
                    viewport_x=vp.x,
                    viewport_y=vp.y,
                    viewport_w=vp.w,
                    viewport_h=vp.h,
                    phase="adaptive",
                    utility_score=vp.score,
                    ess_after=posterior.ess,
                )
            )
            query_counter += 1

        # Resample if ESS collapsed
        if not use_experts:
            posterior = resample_if_needed(
                posterior,
                ess_threshold=6.0,
                seed=base_seed + 2000 + batch_num,
            )
        # Temper if top particle dominates
        posterior = temper_if_collapsed(posterior)

        batch_tensors, _ = predict_all_seeds(
            posterior,
            initial_states[:n_seeds],
            map_height=height,
            map_width=width,
            top_k=min(4, len(posterior.particles)),
            sims_per_seed=16,
            base_seed=base_seed + 4000 + batch_num * 100,
        )
        seed_predictions = {i: tensor for i, tensor in enumerate(batch_tensors)}

    transition_phase(alloc)
    contradiction = check_contradiction_triggers(alloc, posterior, initial_states[:n_seeds])

    reserve_remaining = min(RESERVE_QUERIES, alloc.queries_remaining)
    if reserve_remaining > 0:
        batch_1_size = min(5, reserve_remaining)
        batch_2_size = reserve_remaining - batch_1_size

        reserve_batch_1 = plan_reserve_queries(
            alloc,
            posterior,
            initial_states[:n_seeds],
            seed_predictions=seed_predictions,
            n_queries=batch_1_size,
        )

        for seed_idx, vp in reserve_batch_1:
            if alloc.queries_remaining <= 0:
                break
            try:
                response = _issue_query(
                    adapter,
                    round_id,
                    seed_idx,
                    vp.x,
                    vp.y,
                    vp.w,
                    vp.h,
                )
            except RuntimeError:
                break

            posterior = update_posterior(
                posterior,
                response,
                initial_states[seed_idx],
                n_inner_runs=n_inner_runs,
                base_seed=base_seed + query_counter * 100,
            )
            record_query(alloc, seed_idx, vp, "reserve")
            transcript.append(
                QueryRecord(
                    seed_index=seed_idx,
                    viewport_x=vp.x,
                    viewport_y=vp.y,
                    viewport_w=vp.w,
                    viewport_h=vp.h,
                    phase="reserve",
                    utility_score=vp.score,
                    ess_after=posterior.ess,
                )
            )
            query_counter += 1

        if not use_experts:
            posterior = resample_if_needed(
                posterior,
                ess_threshold=6.0,
                seed=base_seed + 6000,
            )
        posterior = temper_if_collapsed(posterior)

        if batch_2_size > 0:
            reserve_batch_2 = plan_reserve_queries(
                alloc,
                posterior,
                initial_states[:n_seeds],
                seed_predictions=seed_predictions,
                n_queries=batch_2_size,
            )

            for seed_idx, vp in reserve_batch_2:
                if alloc.queries_remaining <= 0:
                    break
                try:
                    response = _issue_query(
                        adapter,
                        round_id,
                        seed_idx,
                        vp.x,
                        vp.y,
                        vp.w,
                        vp.h,
                    )
                except RuntimeError:
                    break

                posterior = update_posterior(
                    posterior,
                    response,
                    initial_states[seed_idx],
                    n_inner_runs=n_inner_runs,
                    base_seed=base_seed + query_counter * 100,
                )
                record_query(alloc, seed_idx, vp, "reserve")
                transcript.append(
                    QueryRecord(
                        seed_index=seed_idx,
                        viewport_x=vp.x,
                        viewport_y=vp.y,
                        viewport_w=vp.w,
                        viewport_h=vp.h,
                        phase="reserve",
                        utility_score=vp.score,
                        ess_after=posterior.ess,
                    )
                )
                query_counter += 1

    transition_phase(alloc)

    # 7. Generate posterior-predictive tensors
    elapsed = time.monotonic() - t_start
    # Rough ceiling: 2.5h = 9000s. If > 80% of that, fallback.
    runtime_fraction = elapsed / 9000.0

    tensors, prediction_metrics = predict_all_seeds(
        posterior,
        initial_states[:n_seeds],
        map_height=height,
        map_width=width,
        top_k=6,
        sims_per_seed=sims_per_seed,
        base_seed=base_seed + 5000,
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
