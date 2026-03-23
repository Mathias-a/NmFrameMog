"""End-to-end solver pipeline.

Accepts an adapter and returns 5 prediction tensors (one per seed).
This module is the single entrypoint that orchestrates:
  1. load round detail
  2. build structural anchors for hybrid fallback
  3. generate bootstrap candidates
  4. issue bootstrap queries
  5. update posterior
  6. iterate adaptive + reserve
  7. compute per-seed confidence
  8. generate final tensors with hybrid anchor blending
  9. return metrics + transcripts
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import SimulateResponse
from astar_twin.contracts.types import MAX_QUERIES, MAX_SEEDS
from astar_twin.solver.inference.confidence import (
    PosteriorConfidence,
    compute_confidence,
)
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
    plan_calibration_bootstrap_queries,
    plan_reserve_queries,
    record_query,
    select_adaptive_batch,
    transition_phase,
)
from astar_twin.solver.predict.hedge import apply_anchor_hedge, compute_blend_weight
from astar_twin.solver.predict.posterior_mc import (
    DEFAULT_SIMS_PER_SEED,
    predict_all_seeds,
)
from astar_twin.solver.predict.structural_anchor import build_structural_anchors


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
class HybridMetadata:
    """Per-seed confidence and hedge decision metadata."""

    confidences: list[PosteriorConfidence]
    hedge_modes: list[str]
    blend_weights: list[float]


@dataclass
class SolveResult:
    """Complete result from one solver run."""

    tensors: list[NDArray[np.float64]]
    transcript: list[QueryRecord] = field(default_factory=list)
    total_queries_used: int = 0
    runtime_seconds: float = 0.0
    final_ess: float = 0.0
    contradiction_triggered: bool = False
    hybrid: HybridMetadata | None = None


def _issue_query(
    adapter: SolverAdapter,
    round_id: str,
    seed_index: int,
    viewport_x: int,
    viewport_y: int,
    viewport_w: int,
    viewport_h: int,
) -> SimulateResponse:
    """Issue a single simulate query through the adapter."""
    return adapter.simulate(
        round_id,
        seed_index,
        viewport_x,
        viewport_y,
        viewport_w,
        viewport_h,
    )


def _compute_entropy_mass(tensor: NDArray[np.float64]) -> float:
    """Compute mean per-cell entropy of a prediction tensor.

    Args:
        tensor: (H, W, C) probability tensor.

    Returns:
        Mean Shannon entropy across all cells.
    """
    # Clamp to avoid log(0)
    safe: NDArray[np.float64] = np.clip(tensor, 1e-10, 1.0)
    per_cell: NDArray[np.float64] = np.sum(safe * np.log(safe), axis=-1)
    cell_entropy: NDArray[np.float64] = -per_cell
    return float(cell_entropy.sum()) / max(cell_entropy.size, 1)


def _compute_seed_confidences(
    posterior_ess: float,
    posterior_top_mass: float,
    n_particles: int,
    particle_tensors: list[NDArray[np.float64]],
    calibration_disagreements: list[float],
) -> list[PosteriorConfidence]:
    """Compute per-seed confidence summaries.

    Uses the shared posterior ESS/top-mass (round-level signals) combined
    with per-seed entropy and disagreement.

    Args:
        posterior_ess: ESS of the final posterior.
        posterior_top_mass: Top particle mass in the final posterior.
        n_particles: Number of particles in the posterior.
        particle_tensors: Per-seed posterior-predictive tensors.
        calibration_disagreements: Per-seed calibration disagreement values.

    Returns:
        One PosteriorConfidence per seed.
    """
    confidences: list[PosteriorConfidence] = []
    for seed_idx, tensor in enumerate(particle_tensors):
        disagreement = (
            calibration_disagreements[seed_idx]
            if seed_idx < len(calibration_disagreements)
            else 0.0
        )
        entropy_mass = _compute_entropy_mass(tensor)
        conf = compute_confidence(
            seed_index=seed_idx,
            ess=posterior_ess,
            top_particle_mass=posterior_top_mass,
            disagreement=disagreement,
            entropy_mass=entropy_mass,
            n_particles=n_particles,
        )
        confidences.append(conf)
    return confidences


def solve(
    adapter: SolverAdapter,
    round_id: str,
    n_particles: int = 24,
    n_inner_runs: int = 6,
    sims_per_seed: int = DEFAULT_SIMS_PER_SEED,
    base_seed: int = 0,
) -> SolveResult:
    """Run the full solver pipeline against the given round.

    Steps:
      1. Load round detail for all seeds
      2. Build structural anchor tensors for hybrid fallback
      3. Generate bootstrap candidates (hotspots)
      4. Issue 10 bootstrap queries (2 per seed)
      5. Update posterior after each observation
      6. Prune and resample after bootstrap
      7. Run 6 adaptive batches of 5 globally-selected queries
      8. Check contradiction triggers -> release reserve if needed
      9. Compute per-seed confidence from posterior state
     10. Generate posterior-predictive tensors for all 5 seeds
     11. Apply hybrid anchor hedge based on confidence
     12. Return structured SolveResult with transcript and hybrid metadata

    Args:
        adapter: SolverAdapter (benchmark or prod).
        round_id: Round identifier.
        n_particles: Initial particle count (default 24).
        n_inner_runs: Inner MC runs per likelihood update (default 6).
        sims_per_seed: Final prediction MC runs per seed (default 64).
        base_seed: Base seed for deterministic execution.

    Returns:
        SolveResult with tensors, transcript, hybrid metadata, and metrics.
    """
    t_start = time.monotonic()
    transcript: list[QueryRecord] = []

    # 1. Load round detail
    detail = adapter.get_round_detail(round_id)
    height = detail.map_height
    width = detail.map_width
    initial_states = detail.initial_states
    n_seeds = min(len(initial_states), MAX_SEEDS)

    # 2. Build structural anchors for hybrid fallback
    anchor_tensors = build_structural_anchors(initial_states[:n_seeds], height, width)

    # 3. Initialize posterior and allocation
    posterior = create_posterior(n_particles=n_particles, seed=base_seed)
    alloc = initialize_allocation(initial_states[:n_seeds], height, width)

    # 4. Bootstrap phase: 2 queries per seed
    bootstrap_plan = plan_calibration_bootstrap_queries(alloc, posterior)
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

    calibration_disagreements: list[float] = []
    particle_count = max(len(posterior.particles), 1)
    normalized_ess = min(posterior.ess / particle_count, 1.0)
    for seed_idx in range(n_seeds):
        seed_queries = [
            q for q in transcript if q.phase == "bootstrap" and q.seed_index == seed_idx
        ]
        if seed_queries:
            calibration_disagreements.append(1.0 - normalized_ess)
        else:
            calibration_disagreements.append(0.0)

    # 5. Post-bootstrap: prune and resample
    transition_phase(alloc)
    posterior = prune_and_resample_bootstrap(
        posterior,
        top_k=8,
        target_n=12,
        seed=base_seed + 1000,
    )

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

    # 6. Adaptive phase: 6 batches of 5

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
    contradiction = check_contradiction_triggers(alloc, posterior)

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

    # 8. Generate posterior-predictive tensors
    elapsed = time.monotonic() - t_start
    # Rough ceiling: 2.5h = 9000s. If > 80% of that, fallback.
    runtime_fraction = elapsed / 9000.0

    particle_tensors, prediction_metrics = predict_all_seeds(
        posterior,
        initial_states[:n_seeds],
        map_height=height,
        map_width=width,
        top_k=6,
        sims_per_seed=sims_per_seed,
        base_seed=base_seed + 5000,
        runtime_fraction=runtime_fraction,
    )

    # 9. Compute per-seed confidence and apply hybrid anchor hedge
    confidences = _compute_seed_confidences(
        posterior_ess=posterior.ess,
        posterior_top_mass=posterior.top_particle_mass,
        n_particles=len(posterior.particles),
        particle_tensors=particle_tensors,
        calibration_disagreements=calibration_disagreements,
    )

    # Apply hybrid blending: confidence-gated anchor hedge
    hybrid_tensors = apply_anchor_hedge(
        particle_tensors=particle_tensors,
        anchor_tensors=anchor_tensors,
        confidences=confidences,
        initial_states=initial_states[:n_seeds],
        height=height,
        width=width,
    )

    hybrid_meta = HybridMetadata(
        confidences=confidences,
        hedge_modes=[c.recommended_mode for c in confidences],
        blend_weights=[compute_blend_weight(c) for c in confidences],
    )

    return SolveResult(
        tensors=hybrid_tensors,
        transcript=transcript,
        total_queries_used=alloc.queries_used,
        runtime_seconds=time.monotonic() - t_start,
        final_ess=posterior.ess,
        contradiction_triggered=contradiction,
        hybrid=hybrid_meta,
    )
