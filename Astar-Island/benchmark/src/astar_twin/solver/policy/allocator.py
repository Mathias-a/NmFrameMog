"""Adaptive viewport allocation policy.

Budget schedule: 10 bootstrap + 30 adaptive + 10 reserve = 50 total.

Bootstrap phase (10 queries, 2 per seed):
  - Query A: top coastal settlement cluster (15x15)
  - Query B: top frontier/corridor growth-conflict cluster (15x15)
  - Falls back to next-best candidate if coastal/corridor unavailable.

Adaptive phase (30 queries, 6 batches of 5 globally selected):
  - Score each candidate window:
      0.45 * entropy_mass
      0.35 * posterior_disagreement
      0.20 * expected_stat_gain
      - 0.25 * overlap_penalty
  - Apply a steep soft penalty to windows with >60% overlap with previously
    queried windows, with lighter penalties during contradiction handling.

Reserve phase (10 queries):
  - Held until contradiction trigger fires.
  - Otherwise released as two final adaptive batches.

Contradiction triggers:
  - Top 2 particles disagree on argmax in >20% of cells in a candidate window
  - ESS < 6
  - Any seed has < 8 total queries after adaptive phase
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import cast

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import MAX_QUERIES, MAX_SEEDS, NUM_CLASSES
from astar_twin.solver.inference.posterior import PosteriorState
from astar_twin.solver.observe.ledger import ObservationLedger
from astar_twin.solver.policy.hotspots import ViewportCandidate, generate_hotspots

# Budget constants
BOOTSTRAP_QUERIES = 10
ADAPTIVE_QUERIES = 30
RESERVE_QUERIES = 10
BOOTSTRAP_PER_SEED = 2
ADAPTIVE_BATCH_SIZE = 5
ADAPTIVE_BATCHES = ADAPTIVE_QUERIES // ADAPTIVE_BATCH_SIZE  # 6

# Scoring weights
W_ENTROPY = 0.45
W_DISAGREEMENT = 0.35
W_STAT_GAIN = 0.20
W_OVERLAP_PENALTY = 0.25

# Thresholds
MAX_OVERLAP_FRACTION = 0.60
CONTRADICTION_CELL_THRESHOLD = 0.20
ESS_CONTRADICTION_THRESHOLD = 6.0
MIN_QUERIES_PER_SEED = 8

# Soft-selection tuning
HIGH_OVERLAP_SOFT_PENALTY = 2.5
RELAXED_HIGH_OVERLAP_SOFT_PENALTY = 1.0
SEED_BALANCE_BONUS_PER_QUERY = 0.06
MAX_SEED_BALANCE_BONUS = 0.30


@dataclass
class QueryRecord:
    """Record of a single query that has been issued."""

    seed_index: int
    viewport: ViewportCandidate
    phase: str  # "bootstrap" | "adaptive" | "reserve"


@dataclass
class AllocationState:
    """Tracks query allocation state across all phases."""

    budget_total: int = MAX_QUERIES
    queries: list[QueryRecord] = field(default_factory=list)
    phase: str = "bootstrap"  # "bootstrap" | "adaptive" | "reserve" | "done"

    # Per-seed candidate pools (generated from hotspots)
    seed_candidates: dict[int, list[ViewportCandidate]] = field(default_factory=dict)

    @property
    def queries_used(self) -> int:
        return len(self.queries)

    @property
    def queries_remaining(self) -> int:
        return self.budget_total - self.queries_used

    def queries_for_seed(self, seed_index: int) -> list[QueryRecord]:
        return [q for q in self.queries if q.seed_index == seed_index]

    def queries_in_phase(self, phase: str) -> list[QueryRecord]:
        return [q for q in self.queries if q.phase == phase]

    @property
    def bootstrap_done(self) -> bool:
        return len(self.queries_in_phase("bootstrap")) >= BOOTSTRAP_QUERIES

    @property
    def adaptive_done(self) -> bool:
        return len(self.queries_in_phase("adaptive")) >= ADAPTIVE_QUERIES


def initialize_allocation(
    initial_states: list[InitialState],
    map_height: int,
    map_width: int,
) -> AllocationState:
    """Create initial allocation state with candidate pools per seed."""
    state = AllocationState()
    for seed_idx in range(min(len(initial_states), MAX_SEEDS)):
        candidates = generate_hotspots(initial_states[seed_idx], map_height, map_width)
        state.seed_candidates[seed_idx] = candidates
    return state


def plan_bootstrap_queries(
    state: AllocationState,
) -> list[tuple[int, ViewportCandidate]]:
    """Plan the 10 bootstrap queries: 2 per seed (coastal + frontier/corridor).

    Returns list of (seed_index, viewport_candidate) tuples.
    """
    planned: list[tuple[int, ViewportCandidate]] = []
    n_seeds = len(state.seed_candidates)

    for seed_idx in range(n_seeds):
        candidates = state.seed_candidates.get(seed_idx, [])
        if not candidates:
            continue

        # Query A: prefer coastal, then corridor, then anything
        query_a = _pick_by_category_priority(
            candidates, ["coastal", "corridor", "frontier", "reclaim", "fallback"], exclude=[]
        )
        # Query B: prefer frontier/corridor, then anything not already chosen
        exclude_b = [query_a] if query_a else []
        query_b = _pick_by_category_priority(
            candidates,
            ["frontier", "corridor", "reclaim", "fallback", "coastal"],
            exclude=exclude_b,
        )

        if query_a:
            planned.append((seed_idx, query_a))
        if query_b:
            planned.append((seed_idx, query_b))

    return planned[:BOOTSTRAP_QUERIES]


def _pick_by_category_priority(
    candidates: list[ViewportCandidate],
    priority: list[str],
    exclude: list[ViewportCandidate],
) -> ViewportCandidate | None:
    """Pick the first candidate matching priority order, excluding already picked."""
    exclude_keys = {(c.x, c.y, c.w, c.h) for c in exclude}
    for cat in priority:
        for c in candidates:
            key = (c.x, c.y, c.w, c.h)
            if c.category == cat and key not in exclude_keys:
                return c
    # Fallback: any not-excluded candidate
    for c in candidates:
        if (c.x, c.y, c.w, c.h) not in exclude_keys:
            return c
    return None


def _clone_candidate(candidate: ViewportCandidate, score: float) -> ViewportCandidate:
    """Return a scored copy of a candidate viewport."""
    return ViewportCandidate(
        x=candidate.x,
        y=candidate.y,
        w=candidate.w,
        h=candidate.h,
        category=candidate.category,
        score=score,
    )


def _compute_overlap_penalty(max_overlap: float, allow_high_overlap: bool) -> float:
    """Convert raw overlap into a soft penalty with steep post-threshold growth."""
    if max_overlap <= MAX_OVERLAP_FRACTION:
        return max_overlap

    normalized_excess = (max_overlap - MAX_OVERLAP_FRACTION) / max(
        1.0 - MAX_OVERLAP_FRACTION,
        1e-6,
    )
    extra_penalty = (
        RELAXED_HIGH_OVERLAP_SOFT_PENALTY if allow_high_overlap else HIGH_OVERLAP_SOFT_PENALTY
    )
    return max_overlap + extra_penalty * normalized_excess * normalized_excess


def _compute_entropy_maps(
    seed_predictions: dict[int, NDArray[np.float64]] | None,
) -> dict[int, NDArray[np.float64]]:
    """Materialize per-seed entropy maps once for a selection pass."""
    entropy_maps: dict[int, NDArray[np.float64]] = {}
    if seed_predictions is None:
        return entropy_maps

    for seed_idx, prediction in seed_predictions.items():
        entropy_maps[seed_idx] = compute_entropy_map(prediction)
    return entropy_maps


def _collect_scored_candidates(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    seed_predictions: dict[int, NDArray[np.float64]] | None = None,
    allow_high_overlap: bool = False,
    observation_ledger: ObservationLedger | None = None,
) -> list[tuple[float, int, ViewportCandidate]]:
    """Score every currently available candidate across seeds."""
    scored: list[tuple[float, int, ViewportCandidate]] = []
    entropy_maps = _compute_entropy_maps(seed_predictions)

    for seed_idx, candidates in state.seed_candidates.items():
        if seed_idx >= len(initial_states):
            continue

        initial_state = initial_states[seed_idx]
        queried_vps = [q.viewport for q in state.queries_for_seed(seed_idx)]
        entropy_map = entropy_maps.get(seed_idx)

        for candidate in candidates:
            score = score_candidate(
                candidate,
                seed_idx,
                posterior,
                initial_state,
                queried_vps,
                entropy_map,
                allow_high_overlap=allow_high_overlap,
                observation_ledger=observation_ledger,
            )
            if np.isfinite(score):
                scored.append((score, seed_idx, candidate))

    return scored


def _pair_overlap_fraction(lhs: ViewportCandidate, rhs: ViewportCandidate) -> float:
    """Symmetric overlap proxy for redundancy checks between two viewports."""
    return max(lhs.overlap_fraction(rhs), rhs.overlap_fraction(lhs))


def _is_redundant_with_selected(
    candidate: ViewportCandidate,
    selected: list[tuple[int, ViewportCandidate]],
) -> bool:
    """Whether a candidate overlaps too strongly with already-selected batch items."""
    return any(
        _pair_overlap_fraction(candidate, selected_candidate) > MAX_OVERLAP_FRACTION
        for _, selected_candidate in selected
    )


def _seed_balance_bonus(
    seed_index: int,
    state: AllocationState,
    selected: list[tuple[int, ViewportCandidate]],
) -> float:
    """Soft bonus for seeds that are behind current allocation levels."""
    if not state.seed_candidates:
        return 0.0

    projected_counts = {
        candidate_seed: len(state.queries_for_seed(candidate_seed))
        for candidate_seed in state.seed_candidates
    }
    for selected_seed, _ in selected:
        projected_counts[selected_seed] = projected_counts.get(selected_seed, 0) + 1

    seed_count = projected_counts.get(seed_index, 0)
    max_count = max(projected_counts.values(), default=seed_count)
    deficit = max_count - seed_count
    if deficit <= 0:
        return 0.0

    return min(MAX_SEED_BALANCE_BONUS, deficit * SEED_BALANCE_BONUS_PER_QUERY)


def _select_scored_batch(
    scored: list[tuple[float, int, ViewportCandidate]],
    state: AllocationState,
    batch_size: int,
    *,
    apply_seed_balance: bool = True,
    avoid_batch_redundancy: bool = True,
) -> list[tuple[int, ViewportCandidate]]:
    """Greedily fill a batch while backfilling around redundant top choices."""
    remaining = list(scored)
    selected: list[tuple[int, ViewportCandidate]] = []

    while remaining and len(selected) < batch_size:
        best_idx: int | None = None
        best_adjusted = float("-inf")
        fallback_idx: int | None = None
        fallback_adjusted = float("-inf")

        for idx, (base_score, seed_idx, candidate) in enumerate(remaining):
            adjusted_score = base_score
            if apply_seed_balance:
                adjusted_score += _seed_balance_bonus(seed_idx, state, selected)

            if adjusted_score > fallback_adjusted:
                fallback_idx = idx
                fallback_adjusted = adjusted_score

            if avoid_batch_redundancy and _is_redundant_with_selected(candidate, selected):
                continue

            if adjusted_score > best_adjusted:
                best_idx = idx
                best_adjusted = adjusted_score

        pick_idx = best_idx if best_idx is not None else fallback_idx
        if pick_idx is None:
            break

        base_score, seed_idx, candidate = remaining.pop(pick_idx)
        final_score = base_score
        if apply_seed_balance:
            final_score += _seed_balance_bonus(seed_idx, state, selected)
        selected.append((seed_idx, _clone_candidate(candidate, final_score)))

    return selected


def _select_top_contradiction_candidate(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    seed_predictions: dict[int, NDArray[np.float64]] | None = None,
    observation_ledger: ObservationLedger | None = None,
) -> tuple[int, ViewportCandidate] | None:
    """Pick the strongest contradiction-resolution candidate, if any."""
    scored = _collect_scored_candidates(
        state,
        posterior,
        initial_states,
        seed_predictions=seed_predictions,
        allow_high_overlap=True,
        observation_ledger=observation_ledger,
    )
    batch = _select_scored_batch(
        scored,
        state,
        batch_size=1,
        apply_seed_balance=True,
        avoid_batch_redundancy=False,
    )
    return batch[0] if batch else None


def score_candidate(
    candidate: ViewportCandidate,
    seed_index: int,
    posterior: PosteriorState,
    initial_state: InitialState,
    queried_viewports: list[ViewportCandidate],
    entropy_map: NDArray[np.float64] | None = None,
    allow_high_overlap: bool = False,
    observation_ledger: ObservationLedger | None = None,
) -> float:
    """Score a candidate viewport for adaptive selection.

    Components:
      - entropy_mass: sum of per-cell entropy in candidate window
      - posterior_disagreement: fraction of cells where top-2 particles disagree on argmax
      - expected_stat_gain: proxy based on alive settlement density in initial state
      - overlap_penalty: max overlap fraction with any previously queried viewport

    All components are normalized to [0, 1] before weighting.
    """
    # --- Overlap penalty ---
    max_overlap = 0.0
    for queried in queried_viewports:
        overlap = candidate.overlap_fraction(queried)
        max_overlap = max(max_overlap, overlap)

    overlap_penalty = _compute_overlap_penalty(max_overlap, allow_high_overlap)

    # --- Entropy mass ---
    entropy_score = 0.0
    if entropy_map is not None:
        h = int(entropy_map.shape[0])
        w = int(entropy_map.shape[1])
        y_start = min(candidate.y, h)
        y_end = min(candidate.y + candidate.h, h)
        x_start = min(candidate.x, w)
        x_end = min(candidate.x + candidate.w, w)
        if y_end > y_start and x_end > x_start:
            window_entropy = cast(NDArray[np.float64], entropy_map[y_start:y_end, x_start:x_end])
            # Normalize: max possible entropy per cell is log(NUM_CLASSES)
            max_entropy = log(NUM_CLASSES) * float(y_end - y_start) * float(x_end - x_start)
            entropy_score = float(np.sum(window_entropy)) / max(max_entropy, 1e-6)

    # --- Posterior disagreement ---
    disagreement = compute_posterior_disagreement(candidate, posterior, initial_state)

    # --- Expected stat gain (proxy) ---
    stat_gain = _estimate_stat_gain(candidate, initial_state)

    score = (
        W_ENTROPY * entropy_score
        + W_DISAGREEMENT * disagreement
        + W_STAT_GAIN * stat_gain
        - W_OVERLAP_PENALTY * overlap_penalty
    )

    # Coverage penalty: reduce score for already-well-observed areas
    if observation_ledger is not None:
        mean_visits = observation_ledger.mean_visit_count_in_window(
            seed_index, candidate.x, candidate.y, candidate.w, candidate.h
        )
        coverage_penalty = min(mean_visits / 3.0, 0.3)
        score -= coverage_penalty

    return float(score)


def compute_posterior_disagreement(
    candidate: ViewportCandidate,
    posterior: PosteriorState,
    initial_state: InitialState,
) -> float:
    """Fraction of viewport cells where top-2 particles disagree on argmax class.

    Runs a lightweight inner MC (2 simulations per particle), aggregates each of the
    top-2 particles into a terrain probability tensor, and compares the per-cell
    argmax terrain class inside the candidate window. Returns a value in [0, 1].
    """
    if len(posterior.particles) < 2:
        return 0.0

    from astar_twin.engine import Simulator
    from astar_twin.mc.aggregate import aggregate_runs
    from astar_twin.mc.runner import MCRunner

    top_indices = posterior.top_k_indices(2)
    if len(top_indices) < 2:
        return 0.0

    map_height = len(initial_state.grid)
    map_width = len(initial_state.grid[0])
    viewport_argmaxes: list[NDArray[np.int64]] = []

    seed_offsets = [99000, 99100]
    for idx, particle_idx in enumerate(top_indices):
        base_seed = seed_offsets[idx]
        particle = posterior.particles[particle_idx]
        simulator = Simulator(params=particle.to_simulation_params())
        runner = MCRunner(simulator)
        runs = runner.run_batch(initial_state, n_runs=2, base_seed=base_seed)
        tensor = aggregate_runs(runs, map_height, map_width)
        window = cast(
            NDArray[np.float64],
            tensor[
                candidate.y : candidate.y + candidate.h,
                candidate.x : candidate.x + candidate.w,
            ],
        )
        viewport_argmaxes.append(cast(NDArray[np.int64], np.argmax(window, axis=2)))

    argmax_1, argmax_2 = viewport_argmaxes
    total_cells = int(argmax_1.size)
    if total_cells == 0:
        return 0.0

    disagreement_count = 0
    height = int(argmax_1.shape[0])
    width = int(argmax_1.shape[1])
    for y in range(height):
        for x in range(width):
            lhs = int(cast(np.int64, argmax_1[y, x]))
            rhs = int(cast(np.int64, argmax_2[y, x]))
            if lhs != rhs:
                disagreement_count += 1
    disagreement = disagreement_count / total_cells
    return float(cast(np.float64, np.clip(disagreement, 0.0, 1.0)))


def _estimate_stat_gain(
    candidate: ViewportCandidate,
    initial_state: InitialState,
) -> float:
    """Proxy for expected settlement-stat info gain.

    Higher if more alive settlements are in the viewport.
    """
    alive_in_window = 0
    total_settlements = 0
    for s in initial_state.settlements:
        if s.alive:
            total_settlements += 1
            if (
                candidate.x <= s.x < candidate.x + candidate.w
                and candidate.y <= s.y < candidate.y + candidate.h
            ):
                alive_in_window += 1

    if total_settlements == 0:
        return 0.0
    return float(alive_in_window) / max(total_settlements, 1)


def compute_entropy_map(
    prediction_tensor: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute per-cell entropy from a H×W×6 prediction tensor.

    Returns H×W array of Shannon entropy values.
    """
    height = int(prediction_tensor.shape[0])
    width = int(prediction_tensor.shape[1])
    entropy = cast(NDArray[np.float64], np.zeros((height, width), dtype=np.float64))
    for y in range(height):
        for x in range(width):
            probs: list[float] = []
            for cls_idx in range(NUM_CLASSES):
                probs.append(max(float(cast(np.float64, prediction_tensor[y, x, cls_idx])), 1e-10))
            total = sum(probs)
            normalized = [prob / total for prob in probs]
            entropy[y, x] = -sum(prob * log(prob) for prob in normalized)
    return entropy


def select_adaptive_batch(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    seed_predictions: dict[int, NDArray[np.float64]] | None = None,
    batch_size: int = ADAPTIVE_BATCH_SIZE,
    observation_ledger: ObservationLedger | None = None,
) -> list[tuple[int, ViewportCandidate]]:
    """Select the next batch of adaptive queries across all seeds.

    Greedily fills the batch with soft seed balancing and non-redundant backfill.
    """
    if state.queries_remaining <= 0:
        return []

    actual_batch = min(batch_size, state.queries_remaining)

    scored = _collect_scored_candidates(
        state,
        posterior,
        initial_states,
        seed_predictions=seed_predictions,
        allow_high_overlap=False,
        observation_ledger=observation_ledger,
    )
    return _select_scored_batch(
        scored,
        state,
        actual_batch,
        apply_seed_balance=True,
        avoid_batch_redundancy=True,
    )


def check_contradiction_triggers(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState] | None = None,
    seed_predictions: dict[int, NDArray[np.float64]] | None = None,
    observation_ledger: ObservationLedger | None = None,
) -> bool:
    """Check if any contradiction trigger fires, releasing reserve queries.

    Triggers:
      - ESS < 6
      - Best contradiction-resolution candidate has >20% argmax disagreement
      - Any seed has < MIN_QUERIES_PER_SEED after adaptive phase
    """
    # ESS trigger
    if posterior.ess < ESS_CONTRADICTION_THRESHOLD:
        return True

    # Disagreement trigger
    if initial_states is not None:
        candidate_plan = _select_top_contradiction_candidate(
            state,
            posterior,
            initial_states,
            seed_predictions=seed_predictions,
            observation_ledger=observation_ledger,
        )
        if candidate_plan is not None:
            seed_idx, candidate = candidate_plan
            if seed_idx < len(initial_states) and check_argmax_disagreement(
                candidate,
                posterior,
                initial_states[seed_idx],
            ):
                return True

    # Under-queried seed trigger (only after adaptive done)
    if state.adaptive_done:
        for seed_idx in state.seed_candidates:
            n_queries = len(state.queries_for_seed(seed_idx))
            if n_queries < MIN_QUERIES_PER_SEED:
                return True

    return False


def check_argmax_disagreement(
    candidate: ViewportCandidate,
    posterior: PosteriorState,
    initial_state: InitialState,
) -> bool:
    """Check if top-2 particles disagree on argmax in >20% of cells.

    Uses the same lightweight cellwise top-2 argmax disagreement computation as
    adaptive scoring.
    """
    if len(posterior.particles) < 2:
        return False

    disagreement = compute_posterior_disagreement(candidate, posterior, initial_state)
    return disagreement > CONTRADICTION_CELL_THRESHOLD


def plan_reserve_queries(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    seed_predictions: dict[int, NDArray[np.float64]] | None = None,
    n_queries: int | None = None,
    observation_ledger: ObservationLedger | None = None,
    contradiction_triggered: bool | None = None,
) -> list[tuple[int, ViewportCandidate]]:
    """Plan reserve query usage.

    If contradiction triggered: use as adaptive queries with high-overlap allowed.
    Otherwise: release as two final adaptive batches of 5.
    """
    remaining = min(
        n_queries if n_queries is not None else RESERVE_QUERIES,
        state.queries_remaining,
    )
    if remaining <= 0:
        return []

    contradiction = contradiction_triggered
    if contradiction is None:
        contradiction = check_contradiction_triggers(
            state,
            posterior,
            initial_states,
            seed_predictions=seed_predictions,
            observation_ledger=observation_ledger,
        )

    if contradiction:
        # Allow high-overlap queries for contradiction resolution
        return _select_contradiction_queries(
            state,
            posterior,
            initial_states,
            seed_predictions,
            remaining,
            observation_ledger=observation_ledger,
        )
    else:
        # Release as normal adaptive batches
        return select_adaptive_batch(
            state,
            posterior,
            initial_states,
            seed_predictions,
            batch_size=remaining,
            observation_ledger=observation_ledger,
        )


def _select_contradiction_queries(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    seed_predictions: dict[int, NDArray[np.float64]] | None,
    n_queries: int,
    observation_ledger: ObservationLedger | None = None,
) -> list[tuple[int, ViewportCandidate]]:
    """Select queries for contradiction resolution with relaxed overlap rules."""
    scored = _collect_scored_candidates(
        state,
        posterior,
        initial_states,
        seed_predictions=seed_predictions,
        allow_high_overlap=True,
        observation_ledger=observation_ledger,
    )
    return _select_scored_batch(
        scored,
        state,
        n_queries,
        apply_seed_balance=True,
        avoid_batch_redundancy=True,
    )


def record_query(
    state: AllocationState,
    seed_index: int,
    viewport: ViewportCandidate,
    phase: str | None = None,
) -> AllocationState:
    """Record a query that has been executed."""
    if phase is None:
        phase = state.phase
    state.queries.append(
        QueryRecord(
            seed_index=seed_index,
            viewport=viewport,
            phase=phase,
        )
    )
    return state


def transition_phase(state: AllocationState) -> AllocationState:
    """Auto-transition phase based on query counts."""
    if state.phase == "bootstrap" and state.bootstrap_done:
        state.phase = "adaptive"
    elif state.phase == "adaptive" and state.adaptive_done:
        state.phase = "reserve"
    elif state.phase == "reserve" and state.queries_remaining <= 0:
        state.phase = "done"
    return state
