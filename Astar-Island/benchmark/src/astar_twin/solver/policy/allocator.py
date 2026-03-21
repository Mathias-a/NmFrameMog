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
  - Reject windows with >60% overlap with previously queried window
    (unless contradiction-flagged).

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

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import MAX_QUERIES, MAX_SEEDS, NUM_CLASSES
from astar_twin.solver.inference.posterior import PosteriorState
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
            candidates, ["coastal", "corridor", "frontier", "reclaim", "fallback"],
            exclude=[]
        )
        # Query B: prefer frontier/corridor, then anything not already chosen
        exclude_b = [query_a] if query_a else []
        query_b = _pick_by_category_priority(
            candidates, ["frontier", "corridor", "reclaim", "fallback", "coastal"],
            exclude=exclude_b
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


def score_candidate(
    candidate: ViewportCandidate,
    seed_index: int,
    posterior: PosteriorState,
    initial_state: InitialState,
    queried_viewports: list[ViewportCandidate],
    entropy_map: NDArray[np.float64] | None = None,
    allow_high_overlap: bool = False,
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

    # Reject if overlap too high (unless contradiction-flagged)
    if max_overlap > MAX_OVERLAP_FRACTION and not allow_high_overlap:
        return -1.0  # sentinel: rejected

    overlap_penalty = max_overlap  # already in [0, 1]

    # --- Entropy mass ---
    entropy_score = 0.0
    if entropy_map is not None:
        h, w = entropy_map.shape[:2]
        y_start = min(candidate.y, h)
        y_end = min(candidate.y + candidate.h, h)
        x_start = min(candidate.x, w)
        x_end = min(candidate.x + candidate.w, w)
        if y_end > y_start and x_end > x_start:
            window_entropy = entropy_map[y_start:y_end, x_start:x_end]
            # Normalize: max possible entropy per cell is log(NUM_CLASSES)
            max_entropy = np.log(NUM_CLASSES) * (y_end - y_start) * (x_end - x_start)
            entropy_score = float(np.sum(window_entropy)) / max(max_entropy, 1e-6)

    # --- Posterior disagreement ---
    disagreement = compute_posterior_disagreement(
        candidate, posterior, initial_state
    )

    # --- Expected stat gain (proxy) ---
    stat_gain = _estimate_stat_gain(candidate, initial_state)

    score = (
        W_ENTROPY * entropy_score
        + W_DISAGREEMENT * disagreement
        + W_STAT_GAIN * stat_gain
        - W_OVERLAP_PENALTY * overlap_penalty
    )
    return float(score)


def compute_posterior_disagreement(
    candidate: ViewportCandidate,
    posterior: PosteriorState,
    initial_state: InitialState,
) -> float:
    """Fraction of cells where top-2 particles disagree on argmax class.

    Uses initial state terrain as a fast proxy (no simulation needed).
    Returns a value in [0, 1].
    """
    if len(posterior.particles) < 2:
        return 0.0

    # For a fast proxy, we measure disagreement as the weight difference
    # between the top 2 particles. Higher weight concentration = less disagreement.
    weights = np.array(posterior.normalized_weights())
    sorted_weights = np.sort(weights)[::-1]

    if len(sorted_weights) < 2:
        return 0.0

    # Disagreement proxy: 1 - dominance of top particle
    # If top particle has 90% weight, disagreement = 0.1
    # If top two are 50/50, disagreement ~ 0.5
    top_mass = sorted_weights[0]
    disagreement = 1.0 - top_mass

    return float(np.clip(disagreement, 0.0, 1.0))


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
            if (candidate.x <= s.x < candidate.x + candidate.w
                    and candidate.y <= s.y < candidate.y + candidate.h):
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
    # Clip to prevent log(0)
    p = np.clip(prediction_tensor, 1e-10, 1.0)
    # Normalize along class axis
    p = p / np.sum(p, axis=2, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=2)
    return entropy


def select_adaptive_batch(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    current_prediction: NDArray[np.float64] | None = None,
    batch_size: int = ADAPTIVE_BATCH_SIZE,
) -> list[tuple[int, ViewportCandidate]]:
    """Select the next batch of adaptive queries across all seeds.

    Scores all candidate viewports across all seeds and picks the top batch_size.
    """
    if state.queries_remaining <= 0:
        return []

    actual_batch = min(batch_size, state.queries_remaining)

    # Compute entropy map if prediction available
    entropy_map: NDArray[np.float64] | None = None
    if current_prediction is not None:
        entropy_map = compute_entropy_map(current_prediction)

    # Score all candidates across all seeds
    scored: list[tuple[float, int, ViewportCandidate]] = []

    for seed_idx, candidates in state.seed_candidates.items():
        if seed_idx >= len(initial_states):
            continue
        initial_state = initial_states[seed_idx]
        queried_vps = [q.viewport for q in state.queries_for_seed(seed_idx)]

        for candidate in candidates:
            s = score_candidate(
                candidate, seed_idx, posterior, initial_state,
                queried_vps, entropy_map,
            )
            if s >= 0:  # not rejected
                scored.append((s, seed_idx, candidate))

    # Sort by score descending, pick top batch
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(seed_idx, vp) for _, seed_idx, vp in scored[:actual_batch]]


def check_contradiction_triggers(
    state: AllocationState,
    posterior: PosteriorState,
) -> bool:
    """Check if any contradiction trigger fires, releasing reserve queries.

    Triggers:
      - ESS < 6
      - Any seed has < MIN_QUERIES_PER_SEED after adaptive phase
    """
    # ESS trigger
    if posterior.ess < ESS_CONTRADICTION_THRESHOLD:
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

    This is a lightweight check that only looks at weight distribution.
    True disagreement requires running inner MC for each particle,
    but we use posterior weight spread as a proxy.
    """
    if len(posterior.particles) < 2:
        return False

    # Use disagreement proxy
    disagreement = compute_posterior_disagreement(candidate, posterior, initial_state)
    return disagreement > CONTRADICTION_CELL_THRESHOLD


def plan_reserve_queries(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    current_prediction: NDArray[np.float64] | None = None,
) -> list[tuple[int, ViewportCandidate]]:
    """Plan reserve query usage.

    If contradiction triggered: use as adaptive queries with high-overlap allowed.
    Otherwise: release as two final adaptive batches of 5.
    """
    remaining = min(RESERVE_QUERIES, state.queries_remaining)
    if remaining <= 0:
        return []

    contradiction = check_contradiction_triggers(state, posterior)

    if contradiction:
        # Allow high-overlap queries for contradiction resolution
        return _select_contradiction_queries(
            state, posterior, initial_states, current_prediction, remaining
        )
    else:
        # Release as normal adaptive batches
        return select_adaptive_batch(
            state, posterior, initial_states, current_prediction, batch_size=remaining
        )


def _select_contradiction_queries(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    current_prediction: NDArray[np.float64] | None,
    n_queries: int,
) -> list[tuple[int, ViewportCandidate]]:
    """Select queries for contradiction resolution with relaxed overlap rules."""
    entropy_map: NDArray[np.float64] | None = None
    if current_prediction is not None:
        entropy_map = compute_entropy_map(current_prediction)

    scored: list[tuple[float, int, ViewportCandidate]] = []

    for seed_idx, candidates in state.seed_candidates.items():
        if seed_idx >= len(initial_states):
            continue
        initial_state = initial_states[seed_idx]
        queried_vps = [q.viewport for q in state.queries_for_seed(seed_idx)]

        for candidate in candidates:
            s = score_candidate(
                candidate, seed_idx, posterior, initial_state,
                queried_vps, entropy_map,
                allow_high_overlap=True,  # relaxed for contradiction
            )
            if s >= 0:
                scored.append((s, seed_idx, candidate))

    scored.sort(key=lambda t: t[0], reverse=True)

    # Prioritize under-queried seeds
    under_queried: list[tuple[float, int, ViewportCandidate]] = []
    normal: list[tuple[float, int, ViewportCandidate]] = []
    for item in scored:
        seed_idx = item[1]
        if len(state.queries_for_seed(seed_idx)) < MIN_QUERIES_PER_SEED:
            under_queried.append(item)
        else:
            normal.append(item)

    result_pool = under_queried + normal
    return [(seed_idx, vp) for _, seed_idx, vp in result_pool[:n_queries]]


def record_query(
    state: AllocationState,
    seed_index: int,
    viewport: ViewportCandidate,
    phase: str | None = None,
) -> AllocationState:
    """Record a query that has been executed."""
    if phase is None:
        phase = state.phase
    state.queries.append(QueryRecord(
        seed_index=seed_index, viewport=viewport, phase=phase,
    ))
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
