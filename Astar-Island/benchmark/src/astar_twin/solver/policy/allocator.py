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
from astar_twin.solver.inference.particles import Particle
from astar_twin.solver.inference.posterior import PosteriorState
from astar_twin.solver.policy.hotspots import ViewportCandidate, generate_hotspots
from astar_twin.solver.policy.legality import (
    apply_legality_filter,
    apply_legality_filter_to_viewport,
)

# Budget constants
BOOTSTRAP_QUERIES = 10
ADAPTIVE_QUERIES = 30
RESERVE_QUERIES = 10
BOOTSTRAP_PER_SEED = 2
ADAPTIVE_BATCH_SIZE = 5
ADAPTIVE_BATCHES = ADAPTIVE_QUERIES // ADAPTIVE_BATCH_SIZE  # 6

# Scoring weights
W_OVERLAP_PENALTY = 0.25

# Thresholds
MAX_OVERLAP_FRACTION = 0.60
CONTRADICTION_CELL_THRESHOLD = 0.20
ESS_CONTRADICTION_THRESHOLD = 6.0
MIN_QUERIES_PER_SEED = 8
HEURISTIC_SHORTLIST_FACTOR = 3
APPROX_EIG_TOP_PARTICLES = 3
APPROX_EIG_PSEUDO_OBSERVATIONS = 2
RERANK_EIG_WEIGHT = 0.35
RERANK_HEURISTIC_WEIGHT = 1.0 - RERANK_EIG_WEIGHT


@dataclass(frozen=True)
class CandidateScore:
    heuristic_score: float
    final_score: float
    seed_index: int
    candidate: ViewportCandidate


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


def get_adaptive_weights(queries_used: int) -> tuple[float, float, float]:
    """Compute soft expert weights for adaptive scoring based on query progress.

    Early adaptive phase (first ~8-12 queries): favors discriminative features (stat gain, entropy).
    Late adaptive phase: shifts to spatial disagreement.

    Returns:
        (w_entropy, w_disagreement, w_stat_gain)
    """
    # Transition happens around queries_used = 18 to 22
    # (which is 8 to 12 queries into the adaptive phase, assuming 10 bootstrap queries)
    transition_start = 18
    transition_end = 22

    # Early weights (discriminative)
    early_entropy = 0.50
    early_stat_gain = 0.40
    early_disagreement = 0.10

    # Late weights (disagreement)
    late_entropy = 0.20
    late_stat_gain = 0.10
    late_disagreement = 0.70

    if queries_used <= transition_start:
        return early_entropy, early_disagreement, early_stat_gain
    elif queries_used >= transition_end:
        return late_entropy, late_disagreement, late_stat_gain
    else:
        progress = (queries_used - transition_start) / (transition_end - transition_start)
        w_entropy = early_entropy + progress * (late_entropy - early_entropy)
        w_stat_gain = early_stat_gain + progress * (late_stat_gain - early_stat_gain)
        w_disagreement = early_disagreement + progress * (late_disagreement - early_disagreement)
        return w_entropy, w_disagreement, w_stat_gain


def score_candidate(
    candidate: ViewportCandidate,
    seed_index: int,
    posterior: PosteriorState,
    initial_state: InitialState,
    queried_viewports: list[ViewportCandidate],
    queries_used: int,
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
    disagreement = compute_posterior_disagreement(candidate, posterior, initial_state)

    # --- Expected stat gain (proxy) ---
    stat_gain = _estimate_stat_gain(candidate, initial_state)

    w_entropy, w_disagreement, w_stat_gain = get_adaptive_weights(queries_used)

    score = (
        w_entropy * entropy_score
        + w_disagreement * disagreement
        + w_stat_gain * stat_gain
        - W_OVERLAP_PENALTY * overlap_penalty
    )
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

    top_indices = posterior.top_k_indices(2)
    if len(top_indices) < 2:
        return 0.0

    viewport_argmaxes: list[NDArray[np.int64]] = []

    seed_offsets = [99000, 99100]
    for idx, particle_idx in enumerate(top_indices):
        particle = posterior.particles[particle_idx]
        window = _simulate_particle_viewport_probs(
            particle,
            candidate,
            initial_state,
            base_seed=seed_offsets[idx],
            n_runs=2,
        )
        viewport_argmaxes.append(np.argmax(window, axis=2))

    argmax_1, argmax_2 = viewport_argmaxes
    total_cells = int(argmax_1.size)
    if total_cells == 0:
        return 0.0

    disagreement = np.sum(argmax_1 != argmax_2) / total_cells
    return float(np.clip(disagreement, 0.0, 1.0))


def approximate_expected_information_gain(
    candidate: ViewportCandidate,
    posterior: PosteriorState,
    initial_state: InitialState,
    seed_index: int,
) -> float:
    top_k = min(APPROX_EIG_TOP_PARTICLES, len(posterior.particles))
    top_indices = posterior.top_k_indices(top_k)
    if len(top_indices) < 2:
        return 0.0

    log_weights = np.array(
        [posterior.particles[idx].log_weight for idx in top_indices], dtype=np.float64
    )
    max_log_weight = float(np.max(log_weights))
    weights = np.exp(log_weights - max_log_weight)
    weights /= np.sum(weights)

    viewport_probs: list[NDArray[np.float64]] = []
    seed_root = 120_000 + seed_index * 10_000 + candidate.x * 101 + candidate.y * 307
    seed_root += candidate.w * 401 + candidate.h * 503

    for rank, particle_idx in enumerate(top_indices):
        viewport_probs.append(
            _simulate_particle_viewport_probs(
                posterior.particles[particle_idx],
                candidate,
                initial_state,
                base_seed=seed_root + rank * 1_000,
                n_runs=APPROX_EIG_PSEUDO_OBSERVATIONS,
            )
        )

    mixture = np.zeros_like(viewport_probs[0])
    expected_entropy = np.zeros(viewport_probs[0].shape[:2], dtype=np.float64)
    for weight, probs in zip(weights, viewport_probs, strict=True):
        mixture += weight * probs
        expected_entropy += weight * _entropy_per_cell(probs)

    eig = np.mean(_entropy_per_cell(mixture) - expected_entropy) / np.log(NUM_CLASSES)
    return float(np.clip(eig, 0.0, 1.0))


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
    # Clip to prevent log(0)
    p = np.clip(prediction_tensor, 1e-10, 1.0)
    # Normalize along class axis
    p = p / np.sum(p, axis=2, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=2)
    return entropy


def _entropy_per_cell(prob_tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    probs = np.clip(prob_tensor, 1e-10, 1.0)
    probs /= np.sum(probs, axis=2, keepdims=True)
    return -np.sum(probs * np.log(probs), axis=2)


def _simulate_particle_viewport_probs(
    particle: Particle,
    candidate: ViewportCandidate,
    initial_state: InitialState,
    base_seed: int,
    n_runs: int,
) -> NDArray[np.float64]:
    from astar_twin.engine import Simulator
    from astar_twin.mc.aggregate import aggregate_runs
    from astar_twin.mc.runner import MCRunner

    map_height = len(initial_state.grid)
    map_width = len(initial_state.grid[0])
    simulator = Simulator(params=particle.to_simulation_params())
    runner = MCRunner(simulator)
    runs = runner.run_batch(initial_state, n_runs=n_runs, base_seed=base_seed)
    tensor = aggregate_runs(runs, map_height, map_width)
    window = tensor[
        candidate.y : candidate.y + candidate.h,
        candidate.x : candidate.x + candidate.w,
    ]
    return apply_legality_filter_to_viewport(window, initial_state, candidate.x, candidate.y)


def _shortlist_size(total_candidates: int, batch_size: int) -> int:
    target = max(batch_size, batch_size * HEURISTIC_SHORTLIST_FACTOR)
    return min(total_candidates, target)


def _copy_candidate_with_score(candidate: ViewportCandidate, score: float) -> ViewportCandidate:
    return ViewportCandidate(
        x=candidate.x,
        y=candidate.y,
        w=candidate.w,
        h=candidate.h,
        category=candidate.category,
        score=score,
    )


def _sorted_candidate_scores(
    scored: list[CandidateScore],
) -> list[CandidateScore]:
    return sorted(
        scored,
        key=lambda item: (
            -item.final_score,
            -item.heuristic_score,
            item.seed_index,
            item.candidate.y,
            item.candidate.x,
            item.candidate.h,
            item.candidate.w,
            item.candidate.category,
        ),
    )


def _build_entropy_map(
    seed_predictions: dict[int, NDArray[np.float64]] | None,
    seed_index: int,
    initial_state: InitialState,
) -> NDArray[np.float64] | None:
    if seed_predictions is None:
        return None

    prediction = seed_predictions.get(seed_index)
    if prediction is None:
        return None

    legal_prediction = apply_legality_filter(prediction, initial_state)
    return compute_entropy_map(legal_prediction)


def _collect_heuristic_candidates(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    seed_predictions: dict[int, NDArray[np.float64]] | None,
    allow_high_overlap: bool,
) -> list[CandidateScore]:
    scored: list[CandidateScore] = []
    queries_used = state.queries_used

    for seed_idx, candidates in state.seed_candidates.items():
        if seed_idx >= len(initial_states):
            continue
        initial_state = initial_states[seed_idx]
        queried_vps = [q.viewport for q in state.queries_for_seed(seed_idx)]
        entropy_map = _build_entropy_map(seed_predictions, seed_idx, initial_state)

        for candidate in candidates:
            heuristic_score = score_candidate(
                candidate,
                seed_idx,
                posterior,
                initial_state,
                queried_vps,
                queries_used,
                entropy_map,
                allow_high_overlap=allow_high_overlap,
            )
            if heuristic_score >= 0:
                scored.append(
                    CandidateScore(
                        heuristic_score=heuristic_score,
                        final_score=heuristic_score,
                        seed_index=seed_idx,
                        candidate=candidate,
                    )
                )

    return _sorted_candidate_scores(scored)


def _rerank_shortlist_with_eig(
    shortlist: list[CandidateScore],
    posterior: PosteriorState,
    initial_states: list[InitialState],
) -> list[CandidateScore]:
    reranked: list[CandidateScore] = []
    for item in shortlist:
        eig_score = approximate_expected_information_gain(
            item.candidate,
            posterior,
            initial_states[item.seed_index],
            seed_index=item.seed_index,
        )
        final_score = RERANK_HEURISTIC_WEIGHT * item.heuristic_score + RERANK_EIG_WEIGHT * eig_score
        reranked.append(
            CandidateScore(
                heuristic_score=item.heuristic_score,
                final_score=final_score,
                seed_index=item.seed_index,
                candidate=item.candidate,
            )
        )

    return _sorted_candidate_scores(reranked)


def _select_ranked_batch(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    seed_predictions: dict[int, NDArray[np.float64]] | None,
    batch_size: int,
    allow_high_overlap: bool,
) -> list[tuple[int, ViewportCandidate]]:
    scored = _collect_heuristic_candidates(
        state,
        posterior,
        initial_states,
        seed_predictions,
        allow_high_overlap=allow_high_overlap,
    )
    if not scored:
        return []

    shortlist = scored[: _shortlist_size(len(scored), batch_size)]
    reranked = _rerank_shortlist_with_eig(shortlist, posterior, initial_states)
    result: list[tuple[int, ViewportCandidate]] = []
    for item in reranked[:batch_size]:
        result.append(
            (item.seed_index, _copy_candidate_with_score(item.candidate, item.final_score))
        )
    return result


def select_adaptive_batch(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    seed_predictions: dict[int, NDArray[np.float64]] | None = None,
    batch_size: int = ADAPTIVE_BATCH_SIZE,
) -> list[tuple[int, ViewportCandidate]]:
    """Select the next batch of adaptive queries across all seeds.

    Scores all candidate viewports across all seeds and picks the top batch_size.
    """
    if state.queries_remaining <= 0:
        return []

    actual_batch = min(batch_size, state.queries_remaining)
    return _select_ranked_batch(
        state,
        posterior,
        initial_states,
        seed_predictions,
        batch_size=actual_batch,
        allow_high_overlap=False,
    )


def check_contradiction_triggers(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState] | None = None,
) -> bool:
    """Check if any contradiction trigger fires, releasing reserve queries.

    Triggers:
      - ESS < 6
      - Top candidate argmax disagreement > threshold in any seed
      - Any seed has < MIN_QUERIES_PER_SEED after adaptive phase
    """
    # ESS trigger
    if posterior.ess < ESS_CONTRADICTION_THRESHOLD:
        return True

    if initial_states is not None:
        for seed_idx, candidates in state.seed_candidates.items():
            if seed_idx >= len(initial_states) or not candidates:
                continue
            best_candidate = candidates[0]
            if check_argmax_disagreement(best_candidate, posterior, initial_states[seed_idx]):
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

    contradiction = check_contradiction_triggers(state, posterior, initial_states)

    if contradiction:
        # Allow high-overlap queries for contradiction resolution
        return _select_contradiction_queries(
            state, posterior, initial_states, seed_predictions, remaining
        )
    else:
        # Release as normal adaptive batches
        return select_adaptive_batch(
            state, posterior, initial_states, seed_predictions, batch_size=remaining
        )


def _select_contradiction_queries(
    state: AllocationState,
    posterior: PosteriorState,
    initial_states: list[InitialState],
    seed_predictions: dict[int, NDArray[np.float64]] | None,
    n_queries: int,
) -> list[tuple[int, ViewportCandidate]]:
    scored = _collect_heuristic_candidates(
        state,
        posterior,
        initial_states,
        seed_predictions,
        allow_high_overlap=True,
    )
    if not scored:
        return []

    shortlist = scored[: _shortlist_size(len(scored), n_queries)]
    reranked = _rerank_shortlist_with_eig(shortlist, posterior, initial_states)

    under_queried: list[CandidateScore] = []
    normal: list[CandidateScore] = []
    for item in reranked:
        if len(state.queries_for_seed(item.seed_index)) < MIN_QUERIES_PER_SEED:
            under_queried.append(item)
        else:
            normal.append(item)

    result: list[tuple[int, ViewportCandidate]] = []
    for item in (under_queried + normal)[:n_queries]:
        result.append(
            (item.seed_index, _copy_candidate_with_score(item.candidate, item.final_score))
        )
    return result


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
