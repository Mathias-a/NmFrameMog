from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams
from astar_twin.solver.policy.hotspots import ViewportCandidate, generate_hotspots
from astar_twin.solver.predict.finalize import finalize_tensor

_CATEGORY_PRIORITY = {
    "coastal": 1.0,
    "corridor": 0.9,
    "frontier": 0.8,
    "reclaim": 0.7,
    "fallback": 0.45,
}

_MAX_REDUCED_ACTIONS = 6
_MAX_CATEGORY_CANDIDATES = 2
_MAX_PLANNED_ACTIONS = 2
_MAX_REFINEMENTS = 1
_ROLLOUT_RUNS = 3
_REFINEMENT_RUNS = 5
_GLOBAL_BASE_RUNS = 8
_GLOBAL_RUNS_PER_REMAINING_BUDGET = 1
_MIN_GLOBAL_RUNS = 6
_ROLLOUT_DISCOUNT = 0.65
_NON_OVERLAP_THRESHOLD = 0.25
_SECOND_ACTION_RESERVE = 3
_REFINEMENT_RESERVE = 2
_W_STATIC = 0.0
_W_FOREST_ISOLATED = 0.18
_W_FOREST_FRONTIER = 0.58
_W_PLAINS_ISOLATED = 0.24
_W_PLAINS_FRONTIER = 0.68
_W_SETTLEMENT = 0.88
_W_PORT = 0.84
_W_RUIN = 0.72
_FRONTIER_RADIUS = 5
_LOW_ENTROPY_THRESHOLD = 0.10
_HIGH_ENTROPY_THRESHOLD = 0.50
_FILTER_INLAND_TEMPLATE = np.array([0.55, 0.18, 0.0, 0.07, 0.20, 0.0], dtype=np.float64)
_FILTER_COASTAL_TEMPLATE = np.array([0.48, 0.17, 0.12, 0.06, 0.17, 0.0], dtype=np.float64)


class ReducedRolloutPlannerStrategy:
    def __init__(self, params: SimulationParams | None = None) -> None:
        self._params = params or SimulationParams()
        self._seed_call_count = 0
        self._learned_attention_profile: NDArray[np.float64] | None = None

    @property
    def name(self) -> str:
        return "reduced_rollout_planner"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        if budget.used == 0:
            self.reset()

        grid = initial_state.grid
        height = len(grid)
        width = len(grid[0])
        is_first_seed = self._seed_call_count == 0
        self._seed_call_count += 1

        prior = self._build_prior(initial_state, height, width)
        dynamism = self._compute_dynamism_map(initial_state, height, width)

        reduced_actions = self._select_reduced_actions(
            generate_hotspots(initial_state, height, width),
            initial_state,
        )
        immediate_scores, rollout_cache = self._score_reduced_actions(
            reduced_actions,
            initial_state,
            height,
            width,
            base_seed,
        )
        executed_actions = self._select_executed_actions(
            reduced_actions,
            immediate_scores,
            budget,
        )
        refinement_actions = self._select_refinement_actions(
            reduced_actions,
            executed_actions,
            immediate_scores,
            rollout_cache,
            budget,
        )

        simulator = Simulator(params=self._params)
        mc_runner = MCRunner(simulator)
        global_runs = max(
            _MIN_GLOBAL_RUNS,
            _GLOBAL_BASE_RUNS + budget.remaining * _GLOBAL_RUNS_PER_REMAINING_BUDGET,
        )
        global_tensor = aggregate_runs(
            mc_runner.run_batch(
                initial_state=initial_state,
                n_runs=global_runs,
                base_seed=base_seed + 20_000,
            ),
            height,
            width,
        )

        focused_tensor, local_attention = self._apply_local_refinements(
            global_tensor,
            initial_state,
            executed_actions,
            refinement_actions,
            height,
            width,
            base_seed,
            rollout_cache,
        )

        focused_entropy = self._compute_entropy(focused_tensor)
        entropy_weights = self._entropy_to_blend_weights(
            focused_entropy,
            initial_state,
            height,
            width,
        )
        if (
            self._learned_attention_profile is not None
            and not is_first_seed
            and self._learned_attention_profile.shape == (height, width)
        ):
            entropy_weights = np.maximum(entropy_weights, 0.35 * self._learned_attention_profile)

        if is_first_seed:
            self._learned_attention_profile = entropy_weights.copy()

        base_blend = np.clip(0.15 + 0.60 * dynamism + 0.20 * local_attention, 0.0, 0.97)
        blend_weights = np.clip(np.maximum(base_blend, entropy_weights), 0.0, 0.97)
        result = (
            blend_weights[:, :, np.newaxis] * focused_tensor
            + (1.0 - blend_weights[:, :, np.newaxis]) * prior
        )
        result = self._apply_hard_limits(result, initial_state, height, width)

        return finalize_tensor(result, height, width, initial_state)

    def _select_reduced_actions(
        self,
        candidates: list[ViewportCandidate],
        initial_state: InitialState,
    ) -> list[ViewportCandidate]:
        ranked = sorted(
            candidates,
            key=lambda candidate: self._candidate_priority(candidate, initial_state),
            reverse=True,
        )
        reduced: list[ViewportCandidate] = []
        category_counts: dict[str, int] = {}

        for candidate in ranked:
            category_count = category_counts.get(candidate.category, 0)
            if category_count >= _MAX_CATEGORY_CANDIDATES:
                continue
            reduced.append(candidate)
            category_counts[candidate.category] = category_count + 1
            if len(reduced) >= _MAX_REDUCED_ACTIONS:
                break

        return reduced

    def _candidate_priority(
        self,
        candidate: ViewportCandidate,
        initial_state: InitialState,
    ) -> tuple[float, float, float, int, int, int, int]:
        learned_attention = self._window_attention(candidate)
        settlement_density = self._settlement_density(candidate, initial_state)
        area = candidate.w * candidate.h
        return (
            _CATEGORY_PRIORITY.get(candidate.category, 0.0),
            learned_attention,
            settlement_density,
            area,
            -candidate.y,
            -candidate.x,
            -len(candidate.category),
        )

    def _score_reduced_actions(
        self,
        reduced_actions: list[ViewportCandidate],
        initial_state: InitialState,
        height: int,
        width: int,
        base_seed: int,
    ) -> tuple[
        dict[tuple[int, int, int, int], float], dict[tuple[int, int, int, int], NDArray[np.float64]]
    ]:
        simulator = Simulator(params=self._params)
        mc_runner = MCRunner(simulator)
        immediate_scores: dict[tuple[int, int, int, int], float] = {}
        rollout_cache: dict[tuple[int, int, int, int], NDArray[np.float64]] = {}

        for index, action in enumerate(reduced_actions):
            key = self._candidate_key(action)
            tensor = aggregate_runs(
                mc_runner.run_batch(
                    initial_state=initial_state,
                    n_runs=_ROLLOUT_RUNS,
                    base_seed=self._candidate_seed(base_seed, action, 1_000 + index * 101),
                ),
                height,
                width,
            )
            rollout_cache[key] = tensor
            immediate_scores[key] = self._immediate_action_score(action, tensor, initial_state)

        return immediate_scores, rollout_cache

    def _select_executed_actions(
        self,
        reduced_actions: list[ViewportCandidate],
        immediate_scores: dict[tuple[int, int, int, int], float],
        budget: Budget,
    ) -> list[ViewportCandidate]:
        if budget.remaining <= 0 or not reduced_actions:
            return []

        ranked = sorted(
            reduced_actions,
            key=lambda action: self._score_first_action_with_rollout(
                action,
                reduced_actions,
                immediate_scores,
            ),
            reverse=True,
        )

        selected: list[ViewportCandidate] = []
        first_action = ranked[0]
        budget.consume()
        selected.append(first_action)

        if len(ranked) > 1 and budget.remaining > _SECOND_ACTION_RESERVE:
            followup = self._best_non_overlapping_followup(
                first_action,
                reduced_actions,
                immediate_scores,
            )
            if followup is not None:
                budget.consume()
                selected.append(followup)

        return selected[:_MAX_PLANNED_ACTIONS]

    def _select_refinement_actions(
        self,
        reduced_actions: list[ViewportCandidate],
        executed_actions: list[ViewportCandidate],
        immediate_scores: dict[tuple[int, int, int, int], float],
        rollout_cache: dict[tuple[int, int, int, int], NDArray[np.float64]],
        budget: Budget,
    ) -> list[ViewportCandidate]:
        if not reduced_actions or budget.remaining <= _REFINEMENT_RESERVE:
            return []

        ranked = sorted(
            reduced_actions,
            key=lambda action: self._refinement_priority(
                action,
                executed_actions,
                immediate_scores,
                rollout_cache,
            ),
            reverse=True,
        )
        refinements = ranked[:_MAX_REFINEMENTS]
        for _ in refinements:
            budget.consume()
        return refinements

    def _refinement_priority(
        self,
        action: ViewportCandidate,
        executed_actions: list[ViewportCandidate],
        immediate_scores: dict[tuple[int, int, int, int], float],
        rollout_cache: dict[tuple[int, int, int, int], NDArray[np.float64]],
    ) -> tuple[float, float, float, int, int]:
        overlap_bonus = 0.0
        if executed_actions:
            overlap_bonus = max(self._overlap_fraction(action, other) for other in executed_actions)
        return (
            self._candidate_entropy_score(action, rollout_cache),
            immediate_scores.get(self._candidate_key(action), 0.0),
            overlap_bonus,
            -action.y,
            -action.x,
        )

    def _candidate_entropy_score(
        self,
        action: ViewportCandidate,
        rollout_cache: dict[tuple[int, int, int, int], NDArray[np.float64]],
    ) -> float:
        tensor = rollout_cache.get(self._candidate_key(action))
        if tensor is None:
            return float("-inf")

        entropy = self._compute_entropy(tensor)
        window = entropy[action.y : action.y + action.h, action.x : action.x + action.w]
        if window.size == 0:
            return float("-inf")
        return float(np.mean(window) / max(float(np.log(NUM_CLASSES)), 1e-10))

    def _score_first_action_with_rollout(
        self,
        first_action: ViewportCandidate,
        reduced_actions: list[ViewportCandidate],
        immediate_scores: dict[tuple[int, int, int, int], float],
    ) -> float:
        immediate_score = immediate_scores[self._candidate_key(first_action)]
        followup = self._best_non_overlapping_followup(
            first_action,
            reduced_actions,
            immediate_scores,
        )
        if followup is None:
            return immediate_score
        followup_score = immediate_scores[self._candidate_key(followup)]
        return immediate_score + _ROLLOUT_DISCOUNT * followup_score

    def _best_non_overlapping_followup(
        self,
        first_action: ViewportCandidate,
        reduced_actions: list[ViewportCandidate],
        immediate_scores: dict[tuple[int, int, int, int], float],
    ) -> ViewportCandidate | None:
        best_action: ViewportCandidate | None = None
        best_score = float("-inf")

        for candidate in reduced_actions:
            if self._candidate_key(candidate) == self._candidate_key(first_action):
                continue
            if self._overlap_fraction(first_action, candidate) > _NON_OVERLAP_THRESHOLD:
                continue
            score = immediate_scores[self._candidate_key(candidate)]
            if score > best_score:
                best_score = score
                best_action = candidate

        return best_action

    def _apply_local_refinements(
        self,
        global_tensor: NDArray[np.float64],
        initial_state: InitialState,
        executed_actions: list[ViewportCandidate],
        refinement_actions: list[ViewportCandidate],
        height: int,
        width: int,
        base_seed: int,
        rollout_cache: dict[tuple[int, int, int, int], NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        focused = global_tensor.copy()
        local_attention = np.zeros((height, width), dtype=np.float64)
        simulator = Simulator(params=self._params)
        mc_runner = MCRunner(simulator)

        for index, action in enumerate(executed_actions):
            cached = rollout_cache.get(self._candidate_key(action))
            if cached is None:
                continue
            self._blend_window(focused, cached, action, weight=0.35)
            local_attention[action.y : action.y + action.h, action.x : action.x + action.w] = (
                np.maximum(
                    local_attention[action.y : action.y + action.h, action.x : action.x + action.w],
                    0.65,
                )
            )

        for index, action in enumerate(refinement_actions):
            refined_tensor = aggregate_runs(
                mc_runner.run_batch(
                    initial_state=initial_state,
                    n_runs=_REFINEMENT_RUNS,
                    base_seed=self._candidate_seed(base_seed, action, 10_000 + index * 131),
                ),
                height,
                width,
            )
            self._blend_window(focused, refined_tensor, action, weight=0.55)
            local_attention[action.y : action.y + action.h, action.x : action.x + action.w] = (
                np.maximum(
                    local_attention[action.y : action.y + action.h, action.x : action.x + action.w],
                    1.0,
                )
            )

        return focused, local_attention

    def _blend_window(
        self,
        target: NDArray[np.float64],
        source: NDArray[np.float64],
        action: ViewportCandidate,
        weight: float,
    ) -> None:
        y_slice = slice(action.y, action.y + action.h)
        x_slice = slice(action.x, action.x + action.w)
        target[y_slice, x_slice] = (1.0 - weight) * target[y_slice, x_slice] + weight * source[
            y_slice,
            x_slice,
        ]

    def _immediate_action_score(
        self,
        action: ViewportCandidate,
        tensor: NDArray[np.float64],
        initial_state: InitialState,
    ) -> float:
        y_slice = slice(action.y, action.y + action.h)
        x_slice = slice(action.x, action.x + action.w)
        window = tensor[y_slice, x_slice]
        if window.size == 0:
            return 0.0

        clipped = np.clip(window, 1e-10, 1.0)
        entropy = -np.sum(clipped * np.log(clipped), axis=2) / np.log(NUM_CLASSES)
        dynamic_mass = np.mean(
            window[:, :, ClassIndex.SETTLEMENT]
            + window[:, :, ClassIndex.PORT]
            + window[:, :, ClassIndex.RUIN]
            + 0.5 * window[:, :, ClassIndex.FOREST]
        )
        settlement_focus = self._settlement_density(action, initial_state)
        category_bonus = 0.10 * _CATEGORY_PRIORITY.get(action.category, 0.0)

        return float(
            0.55 * float(np.mean(entropy))
            + 0.30 * float(dynamic_mass)
            + 0.15 * settlement_focus
            + category_bonus
        )

    def _settlement_density(
        self,
        action: ViewportCandidate,
        initial_state: InitialState,
    ) -> float:
        alive_total = sum(1 for settlement in initial_state.settlements if settlement.alive)
        if alive_total == 0:
            return 0.0

        alive_in_window = 0
        for settlement in initial_state.settlements:
            if not settlement.alive:
                continue
            if (
                action.x <= settlement.x < action.x + action.w
                and action.y <= settlement.y < action.y + action.h
            ):
                alive_in_window += 1

        return alive_in_window / alive_total

    def _window_attention(self, action: ViewportCandidate) -> float:
        if self._learned_attention_profile is None:
            return 0.0

        profile = self._learned_attention_profile
        height, width = profile.shape
        y_start = min(action.y, height)
        y_end = min(action.y + action.h, height)
        x_start = min(action.x, width)
        x_end = min(action.x + action.w, width)
        if y_end <= y_start or x_end <= x_start:
            return 0.0
        return float(np.mean(profile[y_start:y_end, x_start:x_end]))

    def _compute_entropy(self, tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(tensor, 1e-10, 1.0)
        normalized = clipped / np.sum(clipped, axis=2, keepdims=True)
        return -np.sum(normalized * np.log(normalized), axis=2)

    def _entropy_to_blend_weights(
        self,
        entropy: NDArray[np.float64],
        initial_state: InitialState,
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        max_entropy = float(np.log(NUM_CLASSES))
        normalized_entropy = np.clip(entropy / max(max_entropy, 1e-10), 0.0, 1.0)
        weights = np.clip(
            (normalized_entropy - _LOW_ENTROPY_THRESHOLD)
            / max(_HIGH_ENTROPY_THRESHOLD - _LOW_ENTROPY_THRESHOLD, 1e-6),
            0.0,
            1.0,
        )
        weights = 0.10 + 0.85 * weights

        for y in range(height):
            for x in range(width):
                code = initial_state.grid[y][x]
                if code == TerrainCode.OCEAN or code == TerrainCode.MOUNTAIN:
                    weights[y, x] = 0.0

        return weights

    def _candidate_seed(
        self,
        base_seed: int,
        candidate: ViewportCandidate,
        offset: int,
    ) -> int:
        return (
            base_seed
            + offset
            + candidate.x * 17
            + candidate.y * 31
            + candidate.w * 43
            + candidate.h * 59
        )

    def _candidate_key(self, candidate: ViewportCandidate) -> tuple[int, int, int, int]:
        return (candidate.x, candidate.y, candidate.w, candidate.h)

    def _overlap_fraction(self, left: ViewportCandidate, right: ViewportCandidate) -> float:
        return max(left.overlap_fraction(right), right.overlap_fraction(left))

    def _compute_dynamism_map(
        self,
        initial_state: InitialState,
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        grid = initial_state.grid
        dynamism = np.zeros((height, width), dtype=np.float64)
        settlement_positions = [
            (settlement.x, settlement.y)
            for settlement in initial_state.settlements
            if settlement.alive
        ]
        if settlement_positions:
            dist_map = self._settlement_distance_map(settlement_positions, height, width)
        else:
            dist_map = np.full((height, width), float(_FRONTIER_RADIUS + 1), dtype=np.float64)

        for y in range(height):
            for x in range(width):
                code = grid[y][x]
                near_settlement = dist_map[y, x] <= _FRONTIER_RADIUS
                if code == TerrainCode.OCEAN or code == TerrainCode.MOUNTAIN:
                    dynamism[y, x] = _W_STATIC
                elif code == TerrainCode.SETTLEMENT:
                    dynamism[y, x] = _W_SETTLEMENT
                elif code == TerrainCode.PORT:
                    dynamism[y, x] = _W_PORT
                elif code == TerrainCode.RUIN:
                    dynamism[y, x] = _W_RUIN
                elif code == TerrainCode.FOREST:
                    dynamism[y, x] = _W_FOREST_FRONTIER if near_settlement else _W_FOREST_ISOLATED
                else:
                    dynamism[y, x] = _W_PLAINS_FRONTIER if near_settlement else _W_PLAINS_ISOLATED

        return dynamism

    def _settlement_distance_map(
        self,
        positions: list[tuple[int, int]],
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        dist = np.full((height, width), float(height + width), dtype=np.float64)
        for settlement_x, settlement_y in positions:
            for y in range(height):
                for x in range(width):
                    d = max(abs(x - settlement_x), abs(y - settlement_y))
                    if d < dist[y, x]:
                        dist[y, x] = d
        return dist

    def _build_prior(
        self,
        initial_state: InitialState,
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        grid = initial_state.grid
        tensor = np.full((height, width, NUM_CLASSES), 0.01, dtype=np.float64)
        coastal_cells = self._find_coastal_cells(grid, height, width)

        for y in range(height):
            for x in range(width):
                code = grid[y][x]
                is_coastal = (x, y) in coastal_cells
                if code == TerrainCode.OCEAN:
                    tensor[y, x, ClassIndex.EMPTY] = 0.96
                elif code == TerrainCode.MOUNTAIN:
                    tensor[y, x, ClassIndex.MOUNTAIN] = 0.96
                elif code == TerrainCode.SETTLEMENT:
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.48
                    tensor[y, x, ClassIndex.RUIN] = 0.21
                    tensor[y, x, ClassIndex.EMPTY] = 0.12
                    tensor[y, x, ClassIndex.PORT] = 0.12 if is_coastal else 0.03
                    tensor[y, x, ClassIndex.FOREST] = 0.03
                elif code == TerrainCode.PORT:
                    tensor[y, x, ClassIndex.PORT] = 0.44
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.15
                    tensor[y, x, ClassIndex.RUIN] = 0.22
                    tensor[y, x, ClassIndex.EMPTY] = 0.11
                    tensor[y, x, ClassIndex.FOREST] = 0.03
                elif code == TerrainCode.RUIN:
                    tensor[y, x, ClassIndex.RUIN] = 0.31
                    tensor[y, x, ClassIndex.FOREST] = 0.25
                    tensor[y, x, ClassIndex.EMPTY] = 0.23
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.10
                    tensor[y, x, ClassIndex.PORT] = 0.05 if is_coastal else 0.01
                elif code == TerrainCode.FOREST:
                    tensor[y, x, ClassIndex.FOREST] = 0.70
                    tensor[y, x, ClassIndex.EMPTY] = 0.15
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.08
                    tensor[y, x, ClassIndex.PORT] = 0.03 if is_coastal else 0.01
                else:
                    tensor[y, x, ClassIndex.EMPTY] = 0.64
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.13
                    tensor[y, x, ClassIndex.FOREST] = 0.10
                    tensor[y, x, ClassIndex.PORT] = 0.05 if is_coastal else 0.01
                    tensor[y, x, ClassIndex.RUIN] = 0.04

        return tensor / np.maximum(np.sum(tensor, axis=2, keepdims=True), 1e-10)

    def _find_coastal_cells(
        self,
        grid: list[list[int]],
        height: int,
        width: int,
    ) -> set[tuple[int, int]]:
        coastal: set[tuple[int, int]] = set()
        for y in range(height):
            for x in range(width):
                if grid[y][x] == TerrainCode.OCEAN:
                    continue
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and grid[ny][nx] == TerrainCode.OCEAN:
                        coastal.add((x, y))
                        break
        return coastal

    def _apply_hard_limits(
        self,
        tensor: NDArray[np.float64],
        initial_state: InitialState,
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        constrained = tensor.copy()
        coastal = self._find_coastal_cells(initial_state.grid, height, width)

        for y in range(height):
            for x in range(width):
                code = initial_state.grid[y][x]

                if code == TerrainCode.OCEAN:
                    constrained[y, x, :] = 0.0
                    constrained[y, x, ClassIndex.EMPTY] = 1.0
                    continue

                if code == TerrainCode.MOUNTAIN:
                    constrained[y, x, :] = 0.0
                    constrained[y, x, ClassIndex.MOUNTAIN] = 1.0
                    continue

                constrained[y, x, ClassIndex.MOUNTAIN] = 0.0
                if (x, y) not in coastal:
                    constrained[y, x, ClassIndex.PORT] = 0.0

                row_sum = float(np.sum(constrained[y, x]))
                if row_sum <= 0.0:
                    template = (
                        _FILTER_COASTAL_TEMPLATE if (x, y) in coastal else _FILTER_INLAND_TEMPLATE
                    )
                    constrained[y, x, :] = template
                else:
                    constrained[y, x, :] = constrained[y, x, :] / row_sum

        return constrained

    def reset(self) -> None:
        self._seed_call_count = 0
        self._learned_attention_profile = None
