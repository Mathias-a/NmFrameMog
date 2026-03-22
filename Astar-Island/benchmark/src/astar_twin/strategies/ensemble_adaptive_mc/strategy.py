from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams
from astar_twin.solver.predict.finalize import finalize_tensor

_FILTER_INLAND_TEMPLATE = np.array([0.806, 0.136, 0.0, 0.013, 0.045, 0.0], dtype=np.float64)
_FILTER_COASTAL_TEMPLATE = np.array([0.457, 0.094, 0.174, 0.028, 0.247, 0.0], dtype=np.float64)

_EXPERT_WEIGHTS = np.array([0.55, 0.25, 0.20], dtype=np.float64)
_TOTAL_RUN_BANK = 1000
_TARGET_RUNS = 200
_MIN_RUNS_PER_SEED = 160
_MAX_RUNS_PER_SEED = 250
_SCOUT_FRACTION = 0.25
_MAX_DEFAULT_REALLOCATION = 0.15
_TRANSFER_WEIGHT = 0.12
_TRANSFER_CAP = 0.15
_TRANSFER_MIN_SUPPORT = 25
_TRANSFER_MATCH_WEIGHT = 0.06
_NEAR_SETTLEMENT_RADIUS = 5
_CLOSE_PAIR_RADIUS = 6
_HIGH_ENTROPY_THRESHOLD = 0.5
_LOW_ENTROPY_THRESHOLD = 0.1
_EXPERT_SEED_STRIDE = 100000
_SEED_CALL_STRIDE = 1000
_SCOUT_PASS_OFFSET = 0
_EXPLOIT_PASS_OFFSET = 50000

BucketKey = tuple[int, bool, int, bool]


class EnsembleAdaptiveMCStrategy:
    def __init__(self, params: SimulationParams | None = None) -> None:
        self._base_params = params or SimulationParams()
        self._expert_params = self._build_expert_params(self._base_params)
        self._expert_runners = tuple(MCRunner(Simulator(params=p)) for p in self._expert_params)
        self._active_budget_identity: int | None = None
        self.reset()

    @property
    def name(self) -> str:
        return "ensemble_adaptive_mc"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        if budget.used == 0 and self._active_budget_identity != id(budget):
            self.reset()
            self._active_budget_identity = id(budget)
        elif self._active_budget_identity is None:
            self._active_budget_identity = id(budget)

        height = len(initial_state.grid)
        width = len(initial_state.grid[0])
        coastal_cells = self._find_coastal_cells(initial_state.grid, height, width)
        distance_map = self._settlement_distance_map(initial_state, height, width)
        frontier_mask = self._frontier_mask(initial_state.grid, height, width)
        prior = self._build_prior(initial_state, height, width)

        seed_runs = self._allocate_seed_runs(
            self._compute_seed_hardness(
                initial_state=initial_state,
                coastal_cells=coastal_cells,
                frontier_mask=frontier_mask,
                distance_map=distance_map,
                height=height,
                width=width,
            )
        )
        scout_runs = max(3, int(round(seed_runs * _SCOUT_FRACTION)))
        exploit_runs = max(0, seed_runs - scout_runs)

        scout_allocations = self._split_integer_budget(scout_runs, _EXPERT_WEIGHTS)
        scout_tensors = self._run_expert_pass(
            initial_state=initial_state,
            allocations=scout_allocations,
            base_seed=base_seed,
            pass_offset=_SCOUT_PASS_OFFSET + self._seed_call_count * _SEED_CALL_STRIDE,
            height=height,
            width=width,
        )

        dynamic_mask = self._dynamic_cell_mask(initial_state.grid, distance_map, frontier_mask)
        exploit_weights = self._exploit_weights(scout_tensors, scout_allocations, dynamic_mask)
        exploit_allocations = self._split_integer_budget(exploit_runs, exploit_weights)
        exploit_tensors = self._run_expert_pass(
            initial_state=initial_state,
            allocations=exploit_allocations,
            base_seed=base_seed,
            pass_offset=_EXPLOIT_PASS_OFFSET + self._seed_call_count * _SEED_CALL_STRIDE,
            height=height,
            width=width,
        )

        expert_tensors, total_runs = self._merge_pass_tensors(
            scout_tensors=scout_tensors,
            scout_allocations=scout_allocations,
            exploit_tensors=exploit_tensors,
            exploit_allocations=exploit_allocations,
            height=height,
            width=width,
        )
        ensemble_mc_raw = self._combine_expert_tensors(expert_tensors, total_runs)
        ensemble_mc = self._refine_ensemble_tensor(
            ensemble_mc_raw,
            prior,
            initial_state,
            height,
            width,
        )
        result = ensemble_mc

        masked = self._apply_hard_mask(result, initial_state, height, width)
        self._seed_call_count += 1
        return finalize_tensor(masked, height, width, initial_state)

    def reset(self) -> None:
        self._seed_call_count = 0
        self._remaining_run_bank = _TOTAL_RUN_BANK
        self._learned_entropy_profile: NDArray[np.float64] | None = None
        self._bucket_residual_sums: dict[BucketKey, NDArray[np.float64]] = {}
        self._bucket_support: dict[BucketKey, int] = {}

    def _build_expert_params(
        self,
        base_params: SimulationParams,
    ) -> tuple[SimulationParams, SimulationParams, SimulationParams]:
        growth_trade = replace(
            base_params,
            prosperity_threshold_port=1.53,
            prosperity_threshold_expand=1.26,
            expansion_rate=0.216,
            expansion_site_coastal_bonus=0.28,
            trade_range=10,
            trade_value_scale=0.264,
        )
        collapse_reclaim = replace(
            base_params,
            winter_food_loss_flat=0.207,
            winter_food_loss_per_population=0.12,
            collapse_threshold=0.85,
            raid_base_prob=0.072,
            raid_damage_frac=0.216,
            reclaim_rate=0.144,
            ruin_forest_rate=0.115,
        )
        return (base_params, growth_trade, collapse_reclaim)

    def _allocate_seed_runs(self, hardness: float) -> int:
        target_runs = int(
            round(_TARGET_RUNS + (hardness - 0.5) * (_MAX_RUNS_PER_SEED - _MIN_RUNS_PER_SEED))
        )
        target_runs = int(np.clip(target_runs, _MIN_RUNS_PER_SEED, _MAX_RUNS_PER_SEED))
        unseen_after_this_call = max(0, 4 - self._seed_call_count)
        reserve = unseen_after_this_call * _MIN_RUNS_PER_SEED
        available = self._remaining_run_bank - reserve
        if unseen_after_this_call == 0:
            runs = min(target_runs, self._remaining_run_bank)
        else:
            runs = min(target_runs, max(_MIN_RUNS_PER_SEED, available))
        runs = max(1, runs)
        self._remaining_run_bank -= runs
        return runs

    def _compute_seed_hardness(
        self,
        initial_state: InitialState,
        coastal_cells: set[tuple[int, int]],
        frontier_mask: NDArray[np.bool_],
        distance_map: NDArray[np.float64],
        height: int,
        width: int,
    ) -> float:
        alive_positions = [(s.x, s.y) for s in initial_state.settlements if s.alive]
        grid = initial_state.grid
        land_cells = 0
        ruin_cells = 0
        for y in range(height):
            for x in range(width):
                code = grid[y][x]
                if code not in (TerrainCode.OCEAN, TerrainCode.MOUNTAIN):
                    land_cells += 1
                if code == TerrainCode.RUIN:
                    ruin_cells += 1

        alive_count = len(alive_positions)
        close_pairs = 0
        coastal_alive = 0
        for idx, (ax, ay) in enumerate(alive_positions):
            if (ax, ay) in coastal_cells:
                coastal_alive += 1
            for bx, by in alive_positions[idx + 1 :]:
                if max(abs(ax - bx), abs(ay - by)) <= _CLOSE_PAIR_RADIUS:
                    close_pairs += 1

        max_pairs = alive_count * (alive_count - 1) // 2
        norm_alive_settlements = min(alive_count / max(4.0, land_cells / 90.0), 1.0)
        norm_close_pairs = close_pairs / max(float(max_pairs), 1.0)
        coastal_settlement_share = coastal_alive / max(float(alive_count), 1.0)
        frontier_cells = int(np.count_nonzero(frontier_mask))
        norm_frontier_cells = min(frontier_cells / max(1.0, land_cells * 0.35), 1.0)
        norm_ruin_cells = min(ruin_cells / max(1.0, land_cells * 0.10), 1.0)

        hardness = (
            0.35 * norm_alive_settlements
            + 0.25 * norm_close_pairs
            + 0.20 * coastal_settlement_share
            + 0.10 * norm_frontier_cells
            + 0.10 * norm_ruin_cells
        )
        if not alive_positions:
            hardness *= 0.6
            hardness += 0.1 * min(
                np.count_nonzero(distance_map <= _NEAR_SETTLEMENT_RADIUS) / 25.0, 1.0
            )
        return float(np.clip(hardness, 0.0, 1.0))

    def _split_integer_budget(
        self,
        total_runs: int,
        weights: NDArray[np.float64],
    ) -> tuple[int, int, int]:
        if total_runs <= 0:
            return (0, 0, 0)
        normalized = weights / np.maximum(weights.sum(), 1e-12)
        raw = normalized * float(total_runs)
        base = np.floor(raw).astype(np.int64)
        remainder = total_runs - int(base.sum())
        if remainder > 0:
            ordering = np.argsort(-(raw - base))
            for index in ordering[:remainder]:
                base[index] += 1
        return (int(base[0]), int(base[1]), int(base[2]))

    def _run_expert_pass(
        self,
        initial_state: InitialState,
        allocations: tuple[int, int, int],
        base_seed: int,
        pass_offset: int,
        height: int,
        width: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        tensors: list[NDArray[np.float64]] = []
        for expert_idx, n_runs in enumerate(allocations):
            if n_runs <= 0:
                tensors.append(np.zeros((height, width, NUM_CLASSES), dtype=np.float64))
                continue
            runs = self._expert_runners[expert_idx].run_batch(
                initial_state=initial_state,
                n_runs=n_runs,
                base_seed=base_seed + expert_idx * _EXPERT_SEED_STRIDE + pass_offset,
            )
            tensors.append(aggregate_runs(runs, height, width))
        return (tensors[0], tensors[1], tensors[2])

    def _dynamic_cell_mask(
        self,
        grid: list[list[int]],
        distance_map: NDArray[np.float64],
        frontier_mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        height = len(grid)
        width = len(grid[0])
        dynamic = np.zeros((height, width), dtype=np.bool_)
        for y in range(height):
            for x in range(width):
                code = grid[y][x]
                dynamic[y, x] = code not in (TerrainCode.OCEAN, TerrainCode.MOUNTAIN) and (
                    code in (TerrainCode.SETTLEMENT, TerrainCode.PORT, TerrainCode.RUIN)
                    or distance_map[y, x] <= _NEAR_SETTLEMENT_RADIUS
                    or frontier_mask[y, x]
                )
        return dynamic

    def _exploit_weights(
        self,
        scout_tensors: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        scout_allocations: tuple[int, int, int],
        dynamic_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        del scout_allocations
        if not np.any(dynamic_mask):
            return _EXPERT_WEIGHTS.copy()

        default_tensor = scout_tensors[0]
        expert_diffs = np.zeros(3, dtype=np.float64)
        for expert_idx in (1, 2):
            diff = np.abs(scout_tensors[expert_idx] - default_tensor)
            selected = diff[dynamic_mask]
            expert_diffs[expert_idx] = float(np.mean(selected)) if selected.size > 0 else 0.0

        selected_expert = 1 if expert_diffs[1] >= expert_diffs[2] else 2
        selected_diff = expert_diffs[selected_expert]
        if selected_diff <= 0.0:
            return _EXPERT_WEIGHTS.copy()

        reallocation = min(_MAX_DEFAULT_REALLOCATION, 1.5 * selected_diff)
        weights = _EXPERT_WEIGHTS.copy()
        weights[0] -= reallocation
        weights[selected_expert] += reallocation
        return weights / np.maximum(weights.sum(), 1e-12)

    def _merge_pass_tensors(
        self,
        scout_tensors: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        scout_allocations: tuple[int, int, int],
        exploit_tensors: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        exploit_allocations: tuple[int, int, int],
        height: int,
        width: int,
    ) -> tuple[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], tuple[int, int, int]
    ]:
        merged: list[NDArray[np.float64]] = []
        totals: list[int] = []
        for expert_idx in range(3):
            total_runs = scout_allocations[expert_idx] + exploit_allocations[expert_idx]
            totals.append(total_runs)
            if total_runs <= 0:
                merged.append(np.zeros((height, width, NUM_CLASSES), dtype=np.float64))
                continue
            combined = (
                scout_tensors[expert_idx] * float(scout_allocations[expert_idx])
                + exploit_tensors[expert_idx] * float(exploit_allocations[expert_idx])
            ) / float(total_runs)
            merged.append(combined)
        return (merged[0], merged[1], merged[2]), (totals[0], totals[1], totals[2])

    def _combine_expert_tensors(
        self,
        expert_tensors: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        total_runs: tuple[int, int, int],
    ) -> NDArray[np.float64]:
        total = max(sum(total_runs), 1)
        result = np.zeros_like(expert_tensors[0])
        for expert_idx in range(3):
            result += expert_tensors[expert_idx] * (float(total_runs[expert_idx]) / float(total))
        return result

    def _cellwise_disagreement(
        self,
        expert_tensors: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        pairwise = np.stack(
            [
                np.abs(expert_tensors[0] - expert_tensors[1]),
                np.abs(expert_tensors[0] - expert_tensors[2]),
                np.abs(expert_tensors[1] - expert_tensors[2]),
            ],
            axis=0,
        )
        return np.mean(pairwise, axis=(0, 3))

    def _refine_ensemble_tensor(
        self,
        ensemble_mc: NDArray[np.float64],
        prior: NDArray[np.float64],
        initial_state: InitialState,
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        entropy = self._compute_entropy(ensemble_mc)
        if (
            self._learned_entropy_profile is not None
            and self._learned_entropy_profile.shape == (height, width)
            and self._seed_call_count > 0
        ):
            entropy = 0.7 * entropy + 0.3 * self._learned_entropy_profile
        self._learned_entropy_profile = entropy.copy()
        blend_weights = self._entropy_to_blend_weights(entropy, initial_state, height, width)
        return (
            blend_weights[:, :, np.newaxis] * ensemble_mc
            + (1.0 - blend_weights[:, :, np.newaxis]) * prior
        )

    def _compute_entropy(self, tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        probabilities = np.clip(tensor, 1e-10, 1.0)
        probabilities = probabilities / np.sum(probabilities, axis=2, keepdims=True)
        return -np.sum(probabilities * np.log(probabilities), axis=2)

    def _entropy_to_blend_weights(
        self,
        entropy: NDArray[np.float64],
        initial_state: InitialState,
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        max_entropy = float(np.log(NUM_CLASSES))
        normalized = np.clip(entropy / max_entropy, 0.0, 1.0)
        weights = np.clip(
            (normalized - _LOW_ENTROPY_THRESHOLD)
            / max(_HIGH_ENTROPY_THRESHOLD - _LOW_ENTROPY_THRESHOLD, 1e-6),
            0.0,
            1.0,
        )
        weights = 0.1 + 0.65 * weights

        for y in range(height):
            for x in range(width):
                code = initial_state.grid[y][x]
                if code == TerrainCode.OCEAN or code == TerrainCode.MOUNTAIN:
                    weights[y, x] = 0.0
        return weights

    def _alpha_map(
        self,
        initial_state: InitialState,
        distance_map: NDArray[np.float64],
        disagreement: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        height = len(initial_state.grid)
        width = len(initial_state.grid[0])
        alpha = np.zeros((height, width), dtype=np.float64)
        for y in range(height):
            for x in range(width):
                code = initial_state.grid[y][x]
                if code == TerrainCode.OCEAN or code == TerrainCode.MOUNTAIN:
                    base_alpha = 1000.0
                elif code in (TerrainCode.SETTLEMENT, TerrainCode.PORT, TerrainCode.RUIN):
                    base_alpha = 4.0
                elif distance_map[y, x] <= _NEAR_SETTLEMENT_RADIUS:
                    base_alpha = 7.0
                elif code == TerrainCode.FOREST:
                    base_alpha = 15.0
                else:
                    base_alpha = 16.0
                alpha[y, x] = base_alpha * (1.0 + 1.25 * disagreement[y, x])
        return alpha

    def _apply_cross_seed_transfer(
        self,
        result: NDArray[np.float64],
        ensemble_mc: NDArray[np.float64],
        prior: NDArray[np.float64],
        initial_state: InitialState,
        coastal_cells: set[tuple[int, int]],
        distance_map: NDArray[np.float64],
        frontier_mask: NDArray[np.bool_],
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        transferred = result.copy()
        for y in range(height):
            for x in range(width):
                bucket = self._bucket_key(
                    code=initial_state.grid[y][x],
                    is_coastal=(x, y) in coastal_cells,
                    distance=float(distance_map[y, x]),
                    is_frontier=bool(frontier_mask[y, x]),
                )
                support = self._bucket_support.get(bucket, 0)
                if support < _TRANSFER_MIN_SUPPORT:
                    continue
                mean_residual = self._bucket_residual_sums[bucket] / float(support)
                current_residual = ensemble_mc[y, x, :] - prior[y, x, :]
                alignment = float(np.dot(mean_residual, current_residual))
                if alignment <= 0.0:
                    continue
                transfer_weight = min(_TRANSFER_MATCH_WEIGHT, _TRANSFER_WEIGHT, _TRANSFER_CAP)
                transferred[y, x, :] = transferred[y, x, :] + transfer_weight * mean_residual
        return transferred

    def _store_bucket_residuals(
        self,
        initial_state: InitialState,
        ensemble_mc: NDArray[np.float64],
        prior: NDArray[np.float64],
        coastal_cells: set[tuple[int, int]],
        distance_map: NDArray[np.float64],
        frontier_mask: NDArray[np.bool_],
        height: int,
        width: int,
    ) -> None:
        residual = ensemble_mc - prior
        for y in range(height):
            for x in range(width):
                bucket = self._bucket_key(
                    code=initial_state.grid[y][x],
                    is_coastal=(x, y) in coastal_cells,
                    distance=float(distance_map[y, x]),
                    is_frontier=bool(frontier_mask[y, x]),
                )
                if bucket not in self._bucket_residual_sums:
                    self._bucket_residual_sums[bucket] = np.zeros(NUM_CLASSES, dtype=np.float64)
                    self._bucket_support[bucket] = 0
                self._bucket_residual_sums[bucket] = (
                    self._bucket_residual_sums[bucket] + residual[y, x, :]
                )
                self._bucket_support[bucket] += 1

    def _bucket_key(
        self,
        code: int,
        is_coastal: bool,
        distance: float,
        is_frontier: bool,
    ) -> BucketKey:
        distance_bucket = min(int(distance), 10) // 3
        return (int(code), is_coastal, distance_bucket, is_frontier)

    def _build_prior(
        self,
        initial_state: InitialState,
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        grid = initial_state.grid
        tensor = np.full((height, width, NUM_CLASSES), 0.01, dtype=np.float64)

        for y in range(height):
            for x in range(width):
                code = grid[y][x]
                if code == TerrainCode.OCEAN:
                    tensor[y, x, ClassIndex.EMPTY] = 0.97
                elif code == TerrainCode.MOUNTAIN:
                    tensor[y, x, ClassIndex.MOUNTAIN] = 0.97
                elif code == TerrainCode.SETTLEMENT:
                    tensor[y, x, ClassIndex.EMPTY] = 0.440
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.319
                    tensor[y, x, ClassIndex.PORT] = 0.004
                    tensor[y, x, ClassIndex.RUIN] = 0.026
                    tensor[y, x, ClassIndex.FOREST] = 0.210
                elif code == TerrainCode.PORT:
                    tensor[y, x, ClassIndex.EMPTY] = 0.457
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.094
                    tensor[y, x, ClassIndex.PORT] = 0.174
                    tensor[y, x, ClassIndex.RUIN] = 0.028
                    tensor[y, x, ClassIndex.FOREST] = 0.247
                elif code == TerrainCode.RUIN:
                    tensor[y, x, ClassIndex.EMPTY] = 0.500
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.100
                    tensor[y, x, ClassIndex.PORT] = 0.005
                    tensor[y, x, ClassIndex.RUIN] = 0.150
                    tensor[y, x, ClassIndex.FOREST] = 0.245
                elif code == TerrainCode.FOREST:
                    tensor[y, x, ClassIndex.EMPTY] = 0.078
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.143
                    tensor[y, x, ClassIndex.PORT] = 0.009
                    tensor[y, x, ClassIndex.RUIN] = 0.014
                    tensor[y, x, ClassIndex.FOREST] = 0.757
                else:
                    # PLAINS or EMPTY
                    tensor[y, x, ClassIndex.EMPTY] = 0.806
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.136
                    tensor[y, x, ClassIndex.PORT] = 0.009
                    tensor[y, x, ClassIndex.RUIN] = 0.013
                    tensor[y, x, ClassIndex.FOREST] = 0.036

        sums = tensor.sum(axis=2, keepdims=True)
        return tensor / np.maximum(sums, 1e-10)

    def _apply_hard_mask(
        self,
        tensor: NDArray[np.float64],
        initial_state: InitialState,
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        constrained = np.clip(tensor.copy(), 0.0, None)
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

                row_sum = float(np.sum(constrained[y, x, :]))
                if row_sum <= 0.0:
                    template = (
                        _FILTER_COASTAL_TEMPLATE if (x, y) in coastal else _FILTER_INLAND_TEMPLATE
                    )
                    constrained[y, x, :] = template
                else:
                    constrained[y, x, :] = constrained[y, x, :] / row_sum
        return constrained

    def _settlement_distance_map(
        self,
        initial_state: InitialState,
        height: int,
        width: int,
    ) -> NDArray[np.float64]:
        alive_positions = [(s.x, s.y) for s in initial_state.settlements if s.alive]
        if not alive_positions:
            return np.full((height, width), float(height + width), dtype=np.float64)

        distance = np.full((height, width), float(height + width), dtype=np.float64)
        for sx, sy in alive_positions:
            for y in range(height):
                for x in range(width):
                    chebyshev = max(abs(x - sx), abs(y - sy))
                    if chebyshev < distance[y, x]:
                        distance[y, x] = float(chebyshev)
        return distance

    def _frontier_mask(
        self,
        grid: list[list[int]],
        height: int,
        width: int,
    ) -> NDArray[np.bool_]:
        frontier = np.zeros((height, width), dtype=np.bool_)
        for y in range(height):
            for x in range(width):
                code = grid[y][x]
                if code in (TerrainCode.OCEAN, TerrainCode.MOUNTAIN):
                    continue
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = y + dy
                    nx = x + dx
                    if not (0 <= ny < height and 0 <= nx < width):
                        continue
                    neighbor = grid[ny][nx]
                    if code == TerrainCode.FOREST and neighbor in (
                        TerrainCode.EMPTY,
                        TerrainCode.PLAINS,
                        TerrainCode.SETTLEMENT,
                        TerrainCode.PORT,
                        TerrainCode.RUIN,
                    ):
                        frontier[y, x] = True
                        break
                    if (
                        code
                        in (
                            TerrainCode.EMPTY,
                            TerrainCode.PLAINS,
                            TerrainCode.SETTLEMENT,
                            TerrainCode.PORT,
                            TerrainCode.RUIN,
                        )
                        and neighbor == TerrainCode.FOREST
                    ):
                        frontier[y, x] = True
                        break
        return frontier

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
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < height and 0 <= nx < width and grid[ny][nx] == TerrainCode.OCEAN:
                        coastal.add((x, y))
                        break
        return coastal
