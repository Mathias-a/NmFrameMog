from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, TERRAIN_TO_CLASS, ClassIndex, TerrainCode
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams
from astar_twin.solver.predict.finalize import finalize_tensor

_FILTER_INLAND_TEMPLATE = np.array([0.806, 0.136, 0.0, 0.013, 0.045, 0.0], dtype=np.float64)
_FILTER_COASTAL_TEMPLATE = np.array([0.457, 0.094, 0.174, 0.028, 0.247, 0.0], dtype=np.float64)

_TARGET_RUNS = 200
_HIGH_ENTROPY_THRESHOLD = 0.5
_LOW_ENTROPY_THRESHOLD = 0.1


class ActiveSensingEnsembleMCStrategy:
    def __init__(self, params: SimulationParams | None = None) -> None:
        self._base_params = params or SimulationParams()
        self._experts = self._build_experts(self._base_params)
        self._learned_entropy_profile: NDArray[np.float64] | None = None
        self._seed_call_count = 0

    @property
    def name(self) -> str:
        return "active_sensing_ensemble_mc"

    def _build_experts(self, base_params: SimulationParams) -> list[tuple[str, SimulationParams]]:
        return [
            ("default", base_params),
            (
                "growth",
                replace(
                    base_params,
                    prosperity_threshold_port=1.5,
                    prosperity_threshold_expand=1.2,
                    expansion_rate=0.25,
                ),
            ),
            (
                "collapse",
                replace(
                    base_params,
                    winter_food_loss_flat=0.25,
                    winter_food_loss_per_population=0.15,
                    collapse_threshold=0.85,
                    raid_base_prob=0.10,
                ),
            ),
            (
                "static",
                replace(
                    base_params,
                    expansion_rate=0.05,
                    collapse_threshold=1.5,
                    winter_food_loss_flat=0.05,
                ),
            ),
        ]

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        height = len(initial_state.grid)
        width = len(initial_state.grid[0])

        # 1. Identify best 15x15 viewport to observe (max settlements/plains)
        vp_x, vp_y = self._find_best_viewport(initial_state.grid, height, width, 15, 15)

        # 2. Spend queries if available (10 queries per seed)
        empirical_counts = np.zeros((15, 15, NUM_CLASSES), dtype=np.float64)
        queries_spent = 0
        target_queries = 10

        while budget.remaining > 0 and queries_spent < target_queries:
            try:
                resp = budget.observe(
                    seed_index=self._seed_call_count,
                    viewport_x=vp_x,
                    viewport_y=vp_y,
                    viewport_w=15,
                    viewport_h=15,
                )
                for y in range(resp.viewport.h):
                    for x in range(resp.viewport.w):
                        code = resp.grid[y][x]
                        cls_idx = TERRAIN_TO_CLASS.get(code, ClassIndex.EMPTY)
                        empirical_counts[y, x, cls_idx] += 1.0
                queries_spent += 1
            except (RuntimeError, NotImplementedError):
                break

        has_empirical = queries_spent > 0
        empirical_probs: NDArray[np.float64] = np.zeros_like(empirical_counts)
        if has_empirical:
            empirical_probs = empirical_counts / queries_spent

        # 3. Run experts and evaluate likelihood
        expert_tensors = []
        log_likelihoods = []

        for _name, params in self._experts:
            runner = MCRunner(Simulator(params=params))
            runs = runner.run_batch(
                initial_state, n_runs=50, base_seed=base_seed + self._seed_call_count * 1000
            )
            tensor = aggregate_runs(runs, height, width)
            expert_tensors.append(tensor)

            if has_empirical:
                vp_tensor = tensor[vp_y : vp_y + 15, vp_x : vp_x + 15, :]
                vp_tensor = np.clip(vp_tensor, 1e-3, 1.0)
                vp_tensor /= vp_tensor.sum(axis=-1, keepdims=True)

                # Cross-entropy: sum( P_emp * log(P_expert) )
                # Weighted by empirical probability
                valid_y = min(15, height - vp_y)
                valid_x = min(15, width - vp_x)

                ll = 0.0
                for y in range(valid_y):
                    for x in range(valid_x):
                        for c in range(NUM_CLASSES):
                            p_emp = empirical_probs[y, x, c]
                            if p_emp > 0:
                                ll += p_emp * np.log(vp_tensor[y, x, c])

                # Normalize log-likelihood by the number of valid cells
                num_cells = max(1, valid_y * valid_x)
                ll /= num_cells

                log_likelihoods.append(ll)
            else:
                log_likelihoods.append(0.0)

        # 4. Softmax weights
        if has_empirical:
            lls = np.array(log_likelihoods)
            scaled_lls = lls / 0.1
            max_ll = np.max(scaled_lls)
            exp_lls = np.exp(scaled_lls - max_ll)
            weights = exp_lls / np.sum(exp_lls)
        else:
            # Fallback to uniform if no budget
            weights = np.ones(len(self._experts)) / len(self._experts)

        # 5. Combine
        combined_mc = np.zeros((height, width, NUM_CLASSES), dtype=np.float64)
        for i, w in enumerate(weights):
            combined_mc += w * expert_tensors[i]

        # 6. Apply entropy blending with empirical prior
        prior = self._build_prior(initial_state, height, width)
        entropy = self._compute_entropy(combined_mc)

        if self._learned_entropy_profile is not None and self._seed_call_count > 0:
            entropy = 0.7 * entropy + 0.3 * self._learned_entropy_profile
        self._learned_entropy_profile = entropy.copy()

        blend_weights = self._entropy_to_blend_weights(entropy, initial_state, height, width)
        result = (
            blend_weights[:, :, np.newaxis] * combined_mc
            + (1.0 - blend_weights[:, :, np.newaxis]) * prior
        )

        masked = self._apply_hard_mask(result, initial_state, height, width)
        self._seed_call_count += 1
        return finalize_tensor(masked, height, width, initial_state)

    def _find_best_viewport(
        self, grid: list[list[int]], height: int, width: int, vw: int, vh: int
    ) -> tuple[int, int]:
        # Simple heuristic: find 15x15 area with most settlements + plains
        best_score = -1
        best_pos = (0, 0)

        for y in range(0, max(1, height - vh + 1), 5):
            for x in range(0, max(1, width - vw + 1), 5):
                score = 0
                for cy in range(y, min(height, y + vh)):
                    for cx in range(x, min(width, x + vw)):
                        code = grid[cy][cx]
                        if code == TerrainCode.SETTLEMENT or code == TerrainCode.PORT:
                            score += 5
                        elif code == TerrainCode.PLAINS:
                            score += 1
                if score > best_score:
                    best_score = score
                    best_pos = (x, y)
        return best_pos

    def _compute_entropy(self, tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.clip(tensor, 1e-10, 1.0)
        p = p / np.sum(p, axis=2, keepdims=True)
        return -np.sum(p * np.log(p), axis=2)

    def _entropy_to_blend_weights(
        self, entropy: NDArray[np.float64], initial_state: InitialState, height: int, width: int
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

    def _build_prior(
        self, initial_state: InitialState, height: int, width: int
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
                    tensor[y, x, ClassIndex.EMPTY] = 0.806
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.136
                    tensor[y, x, ClassIndex.PORT] = 0.009
                    tensor[y, x, ClassIndex.RUIN] = 0.013
                    tensor[y, x, ClassIndex.FOREST] = 0.036
        sums = tensor.sum(axis=2, keepdims=True)
        return tensor / np.maximum(sums, 1e-10)

    def _apply_hard_mask(
        self, tensor: NDArray[np.float64], initial_state: InitialState, height: int, width: int
    ) -> NDArray[np.float64]:
        constrained = np.clip(tensor.copy(), 0.0, None)
        coastal = set()
        for y in range(height):
            for x in range(width):
                if initial_state.grid[y][x] != TerrainCode.OCEAN:
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and initial_state.grid[ny][nx] == TerrainCode.OCEAN
                        ):
                            coastal.add((x, y))
                            break

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

    def reset(self) -> None:
        self._seed_call_count = 0
        self._learned_entropy_profile = None
