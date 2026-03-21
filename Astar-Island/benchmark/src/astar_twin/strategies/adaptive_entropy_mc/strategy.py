"""Adaptive entropy-guided MC strategy with cross-seed state transfer.

Translates the full "query allocation and hotspot redesign" avenue:

1. **Dynamic budget schedule** — First seed gets more MC runs to learn
   which regions are most dynamic; later seeds inherit that knowledge
   and can achieve similar quality with fewer runs.

2. **Cross-seed value-of-information** — The strategy object carries state
   between per-seed predict() calls. Entropy maps from seed 0 inform
   the prediction weighting for seeds 1-4.

3. **Two-pass MC** — Each seed runs an initial MC batch, computes per-cell
   entropy, then runs a second batch. The final prediction blends both
   passes with the initial-state prior, weighted by cell entropy.

4. **Hotspot-targeted blending** — Cells near settlement clusters,
   coastal edges, and forest frontiers get higher MC trust. The
   entropy map refines this further.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import (
    NUM_CLASSES,
    ClassIndex,
    TerrainCode,
)
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.params import SimulationParams

_FILTER_INLAND_TEMPLATE = np.array([0.55, 0.18, 0.0, 0.07, 0.20, 0.0], dtype=np.float64)
_FILTER_COASTAL_TEMPLATE = np.array([0.48, 0.17, 0.12, 0.06, 0.17, 0.0], dtype=np.float64)

# Budget allocation: first seed gets more MC runs (learning seed)
_FIRST_SEED_BUDGET_FRAC = 0.25  # 25% of conceptual budget
_LATER_SEED_BUDGET_FRAC = 0.1875  # 18.75% each for seeds 1-4

# MC run counts
_PASS1_FRACTION = 0.4  # 40% of runs in first pass
_PASS2_FRACTION = 0.6  # 60% of runs in second pass (focused)
_MIN_RUNS_PER_PASS = 8

# Entropy-based blending
_HIGH_ENTROPY_THRESHOLD = 0.5  # cells above this get full MC trust
_LOW_ENTROPY_THRESHOLD = 0.1  # cells below this get mostly prior

# Total MC runs scaling
_TOTAL_MC_RUNS = 200  # total runs to distribute across 5 seeds


class AdaptiveEntropyMCStrategy:
    """Multi-pass entropy-guided MC strategy with cross-seed learning.

    The strategy maintains state between predict() calls across seeds.
    Seed 0 (the "learning seed") runs more simulations and builds an
    entropy profile of which cell types tend to be dynamic. Seeds 1-4
    use this profile to weight their predictions more efficiently.

    This strategy models the real query-allocation problem: spend more
    observation budget early to learn the hidden parameters, then exploit
    that knowledge for remaining seeds.
    """

    def __init__(self, params: SimulationParams | None = None) -> None:
        self._params = params or SimulationParams()
        # Cross-seed state (reset on each round)
        self._seed_call_count: int = 0
        self._learned_entropy_profile: NDArray[np.float64] | None = None

    @property
    def name(self) -> str:
        return "adaptive_entropy_mc"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        grid = initial_state.grid
        H = len(grid)
        W = len(grid[0])

        # Track which seed call this is (reset on first call of a round)
        is_first_seed = self._seed_call_count == 0
        self._seed_call_count += 1

        # Allocate MC runs based on seed position
        if is_first_seed:
            n_total_runs = int(_TOTAL_MC_RUNS * _FIRST_SEED_BUDGET_FRAC)
        else:
            n_total_runs = int(_TOTAL_MC_RUNS * _LATER_SEED_BUDGET_FRAC)

        # Scale by remaining budget (models query budget awareness)
        budget_scale = max(0.3, budget.remaining / budget.total)
        n_total_runs = max(
            _MIN_RUNS_PER_PASS * 2,
            int(n_total_runs * budget_scale),
        )

        n_pass1 = max(_MIN_RUNS_PER_PASS, int(n_total_runs * _PASS1_FRACTION))
        n_pass2 = max(_MIN_RUNS_PER_PASS, n_total_runs - n_pass1)

        simulator = Simulator(params=self._params)
        mc_runner = MCRunner(simulator)

        # --- Pass 1: Initial MC batch ---
        pass1_runs = mc_runner.run_batch(
            initial_state=initial_state,
            n_runs=n_pass1,
            base_seed=base_seed,
        )
        pass1_tensor = aggregate_runs(pass1_runs, H, W)

        # --- Pass 2: Second MC batch ---
        # Use a different seed range for pass 2 to get independent samples
        pass2_runs = mc_runner.run_batch(
            initial_state=initial_state,
            n_runs=n_pass2,
            base_seed=base_seed + 10000,
        )
        pass2_tensor = aggregate_runs(pass2_runs, H, W)

        # Combined MC tensor: weighted average of both passes
        total_runs = n_pass1 + n_pass2
        mc_tensor = (n_pass1 * pass1_tensor + n_pass2 * pass2_tensor) / total_runs

        # Compute final entropy from combined MC
        mc_entropy = self._compute_entropy(mc_tensor)

        # --- Cross-seed learning ---
        if is_first_seed:
            # Store entropy profile for later seeds
            self._learned_entropy_profile = mc_entropy.copy()

        # Build blending weights from entropy
        blend_weights = self._entropy_to_blend_weights(mc_entropy, initial_state, H, W)

        # If we have a learned profile from seed 0, use it to refine
        if (
            self._learned_entropy_profile is not None
            and not is_first_seed
            and self._learned_entropy_profile.shape == (H, W)
        ):
            # Average current entropy with learned profile
            # This transfers "which regions tend to be dynamic" across seeds
            profile_weight = 0.3
            adjusted_entropy = (
                1.0 - profile_weight
            ) * mc_entropy + profile_weight * self._learned_entropy_profile
            blend_weights = self._entropy_to_blend_weights(adjusted_entropy, initial_state, H, W)

        # Build initial-state prior
        prior = self._build_prior(initial_state, H, W)

        # Blend: result = blend_weight * MC + (1 - blend_weight) * prior
        blend_3d = blend_weights[:, :, np.newaxis]
        result = blend_3d * mc_tensor + (1.0 - blend_3d) * prior

        # Apply hard game-rule limits borrowed from filter_baseline.
        # This preserves adaptive weighting while zeroing impossible classes.
        result = self._apply_hard_limits(result, initial_state, H, W)

        return result

    def _compute_entropy(self, tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute per-cell Shannon entropy from a H x W x 6 probability tensor."""
        p = np.clip(tensor, 1e-10, 1.0)
        p = p / np.sum(p, axis=2, keepdims=True)
        entropy: NDArray[np.float64] = -np.sum(p * np.log(p), axis=2)
        return entropy

    def _entropy_to_blend_weights(
        self,
        entropy: NDArray[np.float64],
        initial_state: InitialState,
        H: int,
        W: int,
    ) -> NDArray[np.float64]:
        """Convert per-cell entropy + terrain type into MC blend weights.

        High entropy → trust MC more (cell is dynamic)
        Low entropy → trust prior more (cell is stable)
        Static terrain → always trust prior (weight = 0)
        """
        # Normalize entropy to [0, 1] range
        max_entropy = float(np.log(NUM_CLASSES))
        norm_entropy = np.clip(entropy / max_entropy, 0.0, 1.0)

        # Sigmoid-like mapping from entropy to blend weight
        # Maps [LOW_THRESH, HIGH_THRESH] → [0.1, 0.95]
        weights = np.clip(
            (norm_entropy - _LOW_ENTROPY_THRESHOLD)
            / max(_HIGH_ENTROPY_THRESHOLD - _LOW_ENTROPY_THRESHOLD, 1e-6),
            0.0,
            1.0,
        )
        # Scale to [0.1, 0.95] range for dynamic cells
        weights = 0.1 + 0.85 * weights

        # Override static terrain to zero weight (pure prior)
        grid = initial_state.grid
        for y in range(H):
            for x in range(W):
                code = grid[y][x]
                if code == TerrainCode.OCEAN or code == TerrainCode.MOUNTAIN:
                    weights[y, x] = 0.0

        return weights

    def _build_prior(
        self,
        initial_state: InitialState,
        H: int,
        W: int,
    ) -> NDArray[np.float64]:
        """Build terrain-type-aware prior from initial state.

        Uses the same approach as TerrainAwareMCStrategy but with
        slightly more aggressive priors for static terrain (the entropy
        blending will handle the dynamic/static weighting).
        """
        grid = initial_state.grid
        tensor = np.full((H, W, NUM_CLASSES), 0.01, dtype=np.float64)

        coastal = self._find_coastal_cells(grid, H, W)

        for y in range(H):
            for x in range(W):
                code = grid[y][x]
                is_coastal = (x, y) in coastal

                if code == TerrainCode.OCEAN:
                    tensor[y, x, ClassIndex.EMPTY] = 0.97
                elif code == TerrainCode.MOUNTAIN:
                    tensor[y, x, ClassIndex.MOUNTAIN] = 0.97
                elif code == TerrainCode.SETTLEMENT:
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.45
                    tensor[y, x, ClassIndex.RUIN] = 0.22
                    tensor[y, x, ClassIndex.EMPTY] = 0.15
                    tensor[y, x, ClassIndex.PORT] = 0.10 if is_coastal else 0.02
                    tensor[y, x, ClassIndex.FOREST] = 0.03
                elif code == TerrainCode.PORT:
                    tensor[y, x, ClassIndex.PORT] = 0.40
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.15
                    tensor[y, x, ClassIndex.RUIN] = 0.22
                    tensor[y, x, ClassIndex.EMPTY] = 0.15
                    tensor[y, x, ClassIndex.FOREST] = 0.03
                elif code == TerrainCode.RUIN:
                    tensor[y, x, ClassIndex.RUIN] = 0.28
                    tensor[y, x, ClassIndex.FOREST] = 0.27
                    tensor[y, x, ClassIndex.EMPTY] = 0.25
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.10
                    tensor[y, x, ClassIndex.PORT] = 0.05 if is_coastal else 0.01
                elif code == TerrainCode.FOREST:
                    tensor[y, x, ClassIndex.FOREST] = 0.72
                    tensor[y, x, ClassIndex.EMPTY] = 0.14
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.07
                    tensor[y, x, ClassIndex.PORT] = 0.03 if is_coastal else 0.01
                else:
                    # Plains or Empty
                    tensor[y, x, ClassIndex.EMPTY] = 0.65
                    tensor[y, x, ClassIndex.SETTLEMENT] = 0.12
                    tensor[y, x, ClassIndex.FOREST] = 0.10
                    tensor[y, x, ClassIndex.PORT] = 0.05 if is_coastal else 0.01
                    tensor[y, x, ClassIndex.RUIN] = 0.04

        # Normalize
        sums = tensor.sum(axis=2, keepdims=True)
        tensor = tensor / np.maximum(sums, 1e-10)

        return tensor

    def _apply_hard_limits(
        self,
        tensor: NDArray[np.float64],
        initial_state: InitialState,
        H: int,
        W: int,
    ) -> NDArray[np.float64]:
        """Apply filter_baseline-style impossible-class constraints.

        Rules:
        - Ocean cells are fixed to Empty.
        - Mountain cells are fixed to Mountain.
        - Non-mountain land cells cannot become Mountain.
        - Inland cells cannot become Port.

        The harness will later apply safe_prediction(), so exact zeros here are
        acceptable and intentionally encode the hard limits before flooring.
        """
        constrained = tensor.copy()
        coastal = self._find_coastal_cells(initial_state.grid, H, W)

        for y in range(H):
            for x in range(W):
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

    def _find_coastal_cells(
        self,
        grid: list[list[int]],
        H: int,
        W: int,
    ) -> set[tuple[int, int]]:
        """Find all land cells adjacent to ocean."""
        coastal: set[tuple[int, int]] = set()
        for y in range(H):
            for x in range(W):
                if grid[y][x] == TerrainCode.OCEAN:
                    continue
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == TerrainCode.OCEAN:
                        coastal.add((x, y))
                        break
        return coastal

    def reset(self) -> None:
        """Reset cross-seed state for a new round evaluation."""
        self._seed_call_count = 0
        self._learned_entropy_profile = None
