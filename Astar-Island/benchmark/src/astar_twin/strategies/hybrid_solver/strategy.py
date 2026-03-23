"""Hybrid solver strategy — wraps the full solve() pipeline for benchmark evaluation.

This strategy bridges the round-level ``solve()`` pipeline (which processes all
5 seeds at once with a shared query budget) into the per-seed ``Strategy``
protocol expected by the benchmark harness.

On the first ``predict()`` call (seed 0), the strategy runs the full solver
pipeline and caches all 5 prediction tensors.  Subsequent calls return the
cached tensor for the requested seed index.

Because the solver needs a ``SolverAdapter`` (which requires a ``RoundFixture``),
this strategy takes the fixture as a constructor argument.  This means it must
be recreated for each fixture/round being benchmarked.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.data.models import RoundFixture
from astar_twin.harness.budget import Budget
from astar_twin.solver.adapters.benchmark import BenchmarkAdapter
from astar_twin.solver.pipeline import SolveResult, solve

# Conservative defaults for tractable runtime on 40x40 maps.
# Full defaults (24 particles, 6 inner, 64 sims) take 20+ min per round.
_DEFAULT_N_PARTICLES = 8
_DEFAULT_N_INNER_RUNS = 2
_DEFAULT_SIMS_PER_SEED = 16


class HybridSolverStrategy:
    """Strategy wrapper around the full hybrid solver pipeline.

    The solver runs once per round (on the first ``predict()`` call) and
    caches all seed tensors.  This mirrors how the solver would operate in
    production: a single round-level solve producing all 5 predictions.

    Args:
        fixture: The round fixture to solve against.  Required because the
                 solver adapter needs simulation params for the digital twin.
        n_particles: Initial particle count (default 8 for speed).
        n_inner_runs: Inner MC runs per likelihood update (default 2).
        sims_per_seed: Final MC runs per seed for prediction (default 16).
    """

    def __init__(
        self,
        fixture: RoundFixture,
        *,
        n_particles: int = _DEFAULT_N_PARTICLES,
        n_inner_runs: int = _DEFAULT_N_INNER_RUNS,
        sims_per_seed: int = _DEFAULT_SIMS_PER_SEED,
    ) -> None:
        self._fixture = fixture
        self._n_particles = n_particles
        self._n_inner_runs = n_inner_runs
        self._sims_per_seed = sims_per_seed

        # Cached after first predict() call
        self._solve_result: SolveResult | None = None
        self._seed_count: int = 0

    @property
    def name(self) -> str:
        return "hybrid_solver"

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        """Return a (H, W, 6) tensor for one seed.

        On the first call, runs the full solver pipeline and caches all
        seed tensors.  Subsequent calls return cached results.
        The ``budget`` is consumed to reflect the queries the solver used.
        """
        seed_idx = self._seed_count
        self._seed_count += 1

        if self._solve_result is None:
            # Run the full solver pipeline once for all seeds
            adapter = BenchmarkAdapter(
                self._fixture,
                n_mc_runs=self._sims_per_seed,
                sim_seed_offset=base_seed,
            )
            self._solve_result = solve(
                adapter,
                self._fixture.id,
                n_particles=self._n_particles,
                n_inner_runs=self._n_inner_runs,
                sims_per_seed=self._sims_per_seed,
                base_seed=base_seed,
            )

            # Consume queries from the harness budget to match what the solver used.
            # The solver tracks its own budget internally via the adapter; we reflect
            # that usage in the harness-level budget so other strategies (if any)
            # sharing the same Budget object see accurate remaining counts.
            queries_used = self._solve_result.total_queries_used
            consumable = min(queries_used, budget.remaining)
            if consumable > 0:
                budget.consume(consumable)

        if seed_idx < len(self._solve_result.tensors):
            return self._solve_result.tensors[seed_idx]

        # Fallback: if seed_idx exceeds cached tensors, return uniform
        h = len(initial_state.grid)
        w = len(initial_state.grid[0])
        uniform: NDArray[np.float64] = np.full((h, w, 6), 1.0 / 6.0, dtype=np.float64)
        return uniform
