"""Benchmark adapter — drives the local digital twin for solver testing.

This adapter internally uses benchmark fixtures and simulator components but
NEVER exposes simulation_params to the solver. The solver sees only the
public contract objects (RoundDetail, SimulateResponse, etc.).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import (
    AnalysisResponse,
    RoundDetail,
    SimSettlement,
    SimulateResponse,
    SubmitResponse,
    ViewportBounds,
)
from astar_twin.contracts.types import MAX_QUERIES
from astar_twin.data.models import ParamsSource, RoundFixture
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.scoring import compute_score, safe_prediction


class BenchmarkAdapter:
    """Adapter that runs the local twin simulator behind the solver protocol.

    The adapter internally knows the fixture's simulation_params, but the
    solver code never sees them — only SimulateResponse objects are returned.
    """

    def __init__(
        self,
        fixture: RoundFixture,
        n_mc_runs: int = 5,
        sim_seed_offset: int = 0,
        require_calibrated_params: bool = False,
    ) -> None:
        if require_calibrated_params and fixture.params_source == ParamsSource.DEFAULT_PRIOR:
            raise ValueError(
                f"BenchmarkAdapter: fixture '{fixture.id}' has params_source=DEFAULT_PRIOR. "
                "The simulator will run with benchmark prior parameters, "
                "not real server-side params. "
                "Pass require_calibrated_params=False to suppress this error, or use a fixture "
                "with params_source=INFERRED or BENCHMARK_TRUTH."
            )
        if fixture.params_source == ParamsSource.DEFAULT_PRIOR:
            warnings.warn(
                f"BenchmarkAdapter: fixture '{fixture.id}' has params_source=DEFAULT_PRIOR. "
                "Simulator results reflect benchmark prior parameters, "
                "not real competition values. "
                "Use this adapter for ground-truth generation only, not calibration.",
                stacklevel=2,
            )
        self._fixture = fixture
        self._simulator = Simulator(params=fixture.simulation_params)
        self._mc_runner = MCRunner(self._simulator)
        self._n_mc_runs = n_mc_runs
        self._sim_seed_offset = sim_seed_offset
        self._queries_used = 0
        self._query_seed_counter: int = 1000 + sim_seed_offset
        self._submissions: dict[int, NDArray[np.float64]] = {}
        self._ground_truths: list[NDArray[np.float64]] | None = None
        if fixture.ground_truths is not None:
            self._ground_truths = [np.array(gt, dtype=np.float64) for gt in fixture.ground_truths]

    def get_round_detail(self, round_id: str) -> RoundDetail:
        return RoundDetail(
            id=self._fixture.id,
            round_number=self._fixture.round_number,
            status=self._fixture.status,
            map_width=self._fixture.map_width,
            map_height=self._fixture.map_height,
            seeds_count=self._fixture.seeds_count,
            initial_states=self._fixture.initial_states,
        )

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> SimulateResponse:
        if self._queries_used >= MAX_QUERIES:
            raise RuntimeError(f"Query budget exhausted: {self._queries_used}/{MAX_QUERIES}")

        self._queries_used += 1
        self._query_seed_counter += 1

        # Run one stochastic simulation
        initial_state = self._fixture.initial_states[seed_index]
        final_state = self._simulator.run(
            initial_state=initial_state,
            sim_seed=self._query_seed_counter,
        )

        # Extract viewport grid
        x0 = max(0, viewport_x)
        y0 = max(0, viewport_y)
        x1 = min(self._fixture.map_width, viewport_x + viewport_w)
        y1 = min(self._fixture.map_height, viewport_y + viewport_h)
        actual_w = max(0, x1 - x0)
        actual_h = max(0, y1 - y0)
        vp_grid = final_state.grid.viewport(viewport_x, viewport_y, viewport_w, viewport_h)
        grid_list = vp_grid.to_list()

        # Extract settlements visible in viewport
        settlements: list[SimSettlement] = []
        for s in final_state.settlements:
            if (
                viewport_x <= s.x < viewport_x + viewport_w
                and viewport_y <= s.y < viewport_y + viewport_h
            ):
                settlements.append(
                    SimSettlement(
                        x=s.x,
                        y=s.y,
                        population=s.population,
                        food=s.food,
                        wealth=s.wealth,
                        defense=s.defense,
                        has_port=s.has_port,
                        alive=s.alive,
                        owner_id=s.owner_id,
                    )
                )

        return SimulateResponse(
            grid=grid_list,
            settlements=settlements,
            viewport=ViewportBounds(
                x=x0,
                y=y0,
                w=actual_w,
                h=actual_h,
            ),
            width=self._fixture.map_width,
            height=self._fixture.map_height,
            queries_used=self._queries_used,
            queries_max=MAX_QUERIES,
        )

    def submit(
        self,
        round_id: str,
        seed_index: int,
        prediction: NDArray[np.float64],
    ) -> SubmitResponse:
        self._submissions[seed_index] = prediction
        return SubmitResponse(
            status="accepted",
            round_id=round_id,
            seed_index=seed_index,
        )

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResponse:
        if self._ground_truths is None:
            # Generate ground truth via MC if not pre-computed
            initial_state = self._fixture.initial_states[seed_index]
            runs = self._mc_runner.run_batch(initial_state, n_runs=200, base_seed=seed_index * 1000)
            gt = safe_prediction(
                aggregate_runs(runs, self._fixture.map_height, self._fixture.map_width)
            )
        else:
            gt = self._ground_truths[seed_index]

        submitted = self._submissions.get(seed_index)
        score = None
        pred_list = None
        if submitted is not None:
            score = compute_score(gt, safe_prediction(submitted))
            pred_list = submitted.tolist()

        return AnalysisResponse(
            prediction=pred_list,
            ground_truth=gt.tolist(),
            score=score,
            width=self._fixture.map_width,
            height=self._fixture.map_height,
            initial_grid=self._fixture.initial_states[seed_index].grid,
        )

    def get_budget(self, round_id: str) -> tuple[int, int]:
        return (self._queries_used, MAX_QUERIES)

    def reset_budget(self) -> None:
        """Reset query budget for repeated solver runs."""
        self._queries_used = 0
        self._query_seed_counter = 1000 + self._sim_seed_offset
        self._submissions.clear()
