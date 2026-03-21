from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from astar_twin.data.models import RoundFixture
from astar_twin.engine import Simulator
from astar_twin.harness.models import BenchmarkReport, SeedResult, StrategyReport
from astar_twin.harness.protocol import Strategy
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.scoring import compute_score, safe_prediction


def _derive_ground_truths(
    fixture: RoundFixture,
    n_mc_runs: int,
    base_seed: int,
) -> list[list[list[list[float]]]]:
    """Run Monte Carlo simulations to produce ground-truth probability tensors.

    Returns ground_truths in the same nested-list format as RoundFixture.ground_truths
    (shape: [seeds_count][H][W][6]).
    """
    simulator = Simulator(params=fixture.simulation_params)
    mc_runner = MCRunner(simulator)
    result: list[list[list[list[float]]]] = []
    for seed_idx in range(fixture.seeds_count):
        initial_state = fixture.initial_states[seed_idx]
        runs = mc_runner.run_batch(
            initial_state=initial_state,
            n_runs=n_mc_runs,
            base_seed=base_seed + seed_idx * n_mc_runs,
        )
        tensor: NDArray[np.float64] = aggregate_runs(runs, fixture.map_height, fixture.map_width)
        result.append(tensor.tolist())
    return result


@dataclass
class BenchmarkRunner:
    """Evaluate one or more strategies against a single fixture.

    If ``fixture.ground_truths`` is ``None``, ground truths are computed
    on-the-fly via ``n_mc_runs`` Monte Carlo simulations with ``base_seed``.

    Example::

        from astar_twin.harness import BenchmarkRunner
        from astar_twin.strategies import REGISTRY

        strategies = [cls() for cls in REGISTRY.values()]
        report = BenchmarkRunner(fixture=my_fixture, base_seed=42).run(strategies)
        for sr in report.ranked():
            print(sr.strategy_name, sr.mean_score)
    """

    fixture: RoundFixture
    base_seed: int = 0
    n_mc_runs: int = 200
    _ground_truths: list[list[list[list[float]]]] | None = field(
        default=None, init=False, repr=False
    )

    def _get_ground_truths(self) -> list[list[list[list[float]]]]:
        if self._ground_truths is not None:
            return self._ground_truths
        if self.fixture.ground_truths is not None:
            self._ground_truths = self.fixture.ground_truths
        else:
            self._ground_truths = _derive_ground_truths(
                self.fixture, self.n_mc_runs, self.base_seed
            )
        return self._ground_truths

    def run(self, strategies: list[Strategy]) -> BenchmarkReport:
        ground_truths = self._get_ground_truths()
        strategy_reports: list[StrategyReport] = []

        for strategy in strategies:
            seed_results: list[SeedResult] = []
            for seed_idx in range(self.fixture.seeds_count):
                initial_state = self.fixture.initial_states[seed_idx]
                raw_prediction = strategy.predict(
                    initial_state=initial_state,
                    budget=50,
                    base_seed=self.base_seed,
                )
                prediction = safe_prediction(raw_prediction)
                gt_tensor = np.array(ground_truths[seed_idx], dtype=np.float64)
                score = compute_score(gt_tensor, prediction)
                seed_results.append(
                    SeedResult(
                        seed_index=seed_idx,
                        score=score,
                        ground_truth=gt_tensor,
                        prediction=prediction,
                    )
                )
            strategy_reports.append(
                StrategyReport.from_seed_results(
                    strategy_name=strategy.name,
                    fixture_id=self.fixture.id,
                    seed_results=seed_results,
                )
            )

        return BenchmarkReport(
            strategy_reports=tuple(strategy_reports),
            fixture_id=self.fixture.id,
            fixture_ids=(self.fixture.id,),
        )
