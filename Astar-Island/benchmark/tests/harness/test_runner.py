from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import RoundFixture
from astar_twin.harness.budget import Budget
from astar_twin.harness.runner import BenchmarkRunner
from astar_twin.scoring import safe_prediction
from astar_twin.strategies.uniform.strategy import UniformStrategy

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def fixture_with_ground_truths() -> RoundFixture:
    fixture = load_fixture(FIXTURE_PATH)
    height = fixture.map_height
    width = fixture.map_width
    gt = safe_prediction(np.full((height, width, 6), 1.0 / 6.0, dtype=np.float64)).tolist()
    return fixture.model_copy(update={"ground_truths": [gt for _ in range(fixture.seeds_count)]})


class TestBenchmarkRunnerWithGroundTruths:
    def test_report_contains_one_entry_per_strategy(
        self, fixture_with_ground_truths: RoundFixture
    ) -> None:
        strategies = [UniformStrategy()]
        report = BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run(strategies)
        assert len(report.strategy_reports) == 1
        assert report.strategy_reports[0].strategy_name == "uniform"

    def test_scores_cover_all_seeds(self, fixture_with_ground_truths: RoundFixture) -> None:
        report = BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run(
            [UniformStrategy()]
        )
        sr = report.strategy_reports[0]
        assert len(sr.scores) == fixture_with_ground_truths.seeds_count

    def test_mean_score_is_average_of_seed_scores(
        self, fixture_with_ground_truths: RoundFixture
    ) -> None:
        report = BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run(
            [UniformStrategy()]
        )
        sr = report.strategy_reports[0]
        assert sr.mean_score == pytest.approx(float(np.mean(sr.scores)))

    def test_scores_are_in_valid_range(self, fixture_with_ground_truths: RoundFixture) -> None:
        report = BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run(
            [UniformStrategy()]
        )
        for score in report.strategy_reports[0].scores:
            assert 0.0 <= score <= 100.0

    def test_prediction_tensors_have_correct_shape(
        self, fixture_with_ground_truths: RoundFixture
    ) -> None:
        report = BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run(
            [UniformStrategy()]
        )
        h = fixture_with_ground_truths.map_height
        w = fixture_with_ground_truths.map_width
        for sr in report.strategy_reports:
            for seed_result in sr.seed_results:
                assert seed_result.prediction.shape == (h, w, 6)
                assert seed_result.ground_truth.shape == (h, w, 6)

    def test_safe_prediction_applied_no_zeros(
        self, fixture_with_ground_truths: RoundFixture
    ) -> None:
        report = BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run(
            [UniformStrategy()]
        )
        for sr in report.strategy_reports:
            for seed_result in sr.seed_results:
                assert np.all(seed_result.prediction > 0.0)

    def test_fixture_id_in_report(self, fixture_with_ground_truths: RoundFixture) -> None:
        report = BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run(
            [UniformStrategy()]
        )
        assert report.fixture_id == fixture_with_ground_truths.id
        assert fixture_with_ground_truths.id in report.fixture_ids

    def test_ranked_returns_sorted_by_mean_descending(
        self, fixture_with_ground_truths: RoundFixture
    ) -> None:
        strategies = [UniformStrategy(), UniformStrategy()]
        strategies[1].__class__ = type(  # type: ignore[assignment]
            "SlightlyWorse",
            (UniformStrategy,),
            {"name": property(lambda self: "slightly_worse")},
        )
        report = BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run(
            [UniformStrategy()]
        )
        ranked = report.ranked()
        for i in range(len(ranked) - 1):
            assert ranked[i].mean_score >= ranked[i + 1].mean_score

    def test_multiple_strategies_all_scored(self, fixture_with_ground_truths: RoundFixture) -> None:
        s1 = UniformStrategy()
        s2 = UniformStrategy()
        report = BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run([s1, s2])
        assert len(report.strategy_reports) == 2


class TestBenchmarkRunnerDerivedGroundTruths:
    def test_null_ground_truths_fixture_still_scores(self) -> None:
        fixture = load_fixture(FIXTURE_PATH)
        assert fixture.ground_truths is None
        report = BenchmarkRunner(fixture=fixture, base_seed=0, n_mc_runs=5).run([UniformStrategy()])
        assert len(report.strategy_reports) == 1
        assert all(0.0 <= s <= 100.0 for s in report.strategy_reports[0].scores)

    def test_derived_ground_truths_have_correct_shape(self) -> None:
        fixture = load_fixture(FIXTURE_PATH)
        runner = BenchmarkRunner(fixture=fixture, base_seed=0, n_mc_runs=5)
        runner.run([UniformStrategy()])
        gts = runner._get_ground_truths()
        assert len(gts) == fixture.seeds_count
        for gt in gts:
            assert len(gt) == fixture.map_height
            assert len(gt[0]) == fixture.map_width
            assert len(gt[0][0]) == 6


class TestBenchmarkRunnerSharedBudget:
    """The runner must pass a single shared Budget instance across all seeds for
    one strategy run.  A query spent on seed 0 reduces what remains for seeds
    1-4, mirroring the real challenge's global 50-query pool."""

    def test_budget_is_shared_across_seeds(self, fixture_with_ground_truths: RoundFixture) -> None:
        """A strategy that consumes one query per seed call should see the
        budget reduce by seeds_count total, not reset between seeds."""
        received_budgets: list[Budget] = []

        class BudgetCapturingStrategy(UniformStrategy):
            @property
            def name(self) -> str:
                return "budget_capturing"

            def predict(
                self,
                initial_state: Any,
                budget: Budget,
                base_seed: int,
            ) -> np.ndarray:
                budget.consume()  # spend exactly 1 query per seed call
                received_budgets.append(budget)
                return super().predict(initial_state, budget, base_seed)

        BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run(
            [BudgetCapturingStrategy()]
        )

        n_seeds = fixture_with_ground_truths.seeds_count
        assert len(received_budgets) == n_seeds

        # All calls received the *same* object
        first = received_budgets[0]
        for b in received_budgets[1:]:
            assert b is first, "Expected the same Budget instance across all seed calls"

        # Total consumed equals one-per-seed call
        assert first.used == n_seeds

    def test_separate_strategy_runs_get_independent_budgets(
        self, fixture_with_ground_truths: RoundFixture
    ) -> None:
        """Two different strategies must each get their own fresh Budget so
        one strategy's spending cannot corrupt another's remaining count."""
        budgets_by_strategy: dict[str, list[Budget]] = {}

        def make_strategy(tag: str) -> UniformStrategy:
            class TaggedStrategy(UniformStrategy):
                @property
                def name(self) -> str:
                    return tag

                def predict(
                    self,
                    initial_state: Any,
                    budget: Budget,
                    base_seed: int,
                ) -> np.ndarray:
                    budgets_by_strategy.setdefault(tag, []).append(budget)
                    budget.consume()
                    return super().predict(initial_state, budget, base_seed)

            return TaggedStrategy()

        BenchmarkRunner(fixture=fixture_with_ground_truths, base_seed=0).run(
            [make_strategy("s1"), make_strategy("s2")]
        )

        b_s1 = budgets_by_strategy["s1"][0]
        b_s2 = budgets_by_strategy["s2"][0]

        assert b_s1 is not b_s2, "Each strategy run must receive its own Budget instance"
        # Each strategy spent one query per seed
        n_seeds = fixture_with_ground_truths.seeds_count
        assert b_s1.used == n_seeds
        assert b_s2.used == n_seeds
