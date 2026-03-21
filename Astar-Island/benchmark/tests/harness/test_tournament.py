"""Tests for the tournament runner and supporting models."""

from __future__ import annotations

from pathlib import Path
from typing import override

import numpy as np
import pytest
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import GroundTruthSource, RoundFixture
from astar_twin.harness.budget import Budget
from astar_twin.harness.tournament import (
    FixtureStrategyResult,
    StrategyTournamentResult,
    TournamentReport,
    TournamentRunner,
    filter_fixtures,
)
from astar_twin.scoring import safe_prediction

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fixture_with_gts(
    fixture_id: str = "test-round-001",
    round_number: int = 1,
    ground_truth_source: GroundTruthSource = GroundTruthSource.API_ANALYSIS,
    round_weight: float = 1.0,
) -> RoundFixture:
    """Load the test fixture and attach uniform ground truths."""
    raw = load_fixture(FIXTURE_PATH)
    h, w = raw.map_height, raw.map_width
    gt = safe_prediction(np.full((h, w, 6), 1.0 / 6.0, dtype=np.float64)).tolist()
    return raw.model_copy(
        update={
            "id": fixture_id,
            "round_number": round_number,
            "ground_truths": [gt for _ in range(raw.seeds_count)],
            "ground_truth_source": ground_truth_source,
            "round_weight": round_weight,
        }
    )


class _ConstantStrategy:
    """Strategy that always returns uniform 1/6 — fast, deterministic."""

    def __init__(self, tag: str = "constant") -> None:
        self._tag = tag

    @property
    def name(self) -> str:
        return self._tag

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        h = len(initial_state.grid)
        w = len(initial_state.grid[0])
        return np.full((h, w, 6), 1.0 / 6.0, dtype=np.float64)


class _SlightlyWorseStrategy(_ConstantStrategy):
    """Intentionally worse: uniform but with first class slightly higher."""

    def __init__(self) -> None:
        super().__init__(tag="slightly_worse")

    @override
    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        h = len(initial_state.grid)
        w = len(initial_state.grid[0])
        pred = np.full((h, w, 6), 1.0 / 6.0, dtype=np.float64)
        # Skew first class up — after safe_prediction, this diverges from uniform GT
        pred[:, :, 0] += 0.3
        return pred


# ---------------------------------------------------------------------------
# filter_fixtures
# ---------------------------------------------------------------------------


class TestFilterFixtures:
    def test_excludes_test_rounds(self) -> None:
        test_f = _make_fixture_with_gts(fixture_id="test-round-001")
        real_f = _make_fixture_with_gts(fixture_id="real-round-abc", round_number=5)
        result = filter_fixtures([test_f, real_f])
        assert len(result) == 1
        assert result[0].id == "real-round-abc"

    def test_excludes_fixtures_without_ground_truths(self) -> None:
        raw = load_fixture(FIXTURE_PATH)
        no_gt = raw.model_copy(update={"id": "real-no-gt", "round_number": 2})
        assert no_gt.ground_truths is None
        result = filter_fixtures([no_gt])
        assert result == []

    def test_filters_by_provenance(self) -> None:
        api_f = _make_fixture_with_gts(
            fixture_id="api-round",
            round_number=1,
            ground_truth_source=GroundTruthSource.API_ANALYSIS,
        )
        local_f = _make_fixture_with_gts(
            fixture_id="local-round",
            round_number=2,
            ground_truth_source=GroundTruthSource.LOCAL_MC,
        )
        result = filter_fixtures([api_f, local_f], provenance=GroundTruthSource.API_ANALYSIS)
        assert len(result) == 1
        assert result[0].id == "api-round"

    def test_no_provenance_filter_returns_all_real(self) -> None:
        f1 = _make_fixture_with_gts(
            fixture_id="round-a",
            round_number=1,
            ground_truth_source=GroundTruthSource.API_ANALYSIS,
        )
        f2 = _make_fixture_with_gts(
            fixture_id="round-b",
            round_number=2,
            ground_truth_source=GroundTruthSource.LOCAL_MC,
        )
        result = filter_fixtures([f1, f2])
        assert len(result) == 2

    def test_sorted_by_round_number(self) -> None:
        f3 = _make_fixture_with_gts(fixture_id="round-c", round_number=3)
        f1 = _make_fixture_with_gts(fixture_id="round-a", round_number=1)
        f2 = _make_fixture_with_gts(fixture_id="round-b", round_number=2)
        result = filter_fixtures([f3, f1, f2])
        assert [f.round_number for f in result] == [1, 2, 3]

    def test_unknown_provenance_filter(self) -> None:
        f = _make_fixture_with_gts(
            fixture_id="unknown-round",
            round_number=1,
            ground_truth_source=GroundTruthSource.UNKNOWN,
        )
        result = filter_fixtures([f], provenance=GroundTruthSource.UNKNOWN)
        assert len(result) == 1

        result_api = filter_fixtures([f], provenance=GroundTruthSource.API_ANALYSIS)
        assert result_api == []


# ---------------------------------------------------------------------------
# TournamentRunner
# ---------------------------------------------------------------------------


class TestTournamentRunner:
    def test_single_strategy_single_fixture(self) -> None:
        fixture = _make_fixture_with_gts(fixture_id="round-001", round_number=1)
        runner = TournamentRunner(fixtures=[fixture], repeats=2, base_seed=42)
        report = runner.run([_ConstantStrategy()])

        assert report.fixtures_evaluated == 1
        assert report.repeats == 2
        assert len(report.strategy_results) == 1
        sr = report.strategy_results[0]
        assert sr.strategy_name == "constant"
        assert len(sr.fixture_results) == 1
        assert len(sr.repeat_weighted_means) == 2

    def test_multiple_strategies_ranked_correctly(self) -> None:
        """The constant strategy should rank higher than the skewed one
        because ground truths are uniform."""
        fixture = _make_fixture_with_gts(fixture_id="round-001", round_number=1)
        runner = TournamentRunner(fixtures=[fixture], repeats=1, base_seed=42)
        report = runner.run([_ConstantStrategy(), _SlightlyWorseStrategy()])

        ranked = report.ranked()
        assert len(ranked) == 2
        # Constant matches the uniform GT better
        assert ranked[0].strategy_name == "constant"
        assert ranked[1].strategy_name == "slightly_worse"
        assert ranked[0].weighted_mean > ranked[1].weighted_mean

    def test_repeats_produce_variance_stats(self) -> None:
        fixture = _make_fixture_with_gts(fixture_id="round-001", round_number=1)
        runner = TournamentRunner(fixtures=[fixture], repeats=3, base_seed=42)
        report = runner.run([_ConstantStrategy()])

        sr = report.strategy_results[0]
        fr = sr.fixture_results[0]
        assert len(fr.repeat_mean_scores) == 3
        # Constant strategy + constant GT → std should be 0 or very near 0
        assert fr.overall_std == pytest.approx(0.0, abs=1e-10)

    def test_empty_fixtures_raises(self) -> None:
        runner = TournamentRunner(fixtures=[], repeats=1)
        with pytest.raises(ValueError, match="No fixtures match filter"):
            runner.run([_ConstantStrategy()])

    def test_only_test_fixtures_raises(self) -> None:
        """Test-round fixtures are filtered out, so tournament should raise."""
        fixture = _make_fixture_with_gts(fixture_id="test-only", round_number=1)
        runner = TournamentRunner(fixtures=[fixture], repeats=1)
        with pytest.raises(ValueError, match="No fixtures match filter"):
            runner.run([_ConstantStrategy()])

    def test_provenance_filter_applied(self) -> None:
        api_f = _make_fixture_with_gts(
            fixture_id="api-round",
            round_number=1,
            ground_truth_source=GroundTruthSource.API_ANALYSIS,
        )
        local_f = _make_fixture_with_gts(
            fixture_id="local-round",
            round_number=2,
            ground_truth_source=GroundTruthSource.LOCAL_MC,
        )
        runner = TournamentRunner(
            fixtures=[api_f, local_f],
            repeats=1,
            provenance_filter=GroundTruthSource.API_ANALYSIS,
        )
        report = runner.run([_ConstantStrategy()])
        assert report.fixtures_evaluated == 1
        assert report.provenance_filter == "api_analysis"

    def test_multiple_fixtures_weighted_aggregation(self) -> None:
        """With two fixtures of different weights, weighted and unweighted means differ."""
        f1 = _make_fixture_with_gts(fixture_id="round-1", round_number=1, round_weight=1.0)
        f2 = _make_fixture_with_gts(fixture_id="round-2", round_number=2, round_weight=1.0)
        runner = TournamentRunner(fixtures=[f1, f2], repeats=1, base_seed=42)
        report = runner.run([_ConstantStrategy()])

        sr = report.strategy_results[0]
        assert len(sr.fixture_results) == 2
        # With equal weights and same GT, weighted ≈ unweighted
        assert sr.weighted_mean == pytest.approx(sr.unweighted_mean, abs=1e-6)

    def test_runtime_tracked(self) -> None:
        fixture = _make_fixture_with_gts(fixture_id="round-001", round_number=1)
        runner = TournamentRunner(fixtures=[fixture], repeats=1, base_seed=42)
        report = runner.run([_ConstantStrategy()])
        assert report.total_runtime_seconds >= 0.0


# ---------------------------------------------------------------------------
# TournamentReport
# ---------------------------------------------------------------------------


class TestTournamentReport:
    @pytest.fixture
    def report(self) -> TournamentReport:
        fixture = _make_fixture_with_gts(fixture_id="round-001", round_number=1)
        runner = TournamentRunner(fixtures=[fixture], repeats=2, base_seed=42)
        return runner.run([_ConstantStrategy(), _SlightlyWorseStrategy()])

    def test_ranked_returns_descending_order(self, report: TournamentReport) -> None:
        ranked = report.ranked()
        for i in range(len(ranked) - 1):
            assert ranked[i].weighted_mean >= ranked[i + 1].weighted_mean

    def test_is_top_rank_stable_with_clear_winner(self, report: TournamentReport) -> None:
        # Constant vs slightly_worse with uniform GT — constant always wins
        assert report.is_top_rank_stable() is True

    def test_is_top_rank_stable_single_strategy(self) -> None:
        fixture = _make_fixture_with_gts(fixture_id="round-001", round_number=1)
        runner = TournamentRunner(fixtures=[fixture], repeats=2, base_seed=42)
        report = runner.run([_ConstantStrategy()])
        assert report.is_top_rank_stable() is True

    def test_to_dict_serializable(self, report: TournamentReport) -> None:
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "fixtures_evaluated" in d
        assert "repeats" in d
        assert "ranking" in d
        assert "top_rank_stable" in d
        assert isinstance(d["ranking"], list)
        assert len(d["ranking"]) == 2

    def test_to_dict_contains_strategy_details(self, report: TournamentReport) -> None:
        d = report.to_dict()
        ranking = d["ranking"]
        assert isinstance(ranking, list)
        first = ranking[0]
        assert isinstance(first, dict)
        assert "strategy_name" in first
        assert "weighted_mean" in first
        assert "unweighted_mean" in first
        assert "weighted_std" in first
        assert "fixtures" in first

    def test_print_summary_does_not_crash(
        self, report: TournamentReport, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report.print_summary()
        captured = capsys.readouterr()
        assert "Tournament:" in captured.out
        assert "Ranking" in captured.out
        assert "constant" in captured.out
        assert "slightly_worse" in captured.out


# ---------------------------------------------------------------------------
# FixtureStrategyResult / StrategyTournamentResult models
# ---------------------------------------------------------------------------


class TestDataclassModels:
    def test_fixture_strategy_result_to_dict(self) -> None:
        fsr = FixtureStrategyResult(
            fixture_id="round-1",
            round_number=1,
            round_weight=1.0,
            ground_truth_source=GroundTruthSource.API_ANALYSIS,
            repeat_mean_scores=(80.0, 82.0, 81.0),
            overall_mean=81.0,
            overall_std=0.82,
        )
        d = fsr.to_dict()
        assert d["fixture_id"] == "round-1"
        assert d["round_number"] == 1
        assert d["ground_truth_source"] == "api_analysis"
        assert d["repeat_mean_scores"] == [80.0, 82.0, 81.0]

    def test_strategy_tournament_result_to_dict(self) -> None:
        fsr = FixtureStrategyResult(
            fixture_id="round-1",
            round_number=1,
            round_weight=1.0,
            ground_truth_source=GroundTruthSource.API_ANALYSIS,
            repeat_mean_scores=(80.0,),
            overall_mean=80.0,
            overall_std=0.0,
        )
        str_result = StrategyTournamentResult(
            strategy_name="test",
            fixture_results=(fsr,),
            weighted_mean=80.0,
            unweighted_mean=80.0,
            weighted_std=0.0,
            repeat_weighted_means=(80.0,),
        )
        d = str_result.to_dict()
        assert d["strategy_name"] == "test"
        assert d["weighted_mean"] == 80.0
        assert isinstance(d["fixtures"], list)
        assert len(d["fixtures"]) == 1
