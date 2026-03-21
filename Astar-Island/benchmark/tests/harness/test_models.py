from __future__ import annotations

import numpy as np

from astar_twin.harness.models import BenchmarkReport, SeedResult, StrategyReport


def _make_seed_result(seed_index: int = 0, score: float = 75.0) -> SeedResult:
    gt = np.full((4, 4, 6), 1.0 / 6.0, dtype=np.float64)
    pred = np.full((4, 4, 6), 1.0 / 6.0, dtype=np.float64)
    return SeedResult(seed_index=seed_index, score=score, ground_truth=gt, prediction=pred)


class TestSeedResult:
    def test_fields_stored_correctly(self) -> None:
        sr = _make_seed_result(seed_index=2, score=42.5)
        assert sr.seed_index == 2
        assert sr.score == 42.5
        assert sr.ground_truth.shape == (4, 4, 6)
        assert sr.prediction.shape == (4, 4, 6)

    def test_frozen(self) -> None:
        sr = _make_seed_result()
        try:
            sr.score = 0.0  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception as exc:  # noqa: BLE001
            assert "frozen" in str(exc).lower() or "cannot" in str(exc).lower()


class TestStrategyReport:
    def test_from_seed_results_computes_mean(self) -> None:
        seeds = [_make_seed_result(i, float(10 * (i + 1))) for i in range(3)]
        report = StrategyReport.from_seed_results("test_strategy", "round-001", seeds)
        assert report.strategy_name == "test_strategy"
        assert report.fixture_id == "round-001"
        assert report.scores == (10.0, 20.0, 30.0)
        assert report.mean_score == 20.0

    def test_from_seed_results_empty(self) -> None:
        report = StrategyReport.from_seed_results("test_strategy", "round-001", [])
        assert report.mean_score == 0.0
        assert report.scores == ()

    def test_frozen(self) -> None:
        seeds = [_make_seed_result()]
        report = StrategyReport.from_seed_results("s", "r", seeds)
        try:
            report.mean_score = 0.0  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception as exc:  # noqa: BLE001
            assert "frozen" in str(exc).lower() or "cannot" in str(exc).lower()


class TestBenchmarkReport:
    def _make_report(self) -> BenchmarkReport:
        r1 = StrategyReport.from_seed_results("alpha", "round-001", [_make_seed_result(0, 80.0)])
        r2 = StrategyReport.from_seed_results("beta", "round-001", [_make_seed_result(0, 50.0)])
        r3 = StrategyReport.from_seed_results("gamma", "round-001", [_make_seed_result(0, 65.0)])
        return BenchmarkReport(
            strategy_reports=(r1, r2, r3),
            fixture_id="round-001",
        )

    def test_ranked_descending(self) -> None:
        report = self._make_report()
        ranked = report.ranked()
        assert len(ranked) == 3
        assert ranked[0].strategy_name == "alpha"
        assert ranked[1].strategy_name == "gamma"
        assert ranked[2].strategy_name == "beta"

    def test_ranked_does_not_mutate_original(self) -> None:
        report = self._make_report()
        ranked = report.ranked()
        assert report.strategy_reports[0].strategy_name == "alpha"
        assert ranked[0].strategy_name == "alpha"

    def test_frozen(self) -> None:
        report = self._make_report()
        try:
            report.fixture_id = "other"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception as exc:  # noqa: BLE001
            assert "frozen" in str(exc).lower() or "cannot" in str(exc).lower()
