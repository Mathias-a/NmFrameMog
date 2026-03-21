"""Integration test: all three strategies evaluated on test-round-001.

Verifies the expected score ordering:
    MCOracleStrategy >= InitialPriorStrategy >= UniformStrategy

Uses n_mc_runs=10 for speed. The ordering constraint is checked with a
small tolerance to avoid flaky failures from statistical noise.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from astar_twin.data.loaders import load_fixture
from astar_twin.harness.models import BenchmarkReport
from astar_twin.harness.runner import BenchmarkRunner
from astar_twin.strategies import REGISTRY

FIXTURE_PATH = (
    Path(__file__).parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture(scope="module")
def benchmark_report() -> BenchmarkReport:
    fixture = load_fixture(FIXTURE_PATH)
    strategies = [cls() for cls in REGISTRY.values()]
    runner = BenchmarkRunner(fixture=fixture, base_seed=42, n_mc_runs=10)
    return runner.run(strategies)


def test_all_three_strategies_present(benchmark_report: BenchmarkReport) -> None:
    names = {sr.strategy_name for sr in benchmark_report.strategy_reports}
    assert "uniform" in names
    assert "initial_prior" in names
    assert "mc_oracle" in names


def test_all_strategies_produce_finite_scores(benchmark_report: BenchmarkReport) -> None:
    for sr in benchmark_report.strategy_reports:
        assert math.isfinite(sr.mean_score), f"{sr.strategy_name} produced non-finite mean_score"


def test_mc_oracle_beats_uniform(benchmark_report: BenchmarkReport) -> None:
    scores = {sr.strategy_name: sr.mean_score for sr in benchmark_report.strategy_reports}
    assert scores["mc_oracle"] >= scores["uniform"] - 1.0, (
        f"mc_oracle ({scores['mc_oracle']:.2f}) should be >= uniform ({scores['uniform']:.2f})"
    )


def test_initial_prior_beats_uniform(benchmark_report: BenchmarkReport) -> None:
    scores = {sr.strategy_name: sr.mean_score for sr in benchmark_report.strategy_reports}
    assert scores["initial_prior"] >= scores["uniform"] - 1.0, (
        f"initial_prior ({scores['initial_prior']:.2f}) should be "
        f">= uniform ({scores['uniform']:.2f})"
    )


def test_mc_oracle_beats_initial_prior(benchmark_report: BenchmarkReport) -> None:
    scores = {sr.strategy_name: sr.mean_score for sr in benchmark_report.strategy_reports}
    assert scores["mc_oracle"] >= scores["initial_prior"] - 1.0, (
        f"mc_oracle ({scores['mc_oracle']:.2f}) should be "
        f">= initial_prior ({scores['initial_prior']:.2f})"
    )


def test_ranked_report_descending(benchmark_report: BenchmarkReport) -> None:
    ranked = benchmark_report.ranked()
    for i in range(len(ranked) - 1):
        assert ranked[i].mean_score >= ranked[i + 1].mean_score


def test_fixture_id_matches(benchmark_report: BenchmarkReport) -> None:
    fixture = load_fixture(FIXTURE_PATH)
    assert benchmark_report.fixture_id == fixture.id
