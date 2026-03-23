from __future__ import annotations

import json
from collections.abc import Callable
from math import isclose
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import RoundFixture
from astar_twin.harness.diagnostics import (
    compute_diagnostic_report,
    compute_seed_diagnostics,
    compute_strategy_diagnostics,
)
from astar_twin.harness.models import BenchmarkReport, SeedResult, StrategyReport
from astar_twin.scoring import safe_prediction

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


def _full_tensor(
    probabilities: tuple[float, float, float, float, float, float],
) -> NDArray[np.float64]:
    raw = np.broadcast_to(np.array(probabilities, dtype=np.float64), (10, 10, 6)).copy()
    return safe_prediction(raw)


def _prob_vector(
    probabilities: tuple[float, float, float, float, float, float],
) -> NDArray[np.float64]:
    return np.asarray(probabilities, dtype=np.float64)


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _make_seed_result(
    seed_index: int,
    score: float,
    ground_truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
) -> SeedResult:
    return SeedResult(
        seed_index=seed_index,
        score=score,
        ground_truth=ground_truth,
        prediction=safe_prediction(prediction),
    )


def _ordered_worst_case_seed_result(score: float = 72.0) -> SeedResult:
    ground_truth = _full_tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    prediction = ground_truth.copy()
    prediction[1, 1] = _prob_vector((0.10, 0.90, 0.0, 0.0, 0.0, 0.0))
    prediction[2, 2] = _prob_vector((0.20, 0.80, 0.0, 0.0, 0.0, 0.0))
    prediction[3, 3] = _prob_vector((0.33, 0.67, 0.0, 0.0, 0.0, 0.0))
    return _make_seed_result(
        seed_index=2,
        score=score,
        ground_truth=ground_truth,
        prediction=prediction,
    )


def _fixture_impl() -> RoundFixture:
    return load_fixture(FIXTURE_PATH)


fixture = pytest.fixture(name="fixture")(_fixture_impl)


def _uniform_ground_truth_impl() -> NDArray[np.float64]:
    return np.full((10, 10, 6), 1.0 / 6.0, dtype=np.float64)


uniform_ground_truth = pytest.fixture(name="uniform_ground_truth")(_uniform_ground_truth_impl)


class TestComputeSeedDiagnostics:
    def test_returns_expected_fields_and_requested_number_of_worst_cells(self) -> None:
        seed_result = _ordered_worst_case_seed_result()

        diagnostics = compute_seed_diagnostics(seed_result, n_worst_cells=3)

        assert diagnostics.seed_index == seed_result.seed_index
        assert diagnostics.score == seed_result.score
        assert diagnostics.weighted_kl > 0.0
        assert diagnostics.n_dynamic_cells == 100
        assert diagnostics.n_static_cells == 0
        assert diagnostics.mean_entropy > 0.0
        assert len(diagnostics.per_class) == 6
        assert len(diagnostics.worst_cells) == 3

    def test_per_class_fractions_sum_to_about_one(self) -> None:
        ground_truth = _full_tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        prediction = _full_tensor((0.33, 0.67, 0.0, 0.0, 0.0, 0.0))
        seed_result = _make_seed_result(
            seed_index=0,
            score=50.0,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        diagnostics = compute_seed_diagnostics(seed_result)
        fraction_sum = sum(metric.fraction_of_total_loss for metric in diagnostics.per_class)

        assert isclose(fraction_sum, 1.0, abs_tol=0.02)

    def test_worst_cells_are_sorted_by_descending_weighted_kl(self) -> None:
        diagnostics = compute_seed_diagnostics(_ordered_worst_case_seed_result(), n_worst_cells=3)

        assert [(cell.row, cell.col) for cell in diagnostics.worst_cells] == [
            (1, 1),
            (2, 2),
            (3, 3),
        ]
        weighted_kls = [cell.weighted_kl for cell in diagnostics.worst_cells]
        assert weighted_kls == sorted(weighted_kls, reverse=True)

    def test_uniform_prediction_against_uniform_ground_truth_has_near_zero_kl(
        self, uniform_ground_truth: NDArray[np.float64]
    ) -> None:
        seed_result = _make_seed_result(
            seed_index=1,
            score=100.0,
            ground_truth=uniform_ground_truth,
            prediction=uniform_ground_truth,
        )

        diagnostics = compute_seed_diagnostics(seed_result)

        assert isclose(diagnostics.weighted_kl, 0.0, abs_tol=1e-12)
        for metric in diagnostics.per_class:
            assert isclose(metric.mean_kl_contribution, 0.0, abs_tol=1e-12)
            assert isclose(metric.total_weighted_kl, 0.0, abs_tol=1e-12)
            assert isclose(metric.fraction_of_total_loss, 0.0, abs_tol=1e-12)


class TestComputeStrategyDiagnostics:
    def test_aggregates_across_seeds_correctly(self) -> None:
        seed_results = [
            _make_seed_result(
                seed_index=0,
                score=81.0,
                ground_truth=_full_tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                prediction=_full_tensor((0.40, 0.60, 0.0, 0.0, 0.0, 0.0)),
            ),
            _make_seed_result(
                seed_index=1,
                score=62.0,
                ground_truth=_full_tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                prediction=_full_tensor((0.33, 0.67, 0.0, 0.0, 0.0, 0.0)),
            ),
            _make_seed_result(
                seed_index=4,
                score=93.0,
                ground_truth=_full_tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                prediction=_full_tensor((0.60, 0.40, 0.0, 0.0, 0.0, 0.0)),
            ),
        ]
        report = StrategyReport.from_seed_results(
            strategy_name="diagnostic-strategy",
            fixture_id="test-round-001",
            seed_results=seed_results,
        )

        diagnostics = compute_strategy_diagnostics(report)
        seed_diagnostics = [compute_seed_diagnostics(seed_result) for seed_result in seed_results]

        assert diagnostics.strategy_name == report.strategy_name
        assert diagnostics.fixture_id == report.fixture_id
        assert isclose(diagnostics.mean_score, report.mean_score, abs_tol=1e-12)
        assert len(diagnostics.seed_diagnostics) == 3
        assert diagnostics.worst_seed_index == 1
        assert diagnostics.best_seed_index == 4

        for class_index, aggregate in enumerate(diagnostics.per_class_aggregate):
            expected_mean_gt = _mean(
                tuple(
                    seed_diag.per_class[class_index].mean_gt_prob for seed_diag in seed_diagnostics
                )
            )
            expected_mean_pred = _mean(
                tuple(
                    seed_diag.per_class[class_index].mean_pred_prob
                    for seed_diag in seed_diagnostics
                )
            )
            expected_mean_kl = _mean(
                tuple(
                    seed_diag.per_class[class_index].mean_kl_contribution
                    for seed_diag in seed_diagnostics
                )
            )
            expected_total_weighted_kl = _mean(
                tuple(
                    seed_diag.per_class[class_index].total_weighted_kl
                    for seed_diag in seed_diagnostics
                )
            )
            expected_fraction = _mean(
                tuple(
                    seed_diag.per_class[class_index].fraction_of_total_loss
                    for seed_diag in seed_diagnostics
                )
            )
            assert isclose(aggregate.mean_gt_prob, expected_mean_gt, abs_tol=1e-12)
            assert isclose(aggregate.mean_pred_prob, expected_mean_pred, abs_tol=1e-12)
            assert isclose(aggregate.mean_kl_contribution, expected_mean_kl, abs_tol=1e-12)
            assert isclose(aggregate.total_weighted_kl, expected_total_weighted_kl, abs_tol=1e-12)
            assert isclose(aggregate.fraction_of_total_loss, expected_fraction, abs_tol=1e-12)


class TestComputeDiagnosticReport:
    def test_wraps_benchmark_report_correctly(self, fixture: RoundFixture) -> None:
        shared_seed_result = _ordered_worst_case_seed_result(score=77.0)
        strategy_reports = (
            StrategyReport.from_seed_results(
                strategy_name="alpha",
                fixture_id=fixture.id,
                seed_results=[shared_seed_result],
            ),
            StrategyReport.from_seed_results(
                strategy_name="beta",
                fixture_id=fixture.id,
                seed_results=[
                    _make_seed_result(
                        seed_index=3,
                        score=88.0,
                        ground_truth=_full_tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                        prediction=_full_tensor((0.50, 0.50, 0.0, 0.0, 0.0, 0.0)),
                    )
                ],
            ),
        )
        benchmark_report = BenchmarkReport(
            strategy_reports=strategy_reports,
            fixture_id=fixture.id,
        )

        diagnostics = compute_diagnostic_report(benchmark_report, n_worst_cells=4)

        assert diagnostics.fixture_id == fixture.id
        assert len(diagnostics.strategy_diagnostics) == 2
        assert [item.strategy_name for item in diagnostics.strategy_diagnostics] == [
            "alpha",
            "beta",
        ]
        assert len(diagnostics.strategy_diagnostics[0].seed_diagnostics[0].worst_cells) == 4

    def test_to_dict_produces_json_serializable_output(self, fixture: RoundFixture) -> None:
        strategy_report = StrategyReport.from_seed_results(
            strategy_name="json-strategy",
            fixture_id=fixture.id,
            seed_results=[_ordered_worst_case_seed_result(score=68.0)],
        )
        benchmark_report = BenchmarkReport(
            strategy_reports=(strategy_report,),
            fixture_id=fixture.id,
        )

        diagnostic_report = compute_diagnostic_report(benchmark_report)
        to_dict = cast(Callable[[], dict[str, object]], diagnostic_report.to_dict)
        report_dict = to_dict()
        encoded = json.dumps(report_dict)
        decoded = cast(dict[str, object], json.loads(encoded))

        assert decoded["fixture_id"] == fixture.id
        strategies = cast(list[dict[str, object]], decoded["strategies"])
        assert len(strategies) == 1
        assert strategies[0]["strategy_name"] == "json-strategy"
