from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

# pyright: reportMissingImports=false
import pytest
from idk_2.astar_island_dr_plan_1.solver.baseline import build_baseline_tensor
from idk_2.astar_island_dr_plan_1.solver.benchmark_suite import (
    BenchmarkHardFailure,
    BenchmarkSuiteReport,
    RobustnessCheckResult,
    evaluate_benchmark_suite,
)
from idk_2.astar_island_dr_plan_1.solver.evaluation_contract import (
    ArtifactHash,
    BenchmarkInput,
    CandidatePredictionBundle,
    CandidateSeedPrediction,
    FrozenDatasetManifest,
    OrganizerSeedAnalysis,
    QueryTraceEntry,
    RoundQueryTrace,
    SeedAnalysisManifestEntry,
    canonical_json_hash,
    canonical_mapping_artifact_hash,
)
from idk_2.astar_island_dr_plan_1.solver.models import Viewport


def test_benchmark_suite_metrics_report_includes_calibration_stability_and_deltas() -> (
    None
):
    benchmark_input = _build_benchmark_input()
    baseline_bundle = _build_candidate_bundle(
        dataset_version="dataset-2026-03-20",
        round_id="round-001",
        candidate_id="candidate-baseline",
        solver_id="baseline-solver",
        prediction_tensors=(
            build_baseline_tensor([[10, 1], [4, 5]]),
            build_baseline_tensor([[11, 3], [4, 5]]),
        ),
    )
    last_blessed_bundle = _build_candidate_bundle(
        dataset_version="dataset-2026-03-20",
        round_id="round-001",
        candidate_id="candidate-last-blessed",
        solver_id="blessed-solver",
        prediction_tensors=(
            benchmark_input.candidate_bundle.predictions[0].prediction,
            build_baseline_tensor([[11, 1], [4, 5]]),
        ),
    )

    report = evaluate_benchmark_suite(
        benchmark_input,
        blessed_baseline_bundle=baseline_bundle,
        last_blessed_bundle=last_blessed_bundle,
    )

    assert report.verdict == "pass"
    assert report.legality.is_legal is True
    assert len(report.per_seed_metrics) == 2
    assert report.aggregate.seed_count == 2
    assert report.aggregate.aggregate_score == pytest.approx(
        report.contract_report.aggregate_metrics.mean_score
    )
    assert report.aggregate.mean_weighted_kl >= 0.0
    assert report.calibration.seed_count == 2
    assert report.calibration.dynamic_cell_count > 0
    assert report.calibration.mean_cross_entropy > 0.0
    assert report.calibration.mean_brier_score >= 0.0
    assert report.calibration.mean_total_variation >= 0.0
    assert report.stability.seed_count == 2
    assert report.stability.score_span >= 0.0
    assert report.fallback.exact_last_blessed_match_seed_indices == (0,)
    assert report.fallback.exact_blessed_baseline_match_seed_indices == ()
    assert report.blessed_baseline.available is True
    assert report.last_blessed_candidate.available is True
    assert len(report.blessed_baseline.per_seed_score_deltas) == 2
    assert len(report.last_blessed_candidate.per_seed_score_deltas) == 2
    assert any(
        abs(delta) > 1e-9 for delta in report.blessed_baseline.per_seed_score_deltas
    )

    payload = report.to_payload()
    assert isinstance(payload["aggregate"], Mapping)
    assert isinstance(payload["calibration"], Mapping)
    assert isinstance(payload["stability"], Mapping)
    assert isinstance(payload["fallback"], Mapping)
    assert isinstance(payload["robustness"], Mapping)
    assert isinstance(payload["verdict_reasons"], list)
    per_seed_metrics = payload["per_seed_metrics"]
    assert isinstance(per_seed_metrics, list)
    first_seed_metric = per_seed_metrics[0]
    assert isinstance(first_seed_metric, Mapping)
    assert isinstance(first_seed_metric["calibration"], Mapping)
    blessed_baseline = payload["blessed_baseline"]
    last_blessed_candidate = payload["last_blessed_candidate"]
    assert isinstance(blessed_baseline, Mapping)
    assert isinstance(last_blessed_candidate, Mapping)
    assert blessed_baseline["available"] is True
    assert last_blessed_candidate["available"] is True


def test_benchmark_suite_robustness_report_labels_report_only_findings() -> None:
    benchmark_input = _build_benchmark_input()
    query_trace = RoundQueryTrace(
        dataset_version=benchmark_input.query_trace.dataset_version,
        round_id=benchmark_input.query_trace.round_id,
        seed_ids=benchmark_input.query_trace.seed_ids,
        entries=(
            benchmark_input.query_trace.entries[0],
            QueryTraceEntry(
                dataset_version=benchmark_input.query_trace.entries[1].dataset_version,
                round_id=benchmark_input.query_trace.entries[1].round_id,
                seed_index=benchmark_input.query_trace.entries[1].seed_index,
                query_index=benchmark_input.query_trace.entries[1].query_index,
                viewport=Viewport(x=1, y=1, width=1, height=1),
                response_artifact_hash=(
                    benchmark_input.query_trace.entries[1].response_artifact_hash
                ),
                queries_used=benchmark_input.query_trace.entries[1].queries_used,
                queries_max=benchmark_input.query_trace.entries[1].queries_max,
            ),
        ),
        trace_artifact_hash=benchmark_input.query_trace.trace_artifact_hash,
        query_budget=benchmark_input.query_trace.query_budget,
    )
    benchmark_input = BenchmarkInput(
        dataset_manifest=benchmark_input.dataset_manifest,
        query_trace=query_trace,
        analyses=benchmark_input.analyses,
        candidate_bundle=_build_candidate_bundle(
            dataset_version="dataset-2026-03-20",
            round_id="round-001",
            candidate_id="candidate-robustness",
            solver_id="solver-under-test",
            prediction_tensors=(
                _candidate_tensor_a_overconfident(),
                _candidate_tensor_b(),
            ),
        ),
    )
    baseline_bundle = _build_candidate_bundle(
        dataset_version="dataset-2026-03-20",
        round_id="round-001",
        candidate_id="candidate-baseline",
        solver_id="baseline-solver",
        prediction_tensors=(
            _candidate_tensor_a(),
            _candidate_tensor_b(),
        ),
    )

    report = evaluate_benchmark_suite(
        benchmark_input,
        blessed_baseline_bundle=baseline_bundle,
    )

    assert report.verdict == "pass"
    robustness_by_name = {
        check.name: check for check in report.robustness.checks
    }
    edge_check = robustness_by_name["edge_clamped_viewports"]
    assert edge_check.severity == "report_only"
    assert edge_check.status == "report_only"
    assert edge_check.affected_seed_indices == (0, 1)
    assert "regressed against the blessed baseline" in edge_check.summary
    overconfidence_check = robustness_by_name["overconfident_predictions"]
    assert overconfidence_check.severity == "report_only"
    assert overconfidence_check.status == "report_only"
    assert overconfidence_check.affected_seed_indices == (0,)
    assert report.verdict_reasons == ()
    payload = report.to_payload()
    robustness_payload = payload["robustness"]
    assert isinstance(robustness_payload, Mapping)
    checks = robustness_payload["checks"]
    assert isinstance(checks, list)
    assert any(
        isinstance(check, Mapping)
        and check["name"] == "edge_clamped_viewports"
        and check["status"] == "report_only"
        for check in checks
    )


def test_benchmark_suite_per_seed_regression_is_report_only_without_gate() -> None:
    benchmark_input = _build_benchmark_input(
        prediction_tensors=(
            _candidate_tensor_a_regressed(),
            _candidate_tensor_b_improved(),
        )
    )
    baseline_bundle = _build_candidate_bundle(
        dataset_version="dataset-2026-03-20",
        round_id="round-001",
        candidate_id="candidate-baseline",
        solver_id="baseline-solver",
        prediction_tensors=(
            _candidate_tensor_a(),
            _candidate_tensor_b(),
        ),
    )

    report = evaluate_benchmark_suite(
        benchmark_input,
        blessed_baseline_bundle=baseline_bundle,
    )

    assert report.verdict == "pass"
    regression_check = _robustness_check(report, "per_seed_regression")
    assert regression_check.severity == "report_only"
    assert regression_check.status == "report_only"
    assert regression_check.affected_seed_indices == (0,)
    assert (
        "Aggregate improvement hides per-seed regressions"
        in regression_check.summary
    )


def test_benchmark_suite_per_seed_regression_gate_can_fail_aggregate_improvement(
) -> None:
    benchmark_input = _build_benchmark_input(
        prediction_tensors=(
            _candidate_tensor_a_regressed(),
            _candidate_tensor_b_improved(),
        )
    )
    baseline_bundle = _build_candidate_bundle(
        dataset_version="dataset-2026-03-20",
        round_id="round-001",
        candidate_id="candidate-baseline",
        solver_id="baseline-solver",
        prediction_tensors=(
            _candidate_tensor_a(),
            _candidate_tensor_b(),
        ),
    )

    report = evaluate_benchmark_suite(
        benchmark_input,
        blessed_baseline_bundle=baseline_bundle,
        per_seed_regression_gate=2.0,
    )

    assert report.blessed_baseline.reference_aggregate_score is not None
    assert (
        report.aggregate.aggregate_score
        > report.blessed_baseline.reference_aggregate_score
    )
    assert report.verdict == "fail"
    regression_check = _robustness_check(report, "per_seed_regression")
    assert regression_check.severity == "hard_gate"
    assert regression_check.status == "fail"
    assert regression_check.affected_seed_indices == (0,)
    assert any(
        "Per-seed regression hard gate failed" in reason
        for reason in report.verdict_reasons
    )
    assert report.contract_report.verdict == "fail"


def test_benchmark_suite_incomplete_seed_sets_are_hard_failure() -> None:
    benchmark_input = _build_benchmark_input()
    object.__setattr__(
        benchmark_input.candidate_bundle,
        "predictions",
        benchmark_input.candidate_bundle.predictions[:1],
    )

    with pytest.raises(BenchmarkHardFailure, match="exactly one prediction per seed"):
        evaluate_benchmark_suite(benchmark_input)


def test_benchmark_suite_mixed_dataset_versions_are_hard_failure() -> None:
    benchmark_input = _build_benchmark_input()
    object.__setattr__(
        benchmark_input.query_trace,
        "dataset_version",
        "dataset-2026-03-21",
    )

    with pytest.raises(BenchmarkHardFailure, match="mixes dataset versions"):
        evaluate_benchmark_suite(benchmark_input)


def test_benchmark_suite_class_order_mismatch_is_hard_failure() -> None:
    benchmark_input = _build_benchmark_input()
    object.__setattr__(
        benchmark_input.candidate_bundle,
        "mapping_artifact_hash",
        canonical_json_hash({"kind": "mapping", "variant": "shuffled-order"}),
    )

    with pytest.raises(
        BenchmarkHardFailure,
        match="possible illegal class ordering",
    ):
        evaluate_benchmark_suite(benchmark_input)


def test_benchmark_suite_reference_dataset_mixing_is_hard_failure() -> None:
    benchmark_input = _build_benchmark_input()
    baseline_bundle = _build_candidate_bundle(
        dataset_version="dataset-2026-03-21",
        round_id="round-001",
        candidate_id="candidate-baseline",
        solver_id="baseline-solver",
        prediction_tensors=(
            _candidate_tensor_a(),
            _candidate_tensor_b(),
        ),
    )

    with pytest.raises(
        BenchmarkHardFailure,
        match="blessed baseline bundle mixes dataset versions",
    ):
        evaluate_benchmark_suite(
            benchmark_input,
            blessed_baseline_bundle=baseline_bundle,
        )


def test_benchmark_suite_illegal_candidate_is_hard_failure() -> None:
    benchmark_input = _build_benchmark_input()
    benchmark_input.candidate_bundle.predictions[0].prediction[0][0] = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    with pytest.raises(BenchmarkHardFailure, match="failed legality check"):
        evaluate_benchmark_suite(benchmark_input)


def _robustness_check(
    report: BenchmarkSuiteReport, name: str
) -> RobustnessCheckResult:
    for check in report.robustness.checks:
        if check.name == name:
            return check
    raise AssertionError(f"Missing robustness check {name!r}")


def _build_benchmark_input(
    *,
    prediction_tensors: tuple[list[list[list[float]]], list[list[list[float]]]]
    | None = None,
) -> BenchmarkInput:
    dataset_version = "dataset-2026-03-20"
    round_id = "round-001"
    seed_ids = (0, 1)
    mapping_hash = canonical_mapping_artifact_hash()
    query_trace_hash = canonical_json_hash({"kind": "trace", "round_id": round_id})
    analysis_hashes = (
        canonical_json_hash({"kind": "analysis", "seed_index": 0}),
        canonical_json_hash({"kind": "analysis", "seed_index": 1}),
    )
    manifest = FrozenDatasetManifest(
        dataset_version=dataset_version,
        capture_timestamp="2026-03-20T10:00:00Z",
        round_id=round_id,
        seed_ids=seed_ids,
        source_endpoints=(
            "/astar-island/rounds/round-001",
            "/astar-island/analysis/round-001/0",
            "/astar-island/analysis/round-001/1",
        ),
        artifact_hashes=(
            ArtifactHash("mapping/terrain_mapping.json", mapping_hash),
            ArtifactHash("query-trace.json", query_trace_hash),
            ArtifactHash("analysis/round-001/seed-00.json", analysis_hashes[0]),
            ArtifactHash("analysis/round-001/seed-01.json", analysis_hashes[1]),
        ),
        solver_id="benchmark-suite-test",
        map_width=2,
        map_height=2,
        mapping_artifact_hash=mapping_hash,
        query_trace_artifact_hash=query_trace_hash,
        seed_analyses=(
            SeedAnalysisManifestEntry(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=0,
                analysis_artifact_name="analysis/round-001/seed-00.json",
                analysis_artifact_hash=analysis_hashes[0],
            ),
            SeedAnalysisManifestEntry(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=1,
                analysis_artifact_name="analysis/round-001/seed-01.json",
                analysis_artifact_hash=analysis_hashes[1],
            ),
        ),
    )
    query_trace = RoundQueryTrace(
        dataset_version=dataset_version,
        round_id=round_id,
        seed_ids=seed_ids,
        entries=(
            QueryTraceEntry(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=0,
                query_index=0,
                viewport=Viewport(x=0, y=0, width=2, height=2),
                response_artifact_hash=canonical_json_hash({"response": 0}),
                queries_used=1,
                queries_max=50,
            ),
            QueryTraceEntry(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=1,
                query_index=1,
                viewport=Viewport(x=0, y=0, width=2, height=2),
                response_artifact_hash=canonical_json_hash({"response": 1}),
                queries_used=2,
                queries_max=50,
            ),
        ),
        trace_artifact_hash=query_trace_hash,
    )
    effective_prediction_tensors = (
        (_candidate_tensor_a(), _candidate_tensor_b())
        if prediction_tensors is None
        else prediction_tensors
    )
    candidate_bundle = _build_candidate_bundle(
        dataset_version=dataset_version,
        round_id=round_id,
        candidate_id="candidate-under-test",
        solver_id="solver-under-test",
        prediction_tensors=effective_prediction_tensors,
    )
    analyses = (
        OrganizerSeedAnalysis(
            dataset_version=dataset_version,
            round_id=round_id,
            seed_index=0,
            width=2,
            height=2,
            analysis_artifact_hash=analysis_hashes[0],
            ground_truth=_ground_truth_tensor_a(),
            organizer_score=81.0,
        ),
        OrganizerSeedAnalysis(
            dataset_version=dataset_version,
            round_id=round_id,
            seed_index=1,
            width=2,
            height=2,
            analysis_artifact_hash=analysis_hashes[1],
            ground_truth=_ground_truth_tensor_b(),
            organizer_score=73.0,
        ),
    )
    return BenchmarkInput(
        dataset_manifest=manifest,
        query_trace=query_trace,
        analyses=analyses,
        candidate_bundle=candidate_bundle,
    )


def _build_candidate_bundle(
    *,
    dataset_version: str,
    round_id: str,
    candidate_id: str,
    solver_id: str,
    prediction_tensors: tuple[list[list[list[float]]], list[list[list[float]]]],
) -> CandidatePredictionBundle:
    prediction_hashes = (
        canonical_json_hash({"candidate": candidate_id, "seed_index": 0}),
        canonical_json_hash({"candidate": candidate_id, "seed_index": 1}),
    )
    return CandidatePredictionBundle(
        dataset_version=dataset_version,
        round_id=round_id,
        solver_id=solver_id,
        candidate_id=candidate_id,
        seed_ids=(0, 1),
        width=2,
        height=2,
        mapping_artifact_hash=canonical_mapping_artifact_hash(),
        query_trace_artifact_hash=canonical_json_hash(
            {"kind": "trace", "round_id": round_id}
        ),
        predictions=(
            CandidateSeedPrediction(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=0,
                width=2,
                height=2,
                prediction=prediction_tensors[0],
                prediction_artifact_hash=prediction_hashes[0],
            ),
            CandidateSeedPrediction(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=1,
                width=2,
                height=2,
                prediction=prediction_tensors[1],
                prediction_artifact_hash=prediction_hashes[1],
            ),
        ),
    )


def _candidate_tensor_a() -> list[list[list[float]]]:
    return [
        [
            [0.60, 0.15, 0.10, 0.05, 0.05, 0.05],
            [0.10, 0.50, 0.15, 0.10, 0.10, 0.05],
        ],
        [
            [0.15, 0.10, 0.10, 0.15, 0.40, 0.10],
            [0.05, 0.05, 0.05, 0.10, 0.10, 0.65],
        ],
    ]


def _candidate_tensor_b() -> list[list[list[float]]]:
    return [
        [
            [0.55, 0.18, 0.07, 0.08, 0.07, 0.05],
            [0.10, 0.25, 0.40, 0.10, 0.10, 0.05],
        ],
        [
            [0.08, 0.12, 0.10, 0.20, 0.40, 0.10],
            [0.10, 0.15, 0.10, 0.25, 0.20, 0.20],
        ],
    ]


def _candidate_tensor_a_regressed() -> list[list[list[float]]]:
    tensor = deepcopy(_candidate_tensor_a())
    tensor[0][0] = [0.80, 0.08, 0.04, 0.03, 0.03, 0.02]
    return tensor


def _candidate_tensor_b_improved() -> list[list[list[float]]]:
    tensor = deepcopy(_candidate_tensor_b())
    tensor[1][1] = [0.03, 0.07, 0.08, 0.17, 0.20, 0.45]
    return tensor


def _candidate_tensor_a_overconfident() -> list[list[list[float]]]:
    tensor = deepcopy(_candidate_tensor_a())
    tensor[0][0] = [0.94, 0.02, 0.01, 0.01, 0.01, 0.01]
    tensor[0][1] = [0.03, 0.82, 0.05, 0.04, 0.03, 0.03]
    return tensor


def _ground_truth_tensor_a() -> list[list[list[float]]]:
    return [
        [
            [0.65, 0.10, 0.08, 0.05, 0.07, 0.05],
            [0.05, 0.60, 0.15, 0.10, 0.05, 0.05],
        ],
        [
            [0.10, 0.10, 0.10, 0.15, 0.45, 0.10],
            [0.05, 0.05, 0.05, 0.05, 0.10, 0.70],
        ],
    ]


def _ground_truth_tensor_b() -> list[list[list[float]]]:
    return [
        [
            [0.60, 0.15, 0.05, 0.10, 0.05, 0.05],
            [0.10, 0.20, 0.45, 0.10, 0.10, 0.05],
        ],
        [
            [0.10, 0.10, 0.10, 0.15, 0.45, 0.10],
            [0.05, 0.10, 0.10, 0.20, 0.20, 0.35],
        ],
    ]
