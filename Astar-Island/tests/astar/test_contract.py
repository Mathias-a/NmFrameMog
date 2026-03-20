from __future__ import annotations

from collections.abc import Mapping

# pyright: reportMissingImports=false
import pytest
from idk_2.astar_island_dr_plan_1.solver.baseline import build_baseline_tensor
from idk_2.astar_island_dr_plan_1.solver.evaluation_contract import (
    ArtifactHash,
    BenchmarkInput,
    BenchmarkReport,
    CandidatePredictionBundle,
    CandidateSeedPrediction,
    FrozenDatasetManifest,
    OrganizerSeedAnalysis,
    QueryTraceEntry,
    RoundQueryTrace,
    SeedAnalysisManifestEntry,
    SeedBenchmarkMetrics,
    canonical_json_hash,
    canonical_mapping_artifact_hash,
)
from idk_2.astar_island_dr_plan_1.solver.models import Viewport


def test_contract_happy_path_builds_manifest_benchmark_input_and_report() -> None:
    dataset_version = "dataset-2026-03-20"
    round_id = "round-001"
    seed_ids = (0, 1)
    mapping_hash = canonical_mapping_artifact_hash()
    query_trace_hash = canonical_json_hash({"kind": "trace", "round_id": round_id})
    analysis_hashes = (
        canonical_json_hash({"kind": "analysis", "seed_index": seed_ids[0]}),
        canonical_json_hash({"kind": "analysis", "seed_index": seed_ids[1]}),
    )
    prediction_hashes = (
        canonical_json_hash({"kind": "prediction", "seed_index": seed_ids[0]}),
        canonical_json_hash({"kind": "prediction", "seed_index": seed_ids[1]}),
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
            ArtifactHash("mapping.json", mapping_hash),
            ArtifactHash("query-trace.json", query_trace_hash),
            ArtifactHash("analysis-00.json", analysis_hashes[0]),
            ArtifactHash("analysis-01.json", analysis_hashes[1]),
        ),
        solver_id="baseline-solver",
        map_width=2,
        map_height=2,
        mapping_artifact_hash=mapping_hash,
        query_trace_artifact_hash=query_trace_hash,
        seed_analyses=(
            SeedAnalysisManifestEntry(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=0,
                analysis_artifact_name="analysis-00.json",
                analysis_artifact_hash=analysis_hashes[0],
            ),
            SeedAnalysisManifestEntry(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=1,
                analysis_artifact_name="analysis-01.json",
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

    predictions = _bundle_predictions(
        dataset_version=dataset_version,
        round_id=round_id,
        prediction_hashes=prediction_hashes,
    )
    candidate_bundle = CandidatePredictionBundle(
        dataset_version=dataset_version,
        round_id=round_id,
        solver_id="baseline-solver",
        candidate_id="candidate-baseline",
        seed_ids=seed_ids,
        width=2,
        height=2,
        mapping_artifact_hash=mapping_hash,
        query_trace_artifact_hash=query_trace_hash,
        predictions=predictions,
    )

    analyses = (
        OrganizerSeedAnalysis(
            dataset_version=dataset_version,
            round_id=round_id,
            seed_index=0,
            width=2,
            height=2,
            analysis_artifact_hash=analysis_hashes[0],
            ground_truth=_ground_truth_tensor(),
            submitted_prediction=predictions[0].prediction,
            organizer_score=78.0,
        ),
        OrganizerSeedAnalysis(
            dataset_version=dataset_version,
            round_id=round_id,
            seed_index=1,
            width=2,
            height=2,
            analysis_artifact_hash=analysis_hashes[1],
            ground_truth=_ground_truth_tensor(),
            submitted_prediction=predictions[1].prediction,
            organizer_score=79.0,
        ),
    )

    benchmark_input = BenchmarkInput(
        dataset_manifest=manifest,
        query_trace=query_trace,
        analyses=analyses,
        candidate_bundle=candidate_bundle,
    )
    report = BenchmarkReport.from_per_seed_metrics(
        dataset_version=dataset_version,
        round_id=round_id,
        candidate_id="candidate-baseline",
        solver_id="baseline-solver",
        verdict="pass",
        per_seed_metrics=(
            SeedBenchmarkMetrics(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=0,
                score=77.5,
                prediction_artifact_hash=prediction_hashes[0],
                analysis_artifact_hash=analysis_hashes[0],
            ),
            SeedBenchmarkMetrics(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=1,
                score=80.5,
                prediction_artifact_hash=prediction_hashes[1],
                analysis_artifact_hash=analysis_hashes[1],
            ),
        ),
        mapping_artifact_hash=mapping_hash,
        query_trace_artifact_hash=query_trace_hash,
    )

    assert benchmark_input.dataset_manifest.dataset_version == dataset_version
    assert report.aggregate_metrics.seed_count == 2
    assert report.aggregate_metrics.mean_score == pytest.approx(79.0)
    payload = report.to_payload()
    per_seed_metrics = payload["per_seed_metrics"]
    aggregate_metrics = payload["aggregate_metrics"]
    assert isinstance(per_seed_metrics, list)
    assert isinstance(aggregate_metrics, Mapping)
    assert payload["dataset_version"] == dataset_version
    assert len(per_seed_metrics) == 2
    assert aggregate_metrics["mean_score"] == pytest.approx(79.0)


def test_contract_rejects_mixed_dataset_versions() -> None:
    dataset_version = "dataset-2026-03-20"
    mixed_dataset_version = "dataset-2026-03-21"
    round_id = "round-001"
    seed_ids = (0, 1)
    mapping_hash = canonical_mapping_artifact_hash()
    query_trace_hash = canonical_json_hash({"kind": "trace", "round_id": round_id})
    analysis_hashes = (
        canonical_json_hash({"kind": "analysis", "seed_index": seed_ids[0]}),
        canonical_json_hash({"kind": "analysis", "seed_index": seed_ids[1]}),
    )
    manifest = FrozenDatasetManifest(
        dataset_version=dataset_version,
        capture_timestamp="2026-03-20T10:00:00Z",
        round_id=round_id,
        seed_ids=seed_ids,
        source_endpoints=("/astar-island/rounds/round-001",),
        artifact_hashes=(
            ArtifactHash("mapping.json", mapping_hash),
            ArtifactHash("query-trace.json", query_trace_hash),
            ArtifactHash("analysis-00.json", analysis_hashes[0]),
            ArtifactHash("analysis-01.json", analysis_hashes[1]),
        ),
        solver_id="baseline-solver",
        map_width=2,
        map_height=2,
        mapping_artifact_hash=mapping_hash,
        query_trace_artifact_hash=query_trace_hash,
        seed_analyses=(
            SeedAnalysisManifestEntry(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=0,
                analysis_artifact_name="analysis-00.json",
                analysis_artifact_hash=analysis_hashes[0],
            ),
            SeedAnalysisManifestEntry(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=1,
                analysis_artifact_name="analysis-01.json",
                analysis_artifact_hash=analysis_hashes[1],
            ),
        ),
    )
    query_trace = RoundQueryTrace(
        dataset_version=dataset_version,
        round_id=round_id,
        seed_ids=seed_ids,
        entries=(),
        trace_artifact_hash=query_trace_hash,
    )
    predictions = _bundle_predictions(
        dataset_version=dataset_version,
        round_id=round_id,
        prediction_hashes=(
            canonical_json_hash({"prediction": 0}),
            canonical_json_hash({"prediction": 1}),
        ),
    )
    candidate_bundle = CandidatePredictionBundle(
        dataset_version=dataset_version,
        round_id=round_id,
        solver_id="baseline-solver",
        candidate_id="candidate-baseline",
        seed_ids=seed_ids,
        width=2,
        height=2,
        mapping_artifact_hash=mapping_hash,
        query_trace_artifact_hash=query_trace_hash,
        predictions=predictions,
    )

    with pytest.raises(ValueError, match="mixes dataset versions"):
        BenchmarkInput(
            dataset_manifest=manifest,
            query_trace=query_trace,
            analyses=(
                OrganizerSeedAnalysis(
                    dataset_version=dataset_version,
                    round_id=round_id,
                    seed_index=0,
                    width=2,
                    height=2,
                    analysis_artifact_hash=analysis_hashes[0],
                    ground_truth=_ground_truth_tensor(),
                ),
                OrganizerSeedAnalysis(
                    dataset_version=mixed_dataset_version,
                    round_id=round_id,
                    seed_index=1,
                    width=2,
                    height=2,
                    analysis_artifact_hash=analysis_hashes[1],
                    ground_truth=_ground_truth_tensor(),
                ),
            ),
            candidate_bundle=candidate_bundle,
        )


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        pytest.param(
            [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            "Tensor floor violation",
            id="zero-probability",
        ),
        pytest.param(
            [
                [[0.80, 0.05, 0.05, 0.04, 0.03, 0.03]],
                [
                    [0.80, 0.05, 0.05, 0.04, 0.03, 0.03],
                ],
            ],
            "Tensor height mismatch",
            id="wrong-height",
        ),
        pytest.param(
            [[[0.20, 0.20, 0.20, 0.20, 0.10, 0.05]]],
            "Tensor normalization error",
            id="bad-normalization",
        ),
    ],
)
def test_contract_rejects_illegal_candidate_tensors(
    prediction: list[list[list[float]]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CandidateSeedPrediction(
            dataset_version="dataset-2026-03-20",
            round_id="round-001",
            seed_index=0,
            width=1,
            height=1,
            prediction=prediction,
            prediction_artifact_hash=canonical_json_hash({"prediction": "illegal"}),
        )


def test_report_requires_complete_per_seed_and_aggregate_metrics() -> None:
    dataset_version = "dataset-2026-03-20"
    round_id = "round-001"
    mapping_hash = canonical_mapping_artifact_hash()
    query_trace_hash = canonical_json_hash({"kind": "trace", "round_id": round_id})
    analysis_hash = canonical_json_hash({"kind": "analysis", "seed_index": 0})
    prediction_hash = canonical_json_hash({"kind": "prediction", "seed_index": 0})

    report = BenchmarkReport.from_per_seed_metrics(
        dataset_version=dataset_version,
        round_id=round_id,
        candidate_id="candidate-baseline",
        solver_id="baseline-solver",
        verdict="pass",
        per_seed_metrics=(
            SeedBenchmarkMetrics(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=0,
                score=82.0,
                prediction_artifact_hash=prediction_hash,
                analysis_artifact_hash=analysis_hash,
            ),
        ),
        mapping_artifact_hash=mapping_hash,
        query_trace_artifact_hash=query_trace_hash,
    )
    assert report.aggregate_metrics.seed_count == 1
    aggregate_metrics = report.to_payload()["aggregate_metrics"]
    assert isinstance(aggregate_metrics, Mapping)
    assert aggregate_metrics["max_score"] == pytest.approx(82.0)

    with pytest.raises(ValueError, match="aggregate seed_count does not match"):
        BenchmarkReport(
            dataset_version=dataset_version,
            round_id=round_id,
            candidate_id="candidate-baseline",
            solver_id="baseline-solver",
            verdict="pass",
            per_seed_metrics=report.per_seed_metrics,
            aggregate_metrics=report.aggregate_metrics.__class__(
                seed_count=2,
                mean_score=82.0,
                min_score=82.0,
                max_score=82.0,
            ),
            mapping_artifact_hash=mapping_hash,
            query_trace_artifact_hash=query_trace_hash,
        )


def _bundle_predictions(
    *, dataset_version: str, round_id: str, prediction_hashes: tuple[str, str]
) -> tuple[CandidateSeedPrediction, CandidateSeedPrediction]:
    tensor = build_baseline_tensor([[10, 1], [4, 5]])
    return (
        CandidateSeedPrediction(
            dataset_version=dataset_version,
            round_id=round_id,
            seed_index=0,
            width=2,
            height=2,
            prediction=tensor,
            prediction_artifact_hash=prediction_hashes[0],
        ),
        CandidateSeedPrediction(
            dataset_version=dataset_version,
            round_id=round_id,
            seed_index=1,
            width=2,
            height=2,
            prediction=tensor,
            prediction_artifact_hash=prediction_hashes[1],
        ),
    )


def _ground_truth_tensor() -> list[list[list[float]]]:
    return [
        [
            [0.70, 0.10, 0.05, 0.05, 0.05, 0.05],
            [0.00, 0.60, 0.20, 0.10, 0.05, 0.05],
        ],
        [
            [0.10, 0.10, 0.10, 0.20, 0.40, 0.10],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.75],
        ],
    ]
