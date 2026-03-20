from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .contract import CLASS_COUNT
from .evaluation_contract import (
    BenchmarkInput,
    BenchmarkReport,
    CandidatePredictionBundle,
    CandidateSeedPrediction,
    OrganizerSeedAnalysis,
    RoundQueryTrace,
    SeedBenchmarkMetrics,
    canonical_mapping_artifact_hash,
)
from .models import Viewport
from .validator import entropy_weighted_kl_score, validate_prediction_tensor

PredictionTensor = list[list[list[float]]]


class BenchmarkHardFailure(ValueError):
    pass


@dataclass(frozen=True)
class LegalitySummary:
    is_legal: bool
    checked_seed_count: int

    def __post_init__(self) -> None:
        if not self.is_legal:
            raise ValueError(
                "LegalitySummary only represents successful hard-gated runs."
            )
        if self.checked_seed_count <= 0:
            raise ValueError("LegalitySummary checked_seed_count must be positive.")

    def to_payload(self) -> dict[str, object]:
        return {
            "is_legal": self.is_legal,
            "checked_seed_count": self.checked_seed_count,
        }


@dataclass(frozen=True)
class SeedCalibrationSummary:
    dynamic_cell_count: int
    mean_cross_entropy: float
    mean_brier_score: float
    mean_total_variation: float

    def __post_init__(self) -> None:
        if self.dynamic_cell_count < 0:
            raise ValueError("Seed calibration dynamic_cell_count cannot be negative.")
        if not all(
            math.isfinite(value)
            for value in (
                self.mean_cross_entropy,
                self.mean_brier_score,
                self.mean_total_variation,
            )
        ):
            raise ValueError("Seed calibration metrics must be finite.")

    def to_payload(self) -> dict[str, object]:
        return {
            "dynamic_cell_count": self.dynamic_cell_count,
            "mean_cross_entropy": self.mean_cross_entropy,
            "mean_brier_score": self.mean_brier_score,
            "mean_total_variation": self.mean_total_variation,
        }


@dataclass(frozen=True)
class SeedMetricReport:
    dataset_version: str
    round_id: str
    seed_index: int
    score: float
    weighted_kl: float
    entropy_weight_sum: float
    calibration: SeedCalibrationSummary
    prediction_artifact_hash: str
    analysis_artifact_hash: str
    matches_blessed_baseline: bool | None = None
    matches_last_blessed_candidate: bool | None = None

    def __post_init__(self) -> None:
        if not self.dataset_version:
            raise ValueError("SeedMetricReport dataset_version must be non-empty.")
        if not self.round_id:
            raise ValueError("SeedMetricReport round_id must be non-empty.")
        if self.seed_index < 0:
            raise ValueError("SeedMetricReport seed_index must be non-negative.")
        if not math.isfinite(self.score):
            raise ValueError("SeedMetricReport score must be finite.")
        if not math.isfinite(self.weighted_kl):
            raise ValueError("SeedMetricReport weighted_kl must be finite.")
        if not math.isfinite(self.entropy_weight_sum) or self.entropy_weight_sum < 0.0:
            raise ValueError(
                "SeedMetricReport entropy_weight_sum must be finite and non-negative."
            )
        if len(self.prediction_artifact_hash) != 64:
            raise ValueError(
                "SeedMetricReport prediction_artifact_hash must be SHA-256 hex."
            )
        if len(self.analysis_artifact_hash) != 64:
            raise ValueError(
                "SeedMetricReport analysis_artifact_hash must be SHA-256 hex."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "seed_index": self.seed_index,
            "score": self.score,
            "weighted_kl": self.weighted_kl,
            "entropy_weight_sum": self.entropy_weight_sum,
            "calibration": self.calibration.to_payload(),
            "prediction_artifact_hash": self.prediction_artifact_hash,
            "analysis_artifact_hash": self.analysis_artifact_hash,
            "matches_blessed_baseline": self.matches_blessed_baseline,
            "matches_last_blessed_candidate": self.matches_last_blessed_candidate,
        }


@dataclass(frozen=True)
class AggregateMetricSummary:
    seed_count: int
    aggregate_score: float
    mean_weighted_kl: float
    min_score: float
    max_score: float

    def __post_init__(self) -> None:
        if self.seed_count <= 0:
            raise ValueError("AggregateMetricSummary seed_count must be positive.")
        if not all(
            math.isfinite(value)
            for value in (
                self.aggregate_score,
                self.mean_weighted_kl,
                self.min_score,
                self.max_score,
            )
        ):
            raise ValueError("AggregateMetricSummary values must be finite.")
        if self.min_score > self.max_score:
            raise ValueError(
                "AggregateMetricSummary min_score cannot exceed max_score."
            )

    @classmethod
    def from_per_seed_metrics(
        cls, per_seed_metrics: tuple[SeedMetricReport, ...]
    ) -> AggregateMetricSummary:
        if not per_seed_metrics:
            raise ValueError("Aggregate metrics require at least one seed metric.")
        scores = [metric.score for metric in per_seed_metrics]
        weighted_kls = [metric.weighted_kl for metric in per_seed_metrics]
        return cls(
            seed_count=len(per_seed_metrics),
            aggregate_score=math.fsum(scores) / len(scores),
            mean_weighted_kl=math.fsum(weighted_kls) / len(weighted_kls),
            min_score=min(scores),
            max_score=max(scores),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "seed_count": self.seed_count,
            "aggregate_score": self.aggregate_score,
            "mean_weighted_kl": self.mean_weighted_kl,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }


@dataclass(frozen=True)
class CalibrationSummary:
    seed_count: int
    dynamic_cell_count: int
    mean_cross_entropy: float
    mean_brier_score: float
    mean_total_variation: float

    def __post_init__(self) -> None:
        if self.seed_count <= 0:
            raise ValueError("CalibrationSummary seed_count must be positive.")
        if self.dynamic_cell_count < 0:
            raise ValueError(
                "CalibrationSummary dynamic_cell_count cannot be negative."
            )
        if not all(
            math.isfinite(value)
            for value in (
                self.mean_cross_entropy,
                self.mean_brier_score,
                self.mean_total_variation,
            )
        ):
            raise ValueError("CalibrationSummary values must be finite.")

    @classmethod
    def from_per_seed_metrics(
        cls, per_seed_metrics: tuple[SeedMetricReport, ...]
    ) -> CalibrationSummary:
        if not per_seed_metrics:
            raise ValueError("Calibration summary requires per-seed metrics.")
        dynamic_cell_count = sum(
            metric.calibration.dynamic_cell_count for metric in per_seed_metrics
        )
        if dynamic_cell_count == 0:
            return cls(
                seed_count=len(per_seed_metrics),
                dynamic_cell_count=0,
                mean_cross_entropy=0.0,
                mean_brier_score=0.0,
                mean_total_variation=0.0,
            )
        cross_entropy_sum = math.fsum(
            metric.calibration.mean_cross_entropy
            * metric.calibration.dynamic_cell_count
            for metric in per_seed_metrics
        )
        brier_sum = math.fsum(
            metric.calibration.mean_brier_score * metric.calibration.dynamic_cell_count
            for metric in per_seed_metrics
        )
        total_variation_sum = math.fsum(
            metric.calibration.mean_total_variation
            * metric.calibration.dynamic_cell_count
            for metric in per_seed_metrics
        )
        return cls(
            seed_count=len(per_seed_metrics),
            dynamic_cell_count=dynamic_cell_count,
            mean_cross_entropy=cross_entropy_sum / dynamic_cell_count,
            mean_brier_score=brier_sum / dynamic_cell_count,
            mean_total_variation=total_variation_sum / dynamic_cell_count,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "seed_count": self.seed_count,
            "dynamic_cell_count": self.dynamic_cell_count,
            "mean_cross_entropy": self.mean_cross_entropy,
            "mean_brier_score": self.mean_brier_score,
            "mean_total_variation": self.mean_total_variation,
        }


@dataclass(frozen=True)
class StabilitySummary:
    seed_count: int
    score_standard_deviation: float
    weighted_kl_standard_deviation: float
    score_span: float

    def __post_init__(self) -> None:
        if self.seed_count <= 0:
            raise ValueError("StabilitySummary seed_count must be positive.")
        if not all(
            math.isfinite(value)
            for value in (
                self.score_standard_deviation,
                self.weighted_kl_standard_deviation,
                self.score_span,
            )
        ):
            raise ValueError("StabilitySummary values must be finite.")

    @classmethod
    def from_per_seed_metrics(
        cls, per_seed_metrics: tuple[SeedMetricReport, ...]
    ) -> StabilitySummary:
        if not per_seed_metrics:
            raise ValueError("Stability summary requires per-seed metrics.")
        scores = [metric.score for metric in per_seed_metrics]
        weighted_kls = [metric.weighted_kl for metric in per_seed_metrics]
        return cls(
            seed_count=len(per_seed_metrics),
            score_standard_deviation=_population_standard_deviation(scores),
            weighted_kl_standard_deviation=_population_standard_deviation(weighted_kls),
            score_span=max(scores) - min(scores),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "seed_count": self.seed_count,
            "score_standard_deviation": self.score_standard_deviation,
            "weighted_kl_standard_deviation": self.weighted_kl_standard_deviation,
            "score_span": self.score_span,
        }


@dataclass(frozen=True)
class FallbackSummary:
    exact_blessed_baseline_match_seed_indices: tuple[int, ...]
    exact_last_blessed_match_seed_indices: tuple[int, ...]

    @property
    def exact_blessed_baseline_fallback(self) -> bool:
        return bool(self.exact_blessed_baseline_match_seed_indices)

    @property
    def exact_last_blessed_fallback(self) -> bool:
        return bool(self.exact_last_blessed_match_seed_indices)

    def to_payload(self) -> dict[str, object]:
        return {
            "exact_blessed_baseline_match_seed_indices": list(
                self.exact_blessed_baseline_match_seed_indices
            ),
            "exact_last_blessed_match_seed_indices": list(
                self.exact_last_blessed_match_seed_indices
            ),
            "exact_blessed_baseline_fallback": self.exact_blessed_baseline_fallback,
            "exact_last_blessed_fallback": self.exact_last_blessed_fallback,
        }


@dataclass(frozen=True)
class ReferenceComparison:
    reference_label: str
    available: bool
    reference_candidate_id: str | None
    reference_solver_id: str | None
    reference_aggregate_score: float | None
    aggregate_score_delta: float | None
    per_seed_score_deltas: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.reference_label:
            raise ValueError("ReferenceComparison reference_label must be non-empty.")
        if self.available:
            if self.reference_candidate_id is None or self.reference_solver_id is None:
                raise ValueError(
                    "Available ReferenceComparison entries require "
                    "candidate and solver ids."
                )
            if (
                self.reference_aggregate_score is None
                or self.aggregate_score_delta is None
            ):
                raise ValueError(
                    "Available ReferenceComparison entries require aggregate scores."
                )
            if not math.isfinite(self.reference_aggregate_score):
                raise ValueError("Reference aggregate score must be finite.")
            if not math.isfinite(self.aggregate_score_delta):
                raise ValueError("Reference aggregate delta must be finite.")
            if not self.per_seed_score_deltas:
                raise ValueError(
                    "Available ReferenceComparison entries require per-seed deltas."
                )
            if not all(math.isfinite(value) for value in self.per_seed_score_deltas):
                raise ValueError("Reference per-seed deltas must be finite.")
        elif self.per_seed_score_deltas:
            raise ValueError(
                "Unavailable ReferenceComparison entries must not "
                "include per-seed deltas."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "reference_label": self.reference_label,
            "available": self.available,
            "reference_candidate_id": self.reference_candidate_id,
            "reference_solver_id": self.reference_solver_id,
            "reference_aggregate_score": self.reference_aggregate_score,
            "aggregate_score_delta": self.aggregate_score_delta,
            "per_seed_score_deltas": list(self.per_seed_score_deltas),
        }


@dataclass(frozen=True)
class RobustnessCheckResult:
    name: str
    severity: str
    status: str
    summary: str
    affected_seed_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RobustnessCheckResult name must be non-empty.")
        if self.severity not in {"hard_gate", "report_only"}:
            raise ValueError(
                "RobustnessCheckResult severity must be 'hard_gate' or 'report_only'."
            )
        if self.status not in {"pass", "fail", "report_only", "not_applicable"}:
            raise ValueError("RobustnessCheckResult status is invalid.")
        if not self.summary:
            raise ValueError("RobustnessCheckResult summary must be non-empty.")
        if any(seed_index < 0 for seed_index in self.affected_seed_indices):
            raise ValueError(
                "RobustnessCheckResult affected_seed_indices must be non-negative."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "severity": self.severity,
            "status": self.status,
            "summary": self.summary,
            "affected_seed_indices": list(self.affected_seed_indices),
        }


@dataclass(frozen=True)
class RobustnessSummary:
    checks: tuple[RobustnessCheckResult, ...]

    def __post_init__(self) -> None:
        if not self.checks:
            raise ValueError("RobustnessSummary requires at least one check.")

    @property
    def hard_gate_failures(self) -> tuple[RobustnessCheckResult, ...]:
        return tuple(
            check
            for check in self.checks
            if check.severity == "hard_gate" and check.status == "fail"
        )

    @property
    def report_only_findings(self) -> tuple[RobustnessCheckResult, ...]:
        return tuple(check for check in self.checks if check.status == "report_only")

    def to_payload(self) -> dict[str, object]:
        return {
            "checks": [check.to_payload() for check in self.checks],
            "hard_gate_failure_count": len(self.hard_gate_failures),
            "report_only_finding_count": len(self.report_only_findings),
        }


@dataclass(frozen=True)
class BenchmarkSuiteReport:
    dataset_version: str
    round_id: str
    candidate_id: str
    solver_id: str
    verdict: str
    legality: LegalitySummary
    per_seed_metrics: tuple[SeedMetricReport, ...]
    aggregate: AggregateMetricSummary
    calibration: CalibrationSummary
    stability: StabilitySummary
    fallback: FallbackSummary
    blessed_baseline: ReferenceComparison
    last_blessed_candidate: ReferenceComparison
    robustness: RobustnessSummary
    verdict_reasons: tuple[str, ...]
    contract_report: BenchmarkReport

    def __post_init__(self) -> None:
        if not self.dataset_version:
            raise ValueError("BenchmarkSuiteReport dataset_version must be non-empty.")
        if not self.round_id:
            raise ValueError("BenchmarkSuiteReport round_id must be non-empty.")
        if not self.candidate_id:
            raise ValueError("BenchmarkSuiteReport candidate_id must be non-empty.")
        if not self.solver_id:
            raise ValueError("BenchmarkSuiteReport solver_id must be non-empty.")
        if not self.verdict:
            raise ValueError("BenchmarkSuiteReport verdict must be non-empty.")
        if not self.per_seed_metrics:
            raise ValueError("BenchmarkSuiteReport requires per-seed metrics.")
        if self.aggregate.seed_count != len(self.per_seed_metrics):
            raise ValueError(
                "BenchmarkSuiteReport aggregate seed_count does not match "
                "per-seed metrics."
            )
        if self.contract_report.dataset_version != self.dataset_version:
            raise ValueError("BenchmarkSuiteReport mixes dataset versions.")
        if self.contract_report.round_id != self.round_id:
            raise ValueError("BenchmarkSuiteReport mixes round ids.")
        if self.contract_report.candidate_id != self.candidate_id:
            raise ValueError("BenchmarkSuiteReport mixes candidate ids.")
        if self.contract_report.solver_id != self.solver_id:
            raise ValueError("BenchmarkSuiteReport mixes solver ids.")
        if self.verdict == "fail" and not self.verdict_reasons:
            raise ValueError(
                "BenchmarkSuiteReport fail verdicts require verdict reasons."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "candidate_id": self.candidate_id,
            "solver_id": self.solver_id,
            "verdict": self.verdict,
            "legality": self.legality.to_payload(),
            "per_seed_metrics": [item.to_payload() for item in self.per_seed_metrics],
            "aggregate": self.aggregate.to_payload(),
            "calibration": self.calibration.to_payload(),
            "stability": self.stability.to_payload(),
            "fallback": self.fallback.to_payload(),
            "blessed_baseline": self.blessed_baseline.to_payload(),
            "last_blessed_candidate": self.last_blessed_candidate.to_payload(),
            "robustness": self.robustness.to_payload(),
            "verdict_reasons": list(self.verdict_reasons),
            "contract_report": self.contract_report.to_payload(),
        }


@dataclass(frozen=True)
class _BundleScoreSummary:
    bundle: CandidatePredictionBundle
    benchmark_report: BenchmarkReport
    per_seed_scores: tuple[float, ...]

    @property
    def aggregate_score(self) -> float:
        return self.benchmark_report.aggregate_metrics.mean_score


def evaluate_benchmark_suite(
    benchmark_input: BenchmarkInput,
    *,
    blessed_baseline_bundle: CandidatePredictionBundle | None = None,
    last_blessed_bundle: CandidatePredictionBundle | None = None,
    per_seed_regression_gate: float | None = None,
) -> BenchmarkSuiteReport:
    if per_seed_regression_gate is not None and per_seed_regression_gate < 0.0:
        raise ValueError("per_seed_regression_gate must be non-negative when set.")
    analyses_by_seed = _validate_benchmark_input_integrity(benchmark_input)
    baseline_predictions_by_seed = _bundle_predictions_by_seed(
        blessed_baseline_bundle,
        benchmark_input=benchmark_input,
        label="blessed baseline",
    )
    last_blessed_predictions_by_seed = _bundle_predictions_by_seed(
        last_blessed_bundle,
        benchmark_input=benchmark_input,
        label="last blessed candidate",
    )

    per_seed_metrics: list[SeedMetricReport] = []
    contract_seed_metrics: list[SeedBenchmarkMetrics] = []
    candidate_predictions_by_seed = {
        prediction.seed_index: prediction
        for prediction in benchmark_input.candidate_bundle.predictions
    }
    for seed_index in benchmark_input.dataset_manifest.seed_ids:
        analysis = analyses_by_seed[seed_index]
        prediction = candidate_predictions_by_seed[seed_index]
        divergence = _entropy_weighted_kl_details(
            prediction=prediction.prediction,
            ground_truth=analysis.ground_truth,
        )
        seed_calibration = _calibration_for_seed(
            prediction=prediction.prediction,
            ground_truth=analysis.ground_truth,
        )
        seed_score = entropy_weighted_kl_score(
            prediction.prediction,
            analysis.ground_truth,
        )
        per_seed_metrics.append(
            SeedMetricReport(
                dataset_version=benchmark_input.dataset_manifest.dataset_version,
                round_id=benchmark_input.dataset_manifest.round_id,
                seed_index=seed_index,
                score=seed_score,
                weighted_kl=divergence.weighted_kl,
                entropy_weight_sum=divergence.entropy_weight_sum,
                calibration=seed_calibration,
                prediction_artifact_hash=prediction.prediction_artifact_hash,
                analysis_artifact_hash=analysis.analysis_artifact_hash,
                matches_blessed_baseline=_matches_reference_seed(
                    seed_index,
                    prediction.prediction,
                    baseline_predictions_by_seed,
                ),
                matches_last_blessed_candidate=_matches_reference_seed(
                    seed_index,
                    prediction.prediction,
                    last_blessed_predictions_by_seed,
                ),
            )
        )
        contract_seed_metrics.append(
            SeedBenchmarkMetrics(
                dataset_version=benchmark_input.dataset_manifest.dataset_version,
                round_id=benchmark_input.dataset_manifest.round_id,
                seed_index=seed_index,
                score=seed_score,
                prediction_artifact_hash=prediction.prediction_artifact_hash,
                analysis_artifact_hash=analysis.analysis_artifact_hash,
            )
        )

    per_seed_metrics_tuple = tuple(per_seed_metrics)
    aggregate = AggregateMetricSummary.from_per_seed_metrics(per_seed_metrics_tuple)
    calibration_summary = CalibrationSummary.from_per_seed_metrics(
        per_seed_metrics_tuple
    )
    stability = StabilitySummary.from_per_seed_metrics(per_seed_metrics_tuple)
    fallback = FallbackSummary(
        exact_blessed_baseline_match_seed_indices=tuple(
            metric.seed_index
            for metric in per_seed_metrics_tuple
            if metric.matches_blessed_baseline is True
        ),
        exact_last_blessed_match_seed_indices=tuple(
            metric.seed_index
            for metric in per_seed_metrics_tuple
            if metric.matches_last_blessed_candidate is True
        ),
    )
    candidate_scores = tuple(metric.score for metric in per_seed_metrics_tuple)
    baseline_summary = _score_reference_bundle(
        blessed_baseline_bundle,
        benchmark_input=benchmark_input,
        analyses_by_seed=analyses_by_seed,
        label="blessed baseline",
    )
    last_blessed_summary = _score_reference_bundle(
        last_blessed_bundle,
        benchmark_input=benchmark_input,
        analyses_by_seed=analyses_by_seed,
        label="last blessed candidate",
    )
    robustness = _build_robustness_summary(
        benchmark_input=benchmark_input,
        candidate_metrics=per_seed_metrics_tuple,
        candidate_predictions_by_seed=candidate_predictions_by_seed,
        analyses_by_seed=analyses_by_seed,
        blessed_baseline_summary=baseline_summary,
        blessed_baseline_predictions_by_seed=baseline_predictions_by_seed,
        last_blessed_summary=last_blessed_summary,
        last_blessed_predictions_by_seed=last_blessed_predictions_by_seed,
        per_seed_regression_gate=per_seed_regression_gate,
    )
    verdict_reasons = tuple(check.summary for check in robustness.hard_gate_failures)
    verdict = "fail" if verdict_reasons else "pass"

    return BenchmarkSuiteReport(
        dataset_version=benchmark_input.dataset_manifest.dataset_version,
        round_id=benchmark_input.dataset_manifest.round_id,
        candidate_id=benchmark_input.candidate_bundle.candidate_id,
        solver_id=benchmark_input.candidate_bundle.solver_id,
        verdict=verdict,
        legality=LegalitySummary(
            is_legal=True,
            checked_seed_count=len(per_seed_metrics_tuple),
        ),
        per_seed_metrics=per_seed_metrics_tuple,
        aggregate=aggregate,
        calibration=calibration_summary,
        stability=stability,
        fallback=fallback,
        blessed_baseline=_build_reference_comparison(
            reference_label="blessed baseline",
            candidate_scores=candidate_scores,
            candidate_aggregate_score=aggregate.aggregate_score,
            summary=baseline_summary,
        ),
        last_blessed_candidate=_build_reference_comparison(
            reference_label="last blessed candidate",
            candidate_scores=candidate_scores,
            candidate_aggregate_score=aggregate.aggregate_score,
            summary=last_blessed_summary,
        ),
        robustness=robustness,
        verdict_reasons=verdict_reasons,
        contract_report=BenchmarkReport.from_per_seed_metrics(
            dataset_version=benchmark_input.dataset_manifest.dataset_version,
            round_id=benchmark_input.dataset_manifest.round_id,
            candidate_id=benchmark_input.candidate_bundle.candidate_id,
            solver_id=benchmark_input.candidate_bundle.solver_id,
            verdict=verdict,
            per_seed_metrics=tuple(contract_seed_metrics),
            mapping_artifact_hash=benchmark_input.dataset_manifest.mapping_artifact_hash,
            query_trace_artifact_hash=benchmark_input.dataset_manifest.query_trace_artifact_hash,
        ),
    )


@dataclass(frozen=True)
class _EntropyWeightedKlDetails:
    weighted_kl: float
    entropy_weight_sum: float


def _validate_benchmark_input_integrity(
    benchmark_input: BenchmarkInput,
) -> dict[int, OrganizerSeedAnalysis]:
    manifest = benchmark_input.dataset_manifest
    if manifest.mapping_artifact_hash != canonical_mapping_artifact_hash():
        raise BenchmarkHardFailure("Benchmark input references a stale class mapping.")

    if benchmark_input.query_trace.dataset_version != manifest.dataset_version:
        raise BenchmarkHardFailure("Benchmark input mixes dataset versions.")
    if benchmark_input.query_trace.round_id != manifest.round_id:
        raise BenchmarkHardFailure("Benchmark input mixes round ids.")
    if (
        benchmark_input.query_trace.trace_artifact_hash
        != manifest.query_trace_artifact_hash
    ):
        raise BenchmarkHardFailure("Benchmark input query trace drift detected.")
    if benchmark_input.candidate_bundle.dataset_version != manifest.dataset_version:
        raise BenchmarkHardFailure("Benchmark input mixes dataset versions.")
    if benchmark_input.candidate_bundle.round_id != manifest.round_id:
        raise BenchmarkHardFailure("Benchmark input mixes round ids.")
    if (
        benchmark_input.candidate_bundle.query_trace_artifact_hash
        != manifest.query_trace_artifact_hash
    ):
        raise BenchmarkHardFailure("Candidate bundle query trace drift detected.")
    if (
        benchmark_input.candidate_bundle.mapping_artifact_hash
        != manifest.mapping_artifact_hash
    ):
        raise BenchmarkHardFailure(
            "Candidate bundle mapping hash drift detected "
            "(possible illegal class ordering)."
        )
    if (
        benchmark_input.candidate_bundle.width != manifest.map_width
        or benchmark_input.candidate_bundle.height != manifest.map_height
    ):
        raise BenchmarkHardFailure(
            "Benchmark input has candidate width/height mismatch."
        )

    seed_ids = benchmark_input.dataset_manifest.seed_ids
    if benchmark_input.query_trace.seed_ids != seed_ids:
        raise BenchmarkHardFailure("Benchmark input mixes query-trace seed coverage.")
    if benchmark_input.candidate_bundle.seed_ids != seed_ids:
        raise BenchmarkHardFailure("Benchmark input mixes candidate seed coverage.")
    if (
        len(benchmark_input.query_trace.entries)
        > benchmark_input.query_trace.query_budget
    ):
        raise BenchmarkHardFailure("Round query trace exceeds the shared query budget.")
    query_indices = tuple(
        entry.query_index for entry in benchmark_input.query_trace.entries
    )
    expected_query_indices = tuple(range(len(benchmark_input.query_trace.entries)))
    if query_indices != expected_query_indices:
        raise BenchmarkHardFailure(
            "Round query trace must use contiguous global query indices."
        )
    for entry in benchmark_input.query_trace.entries:
        if entry.dataset_version != manifest.dataset_version:
            raise BenchmarkHardFailure("Benchmark input mixes dataset versions.")
        if entry.round_id != manifest.round_id:
            raise BenchmarkHardFailure("Benchmark input mixes round ids.")
        if entry.seed_index not in seed_ids:
            raise BenchmarkHardFailure(
                "Round query trace references unknown seed index "
                f"{entry.seed_index}."
            )
        if (
            entry.queries_max is not None
            and entry.queries_max != benchmark_input.query_trace.query_budget
        ):
            raise BenchmarkHardFailure(
                "Round query trace queries_max does not match query_budget."
            )

    analyses_by_seed: dict[int, OrganizerSeedAnalysis] = {}
    expected_analysis_hashes = {
        entry.seed_index: entry.analysis_artifact_hash
        for entry in manifest.seed_analyses
    }
    for analysis in benchmark_input.analyses:
        if analysis.seed_index in analyses_by_seed:
            raise BenchmarkHardFailure(
                "Benchmark input has duplicate analysis for seed "
                f"{analysis.seed_index}."
            )
        if analysis.dataset_version != manifest.dataset_version:
            raise BenchmarkHardFailure("Benchmark input mixes dataset versions.")
        if analysis.round_id != manifest.round_id:
            raise BenchmarkHardFailure("Benchmark input mixes round ids.")
        if (
            analysis.width != manifest.map_width
            or analysis.height != manifest.map_height
        ):
            raise BenchmarkHardFailure(
                "Benchmark input has analysis width/height mismatch."
            )
        if (
            expected_analysis_hashes.get(analysis.seed_index)
            != analysis.analysis_artifact_hash
        ):
            raise BenchmarkHardFailure(
                "Benchmark input analysis hash mismatch detected."
            )
        _validate_reference_tensor(
            analysis.ground_truth,
            width=analysis.width,
            height=analysis.height,
            tensor_name=f"ground_truth for seed {analysis.seed_index}",
        )
        analyses_by_seed[analysis.seed_index] = analysis
    if tuple(analyses_by_seed) != seed_ids:
        raise BenchmarkHardFailure("Benchmark input is missing seed analysis coverage.")

    predictions_by_seed: dict[int, PredictionTensor] = {}
    for prediction in benchmark_input.candidate_bundle.predictions:
        if prediction.seed_index in predictions_by_seed:
            raise BenchmarkHardFailure(
                "Candidate bundle has duplicate prediction for seed "
                f"{prediction.seed_index}."
            )
        try:
            validate_prediction_tensor(
                prediction.prediction,
                width=benchmark_input.candidate_bundle.width,
                height=benchmark_input.candidate_bundle.height,
            )
        except ValueError as error:
            raise BenchmarkHardFailure(
                "Candidate bundle failed legality check for seed "
                f"{prediction.seed_index}: {error}"
            ) from error
        predictions_by_seed[prediction.seed_index] = prediction.prediction
    if tuple(predictions_by_seed) != seed_ids:
        raise BenchmarkHardFailure(
            "Candidate bundle must include exactly one prediction per seed."
        )
    return analyses_by_seed


def _bundle_predictions_by_seed(
    bundle: CandidatePredictionBundle | None,
    *,
    benchmark_input: BenchmarkInput,
    label: str,
) -> dict[int, PredictionTensor] | None:
    if bundle is None:
        return None
    _validate_reference_bundle(bundle, benchmark_input=benchmark_input, label=label)
    return {
        prediction.seed_index: prediction.prediction
        for prediction in bundle.predictions
    }


def _validate_reference_bundle(
    bundle: CandidatePredictionBundle,
    *,
    benchmark_input: BenchmarkInput,
    label: str,
) -> None:
    manifest = benchmark_input.dataset_manifest
    if bundle.dataset_version != manifest.dataset_version:
        raise BenchmarkHardFailure(f"{label} bundle mixes dataset versions.")
    if bundle.round_id != manifest.round_id:
        raise BenchmarkHardFailure(f"{label} bundle mixes round ids.")
    if bundle.seed_ids != manifest.seed_ids:
        raise BenchmarkHardFailure(f"{label} bundle is missing seed coverage.")
    if bundle.width != manifest.map_width or bundle.height != manifest.map_height:
        raise BenchmarkHardFailure(
            f"{label} bundle has width/height mismatch against benchmark input."
        )
    if bundle.mapping_artifact_hash != manifest.mapping_artifact_hash:
        raise BenchmarkHardFailure(
            f"{label} bundle mapping hash drift detected "
            "(possible illegal class ordering)."
        )
    if bundle.query_trace_artifact_hash != manifest.query_trace_artifact_hash:
        raise BenchmarkHardFailure(f"{label} bundle query trace drift detected.")
    for prediction in bundle.predictions:
        try:
            validate_prediction_tensor(
                prediction.prediction,
                width=bundle.width,
                height=bundle.height,
            )
        except ValueError as error:
            raise BenchmarkHardFailure(
                f"{label} bundle failed legality check for seed "
                f"{prediction.seed_index}: {error}"
            ) from error


def _score_reference_bundle(
    bundle: CandidatePredictionBundle | None,
    *,
    benchmark_input: BenchmarkInput,
    analyses_by_seed: dict[int, OrganizerSeedAnalysis],
    label: str,
) -> _BundleScoreSummary | None:
    if bundle is None:
        return None
    _validate_reference_bundle(bundle, benchmark_input=benchmark_input, label=label)
    per_seed_metrics: list[SeedBenchmarkMetrics] = []
    per_seed_scores: list[float] = []
    for prediction in bundle.predictions:
        analysis = analyses_by_seed[prediction.seed_index]
        score = entropy_weighted_kl_score(prediction.prediction, analysis.ground_truth)
        per_seed_scores.append(score)
        per_seed_metrics.append(
            SeedBenchmarkMetrics(
                dataset_version=bundle.dataset_version,
                round_id=bundle.round_id,
                seed_index=prediction.seed_index,
                score=score,
                prediction_artifact_hash=prediction.prediction_artifact_hash,
                analysis_artifact_hash=analysis.analysis_artifact_hash,
            )
        )
    return _BundleScoreSummary(
        bundle=bundle,
        benchmark_report=BenchmarkReport.from_per_seed_metrics(
            dataset_version=bundle.dataset_version,
            round_id=bundle.round_id,
            candidate_id=bundle.candidate_id,
            solver_id=bundle.solver_id,
            verdict="pass",
            per_seed_metrics=tuple(per_seed_metrics),
            mapping_artifact_hash=bundle.mapping_artifact_hash,
            query_trace_artifact_hash=bundle.query_trace_artifact_hash,
        ),
        per_seed_scores=tuple(per_seed_scores),
    )


def _build_reference_comparison(
    *,
    reference_label: str,
    candidate_scores: tuple[float, ...],
    candidate_aggregate_score: float,
    summary: _BundleScoreSummary | None,
) -> ReferenceComparison:
    if summary is None:
        return ReferenceComparison(
            reference_label=reference_label,
            available=False,
            reference_candidate_id=None,
            reference_solver_id=None,
            reference_aggregate_score=None,
            aggregate_score_delta=None,
            per_seed_score_deltas=(),
        )
    if len(candidate_scores) != len(summary.per_seed_scores):
        raise BenchmarkHardFailure(
            f"{reference_label} score coverage does not match candidate seed coverage."
        )
    per_seed_score_deltas = tuple(
        candidate_scores[index] - summary.per_seed_scores[index]
        for index in range(len(candidate_scores))
    )
    return ReferenceComparison(
        reference_label=reference_label,
        available=True,
        reference_candidate_id=summary.bundle.candidate_id,
        reference_solver_id=summary.bundle.solver_id,
        reference_aggregate_score=summary.aggregate_score,
        aggregate_score_delta=candidate_aggregate_score - summary.aggregate_score,
        per_seed_score_deltas=per_seed_score_deltas,
    )


def _matches_reference_seed(
    seed_index: int,
    prediction: PredictionTensor,
    reference_predictions_by_seed: dict[int, PredictionTensor] | None,
) -> bool | None:
    if reference_predictions_by_seed is None:
        return None
    return reference_predictions_by_seed[seed_index] == prediction


@dataclass(frozen=True)
class _RobustnessReference:
    label: str
    summary: _BundleScoreSummary
    predictions_by_seed: dict[int, PredictionTensor]


def _build_robustness_summary(
    *,
    benchmark_input: BenchmarkInput,
    candidate_metrics: tuple[SeedMetricReport, ...],
    candidate_predictions_by_seed: Mapping[int, CandidateSeedPrediction],
    analyses_by_seed: dict[int, OrganizerSeedAnalysis],
    blessed_baseline_summary: _BundleScoreSummary | None,
    blessed_baseline_predictions_by_seed: dict[int, PredictionTensor] | None,
    last_blessed_summary: _BundleScoreSummary | None,
    last_blessed_predictions_by_seed: dict[int, PredictionTensor] | None,
    per_seed_regression_gate: float | None,
) -> RobustnessSummary:
    checks: list[RobustnessCheckResult] = []
    reference = _select_robustness_reference(
        blessed_baseline_summary=blessed_baseline_summary,
        blessed_baseline_predictions_by_seed=blessed_baseline_predictions_by_seed,
        last_blessed_summary=last_blessed_summary,
        last_blessed_predictions_by_seed=last_blessed_predictions_by_seed,
    )
    checks.append(
        _edge_clamped_viewport_check(
            query_trace=benchmark_input.query_trace,
            map_width=benchmark_input.dataset_manifest.map_width,
            map_height=benchmark_input.dataset_manifest.map_height,
            candidate_metrics=candidate_metrics,
            reference=reference,
        )
    )
    checks.append(
        _per_seed_regression_check(
            candidate_metrics=candidate_metrics,
            reference=reference,
            gate=per_seed_regression_gate,
        )
    )
    checks.append(
        _overconfidence_check(
            candidate_metrics=candidate_metrics,
            candidate_predictions_by_seed=candidate_predictions_by_seed,
            analyses_by_seed=analyses_by_seed,
            reference=reference,
        )
    )
    return RobustnessSummary(checks=tuple(checks))


def _select_robustness_reference(
    *,
    blessed_baseline_summary: _BundleScoreSummary | None,
    blessed_baseline_predictions_by_seed: dict[int, PredictionTensor] | None,
    last_blessed_summary: _BundleScoreSummary | None,
    last_blessed_predictions_by_seed: dict[int, PredictionTensor] | None,
) -> _RobustnessReference | None:
    if (
        last_blessed_summary is not None
        and last_blessed_predictions_by_seed is not None
    ):
        return _RobustnessReference(
            label="last blessed candidate",
            summary=last_blessed_summary,
            predictions_by_seed=last_blessed_predictions_by_seed,
        )
    if (
        blessed_baseline_summary is not None
        and blessed_baseline_predictions_by_seed is not None
    ):
        return _RobustnessReference(
            label="blessed baseline",
            summary=blessed_baseline_summary,
            predictions_by_seed=blessed_baseline_predictions_by_seed,
        )
    return None


def _edge_clamped_viewport_check(
    *,
    query_trace: RoundQueryTrace,
    map_width: int,
    map_height: int,
    candidate_metrics: tuple[SeedMetricReport, ...],
    reference: _RobustnessReference | None,
) -> RobustnessCheckResult:
    edge_seed_indices = tuple(
        sorted(
            {
                entry.seed_index
                for entry in query_trace.entries
                if _viewport_touches_edge(
                    viewport=entry.viewport,
                    map_width=map_width,
                    map_height=map_height,
                )
            }
        )
    )
    if not edge_seed_indices:
        return RobustnessCheckResult(
            name="edge_clamped_viewports",
            severity="report_only",
            status="not_applicable",
            summary=(
                "No edge-clamped viewport traces were present in this "
                "benchmark input."
            ),
        )
    if reference is None:
        return RobustnessCheckResult(
            name="edge_clamped_viewports",
            severity="report_only",
            status="pass",
            summary=(
                "Edge-clamped viewport coverage observed on seeds "
                f"{list(edge_seed_indices)}; no reference bundle was available "
                "for adversarial comparison."
            ),
            affected_seed_indices=edge_seed_indices,
        )
    candidate_scores = {metric.seed_index: metric.score for metric in candidate_metrics}
    edge_regressions = tuple(
        seed_index
        for offset, seed_index in enumerate(sorted(reference.predictions_by_seed))
        if seed_index in edge_seed_indices
        and candidate_scores[seed_index] < reference.summary.per_seed_scores[offset]
    )
    status = "report_only" if edge_regressions else "pass"
    if edge_regressions:
        summary = (
            "Edge-clamped viewport seeds "
            f"{list(edge_regressions)} regressed against the {reference.label}."
        )
    else:
        summary = (
            "Edge-clamped viewport coverage observed on seeds "
            f"{list(edge_seed_indices)} without regressions against the "
            f"{reference.label}."
        )
    return RobustnessCheckResult(
        name="edge_clamped_viewports",
        severity="report_only",
        status=status,
        summary=summary,
        affected_seed_indices=edge_seed_indices,
    )


def _per_seed_regression_check(
    *,
    candidate_metrics: tuple[SeedMetricReport, ...],
    reference: _RobustnessReference | None,
    gate: float | None,
) -> RobustnessCheckResult:
    if reference is None:
        return RobustnessCheckResult(
            name="per_seed_regression",
            severity="hard_gate" if gate is not None else "report_only",
            status="not_applicable",
            summary="No reference bundle was available for per-seed regression checks.",
        )
    candidate_scores = tuple(metric.score for metric in candidate_metrics)
    deltas = tuple(
        candidate_scores[index] - reference.summary.per_seed_scores[index]
        for index in range(len(candidate_scores))
    )
    regressed_seed_indices = tuple(
        candidate_metrics[index].seed_index
        for index, delta in enumerate(deltas)
        if delta < 0.0
    )
    aggregate_delta = (
        sum(candidate_scores) / len(candidate_scores)
        - reference.summary.aggregate_score
    )
    if not regressed_seed_indices:
        return RobustnessCheckResult(
            name="per_seed_regression",
            severity="hard_gate" if gate is not None else "report_only",
            status="pass",
            summary=(
                "No per-seed regressions were detected against the "
                f"{reference.label}."
            ),
        )
    worst_regression = min(deltas)
    if gate is not None and worst_regression < -gate:
        return RobustnessCheckResult(
            name="per_seed_regression",
            severity="hard_gate",
            status="fail",
            summary=(
                "Per-seed regression hard gate failed against the "
                f"{reference.label}: aggregate delta {aggregate_delta:.6f}, "
                f"worst seed delta {worst_regression:.6f} exceeded the allowed "
                f"regression gate {gate:.6f}."
            ),
            affected_seed_indices=regressed_seed_indices,
        )
    if aggregate_delta > 0.0:
        summary = (
            "Aggregate improvement hides per-seed regressions against the "
            f"{reference.label}: aggregate delta {aggregate_delta:.6f}, "
            f"regressed seeds {list(regressed_seed_indices)}."
        )
    else:
        summary = (
            "Per-seed regressions were detected against the "
            f"{reference.label}: aggregate delta {aggregate_delta:.6f}, "
            f"regressed seeds {list(regressed_seed_indices)}."
        )
    return RobustnessCheckResult(
        name="per_seed_regression",
        severity="report_only",
        status="report_only",
        summary=summary,
        affected_seed_indices=regressed_seed_indices,
    )


def _overconfidence_check(
    *,
    candidate_metrics: tuple[SeedMetricReport, ...],
    candidate_predictions_by_seed: Mapping[int, CandidateSeedPrediction],
    analyses_by_seed: dict[int, OrganizerSeedAnalysis],
    reference: _RobustnessReference | None,
) -> RobustnessCheckResult:
    if reference is None:
        return RobustnessCheckResult(
            name="overconfident_predictions",
            severity="report_only",
            status="not_applicable",
            summary="No reference bundle was available for overconfidence checks.",
        )
    candidate_metrics_by_seed = {
        metric.seed_index: metric for metric in candidate_metrics
    }
    affected_seed_indices: list[int] = []
    for seed_index, reference_prediction in reference.predictions_by_seed.items():
        candidate_prediction = candidate_predictions_by_seed[seed_index].prediction
        candidate_confidence = _mean_cell_confidence(candidate_prediction)
        reference_confidence = _mean_cell_confidence(reference_prediction)
        reference_kl = _entropy_weighted_kl_details(
            prediction=reference_prediction,
            ground_truth=analyses_by_seed[seed_index].ground_truth,
        ).weighted_kl
        candidate_metric = candidate_metrics_by_seed[seed_index]
        if (
            candidate_confidence > reference_confidence + 1e-12
            and candidate_metric.weighted_kl > reference_kl + 1e-12
        ):
            affected_seed_indices.append(seed_index)
    if not affected_seed_indices:
        return RobustnessCheckResult(
            name="overconfident_predictions",
            severity="report_only",
            status="pass",
            summary=(
                "No overconfident predictions worsened weighted KL against the "
                f"{reference.label}."
            ),
        )
    return RobustnessCheckResult(
        name="overconfident_predictions",
        severity="report_only",
        status="report_only",
        summary=(
            "More decisive predictions worsened weighted KL against the "
            f"{reference.label} on seeds {affected_seed_indices}."
        ),
        affected_seed_indices=tuple(affected_seed_indices),
    )


def _viewport_touches_edge(
    *, viewport: Viewport, map_width: int, map_height: int
) -> bool:
    return (
        viewport.x == 0
        or viewport.y == 0
        or viewport.x + viewport.width == map_width
        or viewport.y + viewport.height == map_height
    )


def _mean_cell_confidence(prediction: PredictionTensor) -> float:
    confidence_sum = 0.0
    cell_count = 0
    for row in prediction:
        for cell in row:
            confidence_sum += max(cell)
            cell_count += 1
    if cell_count == 0:
        raise BenchmarkHardFailure("Prediction tensors must contain at least one cell.")
    return confidence_sum / cell_count


def _calibration_for_seed(
    *,
    prediction: PredictionTensor,
    ground_truth: PredictionTensor,
) -> SeedCalibrationSummary:
    dynamic_cell_count = 0
    cross_entropy_sum = 0.0
    brier_sum = 0.0
    total_variation_sum = 0.0
    for row_index, target_row in enumerate(ground_truth):
        predicted_row = prediction[row_index]
        for column_index, target_cell in enumerate(target_row):
            predicted_cell = predicted_row[column_index]
            if len(target_cell) != len(predicted_cell):
                raise BenchmarkHardFailure(
                    "Calibration tensor class count mismatch at "
                    f"({column_index}, {row_index})."
                )
            entropy = 0.0
            cross_entropy = 0.0
            brier = 0.0
            total_variation = 0.0
            for class_index in range(len(target_cell)):
                target_probability = target_cell[class_index]
                predicted_probability = predicted_cell[class_index]
                if target_probability > 0.0:
                    entropy -= target_probability * math.log(target_probability)
                    cross_entropy -= target_probability * math.log(
                        predicted_probability
                    )
                difference = predicted_probability - target_probability
                brier += difference * difference
                total_variation += abs(difference)
            if entropy <= 0.0:
                continue
            dynamic_cell_count += 1
            cross_entropy_sum += cross_entropy
            brier_sum += brier
            total_variation_sum += 0.5 * total_variation
    if dynamic_cell_count == 0:
        return SeedCalibrationSummary(
            dynamic_cell_count=0,
            mean_cross_entropy=0.0,
            mean_brier_score=0.0,
            mean_total_variation=0.0,
        )
    return SeedCalibrationSummary(
        dynamic_cell_count=dynamic_cell_count,
        mean_cross_entropy=cross_entropy_sum / dynamic_cell_count,
        mean_brier_score=brier_sum / dynamic_cell_count,
        mean_total_variation=total_variation_sum / dynamic_cell_count,
    )


def _entropy_weighted_kl_details(
    *,
    prediction: PredictionTensor,
    ground_truth: PredictionTensor,
) -> _EntropyWeightedKlDetails:
    weighted_divergence = 0.0
    entropy_weight_sum = 0.0
    for row_index, target_row in enumerate(ground_truth):
        predicted_row = prediction[row_index]
        for column_index, target_cell in enumerate(target_row):
            predicted_cell = predicted_row[column_index]
            if len(target_cell) != len(predicted_cell):
                raise BenchmarkHardFailure(
                    f"KL tensor class count mismatch at ({column_index}, {row_index})."
                )
            entropy = 0.0
            divergence = 0.0
            for class_index in range(len(target_cell)):
                target_probability = target_cell[class_index]
                predicted_probability = predicted_cell[class_index]
                if target_probability <= 0.0:
                    continue
                entropy -= target_probability * math.log(target_probability)
                divergence += target_probability * math.log(
                    target_probability / predicted_probability
                )
            if entropy <= 0.0:
                continue
            weighted_divergence += entropy * divergence
            entropy_weight_sum += entropy
    weighted_kl = (
        0.0 if entropy_weight_sum == 0.0 else weighted_divergence / entropy_weight_sum
    )
    return _EntropyWeightedKlDetails(
        weighted_kl=weighted_kl,
        entropy_weight_sum=entropy_weight_sum,
    )


def _validate_reference_tensor(
    tensor: PredictionTensor,
    *,
    width: int,
    height: int,
    tensor_name: str,
) -> None:
    if len(tensor) != height:
        raise BenchmarkHardFailure(
            f"{tensor_name} height mismatch: expected {height}, got {len(tensor)}"
        )
    for row_index, row in enumerate(tensor):
        if len(row) != width:
            raise BenchmarkHardFailure(
                f"{tensor_name} width mismatch on row {row_index}: "
                f"expected {width}, got {len(row)}"
            )
        for column_index, cell in enumerate(row):
            if len(cell) != CLASS_COUNT:
                raise BenchmarkHardFailure(
                    f"{tensor_name} class count mismatch at "
                    f"({column_index}, {row_index}): expected "
                    f"{CLASS_COUNT}, got {len(cell)}"
                )
            if any(probability < 0.0 for probability in cell):
                raise BenchmarkHardFailure(
                    f"{tensor_name} has a negative probability at "
                    f"({column_index}, {row_index})"
                )
            total = math.fsum(cell)
            if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise BenchmarkHardFailure(
                    f"{tensor_name} normalization error at "
                    f"({column_index}, {row_index}): got {total}"
                )


def _population_standard_deviation(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute standard deviation of an empty sample.")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


__all__ = [
    "AggregateMetricSummary",
    "BenchmarkHardFailure",
    "BenchmarkSuiteReport",
    "CalibrationSummary",
    "FallbackSummary",
    "LegalitySummary",
    "ReferenceComparison",
    "RobustnessCheckResult",
    "RobustnessSummary",
    "SeedCalibrationSummary",
    "SeedMetricReport",
    "StabilitySummary",
    "evaluate_benchmark_suite",
]
