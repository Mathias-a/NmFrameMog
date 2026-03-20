from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime

from .contract import CLASS_COUNT, DEFAULT_QUERY_BUDGET, canonical_mapping_artifact
from .models import Viewport
from .validator import validate_prediction_tensor

EVALUATION_SCHEMA_VERSION = "astar-evaluation-v1"

PredictionTensor = list[list[list[float]]]


def canonical_json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_mapping_artifact_hash() -> str:
    return canonical_json_hash(canonical_mapping_artifact())


@dataclass(frozen=True)
class ArtifactHash:
    artifact_name: str
    sha256: str

    def __post_init__(self) -> None:
        _require_non_empty(self.artifact_name, field_name="artifact_name")
        _validate_sha256(
            self.sha256, field_name=f"artifact hash '{self.artifact_name}'"
        )

    def to_payload(self) -> dict[str, object]:
        return {"artifact_name": self.artifact_name, "sha256": self.sha256}


@dataclass(frozen=True)
class SeedAnalysisManifestEntry:
    dataset_version: str
    round_id: str
    seed_index: int
    analysis_artifact_name: str
    analysis_artifact_hash: str

    def __post_init__(self) -> None:
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_non_empty(self.round_id, field_name="round_id")
        _validate_seed_index(self.seed_index)
        _require_non_empty(
            self.analysis_artifact_name, field_name="analysis_artifact_name"
        )
        _validate_sha256(
            self.analysis_artifact_hash,
            field_name=f"analysis hash for seed {self.seed_index}",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "seed_index": self.seed_index,
            "analysis_artifact_name": self.analysis_artifact_name,
            "analysis_artifact_hash": self.analysis_artifact_hash,
        }


@dataclass(frozen=True)
class FrozenDatasetManifest:
    dataset_version: str
    capture_timestamp: str
    round_id: str
    seed_ids: tuple[int, ...]
    source_endpoints: tuple[str, ...]
    artifact_hashes: tuple[ArtifactHash, ...]
    solver_id: str
    schema_version: str = EVALUATION_SCHEMA_VERSION
    map_width: int = 40
    map_height: int = 40
    mapping_artifact_hash: str = ""
    query_trace_artifact_hash: str = ""
    seed_analyses: tuple[SeedAnalysisManifestEntry, ...] = ()

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_iso8601_timestamp(self.capture_timestamp)
        _require_non_empty(self.round_id, field_name="round_id")
        _validate_dimensions(width=self.map_width, height=self.map_height)
        _validate_seed_ids(self.seed_ids)
        _validate_non_empty_strings(
            self.source_endpoints, field_name="source_endpoints"
        )
        hash_registry = _build_hash_registry(self.artifact_hashes)
        _validate_sha256(self.mapping_artifact_hash, field_name="mapping_artifact_hash")
        _validate_sha256(
            self.query_trace_artifact_hash,
            field_name="query_trace_artifact_hash",
        )
        _require_hash_present(
            hash_registry,
            expected_hash=self.mapping_artifact_hash,
            field_name="mapping_artifact_hash",
        )
        _require_hash_present(
            hash_registry,
            expected_hash=self.query_trace_artifact_hash,
            field_name="query_trace_artifact_hash",
        )
        _validate_seed_analysis_manifest_entries(
            seed_analyses=self.seed_analyses,
            dataset_version=self.dataset_version,
            round_id=self.round_id,
            seed_ids=self.seed_ids,
            hash_registry=hash_registry,
        )
        current_mapping_hash = canonical_mapping_artifact_hash()
        if self.mapping_artifact_hash != current_mapping_hash:
            raise ValueError("Dataset manifest references a stale class mapping hash.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "capture_timestamp": self.capture_timestamp,
            "round_id": self.round_id,
            "seed_ids": list(self.seed_ids),
            "source_endpoints": list(self.source_endpoints),
            "artifact_hashes": [item.to_payload() for item in self.artifact_hashes],
            "solver_id": self.solver_id,
            "map_width": self.map_width,
            "map_height": self.map_height,
            "mapping_artifact_hash": self.mapping_artifact_hash,
            "query_trace_artifact_hash": self.query_trace_artifact_hash,
            "seed_analyses": [item.to_payload() for item in self.seed_analyses],
        }


@dataclass(frozen=True)
class QueryTraceEntry:
    dataset_version: str
    round_id: str
    seed_index: int
    query_index: int
    viewport: Viewport
    response_artifact_hash: str
    queries_used: int | None = None
    queries_max: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_non_empty(self.round_id, field_name="round_id")
        _validate_seed_index(self.seed_index)
        if self.query_index < 0:
            raise ValueError("query_index must be non-negative.")
        _validate_viewport(self.viewport)
        _validate_sha256(
            self.response_artifact_hash,
            field_name=f"response hash for query {self.query_index}",
        )
        if self.queries_used is not None and self.queries_used <= 0:
            raise ValueError("queries_used must be positive when provided.")
        if self.queries_max is not None and self.queries_max <= 0:
            raise ValueError("queries_max must be positive when provided.")
        if (
            self.queries_used is not None
            and self.queries_max is not None
            and self.queries_used > self.queries_max
        ):
            raise ValueError("queries_used cannot exceed queries_max.")

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "seed_index": self.seed_index,
            "query_index": self.query_index,
            "viewport": {
                "x": self.viewport.x,
                "y": self.viewport.y,
                "width": self.viewport.width,
                "height": self.viewport.height,
            },
            "response_artifact_hash": self.response_artifact_hash,
            "queries_used": self.queries_used,
            "queries_max": self.queries_max,
        }


@dataclass(frozen=True)
class RoundQueryTrace:
    dataset_version: str
    round_id: str
    seed_ids: tuple[int, ...]
    entries: tuple[QueryTraceEntry, ...]
    trace_artifact_hash: str
    query_budget: int = DEFAULT_QUERY_BUDGET
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_non_empty(self.round_id, field_name="round_id")
        _validate_seed_ids(self.seed_ids)
        _validate_sha256(self.trace_artifact_hash, field_name="trace_artifact_hash")
        if self.query_budget <= 0:
            raise ValueError("query_budget must be positive.")
        if len(self.entries) > self.query_budget:
            raise ValueError("Round query trace exceeds the shared query budget.")
        expected_query_indices = tuple(range(len(self.entries)))
        actual_query_indices = tuple(entry.query_index for entry in self.entries)
        if actual_query_indices != expected_query_indices:
            raise ValueError(
                "Round query trace must use contiguous global query indices."
            )
        for entry in self.entries:
            if entry.dataset_version != self.dataset_version:
                raise ValueError("Round query trace mixes dataset versions.")
            if entry.round_id != self.round_id:
                raise ValueError("Round query trace mixes round ids.")
            if entry.seed_index not in self.seed_ids:
                raise ValueError(
                    "Round query trace references unknown seed index "
                    f"{entry.seed_index}."
                )
            if entry.queries_max is not None and entry.queries_max != self.query_budget:
                raise ValueError(
                    "Round query trace queries_max does not match query_budget."
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "seed_ids": list(self.seed_ids),
            "entries": [entry.to_payload() for entry in self.entries],
            "trace_artifact_hash": self.trace_artifact_hash,
            "query_budget": self.query_budget,
        }


@dataclass(frozen=True)
class CandidateSeedPrediction:
    dataset_version: str
    round_id: str
    seed_index: int
    width: int
    height: int
    prediction: PredictionTensor
    prediction_artifact_hash: str

    def __post_init__(self) -> None:
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_non_empty(self.round_id, field_name="round_id")
        _validate_seed_index(self.seed_index)
        _validate_dimensions(width=self.width, height=self.height)
        _validate_sha256(
            self.prediction_artifact_hash,
            field_name=f"prediction hash for seed {self.seed_index}",
        )
        validate_prediction_tensor(
            self.prediction,
            width=self.width,
            height=self.height,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "seed_index": self.seed_index,
            "width": self.width,
            "height": self.height,
            "prediction": self.prediction,
            "prediction_artifact_hash": self.prediction_artifact_hash,
        }


@dataclass(frozen=True)
class CandidatePredictionBundle:
    dataset_version: str
    round_id: str
    solver_id: str
    candidate_id: str
    seed_ids: tuple[int, ...]
    width: int
    height: int
    mapping_artifact_hash: str
    query_trace_artifact_hash: str
    predictions: tuple[CandidateSeedPrediction, ...]
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_non_empty(self.round_id, field_name="round_id")
        _require_non_empty(self.solver_id, field_name="solver_id")
        _require_non_empty(self.candidate_id, field_name="candidate_id")
        _validate_seed_ids(self.seed_ids)
        _validate_dimensions(width=self.width, height=self.height)
        _validate_sha256(self.mapping_artifact_hash, field_name="mapping_artifact_hash")
        _validate_sha256(
            self.query_trace_artifact_hash,
            field_name="query_trace_artifact_hash",
        )
        expected_seed_ids = self.seed_ids
        actual_seed_ids = tuple(
            prediction.seed_index for prediction in self.predictions
        )
        if actual_seed_ids != expected_seed_ids:
            raise ValueError(
                "Candidate prediction bundle must include exactly one "
                "prediction per seed."
            )
        for prediction in self.predictions:
            if prediction.dataset_version != self.dataset_version:
                raise ValueError("Candidate prediction bundle mixes dataset versions.")
            if prediction.round_id != self.round_id:
                raise ValueError("Candidate prediction bundle mixes round ids.")
            if prediction.width != self.width or prediction.height != self.height:
                raise ValueError(
                    "Candidate prediction bundle has width/height mismatch."
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "solver_id": self.solver_id,
            "candidate_id": self.candidate_id,
            "seed_ids": list(self.seed_ids),
            "width": self.width,
            "height": self.height,
            "mapping_artifact_hash": self.mapping_artifact_hash,
            "query_trace_artifact_hash": self.query_trace_artifact_hash,
            "predictions": [item.to_payload() for item in self.predictions],
        }


@dataclass(frozen=True)
class OrganizerSeedAnalysis:
    dataset_version: str
    round_id: str
    seed_index: int
    width: int
    height: int
    analysis_artifact_hash: str
    ground_truth: PredictionTensor
    submitted_prediction: PredictionTensor | None = None
    organizer_score: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_non_empty(self.round_id, field_name="round_id")
        _validate_seed_index(self.seed_index)
        _validate_dimensions(width=self.width, height=self.height)
        _validate_sha256(
            self.analysis_artifact_hash,
            field_name=f"analysis hash for seed {self.seed_index}",
        )
        _validate_reference_tensor(
            self.ground_truth,
            width=self.width,
            height=self.height,
            tensor_name=f"ground_truth for seed {self.seed_index}",
        )
        if self.submitted_prediction is not None:
            validate_prediction_tensor(
                self.submitted_prediction,
                width=self.width,
                height=self.height,
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "seed_index": self.seed_index,
            "width": self.width,
            "height": self.height,
            "analysis_artifact_hash": self.analysis_artifact_hash,
            "ground_truth": self.ground_truth,
            "submitted_prediction": self.submitted_prediction,
            "organizer_score": self.organizer_score,
        }


@dataclass(frozen=True)
class BenchmarkInput:
    dataset_manifest: FrozenDatasetManifest
    query_trace: RoundQueryTrace
    analyses: tuple[OrganizerSeedAnalysis, ...]
    candidate_bundle: CandidatePredictionBundle

    def __post_init__(self) -> None:
        dataset_version = self.dataset_manifest.dataset_version
        round_id = self.dataset_manifest.round_id
        if self.query_trace.dataset_version != dataset_version:
            raise ValueError("Benchmark input mixes dataset versions.")
        if self.query_trace.round_id != round_id:
            raise ValueError("Benchmark input mixes round ids.")
        if self.candidate_bundle.dataset_version != dataset_version:
            raise ValueError("Benchmark input mixes dataset versions.")
        if self.candidate_bundle.round_id != round_id:
            raise ValueError("Benchmark input mixes round ids.")
        if (
            self.query_trace.trace_artifact_hash
            != self.dataset_manifest.query_trace_artifact_hash
        ):
            raise ValueError("Query trace hash does not match the dataset manifest.")
        if (
            self.candidate_bundle.query_trace_artifact_hash
            != self.dataset_manifest.query_trace_artifact_hash
        ):
            raise ValueError(
                "Candidate bundle query trace hash does not match the dataset manifest."
            )
        if (
            self.candidate_bundle.mapping_artifact_hash
            != self.dataset_manifest.mapping_artifact_hash
        ):
            raise ValueError("Candidate bundle mapping hash drift detected.")
        if (
            self.dataset_manifest.mapping_artifact_hash
            != canonical_mapping_artifact_hash()
        ):
            raise ValueError("Benchmark input references a stale class mapping.")
        if self.dataset_manifest.seed_ids != self.query_trace.seed_ids:
            raise ValueError("Query trace seed ids do not match the dataset manifest.")
        if self.dataset_manifest.seed_ids != self.candidate_bundle.seed_ids:
            raise ValueError(
                "Candidate bundle seed ids do not match the dataset manifest."
            )

        expected_analysis_hashes = {
            item.seed_index: item.analysis_artifact_hash
            for item in self.dataset_manifest.seed_analyses
        }
        actual_analysis_hashes: dict[int, str] = {}
        for analysis in self.analyses:
            if analysis.dataset_version != dataset_version:
                raise ValueError("Benchmark input mixes dataset versions.")
            if analysis.round_id != round_id:
                raise ValueError("Benchmark input mixes round ids.")
            if (
                analysis.width != self.dataset_manifest.map_width
                or analysis.height != self.dataset_manifest.map_height
            ):
                raise ValueError("Benchmark input has analysis width/height mismatch.")
            if analysis.seed_index in actual_analysis_hashes:
                raise ValueError(
                    "Benchmark input has duplicate analysis for seed "
                    f"{analysis.seed_index}."
                )
            actual_analysis_hashes[analysis.seed_index] = (
                analysis.analysis_artifact_hash
            )

        if tuple(actual_analysis_hashes) != self.dataset_manifest.seed_ids:
            raise ValueError("Benchmark input is missing seed analysis coverage.")
        if actual_analysis_hashes != expected_analysis_hashes:
            raise ValueError("Benchmark input analysis hash mismatch detected.")
        if (
            self.candidate_bundle.width != self.dataset_manifest.map_width
            or self.candidate_bundle.height != self.dataset_manifest.map_height
        ):
            raise ValueError("Benchmark input has candidate width/height mismatch.")


@dataclass(frozen=True)
class SeedBenchmarkMetrics:
    dataset_version: str
    round_id: str
    seed_index: int
    score: float
    prediction_artifact_hash: str
    analysis_artifact_hash: str

    def __post_init__(self) -> None:
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_non_empty(self.round_id, field_name="round_id")
        _validate_seed_index(self.seed_index)
        if not math.isfinite(self.score):
            raise ValueError("Per-seed score must be finite.")
        _validate_sha256(
            self.prediction_artifact_hash,
            field_name=f"prediction hash for seed {self.seed_index}",
        )
        _validate_sha256(
            self.analysis_artifact_hash,
            field_name=f"analysis hash for seed {self.seed_index}",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "seed_index": self.seed_index,
            "score": self.score,
            "prediction_artifact_hash": self.prediction_artifact_hash,
            "analysis_artifact_hash": self.analysis_artifact_hash,
        }


@dataclass(frozen=True)
class AggregateBenchmarkMetrics:
    seed_count: int
    mean_score: float
    min_score: float
    max_score: float

    def __post_init__(self) -> None:
        if self.seed_count <= 0:
            raise ValueError("Aggregate seed_count must be positive.")
        if not all(
            math.isfinite(value)
            for value in (self.mean_score, self.min_score, self.max_score)
        ):
            raise ValueError("Aggregate scores must be finite.")
        if self.min_score > self.max_score:
            raise ValueError("Aggregate min_score cannot exceed max_score.")
        if not self.min_score <= self.mean_score <= self.max_score:
            raise ValueError(
                "Aggregate mean_score must fall between min_score and max_score."
            )

    @classmethod
    def from_per_seed_metrics(
        cls, per_seed_metrics: tuple[SeedBenchmarkMetrics, ...]
    ) -> AggregateBenchmarkMetrics:
        if not per_seed_metrics:
            raise ValueError("Cannot build aggregate metrics without per-seed metrics.")
        scores = [metric.score for metric in per_seed_metrics]
        return cls(
            seed_count=len(scores),
            mean_score=math.fsum(scores) / len(scores),
            min_score=min(scores),
            max_score=max(scores),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "seed_count": self.seed_count,
            "mean_score": self.mean_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    dataset_version: str
    round_id: str
    candidate_id: str
    solver_id: str
    verdict: str
    per_seed_metrics: tuple[SeedBenchmarkMetrics, ...]
    aggregate_metrics: AggregateBenchmarkMetrics
    mapping_artifact_hash: str
    query_trace_artifact_hash: str
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_non_empty(self.round_id, field_name="round_id")
        _require_non_empty(self.candidate_id, field_name="candidate_id")
        _require_non_empty(self.solver_id, field_name="solver_id")
        _require_non_empty(self.verdict, field_name="verdict")
        _validate_sha256(self.mapping_artifact_hash, field_name="mapping_artifact_hash")
        _validate_sha256(
            self.query_trace_artifact_hash,
            field_name="query_trace_artifact_hash",
        )
        if self.mapping_artifact_hash != canonical_mapping_artifact_hash():
            raise ValueError("Benchmark report references a stale class mapping.")
        if not self.per_seed_metrics:
            raise ValueError("Benchmark report must include per-seed metrics.")
        seed_ids = tuple(metric.seed_index for metric in self.per_seed_metrics)
        _validate_seed_ids(seed_ids)
        if self.aggregate_metrics.seed_count != len(self.per_seed_metrics):
            raise ValueError(
                "Benchmark report aggregate seed_count does not match per-seed metrics."
            )
        expected_aggregate = AggregateBenchmarkMetrics.from_per_seed_metrics(
            self.per_seed_metrics
        )
        if self.aggregate_metrics != expected_aggregate:
            raise ValueError(
                "Benchmark report aggregate metrics do not match per-seed metrics."
            )
        for metric in self.per_seed_metrics:
            if metric.dataset_version != self.dataset_version:
                raise ValueError("Benchmark report mixes dataset versions.")
            if metric.round_id != self.round_id:
                raise ValueError("Benchmark report mixes round ids.")

    @classmethod
    def from_per_seed_metrics(
        cls,
        *,
        dataset_version: str,
        round_id: str,
        candidate_id: str,
        solver_id: str,
        verdict: str,
        per_seed_metrics: tuple[SeedBenchmarkMetrics, ...],
        mapping_artifact_hash: str,
        query_trace_artifact_hash: str,
    ) -> BenchmarkReport:
        return cls(
            dataset_version=dataset_version,
            round_id=round_id,
            candidate_id=candidate_id,
            solver_id=solver_id,
            verdict=verdict,
            per_seed_metrics=per_seed_metrics,
            aggregate_metrics=AggregateBenchmarkMetrics.from_per_seed_metrics(
                per_seed_metrics
            ),
            mapping_artifact_hash=mapping_artifact_hash,
            query_trace_artifact_hash=query_trace_artifact_hash,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "candidate_id": self.candidate_id,
            "solver_id": self.solver_id,
            "verdict": self.verdict,
            "per_seed_metrics": [item.to_payload() for item in self.per_seed_metrics],
            "aggregate_metrics": self.aggregate_metrics.to_payload(),
            "mapping_artifact_hash": self.mapping_artifact_hash,
            "query_trace_artifact_hash": self.query_trace_artifact_hash,
        }


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty.")


def _require_schema_version(schema_version: str) -> None:
    if schema_version != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {schema_version!r}; "
            f"expected {EVALUATION_SCHEMA_VERSION!r}."
        )


def _require_iso8601_timestamp(timestamp: str) -> None:
    _require_non_empty(timestamp, field_name="capture_timestamp")
    normalized = timestamp.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("capture_timestamp must be ISO 8601.") from error


def _validate_dimensions(*, width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")


def _validate_seed_index(seed_index: int) -> None:
    if seed_index < 0:
        raise ValueError("seed_index must be non-negative.")


def _validate_seed_ids(seed_ids: tuple[int, ...]) -> None:
    if not seed_ids:
        raise ValueError("seed_ids must be non-empty.")
    expected_seed_ids = tuple(range(len(seed_ids)))
    if seed_ids != expected_seed_ids:
        raise ValueError(
            f"seed_ids must be contiguous starting at 0; got {list(seed_ids)}."
        )


def _validate_non_empty_strings(values: tuple[str, ...], *, field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} must be non-empty.")
    for value in values:
        _require_non_empty(value, field_name=field_name)


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _build_hash_registry(artifact_hashes: tuple[ArtifactHash, ...]) -> dict[str, str]:
    if not artifact_hashes:
        raise ValueError("artifact_hashes must be non-empty.")
    registry: dict[str, str] = {}
    for item in artifact_hashes:
        if item.artifact_name in registry:
            raise ValueError(
                f"Duplicate artifact hash entry for {item.artifact_name!r}."
            )
        registry[item.artifact_name] = item.sha256
    return registry


def _require_hash_present(
    hash_registry: dict[str, str], *, expected_hash: str, field_name: str
) -> None:
    if expected_hash not in hash_registry.values():
        raise ValueError(f"{field_name} is missing from artifact_hashes.")


def _validate_seed_analysis_manifest_entries(
    *,
    seed_analyses: tuple[SeedAnalysisManifestEntry, ...],
    dataset_version: str,
    round_id: str,
    seed_ids: tuple[int, ...],
    hash_registry: dict[str, str],
) -> None:
    if tuple(item.seed_index for item in seed_analyses) != seed_ids:
        raise ValueError("Dataset manifest is missing seed analysis coverage.")
    for item in seed_analyses:
        if item.dataset_version != dataset_version:
            raise ValueError("Dataset manifest mixes dataset versions.")
        if item.round_id != round_id:
            raise ValueError("Dataset manifest mixes round ids.")
        if (
            hash_registry.get(item.analysis_artifact_name)
            != item.analysis_artifact_hash
        ):
            raise ValueError(
                f"Dataset manifest analysis hash mismatch for seed {item.seed_index}."
            )


def _validate_viewport(viewport: Viewport) -> None:
    if viewport.x < 0 or viewport.y < 0:
        raise ValueError("Viewport coordinates must be non-negative.")
    _validate_dimensions(width=viewport.width, height=viewport.height)


def _validate_reference_tensor(
    tensor: PredictionTensor,
    *,
    width: int,
    height: int,
    tensor_name: str,
) -> None:
    if len(tensor) != height:
        raise ValueError(
            f"{tensor_name} height mismatch: expected {height}, got {len(tensor)}"
        )
    for row_index, row in enumerate(tensor):
        if len(row) != width:
            raise ValueError(
                f"{tensor_name} width mismatch on row {row_index}: "
                f"expected {width}, got {len(row)}"
            )
        for column_index, cell in enumerate(row):
            if len(cell) != CLASS_COUNT:
                raise ValueError(
                    f"{tensor_name} class count mismatch at "
                    f"({column_index}, {row_index}): expected "
                    f"{CLASS_COUNT}, got {len(cell)}"
                )
            if any(probability < 0.0 for probability in cell):
                raise ValueError(
                    f"{tensor_name} has a negative probability at "
                    f"({column_index}, {row_index})"
                )
            total = math.fsum(cell)
            if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"{tensor_name} normalization error at "
                    f"({column_index}, {row_index}): got {total}"
                )


__all__ = [
    "AggregateBenchmarkMetrics",
    "ArtifactHash",
    "BenchmarkInput",
    "BenchmarkReport",
    "CandidatePredictionBundle",
    "CandidateSeedPrediction",
    "EVALUATION_SCHEMA_VERSION",
    "FrozenDatasetManifest",
    "OrganizerSeedAnalysis",
    "QueryTraceEntry",
    "RoundQueryTrace",
    "SeedAnalysisManifestEntry",
    "SeedBenchmarkMetrics",
    "canonical_json_hash",
    "canonical_mapping_artifact_hash",
]
