from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .evaluation_contract import (
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
)
from .models import Viewport


@dataclass(frozen=True)
class OfflineReplay:
    dataset_manifest: FrozenDatasetManifest
    artifact_hashes: tuple[ArtifactHash, ...]
    query_trace: RoundQueryTrace
    analyses: tuple[OrganizerSeedAnalysis, ...]
    candidate_bundle: CandidatePredictionBundle
    benchmark_input: BenchmarkInput

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_manifest.dataset_version,
            "round_id": self.dataset_manifest.round_id,
            "candidate_id": self.candidate_bundle.candidate_id,
            "solver_id": self.candidate_bundle.solver_id,
            "artifact_hashes": [item.to_payload() for item in self.artifact_hashes],
            "manifest": self.dataset_manifest.to_payload(),
            "query_trace": self.query_trace.to_payload(),
            "analyses": [analysis.to_payload() for analysis in self.analyses],
            "candidate_bundle": self.candidate_bundle.to_payload(),
            "benchmark_input": {
                "dataset_version": (
                    self.benchmark_input.dataset_manifest.dataset_version
                ),
                "round_id": self.benchmark_input.dataset_manifest.round_id,
                "seed_ids": list(self.benchmark_input.dataset_manifest.seed_ids),
                "mapping_artifact_hash": (
                    self.benchmark_input.dataset_manifest.mapping_artifact_hash
                ),
                "query_trace_artifact_hash": (
                    self.benchmark_input.dataset_manifest.query_trace_artifact_hash
                ),
            },
        }

    def to_report_json(self) -> str:
        return render_replay_report(self)


class CandidateBundleAdapter(Protocol):
    def load_candidate_bundle(
        self,
        *,
        dataset_dir: Path,
        dataset_manifest: FrozenDatasetManifest,
        artifact_hash_registry: Mapping[str, str],
        query_trace: RoundQueryTrace,
    ) -> CandidatePredictionBundle: ...


class BenchmarkInputAssembler(Protocol):
    def build_benchmark_input(
        self,
        *,
        dataset_manifest: FrozenDatasetManifest,
        query_trace: RoundQueryTrace,
        analyses: tuple[OrganizerSeedAnalysis, ...],
        candidate_bundle: CandidatePredictionBundle,
    ) -> BenchmarkInput: ...


@dataclass(frozen=True)
class ContractBenchmarkInputAssembler:
    def build_benchmark_input(
        self,
        *,
        dataset_manifest: FrozenDatasetManifest,
        query_trace: RoundQueryTrace,
        analyses: tuple[OrganizerSeedAnalysis, ...],
        candidate_bundle: CandidatePredictionBundle,
    ) -> BenchmarkInput:
        return BenchmarkInput(
            dataset_manifest=dataset_manifest,
            query_trace=query_trace,
            analyses=analyses,
            candidate_bundle=candidate_bundle,
        )


@dataclass(frozen=True)
class FrozenPredictionCandidateAdapter:
    prediction_run_id: str
    candidate_id: str
    solver_id: str

    def load_candidate_bundle(
        self,
        *,
        dataset_dir: Path,
        dataset_manifest: FrozenDatasetManifest,
        artifact_hash_registry: Mapping[str, str],
        query_trace: RoundQueryTrace,
    ) -> CandidatePredictionBundle:
        predictions: list[CandidateSeedPrediction] = []
        for seed_index in dataset_manifest.seed_ids:
            artifact_name = (
                f"predictions/{self.prediction_run_id}/seed-{seed_index:02d}.json"
            )
            expected_hash = _require_artifact_hash(
                artifact_hash_registry,
                artifact_name=artifact_name,
            )
            payload = _load_json_file(
                dataset_dir / artifact_name,
                description=f"prediction artifact for seed {seed_index}",
            )
            actual_hash = canonical_json_hash(payload)
            if actual_hash != expected_hash:
                raise ValueError(
                    "Prediction artifact hash mismatch for seed "
                    f"{seed_index}: expected {expected_hash}, got {actual_hash}."
                )
            predictions.append(
                _parse_candidate_seed_prediction(
                    payload,
                    dataset_version=dataset_manifest.dataset_version,
                    round_id=dataset_manifest.round_id,
                    seed_index=seed_index,
                    width=dataset_manifest.map_width,
                    height=dataset_manifest.map_height,
                    prediction_artifact_hash=expected_hash,
                )
            )
        return CandidatePredictionBundle(
            dataset_version=dataset_manifest.dataset_version,
            round_id=dataset_manifest.round_id,
            solver_id=self.solver_id,
            candidate_id=self.candidate_id,
            seed_ids=dataset_manifest.seed_ids,
            width=dataset_manifest.map_width,
            height=dataset_manifest.map_height,
            mapping_artifact_hash=dataset_manifest.mapping_artifact_hash,
            query_trace_artifact_hash=query_trace.trace_artifact_hash,
            predictions=tuple(predictions),
        )


@dataclass(frozen=True)
class StaticCandidateBundleAdapter:
    candidate_bundle: CandidatePredictionBundle

    def load_candidate_bundle(
        self,
        *,
        dataset_dir: Path,
        dataset_manifest: FrozenDatasetManifest,
        artifact_hash_registry: Mapping[str, str],
        query_trace: RoundQueryTrace,
    ) -> CandidatePredictionBundle:
        del dataset_dir, artifact_hash_registry, query_trace
        if self.candidate_bundle.dataset_version != dataset_manifest.dataset_version:
            raise ValueError("Static candidate bundle mixes dataset versions.")
        if self.candidate_bundle.round_id != dataset_manifest.round_id:
            raise ValueError("Static candidate bundle mixes round ids.")
        return self.candidate_bundle


def load_offline_replay(
    *,
    dataset_dir: Path,
    candidate_bundle_adapter: CandidateBundleAdapter,
    benchmark_input_assembler: BenchmarkInputAssembler | None = None,
) -> OfflineReplay:
    manifest = _load_manifest(dataset_dir / "manifest.json")
    hashes_payload = _load_json_file(
        dataset_dir / "hashes.json",
        description="dataset artifact hashes",
    )
    artifact_hashes, artifact_hash_registry = _load_hashes_registry(
        hashes_payload,
        dataset_manifest=manifest,
    )
    _validate_mapping_artifact(
        dataset_dir=dataset_dir,
        dataset_manifest=manifest,
        artifact_hash_registry=artifact_hash_registry,
    )
    query_trace = _load_query_trace(
        dataset_dir=dataset_dir,
        dataset_manifest=manifest,
        artifact_hash_registry=artifact_hash_registry,
    )
    _validate_query_response_artifacts(
        dataset_dir=dataset_dir,
        round_id=manifest.round_id,
        query_trace=query_trace,
        artifact_hash_registry=artifact_hash_registry,
    )
    analyses = _load_analyses(
        dataset_dir=dataset_dir,
        dataset_manifest=manifest,
        artifact_hash_registry=artifact_hash_registry,
    )
    candidate_bundle = candidate_bundle_adapter.load_candidate_bundle(
        dataset_dir=dataset_dir,
        dataset_manifest=manifest,
        artifact_hash_registry=artifact_hash_registry,
        query_trace=query_trace,
    )
    assembler = (
        ContractBenchmarkInputAssembler()
        if benchmark_input_assembler is None
        else benchmark_input_assembler
    )
    benchmark_input = assembler.build_benchmark_input(
        dataset_manifest=manifest,
        query_trace=query_trace,
        analyses=analyses,
        candidate_bundle=candidate_bundle,
    )
    return OfflineReplay(
        dataset_manifest=manifest,
        artifact_hashes=artifact_hashes,
        query_trace=query_trace,
        analyses=analyses,
        candidate_bundle=candidate_bundle,
        benchmark_input=benchmark_input,
    )


def render_replay_report(replay: OfflineReplay) -> str:
    return json.dumps(replay.to_payload(), indent=2, sort_keys=True)


def _load_manifest(path: Path) -> FrozenDatasetManifest:
    payload = _load_json_file(path, description="dataset manifest")
    payload_mapping = _require_mapping(payload, context="Dataset manifest")
    artifact_hashes = _parse_artifact_hashes(
        payload_mapping.get("artifact_hashes"),
        context="Dataset manifest artifact_hashes",
    )
    seed_analyses = _parse_seed_analyses(
        payload_mapping.get("seed_analyses"),
        context="Dataset manifest seed_analyses",
    )
    seed_ids = _parse_int_tuple(payload_mapping.get("seed_ids"), field_name="seed_ids")
    source_endpoints = _parse_str_tuple(
        payload_mapping.get("source_endpoints"),
        field_name="source_endpoints",
    )
    return FrozenDatasetManifest(
        schema_version=_require_str(
            payload_mapping.get("schema_version"),
            field_name="schema_version",
        ),
        dataset_version=_require_str(
            payload_mapping.get("dataset_version"),
            field_name="dataset_version",
        ),
        capture_timestamp=_require_str(
            payload_mapping.get("capture_timestamp"),
            field_name="capture_timestamp",
        ),
        round_id=_require_str(payload_mapping.get("round_id"), field_name="round_id"),
        seed_ids=seed_ids,
        source_endpoints=source_endpoints,
        artifact_hashes=artifact_hashes,
        solver_id=_require_str(
            payload_mapping.get("solver_id"),
            field_name="solver_id",
        ),
        map_width=_require_int(
            payload_mapping.get("map_width"),
            field_name="map_width",
        ),
        map_height=_require_int(
            payload_mapping.get("map_height"),
            field_name="map_height",
        ),
        mapping_artifact_hash=_require_str(
            payload_mapping.get("mapping_artifact_hash"),
            field_name="mapping_artifact_hash",
        ),
        query_trace_artifact_hash=_require_str(
            payload_mapping.get("query_trace_artifact_hash"),
            field_name="query_trace_artifact_hash",
        ),
        seed_analyses=seed_analyses,
    )


def _load_hashes_registry(
    payload: object, *, dataset_manifest: FrozenDatasetManifest
) -> tuple[tuple[ArtifactHash, ...], dict[str, str]]:
    payload_mapping = _require_mapping(payload, context="Artifact hashes payload")
    dataset_version = _require_str(
        payload_mapping.get("dataset_version"),
        field_name="dataset_version",
    )
    round_id = _require_str(payload_mapping.get("round_id"), field_name="round_id")
    if dataset_version != dataset_manifest.dataset_version:
        raise ValueError("hashes.json dataset_version does not match manifest.json.")
    if round_id != dataset_manifest.round_id:
        raise ValueError("hashes.json round_id does not match manifest.json.")
    artifact_hashes = _parse_artifact_hashes(
        payload_mapping.get("artifact_hashes"),
        context="Artifact hashes payload artifact_hashes",
    )
    artifact_hash_registry = _build_artifact_hash_registry(artifact_hashes)
    manifest_hash_registry = _build_artifact_hash_registry(
        dataset_manifest.artifact_hashes
    )
    if artifact_hash_registry != manifest_hash_registry:
        raise ValueError(
            "hashes.json artifact registry does not match manifest artifact_hashes."
        )
    return (
        tuple(sorted(artifact_hashes, key=lambda item: item.artifact_name)),
        artifact_hash_registry,
    )


def _load_query_trace(
    *,
    dataset_dir: Path,
    dataset_manifest: FrozenDatasetManifest,
    artifact_hash_registry: Mapping[str, str],
) -> RoundQueryTrace:
    payload = _load_json_file(
        dataset_dir / "query-trace.json",
        description="round query trace",
    )
    expected_hash = _require_artifact_hash(
        artifact_hash_registry,
        artifact_name="query-trace.json",
    )
    if expected_hash != dataset_manifest.query_trace_artifact_hash:
        raise ValueError("query-trace.json hash does not match manifest.json.")
    payload_mapping = _require_mapping(payload, context="Round query trace")
    entries_payload = payload_mapping.get("entries")
    if not isinstance(entries_payload, list):
        raise ValueError("Round query trace entries must be a list.")
    entries = tuple(
        _parse_query_trace_entry(item, entry_index=index)
        for index, item in enumerate(entries_payload)
    )
    query_trace = RoundQueryTrace(
        schema_version=_require_str(
            payload_mapping.get("schema_version"),
            field_name="schema_version",
        ),
        dataset_version=_require_str(
            payload_mapping.get("dataset_version"),
            field_name="dataset_version",
        ),
        round_id=_require_str(payload_mapping.get("round_id"), field_name="round_id"),
        seed_ids=_parse_int_tuple(
            payload_mapping.get("seed_ids"),
            field_name="seed_ids",
        ),
        entries=entries,
        trace_artifact_hash=_require_str(
            payload_mapping.get("trace_artifact_hash"),
            field_name="trace_artifact_hash",
        ),
        query_budget=_require_int(
            payload_mapping.get("query_budget"),
            field_name="query_budget",
        ),
    )
    actual_hash = canonical_json_hash(
        {
            "schema_version": query_trace.schema_version,
            "dataset_version": query_trace.dataset_version,
            "round_id": query_trace.round_id,
            "seed_ids": list(query_trace.seed_ids),
            "entries": [entry.to_payload() for entry in query_trace.entries],
            "query_budget": query_trace.query_budget,
        }
    )
    if query_trace.trace_artifact_hash != expected_hash:
        raise ValueError("Round query trace payload hash does not match hashes.json.")
    if actual_hash != expected_hash:
        raise ValueError(
            "Round query trace logical hash mismatch: expected "
            f"{expected_hash}, got {actual_hash}."
        )
    return query_trace


def _validate_mapping_artifact(
    *,
    dataset_dir: Path,
    dataset_manifest: FrozenDatasetManifest,
    artifact_hash_registry: Mapping[str, str],
) -> None:
    artifact_name = "mapping/terrain_mapping.json"
    expected_hash = _require_artifact_hash(
        artifact_hash_registry,
        artifact_name=artifact_name,
    )
    if expected_hash != dataset_manifest.mapping_artifact_hash:
        raise ValueError("Mapping artifact hash does not match manifest.json.")
    payload = _load_json_file(
        dataset_dir / artifact_name,
        description="terrain mapping artifact",
    )
    actual_hash = canonical_json_hash(payload)
    if actual_hash != expected_hash:
        raise ValueError(
            "Mapping artifact hash mismatch: expected "
            f"{expected_hash}, got {actual_hash}."
        )


def _validate_query_response_artifacts(
    *,
    dataset_dir: Path,
    round_id: str,
    query_trace: RoundQueryTrace,
    artifact_hash_registry: Mapping[str, str],
) -> None:
    for entry in query_trace.entries:
        artifact_name = _resolve_query_response_artifact_name(
            round_id=round_id,
            response_artifact_hash=entry.response_artifact_hash,
            artifact_hash_registry=artifact_hash_registry,
        )
        payload = _load_json_file(
            dataset_dir / artifact_name,
            description=(
                "query response artifact for seed "
                f"{entry.seed_index} query {entry.query_index}"
            ),
        )
        actual_hash = canonical_json_hash(payload)
        if actual_hash != entry.response_artifact_hash:
            raise ValueError(
                "Query response artifact hash mismatch for query "
                f"{entry.query_index}: expected {entry.response_artifact_hash}, "
                f"got {actual_hash}."
            )


def _load_analyses(
    *,
    dataset_dir: Path,
    dataset_manifest: FrozenDatasetManifest,
    artifact_hash_registry: Mapping[str, str],
) -> tuple[OrganizerSeedAnalysis, ...]:
    analyses: list[OrganizerSeedAnalysis] = []
    for manifest_entry in dataset_manifest.seed_analyses:
        expected_hash = _require_artifact_hash(
            artifact_hash_registry,
            artifact_name=manifest_entry.analysis_artifact_name,
        )
        if expected_hash != manifest_entry.analysis_artifact_hash:
            raise ValueError(
                f"Analysis hash registry mismatch for seed {manifest_entry.seed_index}."
            )
        payload = _load_json_file(
            dataset_dir / manifest_entry.analysis_artifact_name,
            description=f"analysis artifact for seed {manifest_entry.seed_index}",
        )
        actual_hash = canonical_json_hash(payload)
        if actual_hash != expected_hash:
            raise ValueError(
                "Analysis artifact hash mismatch for seed "
                f"{manifest_entry.seed_index}: expected {expected_hash}, "
                f"got {actual_hash}."
            )
        analyses.append(
            _parse_analysis_payload(
                payload,
                dataset_version=dataset_manifest.dataset_version,
                round_id=dataset_manifest.round_id,
                seed_index=manifest_entry.seed_index,
                analysis_artifact_hash=expected_hash,
            )
        )
    return tuple(analyses)


def _parse_artifact_hashes(
    payload: object, *, context: str
) -> tuple[ArtifactHash, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{context} must be a list.")
    return tuple(
        ArtifactHash(
            artifact_name=_require_str(
                _require_mapping(item, context=f"{context} entry").get("artifact_name"),
                field_name="artifact_name",
            ),
            sha256=_require_str(
                _require_mapping(item, context=f"{context} entry").get("sha256"),
                field_name="sha256",
            ),
        )
        for item in payload
    )


def _parse_seed_analyses(
    payload: object, *, context: str
) -> tuple[SeedAnalysisManifestEntry, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{context} must be a list.")
    entries: list[SeedAnalysisManifestEntry] = []
    for item in payload:
        item_mapping = _require_mapping(item, context=f"{context} entry")
        entries.append(
            SeedAnalysisManifestEntry(
                dataset_version=_require_str(
                    item_mapping.get("dataset_version"),
                    field_name="dataset_version",
                ),
                round_id=_require_str(
                    item_mapping.get("round_id"),
                    field_name="round_id",
                ),
                seed_index=_require_int(
                    item_mapping.get("seed_index"),
                    field_name="seed_index",
                ),
                analysis_artifact_name=_require_str(
                    item_mapping.get("analysis_artifact_name"),
                    field_name="analysis_artifact_name",
                ),
                analysis_artifact_hash=_require_str(
                    item_mapping.get("analysis_artifact_hash"),
                    field_name="analysis_artifact_hash",
                ),
            )
        )
    return tuple(entries)


def _parse_query_trace_entry(payload: object, *, entry_index: int) -> QueryTraceEntry:
    payload_mapping = _require_mapping(
        payload,
        context=f"Query trace entry {entry_index}",
    )
    viewport_payload = _require_mapping(
        payload_mapping.get("viewport"),
        context=f"Query trace entry {entry_index} viewport",
    )
    viewport = Viewport(
        x=_require_int(viewport_payload.get("x"), field_name="viewport.x"),
        y=_require_int(viewport_payload.get("y"), field_name="viewport.y"),
        width=_require_int(viewport_payload.get("width"), field_name="viewport.width"),
        height=_require_int(
            viewport_payload.get("height"),
            field_name="viewport.height",
        ),
    )
    return QueryTraceEntry(
        dataset_version=_require_str(
            payload_mapping.get("dataset_version"),
            field_name="dataset_version",
        ),
        round_id=_require_str(payload_mapping.get("round_id"), field_name="round_id"),
        seed_index=_require_int(
            payload_mapping.get("seed_index"),
            field_name="seed_index",
        ),
        query_index=_require_int(
            payload_mapping.get("query_index"),
            field_name="query_index",
        ),
        viewport=viewport,
        response_artifact_hash=_require_str(
            payload_mapping.get("response_artifact_hash"),
            field_name="response_artifact_hash",
        ),
        queries_used=_optional_int(payload_mapping.get("queries_used")),
        queries_max=_optional_int(payload_mapping.get("queries_max")),
    )


def _parse_analysis_payload(
    payload: object,
    *,
    dataset_version: str,
    round_id: str,
    seed_index: int,
    analysis_artifact_hash: str,
) -> OrganizerSeedAnalysis:
    payload_mapping = _require_mapping(payload, context="Analysis payload")
    width = _require_int(payload_mapping.get("width"), field_name="analysis.width")
    height = _require_int(payload_mapping.get("height"), field_name="analysis.height")
    raw_prediction = payload_mapping.get("prediction")
    submitted_prediction = (
        None
        if raw_prediction is None
        else _parse_prediction_tensor(
            raw_prediction,
            tensor_name="analysis.prediction",
        )
    )
    organizer_score = _optional_float(payload_mapping.get("score"))
    return OrganizerSeedAnalysis(
        dataset_version=dataset_version,
        round_id=round_id,
        seed_index=seed_index,
        width=width,
        height=height,
        analysis_artifact_hash=analysis_artifact_hash,
        ground_truth=_parse_prediction_tensor(
            payload_mapping.get("ground_truth"),
            tensor_name="analysis.ground_truth",
        ),
        submitted_prediction=submitted_prediction,
        organizer_score=organizer_score,
    )


def _parse_candidate_seed_prediction(
    payload: object,
    *,
    dataset_version: str,
    round_id: str,
    seed_index: int,
    width: int,
    height: int,
    prediction_artifact_hash: str,
) -> CandidateSeedPrediction:
    payload_mapping = _require_mapping(payload, context="Prediction payload")
    payload_round_id = _require_str(
        payload_mapping.get("round_id"),
        field_name="prediction.round_id",
    )
    if payload_round_id != round_id:
        raise ValueError(
            "Prediction payload round_id mismatch for seed "
            f"{seed_index}: expected {round_id!r}, got {payload_round_id!r}."
        )
    payload_seed_index = _require_int(
        payload_mapping.get("seed_index"),
        field_name="prediction.seed_index",
    )
    if payload_seed_index != seed_index:
        raise ValueError(
            "Prediction payload seed_index mismatch: expected "
            f"{seed_index}, got {payload_seed_index}."
        )
    return CandidateSeedPrediction(
        dataset_version=dataset_version,
        round_id=round_id,
        seed_index=seed_index,
        width=width,
        height=height,
        prediction=_parse_prediction_tensor(
            payload_mapping.get("prediction"),
            tensor_name=f"prediction seed {seed_index}",
        ),
        prediction_artifact_hash=prediction_artifact_hash,
    )


def _parse_prediction_tensor(
    payload: object, *, tensor_name: str
) -> list[list[list[float]]]:
    if not isinstance(payload, list):
        raise ValueError(f"{tensor_name} must be a nested list.")
    tensor: list[list[list[float]]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ValueError(f"{tensor_name} rows must be lists.")
        parsed_row: list[list[float]] = []
        for cell in row:
            if not isinstance(cell, list):
                raise ValueError(f"{tensor_name} cells must be lists.")
            parsed_row.append(
                [
                    _require_float(probability, field_name=tensor_name)
                    for probability in cell
                ]
            )
        tensor.append(parsed_row)
    return tensor


def _resolve_query_response_artifact_name(
    *,
    round_id: str,
    response_artifact_hash: str,
    artifact_hash_registry: Mapping[str, str],
) -> str:
    prefix = f"queries/{round_id}/"
    matches = sorted(
        artifact_name
        for artifact_name, sha256 in artifact_hash_registry.items()
        if sha256 == response_artifact_hash and artifact_name.startswith(prefix)
    )
    if not matches:
        raise ValueError(
            "Missing query response artifact for hash "
            f"{response_artifact_hash} in round {round_id!r}."
        )
    if len(matches) != 1:
        raise ValueError(
            "Ambiguous query response artifact hash "
            f"{response_artifact_hash}; expected exactly one frozen artifact."
        )
    return matches[0]


def _load_json_file(path: Path, *, description: str) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{description.capitalize()} is not valid JSON: {path}"
        ) from error
    return payload


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _parse_int_tuple(payload: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{field_name} must be a list.")
    return tuple(_require_int(item, field_name=field_name) for item in payload)


def _parse_str_tuple(payload: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{field_name} must be a list.")
    return tuple(_require_str(item, field_name=field_name) for item in payload)


def _build_artifact_hash_registry(
    artifact_hashes: tuple[ArtifactHash, ...],
) -> dict[str, str]:
    registry: dict[str, str] = {}
    for item in artifact_hashes:
        if item.artifact_name in registry:
            raise ValueError(
                f"Duplicate artifact hash entry for {item.artifact_name!r}."
            )
        registry[item.artifact_name] = item.sha256
    return registry


def _require_artifact_hash(
    artifact_hash_registry: Mapping[str, str], *, artifact_name: str
) -> str:
    artifact_hash = artifact_hash_registry.get(artifact_name)
    if artifact_hash is None:
        raise ValueError(f"Missing artifact hash entry for {artifact_name!r}.")
    return artifact_hash


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _require_int(value, field_name="integer field")


def _require_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _require_float(value, field_name="numeric field")


__all__ = [
    "BenchmarkInputAssembler",
    "CandidateBundleAdapter",
    "ContractBenchmarkInputAssembler",
    "FrozenPredictionCandidateAdapter",
    "OfflineReplay",
    "StaticCandidateBundleAdapter",
    "load_offline_replay",
    "render_replay_report",
]
