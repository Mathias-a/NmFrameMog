from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

from .cache import LocalCache
from .contract import DEFAULT_QUERY_BUDGET, canonical_mapping_artifact
from .evaluation_contract import (
    EVALUATION_SCHEMA_VERSION,
    ArtifactHash,
    FrozenDatasetManifest,
    OrganizerSeedAnalysis,
    QueryTraceEntry,
    RoundQueryTrace,
    SeedAnalysisManifestEntry,
    canonical_json_hash,
    canonical_mapping_artifact_hash,
)
from .models import Viewport
from .pipeline import parse_round_detail_payload

HISTORY_REFRESH_SCHEMA_VERSION = "astar-history-refresh-v1"
CURATED_BENCHMARK_SCHEMA_VERSION = "astar-curated-benchmark-v1"


class DatasetRefreshClient(Protocol):
    def get_round_detail(self, round_id: str) -> object: ...

    def get_analysis(self, *, round_id: str, seed_index: int) -> object: ...


class DatasetHistoryRefreshClient(DatasetRefreshClient, Protocol):
    def list_rounds(self) -> object: ...


@dataclass(frozen=True)
class FrozenDatasetSnapshot:
    dataset_version: str
    dataset_dir: Path
    manifest_path: Path
    hashes_path: Path
    query_trace_path: Path
    manifest: FrozenDatasetManifest
    query_trace: RoundQueryTrace


@dataclass(frozen=True)
class HistoryRoundCapture:
    snapshot_version: str
    round_id: str
    seed_ids: tuple[int, ...]
    round_manifest_artifact_name: str
    round_manifest_artifact_hash: str
    query_trace_artifact_name: str
    query_trace_artifact_hash: str
    prediction_run_id: str
    source_endpoints: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.snapshot_version, field_name="snapshot_version")
        _require_non_empty(self.round_id, field_name="round_id")
        _require_non_empty(
            self.round_manifest_artifact_name,
            field_name="round_manifest_artifact_name",
        )
        _require_non_empty(
            self.query_trace_artifact_name,
            field_name="query_trace_artifact_name",
        )
        _require_non_empty(self.prediction_run_id, field_name="prediction_run_id")
        if not self.seed_ids:
            raise ValueError("seed_ids must be non-empty.")
        _validate_sha256(
            self.round_manifest_artifact_hash,
            field_name="round_manifest_artifact_hash",
        )
        _validate_sha256(
            self.query_trace_artifact_hash,
            field_name="query_trace_artifact_hash",
        )
        for endpoint in self.source_endpoints:
            _require_non_empty(endpoint, field_name="source_endpoints")

    def to_payload(self) -> dict[str, object]:
        return {
            "snapshot_version": self.snapshot_version,
            "round_id": self.round_id,
            "seed_ids": list(self.seed_ids),
            "round_manifest_artifact_name": self.round_manifest_artifact_name,
            "round_manifest_artifact_hash": self.round_manifest_artifact_hash,
            "query_trace_artifact_name": self.query_trace_artifact_name,
            "query_trace_artifact_hash": self.query_trace_artifact_hash,
            "prediction_run_id": self.prediction_run_id,
            "source_endpoints": list(self.source_endpoints),
        }


@dataclass(frozen=True)
class HistoryRefreshManifest:
    snapshot_version: str
    capture_timestamp: str
    round_ids: tuple[str, ...]
    artifact_hashes: tuple[ArtifactHash, ...]
    rounds: tuple[HistoryRoundCapture, ...]
    solver_id: str
    schema_version: str = HISTORY_REFRESH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty(self.snapshot_version, field_name="snapshot_version")
        _require_iso8601_timestamp(self.capture_timestamp)
        _require_non_empty(self.solver_id, field_name="solver_id")
        if self.schema_version != HISTORY_REFRESH_SCHEMA_VERSION:
            raise ValueError("Unsupported history refresh manifest schema_version.")
        if not self.rounds:
            raise ValueError(
                "History refresh manifest must include at least one round."
            )
        if tuple(entry.round_id for entry in self.rounds) != self.round_ids:
            raise ValueError("History refresh manifest round_ids do not match rounds.")
        hash_registry = _build_hash_registry(self.artifact_hashes)
        for entry in self.rounds:
            if entry.snapshot_version != self.snapshot_version:
                raise ValueError("History refresh manifest mixes snapshot versions.")
            _require_registry_hash(
                hash_registry,
                artifact_name=entry.round_manifest_artifact_name,
                expected_hash=entry.round_manifest_artifact_hash,
            )
            _require_registry_hash(
                hash_registry,
                artifact_name=entry.query_trace_artifact_name,
                expected_hash=entry.query_trace_artifact_hash,
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "snapshot_version": self.snapshot_version,
            "capture_timestamp": self.capture_timestamp,
            "round_ids": list(self.round_ids),
            "artifact_hashes": [item.to_payload() for item in self.artifact_hashes],
            "rounds": [item.to_payload() for item in self.rounds],
            "solver_id": self.solver_id,
        }


@dataclass(frozen=True)
class HistoryRefreshSnapshot:
    snapshot_version: str
    snapshot_dir: Path
    manifest_path: Path
    hashes_path: Path
    manifest: HistoryRefreshManifest


@dataclass(frozen=True)
class CuratedRoundManifestEntry:
    dataset_version: str
    round_id: str
    seed_ids: tuple[int, ...]
    source_snapshot_version: str
    round_manifest_artifact_name: str
    round_manifest_artifact_hash: str
    query_trace_artifact_name: str
    query_trace_artifact_hash: str
    prediction_run_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_non_empty(self.round_id, field_name="round_id")
        _require_non_empty(
            self.source_snapshot_version,
            field_name="source_snapshot_version",
        )
        _require_non_empty(
            self.round_manifest_artifact_name,
            field_name="round_manifest_artifact_name",
        )
        _require_non_empty(
            self.query_trace_artifact_name,
            field_name="query_trace_artifact_name",
        )
        _require_non_empty(self.prediction_run_id, field_name="prediction_run_id")
        if not self.seed_ids:
            raise ValueError("seed_ids must be non-empty.")
        _validate_sha256(
            self.round_manifest_artifact_hash,
            field_name="round_manifest_artifact_hash",
        )
        _validate_sha256(
            self.query_trace_artifact_hash,
            field_name="query_trace_artifact_hash",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "round_id": self.round_id,
            "seed_ids": list(self.seed_ids),
            "source_snapshot_version": self.source_snapshot_version,
            "round_manifest_artifact_name": self.round_manifest_artifact_name,
            "round_manifest_artifact_hash": self.round_manifest_artifact_hash,
            "query_trace_artifact_name": self.query_trace_artifact_name,
            "query_trace_artifact_hash": self.query_trace_artifact_hash,
            "prediction_run_id": self.prediction_run_id,
        }


@dataclass(frozen=True)
class CuratedBenchmarkManifest:
    dataset_version: str
    capture_timestamp: str
    round_ids: tuple[str, ...]
    source_snapshot_versions: tuple[str, ...]
    artifact_hashes: tuple[ArtifactHash, ...]
    rounds: tuple[CuratedRoundManifestEntry, ...]
    solver_id: str
    schema_version: str = CURATED_BENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty(self.dataset_version, field_name="dataset_version")
        _require_iso8601_timestamp(self.capture_timestamp)
        _require_non_empty(self.solver_id, field_name="solver_id")
        if self.schema_version != CURATED_BENCHMARK_SCHEMA_VERSION:
            raise ValueError("Unsupported curated benchmark manifest schema_version.")
        if not self.source_snapshot_versions:
            raise ValueError(
                "Curated benchmark manifest must reference source snapshots."
            )
        if not self.rounds:
            raise ValueError(
                "Curated benchmark manifest must include at least one round."
            )
        if tuple(entry.round_id for entry in self.rounds) != self.round_ids:
            raise ValueError(
                "Curated benchmark manifest round_ids do not match rounds."
            )
        hash_registry = _build_hash_registry(self.artifact_hashes)
        for entry in self.rounds:
            if entry.dataset_version != self.dataset_version:
                raise ValueError("Curated benchmark manifest mixes dataset versions.")
            _require_registry_hash(
                hash_registry,
                artifact_name=entry.round_manifest_artifact_name,
                expected_hash=entry.round_manifest_artifact_hash,
            )
            _require_registry_hash(
                hash_registry,
                artifact_name=entry.query_trace_artifact_name,
                expected_hash=entry.query_trace_artifact_hash,
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "capture_timestamp": self.capture_timestamp,
            "round_ids": list(self.round_ids),
            "source_snapshot_versions": list(self.source_snapshot_versions),
            "artifact_hashes": [item.to_payload() for item in self.artifact_hashes],
            "rounds": [item.to_payload() for item in self.rounds],
            "solver_id": self.solver_id,
        }


@dataclass(frozen=True)
class CuratedBenchmarkDataset:
    dataset_version: str
    dataset_dir: Path
    manifest_path: Path
    hashes_path: Path
    manifest: CuratedBenchmarkManifest


@dataclass(frozen=True)
class _CapturedRoundArtifacts:
    manifest: FrozenDatasetManifest
    query_trace: RoundQueryTrace
    prediction_run_id: str


def refresh_dataset_snapshot(
    *,
    cache: LocalCache,
    client: DatasetRefreshClient,
    round_id: str,
    dataset_version: str,
    solver_id: str = "astar-dataset-refresh",
    capture_timestamp: datetime | None = None,
) -> FrozenDatasetSnapshot:
    _require_non_empty(dataset_version, field_name="dataset_version")
    _require_non_empty(round_id, field_name="round_id")
    _require_non_empty(solver_id, field_name="solver_id")

    cache.ensure()
    datasets_dir = cache.root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = datasets_dir / dataset_version
    if dataset_dir.exists():
        raise FileExistsError(
            f"Frozen dataset version {dataset_version!r} already exists."
        )

    temp_dataset_dir = Path(
        tempfile.mkdtemp(prefix=f".{dataset_version}.", dir=datasets_dir)
    )
    capture_timestamp_text = _format_capture_timestamp(capture_timestamp)
    artifact_hashes: dict[str, str] = {}

    try:
        _record_json_artifact(
            artifact_hashes=artifact_hashes,
            dataset_root=temp_dataset_dir,
            path=LocalCache(temp_dataset_dir).mapping_artifact_path(),
            payload=canonical_mapping_artifact(),
            precomputed_hash=canonical_mapping_artifact_hash(),
        )

        captured_round = _capture_round_artifacts(
            cache=cache,
            client=client,
            dataset_root=temp_dataset_dir,
            artifact_hashes=artifact_hashes,
            dataset_version=dataset_version,
            round_id=round_id,
            solver_id=solver_id,
            capture_timestamp_text=capture_timestamp_text,
            round_detail_path=temp_dataset_dir / "rounds" / f"{round_id}.json",
            query_trace_path=temp_dataset_dir / "query-trace.json",
            seed_metadata_dir=temp_dataset_dir / "seed-metadata",
        )
        hashes_path = temp_dataset_dir / "hashes.json"
        _save_json(
            hashes_path,
            {
                "dataset_version": dataset_version,
                "round_id": round_id,
                "artifact_hashes": [
                    item.to_payload()
                    for item in captured_round.manifest.artifact_hashes
                ],
            },
        )
        manifest_path = temp_dataset_dir / "manifest.json"
        _save_json(manifest_path, captured_round.manifest.to_payload())

        temp_dataset_dir.rename(dataset_dir)
        return FrozenDatasetSnapshot(
            dataset_version=dataset_version,
            dataset_dir=dataset_dir,
            manifest_path=dataset_dir / "manifest.json",
            hashes_path=dataset_dir / "hashes.json",
            query_trace_path=dataset_dir / "query-trace.json",
            manifest=captured_round.manifest,
            query_trace=captured_round.query_trace,
        )
    except Exception:
        shutil.rmtree(temp_dataset_dir, ignore_errors=True)
        raise


def refresh_history_snapshot(
    *,
    cache: LocalCache,
    client: DatasetHistoryRefreshClient,
    snapshot_version: str,
    round_ids: Sequence[str] | None = None,
    solver_id: str = "astar-history-refresh",
    capture_timestamp: datetime | None = None,
) -> HistoryRefreshSnapshot:
    _require_non_empty(snapshot_version, field_name="snapshot_version")
    _require_non_empty(solver_id, field_name="solver_id")

    cache.ensure()
    raw_history_dir = _history_raw_dir(cache)
    raw_history_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = raw_history_dir / snapshot_version
    if snapshot_dir.exists():
        raise FileExistsError(
            f"History snapshot version {snapshot_version!r} already exists."
        )

    temp_snapshot_dir = Path(
        tempfile.mkdtemp(prefix=f".{snapshot_version}.", dir=raw_history_dir)
    )
    capture_timestamp_text = _format_capture_timestamp(capture_timestamp)
    artifact_hashes: dict[str, str] = {}

    try:
        _record_json_artifact(
            artifact_hashes=artifact_hashes,
            dataset_root=temp_snapshot_dir,
            path=temp_snapshot_dir / "mapping" / "terrain_mapping.json",
            payload=canonical_mapping_artifact(),
            precomputed_hash=canonical_mapping_artifact_hash(),
        )

        selected_round_ids = _select_history_round_ids(
            cache=cache,
            client=client,
            requested_round_ids=round_ids,
        )
        round_entries: list[HistoryRoundCapture] = []
        for round_id in selected_round_ids:
            captured_round = _capture_round_artifacts(
                cache=cache,
                client=client,
                dataset_root=temp_snapshot_dir,
                artifact_hashes=artifact_hashes,
                dataset_version=snapshot_version,
                round_id=round_id,
                solver_id=solver_id,
                capture_timestamp_text=capture_timestamp_text,
                round_detail_path=temp_snapshot_dir / "rounds" / f"{round_id}.json",
                query_trace_path=temp_snapshot_dir
                / "query-traces"
                / f"{round_id}.json",
                seed_metadata_dir=temp_snapshot_dir / "seed-metadata" / round_id,
            )
            round_manifest_path = (
                temp_snapshot_dir / "round-manifests" / f"{round_id}.json"
            )
            round_manifest_hash = _record_json_artifact(
                artifact_hashes=artifact_hashes,
                dataset_root=temp_snapshot_dir,
                path=round_manifest_path,
                payload=captured_round.manifest.to_payload(),
            )
            round_entries.append(
                HistoryRoundCapture(
                    snapshot_version=snapshot_version,
                    round_id=round_id,
                    seed_ids=captured_round.manifest.seed_ids,
                    round_manifest_artifact_name=_artifact_name(
                        round_manifest_path,
                        dataset_root=temp_snapshot_dir,
                    ),
                    round_manifest_artifact_hash=round_manifest_hash,
                    query_trace_artifact_name=_artifact_name(
                        temp_snapshot_dir / "query-traces" / f"{round_id}.json",
                        dataset_root=temp_snapshot_dir,
                    ),
                    query_trace_artifact_hash=captured_round.query_trace.trace_artifact_hash,
                    prediction_run_id=captured_round.prediction_run_id,
                    source_endpoints=captured_round.manifest.source_endpoints,
                )
            )

        manifest = HistoryRefreshManifest(
            snapshot_version=snapshot_version,
            capture_timestamp=capture_timestamp_text,
            round_ids=tuple(selected_round_ids),
            artifact_hashes=tuple(
                ArtifactHash(artifact_name=name, sha256=sha256)
                for name, sha256 in sorted(artifact_hashes.items())
            ),
            rounds=tuple(round_entries),
            solver_id=solver_id,
        )
        hashes_path = temp_snapshot_dir / "hashes.json"
        _save_json(
            hashes_path,
            {
                "snapshot_version": snapshot_version,
                "round_ids": list(selected_round_ids),
                "artifact_hashes": [
                    item.to_payload() for item in manifest.artifact_hashes
                ],
            },
        )
        manifest_path = temp_snapshot_dir / "manifest.json"
        _save_json(manifest_path, manifest.to_payload())

        temp_snapshot_dir.rename(snapshot_dir)
        return HistoryRefreshSnapshot(
            snapshot_version=snapshot_version,
            snapshot_dir=snapshot_dir,
            manifest_path=snapshot_dir / "manifest.json",
            hashes_path=snapshot_dir / "hashes.json",
            manifest=manifest,
        )
    except Exception:
        shutil.rmtree(temp_snapshot_dir, ignore_errors=True)
        raise


def build_curated_history_dataset(
    *,
    cache: LocalCache,
    source_snapshot_versions: Sequence[str],
    dataset_version: str,
    round_ids: Sequence[str] | None = None,
    solver_id: str = "astar-history-curate",
    capture_timestamp: datetime | None = None,
) -> CuratedBenchmarkDataset:
    _require_non_empty(dataset_version, field_name="dataset_version")
    _require_non_empty(solver_id, field_name="solver_id")
    if not source_snapshot_versions:
        raise ValueError("source_snapshot_versions must be non-empty.")

    cache.ensure()
    datasets_dir = cache.root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = datasets_dir / dataset_version
    if dataset_dir.exists():
        raise FileExistsError(
            f"Frozen dataset version {dataset_version!r} already exists."
        )

    temp_dataset_dir = Path(
        tempfile.mkdtemp(prefix=f".{dataset_version}.", dir=datasets_dir)
    )
    capture_timestamp_text = _format_capture_timestamp(capture_timestamp)
    artifact_hashes: dict[str, str] = {}

    try:
        _record_json_artifact(
            artifact_hashes=artifact_hashes,
            dataset_root=temp_dataset_dir,
            path=temp_dataset_dir / "mapping" / "terrain_mapping.json",
            payload=canonical_mapping_artifact(),
            precomputed_hash=canonical_mapping_artifact_hash(),
        )

        round_sources = _select_round_sources(
            cache=cache,
            source_snapshot_versions=source_snapshot_versions,
            requested_round_ids=round_ids,
        )
        curated_rounds: list[CuratedRoundManifestEntry] = []
        for round_source in round_sources:
            source_root = round_source.snapshot_dir
            round_id = round_source.capture.round_id
            source_manifest = _load_frozen_dataset_manifest(
                source_root / round_source.capture.round_manifest_artifact_name
            )
            round_artifact_hashes: dict[str, str] = {
                "mapping/terrain_mapping.json": canonical_mapping_artifact_hash()
            }

            _copy_json_artifact(
                source_root=source_root,
                source_path=source_root / "rounds" / f"{round_id}.json",
                destination_root=temp_dataset_dir,
                destination_path=temp_dataset_dir / "rounds" / f"{round_id}.json",
                artifact_hashes=artifact_hashes,
                round_artifact_hashes=round_artifact_hashes,
            )
            _copy_round_directory(
                source_root=source_root,
                source_dir=source_root / "queries" / round_id,
                destination_root=temp_dataset_dir,
                destination_dir=temp_dataset_dir / "queries" / round_id,
                artifact_hashes=artifact_hashes,
                round_artifact_hashes=round_artifact_hashes,
            )
            _copy_round_directory(
                source_root=source_root,
                source_dir=source_root / "analysis" / round_id,
                destination_root=temp_dataset_dir,
                destination_dir=temp_dataset_dir / "analysis" / round_id,
                artifact_hashes=artifact_hashes,
                round_artifact_hashes=round_artifact_hashes,
            )
            _copy_round_directory(
                source_root=source_root,
                source_dir=source_root
                / "predictions"
                / round_source.capture.prediction_run_id,
                destination_root=temp_dataset_dir,
                destination_dir=temp_dataset_dir
                / "predictions"
                / round_source.capture.prediction_run_id,
                artifact_hashes=artifact_hashes,
                round_artifact_hashes=round_artifact_hashes,
            )
            _copy_round_directory(
                source_root=source_root,
                source_dir=source_root / "seed-metadata" / round_id,
                destination_root=temp_dataset_dir,
                destination_dir=temp_dataset_dir / "seed-metadata" / round_id,
                artifact_hashes=artifact_hashes,
                round_artifact_hashes=round_artifact_hashes,
            )

            source_query_trace = _load_round_query_trace(
                source_root / round_source.capture.query_trace_artifact_name
            )
            curated_entries = tuple(
                QueryTraceEntry(
                    dataset_version=dataset_version,
                    round_id=entry.round_id,
                    seed_index=entry.seed_index,
                    query_index=entry.query_index,
                    viewport=entry.viewport,
                    response_artifact_hash=entry.response_artifact_hash,
                    queries_used=entry.queries_used,
                    queries_max=entry.queries_max,
                )
                for entry in source_query_trace.entries
            )
            curated_query_trace_hash = canonical_json_hash(
                _query_trace_hash_payload(
                    dataset_version=dataset_version,
                    round_id=round_id,
                    seed_ids=source_query_trace.seed_ids,
                    entries=curated_entries,
                    query_budget=source_query_trace.query_budget,
                )
            )
            curated_query_trace = RoundQueryTrace(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_ids=source_query_trace.seed_ids,
                entries=curated_entries,
                trace_artifact_hash=curated_query_trace_hash,
                query_budget=source_query_trace.query_budget,
            )
            query_trace_path = temp_dataset_dir / "query-traces" / f"{round_id}.json"
            _record_json_artifact(
                artifact_hashes=artifact_hashes,
                dataset_root=temp_dataset_dir,
                path=query_trace_path,
                payload=curated_query_trace.to_payload(),
                precomputed_hash=curated_query_trace_hash,
                additional_registries=(round_artifact_hashes,),
            )

            curated_seed_analyses = tuple(
                SeedAnalysisManifestEntry(
                    dataset_version=dataset_version,
                    round_id=entry.round_id,
                    seed_index=entry.seed_index,
                    analysis_artifact_name=entry.analysis_artifact_name,
                    analysis_artifact_hash=entry.analysis_artifact_hash,
                )
                for entry in source_manifest.seed_analyses
            )
            curated_round_manifest = FrozenDatasetManifest(
                dataset_version=dataset_version,
                capture_timestamp=capture_timestamp_text,
                round_id=round_id,
                seed_ids=source_manifest.seed_ids,
                source_endpoints=source_manifest.source_endpoints,
                artifact_hashes=tuple(
                    ArtifactHash(artifact_name=name, sha256=sha256)
                    for name, sha256 in sorted(round_artifact_hashes.items())
                ),
                solver_id=solver_id,
                map_width=source_manifest.map_width,
                map_height=source_manifest.map_height,
                mapping_artifact_hash=canonical_mapping_artifact_hash(),
                query_trace_artifact_hash=curated_query_trace_hash,
                seed_analyses=curated_seed_analyses,
            )
            round_manifest_path = (
                temp_dataset_dir / "round-manifests" / f"{round_id}.json"
            )
            round_manifest_hash = _record_json_artifact(
                artifact_hashes=artifact_hashes,
                dataset_root=temp_dataset_dir,
                path=round_manifest_path,
                payload=curated_round_manifest.to_payload(),
            )
            curated_rounds.append(
                CuratedRoundManifestEntry(
                    dataset_version=dataset_version,
                    round_id=round_id,
                    seed_ids=curated_round_manifest.seed_ids,
                    source_snapshot_version=round_source.snapshot_version,
                    round_manifest_artifact_name=_artifact_name(
                        round_manifest_path,
                        dataset_root=temp_dataset_dir,
                    ),
                    round_manifest_artifact_hash=round_manifest_hash,
                    query_trace_artifact_name=_artifact_name(
                        query_trace_path,
                        dataset_root=temp_dataset_dir,
                    ),
                    query_trace_artifact_hash=curated_query_trace_hash,
                    prediction_run_id=round_source.capture.prediction_run_id,
                )
            )

        manifest = CuratedBenchmarkManifest(
            dataset_version=dataset_version,
            capture_timestamp=capture_timestamp_text,
            round_ids=tuple(entry.round_id for entry in curated_rounds),
            source_snapshot_versions=tuple(source_snapshot_versions),
            artifact_hashes=tuple(
                ArtifactHash(artifact_name=name, sha256=sha256)
                for name, sha256 in sorted(artifact_hashes.items())
            ),
            rounds=tuple(curated_rounds),
            solver_id=solver_id,
        )
        hashes_path = temp_dataset_dir / "hashes.json"
        _save_json(
            hashes_path,
            {
                "dataset_version": dataset_version,
                "round_ids": [entry.round_id for entry in curated_rounds],
                "source_snapshot_versions": list(source_snapshot_versions),
                "artifact_hashes": [
                    item.to_payload() for item in manifest.artifact_hashes
                ],
            },
        )
        manifest_path = temp_dataset_dir / "manifest.json"
        _save_json(manifest_path, manifest.to_payload())

        temp_dataset_dir.rename(dataset_dir)
        return CuratedBenchmarkDataset(
            dataset_version=dataset_version,
            dataset_dir=dataset_dir,
            manifest_path=dataset_dir / "manifest.json",
            hashes_path=dataset_dir / "hashes.json",
            manifest=manifest,
        )
    except Exception:
        shutil.rmtree(temp_dataset_dir, ignore_errors=True)
        raise


def _capture_round_artifacts(
    *,
    cache: LocalCache,
    client: DatasetRefreshClient,
    dataset_root: Path,
    artifact_hashes: dict[str, str],
    dataset_version: str,
    round_id: str,
    solver_id: str,
    capture_timestamp_text: str,
    round_detail_path: Path,
    query_trace_path: Path,
    seed_metadata_dir: Path,
) -> _CapturedRoundArtifacts:
    snapshot_cache = LocalCache(dataset_root)
    round_artifact_hashes = {
        "mapping/terrain_mapping.json": canonical_mapping_artifact_hash()
    }
    source_endpoints: set[str] = set()

    round_endpoint = f"/astar-island/rounds/{round_id}"
    round_payload = _fetch_required_json_payload(
        endpoint=round_endpoint,
        fetcher=lambda: client.get_round_detail(round_id),
    )
    round_detail = parse_round_detail_payload(round_payload)
    if round_detail.round_id != round_id:
        raise ValueError("Fetched round detail does not match the requested round_id.")
    source_endpoints.add(round_endpoint)
    _record_json_artifact(
        artifact_hashes=artifact_hashes,
        dataset_root=dataset_root,
        path=round_detail_path,
        payload=round_payload,
        additional_registries=(round_artifact_hashes,),
    )

    raw_initial_states = _read_round_initial_states(round_payload)
    if len(raw_initial_states) != round_detail.seeds_count:
        raise ValueError(
            "Round detail initial_states length does not match seeds_count."
        )
    for seed_index, initial_state in enumerate(raw_initial_states):
        _record_json_artifact(
            artifact_hashes=artifact_hashes,
            dataset_root=dataset_root,
            path=seed_metadata_dir / f"seed-{seed_index:02d}.json",
            payload={
                "round_id": round_id,
                "seed_index": seed_index,
                "initial_state": initial_state,
            },
            additional_registries=(round_artifact_hashes,),
        )

    seed_ids = tuple(range(round_detail.seeds_count))
    query_trace_entries = _freeze_query_payloads(
        source_cache=cache,
        snapshot_cache=snapshot_cache,
        dataset_root=dataset_root,
        artifact_hashes=artifact_hashes,
        dataset_version=dataset_version,
        round_id=round_id,
        seed_ids=seed_ids,
        additional_registries=(round_artifact_hashes,),
    )
    query_trace_hash = canonical_json_hash(
        _query_trace_hash_payload(
            dataset_version=dataset_version,
            round_id=round_id,
            seed_ids=seed_ids,
            entries=query_trace_entries,
            query_budget=DEFAULT_QUERY_BUDGET,
        )
    )
    query_trace = RoundQueryTrace(
        dataset_version=dataset_version,
        round_id=round_id,
        seed_ids=seed_ids,
        entries=query_trace_entries,
        trace_artifact_hash=query_trace_hash,
    )
    _record_json_artifact(
        artifact_hashes=artifact_hashes,
        dataset_root=dataset_root,
        path=query_trace_path,
        payload=query_trace.to_payload(),
        precomputed_hash=query_trace_hash,
        additional_registries=(round_artifact_hashes,),
    )

    analysis_manifest_entries: list[SeedAnalysisManifestEntry] = []
    prediction_run_id = f"submitted-{round_id}"
    for seed_index in seed_ids:
        analysis_endpoint = f"/astar-island/analysis/{round_id}/{seed_index}"
        analysis_payload = _fetch_required_json_payload(
            endpoint=analysis_endpoint,
            fetcher=_build_analysis_fetcher(
                client=client,
                round_id=round_id,
                seed_index=seed_index,
            ),
        )
        analysis = _parse_analysis_payload(
            payload=analysis_payload,
            dataset_version=dataset_version,
            round_id=round_id,
            seed_index=seed_index,
        )
        source_endpoints.add(analysis_endpoint)
        analysis_path = snapshot_cache.analysis_path(round_id, seed_index)
        analysis_hash = _record_json_artifact(
            artifact_hashes=artifact_hashes,
            dataset_root=dataset_root,
            path=analysis_path,
            payload=analysis_payload,
            additional_registries=(round_artifact_hashes,),
        )
        analysis_manifest_entries.append(
            SeedAnalysisManifestEntry(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=seed_index,
                analysis_artifact_name=_artifact_name(
                    analysis_path, dataset_root=dataset_root
                ),
                analysis_artifact_hash=analysis_hash,
            )
        )

        if analysis.submitted_prediction is not None:
            _record_json_artifact(
                artifact_hashes=artifact_hashes,
                dataset_root=dataset_root,
                path=snapshot_cache.prediction_path(prediction_run_id, seed_index),
                payload={
                    "round_id": round_id,
                    "seed_index": seed_index,
                    "prediction": analysis.submitted_prediction,
                },
                additional_registries=(round_artifact_hashes,),
            )

    return _CapturedRoundArtifacts(
        manifest=FrozenDatasetManifest(
            dataset_version=dataset_version,
            capture_timestamp=capture_timestamp_text,
            round_id=round_id,
            seed_ids=seed_ids,
            source_endpoints=tuple(sorted(source_endpoints)),
            artifact_hashes=tuple(
                ArtifactHash(artifact_name=name, sha256=sha256)
                for name, sha256 in sorted(round_artifact_hashes.items())
            ),
            solver_id=solver_id,
            map_width=round_detail.map_width,
            map_height=round_detail.map_height,
            mapping_artifact_hash=canonical_mapping_artifact_hash(),
            query_trace_artifact_hash=query_trace_hash,
            seed_analyses=tuple(analysis_manifest_entries),
        ),
        query_trace=query_trace,
        prediction_run_id=prediction_run_id,
    )


def _freeze_query_payloads(
    *,
    source_cache: LocalCache,
    snapshot_cache: LocalCache,
    dataset_root: Path,
    artifact_hashes: dict[str, str],
    dataset_version: str,
    round_id: str,
    seed_ids: tuple[int, ...],
    additional_registries: Sequence[dict[str, str]] = (),
) -> tuple[QueryTraceEntry, ...]:
    round_queries_dir = source_cache.queries_dir / round_id
    if not round_queries_dir.exists():
        return ()

    expected_seed_dirs = {f"seed-{seed_index:02d}" for seed_index in seed_ids}
    unexpected_seed_dirs = sorted(
        path.name
        for path in round_queries_dir.iterdir()
        if path.is_dir() and path.name not in expected_seed_dirs
    )
    if unexpected_seed_dirs:
        raise ValueError(
            f"Query cache contains unexpected seed directories: {unexpected_seed_dirs}."
        )

    entries_by_query_index: dict[int, QueryTraceEntry] = {}
    for seed_index in seed_ids:
        seed_dir = round_queries_dir / f"seed-{seed_index:02d}"
        if not seed_dir.exists():
            continue
        for source_path in sorted(seed_dir.glob("*.json")):
            payload = source_cache.load_json(source_path)
            query_metadata = _parse_query_trace_payload(payload)
            if query_metadata.queries_used is None:
                raise ValueError(
                    "Cached query payload is missing queries_used, so the shared round "
                    "query trace identity cannot be reconstructed."
                )
            if query_metadata.queries_max is None:
                raise ValueError(
                    "Cached query payload is missing queries_max, so the shared round "
                    "query trace budget cannot be validated."
                )
            query_index = query_metadata.queries_used - 1
            if query_index in entries_by_query_index:
                raise ValueError(
                    "Duplicate queries_used values prevent deterministic query trace "
                    "reconstruction."
                )

            destination_path = snapshot_cache.query_response_path(
                round_id,
                seed_index,
                source_path.stem,
            )
            response_hash = _record_json_artifact(
                artifact_hashes=artifact_hashes,
                dataset_root=dataset_root,
                path=destination_path,
                payload=payload,
                additional_registries=additional_registries,
            )
            entries_by_query_index[query_index] = QueryTraceEntry(
                dataset_version=dataset_version,
                round_id=round_id,
                seed_index=seed_index,
                query_index=query_index,
                viewport=query_metadata.viewport,
                response_artifact_hash=response_hash,
                queries_used=query_metadata.queries_used,
                queries_max=query_metadata.queries_max,
            )

    return tuple(entry for _, entry in sorted(entries_by_query_index.items()))


@dataclass(frozen=True)
class _ParsedQueryTracePayload:
    viewport: Viewport
    queries_used: int | None
    queries_max: int | None


def _parse_query_trace_payload(payload: object) -> _ParsedQueryTracePayload:
    payload_mapping = _require_mapping(payload, context="Cached query payload")
    _require_grid(payload_mapping.get("grid"), field_name="Cached query payload grid")
    viewport_payload = _require_mapping(
        payload_mapping.get("viewport"),
        context="Cached query payload viewport",
    )
    viewport = Viewport(
        x=_require_int(viewport_payload.get("x"), field_name="viewport.x"),
        y=_require_int(viewport_payload.get("y"), field_name="viewport.y"),
        width=_require_int(
            viewport_payload.get("width", viewport_payload.get("w")),
            field_name="viewport.width",
        ),
        height=_require_int(
            viewport_payload.get("height", viewport_payload.get("h")),
            field_name="viewport.height",
        ),
    )
    return _ParsedQueryTracePayload(
        viewport=viewport,
        queries_used=_optional_int(payload_mapping.get("queries_used")),
        queries_max=_optional_int(payload_mapping.get("queries_max")),
    )


def _parse_analysis_payload(
    *,
    payload: object,
    dataset_version: str,
    round_id: str,
    seed_index: int,
) -> OrganizerSeedAnalysis:
    payload_mapping = _require_mapping(payload, context="Analysis payload")
    width = _require_int(payload_mapping.get("width"), field_name="analysis.width")
    height = _require_int(payload_mapping.get("height"), field_name="analysis.height")
    ground_truth = _parse_prediction_tensor(
        payload_mapping.get("ground_truth"),
        tensor_name="analysis.ground_truth",
    )
    raw_prediction = payload_mapping.get("prediction")
    submitted_prediction = (
        None
        if raw_prediction is None
        else _parse_prediction_tensor(
            raw_prediction,
            tensor_name="analysis.prediction",
        )
    )
    score = payload_mapping.get("score")
    organizer_score = (
        None if score is None else _require_float(score, field_name="score")
    )

    return OrganizerSeedAnalysis(
        dataset_version=dataset_version,
        round_id=round_id,
        seed_index=seed_index,
        width=width,
        height=height,
        analysis_artifact_hash=canonical_json_hash(payload),
        ground_truth=ground_truth,
        submitted_prediction=submitted_prediction,
        organizer_score=organizer_score,
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
            parsed_cell: list[float] = []
            for probability in cell:
                parsed_cell.append(
                    _require_float(probability, field_name=f"{tensor_name} probability")
                )
            parsed_row.append(parsed_cell)
        tensor.append(parsed_row)
    return tensor


def _fetch_required_json_payload(
    *, endpoint: str, fetcher: Callable[[], object]
) -> object:
    try:
        payload = fetcher()
    except Exception as error:
        raise ValueError(
            f"Failed to fetch required payload from {endpoint}."
        ) from error
    if not _is_json_value(payload):
        raise ValueError(f"Endpoint {endpoint} returned a non-JSON payload.")
    return payload


def _build_analysis_fetcher(
    *, client: DatasetRefreshClient, round_id: str, seed_index: int
) -> Callable[[], object]:
    def fetch() -> object:
        return client.get_analysis(round_id=round_id, seed_index=seed_index)

    return fetch


def _query_trace_hash_payload(
    *,
    dataset_version: str,
    round_id: str,
    seed_ids: tuple[int, ...],
    entries: tuple[QueryTraceEntry, ...],
    query_budget: int,
) -> dict[str, object]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "round_id": round_id,
        "seed_ids": list(seed_ids),
        "entries": [entry.to_payload() for entry in entries],
        "query_budget": query_budget,
    }


def _read_round_initial_states(round_payload: object) -> list[object]:
    round_mapping = _require_mapping(round_payload, context="Round detail payload")
    raw_initial_states = round_mapping.get("initial_states")
    if not isinstance(raw_initial_states, list):
        raise ValueError("Round detail payload field 'initial_states' must be a list.")
    return list(raw_initial_states)


def _record_json_artifact(
    *,
    artifact_hashes: dict[str, str],
    dataset_root: Path,
    path: Path,
    payload: object,
    precomputed_hash: str | None = None,
    additional_registries: Sequence[dict[str, str]] = (),
) -> str:
    if not _is_json_value(payload):
        raise ValueError(f"Path {path} received a non-JSON-compatible payload.")
    artifact_name = _artifact_name(path, dataset_root=dataset_root)
    if artifact_name in artifact_hashes:
        raise ValueError(f"Artifact {artifact_name!r} was recorded twice.")
    artifact_hash = (
        canonical_json_hash(payload) if precomputed_hash is None else precomputed_hash
    )
    _save_json(path, payload)
    artifact_hashes[artifact_name] = artifact_hash
    for registry in additional_registries:
        if artifact_name in registry:
            raise ValueError(f"Artifact {artifact_name!r} was recorded twice.")
        registry[artifact_name] = artifact_hash
    return artifact_hash


@dataclass(frozen=True)
class _SelectedRoundSource:
    snapshot_version: str
    snapshot_dir: Path
    capture: HistoryRoundCapture


def _select_history_round_ids(
    *,
    cache: LocalCache,
    client: DatasetHistoryRefreshClient,
    requested_round_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    completed_round_ids = _load_completed_round_ids(client.list_rounds())
    requested = None if requested_round_ids is None else tuple(requested_round_ids)
    if requested is not None:
        missing_round_ids = sorted(set(requested) - set(completed_round_ids))
        if missing_round_ids:
            raise ValueError(
                f"History refresh requested non-completed rounds: {missing_round_ids}."
            )
        candidate_round_ids = requested
    else:
        candidate_round_ids = completed_round_ids

    existing_round_ids = _load_existing_history_round_ids(cache)
    selected_round_ids = tuple(
        round_id
        for round_id in candidate_round_ids
        if round_id not in existing_round_ids
    )
    if not selected_round_ids:
        raise ValueError(
            "History refresh did not find any uncaptured completed rounds."
        )
    return selected_round_ids


def _load_completed_round_ids(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, list):
        raise ValueError("Rounds payload must be a JSON array.")
    completed_round_ids: list[str] = []
    for index, item in enumerate(payload):
        round_mapping = _require_mapping(item, context=f"Rounds payload item {index}")
        round_id = round_mapping.get("id")
        status = round_mapping.get("status")
        if not isinstance(round_id, str) or not round_id:
            raise ValueError(f"Rounds payload item {index} is missing a valid id.")
        if not isinstance(status, str) or not status:
            raise ValueError(f"Rounds payload item {index} is missing a valid status.")
        if status == "completed":
            completed_round_ids.append(round_id)
    return tuple(completed_round_ids)


def _load_existing_history_round_ids(cache: LocalCache) -> set[str]:
    raw_history_dir = _history_raw_dir(cache)
    if not raw_history_dir.exists():
        return set()
    round_ids: set[str] = set()
    for snapshot_dir in sorted(
        path for path in raw_history_dir.iterdir() if path.is_dir()
    ):
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _load_history_refresh_manifest(manifest_path)
        round_ids.update(manifest.round_ids)
    return round_ids


def _select_round_sources(
    *,
    cache: LocalCache,
    source_snapshot_versions: Sequence[str],
    requested_round_ids: Sequence[str] | None,
) -> tuple[_SelectedRoundSource, ...]:
    seen_round_ids: set[str] = set()
    sources_by_round_id: dict[str, _SelectedRoundSource] = {}
    for snapshot_version in source_snapshot_versions:
        snapshot_dir = _history_raw_dir(cache) / snapshot_version
        if not snapshot_dir.exists():
            raise FileNotFoundError(
                f"History snapshot version {snapshot_version!r} does not exist."
            )
        manifest = _load_history_refresh_manifest(snapshot_dir / "manifest.json")
        for capture in manifest.rounds:
            if capture.round_id in seen_round_ids:
                raise ValueError(
                    "Curated dataset cannot merge duplicate round ids across "
                    f"history snapshots: {capture.round_id!r}."
                )
            seen_round_ids.add(capture.round_id)
            sources_by_round_id[capture.round_id] = _SelectedRoundSource(
                snapshot_version=snapshot_version,
                snapshot_dir=snapshot_dir,
                capture=capture,
            )

    if requested_round_ids is None:
        selected_round_ids = tuple(sources_by_round_id)
    else:
        selected_round_ids = tuple(requested_round_ids)
        missing_round_ids = sorted(
            round_id
            for round_id in selected_round_ids
            if round_id not in sources_by_round_id
        )
        if missing_round_ids:
            raise ValueError(
                "Curated dataset requested missing history rounds: "
                f"{missing_round_ids}."
            )
    if not selected_round_ids:
        raise ValueError("Curated dataset must include at least one round.")
    return tuple(sources_by_round_id[round_id] for round_id in selected_round_ids)


def _copy_round_directory(
    *,
    source_root: Path,
    source_dir: Path,
    destination_root: Path,
    destination_dir: Path,
    artifact_hashes: dict[str, str],
    round_artifact_hashes: dict[str, str],
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory {source_dir}.")
    for source_path in sorted(source_dir.rglob("*.json")):
        relative_path = source_path.relative_to(source_dir)
        _copy_json_artifact(
            source_root=source_root,
            source_path=source_path,
            destination_root=destination_root,
            destination_path=destination_dir / relative_path,
            artifact_hashes=artifact_hashes,
            round_artifact_hashes=round_artifact_hashes,
        )


def _copy_json_artifact(
    *,
    source_root: Path,
    source_path: Path,
    destination_root: Path,
    destination_path: Path,
    artifact_hashes: dict[str, str],
    round_artifact_hashes: dict[str, str],
) -> str:
    payload = _load_json(source_path)
    return _record_json_artifact(
        artifact_hashes=artifact_hashes,
        dataset_root=destination_root,
        path=destination_path,
        payload=payload,
        precomputed_hash=canonical_json_hash(payload),
        additional_registries=(round_artifact_hashes,),
    )


def _load_history_refresh_manifest(path: Path) -> HistoryRefreshManifest:
    payload = _require_mapping(_load_json(path), context="History refresh manifest")
    rounds_payload = payload.get("rounds")
    if not isinstance(rounds_payload, list):
        raise ValueError("History refresh manifest rounds must be a list.")
    return HistoryRefreshManifest(
        snapshot_version=_require_str(
            payload.get("snapshot_version"),
            field_name="snapshot_version",
        ),
        capture_timestamp=_require_str(
            payload.get("capture_timestamp"),
            field_name="capture_timestamp",
        ),
        round_ids=tuple(
            _require_str(item, field_name="round_id")
            for item in _require_list(payload.get("round_ids"), field_name="round_ids")
        ),
        artifact_hashes=_parse_artifact_hashes(payload.get("artifact_hashes")),
        rounds=tuple(_parse_history_round_capture(item) for item in rounds_payload),
        solver_id=_require_str(payload.get("solver_id"), field_name="solver_id"),
        schema_version=_require_str(
            payload.get("schema_version"),
            field_name="schema_version",
        ),
    )


def _parse_history_round_capture(payload: object) -> HistoryRoundCapture:
    payload_mapping = _require_mapping(payload, context="History round capture")
    return HistoryRoundCapture(
        snapshot_version=_require_str(
            payload_mapping.get("snapshot_version"),
            field_name="snapshot_version",
        ),
        round_id=_require_str(payload_mapping.get("round_id"), field_name="round_id"),
        seed_ids=tuple(
            _require_int(item, field_name="seed_id")
            for item in _require_list(
                payload_mapping.get("seed_ids"), field_name="seed_ids"
            )
        ),
        round_manifest_artifact_name=_require_str(
            payload_mapping.get("round_manifest_artifact_name"),
            field_name="round_manifest_artifact_name",
        ),
        round_manifest_artifact_hash=_require_str(
            payload_mapping.get("round_manifest_artifact_hash"),
            field_name="round_manifest_artifact_hash",
        ),
        query_trace_artifact_name=_require_str(
            payload_mapping.get("query_trace_artifact_name"),
            field_name="query_trace_artifact_name",
        ),
        query_trace_artifact_hash=_require_str(
            payload_mapping.get("query_trace_artifact_hash"),
            field_name="query_trace_artifact_hash",
        ),
        prediction_run_id=_require_str(
            payload_mapping.get("prediction_run_id"),
            field_name="prediction_run_id",
        ),
        source_endpoints=tuple(
            _require_str(item, field_name="source_endpoint")
            for item in _require_list(
                payload_mapping.get("source_endpoints"),
                field_name="source_endpoints",
            )
        ),
    )


def _load_frozen_dataset_manifest(path: Path) -> FrozenDatasetManifest:
    payload = _require_mapping(_load_json(path), context="Frozen dataset manifest")
    return FrozenDatasetManifest(
        dataset_version=_require_str(
            payload.get("dataset_version"),
            field_name="dataset_version",
        ),
        capture_timestamp=_require_str(
            payload.get("capture_timestamp"),
            field_name="capture_timestamp",
        ),
        round_id=_require_str(payload.get("round_id"), field_name="round_id"),
        seed_ids=tuple(
            _require_int(item, field_name="seed_id")
            for item in _require_list(payload.get("seed_ids"), field_name="seed_ids")
        ),
        source_endpoints=tuple(
            _require_str(item, field_name="source_endpoint")
            for item in _require_list(
                payload.get("source_endpoints"),
                field_name="source_endpoints",
            )
        ),
        artifact_hashes=_parse_artifact_hashes(payload.get("artifact_hashes")),
        solver_id=_require_str(payload.get("solver_id"), field_name="solver_id"),
        schema_version=_require_str(
            payload.get("schema_version"),
            field_name="schema_version",
        ),
        map_width=_require_int(payload.get("map_width"), field_name="map_width"),
        map_height=_require_int(payload.get("map_height"), field_name="map_height"),
        mapping_artifact_hash=_require_str(
            payload.get("mapping_artifact_hash"),
            field_name="mapping_artifact_hash",
        ),
        query_trace_artifact_hash=_require_str(
            payload.get("query_trace_artifact_hash"),
            field_name="query_trace_artifact_hash",
        ),
        seed_analyses=tuple(
            _parse_seed_analysis_manifest_entry(item)
            for item in _require_list(
                payload.get("seed_analyses"),
                field_name="seed_analyses",
            )
        ),
    )


def _parse_seed_analysis_manifest_entry(payload: object) -> SeedAnalysisManifestEntry:
    payload_mapping = _require_mapping(payload, context="Seed analysis manifest entry")
    return SeedAnalysisManifestEntry(
        dataset_version=_require_str(
            payload_mapping.get("dataset_version"),
            field_name="dataset_version",
        ),
        round_id=_require_str(payload_mapping.get("round_id"), field_name="round_id"),
        seed_index=_require_int(
            payload_mapping.get("seed_index"),
            field_name="seed_index",
        ),
        analysis_artifact_name=_require_str(
            payload_mapping.get("analysis_artifact_name"),
            field_name="analysis_artifact_name",
        ),
        analysis_artifact_hash=_require_str(
            payload_mapping.get("analysis_artifact_hash"),
            field_name="analysis_artifact_hash",
        ),
    )


def _load_round_query_trace(path: Path) -> RoundQueryTrace:
    payload = _require_mapping(_load_json(path), context="Round query trace")
    entries_payload = _require_list(payload.get("entries"), field_name="entries")
    return RoundQueryTrace(
        dataset_version=_require_str(
            payload.get("dataset_version"),
            field_name="dataset_version",
        ),
        round_id=_require_str(payload.get("round_id"), field_name="round_id"),
        seed_ids=tuple(
            _require_int(item, field_name="seed_id")
            for item in _require_list(payload.get("seed_ids"), field_name="seed_ids")
        ),
        entries=tuple(_parse_query_trace_entry(item) for item in entries_payload),
        trace_artifact_hash=_require_str(
            payload.get("trace_artifact_hash"),
            field_name="trace_artifact_hash",
        ),
        query_budget=_require_int(
            payload.get("query_budget"),
            field_name="query_budget",
        ),
        schema_version=_require_str(
            payload.get("schema_version"),
            field_name="schema_version",
        ),
    )


def _parse_query_trace_entry(payload: object) -> QueryTraceEntry:
    payload_mapping = _require_mapping(payload, context="Query trace entry")
    viewport_payload = _require_mapping(
        payload_mapping.get("viewport"),
        context="Query trace entry viewport",
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
        viewport=Viewport(
            x=_require_int(viewport_payload.get("x"), field_name="viewport.x"),
            y=_require_int(viewport_payload.get("y"), field_name="viewport.y"),
            width=_require_int(
                viewport_payload.get("width"),
                field_name="viewport.width",
            ),
            height=_require_int(
                viewport_payload.get("height"),
                field_name="viewport.height",
            ),
        ),
        response_artifact_hash=_require_str(
            payload_mapping.get("response_artifact_hash"),
            field_name="response_artifact_hash",
        ),
        queries_used=_optional_int(payload_mapping.get("queries_used")),
        queries_max=_optional_int(payload_mapping.get("queries_max")),
    )


def _parse_artifact_hashes(payload: object) -> tuple[ArtifactHash, ...]:
    return tuple(
        ArtifactHash(
            artifact_name=_require_str(
                _require_mapping(item, context="Artifact hash entry").get(
                    "artifact_name"
                ),
                field_name="artifact_name",
            ),
            sha256=_require_str(
                _require_mapping(item, context="Artifact hash entry").get("sha256"),
                field_name="sha256",
            ),
        )
        for item in _require_list(payload, field_name="artifact_hashes")
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


def _require_registry_hash(
    hash_registry: dict[str, str], *, artifact_name: str, expected_hash: str
) -> None:
    if hash_registry.get(artifact_name) != expected_hash:
        raise ValueError(f"Artifact hash mismatch for {artifact_name!r}.")


def _history_raw_dir(cache: LocalCache) -> Path:
    return cache.root / "history" / "raw"


def _load_json(path: Path) -> object:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not _is_json_value(payload):
        raise ValueError(f"File {path} does not contain JSON-compatible data.")
    return payload


def _artifact_name(path: Path, *, dataset_root: Path) -> str:
    return path.relative_to(dataset_root).as_posix()


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_capture_timestamp(timestamp: datetime | None) -> str:
    effective_timestamp = (
        datetime.now(timezone.utc) if timestamp is None else timestamp  # noqa: UP017
    )
    normalized_timestamp = effective_timestamp.astimezone(timezone.utc).replace(  # noqa: UP017
        microsecond=0
    )
    return normalized_timestamp.isoformat().replace("+00:00", "Z")


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _require_list(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    return list(value)


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_grid(value: object, *, field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    for row in value:
        if not isinstance(row, list):
            raise ValueError(f"{field_name} rows must be lists.")
        for cell in row:
            _require_int(cell, field_name=field_name)


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty.")


def _require_iso8601_timestamp(timestamp: str) -> None:
    _require_non_empty(timestamp, field_name="capture_timestamp")
    normalized = timestamp.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("capture_timestamp must be ISO 8601.") from error


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


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


__all__ = [
    "CURATED_BENCHMARK_SCHEMA_VERSION",
    "CuratedBenchmarkDataset",
    "CuratedBenchmarkManifest",
    "CuratedRoundManifestEntry",
    "DatasetRefreshClient",
    "DatasetHistoryRefreshClient",
    "FrozenDatasetSnapshot",
    "HISTORY_REFRESH_SCHEMA_VERSION",
    "HistoryRefreshManifest",
    "HistoryRefreshSnapshot",
    "HistoryRoundCapture",
    "build_curated_history_dataset",
    "refresh_dataset_snapshot",
    "refresh_history_snapshot",
]
