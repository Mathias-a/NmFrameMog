from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

# pyright: reportMissingImports=false
import pytest
from idk_2.astar_island_dr_plan_1.solver.cache import LocalCache
from idk_2.astar_island_dr_plan_1.solver.dataset_refresh import (
    CuratedBenchmarkDataset,
    FrozenDatasetSnapshot,
    HistoryRefreshSnapshot,
    build_curated_history_dataset,
    refresh_dataset_snapshot,
    refresh_history_snapshot,
)
from idk_2.astar_island_dr_plan_1.solver.evaluation_contract import (
    EVALUATION_SCHEMA_VERSION,
    canonical_json_hash,
    canonical_mapping_artifact_hash,
)


@dataclass(frozen=True)
class _FakeRefreshClient:
    round_payload: object
    analysis_by_seed: dict[int, object]
    rounds_payload: object | None = None

    def get_round_detail(self, round_id: str) -> object:
        return self.round_payload

    def get_analysis(self, *, round_id: str, seed_index: int) -> object:
        if seed_index not in self.analysis_by_seed:
            raise RuntimeError(f"missing analysis for seed {seed_index}")
        return self.analysis_by_seed[seed_index]

    def list_rounds(self) -> object:
        if self.rounds_payload is None:
            raise RuntimeError("missing rounds payload")
        return self.rounds_payload


@dataclass(frozen=True)
class _FakeHistoryRefreshClient:
    rounds_payload: object
    round_payloads: dict[str, object]
    analysis_by_round_and_seed: dict[str, dict[int, object]]

    def list_rounds(self) -> object:
        return self.rounds_payload

    def get_round_detail(self, round_id: str) -> object:
        if round_id not in self.round_payloads:
            raise RuntimeError(f"missing round payload for {round_id}")
        return self.round_payloads[round_id]

    def get_analysis(self, *, round_id: str, seed_index: int) -> object:
        try:
            return self.analysis_by_round_and_seed[round_id][seed_index]
        except KeyError as error:
            raise RuntimeError(
                f"missing analysis for round {round_id} seed {seed_index}"
            ) from error


def test_refresh_dataset_snapshot_freezes_new_versioned_dataset(
    tmp_path: Path,
) -> None:
    cache = LocalCache(tmp_path / ".artifacts" / "astar-island")
    _seed_query_cache(cache)
    client = _FakeRefreshClient(
        round_payload=_round_payload(),
        analysis_by_seed={
            0: _analysis_payload(score=71.0),
            1: _analysis_payload(score=72.0),
        },
    )

    snapshot = refresh_dataset_snapshot(
        cache=cache,
        client=client,
        round_id="round-001",
        dataset_version="dataset-20260320T100000Z",
        solver_id="refresh-test",
        capture_timestamp=_capture_timestamp(),
    )

    assert isinstance(snapshot, FrozenDatasetSnapshot)
    assert snapshot.dataset_dir == cache.root / "datasets" / "dataset-20260320T100000Z"
    assert snapshot.manifest.dataset_version == "dataset-20260320T100000Z"
    assert snapshot.manifest.capture_timestamp == "2026-03-20T10:00:00Z"
    assert snapshot.manifest.seed_ids == (0, 1)
    assert snapshot.query_trace.entries[0].query_index == 0
    assert snapshot.query_trace.entries[1].query_index == 1
    assert snapshot.query_trace.entries[0].seed_index == 0
    assert snapshot.query_trace.entries[1].seed_index == 1
    assert (
        snapshot.query_trace.trace_artifact_hash
        == snapshot.manifest.query_trace_artifact_hash
    )
    assert sorted(snapshot.manifest.source_endpoints) == [
        "/astar-island/analysis/round-001/0",
        "/astar-island/analysis/round-001/1",
        "/astar-island/rounds/round-001",
    ]

    assert (snapshot.dataset_dir / "rounds" / "round-001.json").exists()
    assert (snapshot.dataset_dir / "analysis" / "round-001" / "seed-00.json").exists()
    assert (snapshot.dataset_dir / "analysis" / "round-001" / "seed-01.json").exists()
    assert (
        snapshot.dataset_dir / "queries" / "round-001" / "seed-00" / "q0.json"
    ).exists()
    assert (
        snapshot.dataset_dir / "queries" / "round-001" / "seed-01" / "q1.json"
    ).exists()
    assert (
        snapshot.dataset_dir / "predictions" / "submitted-round-001" / "seed-00.json"
    ).exists()
    assert (
        snapshot.dataset_dir / "predictions" / "submitted-round-001" / "seed-01.json"
    ).exists()
    assert (snapshot.dataset_dir / "seed-metadata" / "seed-00.json").exists()
    assert (snapshot.dataset_dir / "seed-metadata" / "seed-01.json").exists()


def test_refresh_dataset_snapshot_fails_closed_on_partial_analysis(
    tmp_path: Path,
) -> None:
    cache = LocalCache(tmp_path / ".artifacts" / "astar-island")
    _seed_query_cache(cache)
    client = _FakeRefreshClient(
        round_payload=_round_payload(),
        analysis_by_seed={0: _analysis_payload(score=71.0)},
    )

    with pytest.raises(ValueError, match="Failed to fetch required payload"):
        refresh_dataset_snapshot(
            cache=cache,
            client=client,
            round_id="round-001",
            dataset_version="dataset-partial",
        )

    assert not (cache.root / "datasets" / "dataset-partial").exists()


def test_refresh_dataset_snapshot_rejects_duplicate_dataset_version(
    tmp_path: Path,
) -> None:
    cache = LocalCache(tmp_path / ".artifacts" / "astar-island")
    _seed_query_cache(cache)
    client = _FakeRefreshClient(
        round_payload=_round_payload(),
        analysis_by_seed={
            0: _analysis_payload(score=71.0),
            1: _analysis_payload(score=72.0),
        },
    )

    refresh_dataset_snapshot(
        cache=cache,
        client=client,
        round_id="round-001",
        dataset_version="dataset-duplicate",
    )

    with pytest.raises(FileExistsError, match="already exists"):
        refresh_dataset_snapshot(
            cache=cache,
            client=client,
            round_id="round-001",
            dataset_version="dataset-duplicate",
        )


def test_refresh_dataset_snapshot_records_manifest_and_hashes(
    tmp_path: Path,
) -> None:
    cache = LocalCache(tmp_path / ".artifacts" / "astar-island")
    _seed_query_cache(cache)
    client = _FakeRefreshClient(
        round_payload=_round_payload(),
        analysis_by_seed={
            0: _analysis_payload(score=71.0),
            1: _analysis_payload(score=72.0),
        },
    )

    snapshot = refresh_dataset_snapshot(
        cache=cache,
        client=client,
        round_id="round-001",
        dataset_version="dataset-hashes",
        solver_id="refresh-test",
        capture_timestamp=_capture_timestamp(),
    )

    manifest_payload = _load_mapping(snapshot.manifest_path)
    hashes_payload = _load_mapping(snapshot.hashes_path)
    query_trace_payload = _load_mapping(snapshot.query_trace_path)
    analysis_payload = _load_json(
        snapshot.dataset_dir / "analysis" / "round-001" / "seed-00.json"
    )

    assert manifest_payload["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert (
        manifest_payload["mapping_artifact_hash"] == canonical_mapping_artifact_hash()
    )
    assert manifest_payload["query_trace_artifact_hash"] == canonical_json_hash(
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "dataset_version": "dataset-hashes",
            "round_id": "round-001",
            "seed_ids": [0, 1],
            "entries": query_trace_payload["entries"],
            "query_budget": 50,
        }
    )
    assert hashes_payload["dataset_version"] == "dataset-hashes"
    assert hashes_payload["round_id"] == "round-001"

    hashes_entries = cast(list[dict[str, str]], hashes_payload["artifact_hashes"])
    artifact_hashes = {item["artifact_name"]: item["sha256"] for item in hashes_entries}
    assert (
        artifact_hashes["query-trace.json"]
        == manifest_payload["query_trace_artifact_hash"]
    )
    assert (
        artifact_hashes["mapping/terrain_mapping.json"]
        == canonical_mapping_artifact_hash()
    )
    assert artifact_hashes["analysis/round-001/seed-00.json"] == canonical_json_hash(
        analysis_payload
    )
    assert manifest_payload["seed_analyses"] == [
        {
            "dataset_version": "dataset-hashes",
            "round_id": "round-001",
            "seed_index": 0,
            "analysis_artifact_name": "analysis/round-001/seed-00.json",
            "analysis_artifact_hash": artifact_hashes[
                "analysis/round-001/seed-00.json"
            ],
        },
        {
            "dataset_version": "dataset-hashes",
            "round_id": "round-001",
            "seed_index": 1,
            "analysis_artifact_name": "analysis/round-001/seed-01.json",
            "analysis_artifact_hash": artifact_hashes[
                "analysis/round-001/seed-01.json"
            ],
        },
    ]


def test_history_refresh_backfills_completed_rounds_append_only(tmp_path: Path) -> None:
    cache = LocalCache(tmp_path / ".artifacts" / "astar-island")
    _seed_query_cache_for_round(cache, round_id="round-001")
    _seed_query_cache_for_round(cache, round_id="round-002")
    client = _FakeHistoryRefreshClient(
        rounds_payload=[
            {"id": "round-001", "status": "completed"},
            {"id": "round-002", "status": "completed"},
            {"id": "round-active", "status": "active"},
        ],
        round_payloads={
            "round-001": _round_payload(round_id="round-001"),
            "round-002": _round_payload(round_id="round-002"),
        },
        analysis_by_round_and_seed={
            "round-001": {
                0: _analysis_payload(score=71.0),
                1: _analysis_payload(score=72.0),
            },
            "round-002": {
                0: _analysis_payload(score=73.0),
                1: _analysis_payload(score=74.0),
            },
        },
    )

    snapshot = refresh_history_snapshot(
        cache=cache,
        client=client,
        snapshot_version="history-20260320T100000Z",
        capture_timestamp=_capture_timestamp(),
    )

    assert isinstance(snapshot, HistoryRefreshSnapshot)
    assert snapshot.snapshot_dir == (
        cache.root / "history" / "raw" / "history-20260320T100000Z"
    )
    assert snapshot.manifest.round_ids == ("round-001", "round-002")
    assert [entry.round_id for entry in snapshot.manifest.rounds] == [
        "round-001",
        "round-002",
    ]
    assert not (cache.root / "datasets").joinpath("history-20260320T100000Z").exists()
    assert (snapshot.snapshot_dir / "query-traces" / "round-001.json").exists()
    assert (snapshot.snapshot_dir / "query-traces" / "round-002.json").exists()
    assert (snapshot.snapshot_dir / "round-manifests" / "round-001.json").exists()
    assert (snapshot.snapshot_dir / "round-manifests" / "round-002.json").exists()

    with pytest.raises(
        ValueError,
        match="did not find any uncaptured completed rounds",
    ):
        refresh_history_snapshot(
            cache=cache,
            client=client,
            snapshot_version="history-20260320T110000Z",
            capture_timestamp=datetime.fromisoformat("2026-03-20T11:00:00+00:00"),
        )


def test_curated_history_dataset_keeps_older_frozen_versions_immutable(
    tmp_path: Path,
) -> None:
    cache = LocalCache(tmp_path / ".artifacts" / "astar-island")
    _seed_query_cache_for_round(cache, round_id="round-001")
    _seed_query_cache_for_round(cache, round_id="round-002")

    first_history = refresh_history_snapshot(
        cache=cache,
        client=_FakeHistoryRefreshClient(
            rounds_payload=[
                {"id": "round-001", "status": "completed"},
            ],
            round_payloads={
                "round-001": _round_payload(round_id="round-001"),
            },
            analysis_by_round_and_seed={
                "round-001": {
                    0: _analysis_payload(score=71.0),
                    1: _analysis_payload(score=72.0),
                },
            },
        ),
        snapshot_version="history-20260320T100000Z",
        capture_timestamp=_capture_timestamp(),
    )
    curated_v1 = build_curated_history_dataset(
        cache=cache,
        source_snapshot_versions=[first_history.snapshot_version],
        dataset_version="dataset-history-v1",
        capture_timestamp=_capture_timestamp(),
    )
    original_manifest_text = curated_v1.manifest_path.read_text(encoding="utf-8")

    second_history = refresh_history_snapshot(
        cache=cache,
        client=_FakeHistoryRefreshClient(
            rounds_payload=[
                {"id": "round-001", "status": "completed"},
                {"id": "round-002", "status": "completed"},
            ],
            round_payloads={
                "round-001": _round_payload(round_id="round-001"),
                "round-002": _round_payload(round_id="round-002"),
            },
            analysis_by_round_and_seed={
                "round-001": {
                    0: _analysis_payload(score=71.0),
                    1: _analysis_payload(score=72.0),
                },
                "round-002": {
                    0: _analysis_payload(score=73.0),
                    1: _analysis_payload(score=74.0),
                },
            },
        ),
        snapshot_version="history-20260320T120000Z",
        capture_timestamp=datetime.fromisoformat("2026-03-20T12:00:00+00:00"),
    )
    curated_v2 = build_curated_history_dataset(
        cache=cache,
        source_snapshot_versions=[
            first_history.snapshot_version,
            second_history.snapshot_version,
        ],
        dataset_version="dataset-history-v2",
        capture_timestamp=datetime.fromisoformat("2026-03-20T12:05:00+00:00"),
    )

    assert isinstance(curated_v1, CuratedBenchmarkDataset)
    assert isinstance(curated_v2, CuratedBenchmarkDataset)
    assert curated_v1.manifest.round_ids == ("round-001",)
    assert curated_v2.manifest.round_ids == ("round-001", "round-002")
    assert (
        curated_v1.manifest_path.read_text(encoding="utf-8") == original_manifest_text
    )
    assert curated_v1.dataset_dir != curated_v2.dataset_dir
    assert curated_v1.dataset_dir.exists()
    assert curated_v2.dataset_dir.exists()
    assert (curated_v2.dataset_dir / "round-manifests" / "round-002.json").exists()
    assert (
        curated_v1.dataset_dir / "round-manifests" / "round-002.json"
    ).exists() is False
    assert curated_v2.manifest.source_snapshot_versions == (
        "history-20260320T100000Z",
        "history-20260320T120000Z",
    )
    assert [entry.source_snapshot_version for entry in curated_v2.manifest.rounds] == [
        "history-20260320T100000Z",
        "history-20260320T120000Z",
    ]


def _seed_query_cache(cache: LocalCache) -> None:
    _seed_query_cache_for_round(cache, round_id="round-001")


def _seed_query_cache_for_round(cache: LocalCache, *, round_id: str) -> None:
    cache.ensure()
    cache.save_json(
        cache.query_response_path(round_id, 0, "q0"),
        {
            "viewport": {"x": 0, "y": 0, "width": 2, "height": 2},
            "grid": [[4, 11], [1, 3]],
            "settlements": [],
            "queries_used": 1,
            "queries_max": 50,
        },
    )
    cache.save_json(
        cache.query_response_path(round_id, 1, "q1"),
        {
            "viewport": {"x": 1, "y": 1, "width": 2, "height": 2},
            "grid": [[11, 4], [5, 10]],
            "settlements": [],
            "queries_used": 2,
            "queries_max": 50,
        },
    )


def _round_payload(*, round_id: str = "round-001") -> dict[str, object]:
    return {
        "id": round_id,
        "map_width": 2,
        "map_height": 2,
        "seeds_count": 2,
        "initial_states": [
            {"grid": [[10, 11], [4, 5]], "settlements": []},
            {"grid": [[11, 10], [5, 4]], "settlements": []},
        ],
    }


def _analysis_payload(*, score: float) -> dict[str, object]:
    return {
        "prediction": [
            [
                [0.70, 0.10, 0.05, 0.05, 0.05, 0.05],
                [0.10, 0.60, 0.10, 0.10, 0.05, 0.05],
            ],
            [
                [0.10, 0.10, 0.10, 0.20, 0.40, 0.10],
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.75],
            ],
        ],
        "ground_truth": [
            [
                [0.80, 0.05, 0.05, 0.04, 0.03, 0.03],
                [0.05, 0.70, 0.10, 0.05, 0.05, 0.05],
            ],
            [
                [0.10, 0.10, 0.10, 0.10, 0.50, 0.10],
                [0.05, 0.05, 0.05, 0.05, 0.10, 0.70],
            ],
        ],
        "score": score,
        "width": 2,
        "height": 2,
        "initial_grid": [[10, 11], [4, 5]],
    }


def _capture_timestamp() -> datetime:
    return datetime.fromisoformat("2026-03-20T10:00:00+00:00")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mapping(path: Path) -> dict[str, object]:
    payload = _load_json(path)
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)
