from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import cast

# pyright: reportMissingImports=false
import pytest
from idk_2.astar_island_dr_plan_1.solver.cache import LocalCache
from idk_2.astar_island_dr_plan_1.solver.dataset_refresh import refresh_dataset_snapshot
from idk_2.astar_island_dr_plan_1.solver.evaluation_contract import canonical_json_hash
from idk_2.astar_island_dr_plan_1.solver.replay_harness import (
    FrozenPredictionCandidateAdapter,
    load_offline_replay,
)


def test_replay_harness_is_deterministic_across_repeated_loads(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    adapter = FrozenPredictionCandidateAdapter(
        prediction_run_id="submitted-round-001",
        candidate_id="candidate-offline",
        solver_id="offline-solver",
    )

    replay_a = load_offline_replay(
        dataset_dir=dataset_dir,
        candidate_bundle_adapter=adapter,
    )
    replay_b = load_offline_replay(
        dataset_dir=dataset_dir,
        candidate_bundle_adapter=adapter,
    )

    report_a = replay_a.to_report_json()
    report_b = replay_b.to_report_json()
    assert report_a == report_b
    assert replay_a.to_payload() == replay_b.to_payload()
    assert replay_a.query_trace.entries[0].query_index == 0
    assert replay_a.query_trace.entries[1].query_index == 1
    assert tuple(
        prediction.seed_index for prediction in replay_a.candidate_bundle.predictions
    ) == (0, 1)
    benchmark_input_payload = cast(
        dict[str, object],
        replay_a.to_payload()["benchmark_input"],
    )
    assert benchmark_input_payload["seed_ids"] == [0, 1]


def test_replay_harness_fails_on_missing_artifact(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    missing_prediction_path = (
        dataset_dir / "predictions" / "submitted-round-001" / "seed-01.json"
    )
    missing_prediction_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Missing prediction artifact for seed 1",
    ):
        load_offline_replay(
            dataset_dir=dataset_dir,
            candidate_bundle_adapter=FrozenPredictionCandidateAdapter(
                prediction_run_id="submitted-round-001",
                candidate_id="candidate-offline",
                solver_id="offline-solver",
            ),
        )


def test_replay_harness_consumes_only_frozen_artifacts(tmp_path: Path) -> None:
    cache = LocalCache(tmp_path / ".artifacts" / "astar-island")
    _seed_query_cache(cache)
    snapshot = refresh_dataset_snapshot(
        cache=cache,
        client=_FrozenRefreshClient(),
        round_id="round-001",
        dataset_version="dataset-offline-only",
        solver_id="refresh-test",
        capture_timestamp=_capture_timestamp(),
    )
    shutil.rmtree(cache.rounds_dir)
    shutil.rmtree(cache.queries_dir)
    shutil.rmtree(cache.analysis_dir)
    shutil.rmtree(cache.predictions_dir)

    replay = load_offline_replay(
        dataset_dir=snapshot.dataset_dir,
        candidate_bundle_adapter=FrozenPredictionCandidateAdapter(
            prediction_run_id="submitted-round-001",
            candidate_id="candidate-offline",
            solver_id="offline-solver",
        ),
    )

    report_payload = json.loads(replay.to_report_json())
    assert report_payload["manifest"]["source_endpoints"] == [
        "/astar-island/analysis/round-001/0",
        "/astar-island/analysis/round-001/1",
        "/astar-island/rounds/round-001",
    ]
    assert report_payload["candidate_bundle"]["candidate_id"] == "candidate-offline"
    assert canonical_json_hash(report_payload) == canonical_json_hash(
        json.loads(replay.to_report_json())
    )


def _build_dataset(tmp_path: Path) -> Path:
    cache = LocalCache(tmp_path / ".artifacts" / "astar-island")
    _seed_query_cache(cache)
    snapshot = refresh_dataset_snapshot(
        cache=cache,
        client=_FrozenRefreshClient(),
        round_id="round-001",
        dataset_version="dataset-replay",
        solver_id="refresh-test",
        capture_timestamp=_capture_timestamp(),
    )
    return snapshot.dataset_dir


class _FrozenRefreshClient:
    def get_round_detail(self, round_id: str) -> object:
        assert round_id == "round-001"
        return _round_payload()

    def get_analysis(self, *, round_id: str, seed_index: int) -> object:
        assert round_id == "round-001"
        analyses = {
            0: _analysis_payload(score=71.0),
            1: _analysis_payload(score=72.0),
        }
        return analyses[seed_index]


def _seed_query_cache(cache: LocalCache) -> None:
    cache.ensure()
    cache.save_json(
        cache.query_response_path("round-001", 1, "q1"),
        {
            "viewport": {"x": 1, "y": 1, "width": 2, "height": 2},
            "grid": [[11, 4], [5, 10]],
            "settlements": [],
            "queries_used": 2,
            "queries_max": 50,
        },
    )
    cache.save_json(
        cache.query_response_path("round-001", 0, "q0"),
        {
            "viewport": {"x": 0, "y": 0, "width": 2, "height": 2},
            "grid": [[4, 11], [1, 3]],
            "settlements": [],
            "queries_used": 1,
            "queries_max": 50,
        },
    )


def _round_payload() -> dict[str, object]:
    return {
        "id": "round-001",
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
