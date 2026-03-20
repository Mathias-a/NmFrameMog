from __future__ import annotations

import json
import subprocess
from pathlib import Path

# pyright: reportMissingImports=false
import pytest
from idk_2.astar_island_dr_plan_1.solver.cache import LocalCache
from idk_2.astar_island_dr_plan_1.solver.dataset_refresh import refresh_dataset_snapshot
from idk_2.astar_island_dr_plan_1.solver.evaluate_skill import evaluate_solution
from idk_2.astar_island_dr_plan_1.solver.evaluation_contract import canonical_json_hash


def test_evaluate_skill_benchmark_writes_machine_readable_report_and_summary(
    tmp_path: Path,
) -> None:
    cache = _build_cache_with_dataset(tmp_path)
    _write_candidate_registry(
        cache.root / "datasets" / "dataset-evaluate",
        candidates=(
            {
                "candidate_id": "candidate-under-test",
                "solver_id": "refresh-test",
                "prediction_run_id": "submitted-round-001",
            },
        ),
    )

    outputs = evaluate_solution(
        cache_dir=cache.root,
        dataset_version="dataset-evaluate",
        candidate_id="candidate-under-test",
        mode="benchmark",
    )

    assert outputs.report_path.name == "benchmark-report.json"
    assert outputs.summary_path.name == "benchmark-summary.md"
    assert outputs.report_payload["mode"] == "benchmark"
    assert outputs.report_payload["dataset_version"] == "dataset-evaluate"
    assert outputs.report_payload["candidate_id"] == "candidate-under-test"
    assert outputs.report_payload["status"] == "completed"
    assert outputs.report_payload["final_status"] == "pass"
    assert outputs.report_payload["hard_gate_failures"] == []
    assert (
        "benchmark candidate-under-test on dataset-evaluate: pass"
        in outputs.summary_text
    )
    assert (
        json.loads(outputs.report_path.read_text(encoding="utf-8"))
        == outputs.report_payload
    )
    assert (
        outputs.summary_path.read_text(encoding="utf-8").strip()
        == outputs.summary_text
    )
    assert not (
        cache.root / "datasets" / "dataset-evaluate" / "references.json"
    ).exists()


def test_evaluate_skill_promote_regression_returns_fail_verdict_with_reasons(
    tmp_path: Path,
) -> None:
    cache = _build_cache_with_dataset(tmp_path)
    dataset_dir = cache.root / "datasets" / "dataset-evaluate"
    _clone_predictions_run(
        dataset_dir=dataset_dir,
        source_run_id="submitted-round-001",
        destination_run_id="submitted-round-001-better",
    )
    _set_prediction_payload(
        dataset_dir / "predictions" / "submitted-round-001-better" / "seed-00.json",
        prediction=_better_prediction_tensor(),
    )
    _set_prediction_payload(
        dataset_dir / "predictions" / "submitted-round-001-better" / "seed-01.json",
        prediction=_better_prediction_tensor(),
    )
    _register_prediction_hashes(
        dataset_dir=dataset_dir,
        prediction_run_id="submitted-round-001-better",
    )
    _write_candidate_registry(
        dataset_dir,
        candidates=(
            {
                "candidate_id": "candidate-under-test",
                "solver_id": "refresh-test",
                "prediction_run_id": "submitted-round-001",
            },
        ),
    )
    evaluation_dir = cache.root / "evaluation" / "dataset-evaluate"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_blessed_reference(
        evaluation_dir=evaluation_dir,
        role="blessed_baseline",
        dataset_version="dataset-evaluate",
        candidate_id="candidate-baseline",
        solver_id="refresh-test",
        prediction_run_id="submitted-round-001-better",
    )
    _write_blessed_reference(
        evaluation_dir=evaluation_dir,
        role="last_blessed_candidate",
        dataset_version="dataset-evaluate",
        candidate_id="candidate-last-blessed",
        solver_id="refresh-test",
        prediction_run_id="submitted-round-001-better",
    )

    outputs = evaluate_solution(
        cache_dir=cache.root,
        dataset_version="dataset-evaluate",
        candidate_id="candidate-under-test",
        mode="promote",
    )

    assert outputs.report_path.name == "promote-verdict.json"
    assert outputs.summary_path.name == "promote-summary.md"
    assert outputs.report_payload["mode"] == "promote"
    assert outputs.report_payload["promotion_verdict"] == "fail"
    assert outputs.report_payload["final_status"] == "fail"
    reasons = outputs.report_payload["reasons"]
    assert isinstance(reasons, list)
    assert any(
        "Per-seed regression hard gate failed" in reason for reason in reasons
    )
    hard_gate_failures = outputs.report_payload["hard_gate_failures"]
    assert isinstance(hard_gate_failures, list)
    assert hard_gate_failures[0]["name"] == "per_seed_regression"
    assert hard_gate_failures[0]["status"] == "fail"
    baseline_deltas = outputs.report_payload["baseline_deltas"]
    assert isinstance(baseline_deltas, dict)
    assert baseline_deltas["available"] is True
    last_blessed_deltas = outputs.report_payload["last_blessed_deltas"]
    assert isinstance(last_blessed_deltas, dict)
    assert last_blessed_deltas["available"] is True
    assert (
        "promote candidate-under-test on dataset-evaluate: fail"
        in outputs.summary_text
    )


def test_evaluate_skill_promote_requires_blessed_references(tmp_path: Path) -> None:
    cache = _build_cache_with_dataset(tmp_path)
    _write_candidate_registry(
        cache.root / "datasets" / "dataset-evaluate",
        candidates=(
            {
                "candidate_id": "candidate-under-test",
                "solver_id": "refresh-test",
                "prediction_run_id": "submitted-round-001",
            },
        ),
    )
    with pytest.raises(ValueError, match="requires explicit blessed references"):
        evaluate_solution(
            cache_dir=cache.root,
            dataset_version="dataset-evaluate",
            candidate_id="candidate-under-test",
            mode="promote",
        )


def test_evaluate_skill_promote_rejects_incompatible_blessed_references(
    tmp_path: Path,
) -> None:
    cache = _build_cache_with_dataset(tmp_path)
    _write_candidate_registry(
        cache.root / "datasets" / "dataset-evaluate",
        candidates=(
            {
                "candidate_id": "candidate-under-test",
                "solver_id": "refresh-test",
                "prediction_run_id": "submitted-round-001",
            },
        ),
    )
    evaluation_dir = cache.root / "evaluation" / "dataset-evaluate"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_blessed_reference(
        evaluation_dir=evaluation_dir,
        role="blessed_baseline",
        dataset_version="dataset-other",
        candidate_id="candidate-baseline",
        solver_id="refresh-test",
        prediction_run_id="submitted-round-001",
    )
    _write_blessed_reference(
        evaluation_dir=evaluation_dir,
        role="last_blessed_candidate",
        dataset_version="dataset-evaluate",
        candidate_id="candidate-last-blessed",
        solver_id="refresh-test",
        prediction_run_id="submitted-round-001",
    )

    with pytest.raises(ValueError, match="incompatible with frozen dataset version"):
        evaluate_solution(
            cache_dir=cache.root,
            dataset_version="dataset-evaluate",
            candidate_id="candidate-under-test",
            mode="promote",
        )


def test_evaluate_skill_rejects_curated_or_mixed_dataset_inputs(tmp_path: Path) -> None:
    dataset_dir = (
        tmp_path / ".artifacts" / "astar-island" / "datasets" / "dataset-curated"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "astar-curated-benchmark-v1",
                "dataset_version": "dataset-curated",
                "capture_timestamp": "2026-03-20T10:00:00Z",
                "round_ids": ["round-001", "round-002"],
                "source_snapshot_versions": ["history-1"],
                "artifact_hashes": [],
                "rounds": [
                    {
                        "dataset_version": "dataset-curated",
                        "round_id": "round-001",
                        "seed_ids": [0, 1],
                        "source_snapshot_version": "history-1",
                        "round_manifest_artifact_name": (
                            "round-manifests/round-001.json"
                        ),
                        "round_manifest_artifact_hash": "0" * 64,
                        "query_trace_artifact_name": "query-traces/round-001.json",
                        "query_trace_artifact_hash": "1" * 64,
                        "prediction_run_id": "submitted-round-001",
                    },
                    {
                        "dataset_version": "dataset-curated",
                        "round_id": "round-002",
                        "seed_ids": [0, 1],
                        "source_snapshot_version": "history-1",
                        "round_manifest_artifact_name": (
                            "round-manifests/round-002.json"
                        ),
                        "round_manifest_artifact_hash": "2" * 64,
                        "query_trace_artifact_name": "query-traces/round-002.json",
                        "query_trace_artifact_hash": "3" * 64,
                        "prediction_run_id": "submitted-round-002",
                    },
                ],
                "solver_id": "curated-solver",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="single-round dataset snapshot"):
        evaluate_solution(
            cache_dir=tmp_path / ".artifacts" / "astar-island",
            dataset_version="dataset-curated",
            candidate_id="candidate-under-test",
            mode="benchmark",
        )


def test_evaluate_skill_candidate_registry_selects_distinct_prediction_runs(
    tmp_path: Path,
) -> None:
    cache = _build_cache_with_dataset(tmp_path)
    dataset_dir = cache.root / "datasets" / "dataset-evaluate"
    _clone_predictions_run(
        dataset_dir=dataset_dir,
        source_run_id="submitted-round-001",
        destination_run_id="submitted-round-001-better",
    )
    _set_prediction_payload(
        dataset_dir / "predictions" / "submitted-round-001-better" / "seed-00.json",
        prediction=_better_prediction_tensor(),
    )
    _set_prediction_payload(
        dataset_dir / "predictions" / "submitted-round-001-better" / "seed-01.json",
        prediction=_better_prediction_tensor(),
    )
    _register_prediction_hashes(
        dataset_dir=dataset_dir,
        prediction_run_id="submitted-round-001-better",
    )
    _write_candidate_registry(
        dataset_dir,
        candidates=(
            {
                "candidate_id": "candidate-default",
                "solver_id": "refresh-test",
                "prediction_run_id": "submitted-round-001",
            },
            {
                "candidate_id": "candidate-better",
                "solver_id": "refresh-test",
                "prediction_run_id": "submitted-round-001-better",
            },
        ),
    )

    default_outputs = evaluate_solution(
        cache_dir=cache.root,
        dataset_version="dataset-evaluate",
        candidate_id="candidate-default",
        mode="benchmark",
    )
    better_outputs = evaluate_solution(
        cache_dir=cache.root,
        dataset_version="dataset-evaluate",
        candidate_id="candidate-better",
        mode="benchmark",
    )

    default_report = default_outputs.report_payload["benchmark_report"]
    better_report = better_outputs.report_payload["benchmark_report"]
    assert isinstance(default_report, dict)
    assert isinstance(better_report, dict)
    assert default_report["candidate_id"] == "candidate-default"
    assert better_report["candidate_id"] == "candidate-better"
    default_aggregate = default_report["aggregate"]
    better_aggregate = better_report["aggregate"]
    assert isinstance(default_aggregate, dict)
    assert isinstance(better_aggregate, dict)
    assert (
        better_aggregate["aggregate_score"]
        != default_aggregate["aggregate_score"]
    )
    assert better_aggregate["aggregate_score"] > default_aggregate["aggregate_score"]


def test_promote_cli_returns_non_zero_for_failing_verdict(tmp_path: Path) -> None:
    cache = _build_cache_with_dataset(tmp_path)
    dataset_dir = cache.root / "datasets" / "dataset-evaluate"
    _clone_predictions_run(
        dataset_dir=dataset_dir,
        source_run_id="submitted-round-001",
        destination_run_id="submitted-round-001-better",
    )
    _set_prediction_payload(
        dataset_dir / "predictions" / "submitted-round-001-better" / "seed-00.json",
        prediction=_better_prediction_tensor(),
    )
    _set_prediction_payload(
        dataset_dir / "predictions" / "submitted-round-001-better" / "seed-01.json",
        prediction=_better_prediction_tensor(),
    )
    _register_prediction_hashes(
        dataset_dir=dataset_dir,
        prediction_run_id="submitted-round-001-better",
    )
    _write_candidate_registry(
        dataset_dir,
        candidates=(
            {
                "candidate_id": "candidate-under-test",
                "solver_id": "refresh-test",
                "prediction_run_id": "submitted-round-001",
            },
        ),
    )
    evaluation_dir = cache.root / "evaluation" / "dataset-evaluate"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_blessed_reference(
        evaluation_dir=evaluation_dir,
        role="blessed_baseline",
        dataset_version="dataset-evaluate",
        candidate_id="candidate-baseline",
        solver_id="refresh-test",
        prediction_run_id="submitted-round-001-better",
    )
    _write_blessed_reference(
        evaluation_dir=evaluation_dir,
        role="last_blessed_candidate",
        dataset_version="dataset-evaluate",
        candidate_id="candidate-last-blessed",
        solver_id="refresh-test",
        prediction_run_id="submitted-round-001-better",
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "python",
            "-m",
            "nmframemog.astar_island",
            "evaluate-solution",
            "promote",
            "--cache-dir",
            str(cache.root),
            "--dataset-version",
            "dataset-evaluate",
            "--candidate",
            "candidate-under-test",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["report"]["promotion_verdict"] == "fail"


def _build_cache_with_dataset(tmp_path: Path) -> LocalCache:
    cache = LocalCache(tmp_path / ".artifacts" / "astar-island")
    _seed_query_cache(cache)
    refresh_dataset_snapshot(
        cache=cache,
        client=_FrozenRefreshClient(),
        round_id="round-001",
        dataset_version="dataset-evaluate",
        solver_id="refresh-test",
        capture_timestamp=None,
    )
    return cache


class _FrozenRefreshClient:
    def get_round_detail(self, round_id: str) -> object:
        assert round_id == "round-001"
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

    def get_analysis(self, *, round_id: str, seed_index: int) -> object:
        assert round_id == "round-001"
        analyses = {
            0: {
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
                "score": 71.0,
                "width": 2,
                "height": 2,
                "initial_grid": [[10, 11], [4, 5]],
            },
            1: {
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
                "score": 72.0,
                "width": 2,
                "height": 2,
                "initial_grid": [[10, 11], [4, 5]],
            },
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


def _clone_predictions_run(
    *, dataset_dir: Path, source_run_id: str, destination_run_id: str
) -> None:
    source_dir = dataset_dir / "predictions" / source_run_id
    destination_dir = dataset_dir / "predictions" / destination_run_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.glob("*.json"):
        destination_path = destination_dir / path.name
        destination_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _set_prediction_payload(path: Path, *, prediction: list[list[list[float]]]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prediction"] = prediction
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _register_prediction_hashes(*, dataset_dir: Path, prediction_run_id: str) -> None:
    manifest_path = dataset_dir / "manifest.json"
    hashes_path = dataset_dir / "hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes_payload = json.loads(hashes_path.read_text(encoding="utf-8"))
    artifact_hashes = list(manifest["artifact_hashes"])
    registry_hashes = list(hashes_payload["artifact_hashes"])
    for seed_index in (0, 1):
        prediction_path = (
            dataset_dir
            / "predictions"
            / prediction_run_id
            / f"seed-{seed_index:02d}.json"
        )
        artifact_name = f"predictions/{prediction_run_id}/seed-{seed_index:02d}.json"
        artifact_hash = _canonical_json_hash(
            json.loads(prediction_path.read_text(encoding="utf-8"))
        )
        artifact_entry = {"artifact_name": artifact_name, "sha256": artifact_hash}
        artifact_hashes.append(artifact_entry)
        registry_hashes.append(artifact_entry)
    manifest["artifact_hashes"] = artifact_hashes
    hashes_payload["artifact_hashes"] = registry_hashes
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes_path.write_text(
        json.dumps(hashes_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_candidate_registry(
    dataset_dir: Path, *, candidates: tuple[dict[str, str], ...]
) -> None:
    (dataset_dir / "candidates.json").write_text(
        json.dumps(
            {
                "schema_version": "astar-evaluation-candidates-v1",
                "dataset_version": "dataset-evaluate",
                "candidates": list(candidates),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_blessed_reference(
    *,
    evaluation_dir: Path,
    role: str,
    dataset_version: str,
    candidate_id: str,
    solver_id: str,
    prediction_run_id: str,
) -> None:
    filename = {
        "blessed_baseline": "blessed-baseline.json",
        "last_blessed_candidate": "last-blessed-candidate.json",
    }[role]
    (evaluation_dir / filename).write_text(
        json.dumps(
            {
                "schema_version": "astar-blessed-reference-v1",
                "reference_role": role,
                "reference_key": {
                    "dataset_version": dataset_version,
                    "candidate_id": candidate_id,
                },
                "dataset_version_compatibility": {
                    "dataset_version": dataset_version,
                    "status": "exact",
                },
                "solver_id": solver_id,
                "prediction_run_id": prediction_run_id,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _better_prediction_tensor() -> list[list[list[float]]]:
    return [
        [
            [0.80, 0.05, 0.05, 0.04, 0.03, 0.03],
            [0.05, 0.70, 0.10, 0.05, 0.05, 0.05],
        ],
        [
            [0.10, 0.10, 0.10, 0.10, 0.50, 0.10],
            [0.05, 0.05, 0.05, 0.05, 0.10, 0.70],
        ],
    ]


def _canonical_json_hash(payload: object) -> str:
    return canonical_json_hash(payload)
