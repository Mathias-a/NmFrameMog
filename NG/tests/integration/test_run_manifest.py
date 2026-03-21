from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast

RUN_NAME = "yolov8m-search-baseline"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_run_manifest_main() -> Any:
    module_path = _repo_root() / "src/ng_data/cloud/run_manifest.py"
    spec = importlib.util.spec_from_file_location("ng_cloud_run_manifest", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load run manifest module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_run_manifest_writes_required_snapshots_and_optional_artifacts(
    tmp_path: Path, capsys: object
) -> None:
    typed_capsys = cast(Any, capsys)
    repo_root = _repo_root()
    out_path = tmp_path / "artifacts/run_manifest.json"
    detector_output = repo_root / f"artifacts/models/detector/{RUN_NAME}"
    run_manifest_main = _load_run_manifest_main()

    exit_code = run_manifest_main(
        [
            "--config",
            str(repo_root / "configs/cloud/main.json"),
            "--detector-output",
            str(detector_output),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    payload = cast(dict[str, Any], json.loads(typed_capsys.readouterr().out))

    assert payload["schema_version"] == 1
    assert payload["run_name"] == RUN_NAME
    assert payload["created_at_utc"].endswith("Z")
    assert payload["updated_at_utc"] == payload["created_at_utc"]
    assert payload["cloud"] == {
        "config_path": "configs/cloud/main.json",
        "project_id": "ainm26osl-707",
        "region": "europe-west1",
        "zone": "europe-west1-b",
    }
    assert payload["detector"] == {
        "config": {
            "path": "configs/detector/yolov8m-search.json",
            "snapshot": payload["detector"]["config"]["snapshot"],
        },
        "output_dir": f"artifacts/models/detector/{RUN_NAME}",
    }
    assert payload["dataset_manifest_snapshot"]["path"] == (
        "data/processed/manifests/dataset_manifest.json"
    )
    assert payload["split_manifest_snapshot"]["path"] == (
        "data/processed/manifests/splits.json"
    )
    assert payload["output_artifacts"]["best_pt"]["path"] == (
        f"artifacts/models/detector/{RUN_NAME}/best.pt"
    )
    assert payload["output_artifacts"]["train_summary_json"]["path"] == (
        f"artifacts/models/detector/{RUN_NAME}/train_summary.json"
    )
    assert payload["output_artifacts"]["evaluation_report"]["path"] == (
        "artifacts/eval/detector_holdout_metrics.json"
    )
    assert (
        payload["output_artifacts"]["submission_zip"]["path"]
        == "dist/ng_detector_only_submission.zip"
    )
    assert payload["run_manifest"] == {
        "kind": "detector_run_manifest",
        "path": out_path.as_posix(),
    }
    assert payload["provenance"]["cwd"] == repo_root.as_posix()
    assert set(payload["provenance"]["referenced_files"]) == {
        "cloud_config",
        "dataset_manifest",
        "detector_config",
        "evaluation_report",
        "split_manifest",
        "submission_zip",
        "train_summary",
    }
    assert json.loads(out_path.read_text(encoding="utf-8")) == payload


def test_run_manifest_fails_when_required_best_weights_are_missing(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    detector_output = tmp_path / "artifacts/models/detector/yolov8m-search-baseline"
    run_manifest_main = _load_run_manifest_main()
    detector_output.mkdir(parents=True)
    summary_source = (
        repo_root
        / "artifacts/models/detector/yolov8m-search-baseline/train_summary.json"
    )
    summary_payload = cast(
        dict[str, Any], json.loads(summary_source.read_text(encoding="utf-8"))
    )
    summary_payload["output_artifacts"] = {
        "best_weights": str(detector_output / "best.pt"),
        "summary_json": str(detector_output / "train_summary.json"),
    }
    detector_output.joinpath("train_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        run_manifest_main(
            [
                "--config",
                str(repo_root / "configs/cloud/main.json"),
                "--detector-output",
                str(detector_output),
                "--out",
                str(tmp_path / "artifacts/run_manifest.json"),
            ]
        )
    except SystemExit as error:
        assert "Expected file does not exist" in str(error)
        assert "best.pt" in str(error)
    else:
        raise AssertionError("Expected run manifest generation to fail without best.pt")
