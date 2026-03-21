from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, cast
from unittest import TestCase

from src.ng_data.submission import (
    SubmissionBuildError,
    build_submission_zip,
    run_smoke_submission,
)
from src.ng_data.submission.budget_check import collect_budget_evidence

ASSERTIONS = TestCase()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_detector_only_submission_bundle_includes_expected_detector_artifacts() -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/submission/detector_only.json"

    zip_path = build_submission_zip(config_path)

    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()

    assert members == ["run.py", "best.pt", "train_summary.json"]
    assert all("/" not in member for member in members)


def test_detector_only_submission_smoke_run_and_budget_check_pass(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/submission/detector_only.json"
    fixture_dir = repo_root / "tests/fixtures/submission_images"
    output_path = tmp_path / "detector_predictions.json"

    zip_path = build_submission_zip(config_path)
    predictions = run_smoke_submission(zip_path, fixture_dir, output_path)
    budget_evidence = collect_budget_evidence(zip_path, fixture_dir)

    payload = cast(
        list[dict[str, Any]], json.loads(output_path.read_text(encoding="utf-8"))
    )

    assert output_path.is_file()
    assert payload == predictions
    assert isinstance(payload, list)
    for prediction in payload:
        assert set(prediction) == {"image_id", "category_id", "bbox", "score"}
        assert isinstance(prediction["image_id"], int)
        assert isinstance(prediction["category_id"], int)
        assert isinstance(prediction["score"], float)
        bbox = prediction["bbox"]
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert all(isinstance(value, float) for value in bbox)

    assert budget_evidence["status"] == "pass"
    assert budget_evidence["failed_checks"] == []
    assert budget_evidence["smoke_error"] is None
    assert budget_evidence["observed"]["file_count"] == 3
    assert budget_evidence["observed"]["weight_file_count"] == 1
    assert budget_evidence["observed"]["prediction_count"] == len(payload)
    assert cast(float, budget_evidence["observed"]["smoke_runtime_seconds"]) >= 0.0


def test_detector_only_submission_build_fails_when_detector_artifact_path_is_missing(
    tmp_path: Path,
) -> None:
    temp_repo_root = tmp_path / "repo"
    submission_dir = temp_repo_root / "submission"
    detector_dir = temp_repo_root / "artifacts/models/detector/yolov8m-search-baseline"
    config_dir = temp_repo_root / "configs/submission"
    submission_dir.mkdir(parents=True)
    detector_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    (submission_dir / "run.py").write_text(
        "from pathlib import Path\n",
        encoding="utf-8",
    )
    (detector_dir / "train_summary.json").write_text("{}\n", encoding="utf-8")

    config_path = config_dir / "detector_only_missing_best.json"
    missing_relative_path = (
        "artifacts/models/detector/yolov8m-search-baseline/missing-best.pt"
    )
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "output_zip": "dist/ng_detector_only_submission.zip",
                "source_files": [
                    "submission/run.py",
                    missing_relative_path,
                    "artifacts/models/detector/yolov8m-search-baseline/train_summary.json",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with ASSERTIONS.assertRaisesRegex(
        SubmissionBuildError,
        f"Configured bundle source does not exist: {missing_relative_path}",
    ):
        build_submission_zip(config_path)


def test_detector_only_submission_build_fails_when_weight_file_limit_is_exceeded(
    tmp_path: Path,
) -> None:
    temp_repo_root = tmp_path / "repo"
    submission_dir = temp_repo_root / "submission"
    detector_dir = temp_repo_root / "artifacts/models/detector/yolov8m-search-baseline"
    extra_weights_dir = temp_repo_root / "artifacts/models/detector/extra"
    config_dir = temp_repo_root / "configs/submission"
    submission_dir.mkdir(parents=True)
    detector_dir.mkdir(parents=True)
    extra_weights_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    (submission_dir / "run.py").write_text(
        "from pathlib import Path\n",
        encoding="utf-8",
    )
    (detector_dir / "best.pt").write_bytes(b"detector")
    (detector_dir / "train_summary.json").write_text("{}\n", encoding="utf-8")
    (extra_weights_dir / "two.pt").write_bytes(b"two")
    (extra_weights_dir / "three.pt").write_bytes(b"three")
    (extra_weights_dir / "four.pt").write_bytes(b"four")

    config_path = config_dir / "detector_only_too_many_weights.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "output_zip": "dist/ng_detector_only_submission.zip",
                "source_files": [
                    "submission/run.py",
                    "artifacts/models/detector/yolov8m-search-baseline/best.pt",
                    "artifacts/models/detector/yolov8m-search-baseline/train_summary.json",
                    "artifacts/models/detector/extra/two.pt",
                    "artifacts/models/detector/extra/three.pt",
                    "artifacts/models/detector/extra/four.pt",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with ASSERTIONS.assertRaisesRegex(
        SubmissionBuildError,
        "Submission bundle exceeds max weight file count of 3.",
    ):
        build_submission_zip(config_path)
