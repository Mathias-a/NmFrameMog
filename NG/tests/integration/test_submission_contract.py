from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, cast

from src.ng_data.submission import (
    build_submission_zip,
    infer_image_id,
    list_image_files,
    run_smoke_submission,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_submission_smoke_bundle_runs_offline_against_local_fixtures(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/submission/smoke.json"
    fixture_dir = repo_root / "tests/fixtures/submission_images"
    output_path = tmp_path / "predictions.json"

    zip_path = build_submission_zip(config_path)
    predictions = run_smoke_submission(zip_path, fixture_dir, output_path)

    assert zip_path.is_file()
    assert output_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["run.py"]

    payload = cast(
        list[dict[str, Any]], json.loads(output_path.read_text(encoding="utf-8"))
    )
    fixture_files = list_image_files(fixture_dir)
    expected_image_ids = [
        infer_image_id(image_path, fallback_id)
        for fallback_id, image_path in enumerate(fixture_files, start=1)
    ]

    assert len(predictions) == len(fixture_files) == len(payload)
    assert [prediction["image_id"] for prediction in predictions] == expected_image_ids
    assert payload == predictions
    assert payload == [
        {
            "image_id": 1,
            "category_id": 0,
            "bbox": [13.0, 17.0, 48.0, 34.0],
            "score": 0.58,
        },
        {
            "image_id": 2,
            "category_id": 0,
            "bbox": [26.0, 34.0, 56.0, 40.0],
            "score": 0.61,
        },
        {
            "image_id": 7,
            "category_id": 0,
            "bbox": [91.0, 39.0, 56.0, 28.0],
            "score": 0.76,
        },
    ]


def test_submission_bundle_keeps_run_py_at_zip_root() -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/submission/smoke.json"
    zip_path = build_submission_zip(config_path)

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()

    assert members[0] == "run.py"
    assert all("/" not in member for member in members)
