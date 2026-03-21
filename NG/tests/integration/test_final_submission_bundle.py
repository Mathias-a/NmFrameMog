from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, cast

from src.ng_data.submission import build_submission_zip, run_smoke_submission
from src.ng_data.submission.budget_check import collect_budget_evidence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_final_submission_bundle_is_root_run_py_only_and_smoke_valid(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/submission/final.json"
    fixture_dir = repo_root / "tests/fixtures/submission_images"
    predictions_path = tmp_path / "final_predictions.json"

    zip_path = build_submission_zip(config_path)
    predictions = run_smoke_submission(zip_path, fixture_dir, predictions_path)
    budget_evidence = collect_budget_evidence(zip_path, fixture_dir)

    assert zip_path == repo_root / "dist/ng_submission.zip"
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()

    assert members == ["run.py"]
    assert all("/" not in member for member in members)
    assert predictions_path.is_file()

    payload = cast(
        list[dict[str, Any]], json.loads(predictions_path.read_text(encoding="utf-8"))
    )
    assert payload == predictions
    assert payload == []
    assert budget_evidence["status"] == "pass"
    assert budget_evidence["failed_checks"] == []
    assert budget_evidence["observed"]["prediction_count"] == len(payload)
    assert cast(float, budget_evidence["observed"]["smoke_runtime_seconds"]) >= 0.0
