from __future__ import annotations

import json
from pathlib import Path

from src.ng_data.data.manifest import write_json
from src.ng_data.pipeline.final_train import main as final_train_main


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_final_train_rejects_frozen_config_drift(tmp_path: Path) -> None:
    repo_root = _repo_root()
    config_payload = json.loads(
        (repo_root / "configs/final/frozen_pipeline.json").read_text(encoding="utf-8")
    )
    config_payload["artifacts"]["decision"]["snapshot"]["sha256"] = "0" * 64
    drifted_config_path = tmp_path / "configs/final/frozen_pipeline.json"
    write_json(drifted_config_path, config_payload)

    try:
        final_train_main(
            [
                "--config",
                str(drifted_config_path),
                "--out",
                str(tmp_path / "artifacts/release/final_manifest.json"),
            ]
        )
    except SystemExit as error:
        assert "Frozen config drift detected for decision" in str(error)
    else:
        raise AssertionError("Expected frozen config drift to abort final_train")


def test_final_train_rejects_ineligible_evidence_without_hold_mode(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    config_payload = json.loads(
        (repo_root / "configs/final/frozen_pipeline.json").read_text(encoding="utf-8")
    )
    config_payload["final_train"]["hold_manifest_only"] = False
    strict_config_path = tmp_path / "configs/final/frozen_pipeline.json"
    write_json(strict_config_path, config_payload)

    try:
        final_train_main(
            [
                "--config",
                str(strict_config_path),
                "--out",
                str(tmp_path / "artifacts/release/final_manifest.json"),
            ]
        )
    except SystemExit as error:
        assert str(error) == (
            "Final retraining blocked: decision.selected_variant_eligible=false and "
            "frozen config requires an eligible selected variant."
        )
    else:
        raise AssertionError("Expected ineligible evidence to abort final_train")
