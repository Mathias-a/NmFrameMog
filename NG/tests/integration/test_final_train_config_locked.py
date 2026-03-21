from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from src.ng_data.pipeline.final_train import main as final_train_main


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_final_train_writes_truthful_blocked_manifest_from_locked_config(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/final/frozen_pipeline.json"
    out_path = tmp_path / "artifacts/release/final_manifest.json"

    exit_code = final_train_main(
        [
            "--config",
            str(config_path),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    payload = cast(dict[str, Any], json.loads(out_path.read_text(encoding="utf-8")))

    assert payload["schema_version"] == 1
    assert payload["status"] == "blocked_hold_manifest"
    assert payload["final_training"] == {
        "attempted": False,
        "blocked": True,
        "hold_manifest_only": True,
        "promotion_applied": False,
        "selected_variant": "detector_only",
        "selected_variant_eligible": False,
        "selected_variant_must_be_eligible": True,
        "blocking_reasons": [
            "No variant has promotion-grade end-to-end evidence, so the "
            "baseline stays detector-only.",
            "detector evaluation.mode=smoke is not promotion-grade evidence",
            "detector export.placeholder=true is not promotion-grade evidence",
            "detector predictions are detection-only category_id=0 outputs",
            "detector train summary mode=smoke blocks real final retraining",
            "detector train summary training.placeholder=true blocks real "
            "final retraining",
        ],
    }
    assert (
        payload["artifact_references"]["decision"]["path"]
        == "artifacts/eval/final_variant_decision.json"
    )
    assert payload["artifact_references"]["decision"]["decision"] == {
        "promotion_applied": False,
        "selected_variant": "detector_only",
        "selected_variant_eligible": False,
        "strategy": "keep_detector_only",
    }
    assert payload["artifact_references"]["detector_train_summary"]["summary"] == {
        "mode": "smoke",
        "output_dir": "artifacts/models/detector/yolov8m-search-baseline",
        "run_name": "yolov8m-search-baseline",
        "runtime_version": "detector-train-smoke-v1",
        "training_placeholder": True,
    }
    assert payload["artifact_references"]["classifier_train_summary"]["summary"] == {
        "mode": "timm_classifier_baseline",
        "output_dir": "artifacts/models/classifier",
        "weights_artifact_name": "best.pt",
    }
    assert payload["release_manifest"] == {
        "kind": "final_manifest",
        "path": str(out_path),
    }
    assert payload["frozen_config"]["path"] == str(config_path)
    assert payload["provenance"]["referenced_files"] == {
        "classifier_train_summary": {
            "sha256": (
                "90e90e9539d39a77a161fb72f49ec4d1080bc11d0d0653f5f3f196742e290784"
            ),
            "size_bytes": 2205,
        },
        "decision": {
            "sha256": (
                "03858f1ee47f6577f1734b932b25fdc5814ed273c89585e51d6d61f324dfbee2"
            ),
            "size_bytes": 2196,
        },
        "detector_train_summary": {
            "sha256": (
                "7494188697fbe0ecaa28f3bdb4f8b446ea757cf6fda29a5901b9bdf030f5401b"
            ),
            "size_bytes": 1419,
        },
    }
