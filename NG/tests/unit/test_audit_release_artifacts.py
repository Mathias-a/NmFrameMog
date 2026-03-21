from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.ng_data.data.manifest import file_snapshot, write_json
from src.ng_data.pipeline.audit_release_artifacts import (
    main as audit_release_artifacts_main,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _copy_release_audit_repo(tmp_path: Path) -> Path:
    repo_root = _repo_root()
    copied_root = tmp_path / "repo"
    for relative_path in (
        "artifacts/eval/final_variant_decision.json",
        "artifacts/models/classifier/train_summary.json",
        "artifacts/models/detector/yolov8m-search-baseline/train_summary.json",
        "artifacts/release/final_manifest.json",
        "configs/final/frozen_pipeline.json",
    ):
        source = repo_root / relative_path
        destination = copied_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    frozen_config_path = copied_root / "configs/final/frozen_pipeline.json"
    release_manifest_path = copied_root / "artifacts/release/final_manifest.json"
    frozen_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))

    frozen_config["artifacts"]["classifier_train_summary"]["snapshot"] = file_snapshot(
        copied_root / "artifacts/models/classifier/train_summary.json"
    )
    frozen_config["artifacts"]["decision"]["snapshot"] = file_snapshot(
        copied_root / "artifacts/eval/final_variant_decision.json"
    )
    frozen_config["artifacts"]["detector_train_summary"]["snapshot"] = file_snapshot(
        copied_root
        / "artifacts/models/detector/yolov8m-search-baseline/train_summary.json"
    )
    write_json(frozen_config_path, frozen_config)

    release_manifest["frozen_config"]["snapshot"] = file_snapshot(frozen_config_path)
    release_manifest["artifact_references"]["classifier_train_summary"]["snapshot"] = (
        file_snapshot(copied_root / "artifacts/models/classifier/train_summary.json")
    )
    release_manifest["artifact_references"]["decision"]["snapshot"] = file_snapshot(
        copied_root / "artifacts/eval/final_variant_decision.json"
    )
    release_manifest["artifact_references"]["detector_train_summary"]["snapshot"] = (
        file_snapshot(
            copied_root
            / "artifacts/models/detector/yolov8m-search-baseline/train_summary.json"
        )
    )
    release_manifest["final_training"]["selected_variant_must_be_eligible"] = True
    release_manifest["release_manifest"] = {
        "kind": "final_manifest",
        "path": release_manifest_path.as_posix(),
    }
    release_manifest["provenance"]["cwd"] = str(copied_root)
    release_manifest["provenance"]["referenced_files"] = {
        "classifier_train_summary": file_snapshot(
            copied_root / "artifacts/models/classifier/train_summary.json"
        ),
        "decision": file_snapshot(
            copied_root / "artifacts/eval/final_variant_decision.json"
        ),
        "detector_train_summary": file_snapshot(
            copied_root
            / "artifacts/models/detector/yolov8m-search-baseline/train_summary.json"
        ),
    }
    write_json(release_manifest_path, release_manifest)
    return copied_root


def test_audit_release_artifacts_accepts_truthful_blocked_hold_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    copied_root = _copy_release_audit_repo(tmp_path)

    exit_code = audit_release_artifacts_main(
        ["--manifest", str(copied_root / "artifacts/release/final_manifest.json")]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["release_state"] == "blocked_hold_manifest"
    assert payload["checked_files"] == [
        {
            "label": "frozen_config",
            "path": (copied_root / "configs/final/frozen_pipeline.json").as_posix(),
            "sha256": file_snapshot(copied_root / "configs/final/frozen_pipeline.json")[
                "sha256"
            ],
            "size_bytes": file_snapshot(
                copied_root / "configs/final/frozen_pipeline.json"
            )["size_bytes"],
        },
        {
            "label": "classifier_train_summary",
            "path": (
                copied_root / "artifacts/models/classifier/train_summary.json"
            ).as_posix(),
            "sha256": file_snapshot(
                copied_root / "artifacts/models/classifier/train_summary.json"
            )["sha256"],
            "size_bytes": file_snapshot(
                copied_root / "artifacts/models/classifier/train_summary.json"
            )["size_bytes"],
        },
        {
            "label": "decision",
            "path": (
                copied_root / "artifacts/eval/final_variant_decision.json"
            ).as_posix(),
            "sha256": file_snapshot(
                copied_root / "artifacts/eval/final_variant_decision.json"
            )["sha256"],
            "size_bytes": file_snapshot(
                copied_root / "artifacts/eval/final_variant_decision.json"
            )["size_bytes"],
        },
        {
            "label": "detector_train_summary",
            "path": (
                copied_root
                / "artifacts/models/detector/yolov8m-search-baseline/train_summary.json"
            ).as_posix(),
            "sha256": file_snapshot(
                copied_root
                / "artifacts/models/detector/yolov8m-search-baseline/train_summary.json"
            )["sha256"],
            "size_bytes": file_snapshot(
                copied_root
                / "artifacts/models/detector/yolov8m-search-baseline/train_summary.json"
            )["size_bytes"],
        },
    ]
    assert payload["cannot_validate"] == [
        (
            "release-grade final weights cannot be validated because final "
            "retraining was blocked and never ran"
        ),
        (
            "release-grade final-training provenance cannot be validated because "
            "final retraining was blocked and never ran"
        ),
    ]


def test_audit_release_artifacts_rejects_snapshot_mismatch(
    tmp_path: Path,
) -> None:
    copied_root = _copy_release_audit_repo(tmp_path)
    decision_path = copied_root / "artifacts/eval/final_variant_decision.json"
    original = decision_path.read_text(encoding="utf-8")
    decision_path.write_text(original + " \n", encoding="utf-8")

    try:
        audit_release_artifacts_main(
            ["--manifest", str(copied_root / "artifacts/release/final_manifest.json")]
        )
    except SystemExit as error:
        assert "Artifact snapshot mismatch for 'decision'" in str(error)
    else:
        raise AssertionError("Expected audit to fail on referenced artifact drift")


def test_audit_release_artifacts_rejects_provenance_cwd_drift(
    tmp_path: Path,
) -> None:
    copied_root = _copy_release_audit_repo(tmp_path)
    manifest_path = copied_root / "artifacts/release/final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["cwd"] = "/tmp/not-the-real-repo"
    write_json(manifest_path, manifest)

    try:
        audit_release_artifacts_main(["--manifest", str(manifest_path)])
    except SystemExit as error:
        assert "Release manifest provenance.cwd does not match" in str(error)
    else:
        raise AssertionError("Expected audit to fail on provenance cwd drift")
