from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from src.ng_data.cloud.config import ConfigValidationError, load_cloud_config
from src.ng_data.cloud.validation import validate_cloud_config
from src.ng_data.data.manifest import file_snapshot, load_manifest, write_json

JsonDict = dict[str, Any]
SCHEMA_VERSION = 1
RUN_NAME = "yolov8m-search-baseline"
DEFAULT_CONFIG_PATH = "configs/cloud/main.json"
DEFAULT_DETECTOR_CONFIG_PATH = "configs/detector/yolov8m-search.json"
DEFAULT_DATASET_MANIFEST_PATH = "data/processed/manifests/dataset_manifest.json"
DEFAULT_SPLIT_MANIFEST_PATH = "data/processed/manifests/splits.json"
DEFAULT_EVALUATION_REPORT_PATH = "artifacts/eval/detector_holdout_metrics.json"
DEFAULT_SUBMISSION_ZIP_PATH = "dist/ng_detector_only_submission.zip"


class RunManifestError(ValueError):
    pass


class RunManifestArgs(argparse.Namespace):
    config: str
    detector_output: str
    out: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a first-iteration detector run manifest with file snapshots."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the cloud config JSON file.",
    )
    parser.add_argument(
        "--detector-output",
        required=True,
        help="Directory containing the detector run artifacts.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the run manifest JSON output.",
    )
    return parser


def _load_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RunManifestError(f"Expected file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise RunManifestError(f"Invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise RunManifestError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise RunManifestError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise RunManifestError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RunManifestError(f"Expected '{key}' to be an integer.")
    return value


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)  # noqa: UP017
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _repo_root_from_cloud_config(config_path: Path) -> Path:
    resolved_config = config_path.resolve()
    if len(resolved_config.parents) >= 3:
        return resolved_config.parents[2]
    return Path.cwd().resolve()


def _resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else repo_root / path


def _relative_repo_path(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_repo_root = repo_root.resolve()
    try:
        return resolved_path.relative_to(resolved_repo_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _snapshot_entry(path: Path, repo_root: Path) -> JsonDict:
    return {
        "path": _relative_repo_path(path, repo_root),
        "snapshot": file_snapshot(path),
    }


def _optional_snapshot_entry(path: Path, repo_root: Path) -> JsonDict | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise RunManifestError(f"Expected a file path: {path}")
    return _snapshot_entry(path, repo_root)


def _validate_dataset_manifest(path: Path) -> None:
    manifest = load_manifest(path)
    if _require_int(manifest, "schema_version") != 1:
        raise RunManifestError("Unsupported dataset manifest schema version.")
    counts = _require_mapping(manifest, "counts")
    for key in (
        "annotation_count",
        "category_count",
        "image_count",
        "reference_product_count",
    ):
        _require_int(counts, key)


def _validate_split_manifest(path: Path) -> None:
    manifest = _load_json_object(path)
    if _require_int(manifest, "schema_version") != 1:
        raise RunManifestError("Unsupported split manifest schema version.")
    counts = _require_mapping(manifest, "counts")
    for key in ("cv_folds", "cv_pool_images", "holdout_images", "total_images"):
        _require_int(counts, key)


def _load_train_summary(path: Path) -> JsonDict:
    summary = _load_json_object(path)
    output_artifacts = _require_mapping(summary, "output_artifacts")
    _require_string(summary, "config_path")
    _require_string(summary, "output_dir")
    _require_string(_require_mapping(summary, "search"), "run_name")
    _require_string(output_artifacts, "best_weights")
    _require_string(output_artifacts, "summary_json")
    return summary


def build_run_manifest(
    *, config_path: Path, detector_output_path: Path, out_path: Path
) -> JsonDict:
    config = load_cloud_config(config_path)
    validate_cloud_config(config)
    repo_root = _repo_root_from_cloud_config(config_path)

    detector_output_resolved = detector_output_path.resolve()
    if not detector_output_resolved.exists() or not detector_output_resolved.is_dir():
        raise RunManifestError(
            f"Expected detector output directory does not exist: {detector_output_path}"
        )
    if detector_output_resolved.name != RUN_NAME:
        raise RunManifestError(
            "Detector output directory must be the canonical run directory "
            f"'{RUN_NAME}': {detector_output_path}"
        )

    best_weights_path = detector_output_resolved / "best.pt"
    train_summary_path = detector_output_resolved / "train_summary.json"
    train_summary = _load_train_summary(train_summary_path)
    if (
        _require_string(_require_mapping(train_summary, "search"), "run_name")
        != RUN_NAME
    ):
        raise RunManifestError(
            f"Detector train summary must declare search.run_name='{RUN_NAME}'."
        )

    output_artifacts = _require_mapping(train_summary, "output_artifacts")
    declared_best_weights = Path(
        _require_string(output_artifacts, "best_weights")
    ).resolve()
    declared_summary = Path(_require_string(output_artifacts, "summary_json")).resolve()
    if declared_best_weights != best_weights_path.resolve():
        raise RunManifestError(
            "Detector train summary best_weights path does not match the provided "
            f"detector output directory: expected '{best_weights_path.resolve()}', got "
            f"'{declared_best_weights}'."
        )
    if declared_summary != train_summary_path.resolve():
        raise RunManifestError(
            "Detector train summary summary_json path does not match the provided "
            "detector output directory: expected "
            f"'{train_summary_path.resolve()}', got "
            f"'{declared_summary}'."
        )

    detector_config_path = _resolve_repo_path(
        repo_root, _require_string(train_summary, "config_path")
    )
    dataset_manifest_path = _resolve_repo_path(
        repo_root, _require_string(train_summary, "processed_manifest_path")
    )
    split_manifest_path = _resolve_repo_path(
        repo_root, _require_string(train_summary, "splits_path")
    )
    evaluation_report_path = repo_root / DEFAULT_EVALUATION_REPORT_PATH
    submission_zip_path = repo_root / DEFAULT_SUBMISSION_ZIP_PATH

    _validate_dataset_manifest(dataset_manifest_path)
    _validate_split_manifest(split_manifest_path)

    created_at = _utc_timestamp()
    manifest: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "run_name": RUN_NAME,
        "created_at_utc": created_at,
        "updated_at_utc": created_at,
        "cloud": {
            "config_path": _relative_repo_path(config_path, repo_root),
            "project_id": config.project_id,
            "region": config.region,
            "zone": config.compute_engine.zone,
        },
        "detector": {
            "config": _snapshot_entry(detector_config_path, repo_root),
            "output_dir": _relative_repo_path(detector_output_resolved, repo_root),
        },
        "dataset_manifest_snapshot": _snapshot_entry(dataset_manifest_path, repo_root),
        "split_manifest_snapshot": _snapshot_entry(split_manifest_path, repo_root),
        "output_artifacts": {
            "best_pt": _snapshot_entry(best_weights_path, repo_root),
            "train_summary_json": _snapshot_entry(train_summary_path, repo_root),
            "evaluation_report": _optional_snapshot_entry(
                evaluation_report_path, repo_root
            ),
            "submission_zip": _optional_snapshot_entry(submission_zip_path, repo_root),
        },
        "provenance": {
            "cwd": repo_root.as_posix(),
            "referenced_files": {
                "cloud_config": file_snapshot(config_path),
                "dataset_manifest": file_snapshot(dataset_manifest_path),
                "detector_config": file_snapshot(detector_config_path),
                "split_manifest": file_snapshot(split_manifest_path),
                "train_summary": file_snapshot(train_summary_path),
            },
        },
        "run_manifest": {
            "kind": "detector_run_manifest",
            "path": _relative_repo_path(out_path, repo_root),
        },
    }
    if manifest["output_artifacts"]["evaluation_report"] is not None:
        manifest["provenance"]["referenced_files"]["evaluation_report"] = file_snapshot(
            evaluation_report_path
        )
    if manifest["output_artifacts"]["submission_zip"] is not None:
        manifest["provenance"]["referenced_files"]["submission_zip"] = file_snapshot(
            submission_zip_path
        )
    return manifest


def run_manifest(
    *, config_path: str | Path, detector_output_path: str | Path, out_path: str | Path
) -> JsonDict:
    config = Path(config_path)
    detector_output = Path(detector_output_path)
    out = Path(out_path)
    manifest = build_run_manifest(
        config_path=config,
        detector_output_path=detector_output,
        out_path=out,
    )
    write_json(out, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(RunManifestArgs, parser.parse_args(argv))
    try:
        manifest = run_manifest(
            config_path=args.config,
            detector_output_path=args.detector_output,
            out_path=args.out,
        )
    except (ConfigValidationError, RunManifestError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
