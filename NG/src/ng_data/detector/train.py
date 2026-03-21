from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from src.ng_data.data.manifest import (
    directory_snapshot,
    file_snapshot,
    load_manifest,
    write_json,
)
from src.ng_data.detector.config import (
    DetectorConfig,
    DetectorConfigValidationError,
    load_and_validate_detector_config,
)
from src.ng_data.detector.prepare_dataset import prepare_detector_dataset

JsonDict = dict[str, Any]
PLACEHOLDER_MODE = "smoke"
REAL_MODE = "real"
PLACEHOLDER_RUNTIME_VERSION = "detector-train-smoke-v1"
EXPECTED_RUN_NAME = "yolov8m-search-baseline"
ULTRALYTICS_PROJECT = Path("artifacts/runs/detector")


class DetectorTrainSmokeError(ValueError):
    pass


class TrainArgs(argparse.Namespace):
    config: str
    manifest: str
    mode: str
    splits: str
    output_dir: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write deterministic placeholder detector training artifacts."
    )
    parser.add_argument(
        "--config",
        default="configs/detector/yolov8m-search.json",
        help="Path to the detector config JSON file.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifests/dataset_manifest.json",
        help="Path to the processed dataset manifest JSON file.",
    )
    parser.add_argument(
        "--mode",
        choices=(PLACEHOLDER_MODE, REAL_MODE),
        default=PLACEHOLDER_MODE,
        help="Whether to run deterministic smoke training or real Ultralytics training.",
    )
    parser.add_argument(
        "--splits",
        default="data/processed/manifests/splits.json",
        help="Path to the processed split manifest JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/detector",
        help="Directory where normalized detector training artifacts should be written.",
    )
    return parser


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise DetectorTrainSmokeError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_list(data: JsonDict, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise DetectorTrainSmokeError(f"Expected '{key}' to be a list.")
    return cast(list[object], value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise DetectorTrainSmokeError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DetectorTrainSmokeError(f"Expected '{key}' to be an integer.")
    return value


def _load_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DetectorTrainSmokeError(f"Invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise DetectorTrainSmokeError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _search_summary(config: DetectorConfig) -> JsonDict:
    return {
        "batch_size": config.search.batch_size,
        "device": config.search.device,
        "epochs": config.search.epochs,
        "image_size": config.search.image_size,
        "patience": config.search.patience,
        "run_name": config.search.run_name,
    }


def _runtime_summary(config: DetectorConfig) -> JsonDict:
    return {
        "export_format": config.runtime.export_format,
        "framework": config.runtime.framework,
        "version": config.runtime.version,
    }


def _validate_run_name(config: DetectorConfig) -> None:
    if config.search.run_name != EXPECTED_RUN_NAME:
        raise DetectorTrainSmokeError(
            "Detector training requires "
            f"search.run_name='{EXPECTED_RUN_NAME}', got "
            f"'{config.search.run_name}'."
        )


def _prepared_dataset_root(run_name: str) -> Path:
    return ULTRALYTICS_PROJECT / run_name / "dataset"


def _ultralytics_run_dir(run_name: str) -> Path:
    return ULTRALYTICS_PROJECT / run_name


def _ultralytics_best_weights_path(run_name: str) -> Path:
    return _ultralytics_run_dir(run_name) / "weights" / "best.pt"


def _normalized_output_paths(
    output_dir: Path, run_name: str
) -> tuple[Path, Path, Path]:
    run_dir = output_dir / run_name
    return run_dir, run_dir / "best.pt", run_dir / "train_summary.json"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_training_metrics(result: object) -> JsonDict:
    results_dict = getattr(result, "results_dict", None)
    if isinstance(results_dict, dict):
        return cast(JsonDict, _json_safe(results_dict))
    if isinstance(result, dict):
        return cast(JsonDict, _json_safe(result))
    return {}


def _load_ultralytics_yolo() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError as error:
        raise DetectorTrainSmokeError(
            "Real detector training requires ultralytics==8.1.0 to be installed."
        ) from error
    return YOLO


def _run_ultralytics_training(
    *, config: DetectorConfig, prepared_dataset_yaml: str
) -> object:
    return _load_ultralytics_yolo()(config.model.weights).train(
        data=prepared_dataset_yaml,
        epochs=config.search.epochs,
        imgsz=config.search.image_size,
        batch=config.search.batch_size,
        device=config.search.device,
        patience=config.search.patience,
        project="artifacts/runs/detector",
        name=config.search.run_name,
        exist_ok=True,
        verbose=False,
    )


def _copy_best_weights(*, source_path: Path, destination_path: Path) -> None:
    if not source_path.exists() or not source_path.is_file():
        raise DetectorTrainSmokeError(
            "Ultralytics training did not produce best.pt at the expected path: "
            f"{source_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _validate_dataset_inputs(manifest_path: Path) -> tuple[JsonDict, str, Path]:
    manifest = load_manifest(manifest_path)
    if _require_int(manifest, "schema_version") != 1:
        raise DetectorTrainSmokeError("Unsupported processed manifest schema version.")

    counts = _require_mapping(manifest, "counts")
    for key in (
        "annotation_count",
        "category_count",
        "image_count",
        "reference_product_count",
    ):
        _require_int(counts, key)

    processed_outputs = _require_mapping(manifest, "processed_outputs")
    annotations_entry = _require_mapping(processed_outputs, "annotations")
    annotations_relative = _require_string(annotations_entry, "path")
    processed_root = manifest_path.parents[1]
    annotations_path = processed_root / annotations_relative
    if not annotations_path.exists() or not annotations_path.is_file():
        raise DetectorTrainSmokeError(
            f"Processed annotations file does not exist: {annotations_path}"
        )

    return manifest, annotations_relative, annotations_path


def _validate_splits_inputs(
    *, manifest_path: Path, expected_annotations_path: str, splits_path: Path
) -> JsonDict:
    payload = _load_json_object(splits_path)
    if _require_int(payload, "schema_version") != 1:
        raise DetectorTrainSmokeError("Unsupported split manifest schema version.")

    counts = _require_mapping(payload, "counts")
    for key in ("cv_folds", "cv_pool_images", "holdout_images", "total_images"):
        _require_int(counts, key)

    _require_int(payload, "seed")
    _require_mapping(payload, "holdout")
    _require_list(payload, "folds")
    _require_list(payload, "cv_pool_image_ids")

    source_manifest = _require_string(payload, "source_manifest")
    source_annotations = _require_string(payload, "source_annotations")

    processed_root = manifest_path.parents[1]
    expected_manifest = manifest_path.relative_to(processed_root).as_posix()
    if source_manifest != expected_manifest:
        raise DetectorTrainSmokeError(
            "Split manifest source_manifest does not match the processed dataset "
            f"manifest: expected '{expected_manifest}', got '{source_manifest}'."
        )
    if source_annotations != expected_annotations_path:
        raise DetectorTrainSmokeError(
            "Split manifest source_annotations does not match the processed "
            f"annotations path: expected '{expected_annotations_path}', got "
            f"'{source_annotations}'."
        )

    return payload


def _build_placeholder_payload(
    *,
    config: DetectorConfig,
    dataset_manifest: JsonDict,
    split_manifest: JsonDict,
    annotations_path: Path,
) -> JsonDict:
    counts = _require_mapping(dataset_manifest, "counts")
    split_counts = _require_mapping(split_manifest, "counts")
    return {
        "annotation_count": _require_int(counts, "annotation_count"),
        "annotations_path": annotations_path.as_posix(),
        "batch_size": config.search.batch_size,
        "cv_folds": _require_int(split_counts, "cv_folds"),
        "device": config.search.device,
        "epochs": config.search.epochs,
        "holdout_images": _require_int(split_counts, "holdout_images"),
        "image_count": _require_int(counts, "image_count"),
        "image_size": config.search.image_size,
        "mode": PLACEHOLDER_MODE,
        "model_name": config.model.name,
        "patience": config.search.patience,
        "run_name": config.search.run_name,
        "runtime": {
            "export_format": config.runtime.export_format,
            "framework": config.runtime.framework,
            "version": config.runtime.version,
        },
        "seed": _require_int(split_manifest, "seed"),
    }


def _build_placeholder_weights_bytes(payload: JsonDict) -> bytes:
    payload_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload_digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    header = (
        "placeholder-detector-weights\n"
        f"contract_version={PLACEHOLDER_RUNTIME_VERSION}\n"
        f"payload_sha256={payload_digest}\n"
    )
    return (header + payload_text).encode("utf-8")


def run_smoke_training(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
) -> JsonDict:
    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    splits_path = Path(splits_path)
    output_dir = Path(output_dir)

    config = load_and_validate_detector_config(config_path)
    _validate_run_name(config)
    dataset_manifest, annotations_relative, annotations_path = _validate_dataset_inputs(
        manifest_path
    )
    split_manifest = _validate_splits_inputs(
        manifest_path=manifest_path,
        expected_annotations_path=annotations_relative,
        splits_path=splits_path,
    )

    run_dir = output_dir / config.search.run_name
    best_weights_path = run_dir / "best.pt"
    summary_path = run_dir / "train_summary.json"
    placeholder_payload = _build_placeholder_payload(
        config=config,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
        annotations_path=annotations_path,
    )
    placeholder_bytes = _build_placeholder_weights_bytes(placeholder_payload)

    run_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path.write_bytes(placeholder_bytes)

    summary: JsonDict = {
        "config_path": config_path.as_posix(),
        "mode": PLACEHOLDER_MODE,
        "model_name": config.model.name,
        "output_artifacts": {
            "best_weights": best_weights_path.as_posix(),
            "summary_json": summary_path.as_posix(),
        },
        "output_dir": run_dir.as_posix(),
        "placeholder_bytes": len(placeholder_bytes),
        "placeholder_sha256": hashlib.sha256(placeholder_bytes).hexdigest(),
        "processed_manifest_path": manifest_path.as_posix(),
        "runtime": _runtime_summary(config),
        "runtime_version": PLACEHOLDER_RUNTIME_VERSION,
        "search": _search_summary(config),
        "split_counts": _require_mapping(split_manifest, "counts"),
        "splits_path": splits_path.as_posix(),
        "source_annotations": _require_string(split_manifest, "source_annotations"),
        "source_manifest": _require_string(split_manifest, "source_manifest"),
        "summary_format": "json",
        "training": {
            "counts": _require_mapping(dataset_manifest, "counts"),
            "placeholder": True,
            "weights": config.model.weights,
        },
    }
    write_json(summary_path, summary)
    return summary


def run_real_training(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
) -> JsonDict:
    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    splits_path = Path(splits_path)
    output_dir = Path(output_dir)

    config = load_and_validate_detector_config(config_path)
    _validate_run_name(config)
    dataset_manifest, annotations_relative, annotations_path = _validate_dataset_inputs(
        manifest_path
    )
    split_manifest = _validate_splits_inputs(
        manifest_path=manifest_path,
        expected_annotations_path=annotations_relative,
        splits_path=splits_path,
    )

    prepared_dataset = prepare_detector_dataset(
        config_path=config_path,
        manifest_path=manifest_path,
        out_path=_prepared_dataset_root(config.search.run_name),
        splits_path=splits_path,
    )
    prepared_dataset_yaml = _require_string(prepared_dataset, "dataset_yaml")

    started_at_utc = _utc_timestamp()
    started_at = time.perf_counter()
    train_result = _run_ultralytics_training(
        config=config,
        prepared_dataset_yaml=prepared_dataset_yaml,
    )
    duration_seconds = time.perf_counter() - started_at
    completed_at_utc = _utc_timestamp()

    ultralytics_best_weights = _ultralytics_best_weights_path(config.search.run_name)
    run_dir, best_weights_path, summary_path = _normalized_output_paths(
        output_dir, config.search.run_name
    )
    _copy_best_weights(
        source_path=ultralytics_best_weights,
        destination_path=best_weights_path,
    )

    prepared_dataset_dir = Path(_require_string(prepared_dataset, "output_dir"))
    summary: JsonDict = {
        "artifact_provenance": {
            "dataset_yaml": {
                "path": prepared_dataset_yaml,
                **file_snapshot(Path(prepared_dataset_yaml)),
            },
            "normalized_best_weights": {
                "path": best_weights_path.as_posix(),
                **file_snapshot(best_weights_path),
            },
            "prepared_dataset_dir": {
                "path": prepared_dataset_dir.as_posix(),
                **directory_snapshot(prepared_dataset_dir),
            },
            "ultralytics_best_weights": {
                "path": ultralytics_best_weights.as_posix(),
                **file_snapshot(ultralytics_best_weights),
            },
        },
        "config_path": config_path.as_posix(),
        "dataset_preparation": {
            "dataset_yaml": prepared_dataset_yaml,
            "names": prepared_dataset.get("names"),
            "nc": prepared_dataset.get("nc"),
            "output_dir": _require_string(prepared_dataset, "output_dir"),
            "source_annotations": _require_string(
                prepared_dataset, "source_annotations"
            ),
            "source_manifest": _require_string(prepared_dataset, "source_manifest"),
            "split_counts": _require_mapping(prepared_dataset, "split_counts"),
            "splits_path": _require_string(prepared_dataset, "splits_path"),
        },
        "mode": REAL_MODE,
        "model_name": config.model.name,
        "output_artifacts": {
            "best_weights": best_weights_path.as_posix(),
            "summary_json": summary_path.as_posix(),
        },
        "output_dir": run_dir.as_posix(),
        "processed_manifest_path": manifest_path.as_posix(),
        "runtime": _runtime_summary(config),
        "runtime_version": config.runtime.version,
        "search": _search_summary(config),
        "split_counts": _require_mapping(split_manifest, "counts"),
        "splits_path": splits_path.as_posix(),
        "source_annotations": _require_string(split_manifest, "source_annotations"),
        "source_manifest": _require_string(split_manifest, "source_manifest"),
        "summary_format": "json",
        "training": {
            "counts": _require_mapping(dataset_manifest, "counts"),
            "metrics": _extract_training_metrics(train_result),
            "paths": {
                "best_weights": best_weights_path.as_posix(),
                "prepared_dataset_yaml": prepared_dataset_yaml,
                "ultralytics_best_weights": ultralytics_best_weights.as_posix(),
                "ultralytics_project_dir": ULTRALYTICS_PROJECT.as_posix(),
                "ultralytics_run_dir": _ultralytics_run_dir(
                    config.search.run_name
                ).as_posix(),
            },
            "placeholder": False,
            "runtime_summary": {
                "completed_at_utc": completed_at_utc,
                "duration_seconds": round(duration_seconds, 6),
                "started_at_utc": started_at_utc,
            },
            "weights": config.model.weights,
        },
    }
    write_json(summary_path, summary)
    return summary


def run_training(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    mode: str,
    splits_path: str | Path,
    output_dir: str | Path,
) -> JsonDict:
    if mode == PLACEHOLDER_MODE:
        return run_smoke_training(
            config_path=config_path,
            manifest_path=manifest_path,
            splits_path=splits_path,
            output_dir=output_dir,
        )
    if mode == REAL_MODE:
        return run_real_training(
            config_path=config_path,
            manifest_path=manifest_path,
            splits_path=splits_path,
            output_dir=output_dir,
        )
    raise DetectorTrainSmokeError(f"Unsupported detector training mode: {mode}.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(TrainArgs, parser.parse_args(argv))

    try:
        summary = run_training(
            config_path=args.config,
            manifest_path=args.manifest,
            mode=args.mode,
            splits_path=args.splits,
            output_dir=args.output_dir,
        )
    except (
        DetectorConfigValidationError,
        DetectorTrainSmokeError,
        ValueError,
    ) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
