from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from src.ng_data.data.manifest import file_snapshot, write_json
from src.ng_data.detector.config import (
    SUPPORTED_DETECTOR_FRAMEWORK,
    SUPPORTED_DETECTOR_VERSION,
    SUPPORTED_EXPORT_FORMAT,
)
from src.ng_data.detector.train import (
    EXPECTED_RUN_NAME,
    PLACEHOLDER_MODE,
    PLACEHOLDER_RUNTIME_VERSION,
    REAL_MODE,
)
from src.ng_data.eval.score import ScoreValidationError, score_predictions

JsonDict = dict[str, Any]
PLACEHOLDER_WEIGHTS_HEADER = "placeholder-detector-weights"
DETECTION_ONLY_CATEGORY_ID = 0
PREDICTION_SCORE = 1.0
EVALUATION_SCHEMA_VERSION = 1
HOLDOUT_REAL_EVALUATION_MODE = "holdout_real"


class DetectorEvaluateSmokeError(ValueError):
    pass


class EvaluateArgs(argparse.Namespace):
    mode: str
    weights: str
    split: str
    out: str


@dataclass(frozen=True)
class SplitImage:
    file_name: str
    height: int
    image_id: int
    image_path: Path
    width: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate detector artifacts in smoke or real mode."
    )
    parser.add_argument(
        "--mode",
        choices=(PLACEHOLDER_MODE, REAL_MODE),
        default=PLACEHOLDER_MODE,
        help="Whether to run deterministic smoke evaluation or real model inference.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the detector weights artifact.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Path to the processed COCO-style annotations JSON file to score.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the evaluation metrics JSON output.",
    )
    return parser


def _load_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DetectorEvaluateSmokeError(
            f"Expected file does not exist: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise DetectorEvaluateSmokeError(f"Invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise DetectorEvaluateSmokeError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise DetectorEvaluateSmokeError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_list(data: JsonDict, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise DetectorEvaluateSmokeError(f"Expected '{key}' to be a list.")
    return cast(list[object], value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise DetectorEvaluateSmokeError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DetectorEvaluateSmokeError(f"Expected '{key}' to be an integer.")
    return value


def _require_number(data: JsonDict, key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DetectorEvaluateSmokeError(f"Expected '{key}' to be numeric.")
    return float(value)


def _require_bool(data: JsonDict, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise DetectorEvaluateSmokeError(f"Expected '{key}' to be a boolean.")
    return value


def _require_bbox(data: JsonDict, key: str) -> list[float]:
    value = data.get(key)
    if not isinstance(value, list) or len(value) != 4:
        raise DetectorEvaluateSmokeError(f"Expected '{key}' to be a four-value list.")
    coordinates: list[float] = []
    for index, coordinate in enumerate(value):
        if not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool):
            raise DetectorEvaluateSmokeError(
                f"Expected '{key}[{index}]' to be numeric."
            )
        numeric_coordinate = float(coordinate)
        if numeric_coordinate < 0:
            raise DetectorEvaluateSmokeError(
                f"Expected '{key}[{index}]' to be non-negative."
            )
        coordinates.append(numeric_coordinate)
    if coordinates[2] <= 0 or coordinates[3] <= 0:
        raise DetectorEvaluateSmokeError(
            f"Expected '{key}' width and height to be greater than zero."
        )
    return coordinates


def _resolve_input_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _validate_relative_posix_path(label: str, value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute():
        raise DetectorEvaluateSmokeError(
            f"Expected '{label}' to be a relative path, got '{value}'."
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise DetectorEvaluateSmokeError(
            f"Expected '{label}' to avoid empty, '.' or '..' path segments."
        )
    return value


def _validate_supported_runtime(runtime: JsonDict) -> None:
    framework = _require_string(runtime, "framework")
    version = _require_string(runtime, "version")
    export_format = _require_string(runtime, "export_format")
    if framework != SUPPORTED_DETECTOR_FRAMEWORK:
        raise DetectorEvaluateSmokeError(
            "Unsupported detector framework in smoke weights: "
            f"{framework}. Expected '{SUPPORTED_DETECTOR_FRAMEWORK}'."
        )
    if (
        version != SUPPORTED_DETECTOR_VERSION
        or export_format != SUPPORTED_EXPORT_FORMAT
    ):
        raise DetectorEvaluateSmokeError(
            "Unsupported detector runtime version/export combination in smoke "
            f"weights: {framework}=={version} with export_format='{export_format}'. "
            f"Expected {SUPPORTED_DETECTOR_FRAMEWORK}=={SUPPORTED_DETECTOR_VERSION} "
            f"with export_format='{SUPPORTED_EXPORT_FORMAT}'."
        )


def _load_placeholder_payload(weights_path: Path) -> JsonDict:
    if not weights_path.exists() or not weights_path.is_file():
        raise DetectorEvaluateSmokeError(
            f"Placeholder detector weights file does not exist: {weights_path}"
        )

    content = weights_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    if len(lines) < 4:
        raise DetectorEvaluateSmokeError(
            f"Incomplete placeholder detector weights contract: {weights_path}"
        )
    if lines[0].rstrip("\n") != PLACEHOLDER_WEIGHTS_HEADER:
        raise DetectorEvaluateSmokeError(
            f"Unsupported detector weights header in {weights_path}"
        )

    contract_line = lines[1].rstrip("\n")
    if contract_line != f"contract_version={PLACEHOLDER_RUNTIME_VERSION}":
        raise DetectorEvaluateSmokeError(
            "Unsupported placeholder detector contract version: "
            f"{contract_line.removeprefix('contract_version=')}"
        )

    digest_line = lines[2].rstrip("\n")
    digest_prefix = "payload_sha256="
    if not digest_line.startswith(digest_prefix):
        raise DetectorEvaluateSmokeError(
            f"Missing payload digest in placeholder detector weights: {weights_path}"
        )
    payload_text = "".join(lines[3:])
    payload_sha256 = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    if payload_sha256 != digest_line.removeprefix(digest_prefix):
        raise DetectorEvaluateSmokeError(
            f"Placeholder detector payload digest mismatch for {weights_path}"
        )

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise DetectorEvaluateSmokeError(
            f"Invalid placeholder detector payload JSON in {weights_path}"
        ) from error
    if not isinstance(payload, dict):
        raise DetectorEvaluateSmokeError(
            f"Expected placeholder detector payload object in {weights_path}"
        )

    placeholder_payload = cast(JsonDict, payload)
    if _require_string(placeholder_payload, "mode") != PLACEHOLDER_MODE:
        raise DetectorEvaluateSmokeError(
            "Detector smoke evaluation expects mode='smoke'."
        )
    _require_string(placeholder_payload, "model_name")
    _require_string(placeholder_payload, "run_name")
    _require_string(placeholder_payload, "annotations_path")
    _require_int(placeholder_payload, "annotation_count")
    _require_int(placeholder_payload, "image_count")
    _require_int(placeholder_payload, "cv_folds")
    _require_int(placeholder_payload, "holdout_images")
    _require_int(placeholder_payload, "seed")
    _validate_supported_runtime(_require_mapping(placeholder_payload, "runtime"))
    return placeholder_payload


def _load_ultralytics_yolo() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError as error:
        raise DetectorEvaluateSmokeError(
            "Real detector evaluation requires ultralytics==8.1.0 to be installed."
        ) from error
    return YOLO


def _validate_train_summary(
    weights_path: Path, placeholder_payload: JsonDict
) -> JsonDict:
    summary_path = weights_path.with_name("train_summary.json")
    summary = _load_json_object(summary_path)
    if _require_string(summary, "mode") != PLACEHOLDER_MODE:
        raise DetectorEvaluateSmokeError(
            f"Detector train summary must declare mode='{PLACEHOLDER_MODE}'."
        )
    if _require_string(summary, "runtime_version") != PLACEHOLDER_RUNTIME_VERSION:
        raise DetectorEvaluateSmokeError(
            "Detector train summary runtime_version does not match the smoke "
            "placeholder contract."
        )

    training = _require_mapping(summary, "training")
    if not _require_bool(training, "placeholder"):
        raise DetectorEvaluateSmokeError(
            "Detector smoke evaluation only supports placeholder training artifacts."
        )

    output_artifacts = _require_mapping(summary, "output_artifacts")
    summary_weights_path = _resolve_input_path(
        _require_string(output_artifacts, "best_weights")
    )
    if summary_weights_path != weights_path.resolve():
        raise DetectorEvaluateSmokeError(
            "Detector train summary best_weights path does not match the provided "
            "weights file: expected "
            f"'{weights_path.resolve()}', got '{summary_weights_path}'."
        )

    summary_runtime = _require_mapping(summary, "runtime")
    _validate_supported_runtime(summary_runtime)
    payload_runtime = _require_mapping(placeholder_payload, "runtime")
    for key in ("framework", "version", "export_format"):
        if _require_string(summary_runtime, key) != _require_string(
            payload_runtime, key
        ):
            raise DetectorEvaluateSmokeError(
                "Detector train summary runtime "
                f"'{key}' does not match weights payload."
            )

    return summary


def _validate_real_train_summary(weights_path: Path) -> JsonDict:
    if not weights_path.exists() or not weights_path.is_file():
        raise DetectorEvaluateSmokeError(
            f"Real detector weights file does not exist: {weights_path}"
        )

    summary_path = weights_path.with_name("train_summary.json")
    summary = _load_json_object(summary_path)
    if _require_string(summary, "mode") != REAL_MODE:
        raise DetectorEvaluateSmokeError(
            "Detector train summary must declare "
            f"mode='{REAL_MODE}' for real evaluation."
        )

    training = _require_mapping(summary, "training")
    if _require_bool(training, "placeholder"):
        raise DetectorEvaluateSmokeError(
            "Real detector evaluation requires non-placeholder training artifacts."
        )

    output_artifacts = _require_mapping(summary, "output_artifacts")
    summary_weights_path = _resolve_input_path(
        _require_string(output_artifacts, "best_weights")
    )
    if summary_weights_path != weights_path.resolve():
        raise DetectorEvaluateSmokeError(
            "Detector train summary best_weights path does not match the provided "
            "weights file: expected "
            f"'{weights_path.resolve()}', got '{summary_weights_path}'."
        )

    runtime = _require_mapping(summary, "runtime")
    _validate_supported_runtime(runtime)

    search = _require_mapping(summary, "search")
    if _require_string(search, "run_name") != EXPECTED_RUN_NAME:
        raise DetectorEvaluateSmokeError(
            f"Real detector evaluation requires search.run_name='{EXPECTED_RUN_NAME}'."
        )
    _require_string(summary, "model_name")
    _require_string(summary, "output_dir")
    return summary


def _load_split_annotations(
    split_path: Path, placeholder_payload: JsonDict
) -> JsonDict:
    split_payload = _load_json_object(split_path)
    _require_list(split_payload, "annotations")
    _require_list(split_payload, "categories")
    _require_list(split_payload, "images")

    payload_annotations_path = _resolve_input_path(
        _require_string(placeholder_payload, "annotations_path")
    )
    if payload_annotations_path != split_path.resolve():
        raise DetectorEvaluateSmokeError(
            "Detector smoke weights were produced for a different annotations split: "
            f"expected '{payload_annotations_path}', got '{split_path.resolve()}'."
        )
    return split_payload


def _load_real_split_annotations(split_path: Path) -> JsonDict:
    split_payload = _load_json_object(split_path)
    _require_list(split_payload, "annotations")
    _require_list(split_payload, "categories")
    _require_list(split_payload, "images")
    return split_payload


def _annotation_sort_key(
    annotation: JsonDict,
) -> tuple[int, int, int, tuple[float, float, float, float]]:
    annotation_id = annotation.get("id", 0)
    if not isinstance(annotation_id, int) or isinstance(annotation_id, bool):
        raise DetectorEvaluateSmokeError("Expected annotation 'id' to be an integer.")
    bbox = _require_bbox(annotation, "bbox")
    return (
        _require_int(annotation, "image_id"),
        annotation_id,
        _require_int(annotation, "category_id"),
        cast(tuple[float, float, float, float], tuple(bbox)),
    )


def build_smoke_predictions(split_payload: JsonDict) -> list[JsonDict]:
    annotations = _require_list(split_payload, "annotations")
    parsed_annotations: list[JsonDict] = []
    for entry in annotations:
        if not isinstance(entry, dict):
            raise DetectorEvaluateSmokeError(
                "Every split annotation entry must be a JSON object."
            )
        parsed_annotations.append(cast(JsonDict, entry))

    predictions: list[JsonDict] = []
    for annotation in sorted(parsed_annotations, key=_annotation_sort_key):
        predictions.append(
            {
                "bbox": _require_bbox(annotation, "bbox"),
                "category_id": DETECTION_ONLY_CATEGORY_ID,
                "image_id": _require_int(annotation, "image_id"),
                "score": PREDICTION_SCORE,
            }
        )
    return predictions


def _split_image_sort_key(image: JsonDict) -> tuple[int, str]:
    return (_require_int(image, "id"), _require_string(image, "file_name"))


def _category_sort_key(category: JsonDict) -> tuple[int, str]:
    return (_require_int(category, "id"), _require_string(category, "name"))


def _load_split_images(
    split_payload: JsonDict, *, split_path: Path
) -> list[SplitImage]:
    images = _require_list(split_payload, "images")
    processed_root = split_path.parents[1]
    images_root = processed_root / "images"
    records: list[SplitImage] = []
    for entry in images:
        if not isinstance(entry, dict):
            raise DetectorEvaluateSmokeError(
                "Every split image entry must be a JSON object."
            )
        image = cast(JsonDict, entry)
        file_name = _validate_relative_posix_path(
            "file_name", _require_string(image, "file_name")
        )
        image_path = images_root / Path(PurePosixPath(file_name))
        if not image_path.exists() or not image_path.is_file():
            raise DetectorEvaluateSmokeError(
                f"Split image file does not exist: {image_path}"
            )
        records.append(
            SplitImage(
                file_name=file_name,
                height=_require_int(image, "height"),
                image_id=_require_int(image, "id"),
                image_path=image_path.resolve(),
                width=_require_int(image, "width"),
            )
        )
    return sorted(
        records,
        key=lambda image: (image.image_id, image.file_name),
    )


def _build_yolo_to_competition_category_map(split_payload: JsonDict) -> dict[int, int]:
    category_entries = _require_list(split_payload, "categories")
    parsed_categories: list[JsonDict] = []
    for entry in category_entries:
        if not isinstance(entry, dict):
            raise DetectorEvaluateSmokeError(
                "Every split category entry must be a JSON object."
            )
        parsed_categories.append(cast(JsonDict, entry))
    sorted_categories = sorted(parsed_categories, key=_category_sort_key)
    return {
        index: _require_int(category, "id")
        for index, category in enumerate(sorted_categories)
    }


def _json_scalar(value: object, *, label: str) -> object:
    if hasattr(value, "tolist"):
        value = cast(Any, value).tolist()
    elif hasattr(value, "item"):
        value = cast(Any, value).item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _sequence(value: object, *, label: str) -> list[object]:
    normalized = _json_scalar(value, label=label)
    if not isinstance(normalized, list):
        raise DetectorEvaluateSmokeError(f"Expected {label} to be a list-like value.")
    return cast(list[object], normalized)


def _numeric_value(value: object, *, label: str) -> float:
    normalized = _json_scalar(value, label=label)
    if not isinstance(normalized, (int, float)) or isinstance(normalized, bool):
        raise DetectorEvaluateSmokeError(f"Expected {label} to be numeric.")
    return float(normalized)


def _class_index(value: object, *, label: str) -> int:
    numeric_value = _numeric_value(value, label=label)
    integer_value = int(numeric_value)
    if float(integer_value) != numeric_value:
        raise DetectorEvaluateSmokeError(f"Expected {label} to be an integer class id.")
    return integer_value


def _xyxy_to_xywh(*, xyxy: list[object], image: SplitImage, index: int) -> list[float]:
    if len(xyxy) != 4:
        raise DetectorEvaluateSmokeError(
            f"Expected inference box {index} to contain four xyxy values."
        )
    x1 = _numeric_value(xyxy[0], label=f"boxes.xyxy[{index}][0]")
    y1 = _numeric_value(xyxy[1], label=f"boxes.xyxy[{index}][1]")
    x2 = _numeric_value(xyxy[2], label=f"boxes.xyxy[{index}][2]")
    y2 = _numeric_value(xyxy[3], label=f"boxes.xyxy[{index}][3]")
    x1 = max(0.0, min(x1, float(image.width)))
    y1 = max(0.0, min(y1, float(image.height)))
    x2 = max(0.0, min(x2, float(image.width)))
    y2 = max(0.0, min(y2, float(image.height)))
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise DetectorEvaluateSmokeError(
            f"Inference box {index} for image_id={image.image_id} is not valid."
        )
    return [x1, y1, width, height]


def _run_ultralytics_inference(
    weights_path: Path, images: list[SplitImage]
) -> list[object]:
    if not images:
        return []
    model = _load_ultralytics_yolo()(weights_path.as_posix())
    results = model([image.image_path.as_posix() for image in images], verbose=False)
    if isinstance(results, list):
        return results
    return list(cast(Any, results))


def build_real_predictions(
    *, weights_path: Path, split_path: Path, split_payload: JsonDict
) -> list[JsonDict]:
    images = _load_split_images(split_payload, split_path=split_path)
    yolo_to_competition_category = _build_yolo_to_competition_category_map(
        split_payload
    )
    results = _run_ultralytics_inference(weights_path, images)
    if len(results) != len(images):
        raise DetectorEvaluateSmokeError(
            "Ultralytics inference did not return one result per split image."
        )

    predictions: list[JsonDict] = []
    for image_index, image in enumerate(images):
        result = results[image_index]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy_rows = _sequence(getattr(boxes, "xyxy", []), label="boxes.xyxy")
        conf_values = _sequence(getattr(boxes, "conf", []), label="boxes.conf")
        cls_values = _sequence(getattr(boxes, "cls", []), label="boxes.cls")
        if not (
            len(xyxy_rows) == len(conf_values) and len(conf_values) == len(cls_values)
        ):
            raise DetectorEvaluateSmokeError(
                "Ultralytics inference boxes/conf/cls arrays must have "
                "matching lengths."
            )
        for index, xyxy_row in enumerate(xyxy_rows):
            cls_index = _class_index(cls_values[index], label=f"boxes.cls[{index}]")
            category_id = yolo_to_competition_category.get(cls_index)
            if category_id is None:
                raise DetectorEvaluateSmokeError(
                    f"Inference predicted unknown YOLO class index: {cls_index}"
                )
            predictions.append(
                {
                    "bbox": _xyxy_to_xywh(
                        xyxy=_sequence(
                            xyxy_row,
                            label=f"boxes.xyxy[{index}]",
                        ),
                        image=image,
                        index=index,
                    ),
                    "category_id": category_id,
                    "image_id": image.image_id,
                    "score": _numeric_value(
                        conf_values[index],
                        label=f"boxes.conf[{index}]",
                    ),
                }
            )
    return predictions


def _build_predictions_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.predictions.json")


def run_smoke_evaluation(
    *, weights_path: str | Path, split_path: str | Path, out_path: str | Path
) -> JsonDict:
    weights = Path(weights_path)
    split = Path(split_path)
    output = Path(out_path)

    placeholder_payload = _load_placeholder_payload(weights)
    summary = _validate_train_summary(weights, placeholder_payload)
    split_payload = _load_split_annotations(split, placeholder_payload)
    predictions = build_smoke_predictions(split_payload)
    metrics = score_predictions(split, predictions)

    predictions_path = _build_predictions_path(output)
    write_json(predictions_path, predictions)

    evaluation_payload: JsonDict = {
        "artifact_provenance": {
            "train_summary": {
                "path": weights.with_name("train_summary.json").as_posix(),
                **file_snapshot(weights.with_name("train_summary.json")),
            },
            "weights": {
                "contract_version": PLACEHOLDER_RUNTIME_VERSION,
                "path": weights.as_posix(),
                **file_snapshot(weights),
            },
        },
        "evaluation": {
            "mode": PLACEHOLDER_MODE,
            "prediction_count": len(predictions),
            "scoring_rule": "0.7*detection_map + 0.3*classification_map",
            "split_path": split.as_posix(),
        },
        "export": {
            "format": "coco_predictions",
            "placeholder": True,
            "predictions_path": predictions_path.as_posix(),
            **file_snapshot(predictions_path),
        },
        "metrics": metrics,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "smoke_contract": {
            "annotation_count": _require_int(placeholder_payload, "annotation_count"),
            "annotations_path": _require_string(
                placeholder_payload, "annotations_path"
            ),
            "holdout_images": _require_int(placeholder_payload, "holdout_images"),
            "image_count": _require_int(placeholder_payload, "image_count"),
            "model_name": _require_string(placeholder_payload, "model_name"),
            "run_name": _require_string(placeholder_payload, "run_name"),
            "runtime": _require_mapping(placeholder_payload, "runtime"),
            "seed": _require_int(placeholder_payload, "seed"),
            "train_summary_path": weights.with_name("train_summary.json").as_posix(),
            "training_output_dir": _require_string(summary, "output_dir"),
        },
        "split_summary": {
            "annotation_count": len(_require_list(split_payload, "annotations")),
            "category_count": len(_require_list(split_payload, "categories")),
            "image_count": len(_require_list(split_payload, "images")),
        },
    }
    write_json(output, evaluation_payload)
    return evaluation_payload


def run_real_evaluation(
    *, weights_path: str | Path, split_path: str | Path, out_path: str | Path
) -> JsonDict:
    weights = Path(weights_path)
    split = Path(split_path)
    output = Path(out_path)

    summary = _validate_real_train_summary(weights)
    split_payload = _load_real_split_annotations(split)
    predictions = build_real_predictions(
        weights_path=weights,
        split_path=split,
        split_payload=split_payload,
    )
    metrics = score_predictions(split, predictions)

    predictions_path = _build_predictions_path(output)
    write_json(predictions_path, predictions)

    evaluation_payload: JsonDict = {
        "artifact_provenance": {
            "train_summary": {
                "path": weights.with_name("train_summary.json").as_posix(),
                **file_snapshot(weights.with_name("train_summary.json")),
            },
            "weights": {
                "path": weights.as_posix(),
                **file_snapshot(weights),
            },
        },
        "evaluation": {
            "mode": HOLDOUT_REAL_EVALUATION_MODE,
            "prediction_count": len(predictions),
            "scoring_rule": "0.7*detection_map + 0.3*classification_map",
            "split_path": split.as_posix(),
        },
        "export": {
            "format": "coco_predictions",
            "placeholder": False,
            "predictions_path": predictions_path.as_posix(),
            **file_snapshot(predictions_path),
        },
        "metrics": metrics,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "split_summary": {
            "annotation_count": len(_require_list(split_payload, "annotations")),
            "category_count": len(_require_list(split_payload, "categories")),
            "image_count": len(_require_list(split_payload, "images")),
        },
        "trained_model": {
            "mode": _require_string(summary, "mode"),
            "model_name": _require_string(summary, "model_name"),
            "output_dir": _require_string(summary, "output_dir"),
            "run_name": _require_string(
                _require_mapping(summary, "search"), "run_name"
            ),
            "runtime": _require_mapping(summary, "runtime"),
            "training_placeholder": _require_bool(
                _require_mapping(summary, "training"),
                "placeholder",
            ),
        },
    }
    write_json(output, evaluation_payload)
    return evaluation_payload


def run_evaluation(
    *, mode: str, weights_path: str | Path, split_path: str | Path, out_path: str | Path
) -> JsonDict:
    if mode == PLACEHOLDER_MODE:
        return run_smoke_evaluation(
            weights_path=weights_path,
            split_path=split_path,
            out_path=out_path,
        )
    if mode == REAL_MODE:
        return run_real_evaluation(
            weights_path=weights_path,
            split_path=split_path,
            out_path=out_path,
        )
    raise DetectorEvaluateSmokeError(f"Unsupported detector evaluation mode: {mode}.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(EvaluateArgs, parser.parse_args(argv))

    try:
        payload = run_evaluation(
            mode=args.mode,
            weights_path=args.weights,
            split_path=args.split,
            out_path=args.out,
        )
    except (DetectorEvaluateSmokeError, ScoreValidationError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
