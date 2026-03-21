from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.ng_data.data.manifest import load_manifest, write_json

JsonDict = dict[str, Any]


class SplitConfigValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SplitConfig:
    schema_version: int
    seed: int
    holdout_fraction: float
    cv_folds: int
    output_path: str


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise SplitConfigValidationError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_list(data: JsonDict, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise SplitConfigValidationError(f"Expected '{key}' to be a list.")
    return cast(list[object], value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise SplitConfigValidationError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SplitConfigValidationError(f"Expected '{key}' to be an integer.")
    return value


def _require_number(data: JsonDict, key: str) -> int | float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SplitConfigValidationError(f"Expected '{key}' to be numeric.")
    return cast(int | float, value)


def _validate_relative_path(label: str, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        raise SplitConfigValidationError(
            f"Expected '{label}' to be a relative path, got '{value}'."
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SplitConfigValidationError(
            f"Expected '{label}' to avoid empty, '.' or '..' path segments."
        )
    return value


def parse_split_config(data: JsonDict) -> SplitConfig:
    return SplitConfig(
        schema_version=_require_int(data, "schema_version"),
        seed=_require_int(data, "seed"),
        holdout_fraction=float(_require_number(data, "holdout_fraction")),
        cv_folds=_require_int(data, "cv_folds"),
        output_path=_validate_relative_path(
            "output_path", _require_string(data, "output_path")
        ),
    )


def load_split_config(config_path: str | Path) -> SplitConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        data = cast(JsonDict, json.load(config_file))
    return parse_split_config(data)


def validate_split_config(config: SplitConfig) -> None:
    if config.schema_version != 1:
        raise SplitConfigValidationError(
            f"Unsupported schema version: {config.schema_version}"
        )
    if not 0 < config.holdout_fraction < 1:
        raise SplitConfigValidationError("holdout_fraction must be between 0 and 1")
    if config.cv_folds < 2:
        raise SplitConfigValidationError("cv_folds must be at least 2")


def _load_coco_payload(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SplitConfigValidationError(f"Invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise SplitConfigValidationError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _load_annotation_index(annotations_path: Path) -> tuple[list[int], dict[int, int]]:
    payload = _load_coco_payload(annotations_path)
    images = _require_list(payload, "images")
    annotations = _require_list(payload, "annotations")

    image_ids: list[int] = []
    seen_image_ids: set[int] = set()
    annotation_counts_by_image: dict[int, int] = {}
    for item in images:
        if not isinstance(item, dict):
            raise SplitConfigValidationError("Every image entry must be an object.")
        image = cast(JsonDict, item)
        image_id = _require_int(image, "id")
        if image_id in seen_image_ids:
            raise SplitConfigValidationError(f"Duplicate image id: {image_id}")
        seen_image_ids.add(image_id)
        image_ids.append(image_id)
        annotation_counts_by_image[image_id] = 0

    if not image_ids:
        raise SplitConfigValidationError("Expected at least one image in annotations.")

    for item in annotations:
        if not isinstance(item, dict):
            raise SplitConfigValidationError(
                "Every annotation entry must be an object."
            )
        annotation = cast(JsonDict, item)
        image_id = _require_int(annotation, "image_id")
        if image_id not in annotation_counts_by_image:
            raise SplitConfigValidationError(
                f"Annotation references unknown image_id {image_id}"
            )
        annotation_counts_by_image[image_id] += 1

    return sorted(image_ids), annotation_counts_by_image


def _resolve_annotations_path(dataset_manifest_path: Path) -> Path:
    manifest = load_manifest(dataset_manifest_path)
    processed_outputs = _require_mapping(manifest, "processed_outputs")
    annotations_entry = _require_mapping(processed_outputs, "annotations")
    relative_path = _require_string(annotations_entry, "path")
    processed_root = dataset_manifest_path.parents[1]
    return processed_root / relative_path


def _stable_image_order(image_ids: list[int], seed: int) -> list[int]:
    def sort_key(image_id: int) -> tuple[str, int]:
        digest = hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()
        return digest, image_id

    return sorted(image_ids, key=sort_key)


def _compute_holdout_count(total_images: int, holdout_fraction: float) -> int:
    if total_images == 0:
        return 0
    requested = math.ceil(total_images * holdout_fraction)
    if total_images == 1:
        return 1
    return min(max(1, requested), total_images - 1)


def _annotation_count(image_ids: list[int], counts_by_image: dict[int, int]) -> int:
    return sum(counts_by_image[image_id] for image_id in image_ids)


def validate_split_manifest(payload: JsonDict) -> None:
    holdout = _require_mapping(payload, "holdout")
    holdout_ids = [
        _require_int({"image_id": value}, "image_id")
        for value in _require_list(holdout, "image_ids")
    ]
    if len(set(holdout_ids)) != len(holdout_ids):
        raise SplitConfigValidationError("Holdout image_ids must be unique.")

    folds_payload = _require_list(payload, "folds")
    if not folds_payload:
        raise SplitConfigValidationError("Expected at least one CV fold.")

    expected_cv_pool_ids: set[int] | None = None
    if "cv_pool_image_ids" in payload:
        expected_cv_pool_ids = {
            _require_int({"image_id": value}, "image_id")
            for value in _require_list(payload, "cv_pool_image_ids")
        }
        if expected_cv_pool_ids & set(holdout_ids):
            raise SplitConfigValidationError(
                "cv_pool_image_ids must not overlap with holdout image_ids."
            )

    cv_validation_ids: set[int] = set()
    for index, item in enumerate(folds_payload):
        if not isinstance(item, dict):
            raise SplitConfigValidationError("Every fold entry must be an object.")
        fold = cast(JsonDict, item)
        train_ids = [
            _require_int({"image_id": value}, "image_id")
            for value in _require_list(fold, "train_image_ids")
        ]
        validation_ids = [
            _require_int({"image_id": value}, "image_id")
            for value in _require_list(fold, "val_image_ids")
        ]

        train_set = set(train_ids)
        validation_set = set(validation_ids)
        if len(train_set) != len(train_ids):
            raise SplitConfigValidationError(
                f"Fold {index} train_image_ids must be unique."
            )
        if len(validation_set) != len(validation_ids):
            raise SplitConfigValidationError(
                f"Fold {index} val_image_ids must be unique."
            )
        if train_set & validation_set:
            raise SplitConfigValidationError(
                f"Fold {index} leaks image_ids between train and validation."
            )
        if validation_set & set(holdout_ids):
            raise SplitConfigValidationError(
                f"Fold {index} validation set contains holdout image_ids."
            )
        if train_set & set(holdout_ids):
            raise SplitConfigValidationError(
                f"Fold {index} train set contains holdout image_ids."
            )
        overlap = cv_validation_ids & validation_set
        if overlap:
            overlap_display = ", ".join(str(image_id) for image_id in sorted(overlap))
            raise SplitConfigValidationError(
                f"Image leakage across validation folds: {overlap_display}"
            )
        cv_validation_ids.update(validation_set)

    if expected_cv_pool_ids is not None and cv_validation_ids != expected_cv_pool_ids:
        raise SplitConfigValidationError(
            "Validation fold image_ids must match cv_pool_image_ids exactly."
        )


def generate_split_manifest(
    config: SplitConfig, dataset_manifest_path: str | Path
) -> JsonDict:
    validate_split_config(config)
    manifest_path = Path(dataset_manifest_path)
    annotations_path = _resolve_annotations_path(manifest_path)
    image_ids, annotation_counts_by_image = _load_annotation_index(annotations_path)

    ordered_image_ids = _stable_image_order(image_ids, config.seed)
    holdout_count = _compute_holdout_count(
        len(ordered_image_ids), config.holdout_fraction
    )
    holdout_ids = sorted(ordered_image_ids[:holdout_count])
    cv_pool_ids = ordered_image_ids[holdout_count:]

    validation_assignments: list[list[int]] = [[] for _ in range(config.cv_folds)]
    for index, image_id in enumerate(cv_pool_ids):
        validation_assignments[index % config.cv_folds].append(image_id)

    sorted_cv_pool_ids = sorted(cv_pool_ids)
    folds: list[JsonDict] = []
    for fold_index, validation_ids in enumerate(validation_assignments):
        sorted_validation_ids = sorted(validation_ids)
        validation_id_set = set(sorted_validation_ids)
        train_ids = [
            image_id
            for image_id in sorted_cv_pool_ids
            if image_id not in validation_id_set
        ]
        folds.append(
            {
                "fold_index": fold_index,
                "train_annotation_count": _annotation_count(
                    train_ids, annotation_counts_by_image
                ),
                "train_image_ids": train_ids,
                "val_annotation_count": _annotation_count(
                    sorted_validation_ids, annotation_counts_by_image
                ),
                "val_image_ids": sorted_validation_ids,
            }
        )

    payload: JsonDict = {
        "counts": {
            "cv_folds": config.cv_folds,
            "cv_pool_images": len(cv_pool_ids),
            "holdout_images": len(holdout_ids),
            "total_images": len(image_ids),
        },
        "cv_pool_image_ids": sorted_cv_pool_ids,
        "folds": folds,
        "holdout": {
            "annotation_count": _annotation_count(
                holdout_ids, annotation_counts_by_image
            ),
            "image_ids": holdout_ids,
        },
        "holdout_fraction": config.holdout_fraction,
        "schema_version": 1,
        "seed": config.seed,
        "source_annotations": annotations_path.relative_to(
            manifest_path.parents[1]
        ).as_posix(),
        "source_manifest": manifest_path.relative_to(
            manifest_path.parents[1]
        ).as_posix(),
    }
    validate_split_manifest(payload)
    return payload


def write_split_manifest(
    config: SplitConfig, dataset_manifest_path: str | Path
) -> tuple[Path, JsonDict]:
    manifest_path = Path(dataset_manifest_path)
    processed_root = manifest_path.parents[1]
    payload = generate_split_manifest(config, manifest_path)
    output_path = processed_root / config.output_path
    write_json(output_path, payload)
    return output_path, payload
