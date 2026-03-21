from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from src.ng_data.data.manifest import load_manifest
from src.ng_data.detector.config import (
    DetectorConfigValidationError,
    load_and_validate_detector_config,
)
from src.ng_data.eval.splits import SplitConfigValidationError, validate_split_manifest

JsonDict = dict[str, Any]
EXPECTED_RUN_NAME = "yolov8m-search-baseline"
DEFAULT_OUTPUT_DIR = f"artifacts/runs/detector/{EXPECTED_RUN_NAME}/dataset"


class DetectorDatasetPreparationError(ValueError):
    pass


@dataclass(frozen=True)
class CategoryDefinition:
    competition_id: int
    name: str
    yolo_index: int


@dataclass(frozen=True)
class ImageDefinition:
    file_name: str
    height: int
    image_id: int
    source_path: Path
    width: int


@dataclass(frozen=True)
class AnnotationDefinition:
    annotation_id: int
    bbox: tuple[float, float, float, float]
    yolo_index: int


@dataclass(frozen=True)
class PreparedDatasetInputs:
    annotations_path: Path
    categories_path: Path
    images_root: Path
    manifest_path: Path
    manifest_relative_path: str
    manifest_counts: JsonDict
    processed_root: Path
    source_annotations_relative_path: str


@dataclass(frozen=True)
class SplitSelection:
    test_image_ids: tuple[int, ...]
    train_image_ids: tuple[int, ...]
    val_image_ids: tuple[int, ...]


class PrepareDatasetArgs(argparse.Namespace):
    config: str
    manifest: str
    out: str
    splits: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert canonical processed COCO data into a deterministic YOLO "
            "training workspace."
        )
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
        "--splits",
        default="data/processed/manifests/splits.json",
        help="Path to the processed split manifest JSON file.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the YOLO dataset workspace should be written.",
    )
    return parser


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise DetectorDatasetPreparationError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_list(data: JsonDict, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise DetectorDatasetPreparationError(f"Expected '{key}' to be a list.")
    return cast(list[object], value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise DetectorDatasetPreparationError(
            f"Expected '{key}' to be a non-empty string."
        )
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DetectorDatasetPreparationError(f"Expected '{key}' to be an integer.")
    return value


def _load_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DetectorDatasetPreparationError(
            f"Expected file does not exist: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise DetectorDatasetPreparationError(f"Invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise DetectorDatasetPreparationError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _load_json_list(path: Path) -> list[object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DetectorDatasetPreparationError(
            f"Expected file does not exist: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise DetectorDatasetPreparationError(f"Invalid JSON file: {path}") from error
    if not isinstance(payload, list):
        raise DetectorDatasetPreparationError(f"Expected JSON list in {path}")
    return cast(list[object], payload)


def _require_number(data: JsonDict, key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DetectorDatasetPreparationError(f"Expected '{key}' to be numeric.")
    return float(value)


def _validate_relative_posix_path(label: str, value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute():
        raise DetectorDatasetPreparationError(
            f"Expected '{label}' to be a relative path, got '{value}'."
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise DetectorDatasetPreparationError(
            f"Expected '{label}' to avoid empty, '.' or '..' path segments."
        )
    return value


def _resolve_processed_output(
    processed_outputs: JsonDict, processed_root: Path, name: str
) -> tuple[str, Path]:
    entry = _require_mapping(processed_outputs, name)
    relative_path = _validate_relative_posix_path(
        "path", _require_string(entry, "path")
    )
    path = processed_root / Path(PurePosixPath(relative_path))
    if not path.exists():
        raise DetectorDatasetPreparationError(
            f"Processed output '{name}' does not exist: {path}"
        )
    return relative_path, path


def _load_prepared_inputs(manifest_path: Path) -> PreparedDatasetInputs:
    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError as error:
        raise DetectorDatasetPreparationError(
            f"Expected file does not exist: {manifest_path}"
        ) from error
    if _require_int(manifest, "schema_version") != 1:
        raise DetectorDatasetPreparationError(
            "Unsupported processed manifest schema version."
        )

    processed_root = manifest_path.parents[1]
    manifest_relative_path = manifest_path.relative_to(processed_root).as_posix()
    counts = _require_mapping(manifest, "counts")
    for key in (
        "annotation_count",
        "category_count",
        "image_count",
        "reference_product_count",
    ):
        _require_int(counts, key)

    processed_outputs = _require_mapping(manifest, "processed_outputs")
    annotations_relative_path, annotations_path = _resolve_processed_output(
        processed_outputs, processed_root, "annotations"
    )
    _, images_root = _resolve_processed_output(
        processed_outputs, processed_root, "images"
    )
    _, categories_path = _resolve_processed_output(
        processed_outputs, processed_root, "categories"
    )
    if not images_root.is_dir():
        raise DetectorDatasetPreparationError(
            f"Processed output 'images' must be a directory: {images_root}"
        )
    if not annotations_path.is_file():
        raise DetectorDatasetPreparationError(
            f"Processed output 'annotations' must be a file: {annotations_path}"
        )
    if not categories_path.is_file():
        raise DetectorDatasetPreparationError(
            f"Processed output 'categories' must be a file: {categories_path}"
        )

    return PreparedDatasetInputs(
        annotations_path=annotations_path,
        categories_path=categories_path,
        images_root=images_root,
        manifest_path=manifest_path,
        manifest_relative_path=manifest_relative_path,
        manifest_counts=counts,
        processed_root=processed_root,
        source_annotations_relative_path=annotations_relative_path,
    )


def _validate_run_name(config_path: Path) -> None:
    config = load_and_validate_detector_config(config_path)
    if config.search.run_name != EXPECTED_RUN_NAME:
        raise DetectorDatasetPreparationError(
            "Detector dataset preparation requires "
            f"search.run_name='{EXPECTED_RUN_NAME}', got '{config.search.run_name}'."
        )


def _load_split_selection(
    *,
    manifest_relative_path: str,
    source_annotations_relative_path: str,
    splits_path: Path,
) -> SplitSelection:
    payload = _load_json_object(splits_path)
    if _require_int(payload, "schema_version") != 1:
        raise DetectorDatasetPreparationError(
            "Unsupported split manifest schema version."
        )

    source_manifest = _require_string(payload, "source_manifest")
    if source_manifest != manifest_relative_path:
        raise DetectorDatasetPreparationError(
            "Split manifest source_manifest does not match the processed dataset "
            f"manifest: expected '{manifest_relative_path}', got '{source_manifest}'."
        )

    source_annotations = _require_string(payload, "source_annotations")
    if source_annotations != source_annotations_relative_path:
        raise DetectorDatasetPreparationError(
            "Split manifest source_annotations does not match the processed "
            f"annotations path: expected '{source_annotations_relative_path}', got "
            f"'{source_annotations}'."
        )

    try:
        validate_split_manifest(payload)
    except SplitConfigValidationError as error:
        raise DetectorDatasetPreparationError(str(error)) from error

    folds = _require_list(payload, "folds")
    if not folds:
        raise DetectorDatasetPreparationError("Expected at least one CV fold.")
    first_fold = folds[0]
    if not isinstance(first_fold, dict):
        raise DetectorDatasetPreparationError("Every fold entry must be an object.")
    fold = cast(JsonDict, first_fold)
    holdout = _require_mapping(payload, "holdout")

    train_image_ids = tuple(
        _require_int({"image_id": value}, "image_id")
        for value in _require_list(fold, "train_image_ids")
    )
    val_image_ids = tuple(
        _require_int({"image_id": value}, "image_id")
        for value in _require_list(fold, "val_image_ids")
    )
    test_image_ids = tuple(
        _require_int({"image_id": value}, "image_id")
        for value in _require_list(holdout, "image_ids")
    )
    return SplitSelection(
        test_image_ids=test_image_ids,
        train_image_ids=train_image_ids,
        val_image_ids=val_image_ids,
    )


def _load_categories(
    categories_path: Path,
) -> tuple[list[CategoryDefinition], dict[int, int]]:
    payload = _load_json_list(categories_path)
    parsed_categories: list[tuple[int, str]] = []
    seen_ids: set[int] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise DetectorDatasetPreparationError(
                "Every category entry must be a JSON object."
            )
        category = cast(JsonDict, item)
        category_id = _require_int(category, "id")
        if category_id in seen_ids:
            raise DetectorDatasetPreparationError(
                f"Duplicate category id: {category_id}"
            )
        seen_ids.add(category_id)
        parsed_categories.append((category_id, _require_string(category, "name")))

    sorted_categories = sorted(parsed_categories, key=lambda value: value[0])
    definitions = [
        CategoryDefinition(
            competition_id=category_id,
            name=name,
            yolo_index=index,
        )
        for index, (category_id, name) in enumerate(sorted_categories)
    ]
    return definitions, {
        category.competition_id: category.yolo_index for category in definitions
    }


def _load_annotations(
    *,
    annotations_path: Path,
    category_index_by_id: dict[int, int],
    expected_categories: list[CategoryDefinition],
    images_root: Path,
) -> tuple[dict[int, ImageDefinition], dict[int, list[AnnotationDefinition]], JsonDict]:
    payload = _load_json_object(annotations_path)
    images_payload = _require_list(payload, "images")
    annotations_payload = _require_list(payload, "annotations")
    categories_payload = _require_list(payload, "categories")

    embedded_categories: list[tuple[int, str]] = []
    seen_category_ids: set[int] = set()
    for item in categories_payload:
        if not isinstance(item, dict):
            raise DetectorDatasetPreparationError(
                "Every embedded category entry must be a JSON object."
            )
        category = cast(JsonDict, item)
        category_id = _require_int(category, "id")
        if category_id in seen_category_ids:
            raise DetectorDatasetPreparationError(
                f"Duplicate embedded category id: {category_id}"
            )
        seen_category_ids.add(category_id)
        embedded_categories.append((category_id, _require_string(category, "name")))

    expected_category_pairs = [
        (category.competition_id, category.name) for category in expected_categories
    ]
    if sorted(embedded_categories) != expected_category_pairs:
        raise DetectorDatasetPreparationError(
            "Processed categories.json does not match the categories embedded in "
            f"{annotations_path}."
        )

    images_by_id: dict[int, ImageDefinition] = {}
    for item in images_payload:
        if not isinstance(item, dict):
            raise DetectorDatasetPreparationError(
                "Every image entry must be an object."
            )
        image = cast(JsonDict, item)
        image_id = _require_int(image, "id")
        if image_id in images_by_id:
            raise DetectorDatasetPreparationError(f"Duplicate image id: {image_id}")
        width = _require_int(image, "width")
        height = _require_int(image, "height")
        if width <= 0 or height <= 0:
            raise DetectorDatasetPreparationError(
                f"Image dimensions must be positive for id {image_id}"
            )
        file_name = _validate_relative_posix_path(
            "file_name", _require_string(image, "file_name")
        )
        source_path = images_root / Path(PurePosixPath(file_name))
        if not source_path.exists() or not source_path.is_file():
            raise DetectorDatasetPreparationError(
                f"Processed image file does not exist: {source_path}"
            )
        images_by_id[image_id] = ImageDefinition(
            file_name=file_name,
            height=height,
            image_id=image_id,
            source_path=source_path.resolve(),
            width=width,
        )

    annotations_by_image: dict[int, list[AnnotationDefinition]] = {
        image_id: [] for image_id in images_by_id
    }
    seen_annotation_ids: set[int] = set()
    for item in annotations_payload:
        if not isinstance(item, dict):
            raise DetectorDatasetPreparationError(
                "Every annotation entry must be a JSON object."
            )
        annotation = cast(JsonDict, item)
        annotation_id = _require_int(annotation, "id")
        if annotation_id in seen_annotation_ids:
            raise DetectorDatasetPreparationError(
                f"Duplicate annotation id: {annotation_id}"
            )
        seen_annotation_ids.add(annotation_id)
        image_id = _require_int(annotation, "image_id")
        if image_id not in images_by_id:
            raise DetectorDatasetPreparationError(
                f"Annotation {annotation_id} references unknown image_id {image_id}"
            )
        category_id = _require_int(annotation, "category_id")
        if category_id not in category_index_by_id:
            raise DetectorDatasetPreparationError(
                "Annotation "
                f"{annotation_id} references unknown category_id {category_id}"
            )
        bbox_payload = annotation.get("bbox")
        if not isinstance(bbox_payload, list) or len(bbox_payload) != 4:
            raise DetectorDatasetPreparationError(
                f"Annotation {annotation_id} must contain a four-value bbox list."
            )
        bbox = (
            float(_require_number({"value": bbox_payload[0]}, "value")),
            float(_require_number({"value": bbox_payload[1]}, "value")),
            float(_require_number({"value": bbox_payload[2]}, "value")),
            float(_require_number({"value": bbox_payload[3]}, "value")),
        )
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
            raise DetectorDatasetPreparationError(
                f"Annotation {annotation_id} must contain a positive bbox."
            )
        annotations_by_image[image_id].append(
            AnnotationDefinition(
                annotation_id=annotation_id,
                bbox=bbox,
                yolo_index=category_index_by_id[category_id],
            )
        )

    for image_annotations in annotations_by_image.values():
        image_annotations.sort(key=lambda value: value.annotation_id)
    return images_by_id, annotations_by_image, payload


def _validate_manifest_counts(
    *,
    coco_payload: JsonDict,
    manifest_counts: JsonDict,
    expected_category_count: int,
) -> None:
    actual_annotation_count = len(_require_list(coco_payload, "annotations"))
    actual_image_count = len(_require_list(coco_payload, "images"))
    actual_category_count = expected_category_count
    if _require_int(manifest_counts, "annotation_count") != actual_annotation_count:
        raise DetectorDatasetPreparationError(
            "Processed manifest annotation_count does not match annotations payload: "
            f"expected {_require_int(manifest_counts, 'annotation_count')}, got "
            f"{actual_annotation_count}."
        )
    if _require_int(manifest_counts, "image_count") != actual_image_count:
        raise DetectorDatasetPreparationError(
            "Processed manifest image_count does not match annotations payload: "
            f"expected {_require_int(manifest_counts, 'image_count')}, got "
            f"{actual_image_count}."
        )
    if _require_int(manifest_counts, "category_count") != actual_category_count:
        raise DetectorDatasetPreparationError(
            "Processed manifest category_count does not match categories payload: "
            f"expected {_require_int(manifest_counts, 'category_count')}, got "
            f"{actual_category_count}."
        )


def _validate_selected_image_ids(
    *,
    images_by_id: dict[int, ImageDefinition],
    selection: SplitSelection,
) -> None:
    all_selected = (
        list(selection.train_image_ids)
        + list(selection.val_image_ids)
        + list(selection.test_image_ids)
    )
    unique_selected = set(all_selected)
    if len(unique_selected) != len(all_selected):
        raise DetectorDatasetPreparationError(
            "Selected train/val/test image ids must not overlap."
        )

    known_image_ids = set(images_by_id)
    unknown_image_ids = sorted(unique_selected - known_image_ids)
    if unknown_image_ids:
        missing_display = ", ".join(str(image_id) for image_id in unknown_image_ids)
        raise DetectorDatasetPreparationError(
            f"Split manifest references unknown image ids: {missing_display}"
        )

    if unique_selected != known_image_ids:
        missing_from_selection = sorted(known_image_ids - unique_selected)
        missing_display = ", ".join(
            str(image_id) for image_id in missing_from_selection
        )
        raise DetectorDatasetPreparationError(
            "The v1 detector split selection must cover every processed image id. "
            f"Missing: {missing_display}"
        )


def _format_float(value: float) -> str:
    formatted = f"{value:.10f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _build_label_text(
    *, image: ImageDefinition, annotations: list[AnnotationDefinition]
) -> str:
    lines: list[str] = []
    for annotation in annotations:
        x, y, width, height = annotation.bbox
        x_center = (x + (width / 2.0)) / image.width
        y_center = (y + (height / 2.0)) / image.height
        normalized_width = width / image.width
        normalized_height = height / image.height
        lines.append(
            " ".join(
                [
                    str(annotation.yolo_index),
                    _format_float(x_center),
                    _format_float(y_center),
                    _format_float(normalized_width),
                    _format_float(normalized_height),
                ]
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_image_link(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.symlink_to(source_path)


def _write_split_workspace(
    *,
    annotations_by_image: dict[int, list[AnnotationDefinition]],
    image_ids: tuple[int, ...],
    images_by_id: dict[int, ImageDefinition],
    output_root: Path,
    split_name: str,
) -> None:
    for image_id in image_ids:
        image = images_by_id[image_id]
        relative_path = Path(PurePosixPath(image.file_name))
        image_path = output_root / "images" / split_name / relative_path
        label_path = (
            output_root / "labels" / split_name / relative_path.with_suffix(".txt")
        )
        _write_image_link(image.source_path, image_path)
        _write_text(
            label_path,
            _build_label_text(
                image=image,
                annotations=annotations_by_image.get(image_id, []),
            ),
        )


def _build_dataset_yaml(*, names: list[str], output_root: Path) -> str:
    output_root_text = json.dumps(output_root.as_posix(), ensure_ascii=False)
    lines = [
        f"path: {output_root_text}",
        f"train: {json.dumps((output_root / 'images/train').as_posix(), ensure_ascii=False)}",
        f"val: {json.dumps((output_root / 'images/val').as_posix(), ensure_ascii=False)}",
        f"test: {json.dumps((output_root / 'images/test').as_posix(), ensure_ascii=False)}",
        "names:",
    ]
    for name in names:
        lines.append(f"  - {json.dumps(name, ensure_ascii=False)}")
    lines.append(f"nc: {len(names)}")
    return "\n".join(lines) + "\n"


def prepare_detector_dataset(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    out_path: str | Path,
    splits_path: str | Path,
) -> JsonDict:
    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    splits_path = Path(splits_path)
    output_root = Path(out_path).resolve()

    _validate_run_name(config_path)
    prepared_inputs = _load_prepared_inputs(manifest_path)
    selection = _load_split_selection(
        manifest_relative_path=prepared_inputs.manifest_relative_path,
        source_annotations_relative_path=prepared_inputs.source_annotations_relative_path,
        splits_path=splits_path,
    )
    categories, category_index_by_id = _load_categories(prepared_inputs.categories_path)
    images_by_id, annotations_by_image, coco_payload = _load_annotations(
        annotations_path=prepared_inputs.annotations_path,
        category_index_by_id=category_index_by_id,
        expected_categories=categories,
        images_root=prepared_inputs.images_root,
    )
    _validate_manifest_counts(
        coco_payload=coco_payload,
        manifest_counts=prepared_inputs.manifest_counts,
        expected_category_count=len(categories),
    )
    _validate_selected_image_ids(images_by_id=images_by_id, selection=selection)

    staging_root = output_root.parent / f".{output_root.name}.tmp"
    shutil.rmtree(staging_root, ignore_errors=True)
    try:
        staging_root.mkdir(parents=True, exist_ok=True)
        for split_name, image_ids in (
            ("train", selection.train_image_ids),
            ("val", selection.val_image_ids),
            ("test", selection.test_image_ids),
        ):
            _write_split_workspace(
                annotations_by_image=annotations_by_image,
                image_ids=image_ids,
                images_by_id=images_by_id,
                output_root=staging_root,
                split_name=split_name,
            )
        _write_text(
            staging_root / "dataset.yaml",
            _build_dataset_yaml(
                names=[category.name for category in categories],
                output_root=output_root,
            ),
        )
        shutil.rmtree(output_root, ignore_errors=True)
        staging_root.rename(output_root)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return {
        "config_path": config_path.as_posix(),
        "dataset_yaml": (output_root / "dataset.yaml").as_posix(),
        "names": [category.name for category in categories],
        "nc": len(categories),
        "output_dir": output_root.as_posix(),
        "processed_manifest_path": manifest_path.as_posix(),
        "source_annotations": prepared_inputs.source_annotations_relative_path,
        "source_manifest": prepared_inputs.manifest_relative_path,
        "split_counts": {
            "test": len(selection.test_image_ids),
            "train": len(selection.train_image_ids),
            "val": len(selection.val_image_ids),
        },
        "splits_path": splits_path.as_posix(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(PrepareDatasetArgs, parser.parse_args(argv))
    try:
        payload = prepare_detector_dataset(
            config_path=args.config,
            manifest_path=args.manifest,
            out_path=args.out,
            splits_path=args.splits,
        )
    except (
        DetectorConfigValidationError,
        DetectorDatasetPreparationError,
        ValueError,
    ) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
