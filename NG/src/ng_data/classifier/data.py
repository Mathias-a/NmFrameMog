from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.ng_data.data.manifest import load_manifest, write_json

JsonDict = dict[str, Any]

SUPPORTED_CLASSIFIER_BASELINE = "classifier_gt_crops_search"
SUPPORTED_BACKBONE_LIBRARY = "timm"
SUPPORTED_BACKBONE_LIBRARY_VERSION = "0.9.12"
SUPPORTED_CROP_SOURCE = "ground_truth_boxes"


class ClassifierDataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ClassifierRuntimeConfig:
    backbone_library: str
    backbone_library_version: str


@dataclass(frozen=True)
class ClassifierModelConfig:
    backbone: str
    image_size: int


@dataclass(frozen=True)
class ClassifierInputConfig:
    processed_manifest_path: str
    annotations_path: str
    categories_path: str
    product_index_path: str
    reference_metadata_path: str
    crop_source: str


@dataclass(frozen=True)
class ClassifierOutputConfig:
    class_map_path: str
    crop_manifest_path: str
    crops_dir: str


@dataclass(frozen=True)
class ClassifierDataConfig:
    schema_version: int
    baseline: str
    runtime: ClassifierRuntimeConfig
    model: ClassifierModelConfig
    data: ClassifierInputConfig
    outputs: ClassifierOutputConfig


@dataclass(frozen=True)
class _ClassMapEntry:
    class_id: int
    category_ids: tuple[int, ...]
    product_code: str
    product_name: str
    reference_image_paths: tuple[str, ...]


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ClassifierDataValidationError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_list(data: JsonDict, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ClassifierDataValidationError(f"Expected '{key}' to be a list.")
    return cast(list[object], value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise ClassifierDataValidationError(
            f"Expected '{key}' to be a non-empty string."
        )
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ClassifierDataValidationError(f"Expected '{key}' to be an integer.")
    return value


def _require_bool(data: JsonDict, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ClassifierDataValidationError(f"Expected '{key}' to be a boolean.")
    return value


def _validate_relative_path(label: str, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        raise ClassifierDataValidationError(
            f"Expected '{label}' to be a relative path, got '{value}'."
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ClassifierDataValidationError(
            f"Expected '{label}' to avoid empty, '.' or '..' path segments."
        )
    return value


def _require_string_list(data: JsonDict, key: str) -> list[str]:
    values = _require_list(data, key)
    strings: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or value == "":
            raise ClassifierDataValidationError(
                f"Expected '{key}[{index}]' to be a non-empty string."
            )
        strings.append(value)
    return strings


def _require_int_list(data: JsonDict, key: str) -> list[int]:
    values = _require_list(data, key)
    integers: list[int] = []
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ClassifierDataValidationError(
                f"Expected '{key}[{index}]' to be an integer."
            )
        integers.append(value)
    return integers


def _require_bbox(data: JsonDict, key: str) -> list[float]:
    values = _require_list(data, key)
    if len(values) != 4:
        raise ClassifierDataValidationError(
            f"Expected '{key}' to contain exactly four numeric values."
        )
    bbox: list[float] = []
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ClassifierDataValidationError(
                f"Expected '{key}[{index}]' to be numeric."
            )
        bbox.append(float(value))
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ClassifierDataValidationError(
            f"Expected '{key}' width and height to be positive."
        )
    return bbox


def _parse_runtime(data: JsonDict) -> ClassifierRuntimeConfig:
    return ClassifierRuntimeConfig(
        backbone_library=_require_string(data, "backbone_library"),
        backbone_library_version=_require_string(data, "backbone_library_version"),
    )


def _parse_model(data: JsonDict) -> ClassifierModelConfig:
    return ClassifierModelConfig(
        backbone=_require_string(data, "backbone"),
        image_size=_require_int(data, "image_size"),
    )


def _parse_data(data: JsonDict) -> ClassifierInputConfig:
    return ClassifierInputConfig(
        processed_manifest_path=_validate_relative_path(
            "processed_manifest_path", _require_string(data, "processed_manifest_path")
        ),
        annotations_path=_validate_relative_path(
            "annotations_path", _require_string(data, "annotations_path")
        ),
        categories_path=_validate_relative_path(
            "categories_path", _require_string(data, "categories_path")
        ),
        product_index_path=_validate_relative_path(
            "product_index_path", _require_string(data, "product_index_path")
        ),
        reference_metadata_path=_validate_relative_path(
            "reference_metadata_path",
            _require_string(data, "reference_metadata_path"),
        ),
        crop_source=_require_string(data, "crop_source"),
    )


def _parse_outputs(data: JsonDict) -> ClassifierOutputConfig:
    return ClassifierOutputConfig(
        class_map_path=_validate_relative_path(
            "class_map_path", _require_string(data, "class_map_path")
        ),
        crop_manifest_path=_validate_relative_path(
            "crop_manifest_path", _require_string(data, "crop_manifest_path")
        ),
        crops_dir=_validate_relative_path(
            "crops_dir", _require_string(data, "crops_dir")
        ),
    )


def parse_classifier_data_config(data: JsonDict) -> ClassifierDataConfig:
    return ClassifierDataConfig(
        schema_version=_require_int(data, "schema_version"),
        baseline=_require_string(data, "baseline"),
        runtime=_parse_runtime(_require_mapping(data, "runtime")),
        model=_parse_model(_require_mapping(data, "model")),
        data=_parse_data(_require_mapping(data, "data")),
        outputs=_parse_outputs(_require_mapping(data, "outputs")),
    )


def load_classifier_data_config(config_path: str | Path) -> ClassifierDataConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        data = cast(JsonDict, json.load(config_file))
    return parse_classifier_data_config(data)


def _load_json_object(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ClassifierDataValidationError(f"Invalid JSON file: {path}") from error
    if not isinstance(data, dict):
        raise ClassifierDataValidationError(f"Expected JSON object in {path}")
    return cast(JsonDict, data)


def _load_json_list(path: Path) -> list[JsonDict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ClassifierDataValidationError(f"Invalid JSON file: {path}") from error
    if not isinstance(data, list):
        raise ClassifierDataValidationError(f"Expected JSON list in {path}")
    records: list[JsonDict] = []
    for index, value in enumerate(data):
        if not isinstance(value, dict):
            raise ClassifierDataValidationError(
                f"Expected '{path}[{index}]' to be a JSON object."
            )
        records.append(cast(JsonDict, value))
    return records


def _resolve_processed_inputs(
    config: ClassifierDataConfig, processed_root: Path
) -> tuple[Path, Path, Path, Path, str]:
    manifest_path = processed_root / config.data.processed_manifest_path
    manifest = load_manifest(manifest_path)
    if _require_int(manifest, "schema_version") != 1:
        raise ClassifierDataValidationError(
            "Unsupported processed manifest schema version."
        )

    processed_outputs = _require_mapping(manifest, "processed_outputs")
    required_paths = {
        "annotations": config.data.annotations_path,
        "categories": config.data.categories_path,
        "product_index": config.data.product_index_path,
        "reference_metadata": config.data.reference_metadata_path,
    }
    for key, expected_path in sorted(required_paths.items()):
        output_entry = _require_mapping(processed_outputs, key)
        actual_path = _require_string(output_entry, "path")
        if actual_path != expected_path:
            raise ClassifierDataValidationError(
                f"Processed manifest entry '{key}' must point to '{expected_path}', "
                f"got '{actual_path}'."
            )

    images_entry = _require_mapping(processed_outputs, "images")
    images_dir = _require_string(images_entry, "path")

    annotations_path = processed_root / config.data.annotations_path
    categories_path = processed_root / config.data.categories_path
    product_index_path = processed_root / config.data.product_index_path
    reference_metadata_path = processed_root / config.data.reference_metadata_path
    for path in (
        manifest_path,
        annotations_path,
        categories_path,
        product_index_path,
        reference_metadata_path,
    ):
        if not path.exists() or not path.is_file():
            raise ClassifierDataValidationError(
                f"Expected processed metadata file: {path}"
            )

    return (
        annotations_path,
        categories_path,
        product_index_path,
        reference_metadata_path,
        images_dir,
    )


def _build_class_map_entries(
    *,
    categories: list[JsonDict],
    product_index: list[JsonDict],
    reference_metadata: JsonDict,
) -> tuple[list[_ClassMapEntry], dict[int, _ClassMapEntry]]:
    categories_by_id: dict[int, str] = {}
    for category in categories:
        category_id = _require_int(category, "id")
        if category_id in categories_by_id:
            raise ClassifierDataValidationError(
                f"Duplicate category id in categories.json: {category_id}"
            )
        categories_by_id[category_id] = _require_string(category, "name")

    metadata_products = _require_list(reference_metadata, "products")
    metadata_by_code: dict[str, JsonDict] = {}
    for index, product in enumerate(metadata_products):
        if not isinstance(product, dict):
            raise ClassifierDataValidationError(
                f"Expected 'products[{index}]' to be a JSON object."
            )
        product_record = cast(JsonDict, product)
        product_code = _require_string(product_record, "product_code")
        if product_code in metadata_by_code:
            raise ClassifierDataValidationError(
                f"Duplicate reference metadata product_code: {product_code}"
            )
        metadata_by_code[product_code] = product_record

    parsed_products: list[tuple[tuple[int, ...], str, str, tuple[str, ...]]] = []
    seen_category_ids: set[int] = set()
    for product in product_index:
        product_code = _require_string(product, "product_code")
        product_name = _require_string(product, "product_name")
        category_ids = tuple(sorted(set(_require_int_list(product, "category_ids"))))
        if not category_ids:
            raise ClassifierDataValidationError(
                "Expected product_index entry "
                f"'{product_code}' to declare category_ids."
            )

        for category_id in category_ids:
            if category_id not in categories_by_id:
                raise ClassifierDataValidationError(
                    "Product "
                    f"'{product_code}' references missing category_id {category_id}."
                )
            if categories_by_id[category_id] != product_name:
                raise ClassifierDataValidationError(
                    f"Category/product name mismatch for category_id {category_id}: "
                    f"expected '{categories_by_id[category_id]}', got '{product_name}'."
                )
            if category_id in seen_category_ids:
                raise ClassifierDataValidationError(
                    f"Duplicate classifier class mapping for category_id {category_id}."
                )
            seen_category_ids.add(category_id)

        metadata_product = metadata_by_code.get(product_code)
        if metadata_product is None:
            raise ClassifierDataValidationError(
                f"Missing reference metadata for product_code '{product_code}'."
            )
        if _require_string(metadata_product, "product_name") != product_name:
            raise ClassifierDataValidationError(
                f"Reference metadata name mismatch for product_code '{product_code}'."
            )

        image_names = sorted(set(_require_string_list(metadata_product, "image_names")))
        expected_reference_paths = tuple(
            f"reference_images/{product_code}/{image_name}"
            for image_name in image_names
        )
        actual_reference_paths = tuple(
            sorted(set(_require_string_list(product, "reference_image_paths")))
        )
        if actual_reference_paths != expected_reference_paths:
            raise ClassifierDataValidationError(
                f"Reference image mapping mismatch for product_code '{product_code}'."
            )

        parsed_products.append(
            (category_ids, product_code, product_name, actual_reference_paths)
        )

    class_entries: list[_ClassMapEntry] = []
    class_by_category_id: dict[int, _ClassMapEntry] = {}
    for class_id, (
        category_ids,
        product_code,
        product_name,
        reference_image_paths,
    ) in enumerate(
        sorted(parsed_products, key=lambda item: (item[0][0], item[1], item[2]))
    ):
        entry = _ClassMapEntry(
            class_id=class_id,
            category_ids=category_ids,
            product_code=product_code,
            product_name=product_name,
            reference_image_paths=reference_image_paths,
        )
        class_entries.append(entry)
        for category_id in category_ids:
            class_by_category_id[category_id] = entry

    return class_entries, class_by_category_id


def validate_classifier_data_config(
    config: ClassifierDataConfig, processed_root: str | Path
) -> None:
    if config.schema_version != 1:
        raise ClassifierDataValidationError(
            f"Unsupported schema version: {config.schema_version}"
        )
    if config.baseline != SUPPORTED_CLASSIFIER_BASELINE:
        raise ClassifierDataValidationError(
            "Unsupported classifier baseline: "
            f"{config.baseline}. Expected '{SUPPORTED_CLASSIFIER_BASELINE}'."
        )
    if config.runtime.backbone_library != SUPPORTED_BACKBONE_LIBRARY:
        raise ClassifierDataValidationError(
            "Unsupported classifier backbone library: "
            f"{config.runtime.backbone_library}. Expected "
            f"'{SUPPORTED_BACKBONE_LIBRARY}'."
        )
    if config.runtime.backbone_library_version != SUPPORTED_BACKBONE_LIBRARY_VERSION:
        raise ClassifierDataValidationError(
            "Unsupported classifier backbone runtime pin: "
            f"{config.runtime.backbone_library}=="
            f"{config.runtime.backbone_library_version}. Expected "
            f"{SUPPORTED_BACKBONE_LIBRARY}=="
            f"{SUPPORTED_BACKBONE_LIBRARY_VERSION}."
        )
    if config.model.image_size < 32:
        raise ClassifierDataValidationError("image_size must be at least 32")
    if config.data.crop_source != SUPPORTED_CROP_SOURCE:
        raise ClassifierDataValidationError(
            "Unsupported classifier crop source: "
            f"{config.data.crop_source}. Expected '{SUPPORTED_CROP_SOURCE}'."
        )

    processed_root = Path(processed_root)
    (
        annotations_path,
        categories_path,
        product_index_path,
        reference_metadata_path,
        _,
    ) = _resolve_processed_inputs(config, processed_root)
    categories = _load_json_list(categories_path)
    product_index = _load_json_list(product_index_path)
    reference_metadata = _load_json_object(reference_metadata_path)
    _build_class_map_entries(
        categories=categories,
        product_index=product_index,
        reference_metadata=reference_metadata,
    )

    annotations_payload = _load_json_object(annotations_path)
    _require_list(annotations_payload, "annotations")
    _require_list(annotations_payload, "images")


def load_and_validate_classifier_data_config(
    config_path: str | Path, processed_root: str | Path
) -> ClassifierDataConfig:
    config = load_classifier_data_config(config_path)
    validate_classifier_data_config(config, processed_root)
    return config


def build_classifier_crop_dataset(
    *,
    config_path: str | Path,
    processed_root: str | Path,
    output_root: str | Path,
) -> JsonDict:
    processed_root = Path(processed_root)
    output_root = Path(output_root)
    config = load_and_validate_classifier_data_config(config_path, processed_root)
    (
        annotations_path,
        categories_path,
        product_index_path,
        reference_metadata_path,
        images_dir,
    ) = _resolve_processed_inputs(config, processed_root)

    categories = _load_json_list(categories_path)
    product_index = _load_json_list(product_index_path)
    reference_metadata = _load_json_object(reference_metadata_path)
    class_entries, class_by_category_id = _build_class_map_entries(
        categories=categories,
        product_index=product_index,
        reference_metadata=reference_metadata,
    )

    annotations_payload = _load_json_object(annotations_path)
    image_records = _require_list(annotations_payload, "images")
    images_by_id: dict[int, str] = {}
    for index, image in enumerate(image_records):
        if not isinstance(image, dict):
            raise ClassifierDataValidationError(
                f"Expected 'images[{index}]' to be a JSON object."
            )
        image_record = cast(JsonDict, image)
        image_id = _require_int(image_record, "id")
        if image_id in images_by_id:
            raise ClassifierDataValidationError(f"Duplicate image id: {image_id}")
        images_by_id[image_id] = _require_string(image_record, "file_name")

    crop_records: list[JsonDict] = []
    seen_image_ids: set[int] = set()
    annotations = _require_list(annotations_payload, "annotations")
    for index, annotation in enumerate(
        sorted(annotations, key=lambda item: cast(JsonDict, item).get("id", -1))
    ):
        if not isinstance(annotation, dict):
            raise ClassifierDataValidationError(
                f"Expected 'annotations[{index}]' to be a JSON object."
            )
        annotation_record = cast(JsonDict, annotation)
        annotation_id = _require_int(annotation_record, "id")
        image_id = _require_int(annotation_record, "image_id")
        category_id = _require_int(annotation_record, "category_id")
        product_code = _require_string(annotation_record, "product_code")
        product_name = _require_string(annotation_record, "product_name")
        corrected = _require_bool(annotation_record, "corrected")
        bbox = _require_bbox(annotation_record, "bbox")

        class_entry = class_by_category_id.get(category_id)
        if class_entry is None:
            raise ClassifierDataValidationError(
                "Missing classifier class-map entry for annotation "
                f"{annotation_id}: category_id={category_id}, "
                f"product_code='{product_code}', product_name='{product_name}'."
            )
        if (
            class_entry.product_code != product_code
            or class_entry.product_name != product_name
        ):
            raise ClassifierDataValidationError(
                "Classifier product mapping mismatch for annotation "
                f"{annotation_id}: category_id={category_id}, expected "
                f"product_code='{class_entry.product_code}' and "
                f"product_name='{class_entry.product_name}'."
            )

        image_file_name = images_by_id.get(image_id)
        if image_file_name is None:
            raise ClassifierDataValidationError(
                "Missing image metadata for annotation "
                f"{annotation_id}: image_id={image_id}."
            )
        seen_image_ids.add(image_id)
        image_suffix = Path(image_file_name).suffix or ".jpg"
        crop_path = (
            f"{config.outputs.crops_dir}/{class_entry.class_id:06d}/"
            f"{annotation_id:06d}_{product_code}{image_suffix}"
        )
        crop_records.append(
            {
                "annotation_id": annotation_id,
                "bbox_xywh": bbox,
                "category_id": category_id,
                "class_id": class_entry.class_id,
                "corrected": corrected,
                "crop_path": crop_path,
                "image_id": image_id,
                "product_code": product_code,
                "product_name": product_name,
                "reference_image_paths": list(class_entry.reference_image_paths),
                "source_image_path": f"{images_dir}/{image_file_name}",
            }
        )

    class_map_payload: JsonDict = {
        "baseline": config.baseline,
        "classes": [
            {
                "category_ids": list(entry.category_ids),
                "class_id": entry.class_id,
                "product_code": entry.product_code,
                "product_name": entry.product_name,
                "reference_image_paths": list(entry.reference_image_paths),
            }
            for entry in class_entries
        ],
        "crop_source": config.data.crop_source,
        "index_by_category_id": {
            str(category_id): entry.class_id
            for category_id, entry in sorted(class_by_category_id.items())
        },
        "index_by_product_code": {
            entry.product_code: entry.class_id
            for entry in sorted(class_entries, key=lambda item: item.class_id)
        },
        "runtime": {
            "backbone_library": config.runtime.backbone_library,
            "backbone_library_version": config.runtime.backbone_library_version,
        },
        "schema_version": 1,
    }
    crop_manifest_payload: JsonDict = {
        "baseline": config.baseline,
        "class_map_path": config.outputs.class_map_path,
        "counts": {
            "class_count": len(class_entries),
            "crop_count": len(crop_records),
            "image_count": len(seen_image_ids),
        },
        "crop_source": config.data.crop_source,
        "crops": crop_records,
        "model": {
            "backbone": config.model.backbone,
            "image_size": config.model.image_size,
        },
        "outputs": {
            "crops_dir": config.outputs.crops_dir,
            "manifest_path": config.outputs.crop_manifest_path,
        },
        "processed_inputs": {
            "annotations_path": config.data.annotations_path,
            "categories_path": config.data.categories_path,
            "processed_manifest_path": config.data.processed_manifest_path,
            "product_index_path": config.data.product_index_path,
            "reference_metadata_path": config.data.reference_metadata_path,
        },
        "runtime": {
            "backbone_library": config.runtime.backbone_library,
            "backbone_library_version": config.runtime.backbone_library_version,
        },
        "schema_version": 1,
    }

    class_map_path = output_root / config.outputs.class_map_path
    crop_manifest_path = output_root / config.outputs.crop_manifest_path
    write_json(class_map_path, class_map_payload)
    write_json(crop_manifest_path, crop_manifest_payload)
    return crop_manifest_payload
