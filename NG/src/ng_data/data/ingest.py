from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from src.ng_data.data.config import DataConfig, load_data_config
from src.ng_data.data.manifest import build_dataset_manifest, write_json
from src.ng_data.data.validation import validate_data_config

JsonDict = dict[str, Any]
IMAGE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
REFERENCE_IMAGE_NAMES = frozenset(
    {
        "back.jpg",
        "bottom.jpg",
        "front.jpg",
        "left.jpg",
        "main.jpg",
        "right.jpg",
        "top.jpg",
    }
)


class DataIngestError(ValueError):
    pass


@dataclass(frozen=True)
class IngestPaths:
    raw_root: Path
    processed_root: Path
    coco_archive: Path
    reference_archive: Path
    coco_extract_root: Path
    reference_extract_root: Path
    images_dir: Path
    annotations_path: Path
    categories_path: Path
    reference_images_dir: Path
    reference_metadata_path: Path
    product_index_path: Path
    manifest_path: Path


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise DataIngestError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_list(data: JsonDict, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise DataIngestError(f"Expected '{key}' to be a list.")
    return cast(list[object], value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise DataIngestError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataIngestError(f"Expected '{key}' to be an integer.")
    return value


def _require_bool(data: JsonDict, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise DataIngestError(f"Expected '{key}' to be a boolean.")
    return value


def _require_number(data: JsonDict, key: str) -> int | float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DataIngestError(f"Expected '{key}' to be numeric.")
    return cast(int | float, value)


def _load_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DataIngestError(f"Invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise DataIngestError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def resolve_ingest_paths(
    config: DataConfig, raw_root: Path, processed_root: Path
) -> IngestPaths:
    layout = config.processed_layout
    extract_root = raw_root / config.raw_layout.extract_root
    return IngestPaths(
        raw_root=raw_root,
        processed_root=processed_root,
        coco_archive=raw_root / config.raw_archives.coco_dataset,
        reference_archive=raw_root / config.raw_archives.product_images,
        coco_extract_root=extract_root / Path(config.raw_archives.coco_dataset).stem,
        reference_extract_root=extract_root
        / Path(config.raw_archives.product_images).stem,
        images_dir=processed_root / layout.images_dir,
        annotations_path=processed_root / layout.annotations_path,
        categories_path=processed_root / layout.categories_path,
        reference_images_dir=processed_root / layout.reference_images_dir,
        reference_metadata_path=processed_root / layout.reference_metadata_path,
        product_index_path=processed_root / layout.product_index_path,
        manifest_path=processed_root / layout.manifest_path,
    )


def _validate_archive_member(member_name: str) -> PurePosixPath:
    member_path = PurePosixPath(member_name)
    if member_path.is_absolute():
        raise DataIngestError(f"Archive member must be relative: {member_name}")
    if any(part in {"", ".", ".."} for part in member_path.parts):
        raise DataIngestError(
            f"Archive member contains unsafe path segments: {member_name}"
        )
    return member_path


def _extract_archive(archive_path: Path, destination_root: Path) -> None:
    if not archive_path.exists():
        raise DataIngestError(f"Missing required archive: {archive_path}")
    if not archive_path.is_file():
        raise DataIngestError(f"Expected archive path to be a file: {archive_path}")

    shutil.rmtree(destination_root, ignore_errors=True)
    destination_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = _validate_archive_member(member.filename)
                target_path = destination_root / Path(*member_path.parts)
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source_handle:
                    target_path.write_bytes(source_handle.read())
    except zipfile.BadZipFile as error:
        raise DataIngestError(f"Invalid zip archive: {archive_path}") from error


def _find_single_file(root: Path, filename: str) -> Path:
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        raise DataIngestError(f"Missing required file '{filename}' under {root}")
    if len(matches) > 1:
        raise DataIngestError(f"Expected exactly one '{filename}' under {root}")
    return matches[0]


def _build_file_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_relative_path: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(root).as_posix()
        by_relative_path[relative_path] = path
        by_name.setdefault(path.name, []).append(path)
    return by_relative_path, by_name


def _resolve_image_source(root: Path, image_name: str) -> Path:
    by_relative_path, by_name = _build_file_index(root)
    direct_match = by_relative_path.get(image_name)
    if direct_match is not None:
        return direct_match

    image_basename = PurePosixPath(image_name).name
    matches = by_name.get(image_basename, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise DataIngestError(
            f"Ambiguous image file reference '{image_name}' in COCO data"
        )
    raise DataIngestError(f"Missing image file '{image_name}' in COCO archive")


def _normalize_images(images_payload: list[object]) -> list[JsonDict]:
    images: list[JsonDict] = []
    seen_ids: set[int] = set()
    for item in images_payload:
        if not isinstance(item, dict):
            raise DataIngestError("Every COCO image entry must be an object.")
        image = cast(JsonDict, item)
        image_id = _require_int(image, "id")
        if image_id in seen_ids:
            raise DataIngestError(f"Duplicate COCO image id: {image_id}")
        seen_ids.add(image_id)
        width = _require_int(image, "width")
        height = _require_int(image, "height")
        if width <= 0 or height <= 0:
            raise DataIngestError(
                f"Image dimensions must be positive for id {image_id}"
            )
        images.append(
            {
                "file_name": _require_string(image, "file_name"),
                "height": height,
                "id": image_id,
                "width": width,
            }
        )
    return sorted(images, key=lambda value: cast(int, value["id"]))


def _normalize_categories(categories_payload: list[object]) -> list[JsonDict]:
    categories: list[JsonDict] = []
    seen_ids: set[int] = set()
    for item in categories_payload:
        if not isinstance(item, dict):
            raise DataIngestError("Every COCO category entry must be an object.")
        category = cast(JsonDict, item)
        category_id = _require_int(category, "id")
        if category_id in seen_ids:
            raise DataIngestError(f"Duplicate COCO category id: {category_id}")
        seen_ids.add(category_id)
        categories.append(
            {
                "id": category_id,
                "name": _require_string(category, "name"),
                "supercategory": _require_string(category, "supercategory"),
            }
        )
    return sorted(categories, key=lambda value: cast(int, value["id"]))


def _normalize_annotations(
    annotations_payload: list[object], image_ids: set[int], category_ids: set[int]
) -> tuple[list[JsonDict], dict[str, int], dict[str, set[int]], dict[str, str]]:
    annotations: list[JsonDict] = []
    annotation_counts: dict[str, int] = {}
    product_categories: dict[str, set[int]] = {}
    annotation_names: dict[str, str] = {}
    seen_ids: set[int] = set()

    for item in annotations_payload:
        if not isinstance(item, dict):
            raise DataIngestError("Every COCO annotation entry must be an object.")
        annotation = cast(JsonDict, item)
        annotation_id = _require_int(annotation, "id")
        if annotation_id in seen_ids:
            raise DataIngestError(f"Duplicate COCO annotation id: {annotation_id}")
        seen_ids.add(annotation_id)
        image_id = _require_int(annotation, "image_id")
        category_id = _require_int(annotation, "category_id")
        if image_id not in image_ids:
            raise DataIngestError(
                f"Annotation {annotation_id} references unknown image_id {image_id}"
            )
        if category_id not in category_ids:
            raise DataIngestError(
                "Annotation "
                f"{annotation_id} references unknown category_id {category_id}"
            )
        bbox_value = annotation.get("bbox")
        if not isinstance(bbox_value, list) or len(bbox_value) != 4:
            raise DataIngestError(
                f"Annotation {annotation_id} must contain a four-value bbox list."
            )
        bbox: list[int | float] = []
        for index, coordinate in enumerate(bbox_value):
            if not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool):
                raise DataIngestError(
                    f"Annotation {annotation_id} bbox index {index} must be numeric."
                )
            if coordinate < 0:
                raise DataIngestError(
                    "Annotation "
                    f"{annotation_id} bbox index {index} must be non-negative."
                )
            bbox.append(cast(int | float, coordinate))

        area = _require_number(annotation, "area")
        if area < 0:
            raise DataIngestError(
                f"Annotation {annotation_id} area must be non-negative."
            )

        iscrowd = _require_int(annotation, "iscrowd")
        if iscrowd not in {0, 1}:
            raise DataIngestError(
                f"Annotation {annotation_id} iscrowd must be 0 or 1, got {iscrowd}"
            )

        product_code = _require_string(annotation, "product_code")
        product_name = _require_string(annotation, "product_name")
        corrected = _require_bool(annotation, "corrected")

        previous_name = annotation_names.get(product_code)
        if previous_name is not None and previous_name != product_name:
            raise DataIngestError(
                f"Conflicting product_name values for product_code '{product_code}'"
            )
        annotation_names[product_code] = product_name
        annotation_counts[product_code] = annotation_counts.get(product_code, 0) + 1
        product_categories.setdefault(product_code, set()).add(category_id)
        annotations.append(
            {
                "area": area,
                "bbox": bbox,
                "category_id": category_id,
                "corrected": corrected,
                "id": annotation_id,
                "image_id": image_id,
                "iscrowd": iscrowd,
                "product_code": product_code,
                "product_name": product_name,
            }
        )

    return (
        sorted(annotations, key=lambda value: cast(int, value["id"])),
        annotation_counts,
        product_categories,
        annotation_names,
    )


def validate_coco_annotations(
    annotations_path: Path,
) -> tuple[JsonDict, dict[str, int], dict[str, set[int]], dict[str, str]]:
    payload = _load_json(annotations_path)
    images = _normalize_images(_require_list(payload, "images"))
    categories = _normalize_categories(_require_list(payload, "categories"))
    image_ids = {cast(int, image["id"]) for image in images}
    category_ids = {cast(int, category["id"]) for category in categories}
    normalized_annotations, annotation_counts, product_categories, annotation_names = (
        _normalize_annotations(
            _require_list(payload, "annotations"), image_ids, category_ids
        )
    )
    return (
        {
            "annotations": normalized_annotations,
            "categories": categories,
            "images": images,
        },
        annotation_counts,
        product_categories,
        annotation_names,
    )


def _normalize_metadata_products(payload: object) -> list[JsonDict]:
    products_payload: list[object]
    if isinstance(payload, dict):
        payload_dict = cast(JsonDict, payload)
        products_value = payload_dict.get("products")
        if isinstance(products_value, list):
            products_payload = cast(list[object], products_value)
        else:
            products_payload = []
            for product_code, value in sorted(payload_dict.items()):
                if product_code == "products":
                    continue
                if not isinstance(value, dict):
                    raise DataIngestError(
                        "Reference metadata mapping values must be objects."
                    )
                candidate = dict(cast(JsonDict, value))
                candidate["product_code"] = product_code
                products_payload.append(candidate)
    elif isinstance(payload, list):
        products_payload = cast(list[object], payload)
    else:
        raise DataIngestError("Reference metadata must be a JSON object or list.")

    normalized: list[JsonDict] = []
    seen_codes: set[str] = set()
    for item in products_payload:
        if not isinstance(item, dict):
            raise DataIngestError("Reference metadata product entries must be objects.")
        product = cast(JsonDict, item)
        product_code = _require_string(product, "product_code")
        if product_code in seen_codes:
            raise DataIngestError(f"Duplicate metadata product_code: {product_code}")
        seen_codes.add(product_code)
        annotation_count = _require_int(product, "annotation_count")
        if annotation_count < 0:
            raise DataIngestError(
                f"annotation_count must be non-negative for '{product_code}'"
            )
        normalized.append(
            {
                "annotation_count": annotation_count,
                "product_code": product_code,
                "product_name": _require_string(product, "product_name"),
            }
        )
    return sorted(normalized, key=lambda value: cast(str, value["product_code"]))


def validate_reference_archive(
    reference_root: Path,
) -> tuple[Path, list[JsonDict], dict[str, list[str]]]:
    metadata_path = _find_single_file(reference_root, "metadata.json")
    try:
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DataIngestError(f"Invalid JSON file: {metadata_path}") from error
    products = _normalize_metadata_products(metadata_payload)
    reference_base_root = metadata_path.parent

    product_directories = {
        path.name: path
        for path in sorted(reference_base_root.iterdir())
        if path.is_dir()
    }
    metadata_codes = {cast(str, product["product_code"]) for product in products}
    directory_codes = set(product_directories)
    if metadata_codes != directory_codes:
        missing_dirs = sorted(metadata_codes - directory_codes)
        missing_meta = sorted(directory_codes - metadata_codes)
        details: list[str] = []
        if missing_dirs:
            details.append(f"missing directories for {', '.join(missing_dirs)}")
        if missing_meta:
            details.append(f"missing metadata entries for {', '.join(missing_meta)}")
        raise DataIngestError(f"Reference metadata mismatch: {'; '.join(details)}")

    reference_images: dict[str, list[str]] = {}
    for product in products:
        product_code = cast(str, product["product_code"])
        directory = product_directories[product_code]
        image_names = sorted(
            path.name
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_names:
            raise DataIngestError(
                f"Reference product '{product_code}' does not contain any image files."
            )
        if "main.jpg" not in image_names:
            raise DataIngestError(
                f"Reference product '{product_code}' must include main.jpg"
            )
        unexpected = [
            name for name in image_names if name.lower() not in REFERENCE_IMAGE_NAMES
        ]
        if unexpected:
            names_display = ", ".join(unexpected)
            raise DataIngestError(
                "Reference product "
                f"'{product_code}' contains unsupported images: {names_display}"
            )
        reference_images[product_code] = image_names

    return reference_base_root, products, reference_images


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_coco_images(
    images: list[JsonDict], source_root: Path, destination_root: Path
) -> None:
    shutil.rmtree(destination_root, ignore_errors=True)
    destination_root.mkdir(parents=True, exist_ok=True)
    for image in images:
        file_name = cast(str, image["file_name"])
        source_path = _resolve_image_source(source_root, file_name)
        destination_path = destination_root / Path(PurePosixPath(file_name))
        _copy_file(source_path, destination_path)


def _copy_reference_images(
    products: list[JsonDict], reference_root: Path, destination_root: Path
) -> dict[str, list[str]]:
    shutil.rmtree(destination_root, ignore_errors=True)
    destination_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, list[str]] = {}
    for product in products:
        product_code = cast(str, product["product_code"])
        source_dir = reference_root / product_code
        destination_dir = destination_root / product_code
        image_paths: list[str] = []
        for source_path in sorted(
            path for path in source_dir.iterdir() if path.is_file()
        ):
            if source_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            destination_path = destination_dir / source_path.name
            _copy_file(source_path, destination_path)
            image_paths.append(
                destination_path.relative_to(destination_root.parent).as_posix()
            )
        copied[product_code] = image_paths
    return copied


def build_product_index(
    *,
    metadata_products: list[JsonDict],
    annotation_counts: dict[str, int],
    product_categories: dict[str, set[int]],
    annotation_names: dict[str, str],
    reference_images: dict[str, list[str]],
) -> list[JsonDict]:
    all_codes = sorted(
        set(annotation_counts)
        | {cast(str, product["product_code"]) for product in metadata_products}
    )
    metadata_by_code = {
        cast(str, product["product_code"]): product for product in metadata_products
    }
    product_index: list[JsonDict] = []
    for product_code in all_codes:
        metadata = metadata_by_code.get(product_code)
        product_name = annotation_names.get(product_code)
        if metadata is not None:
            product_name = cast(str, metadata["product_name"])
        if product_name is None:
            raise DataIngestError(
                f"Could not determine a product_name for product_code '{product_code}'"
            )
        product_index.append(
            {
                "annotation_count": annotation_counts.get(product_code, 0),
                "category_ids": sorted(product_categories.get(product_code, set())),
                "metadata_annotation_count": (
                    cast(int, metadata["annotation_count"])
                    if metadata is not None
                    else 0
                ),
                "product_code": product_code,
                "product_name": product_name,
                "reference_image_paths": reference_images.get(product_code, []),
            }
        )
    return product_index


def ingest_official_archives(
    config: DataConfig, raw_root: Path, processed_root: Path
) -> JsonDict:
    validate_data_config(config)
    paths = resolve_ingest_paths(config, raw_root, processed_root)

    paths.raw_root.mkdir(parents=True, exist_ok=True)
    paths.processed_root.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.unlink(missing_ok=True)

    _extract_archive(paths.coco_archive, paths.coco_extract_root)
    _extract_archive(paths.reference_archive, paths.reference_extract_root)

    annotations_source = _find_single_file(paths.coco_extract_root, "annotations.json")
    coco_payload, annotation_counts, product_categories, annotation_names = (
        validate_coco_annotations(annotations_source)
    )
    reference_base_root, metadata_products, validated_reference_images = (
        validate_reference_archive(paths.reference_extract_root)
    )

    _copy_coco_images(
        cast(list[JsonDict], coco_payload["images"]),
        paths.coco_extract_root,
        paths.images_dir,
    )
    copied_reference_images = _copy_reference_images(
        metadata_products,
        reference_base_root,
        paths.reference_images_dir,
    )

    categories = cast(list[JsonDict], coco_payload["categories"])
    write_json(paths.annotations_path, coco_payload)
    write_json(paths.categories_path, categories)

    reference_metadata = {
        "products": [
            {
                "annotation_count": cast(int, product["annotation_count"]),
                "image_names": validated_reference_images[
                    cast(str, product["product_code"])
                ],
                "product_code": cast(str, product["product_code"]),
                "product_name": cast(str, product["product_name"]),
            }
            for product in metadata_products
        ]
    }
    write_json(paths.reference_metadata_path, reference_metadata)

    product_index = build_product_index(
        metadata_products=metadata_products,
        annotation_counts=annotation_counts,
        product_categories=product_categories,
        annotation_names=annotation_names,
        reference_images=copied_reference_images,
    )
    write_json(paths.product_index_path, product_index)

    counts = {
        "annotation_count": len(cast(list[JsonDict], coco_payload["annotations"])),
        "category_count": len(categories),
        "image_count": len(cast(list[JsonDict], coco_payload["images"])),
        "reference_product_count": len(metadata_products),
    }
    manifest = build_dataset_manifest(
        processed_root=paths.processed_root,
        raw_root=paths.raw_root,
        raw_archives={
            "coco_dataset": paths.coco_archive,
            "product_images": paths.reference_archive,
        },
        extracted_roots={
            "coco_dataset": paths.coco_extract_root,
            "product_images": paths.reference_extract_root,
        },
        output_paths={
            "annotations": paths.annotations_path,
            "categories": paths.categories_path,
            "images": paths.images_dir,
            "product_index": paths.product_index_path,
            "reference_images": paths.reference_images_dir,
            "reference_metadata": paths.reference_metadata_path,
        },
        counts=counts,
    )
    write_json(paths.manifest_path, manifest)
    return manifest


class IngestArgs(argparse.Namespace):
    config: str
    processed: str
    raw: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest the official NorgesGruppen archives into canonical local layouts."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/data/main.json",
        help="Path to the data ingestion config JSON file.",
    )
    parser.add_argument(
        "--raw",
        default="data/raw",
        help=(
            "Raw data directory containing NM_NGD_coco_dataset.zip and "
            "NM_NGD_product_images.zip."
        ),
    )
    parser.add_argument(
        "--processed",
        default="data/processed",
        help=(
            "Processed output directory for canonical annotations, images, "
            "and manifests."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(IngestArgs, parser.parse_args(argv))
    try:
        config = load_data_config(args.config)
        manifest = ingest_official_archives(
            config,
            raw_root=Path(args.raw),
            processed_root=Path(args.processed),
        )
    except (DataIngestError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
