from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.ng_data.data.ingest import REFERENCE_IMAGE_NAMES
from src.ng_data.data.manifest import file_snapshot, load_manifest, write_json

JsonDict = dict[str, Any]
np = importlib.import_module("numpy")


class GalleryBuildError(ValueError):
    pass


@dataclass(frozen=True)
class GalleryOutputConfig:
    index_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class GalleryConfig:
    embedding_dim: int
    normalize: bool
    output: GalleryOutputConfig
    processed_manifest_path: Path
    processed_root: Path
    prototype_strategy: str
    schema_version: int


@dataclass(frozen=True)
class ProductRecord:
    category_ids: tuple[int, ...]
    product_code: str
    product_name: str
    reference_image_paths: tuple[str, ...]


@dataclass(frozen=True)
class MetadataRecord:
    image_names: tuple[str, ...]
    product_code: str
    product_name: str


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GalleryBuildError(f"Expected file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise GalleryBuildError(f"Invalid JSON file: {path}") from error


def _load_json_object(path: Path) -> JsonDict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise GalleryBuildError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _load_json_list(path: Path) -> list[object]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise GalleryBuildError(f"Expected JSON list in {path}")
    return cast(list[object], payload)


def _require_bool(data: JsonDict, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise GalleryBuildError(f"Expected '{key}' to be a boolean.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise GalleryBuildError(f"Expected '{key}' to be an integer.")
    return value


def _require_list(data: JsonDict, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise GalleryBuildError(f"Expected '{key}' to be a list.")
    return cast(list[object], value)


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise GalleryBuildError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise GalleryBuildError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_string_list(data: JsonDict, key: str) -> list[str]:
    values = _require_list(data, key)
    strings: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or value == "":
            raise GalleryBuildError(
                f"Expected '{key}[{index}]' to be a non-empty string."
            )
        strings.append(value)
    return strings


def load_gallery_config(config_path: str | Path) -> GalleryConfig:
    path = Path(config_path)
    payload = _load_json_object(path)
    output_payload = _require_mapping(payload, "output")
    gallery_payload = _require_mapping(payload, "gallery")
    processed_root = Path(_require_string(payload, "processed_root"))
    return GalleryConfig(
        embedding_dim=_require_int(gallery_payload, "embedding_dim"),
        normalize=_require_bool(gallery_payload, "normalize"),
        output=GalleryOutputConfig(
            index_path=Path(_require_string(output_payload, "index_path")),
            manifest_path=Path(_require_string(output_payload, "manifest_path")),
        ),
        processed_manifest_path=Path(
            _require_string(payload, "processed_manifest_path")
        ),
        processed_root=processed_root,
        prototype_strategy=_require_string(gallery_payload, "prototype_strategy"),
        schema_version=_require_int(payload, "schema_version"),
    )


def _resolve_processed_inputs(config: GalleryConfig) -> tuple[Path, Path, Path]:
    manifest_path = config.processed_root / config.processed_manifest_path
    manifest = load_manifest(manifest_path)
    processed_outputs = _require_mapping(manifest, "processed_outputs")

    product_index_entry = _require_mapping(processed_outputs, "product_index")
    _require_mapping(processed_outputs, "reference_images")
    reference_metadata_entry = _require_mapping(processed_outputs, "reference_metadata")
    return (
        manifest_path,
        config.processed_root / _require_string(product_index_entry, "path"),
        config.processed_root / _require_string(reference_metadata_entry, "path"),
    )


def _load_product_records(path: Path) -> list[ProductRecord]:
    payload = _load_json_list(path)
    records: list[ProductRecord] = []
    for index, value in enumerate(payload):
        if not isinstance(value, dict):
            raise GalleryBuildError(
                f"Expected product_index[{index}] to be a JSON object."
            )
        product = cast(JsonDict, value)
        category_ids = tuple(sorted(_require_int_list(product, "category_ids")))
        records.append(
            ProductRecord(
                category_ids=category_ids,
                product_code=_require_string(product, "product_code"),
                product_name=_require_string(product, "product_name"),
                reference_image_paths=tuple(
                    sorted(_require_string_list(product, "reference_image_paths"))
                ),
            )
        )
    return sorted(records, key=lambda item: (item.category_ids, item.product_code))


def _require_int_list(data: JsonDict, key: str) -> list[int]:
    values = _require_list(data, key)
    integers: list[int] = []
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise GalleryBuildError(f"Expected '{key}[{index}]' to be an integer.")
        integers.append(value)
    return integers


def _load_reference_metadata(path: Path) -> dict[str, MetadataRecord]:
    payload = _load_json_object(path)
    products = _require_list(payload, "products")
    records: dict[str, MetadataRecord] = {}
    for index, value in enumerate(products):
        if not isinstance(value, dict):
            raise GalleryBuildError(
                f"Expected reference_metadata.products[{index}] to be an object."
            )
        record = cast(JsonDict, value)
        product_code = _require_string(record, "product_code")
        records[product_code] = MetadataRecord(
            image_names=tuple(sorted(set(_require_string_list(record, "image_names")))),
            product_code=product_code,
            product_name=_require_string(record, "product_name"),
        )
    return records


def _vector_from_bytes(raw: bytes, dimension: int) -> Any:
    if dimension < 1:
        raise GalleryBuildError("gallery.embedding_dim must be at least 1")
    seed = raw or b"empty"
    buffer = bytearray()
    counter = 0
    while len(buffer) < dimension:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        buffer.extend(digest)
        counter += 1
    values = np.frombuffer(bytes(buffer[:dimension]), dtype=np.uint8).astype(np.float32)
    return values / 255.0


def _load_reference_vector(path: Path, dimension: int) -> Any:
    raw = path.read_bytes() if path.exists() else path.as_posix().encode("utf-8")
    return _vector_from_bytes(raw, dimension)


def _normalize(vector: Any) -> Any:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def build_gallery(config_path: str | Path) -> JsonDict:
    config = load_gallery_config(config_path)
    if config.schema_version != 1:
        raise GalleryBuildError("Unsupported gallery config schema version.")

    manifest_path, product_index_path, reference_metadata_path = (
        _resolve_processed_inputs(config)
    )
    product_records = _load_product_records(product_index_path)
    metadata_by_code = _load_reference_metadata(reference_metadata_path)
    reference_views = sorted(name for name in REFERENCE_IMAGE_NAMES)

    embeddings: list[Any] = []
    prototype_codes: list[str] = []
    prototype_category_ids: list[int] = []
    products_payload: list[JsonDict] = []
    products_with_missing_views = 0
    products_without_prototypes = 0

    for product in product_records:
        metadata = metadata_by_code.get(product.product_code)
        if metadata is None:
            raise GalleryBuildError(
                f"Missing reference metadata for product_code '{product.product_code}'."
            )
        if metadata.product_name != product.product_name:
            raise GalleryBuildError(
                "Reference metadata name mismatch for product_code "
                f"'{product.product_code}'."
            )

        available_views = sorted(
            {
                Path(reference_path).name
                for reference_path in product.reference_image_paths
                if (config.processed_root / reference_path).exists()
            }
        )
        missing_views = sorted(set(reference_views) - set(available_views))
        products_with_missing_views += int(bool(missing_views))

        start_index = len(embeddings)
        existing_paths = [
            config.processed_root / reference_path
            for reference_path in product.reference_image_paths
            if (config.processed_root / reference_path).exists()
        ]
        if existing_paths:
            vectors = [
                _load_reference_vector(path, config.embedding_dim)
                for path in existing_paths
            ]
            prototype = np.mean(np.stack(vectors, axis=0), axis=0)
            if config.normalize:
                prototype = _normalize(prototype)
            embeddings.append(prototype.astype(np.float32))
            prototype_codes.append(product.product_code)
            prototype_category_ids.append(
                product.category_ids[0] if product.category_ids else -1
            )
        else:
            products_without_prototypes += 1
        end_index = len(embeddings)

        products_payload.append(
            {
                "available_views": available_views,
                "category_ids": list(product.category_ids),
                "missing_views": missing_views,
                "product_code": product.product_code,
                "product_name": product.product_name,
                "prototype_count": end_index - start_index,
                "prototype_index_range": [start_index, end_index],
                "reference_image_paths": list(product.reference_image_paths),
            }
        )

    embedding_array = (
        np.stack(embeddings, axis=0).astype(np.float32)
        if embeddings
        else np.zeros((0, config.embedding_dim), dtype=np.float32)
    )

    config.output.index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output.index_path,
        embeddings=embedding_array,
        prototype_category_ids=np.asarray(prototype_category_ids, dtype=np.int64),
        prototype_product_codes=np.asarray(prototype_codes),
    )

    manifest_payload: JsonDict = {
        "config_path": str(config_path),
        "counts": {
            "product_count": len(products_payload),
            "products_with_missing_views": products_with_missing_views,
            "products_without_prototypes": products_without_prototypes,
            "prototype_count": int(embedding_array.shape[0]),
        },
        "coverage": {
            "reference_view_vocabulary": reference_views,
        },
        "index": {
            "embedding_dim": config.embedding_dim,
            "format": "npz",
            "normalize": config.normalize,
            "path": config.output.index_path.as_posix(),
            "prototype_strategy": config.prototype_strategy,
            **file_snapshot(config.output.index_path),
        },
        "processed_inputs": {
            "processed_manifest_path": manifest_path.as_posix(),
            "product_index_path": product_index_path.as_posix(),
            "reference_metadata_path": reference_metadata_path.as_posix(),
        },
        "products": products_payload,
        "schema_version": 1,
    }
    write_json(config.output.manifest_path, manifest_payload)
    return manifest_payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic retrieval gallery"
    )
    parser.add_argument("--config", required=True, help="Path to gallery config JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    build_gallery(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
