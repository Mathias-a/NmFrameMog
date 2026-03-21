from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]
np = importlib.import_module("numpy")


class GalleryAuditError(ValueError):
    pass


def _load_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GalleryAuditError(f"Expected file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise GalleryAuditError(f"Invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise GalleryAuditError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise GalleryAuditError(f"Expected '{key}' to be an integer.")
    return value


def _require_list(data: JsonDict, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise GalleryAuditError(f"Expected '{key}' to be a list.")
    return cast(list[object], value)


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise GalleryAuditError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise GalleryAuditError(f"Expected '{key}' to be a non-empty string.")
    return value


def audit_gallery(*, index_path: str | Path, manifest_path: str | Path) -> JsonDict:
    index_file = Path(index_path)
    manifest_file = Path(manifest_path)
    manifest = _load_json_object(manifest_file)
    if _require_int(manifest, "schema_version") != 1:
        raise GalleryAuditError("Unsupported gallery manifest schema version.")

    products = _require_list(manifest, "products")
    counts = _require_mapping(manifest, "counts")
    index = _require_mapping(manifest, "index")

    with np.load(index_file) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        product_codes = np.asarray(data["prototype_product_codes"])
        category_ids = np.asarray(data["prototype_category_ids"], dtype=np.int64)

    if embeddings.ndim != 2:
        raise GalleryAuditError("Gallery embeddings must be a 2D matrix.")
    if (
        embeddings.shape[0] != product_codes.shape[0]
        or embeddings.shape[0] != category_ids.shape[0]
    ):
        raise GalleryAuditError(
            "Gallery index arrays must have matching first dimension."
        )
    if embeddings.shape[0] != _require_int(counts, "prototype_count"):
        raise GalleryAuditError("Prototype count mismatch between manifest and index.")
    if embeddings.shape[1] != _require_int(index, "embedding_dim"):
        raise GalleryAuditError(
            "Embedding dimension mismatch between manifest and index."
        )
    if _require_string(index, "path") != index_file.as_posix():
        raise GalleryAuditError(
            "Manifest index.path does not match the audited index path."
        )

    covered_rows = 0
    for product_index, value in enumerate(products):
        if not isinstance(value, dict):
            raise GalleryAuditError(
                f"Expected products[{product_index}] to be an object."
            )
        product = cast(JsonDict, value)
        prototype_range = _require_list(product, "prototype_index_range")
        if len(prototype_range) != 2:
            raise GalleryAuditError("prototype_index_range must contain [start, end].")
        start = prototype_range[0]
        end = prototype_range[1]
        if not isinstance(start, int) or not isinstance(end, int):
            raise GalleryAuditError("prototype_index_range values must be integers.")
        if start < 0 or end < start or end > embeddings.shape[0]:
            raise GalleryAuditError("prototype_index_range is out of bounds.")
        if end - start != _require_int(product, "prototype_count"):
            raise GalleryAuditError(
                "prototype_count does not match prototype_index_range."
            )
        covered_rows += end - start

    if covered_rows != embeddings.shape[0]:
        raise GalleryAuditError(
            "Prototype ranges do not cover the gallery index rows exactly."
        )

    return {
        "embedding_dim": int(embeddings.shape[1]),
        "manifest": manifest_file.as_posix(),
        "prototype_count": int(embeddings.shape[0]),
        "status": "ok",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit deterministic retrieval gallery"
    )
    parser.add_argument("--index", required=True, help="Path to gallery_index.npz")
    parser.add_argument(
        "--manifest", required=True, help="Path to gallery_manifest.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    audit_gallery(index_path=args.index, manifest_path=args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
