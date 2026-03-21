from __future__ import annotations

import json
from pathlib import Path

from src.ng_data.data.manifest import write_json
from src.ng_data.retrieval.audit_gallery import main as audit_gallery_main
from src.ng_data.retrieval.build_gallery import main as build_gallery_main


def _write_processed_fixture(tmp_path: Path, *, include_empty_product: bool) -> Path:
    processed_root = tmp_path / "data/processed"
    manifest_path = processed_root / "manifests/dataset_manifest.json"
    product_index_path = processed_root / "product_index.json"
    reference_metadata_path = processed_root / "reference_metadata.json"

    (processed_root / "reference_images/1111111111111").mkdir(
        parents=True, exist_ok=True
    )
    (processed_root / "reference_images/2222222222222").mkdir(
        parents=True, exist_ok=True
    )
    (processed_root / "reference_images/1111111111111/main.jpg").write_bytes(
        b"main-one"
    )
    (processed_root / "reference_images/1111111111111/front.jpg").write_bytes(
        b"front-one"
    )
    (processed_root / "reference_images/2222222222222/main.jpg").write_bytes(
        b"main-two"
    )

    products: list[dict[str, object]] = [
        {
            "annotation_count": 1,
            "category_ids": [0],
            "metadata_annotation_count": 1,
            "product_code": "1111111111111",
            "product_name": "TEST PRODUCT ONE",
            "reference_image_paths": [
                "reference_images/1111111111111/front.jpg",
                "reference_images/1111111111111/main.jpg",
            ],
        },
        {
            "annotation_count": 1,
            "category_ids": [356],
            "metadata_annotation_count": 1,
            "product_code": "2222222222222",
            "product_name": "unknown_product",
            "reference_image_paths": ["reference_images/2222222222222/main.jpg"],
        },
    ]
    metadata_products: list[dict[str, object]] = [
        {
            "annotation_count": 1,
            "image_names": ["front.jpg", "main.jpg"],
            "product_code": "1111111111111",
            "product_name": "TEST PRODUCT ONE",
        },
        {
            "annotation_count": 1,
            "image_names": ["main.jpg"],
            "product_code": "2222222222222",
            "product_name": "unknown_product",
        },
    ]

    if include_empty_product:
        products.append(
            {
                "annotation_count": 0,
                "category_ids": [999],
                "metadata_annotation_count": 0,
                "product_code": "3333333333333",
                "product_name": "MISSING REFS",
                "reference_image_paths": [],
            }
        )
        metadata_products.append(
            {
                "annotation_count": 0,
                "image_names": [],
                "product_code": "3333333333333",
                "product_name": "MISSING REFS",
            }
        )

    write_json(product_index_path, products)
    write_json(reference_metadata_path, {"products": metadata_products})
    write_json(
        manifest_path,
        {
            "counts": {
                "annotation_count": 0,
                "category_count": len(products),
                "image_count": 0,
                "reference_product_count": len(products),
            },
            "processed_outputs": {
                "product_index": {"kind": "file", "path": "product_index.json"},
                "reference_images": {"kind": "directory", "path": "reference_images"},
                "reference_metadata": {
                    "kind": "file",
                    "path": "reference_metadata.json",
                },
            },
            "raw_archives": {},
            "raw_root_relative": "../raw",
            "schema_version": 1,
        },
    )
    return processed_root


def _write_gallery_config(tmp_path: Path, processed_root: Path) -> Path:
    config_path = tmp_path / "configs/retrieval/gallery.json"
    write_json(
        config_path,
        {
            "gallery": {
                "embedding_dim": 16,
                "normalize": True,
                "prototype_strategy": "mean_reference_hash",
            },
            "output": {
                "index_path": str(tmp_path / "artifacts/retrieval/gallery_index.npz"),
                "manifest_path": str(
                    tmp_path / "artifacts/retrieval/gallery_manifest.json"
                ),
            },
            "processed_manifest_path": "manifests/dataset_manifest.json",
            "processed_root": str(processed_root),
            "schema_version": 1,
        },
    )
    return config_path


def test_build_gallery_creates_index_and_manifest(tmp_path: Path) -> None:
    import importlib

    np = importlib.import_module("numpy")

    processed_root = _write_processed_fixture(tmp_path, include_empty_product=False)
    config_path = _write_gallery_config(tmp_path, processed_root)

    exit_code = build_gallery_main(["--config", str(config_path)])
    assert exit_code == 0

    index_path = tmp_path / "artifacts/retrieval/gallery_index.npz"
    manifest_path = tmp_path / "artifacts/retrieval/gallery_manifest.json"
    assert index_path.is_file()
    assert manifest_path.is_file()

    audit_exit = audit_gallery_main(
        ["--index", str(index_path), "--manifest", str(manifest_path)]
    )
    assert audit_exit == 0

    with np.load(index_path) as payload:
        embeddings = payload["embeddings"]
        prototype_codes = payload["prototype_product_codes"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert embeddings.shape == (2, 16)
    assert prototype_codes.tolist() == ["1111111111111", "2222222222222"]
    assert manifest["counts"] == {
        "product_count": 2,
        "products_with_missing_views": 2,
        "products_without_prototypes": 0,
        "prototype_count": 2,
    }
    assert manifest["products"][0]["prototype_index_range"] == [0, 1]
    assert manifest["products"][1]["prototype_index_range"] == [1, 2]
    assert manifest["products"][1]["missing_views"]
