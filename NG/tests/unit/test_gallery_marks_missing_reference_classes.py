from __future__ import annotations

import json
from pathlib import Path

from src.ng_data.data.manifest import write_json
from src.ng_data.retrieval.build_gallery import main as build_gallery_main


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    processed_root = tmp_path / "data/processed"
    (processed_root / "reference_images/1111111111111").mkdir(
        parents=True, exist_ok=True
    )
    (processed_root / "reference_images/1111111111111/main.jpg").write_bytes(
        b"only-main"
    )
    write_json(
        processed_root / "product_index.json",
        [
            {
                "annotation_count": 1,
                "category_ids": [10],
                "metadata_annotation_count": 1,
                "product_code": "1111111111111",
                "product_name": "HAS REFS",
                "reference_image_paths": ["reference_images/1111111111111/main.jpg"],
            },
            {
                "annotation_count": 0,
                "category_ids": [11],
                "metadata_annotation_count": 0,
                "product_code": "9999999999999",
                "product_name": "NO REFS",
                "reference_image_paths": [],
            },
        ],
    )
    write_json(
        processed_root / "reference_metadata.json",
        {
            "products": [
                {
                    "annotation_count": 1,
                    "image_names": ["main.jpg"],
                    "product_code": "1111111111111",
                    "product_name": "HAS REFS",
                },
                {
                    "annotation_count": 0,
                    "image_names": [],
                    "product_code": "9999999999999",
                    "product_name": "NO REFS",
                },
            ]
        },
    )
    write_json(
        processed_root / "manifests/dataset_manifest.json",
        {
            "counts": {
                "annotation_count": 0,
                "category_count": 2,
                "image_count": 0,
                "reference_product_count": 2,
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
    config_path = tmp_path / "configs/retrieval/gallery.json"
    write_json(
        config_path,
        {
            "gallery": {
                "embedding_dim": 8,
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
    return processed_root, config_path


def test_gallery_manifest_reports_missing_reference_coverage(tmp_path: Path) -> None:
    _, config_path = _write_fixture(tmp_path)

    exit_code = build_gallery_main(["--config", str(config_path)])
    assert exit_code == 0

    manifest = json.loads(
        (tmp_path / "artifacts/retrieval/gallery_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    missing_product = next(
        product
        for product in manifest["products"]
        if product["product_code"] == "9999999999999"
    )

    assert missing_product["prototype_count"] == 0
    assert missing_product["prototype_index_range"] == [1, 1]
    assert missing_product["available_views"] == []
    assert "main.jpg" in missing_product["missing_views"]
    assert manifest["counts"]["products_without_prototypes"] == 1
