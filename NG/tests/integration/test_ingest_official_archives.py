from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, cast

from src.ng_data.data.audit_manifest import main as audit_manifest_main
from src.ng_data.data.ingest import main as ingest_main

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures/ingest_official"


def _write_zip_from_tree(source_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for source_path in sorted(
            path for path in source_root.rglob("*") if path.is_file()
        ):
            archive.write(
                source_path, arcname=source_path.relative_to(source_root).as_posix()
            )


def _create_valid_archives(raw_root: Path) -> None:
    _write_zip_from_tree(
        FIXTURE_ROOT / "coco_source",
        raw_root / "NM_NGD_coco_dataset.zip",
    )
    _write_zip_from_tree(
        FIXTURE_ROOT / "reference_source",
        raw_root / "NM_NGD_product_images.zip",
    )


def test_official_archives_ingest_into_canonical_layout(tmp_path: Path) -> None:
    raw_root = tmp_path / "data/raw"
    processed_root = tmp_path / "data/processed"
    _create_valid_archives(raw_root)

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs/data/main.json"

    first_exit = ingest_main(
        [
            "--config",
            str(config_path),
            "--raw",
            str(raw_root),
            "--processed",
            str(processed_root),
        ]
    )
    manifest_path = processed_root / "manifests/dataset_manifest.json"
    first_manifest = manifest_path.read_text(encoding="utf-8")

    second_exit = ingest_main(
        [
            "--config",
            str(config_path),
            "--raw",
            str(raw_root),
            "--processed",
            str(processed_root),
        ]
    )
    second_manifest = manifest_path.read_text(encoding="utf-8")

    assert first_exit == 0
    assert second_exit == 0
    assert first_manifest == second_manifest

    payload = cast(dict[str, Any], json.loads(first_manifest))
    assert payload["counts"] == {
        "annotation_count": 2,
        "category_count": 2,
        "image_count": 2,
        "reference_product_count": 2,
    }
    assert (
        payload["processed_outputs"]["annotations"]["path"]
        == "annotations/instances.coco.json"
    )
    assert payload["processed_outputs"]["categories"]["path"] == "categories.json"
    assert payload["processed_outputs"]["images"]["path"] == "images"
    assert (
        payload["processed_outputs"]["reference_images"]["path"] == "reference_images"
    )
    assert (
        payload["processed_outputs"]["reference_metadata"]["path"]
        == "reference_metadata.json"
    )
    assert payload["processed_outputs"]["product_index"]["path"] == "product_index.json"

    categories = cast(
        list[dict[str, Any]],
        json.loads((processed_root / "categories.json").read_text(encoding="utf-8")),
    )
    assert categories == [
        {"id": 0, "name": "TEST PRODUCT ONE", "supercategory": "product"},
        {"id": 356, "name": "unknown_product", "supercategory": "product"},
    ]

    product_index = cast(
        list[dict[str, Any]],
        json.loads((processed_root / "product_index.json").read_text(encoding="utf-8")),
    )
    assert product_index == [
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

    assert (processed_root / "images/shelves/img_00001.jpg").read_text(
        encoding="utf-8"
    ) == "fake-jpeg-image-00001\n"
    assert (processed_root / "images/shelves/img_00002.jpg").read_text(
        encoding="utf-8"
    ) == "fake-jpeg-image-00002\n"

    audit_exit = audit_manifest_main(["--manifest", str(manifest_path)])
    assert audit_exit == 0
