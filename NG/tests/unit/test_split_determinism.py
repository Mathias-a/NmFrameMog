from __future__ import annotations

import json
from pathlib import Path

from src.ng_data.data.manifest import write_json
from src.ng_data.eval.make_splits import main as make_splits_main


def _write_dataset(tmp_path: Path, image_ids: list[int]) -> Path:
    processed_root = tmp_path / "data/processed"
    annotations_path = processed_root / "annotations/instances.coco.json"
    manifest_path = processed_root / "manifests/dataset_manifest.json"

    images = [
        {
            "file_name": f"shelves/img_{image_id:05d}.jpg",
            "height": 480,
            "id": image_id,
            "width": 640,
        }
        for image_id in image_ids
    ]
    annotations = [
        {
            "area": 400.0,
            "bbox": [10.0, 20.0, 20.0, 20.0],
            "category_id": image_id % 2,
            "corrected": True,
            "id": index,
            "image_id": image_id,
            "iscrowd": 0,
            "product_code": f"code-{image_id}",
            "product_name": f"product-{image_id}",
        }
        for index, image_id in enumerate(image_ids, start=1)
    ]
    write_json(
        annotations_path,
        {
            "annotations": annotations,
            "categories": [
                {"id": 0, "name": "zero", "supercategory": "product"},
                {"id": 1, "name": "one", "supercategory": "product"},
            ],
            "images": images,
        },
    )
    write_json(
        manifest_path,
        {
            "counts": {
                "annotation_count": len(annotations),
                "category_count": 2,
                "image_count": len(images),
                "reference_product_count": len(images),
            },
            "processed_outputs": {
                "annotations": {
                    "kind": "file",
                    "path": "annotations/instances.coco.json",
                    "sha256": "unused",
                    "size_bytes": 0,
                }
            },
            "raw_archives": {},
            "raw_root_relative": "../raw",
            "schema_version": 1,
        },
    )
    return manifest_path


def _write_split_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "configs/splits.json"
    write_json(
        config_path,
        {
            "cv_folds": 4,
            "holdout_fraction": 0.2,
            "output_path": "manifests/splits.json",
            "schema_version": 1,
            "seed": 7,
        },
    )
    return config_path


def test_make_splits_is_deterministic_and_image_level(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path, [11, 12, 13, 14, 15])
    config_path = _write_split_config(tmp_path)
    output_path = manifest_path.parent / "splits.json"

    first_exit = make_splits_main(
        ["--config", str(config_path), "--manifest", str(manifest_path)]
    )
    first_payload_text = output_path.read_text(encoding="utf-8")

    second_exit = make_splits_main(
        ["--config", str(config_path), "--manifest", str(manifest_path)]
    )
    second_payload_text = output_path.read_text(encoding="utf-8")

    assert first_exit == 0
    assert second_exit == 0
    assert first_payload_text == second_payload_text

    payload = json.loads(first_payload_text)
    assert payload["counts"] == {
        "cv_folds": 4,
        "cv_pool_images": 4,
        "holdout_images": 1,
        "total_images": 5,
    }
    holdout_ids = payload["holdout"]["image_ids"]
    assert len(holdout_ids) == 1
    assert set(holdout_ids).issubset({11, 12, 13, 14, 15})

    fold_validation_ids = sorted(
        image_id for fold in payload["folds"] for image_id in fold["val_image_ids"]
    )
    assert fold_validation_ids == sorted(payload["cv_pool_image_ids"])
    assert sorted(holdout_ids + fold_validation_ids) == [11, 12, 13, 14, 15]

    for fold in payload["folds"]:
        assert set(fold["train_image_ids"]).isdisjoint(fold["val_image_ids"])
        assert set(fold["train_image_ids"]) | set(fold["val_image_ids"]) == set(
            payload["cv_pool_image_ids"]
        )
