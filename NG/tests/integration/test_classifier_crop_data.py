from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

from src.ng_data.classifier.data import (
    ClassifierDataValidationError,
    build_classifier_crop_dataset,
    load_and_validate_classifier_data_config,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_classifier_crop_dataset_builds_deterministic_gt_manifest(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/classifier/search.json"
    processed_root = repo_root / "data/processed"
    output_root = tmp_path / "artifacts/classifier"

    config = load_and_validate_classifier_data_config(config_path, processed_root)
    first_manifest = build_classifier_crop_dataset(
        config_path=config_path,
        processed_root=processed_root,
        output_root=output_root,
    )
    class_map_path = output_root / config.outputs.class_map_path
    crop_manifest_path = output_root / config.outputs.crop_manifest_path
    first_class_map_text = class_map_path.read_text(encoding="utf-8")
    first_crop_manifest_text = crop_manifest_path.read_text(encoding="utf-8")

    second_manifest = build_classifier_crop_dataset(
        config_path=config_path,
        processed_root=processed_root,
        output_root=output_root,
    )

    assert config.runtime.backbone_library == "timm"
    assert config.runtime.backbone_library_version == "0.9.12"
    assert first_manifest == second_manifest
    assert first_class_map_text == class_map_path.read_text(encoding="utf-8")
    assert first_crop_manifest_text == crop_manifest_path.read_text(encoding="utf-8")

    class_map = cast(dict[str, Any], json.loads(first_class_map_text))
    crop_manifest = cast(dict[str, Any], json.loads(first_crop_manifest_text))

    assert class_map["index_by_category_id"] == {"0": 0, "356": 1}
    assert class_map["index_by_product_code"] == {
        "1111111111111": 0,
        "2222222222222": 1,
    }
    assert class_map["classes"] == [
        {
            "category_ids": [0],
            "class_id": 0,
            "product_code": "1111111111111",
            "product_name": "TEST PRODUCT ONE",
            "reference_image_paths": [
                "reference_images/1111111111111/front.jpg",
                "reference_images/1111111111111/main.jpg",
            ],
        },
        {
            "category_ids": [356],
            "class_id": 1,
            "product_code": "2222222222222",
            "product_name": "unknown_product",
            "reference_image_paths": ["reference_images/2222222222222/main.jpg"],
        },
    ]

    assert crop_manifest["counts"] == {
        "class_count": 2,
        "crop_count": 2,
        "image_count": 2,
    }
    assert crop_manifest["class_map_path"] == "manifests/classifier_class_map.json"
    assert crop_manifest["crop_source"] == "ground_truth_boxes"
    assert crop_manifest["processed_inputs"] == {
        "annotations_path": "annotations/instances.coco.json",
        "categories_path": "categories.json",
        "processed_manifest_path": "manifests/dataset_manifest.json",
        "product_index_path": "product_index.json",
        "reference_metadata_path": "reference_metadata.json",
    }
    assert crop_manifest["crops"] == [
        {
            "annotation_id": 1,
            "bbox_xywh": [10.0, 20.0, 30.0, 40.0],
            "category_id": 0,
            "class_id": 0,
            "corrected": True,
            "crop_path": "crops/gt/000000/000001_1111111111111.jpg",
            "image_id": 1,
            "product_code": "1111111111111",
            "product_name": "TEST PRODUCT ONE",
            "reference_image_paths": [
                "reference_images/1111111111111/front.jpg",
                "reference_images/1111111111111/main.jpg",
            ],
            "source_image_path": "images/shelves/img_00001.jpg",
        },
        {
            "annotation_id": 2,
            "bbox_xywh": [5.0, 6.0, 7.0, 8.0],
            "category_id": 356,
            "class_id": 1,
            "corrected": False,
            "crop_path": "crops/gt/000001/000002_2222222222222.jpg",
            "image_id": 2,
            "product_code": "2222222222222",
            "product_name": "unknown_product",
            "reference_image_paths": ["reference_images/2222222222222/main.jpg"],
            "source_image_path": "images/shelves/img_00002.jpg",
        },
    ]


def test_classifier_crop_dataset_rejects_missing_class_map_entry(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/classifier/search.json"
    processed_root = tmp_path / "processed"
    shutil.copytree(repo_root / "data/processed", processed_root)

    product_index_path = processed_root / "product_index.json"
    product_index = cast(
        list[dict[str, Any]], json.loads(product_index_path.read_text(encoding="utf-8"))
    )
    product_index_path.write_text(
        json.dumps(product_index[:1], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        build_classifier_crop_dataset(
            config_path=config_path,
            processed_root=processed_root,
            output_root=tmp_path / "artifacts/classifier",
        )
    except ClassifierDataValidationError as error:
        message = str(error)
    else:
        raise AssertionError("Expected classifier crop dataset build to fail.")

    assert "Missing classifier class-map entry for annotation 2" in message
