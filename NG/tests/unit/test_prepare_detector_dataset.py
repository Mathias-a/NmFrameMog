from __future__ import annotations

import json
from pathlib import Path

from src.ng_data.data.manifest import write_json
from src.ng_data.detector.prepare_dataset import (
    DetectorDatasetPreparationError,
    main as prepare_dataset_main,
    prepare_detector_dataset,
)


def _write_detector_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    processed_root = tmp_path / "data/processed"
    images_root = processed_root / "images"
    annotations_path = processed_root / "annotations/instances.coco.json"
    categories_path = processed_root / "categories.json"
    manifest_path = processed_root / "manifests/dataset_manifest.json"
    splits_path = processed_root / "manifests/splits.json"
    config_path = tmp_path / "configs/detector.json"

    (images_root / "shelves").mkdir(parents=True, exist_ok=True)
    (images_root / "shelves/img_00001.jpg").write_bytes(b"image-1\n")
    (images_root / "shelves/img_00002.jpg").write_bytes(b"image-2\n")
    (images_root / "shelves/img_00003.jpg").write_bytes(b"image-3\n")

    categories = [
        {"id": 0, "name": "alpha", "supercategory": "product"},
        {"id": 356, "name": "omega", "supercategory": "product"},
    ]
    write_json(categories_path, categories)
    write_json(
        annotations_path,
        {
            "annotations": [
                {
                    "area": 1200.0,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "category_id": 0,
                    "corrected": True,
                    "id": 1,
                    "image_id": 1,
                    "iscrowd": 0,
                    "product_code": "111",
                    "product_name": "alpha",
                },
                {
                    "area": 56.0,
                    "bbox": [5.0, 6.0, 7.0, 8.0],
                    "category_id": 356,
                    "corrected": False,
                    "id": 2,
                    "image_id": 2,
                    "iscrowd": 0,
                    "product_code": "222",
                    "product_name": "omega",
                },
            ],
            "categories": categories,
            "images": [
                {
                    "file_name": "shelves/img_00001.jpg",
                    "height": 480,
                    "id": 1,
                    "width": 640,
                },
                {
                    "file_name": "shelves/img_00002.jpg",
                    "height": 240,
                    "id": 2,
                    "width": 320,
                },
                {
                    "file_name": "shelves/img_00003.jpg",
                    "height": 240,
                    "id": 3,
                    "width": 320,
                },
            ],
        },
    )
    write_json(
        manifest_path,
        {
            "counts": {
                "annotation_count": 2,
                "category_count": 2,
                "image_count": 3,
                "reference_product_count": 2,
            },
            "processed_outputs": {
                "annotations": {
                    "kind": "file",
                    "path": "annotations/instances.coco.json",
                    "sha256": "unused",
                    "size_bytes": 0,
                },
                "categories": {
                    "kind": "file",
                    "path": "categories.json",
                    "sha256": "unused",
                    "size_bytes": 0,
                },
                "images": {
                    "file_count": 3,
                    "kind": "directory",
                    "path": "images",
                    "sha256": "unused",
                    "total_bytes": 0,
                },
            },
            "raw_archives": {},
            "raw_root_relative": "../raw",
            "schema_version": 1,
        },
    )
    write_json(
        splits_path,
        {
            "counts": {
                "cv_folds": 2,
                "cv_pool_images": 2,
                "holdout_images": 1,
                "total_images": 3,
            },
            "cv_pool_image_ids": [2, 3],
            "folds": [
                {
                    "fold_index": 0,
                    "train_annotation_count": 0,
                    "train_image_ids": [3],
                    "val_annotation_count": 1,
                    "val_image_ids": [2],
                },
                {
                    "fold_index": 1,
                    "train_annotation_count": 1,
                    "train_image_ids": [2],
                    "val_annotation_count": 0,
                    "val_image_ids": [3],
                },
            ],
            "holdout": {
                "annotation_count": 1,
                "image_ids": [1],
            },
            "holdout_fraction": 0.34,
            "schema_version": 1,
            "seed": 7,
            "source_annotations": "annotations/instances.coco.json",
            "source_manifest": "manifests/dataset_manifest.json",
        },
    )
    write_json(
        config_path,
        {
            "schema_version": 1,
            "baseline": "detector_only_search",
            "runtime": {
                "framework": "ultralytics",
                "version": "8.1.0",
                "export_format": "pt",
            },
            "model": {"name": "yolov8m", "weights": "yolov8m.pt"},
            "search": {
                "device": "cpu",
                "epochs": 1,
                "image_size": 640,
                "batch_size": 2,
                "patience": 0,
                "run_name": "yolov8m-search-baseline",
            },
        },
    )
    return config_path, manifest_path, splits_path, processed_root


def test_prepare_detector_dataset_is_deterministic(tmp_path: Path) -> None:
    config_path, manifest_path, splits_path, processed_root = _write_detector_fixture(
        tmp_path
    )
    output_dir = tmp_path / "artifacts/runs/detector/yolov8m-search-baseline/dataset"

    first_exit = prepare_dataset_main(
        [
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--splits",
            str(splits_path),
            "--out",
            str(output_dir),
        ]
    )
    first_files = {
        path.relative_to(output_dir).as_posix(): (
            path.readlink().as_posix()
            if path.is_symlink()
            else path.read_text(encoding="utf-8")
        )
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() or path.is_symlink()
    }

    second_exit = prepare_dataset_main(
        [
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--splits",
            str(splits_path),
            "--out",
            str(output_dir),
        ]
    )
    second_files = {
        path.relative_to(output_dir).as_posix(): (
            path.readlink().as_posix()
            if path.is_symlink()
            else path.read_text(encoding="utf-8")
        )
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() or path.is_symlink()
    }

    assert first_exit == 0
    assert second_exit == 0
    assert first_files == second_files

    dataset_yaml = (output_dir / "dataset.yaml").read_text(encoding="utf-8")
    assert dataset_yaml == "\n".join(
        [
            f'path: "{output_dir.as_posix()}"',
            f'train: "{(output_dir / "images/train").as_posix()}"',
            f'val: "{(output_dir / "images/val").as_posix()}"',
            f'test: "{(output_dir / "images/test").as_posix()}"',
            "names:",
            '  - "alpha"',
            '  - "omega"',
            "nc: 2",
            "",
        ]
    )
    assert (output_dir / "labels/test/shelves/img_00001.txt").read_text(
        encoding="utf-8"
    ) == "0 0.0390625 0.0833333333 0.046875 0.0833333333\n"
    assert (output_dir / "labels/val/shelves/img_00002.txt").read_text(
        encoding="utf-8"
    ) == "1 0.0265625 0.0416666667 0.021875 0.0333333333\n"
    assert (output_dir / "labels/train/shelves/img_00003.txt").read_text(
        encoding="utf-8"
    ) == ""
    assert (output_dir / "images/test/shelves/img_00001.jpg").readlink() == (
        processed_root / "images/shelves/img_00001.jpg"
    ).resolve()


def test_prepare_detector_dataset_rejects_split_manifest_mismatch(
    tmp_path: Path,
) -> None:
    config_path, manifest_path, splits_path, _ = _write_detector_fixture(tmp_path)
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    payload["source_manifest"] = "manifests/other.json"
    write_json(splits_path, payload)

    try:
        prepare_detector_dataset(
            config_path=config_path,
            manifest_path=manifest_path,
            splits_path=splits_path,
            out_path=tmp_path / "artifacts/out",
        )
    except DetectorDatasetPreparationError as error:
        message = str(error)
    else:
        raise AssertionError("Expected split manifest mismatch to fail.")

    assert "source_manifest does not match" in message
    assert not (tmp_path / "artifacts/out").exists()


def test_prepare_detector_dataset_rejects_missing_processed_source(
    tmp_path: Path,
) -> None:
    config_path, manifest_path, splits_path, processed_root = _write_detector_fixture(
        tmp_path
    )
    (processed_root / "annotations/instances.coco.json").unlink()

    try:
        prepare_detector_dataset(
            config_path=config_path,
            manifest_path=manifest_path,
            splits_path=splits_path,
            out_path=tmp_path / "artifacts/out",
        )
    except DetectorDatasetPreparationError as error:
        message = str(error)
    else:
        raise AssertionError("Expected missing processed annotations to fail.")

    assert "Processed output 'annotations' does not exist" in message
    assert not (tmp_path / "artifacts/out").exists()
