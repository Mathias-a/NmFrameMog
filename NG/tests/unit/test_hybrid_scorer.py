from __future__ import annotations

from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from src.ng_data.data.manifest import write_json
from src.ng_data.eval.score import score_predictions


def _write_ground_truth(tmp_path: Path) -> Path:
    ground_truth_path = tmp_path / "instances.coco.json"
    write_json(
        ground_truth_path,
        {
            "annotations": [
                {
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "category_id": 1,
                    "id": 1,
                    "image_id": 1,
                },
                {
                    "bbox": [20.0, 20.0, 10.0, 10.0],
                    "category_id": 2,
                    "id": 2,
                    "image_id": 2,
                },
            ],
            "categories": [
                {"id": 1, "name": "one", "supercategory": "product"},
                {"id": 2, "name": "two", "supercategory": "product"},
            ],
            "images": [
                {"file_name": "img_00001.jpg", "height": 10, "id": 1, "width": 10},
                {"file_name": "img_00002.jpg", "height": 10, "id": 2, "width": 10},
            ],
        },
    )
    return ground_truth_path


def test_hybrid_scorer_counts_category_mismatch_for_detection_only(
    tmp_path: Path,
) -> None:
    ground_truth_path = _write_ground_truth(tmp_path)
    summary = score_predictions(
        ground_truth_path,
        [
            {
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "category_id": 99,
                "image_id": 1,
                "score": 0.95,
            },
            {
                "bbox": [20.0, 20.0, 10.0, 10.0],
                "category_id": 2,
                "image_id": 2,
                "score": 0.9,
            },
        ],
    )

    assert summary == pytest.approx(
        {
            "classification_map": 0.5,
            "detection_map": 1.0,
            "hybrid_score": 0.85,
        }
    )


def test_hybrid_scorer_returns_perfect_score_for_perfect_predictions(
    tmp_path: Path,
) -> None:
    ground_truth_path = _write_ground_truth(tmp_path)
    summary = score_predictions(
        ground_truth_path,
        [
            {
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "category_id": 1,
                "image_id": 1,
                "score": 0.95,
            },
            {
                "bbox": [20.0, 20.0, 10.0, 10.0],
                "category_id": 2,
                "image_id": 2,
                "score": 0.9,
            },
        ],
    )

    assert summary == {
        "classification_map": 1.0,
        "detection_map": 1.0,
        "hybrid_score": 1.0,
    }
