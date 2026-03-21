from __future__ import annotations

from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from src.ng_data.data.manifest import write_json
from src.ng_data.eval.score import ScoreValidationError, score_predictions


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
                }
            ],
            "categories": [{"id": 1, "name": "one", "supercategory": "product"}],
            "images": [
                {"file_name": "img_00001.jpg", "height": 10, "id": 1, "width": 10}
            ],
        },
    )
    return ground_truth_path


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (
            [{"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 10.0], "score": 0.5}],
            "four-value list",
        ),
        (
            [{"image_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.5}],
            "category_id",
        ),
        (
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "score": 1.1,
                }
            ],
            "between 0 and 1",
        ),
        (
            [
                {
                    "image_id": 99,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "score": 0.5,
                }
            ],
            "Unknown prediction image_id",
        ),
    ],
)
def test_hybrid_scorer_rejects_invalid_predictions(
    tmp_path: Path, predictions: list[dict[str, object]], message: str
) -> None:
    ground_truth_path = _write_ground_truth(tmp_path)

    with pytest.raises(ScoreValidationError, match=message):
        score_predictions(ground_truth_path, predictions)
