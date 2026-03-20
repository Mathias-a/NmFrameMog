from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from PIL import Image

IMAGE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


class CocoPredictionDict(TypedDict):
    image_id: int
    category_id: int
    bbox: list[int]
    score: float


@dataclass(frozen=True)
class CocoPrediction:
    image_id: int
    category_id: int
    bbox: list[int]
    score: float

    def to_dict(self) -> CocoPredictionDict:
        return {
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": self.bbox,
            "score": self.score,
        }


def list_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def infer_image_id(image_path: Path, fallback_id: int) -> int:
    matches = re.findall(r"\d+", image_path.stem)
    if matches:
        return int(matches[-1])
    return fallback_id


def _build_center_box(width: int, height: int) -> list[int]:
    box_width = max(1, width // 2)
    box_height = max(1, height // 2)
    x = max(0, (width - box_width) // 2)
    y = max(0, (height - box_height) // 2)
    return [x, y, box_width, box_height]


def generate_predictions(input_dir: Path) -> list[CocoPrediction]:
    predictions: list[CocoPrediction] = []
    for fallback_id, image_path in enumerate(list_image_files(input_dir), start=1):
        image = Image.open(image_path)
        try:
            width, height = image.size
        finally:
            image.close()

        predictions.append(
            CocoPrediction(
                image_id=infer_image_id(image_path, fallback_id),
                category_id=1,
                bbox=_build_center_box(width, height),
                score=0.5,
            )
        )

    return predictions


def write_predictions_json(
    predictions: list[CocoPrediction], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [prediction.to_dict() for prediction in predictions]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
