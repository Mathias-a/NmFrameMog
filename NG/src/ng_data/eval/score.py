from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]
IOU_THRESHOLD = 0.5


class ScoreValidationError(ValueError):
    pass


@dataclass(frozen=True)
class GroundTruthBox:
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class PredictionBox:
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ScoreValidationError(f"Invalid JSON file: {path}") from error


def _require_list(data: JsonDict, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ScoreValidationError(f"Expected '{key}' to be a list.")
    return cast(list[object], value)


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ScoreValidationError(f"Expected '{key}' to be an integer.")
    return value


def _require_score(data: JsonDict, key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScoreValidationError(f"Expected '{key}' to be numeric.")
    score = float(value)
    if score < 0 or score > 1:
        raise ScoreValidationError(f"Expected '{key}' to be between 0 and 1.")
    return score


def _require_bbox(data: JsonDict, key: str) -> tuple[float, float, float, float]:
    value = data.get(key)
    if not isinstance(value, list) or len(value) != 4:
        raise ScoreValidationError(f"Expected '{key}' to be a four-value list.")
    coordinates: list[float] = []
    for index, coordinate in enumerate(value):
        if not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool):
            raise ScoreValidationError(f"Expected '{key}[{index}]' to be numeric.")
        numeric_coordinate = float(coordinate)
        if numeric_coordinate < 0:
            raise ScoreValidationError(f"Expected '{key}[{index}]' to be non-negative.")
        coordinates.append(numeric_coordinate)
    if coordinates[2] <= 0 or coordinates[3] <= 0:
        raise ScoreValidationError(
            f"Expected '{key}' width and height to be greater than zero."
        )
    return cast(tuple[float, float, float, float], tuple(coordinates))


def load_ground_truth_boxes(ground_truth_path: str | Path) -> list[GroundTruthBox]:
    payload = _load_json(Path(ground_truth_path))
    if not isinstance(payload, dict):
        raise ScoreValidationError("Ground truth root must be a JSON object.")
    data = cast(JsonDict, payload)
    annotations = _require_list(data, "annotations")

    boxes: list[GroundTruthBox] = []
    for item in annotations:
        if not isinstance(item, dict):
            raise ScoreValidationError("Every annotation entry must be an object.")
        annotation = cast(JsonDict, item)
        boxes.append(
            GroundTruthBox(
                image_id=_require_int(annotation, "image_id"),
                category_id=_require_int(annotation, "category_id"),
                bbox=_require_bbox(annotation, "bbox"),
            )
        )
    return boxes


def validate_predictions(
    predictions: object, *, valid_image_ids: set[int] | None = None
) -> list[PredictionBox]:
    if not isinstance(predictions, list):
        raise ScoreValidationError("Predictions root must be a JSON array.")

    parsed: list[PredictionBox] = []
    for index, item in enumerate(predictions):
        if not isinstance(item, dict):
            raise ScoreValidationError(
                f"Prediction at index {index} must be a JSON object."
            )
        prediction = cast(JsonDict, item)
        image_id = _require_int(prediction, "image_id")
        if valid_image_ids is not None and image_id not in valid_image_ids:
            raise ScoreValidationError(f"Unknown prediction image_id: {image_id}")
        parsed.append(
            PredictionBox(
                image_id=image_id,
                category_id=_require_int(prediction, "category_id"),
                bbox=_require_bbox(prediction, "bbox"),
                score=_require_score(prediction, "score"),
            )
        )
    return parsed


def load_predictions(predictions_path: str | Path) -> list[PredictionBox]:
    return validate_predictions(_load_json(Path(predictions_path)))


def _to_xyxy(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, width, height = bbox
    return x, y, x + width, y + height


def compute_iou(
    first_bbox: tuple[float, float, float, float],
    second_bbox: tuple[float, float, float, float],
) -> float:
    first_x1, first_y1, first_x2, first_y2 = _to_xyxy(first_bbox)
    second_x1, second_y1, second_x2, second_y2 = _to_xyxy(second_bbox)

    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0

    first_area = (first_x2 - first_x1) * (first_y2 - first_y1)
    second_area = (second_x2 - second_x1) * (second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def _compute_average_precision(
    predictions: list[PredictionBox],
    ground_truth: list[GroundTruthBox],
    *,
    match_category: bool,
) -> float:
    if not ground_truth:
        return 0.0

    matched_ground_truth: set[int] = set()
    sorted_predictions = sorted(
        predictions,
        key=lambda prediction: (
            -prediction.score,
            prediction.image_id,
            prediction.category_id,
            prediction.bbox,
        ),
    )

    true_positives: list[float] = []
    false_positives: list[float] = []
    for prediction in sorted_predictions:
        best_iou = 0.0
        best_index: int | None = None
        for index, target in enumerate(ground_truth):
            if index in matched_ground_truth:
                continue
            if prediction.image_id != target.image_id:
                continue
            if match_category and prediction.category_id != target.category_id:
                continue
            iou = compute_iou(prediction.bbox, target.bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index is not None and best_iou >= IOU_THRESHOLD:
            matched_ground_truth.add(best_index)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    cumulative_true_positives: list[float] = []
    cumulative_false_positives: list[float] = []
    tp_total = 0.0
    fp_total = 0.0
    for index in range(len(true_positives)):
        tp_total += true_positives[index]
        fp_total += false_positives[index]
        cumulative_true_positives.append(tp_total)
        cumulative_false_positives.append(fp_total)

    recalls = [value / len(ground_truth) for value in cumulative_true_positives]
    precisions = [
        cumulative_true_positives[index]
        / (cumulative_true_positives[index] + cumulative_false_positives[index])
        for index in range(len(cumulative_true_positives))
    ]

    monotonic_precisions = [0.0, *precisions, 0.0]
    monotonic_recalls = [0.0, *recalls, 1.0]
    for index in range(len(monotonic_precisions) - 2, -1, -1):
        monotonic_precisions[index] = max(
            monotonic_precisions[index], monotonic_precisions[index + 1]
        )

    average_precision = 0.0
    for index in range(1, len(monotonic_recalls)):
        recall_delta = monotonic_recalls[index] - monotonic_recalls[index - 1]
        if recall_delta > 0:
            average_precision += recall_delta * monotonic_precisions[index]
    return average_precision


def score_predictions(
    ground_truth_path: str | Path, predictions: object
) -> dict[str, float]:
    ground_truth = load_ground_truth_boxes(ground_truth_path)
    valid_image_ids = {box.image_id for box in ground_truth}
    parsed_predictions = validate_predictions(
        predictions, valid_image_ids=valid_image_ids
    )

    detection_map = _compute_average_precision(
        parsed_predictions, ground_truth, match_category=False
    )

    category_ids = sorted({box.category_id for box in ground_truth})
    classification_scores: list[float] = []
    for category_id in category_ids:
        category_ground_truth = [
            box for box in ground_truth if box.category_id == category_id
        ]
        category_predictions = [
            box for box in parsed_predictions if box.category_id == category_id
        ]
        classification_scores.append(
            _compute_average_precision(
                category_predictions,
                category_ground_truth,
                match_category=True,
            )
        )
    classification_map = sum(classification_scores) / len(classification_scores)
    hybrid_score = 0.7 * detection_map + 0.3 * classification_map
    return {
        "classification_map": classification_map,
        "detection_map": detection_map,
        "hybrid_score": hybrid_score,
    }
