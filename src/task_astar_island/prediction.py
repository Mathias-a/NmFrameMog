from __future__ import annotations

import math

CLASS_COUNT = 6
EPSILON = 1e-6


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _find_dimensions(mapping: dict[str, object]) -> tuple[int, int] | None:
    key_pairs = (
        ("width", "height"),
        ("gridWidth", "gridHeight"),
        ("columns", "rows"),
        ("x", "y"),
    )
    for width_key, height_key in key_pairs:
        width = _coerce_positive_int(mapping.get(width_key))
        height = _coerce_positive_int(mapping.get(height_key))
        if width is not None and height is not None:
            return width, height

    for value in mapping.values():
        if isinstance(value, dict):
            nested = _find_dimensions(value)
            if nested is not None:
                return nested
    return None


def infer_grid_dimensions(payload: dict[str, object]) -> tuple[int, int]:
    dimensions = _find_dimensions(payload)
    if dimensions is None:
        raise ValueError(
            "Could not infer grid dimensions from the supplied round payload."
        )
    return dimensions


def extract_budget_hint(payload: object) -> float | None:
    if isinstance(payload, bool) or payload is None:
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, str):
        try:
            return float(payload)
        except ValueError:
            return None
    if isinstance(payload, list):
        for item in payload:
            budget = extract_budget_hint(item)
            if budget is not None:
                return budget
        return None

    if not isinstance(payload, dict):
        return None

    for key in ("budget", "remainingBudget", "value", "remaining"):
        if key in payload:
            budget = extract_budget_hint(payload[key])
            if budget is not None:
                return budget

    for value in payload.values():
        budget = extract_budget_hint(value)
        if budget is not None:
            return budget
    return None


def _normalized_probabilities(weights: list[float]) -> list[float]:
    positive_weights = [max(weight, EPSILON) for weight in weights]
    total = sum(positive_weights)
    normalized = [weight / total for weight in positive_weights]
    normalized[-1] += 1.0 - sum(normalized)
    return normalized


def _cell_weights(
    x: int, y: int, width: int, height: int, budget: float | None
) -> list[float]:
    budget_factor = 0.0 if budget is None else (budget % 19.0) / 100.0
    horizontal = (x + 1) / (width + 1)
    vertical = (y + 1) / (height + 1)
    mirrored_horizontal = (width - x) / (width + 1)
    mirrored_vertical = (height - y) / (height + 1)
    parity = 1.0 if (x + y) % 2 == 0 else 1.4
    diagonal = 1.0 + ((x + 1) * (y + 1)) / ((width + 1) * (height + 1))
    return [
        1.0 + horizontal + budget_factor,
        1.0 + vertical,
        parity,
        diagonal,
        1.0 + mirrored_horizontal,
        1.0 + mirrored_vertical,
    ]


def build_probability_grid(
    width: int, height: int, budget: float | None = None
) -> list[list[list[float]]]:
    if width <= 0 or height <= 0:
        raise ValueError("Grid dimensions must be positive integers.")

    grid: list[list[list[float]]] = []
    for y in range(height):
        row: list[list[float]] = []
        for x in range(width):
            row.append(
                _normalized_probabilities(_cell_weights(x, y, width, height, budget))
            )
        grid.append(row)

    validate_probability_grid(grid)
    return grid


def validate_probability_grid(
    grid: list[list[list[float]]], class_count: int = CLASS_COUNT
) -> None:
    if not grid or not grid[0]:
        raise ValueError("Prediction grid must contain at least one cell.")

    row_width = len(grid[0])
    for row in grid:
        if len(row) != row_width:
            raise ValueError("Prediction grid rows must all have the same width.")
        for cell in row:
            if len(cell) != class_count:
                raise ValueError(
                    f"Each cell must contain exactly {class_count} class probabilities."
                )
            if any(probability <= 0.0 for probability in cell):
                raise ValueError(
                    "Every class probability must be strictly greater than zero."
                )
            if not math.isclose(sum(cell), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Each cell must sum to exactly one.")


def build_submission_body(
    prediction: list[list[list[float]]],
    *,
    round_id: str | int,
    seed_index: int,
) -> dict[str, object]:
    validate_probability_grid(prediction)
    prediction_json: list[object] = []
    for row in prediction:
        row_json: list[object] = []
        for cell in row:
            cell_json: list[object] = [float(probability) for probability in cell]
            row_json.append(cell_json)
        prediction_json.append(row_json)

    return {
        "round_id": str(round_id),
        "seed_index": seed_index,
        "prediction": prediction_json,
    }
