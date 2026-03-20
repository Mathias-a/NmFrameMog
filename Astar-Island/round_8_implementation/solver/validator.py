from __future__ import annotations

import math

from .contract import (
    CLASS_COUNT,
    PROBABILITY_FLOOR,
    TERRAIN_CODE_TO_CLASS,
    canonical_mapping_artifact,
)


def validate_mapping() -> None:
    expected_codes = {0, 1, 2, 3, 4, 5, 10, 11}
    actual_codes = set(TERRAIN_CODE_TO_CLASS)
    if actual_codes != expected_codes:
        raise ValueError(
            f"Terrain mapping coverage mismatch: expected {sorted(expected_codes)}, got {sorted(actual_codes)}"
        )
    artifact = canonical_mapping_artifact()
    internal_code_count = artifact.get("internal_code_count")
    if not isinstance(internal_code_count, int):
        raise ValueError("Mapping artifact is missing an integer internal code count.")
    if internal_code_count != len(expected_codes):
        raise ValueError("Mapping artifact reports the wrong internal code count.")


def validate_grid(
    grid: list[list[int]], *, width: int | None = None, height: int | None = None
) -> None:
    if not grid or not grid[0]:
        raise ValueError("Grid must be non-empty.")
    row_width = len(grid[0])
    for row in grid:
        if len(row) != row_width:
            raise ValueError("Grid must be rectangular.")
        for terrain_code in row:
            if terrain_code not in TERRAIN_CODE_TO_CLASS:
                raise ValueError(f"Unsupported terrain code in grid: {terrain_code}")
    if width is not None and row_width != width:
        raise ValueError(f"Grid width mismatch: expected {width}, got {row_width}")
    if height is not None and len(grid) != height:
        raise ValueError(f"Grid height mismatch: expected {height}, got {len(grid)}")


def validate_prediction_tensor(
    tensor: list[list[list[float]]],
    *,
    width: int,
    height: int,
    probability_floor: float = PROBABILITY_FLOOR,
) -> None:
    if len(tensor) != height:
        raise ValueError(
            f"Tensor height mismatch: expected {height}, got {len(tensor)}"
        )

    for row_index, row in enumerate(tensor):
        if len(row) != width:
            raise ValueError(
                f"Tensor width mismatch on row {row_index}: expected {width}, got {len(row)}"
            )
        for column_index, cell in enumerate(row):
            if len(cell) != CLASS_COUNT:
                raise ValueError(
                    f"Tensor class count mismatch at ({column_index}, {row_index}): expected {CLASS_COUNT}, got {len(cell)}"
                )
            if any(probability < probability_floor for probability in cell):
                raise ValueError(
                    f"Tensor floor violation at ({column_index}, {row_index}) with floor {probability_floor}"
                )
            total = math.fsum(cell)
            if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"Tensor normalization error at ({column_index}, {row_index}): got {total}"
                )


def entropy_weighted_kl_score(
    prediction: list[list[list[float]]],
    ground_truth: list[list[list[float]]],
) -> float:
    if len(prediction) != len(ground_truth):
        raise ValueError("Prediction and ground truth height differ.")

    weighted_divergence = 0.0
    entropy_weight_sum = 0.0
    for row_index, row in enumerate(ground_truth):
        if len(prediction[row_index]) != len(row):
            raise ValueError("Prediction and ground truth width differ.")
        for column_index, target_cell in enumerate(row):
            predicted_cell = prediction[row_index][column_index]
            if len(predicted_cell) != len(target_cell):
                raise ValueError("Prediction and ground truth class counts differ.")
            entropy = 0.0
            divergence = 0.0
            if len(predicted_cell) != len(target_cell):
                raise ValueError("Prediction and ground truth class counts differ.")
            for index, target_probability in enumerate(target_cell):
                predicted_probability = predicted_cell[index]
                if target_probability <= 0.0:
                    continue
                entropy -= target_probability * math.log(target_probability)
                divergence += target_probability * math.log(
                    target_probability / predicted_probability
                )
            if entropy <= 0.0:
                continue
            weighted_divergence += entropy * divergence
            entropy_weight_sum += entropy
    if entropy_weight_sum == 0.0:
        return 100.0
    weighted_kl = weighted_divergence / entropy_weight_sum
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * weighted_kl)))
