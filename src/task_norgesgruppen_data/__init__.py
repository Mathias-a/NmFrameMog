"""Baseline offline runner for the NorgesGruppen data task."""

from task_norgesgruppen_data.predictor import (
    CocoPrediction,
    generate_predictions,
    infer_image_id,
    write_predictions_json,
)

__all__ = [
    "CocoPrediction",
    "generate_predictions",
    "infer_image_id",
    "write_predictions_json",
]
