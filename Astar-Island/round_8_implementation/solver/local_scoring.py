from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .emulator import AstarIslandEmulator


def score_prediction_locally(
    *,
    prediction_payload: object,
    fixture_paths: Sequence[Path] | None = None,
    round_id: str | None = None,
    seed_index: int | None = None,
    random_seed: int = 20260320,
    analysis_rollout_count: int = 128,
) -> dict[str, object]:
    parsed_prediction, resolved_round_id, resolved_seed_index = (
        _resolve_submission_payload(
            prediction_payload,
            round_id=round_id,
            seed_index=seed_index,
        )
    )
    emulator = AstarIslandEmulator.from_fixture_paths(
        fixture_paths,
        active_round_id=resolved_round_id,
        random_seed=random_seed,
        analysis_rollout_count=analysis_rollout_count,
    )
    effective_round_id = resolved_round_id or emulator.active_round_id
    effective_seed_index = 0 if resolved_seed_index is None else resolved_seed_index
    emulator.submit(
        {
            "round_id": effective_round_id,
            "seed_index": effective_seed_index,
            "prediction": parsed_prediction,
        }
    )
    analysis = emulator.get_analysis(effective_round_id, effective_seed_index)
    return {
        "round_id": effective_round_id,
        "seed_index": effective_seed_index,
        **analysis,
    }


def load_json_payload(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_submission_payload(
    prediction_payload: object,
    *,
    round_id: str | None,
    seed_index: int | None,
) -> tuple[list[list[list[float]]], str | None, int | None]:
    resolved_round_id = round_id
    resolved_seed_index = seed_index

    if isinstance(prediction_payload, dict):
        raw_prediction = prediction_payload.get("prediction")
        if raw_prediction is None:
            raise ValueError(
                "Prediction payload object must contain a 'prediction' field."
            )
        if resolved_round_id is None:
            raw_round_id = prediction_payload.get("round_id") or prediction_payload.get(
                "roundId"
            )
            if raw_round_id is not None:
                resolved_round_id = str(raw_round_id)
        if resolved_seed_index is None:
            raw_seed_index = prediction_payload.get(
                "seed_index"
            ) or prediction_payload.get("seedIndex")
            if isinstance(raw_seed_index, int):
                resolved_seed_index = raw_seed_index
        prediction_payload = raw_prediction

    prediction = _coerce_prediction_tensor(prediction_payload)
    return prediction, resolved_round_id, resolved_seed_index


def _coerce_prediction_tensor(payload: object) -> list[list[list[float]]]:
    if not isinstance(payload, list):
        raise ValueError(
            "Prediction payload must be a nested list or object with 'prediction'."
        )
    tensor: list[list[list[float]]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ValueError("Prediction rows must be lists.")
        parsed_row: list[list[float]] = []
        for cell in row:
            if not isinstance(cell, list):
                raise ValueError("Prediction cells must be lists.")
            parsed_cell: list[float] = []
            for probability in cell:
                if isinstance(probability, bool) or not isinstance(
                    probability, (int, float)
                ):
                    raise ValueError("Prediction probabilities must be numeric.")
                parsed_cell.append(float(probability))
            parsed_row.append(parsed_cell)
        tensor.append(parsed_row)
    return tensor
