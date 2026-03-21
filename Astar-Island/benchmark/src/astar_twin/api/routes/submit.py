from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from astar_twin.api.store import RoundStore, SubmissionStore
from astar_twin.contracts.api_models import SubmitRequest, SubmitResponse

router = APIRouter()


def get_round_store(request: Request) -> RoundStore:
    return request.app.state.round_store


def get_submission_store(request: Request) -> SubmissionStore:
    return request.app.state.submission_store


RoundStoreDep = Annotated[RoundStore, Depends(get_round_store)]
SubmissionStoreDep = Annotated[SubmissionStore, Depends(get_submission_store)]


def _validate_prediction_shape(
    prediction: list[list[list[float]]], height: int, width: int
) -> None:
    if len(prediction) != height:
        raise HTTPException(
            status_code=422, detail=f"Expected {height} rows, got {len(prediction)}"
        )
    for y, row in enumerate(prediction):
        if len(row) != width:
            raise HTTPException(
                status_code=422,
                detail=f"Row {y}: expected {width} cols, got {len(row)}",
            )
        for x, cell in enumerate(row):
            if len(cell) != 6:
                raise HTTPException(
                    status_code=422,
                    detail=f"Cell ({y},{x}): expected 6 probs, got {len(cell)}",
                )
            if any(prob < 0.0 for prob in cell):
                raise HTTPException(
                    status_code=422,
                    detail=f"Cell ({y},{x}): negative probability",
                )
            total = sum(cell)
            if abs(total - 1.0) > 0.01:
                raise HTTPException(
                    status_code=422,
                    detail=f"Cell ({y},{x}): probs sum to {total}, expected 1.0",
                )


@router.post("/astar-island/submit", response_model=SubmitResponse)
def submit(
    body: SubmitRequest,
    round_store: RoundStoreDep,
    submission_store: SubmissionStoreDep,
) -> SubmitResponse:
    fixture = round_store.get(body.round_id)
    if fixture is None:
        raise HTTPException(status_code=404, detail="Round not found")
    if fixture.status != "active":
        raise HTTPException(status_code=400, detail="Round is not active")
    if body.seed_index < 0 or body.seed_index >= fixture.seeds_count:
        raise HTTPException(status_code=400, detail="seed_index must be 0-4")

    _validate_prediction_shape(body.prediction, fixture.map_height, fixture.map_width)
    submission_store.upsert(body.round_id, body.seed_index, body.prediction)
    return SubmitResponse(status="accepted", round_id=body.round_id, seed_index=body.seed_index)
