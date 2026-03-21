from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from astar_twin.api.store import RoundStore, SubmissionStore
from astar_twin.contracts.api_models import AnalysisResponse
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.scoring import compute_score, safe_prediction

router = APIRouter()


def get_round_store(request: Request) -> RoundStore:
    return request.app.state.round_store


def get_submission_store(request: Request) -> SubmissionStore:
    return request.app.state.submission_store


def get_mc_runner(request: Request) -> MCRunner:
    return request.app.state.mc_runner


def get_n_mc_runs(request: Request) -> int:
    return request.app.state.n_mc_runs


RoundStoreDep = Annotated[RoundStore, Depends(get_round_store)]
SubmissionStoreDep = Annotated[SubmissionStore, Depends(get_submission_store)]
MCRunnerDep = Annotated[MCRunner, Depends(get_mc_runner)]
RunsDep = Annotated[int, Depends(get_n_mc_runs)]


@router.get("/astar-island/analysis/{round_id}/{seed_index}", response_model=AnalysisResponse)
def analysis(
    round_id: str,
    seed_index: int,
    round_store: RoundStoreDep,
    submission_store: SubmissionStoreDep,
    mc_runner: MCRunnerDep,
    n_mc_runs: RunsDep,
) -> AnalysisResponse:
    fixture = round_store.get(round_id)
    if fixture is None:
        raise HTTPException(status_code=404, detail="Round not found")
    if fixture.status not in ("scoring", "completed"):
        raise HTTPException(status_code=400, detail="Round is not available for analysis")
    if seed_index < 0 or seed_index >= fixture.seeds_count:
        raise HTTPException(status_code=400, detail="seed_index must be 0-4")

    if fixture.ground_truths is not None:
        ground_truth_list = fixture.ground_truths[seed_index]
        ground_truth_tensor = np.array(ground_truth_list, dtype=np.float64)
    else:
        initial_state = fixture.initial_states[seed_index]
        runs = mc_runner.run_batch(initial_state=initial_state, n_runs=n_mc_runs)
        ground_truth_tensor = safe_prediction(
            aggregate_runs(runs, fixture.map_height, fixture.map_width)
        )
        ground_truth_list = ground_truth_tensor.tolist()

    prediction = submission_store.get(round_id, seed_index)
    score: float | None = None
    if prediction is not None:
        prediction_tensor = np.array(prediction, dtype=np.float64)
        score = compute_score(ground_truth_tensor, prediction_tensor)

    return AnalysisResponse(
        prediction=prediction,
        ground_truth=ground_truth_list,
        score=score,
        width=fixture.map_width,
        height=fixture.map_height,
        initial_grid=fixture.initial_states[0].grid,
    )
