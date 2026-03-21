from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from astar_twin.api.store import RoundStore
from astar_twin.contracts.api_models import RoundDetail, RoundSummary

router = APIRouter()


def get_round_store(request: Request) -> RoundStore:
    return request.app.state.round_store


RoundStoreDep = Annotated[RoundStore, Depends(get_round_store)]


def _to_round_summary(fixture: object) -> RoundSummary:
    from astar_twin.data.models import RoundFixture

    round_fixture = (
        fixture if isinstance(fixture, RoundFixture) else RoundFixture.model_validate(fixture)
    )
    return RoundSummary(
        id=round_fixture.id,
        round_number=round_fixture.round_number,
        event_date=round_fixture.event_date,
        status=round_fixture.status,
        map_width=round_fixture.map_width,
        map_height=round_fixture.map_height,
        prediction_window_minutes=165,
        started_at=None,
        closes_at=None,
        round_weight=round_fixture.round_weight,
        created_at="",
    )


def _to_round_detail(fixture: object) -> RoundDetail:
    from astar_twin.data.models import RoundFixture

    round_fixture = (
        fixture if isinstance(fixture, RoundFixture) else RoundFixture.model_validate(fixture)
    )
    return RoundDetail(
        id=round_fixture.id,
        round_number=round_fixture.round_number,
        status=round_fixture.status,
        map_width=round_fixture.map_width,
        map_height=round_fixture.map_height,
        seeds_count=round_fixture.seeds_count,
        initial_states=round_fixture.initial_states,
    )


@router.get("/astar-island/rounds", response_model=list[RoundSummary])
def list_rounds(round_store: RoundStoreDep) -> list[RoundSummary]:
    fixtures = sorted(round_store.list(), key=lambda fixture: fixture.round_number)
    return [_to_round_summary(fixture) for fixture in fixtures]


@router.get("/astar-island/rounds/{round_id}", response_model=RoundDetail)
def get_round(round_id: str, round_store: RoundStoreDep) -> RoundDetail:
    fixture = round_store.get(round_id)
    if fixture is None:
        raise HTTPException(status_code=404, detail="Round not found")
    return _to_round_detail(fixture)
