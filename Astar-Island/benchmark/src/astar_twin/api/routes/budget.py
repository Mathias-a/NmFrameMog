from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from astar_twin.api.store import BudgetStore, RoundStore
from astar_twin.contracts.api_models import BudgetResponse

router = APIRouter()


def get_round_store(request: Request) -> RoundStore:
    return request.app.state.round_store


def get_budget_store(request: Request) -> BudgetStore:
    return request.app.state.budget_store


RoundStoreDep = Annotated[RoundStore, Depends(get_round_store)]
BudgetStoreDep = Annotated[BudgetStore, Depends(get_budget_store)]


@router.get("/astar-island/budget", response_model=BudgetResponse)
def get_budget(round_store: RoundStoreDep, budget_store: BudgetStoreDep) -> BudgetResponse:
    fixture = round_store.get_active()
    if fixture is None:
        raise HTTPException(status_code=404, detail="No active round")
    return BudgetResponse(
        round_id=fixture.id,
        queries_used=budget_store.used(fixture.id),
        queries_max=budget_store.max_queries,
        active=True,
    )
