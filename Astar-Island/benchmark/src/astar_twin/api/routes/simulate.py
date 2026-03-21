from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from numpy.random import default_rng

from astar_twin.api.store import BudgetStore, RoundStore
from astar_twin.contracts.api_models import (
    SimSettlement,
    SimulateRequest,
    SimulateResponse,
    ViewportBounds,
)
from astar_twin.contracts.types import MAX_QUERIES
from astar_twin.engine import Simulator

router = APIRouter()
_SIM_SEED_RNG = default_rng(secrets.randbits(64))


def get_round_store(request: Request) -> RoundStore:
    return request.app.state.round_store


def get_budget_store(request: Request) -> BudgetStore:
    return request.app.state.budget_store


def get_simulator(request: Request) -> Simulator:
    return request.app.state.simulator


RoundStoreDep = Annotated[RoundStore, Depends(get_round_store)]
BudgetStoreDep = Annotated[BudgetStore, Depends(get_budget_store)]
SimulatorDep = Annotated[Simulator, Depends(get_simulator)]


@router.post("/astar-island/simulate", response_model=SimulateResponse)
def simulate(
    body: SimulateRequest,
    round_store: RoundStoreDep,
    budget_store: BudgetStoreDep,
    simulator: SimulatorDep,
) -> SimulateResponse:
    fixture = round_store.get(body.round_id)
    if fixture is None:
        raise HTTPException(status_code=404, detail="Round not found")
    if fixture.status != "active":
        raise HTTPException(status_code=400, detail="Round is not active")
    if body.seed_index < 0 or body.seed_index >= fixture.seeds_count:
        raise HTTPException(status_code=400, detail="seed_index must be 0-4")
    if budget_store.used(fixture.id) >= MAX_QUERIES:
        raise HTTPException(status_code=429, detail="Query budget exhausted")

    initial_state = fixture.initial_states[body.seed_index]
    sim_seed = int(_SIM_SEED_RNG.integers(0, 2**32, endpoint=False))
    final_state = simulator.run(initial_state=initial_state, sim_seed=sim_seed)

    x0 = max(0, body.viewport_x)
    y0 = max(0, body.viewport_y)
    x1 = min(fixture.map_width, body.viewport_x + body.viewport_w)
    y1 = min(fixture.map_height, body.viewport_y + body.viewport_h)
    actual_w = max(0, x1 - x0)
    actual_h = max(0, y1 - y0)
    viewport = final_state.grid.viewport(
        body.viewport_x, body.viewport_y, body.viewport_w, body.viewport_h
    )

    settlements = [
        SimSettlement(
            x=settlement.x,
            y=settlement.y,
            population=settlement.population,
            food=settlement.food,
            wealth=settlement.wealth,
            defense=settlement.defense,
            has_port=settlement.has_port,
            alive=settlement.alive,
            owner_id=settlement.owner_id,
        )
        for settlement in final_state.settlements
        if x0 <= settlement.x < x1 and y0 <= settlement.y < y1
    ]

    budget_store.increment(fixture.id)
    return SimulateResponse(
        grid=viewport.to_list(),
        settlements=settlements,
        viewport=ViewportBounds(x=x0, y=y0, w=actual_w, h=actual_h),
        width=fixture.map_width,
        height=fixture.map_height,
        queries_used=budget_store.used(fixture.id),
        queries_max=budget_store.max_queries,
    )
