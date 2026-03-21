from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from astar_twin.api.routes.analysis import router as analysis_router
from astar_twin.api.routes.budget import router as budget_router
from astar_twin.api.routes.rounds import router as rounds_router
from astar_twin.api.routes.simulate import router as simulate_router
from astar_twin.api.routes.submit import router as submit_router
from astar_twin.api.store import BudgetStore, RoundStore, SubmissionStore
from astar_twin.data import list_fixtures
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner


def create_app(
    round_store: RoundStore | None = None,
    submission_store: SubmissionStore | None = None,
    budget_store: BudgetStore | None = None,
    simulator: Simulator | None = None,
    mc_runner: MCRunner | None = None,
    n_mc_runs: int = 200,
    data_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.round_store = round_store or RoundStore()
    app.state.submission_store = submission_store or SubmissionStore()
    app.state.budget_store = budget_store or BudgetStore()
    app.state.simulator = simulator
    app.state.mc_runner = mc_runner
    app.state.n_mc_runs = n_mc_runs

    if data_dir is not None:
        for fixture in list_fixtures(data_dir):
            app.state.round_store.add(fixture)

    app.include_router(rounds_router)
    app.include_router(budget_router)
    app.include_router(simulate_router)
    app.include_router(submit_router)
    app.include_router(analysis_router)
    return app


app = create_app()
