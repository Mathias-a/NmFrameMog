from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from astar_twin.api.app import create_app
from astar_twin.api.store import BudgetStore, RoundStore, SubmissionStore
from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import RoundFixture

FIXTURE_PATH = (
    Path(__file__).parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def fixture() -> RoundFixture:
    return load_fixture(FIXTURE_PATH)


@pytest.fixture
def active_fixture(fixture: RoundFixture) -> RoundFixture:
    return fixture.model_copy(update={"status": "active"})


@pytest.fixture
def client(active_fixture: RoundFixture) -> TestClient:
    round_store = RoundStore()
    round_store.add(active_fixture)
    app = create_app(
        round_store=round_store,
        submission_store=SubmissionStore(),
        budget_store=BudgetStore(),
        n_mc_runs=5,
    )
    return TestClient(app)


@pytest.fixture
def completed_client(fixture: RoundFixture) -> TestClient:
    round_store = RoundStore()
    round_store.add(fixture)
    app = create_app(
        round_store=round_store,
        submission_store=SubmissionStore(),
        budget_store=BudgetStore(),
        n_mc_runs=5,
    )
    return TestClient(app)
