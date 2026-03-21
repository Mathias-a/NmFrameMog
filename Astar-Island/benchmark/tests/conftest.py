from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from astar_twin.api.app import create_app
from astar_twin.api.store import BudgetStore, RoundStore, SubmissionStore
from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import RoundFixture
from astar_twin.scoring import safe_prediction

FIXTURE_PATH = (
    Path(__file__).parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def fixture() -> RoundFixture:
    return load_fixture(FIXTURE_PATH)


@pytest.fixture
def fixture_with_ground_truths(fixture: RoundFixture) -> RoundFixture:
    height = fixture.map_height
    width = fixture.map_width
    empty_prediction = safe_prediction(np.full((height, width, 6), 1.0 / 6.0, dtype=np.float64))
    ground_truth = empty_prediction.tolist()
    return fixture.model_copy(
        update={"ground_truths": [ground_truth for _ in range(fixture.seeds_count)]}
    )


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
def completed_client(fixture_with_ground_truths: RoundFixture) -> TestClient:
    round_store = RoundStore()
    round_store.add(fixture_with_ground_truths)
    app = create_app(
        round_store=round_store,
        submission_store=SubmissionStore(),
        budget_store=BudgetStore(),
        n_mc_runs=5,
    )
    return TestClient(app)
