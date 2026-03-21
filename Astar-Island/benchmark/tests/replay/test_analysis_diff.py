from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from astar_twin.api.app import create_app
from astar_twin.api.store import BudgetStore, RoundStore, SubmissionStore
from astar_twin.data.models import RoundFixture
from astar_twin.scoring import safe_prediction


def test_analysis_returns_score_between_zero_and_hundred(fixture: RoundFixture) -> None:
    fixture = fixture.model_copy(
        update={
            "ground_truths": [
                [
                    [[1.0 / 6.0] * 6 for _ in range(fixture.map_width)]
                    for _ in range(fixture.map_height)
                ]
                for _ in range(fixture.seeds_count)
            ]
        }
    )
    round_store = RoundStore()
    round_store.add(fixture)
    submission_store = SubmissionStore()
    prediction = safe_prediction(np.full((10, 10, 6), 1.0 / 6.0, dtype=np.float64)).tolist()
    submission_store.upsert(fixture.id, 0, prediction)
    app = create_app(
        round_store=round_store,
        submission_store=submission_store,
        budget_store=BudgetStore(),
        n_mc_runs=5,
    )
    client = TestClient(app)

    response = client.get(f"/astar-island/analysis/{fixture.id}/0")
    assert response.status_code == 200
    score = response.json()["score"]
    assert score is not None
    assert 0.0 <= score <= 100.0
