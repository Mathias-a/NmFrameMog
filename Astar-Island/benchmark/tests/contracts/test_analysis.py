from __future__ import annotations

from fastapi.testclient import TestClient

from astar_twin.data.models import RoundFixture


def test_analysis_400_when_round_not_completed(client: TestClient) -> None:
    response = client.get("/astar-island/analysis/test-round-001/0")
    assert response.status_code == 400


def test_analysis_404_when_round_not_found(completed_client: TestClient) -> None:
    response = completed_client.get("/astar-island/analysis/missing-round/0")
    assert response.status_code == 404


def test_analysis_400_when_seed_index_out_of_range(completed_client: TestClient) -> None:
    response = completed_client.get("/astar-island/analysis/test-round-001/5")
    assert response.status_code == 400


def test_analysis_successful_response_shape(completed_client: TestClient) -> None:
    response = completed_client.get("/astar-island/analysis/test-round-001/0")
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] is None
    assert len(payload["ground_truth"]) == 10
    assert len(payload["ground_truth"][0]) == 10
    assert len(payload["ground_truth"][0][0]) == 6
    assert payload["width"] == 10
    assert payload["height"] == 10
    assert len(payload["initial_grid"]) == 10


def test_analysis_409_when_ground_truth_missing(active_fixture: RoundFixture) -> None:
    unavailable_fixture = active_fixture.model_copy(update={"status": "completed"})
    from astar_twin.api.app import create_app
    from astar_twin.api.store import BudgetStore, RoundStore, SubmissionStore

    round_store = RoundStore()
    round_store.add(unavailable_fixture)
    client = TestClient(
        create_app(
            round_store=round_store,
            submission_store=SubmissionStore(),
            budget_store=BudgetStore(),
            n_mc_runs=5,
        )
    )

    response = client.get(f"/astar-island/analysis/{unavailable_fixture.id}/0")
    assert response.status_code == 409
    assert response.json() == {
        "detail": "Analysis unavailable: ground truth not available for this round"
    }


def test_analysis_uses_requested_seed_initial_grid(
    fixture_with_ground_truths: RoundFixture,
) -> None:
    custom_grid = [row[:] for row in fixture_with_ground_truths.initial_states[1].grid]
    custom_grid[0][0] = 99
    custom_initial_states = list(fixture_with_ground_truths.initial_states)
    custom_initial_states[1] = custom_initial_states[1].model_copy(update={"grid": custom_grid})
    custom_fixture = fixture_with_ground_truths.model_copy(
        update={"initial_states": custom_initial_states}
    )

    from astar_twin.api.app import create_app
    from astar_twin.api.store import BudgetStore, RoundStore, SubmissionStore

    round_store = RoundStore()
    round_store.add(custom_fixture)
    client = TestClient(
        create_app(
            round_store=round_store,
            submission_store=SubmissionStore(),
            budget_store=BudgetStore(),
            n_mc_runs=5,
        )
    )

    response = client.get(f"/astar-island/analysis/{custom_fixture.id}/1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["initial_grid"] == custom_grid
    assert payload["initial_grid"] != custom_fixture.initial_states[0].grid
