from __future__ import annotations

from fastapi.testclient import TestClient

from astar_twin.api.app import create_app
from astar_twin.api.store import BudgetStore, RoundStore, SubmissionStore
from astar_twin.data.models import RoundFixture


def test_simulate_success_returns_correct_viewport_shape(client: TestClient) -> None:
    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "viewport_x": 1,
            "viewport_y": 2,
            "viewport_w": 5,
            "viewport_h": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["grid"]) == 5
    assert all(len(row) == 5 for row in payload["grid"])


def test_simulate_settlements_stay_within_viewport_bounds(client: TestClient) -> None:
    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 10,
            "viewport_h": 10,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    viewport = payload["viewport"]
    for settlement in payload["settlements"]:
        assert viewport["x"] <= settlement["x"] < viewport["x"] + viewport["w"]
        assert viewport["y"] <= settlement["y"] < viewport["y"] + viewport["h"]


def test_simulate_clamps_viewport_to_map_bounds(client: TestClient) -> None:
    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "viewport_x": 9,
            "viewport_y": 9,
            "viewport_w": 15,
            "viewport_h": 15,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["viewport"] == {"x": 9, "y": 9, "w": 1, "h": 1}
    assert len(payload["grid"]) == 1
    assert len(payload["grid"][0]) == 1


def test_simulate_404_for_missing_round(client: TestClient) -> None:
    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": "missing-round",
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 5,
            "viewport_h": 5,
        },
    )
    assert response.status_code == 404


def test_simulate_400_for_non_active_round(completed_client: TestClient) -> None:
    response = completed_client.post(
        "/astar-island/simulate",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 5,
            "viewport_h": 5,
        },
    )
    assert response.status_code == 400


def test_simulate_429_when_budget_exhausted(active_fixture: RoundFixture) -> None:
    round_store = RoundStore()
    round_store.add(active_fixture)
    budget_store = BudgetStore()
    for _ in range(50):
        budget_store.increment(active_fixture.id)
    app = create_app(
        round_store=round_store,
        submission_store=SubmissionStore(),
        budget_store=budget_store,
        n_mc_runs=5,
    )
    client = TestClient(app)

    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 5,
            "viewport_h": 5,
        },
    )
    assert response.status_code == 429
