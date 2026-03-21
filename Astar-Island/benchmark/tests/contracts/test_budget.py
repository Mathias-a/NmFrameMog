from __future__ import annotations

from fastapi.testclient import TestClient


def test_budget_endpoint_returns_active_round_budget(client: TestClient) -> None:
    response = client.get("/astar-island/budget")
    assert response.status_code == 200
    assert response.json() == {
        "round_id": "test-round-001",
        "queries_used": 0,
        "queries_max": 50,
        "active": True,
    }


def test_budget_increments_after_simulate(client: TestClient) -> None:
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
    assert response.status_code == 200

    budget_response = client.get("/astar-island/budget")
    assert budget_response.status_code == 200
    assert budget_response.json()["queries_used"] == 1
