from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_rounds_returns_round_summary(client: TestClient) -> None:
    response = client.get("/astar-island/rounds")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload[0]["id"] == "test-round-001"
    assert payload[0]["round_number"] == 1
    assert payload[0]["status"] == "active"
    assert payload[0]["map_width"] == 10
    assert payload[0]["map_height"] == 10
    assert "simulation_params" not in payload[0]


def test_get_round_detail_returns_initial_states(client: TestClient) -> None:
    response = client.get("/astar-island/rounds/test-round-001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "test-round-001"
    assert payload["seeds_count"] == 5
    assert len(payload["initial_states"]) == 5
    assert len(payload["initial_states"][0]["grid"]) == 10
    assert "simulation_params" not in payload


def test_get_round_detail_404_for_missing_round(client: TestClient) -> None:
    response = client.get("/astar-island/rounds/nonexistent")
    assert response.status_code == 404
