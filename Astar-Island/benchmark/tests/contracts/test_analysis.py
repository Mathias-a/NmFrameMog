from __future__ import annotations

from fastapi.testclient import TestClient


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
