from __future__ import annotations

from fastapi.testclient import TestClient


def _uniform_prediction(height: int = 10, width: int = 10) -> list[list[list[float]]]:
    return [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(width)] for _ in range(height)]


def _valid_prediction(height: int = 10, width: int = 10) -> list[list[list[float]]]:
    return [[[0.5, 0.1, 0.1, 0.1, 0.1, 0.1] for _ in range(width)] for _ in range(height)]


def test_valid_submission_returns_accepted(client: TestClient) -> None:
    response = client.post(
        "/astar-island/submit",
        json={"round_id": "test-round-001", "seed_index": 0, "prediction": _valid_prediction()},
    )
    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "round_id": "test-round-001",
        "seed_index": 0,
    }


def test_resubmit_overwrites_existing_prediction(client: TestClient) -> None:
    first = client.post(
        "/astar-island/submit",
        json={"round_id": "test-round-001", "seed_index": 0, "prediction": _valid_prediction()},
    )
    second_prediction = [[[0.6, 0.1, 0.1, 0.1, 0.05, 0.05] for _ in range(10)] for _ in range(10)]
    second = client.post(
        "/astar-island/submit",
        json={"round_id": "test-round-001", "seed_index": 0, "prediction": second_prediction},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "accepted"


def test_submit_wrong_row_count(client: TestClient) -> None:
    response = client.post(
        "/astar-island/submit",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "prediction": _valid_prediction(height=9),
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Expected 10 rows, got 9"


def test_submit_wrong_col_count(client: TestClient) -> None:
    response = client.post(
        "/astar-island/submit",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "prediction": _valid_prediction(width=5),
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Row 0: expected 10 cols, got 5"


def test_submit_wrong_prob_length(client: TestClient) -> None:
    prediction = _valid_prediction()
    prediction[0][0] = [0.4, 0.3, 0.3]
    response = client.post(
        "/astar-island/submit",
        json={"round_id": "test-round-001", "seed_index": 0, "prediction": prediction},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Cell (0,0): expected 6 probs, got 3"


def test_submit_negative_probability(client: TestClient) -> None:
    prediction = _valid_prediction()
    prediction[0][0] = [0.7, -0.1, 0.1, 0.1, 0.1, 0.1]
    response = client.post(
        "/astar-island/submit",
        json={"round_id": "test-round-001", "seed_index": 0, "prediction": prediction},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Cell (0,0): negative probability"


def test_submit_probabilities_must_sum_to_one(client: TestClient) -> None:
    prediction = _valid_prediction()
    prediction[0][0] = [0.5, 0.2, 0.1, 0.1, 0.1, 0.1]
    response = client.post(
        "/astar-island/submit",
        json={"round_id": "test-round-001", "seed_index": 0, "prediction": prediction},
    )
    assert response.status_code == 422
    assert "Cell (0,0): probs sum to" in response.json()["detail"]
