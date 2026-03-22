"""Tests for the production adapter — httpx-mocked API interactions."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_httpx import HTTPXMock

from astar_twin.contracts.api_models import (
    AnalysisResponse,
    RoundDetail,
    SimulateResponse,
    SubmitResponse,
)
from astar_twin.contracts.types import MAX_QUERIES, NUM_CLASSES
from astar_twin.solver.adapters.prod import ProdAdapter, _resolve_token
from astar_twin.solver.interfaces import SolverAdapter

BASE = "https://api.ainm.no"

# ---- Fixtures ---------------------------------------------------------------

ROUND_ID = "test-round-abc"

ROUND_DETAIL_RAW = {
    "id": ROUND_ID,
    "round_number": 7,
    "status": "active",
    "map_width": 10,
    "map_height": 10,
    "seeds_count": 5,
    "event_date": "2025-01-01",
    "prediction_window_minutes": 180,
    "started_at": "2025-01-01T12:00:00Z",
    "closes_at": "2025-01-01T15:00:00Z",
    "initial_states": [
        {
            "grid": [[0] * 10 for _ in range(10)],
            "settlements": [{"x": 2, "y": 3, "has_port": False, "alive": True}],
        }
        for _ in range(5)
    ],
}

SIMULATE_RESPONSE_RAW = {
    "grid": [[0] * 10 for _ in range(10)],
    "settlements": [],
    "viewport": {"x": 0, "y": 0, "w": 10, "h": 10},
    "width": 10,
    "height": 10,
    "queries_used": 1,
    "queries_max": MAX_QUERIES,
}

BUDGET_RESPONSE_RAW = {
    "round_id": ROUND_ID,
    "queries_used": 3,
    "queries_max": MAX_QUERIES,
    "active": True,
}

ANALYSIS_RESPONSE_RAW = {
    "prediction": None,
    "ground_truth": [[[1.0 / NUM_CLASSES] * NUM_CLASSES for _ in range(10)] for _ in range(10)],
    "score": None,
    "width": 10,
    "height": 10,
    "initial_grid": [[0] * 10 for _ in range(10)],
}


@pytest.fixture
def adapter() -> ProdAdapter:
    """ProdAdapter with an explicit token (no env lookup)."""
    return ProdAdapter(token="test-token-123", base_url=BASE)


# ---- Protocol compliance ----------------------------------------------------


def test_prod_adapter_satisfies_protocol(adapter: ProdAdapter) -> None:
    """ProdAdapter must satisfy the SolverAdapter protocol."""
    assert isinstance(adapter, SolverAdapter)


# ---- get_round_detail --------------------------------------------------------


def test_get_round_detail(adapter: ProdAdapter, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/rounds/{ROUND_ID}",
        json=ROUND_DETAIL_RAW,
    )
    detail = adapter.get_round_detail(ROUND_ID)
    assert isinstance(detail, RoundDetail)
    assert detail.id == ROUND_ID
    assert detail.round_number == 7
    assert detail.seeds_count == 5
    assert detail.map_width == 10
    assert detail.map_height == 10
    assert len(detail.initial_states) == 5
    # Extra fields from raw API should NOT appear on the model
    assert not hasattr(detail, "event_date")


def test_get_round_detail_parses_initial_states(
    adapter: ProdAdapter, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/rounds/{ROUND_ID}",
        json=ROUND_DETAIL_RAW,
    )
    detail = adapter.get_round_detail(ROUND_ID)
    state = detail.initial_states[0]
    assert len(state.grid) == 10
    assert len(state.grid[0]) == 10
    assert len(state.settlements) == 1
    assert state.settlements[0].x == 2
    assert state.settlements[0].y == 3


# ---- simulate ----------------------------------------------------------------


def test_simulate(adapter: ProdAdapter, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/simulate",
        json=SIMULATE_RESPONSE_RAW,
    )
    resp = adapter.simulate(ROUND_ID, 0, 0, 0, 10, 10)
    assert isinstance(resp, SimulateResponse)
    assert len(resp.grid) == 10
    assert resp.queries_used == 1
    assert resp.queries_max == MAX_QUERIES


def test_simulate_sends_auth_header(adapter: ProdAdapter, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/simulate",
        json=SIMULATE_RESPONSE_RAW,
    )
    adapter.simulate(ROUND_ID, 0, 0, 0, 10, 10)
    request = httpx_mock.get_request()
    assert request is not None
    assert request.headers["authorization"] == "Bearer test-token-123"


def test_simulate_sends_correct_payload(adapter: ProdAdapter, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/simulate",
        json=SIMULATE_RESPONSE_RAW,
    )
    adapter.simulate(ROUND_ID, 2, 5, 10, 15, 12)
    request = httpx_mock.get_request()
    assert request is not None
    import json

    body = json.loads(request.content)
    assert body["round_id"] == ROUND_ID
    assert body["seed_index"] == 2
    assert body["viewport_x"] == 5
    assert body["viewport_y"] == 10
    assert body["viewport_w"] == 15
    assert body["viewport_h"] == 12


def test_simulate_429_raises_runtime_error(adapter: ProdAdapter, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/simulate",
        status_code=429,
        text="Query budget exhausted",
    )
    with pytest.raises(RuntimeError, match="budget exhausted"):
        adapter.simulate(ROUND_ID, 0, 0, 0, 10, 10)


# ---- submit ------------------------------------------------------------------


def test_submit(adapter: ProdAdapter, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/submit",
        json={"status": "accepted", "round_id": ROUND_ID, "seed_index": 0},
    )
    prediction = np.full((10, 10, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    resp = adapter.submit(ROUND_ID, 0, prediction)
    assert isinstance(resp, SubmitResponse)
    assert resp.status == "accepted"


def test_submit_sends_prediction_as_list(adapter: ProdAdapter, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/submit",
        json={"status": "accepted", "round_id": ROUND_ID, "seed_index": 0},
    )
    prediction = np.full((10, 10, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    adapter.submit(ROUND_ID, 0, prediction)
    request = httpx_mock.get_request()
    assert request is not None
    import json

    body = json.loads(request.content)
    assert isinstance(body["prediction"], list)
    assert len(body["prediction"]) == 10
    assert len(body["prediction"][0]) == 10
    assert len(body["prediction"][0][0]) == NUM_CLASSES


# ---- get_analysis ------------------------------------------------------------


def test_get_analysis(adapter: ProdAdapter, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/analysis/{ROUND_ID}/0",
        json=ANALYSIS_RESPONSE_RAW,
    )
    resp = adapter.get_analysis(ROUND_ID, 0)
    assert isinstance(resp, AnalysisResponse)
    assert resp.width == 10
    assert resp.height == 10
    assert len(resp.ground_truth) == 10


def test_get_analysis_sends_auth(adapter: ProdAdapter, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/analysis/{ROUND_ID}/0",
        json=ANALYSIS_RESPONSE_RAW,
    )
    adapter.get_analysis(ROUND_ID, 0)
    request = httpx_mock.get_request()
    assert request is not None
    assert request.headers["authorization"] == "Bearer test-token-123"


# ---- get_budget --------------------------------------------------------------


def test_get_budget(adapter: ProdAdapter, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/astar-island/budget",
        json=BUDGET_RESPONSE_RAW,
    )
    used, max_q = adapter.get_budget(ROUND_ID)
    assert used == 3
    assert max_q == MAX_QUERIES


# ---- Token resolution --------------------------------------------------------


def test_resolve_token_explicit() -> None:
    assert _resolve_token("my-token") == "my-token"


def test_resolve_token_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-token")
    assert _resolve_token() == "env-token"


def test_resolve_token_no_source_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="No API token found"):
        _resolve_token()


# ---- Context manager ---------------------------------------------------------


def test_context_manager_closes(httpx_mock: HTTPXMock) -> None:
    with ProdAdapter(token="t", base_url=BASE) as adapter:
        assert isinstance(adapter, ProdAdapter)
    # After exiting, the client should be closed.
    assert adapter._client.is_closed
