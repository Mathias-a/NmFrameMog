from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from astar_twin.contracts.api_models import RoundDetail
from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.solver.adapters.prod import (
    ProdAdapter,
    ProdAdapterConfig,
    ProdAdapterError,
    QueryBudgetExhaustedError,
)
from astar_twin.solver.interfaces import SolverAdapter


def _make_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/astar-island/rounds/round-001":
            return httpx.Response(
                200,
                json={
                    "id": "round-001",
                    "round_number": 1,
                    "status": "active",
                    "map_width": 40,
                    "map_height": 40,
                    "seeds_count": 5,
                    "initial_states": [
                        {
                            "grid": [[10] * 40 for _ in range(40)],
                            "settlements": [],
                        }
                        for _ in range(5)
                    ],
                    "event_date": "2026-03-21",
                    "prediction_window_minutes": 165,
                    "started_at": None,
                    "closes_at": None,
                },
            )
        if request.url.path == "/astar-island/simulate":
            return httpx.Response(
                200,
                json={
                    "grid": [[0] * 5 for _ in range(5)],
                    "settlements": [],
                    "viewport": {"x": 0, "y": 0, "w": 5, "h": 5},
                    "width": 40,
                    "height": 40,
                    "queries_used": 1,
                    "queries_max": 50,
                },
            )
        if request.url.path == "/astar-island/budget":
            return httpx.Response(
                200,
                json={
                    "round_id": "round-001",
                    "queries_used": 1,
                    "queries_max": 50,
                    "active": True,
                },
            )
        if request.url.path == "/astar-island/submit":
            payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "status": "accepted",
                    "round_id": payload["round_id"],
                    "seed_index": payload["seed_index"],
                },
            )
        if request.url.path == "/astar-island/analysis/round-001/0":
            return httpx.Response(
                200,
                json={
                    "prediction": None,
                    "ground_truth": [[[1 / 6] * 6 for _ in range(40)] for _ in range(40)],
                    "score": None,
                    "width": 40,
                    "height": 40,
                    "initial_grid": [[10] * 40 for _ in range(40)],
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler)


def test_prod_adapter_parses_round_detail() -> None:
    client = httpx.Client(transport=_make_transport(), base_url="https://api.ainm.no")
    adapter = ProdAdapter(
        ProdAdapterConfig(base_url="https://api.ainm.no", token="token"),
        client=client,
    )
    detail = adapter.get_round_detail("round-001")
    assert isinstance(detail, RoundDetail)
    assert detail.id == "round-001"
    assert detail.seeds_count == 5
    adapter.close()


def test_prod_adapter_satisfies_protocol() -> None:
    client = httpx.Client(transport=_make_transport(), base_url="https://api.ainm.no")
    adapter = ProdAdapter(
        ProdAdapterConfig(base_url="https://api.ainm.no", token="token"),
        client=client,
    )
    assert isinstance(adapter, SolverAdapter)
    adapter.close()


def test_prod_adapter_budget_and_simulate() -> None:
    client = httpx.Client(transport=_make_transport(), base_url="https://api.ainm.no")
    adapter = ProdAdapter(
        ProdAdapterConfig(base_url="https://api.ainm.no", token="token"),
        client=client,
    )
    response = adapter.simulate("round-001", 0, 0, 0, 5, 5)
    assert response.queries_used == 1
    assert adapter.get_budget("round-001") == (1, 50)
    adapter.close()


def test_prod_adapter_submit_and_analysis() -> None:
    client = httpx.Client(transport=_make_transport(), base_url="https://api.ainm.no")
    adapter = ProdAdapter(
        ProdAdapterConfig(base_url="https://api.ainm.no", token="token", submit_enabled=True),
        client=client,
    )
    prediction = np.full((40, 40, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    submit_response = adapter.submit("round-001", 0, prediction)
    analysis = adapter.get_analysis("round-001", 0)
    assert submit_response.status == "accepted"
    assert analysis.width == 40
    adapter.close()


def test_prod_adapter_submit_disabled_by_default() -> None:
    client = httpx.Client(transport=_make_transport(), base_url="https://api.ainm.no")
    adapter = ProdAdapter(
        ProdAdapterConfig(base_url="https://api.ainm.no", token="token"),
        client=client,
    )
    prediction = np.full((40, 40, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    with pytest.raises(RuntimeError, match="Submission disabled"):
        adapter.submit("round-001", 0, prediction)
    adapter.close()


def test_prod_adapter_normalizes_astar_suffix() -> None:
    client = httpx.Client(transport=_make_transport(), base_url="https://api.ainm.no")
    adapter = ProdAdapter(
        ProdAdapterConfig(base_url="https://api.ainm.no/astar-island", token="token"),
        client=client,
    )
    detail = adapter.get_round_detail("round-001")
    assert detail.id == "round-001"
    adapter.close()


def test_prod_adapter_429_maps_to_budget_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(429, text="budget exhausted")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.ainm.no")
    adapter = ProdAdapter(
        ProdAdapterConfig(base_url="https://api.ainm.no", token="token"),
        client=client,
    )
    with pytest.raises(QueryBudgetExhaustedError):
        adapter.simulate("round-001", 0, 0, 0, 5, 5)
    adapter.close()


def test_prod_adapter_non_budget_http_error_is_distinct() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(403, text="forbidden")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.ainm.no")
    adapter = ProdAdapter(
        ProdAdapterConfig(base_url="https://api.ainm.no", token="token"),
        client=client,
    )
    with pytest.raises(ProdAdapterError):
        adapter.simulate("round-001", 0, 0, 0, 5, 5)
    adapter.close()
