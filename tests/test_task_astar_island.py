from __future__ import annotations

import asyncio
import math
from typing import cast

import httpx

from task_astar_island.client import AstarIslandClient
from task_astar_island.models import AuthConfig
from task_astar_island.prediction import (
    build_probability_grid,
    build_submission_body,
    infer_grid_dimensions,
    validate_probability_grid,
)


def test_build_probability_grid_produces_normalized_non_zero_cells() -> None:
    prediction = build_probability_grid(width=3, height=2, budget=17.0)

    assert len(prediction) == 2
    assert len(prediction[0]) == 3
    for row in prediction:
        for cell in row:
            assert len(cell) == 6
            assert all(probability > 0.0 for probability in cell)
            assert math.isclose(sum(cell), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_infer_grid_dimensions_supports_nested_payloads() -> None:
    payload = {"round": {"grid": {"width": 4, "height": 5}}}
    assert infer_grid_dimensions(cast(dict[str, object], payload)) == (4, 5)


def test_astar_client_uses_auth_headers_and_submit_path() -> None:
    seen: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers["Authorization"]))
        return httpx.Response(status_code=200, json={"accepted": True})

    transport = httpx.MockTransport(handler)
    auth = AuthConfig(token="token-123")

    async def run_client() -> None:
        async with AstarIslandClient(auth, transport=transport) as client:
            submission = build_submission_body(
                build_probability_grid(1, 1), round_id="r1", seed_index=3
            )
            await client.submit(submission)

    asyncio.run(run_client())

    assert seen == [("/astar-island/submit", "Bearer token-123")]


def test_submission_body_matches_documented_shape() -> None:
    submission = build_submission_body(
        build_probability_grid(2, 1), round_id="round-42", seed_index=4
    )

    assert submission["round_id"] == "round-42"
    assert submission["seed_index"] == 4
    prediction = cast(list[list[list[float]]], submission["prediction"])
    assert len(prediction) == 1
    assert len(prediction[0]) == 2


def test_submission_body_rejects_zero_probability_cells() -> None:
    invalid_prediction = [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]

    try:
        validate_probability_grid(invalid_prediction)
    except ValueError as exc:
        assert "strictly greater than zero" in str(exc)
    else:
        raise AssertionError("Expected invalid prediction grid to be rejected.")
