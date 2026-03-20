from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from http.client import HTTPResponse
from typing import cast

from .models import QueryResponse, SettlementObservation, Viewport


class AstarIslandClient:
    def __init__(self, *, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def get_rounds(self) -> object:
        return self._request_json("GET", "/rounds")

    def get_round_detail(self, round_id: str) -> object:
        return self._request_json("GET", f"/rounds/{round_id}")

    def simulate(
        self, *, round_id: str, seed_index: int, viewport: Viewport
    ) -> QueryResponse:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport.x,
            "viewport_y": viewport.y,
            "viewport_w": viewport.width,
            "viewport_h": viewport.height,
        }
        response = self._request_json("POST", "/simulate", payload)
        if not isinstance(response, dict):
            raise ValueError("Simulate response must be a JSON object.")
        response_mapping = cast(Mapping[str, object], response)

        raw_grid = response_mapping.get("grid")
        raw_settlements_obj = response_mapping.get("settlements")
        raw_settlements = [] if raw_settlements_obj is None else raw_settlements_obj
        if not isinstance(raw_settlements, list):
            raise ValueError("Simulate response field 'settlements' must be a list.")

        return QueryResponse(
            viewport=viewport,
            grid=_parse_grid(raw_grid),
            settlements=tuple(
                _parse_settlement_observation(item) for item in raw_settlements
            ),
            queries_used=_optional_int(response_mapping.get("queries_used")),
            queries_max=_optional_int(response_mapping.get("queries_max")),
        )

    def submit_prediction(
        self, *, round_id: str, seed_index: int, prediction: list[list[list[float]]]
    ) -> object:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        return self._request_json("POST", "/submit", payload)

    def get_analysis(self, *, round_id: str, seed_index: int) -> object:
        return self._request_json("GET", f"/analysis/{round_id}/{seed_index}")

    def _request_json(
        self, method: str, path: str, payload: object | None = None
    ) -> object:
        if payload is not None and not _is_json_value(payload):
            raise ValueError(f"Request payload for {path} is not JSON-compatible.")

        data = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            method=method,
            headers=headers,
            data=data,
        )
        try:
            raw_response: object = urllib.request.urlopen(request)
            http_response = _require_http_response(raw_response)
            with http_response:
                decoded: object = json.loads(http_response.read().decode("utf-8"))
            if not _is_json_value(decoded):
                raise ValueError(f"Response from {path} is not JSON-compatible.")
            return decoded
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code} for {path}: {body}") from error


def query_response_to_payload(response: QueryResponse) -> dict[str, object]:
    return {
        "viewport": {
            "x": response.viewport.x,
            "y": response.viewport.y,
            "width": response.viewport.width,
            "height": response.viewport.height,
        },
        "grid": response.grid,
        "settlements": [
            {
                "x": settlement.x,
                "y": settlement.y,
                "population": settlement.population,
                "food": settlement.food,
                "wealth": settlement.wealth,
                "defense": settlement.defense,
                "has_port": settlement.has_port,
                "alive": settlement.alive,
                "owner_id": settlement.owner_id,
            }
            for settlement in response.settlements
        ],
        "queries_used": response.queries_used,
        "queries_max": response.queries_max,
    }


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Expected integer value, got {value!r}")
    return value


def _parse_grid(raw_grid: object) -> list[list[int]]:
    if not isinstance(raw_grid, list):
        raise ValueError("Grid payload must be a list.")
    grid: list[list[int]] = []
    for row in raw_grid:
        if not isinstance(row, list):
            raise ValueError("Each grid row must be a list.")
        grid.append([_require_int(cell, field_name="grid cell") for cell in row])
    return grid


def _parse_settlement_observation(item: object) -> SettlementObservation:
    if not isinstance(item, dict):
        raise ValueError("Settlement entry must be an object.")
    item_mapping = cast(Mapping[str, object], item)
    return SettlementObservation(
        x=_require_int(item_mapping.get("x"), field_name="settlement.x"),
        y=_require_int(item_mapping.get("y"), field_name="settlement.y"),
        population=_require_float(
            item_mapping.get("population"), field_name="settlement.population"
        ),
        food=_require_float(item_mapping.get("food"), field_name="settlement.food"),
        wealth=_require_float(
            item_mapping.get("wealth"), field_name="settlement.wealth"
        ),
        defense=_require_float(
            item_mapping.get("defense"), field_name="settlement.defense"
        ),
        has_port=_require_bool(
            item_mapping.get("has_port"), field_name="settlement.has_port"
        ),
        alive=_require_bool(item_mapping.get("alive"), field_name="settlement.alive"),
        owner_id=_require_int(
            item_mapping.get("owner_id"), field_name="settlement.owner_id"
        ),
    )


def _require_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Field '{field_name}' must be an integer.")
    return value


def _require_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Field '{field_name}' must be numeric.")
    return float(value)


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Field '{field_name}' must be a boolean.")
    return value


def _require_http_response(response: object) -> HTTPResponse:
    if not isinstance(response, HTTPResponse):
        raise ValueError("Expected an HTTP response object.")
    return response


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False
