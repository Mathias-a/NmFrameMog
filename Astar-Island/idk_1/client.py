from __future__ import annotations

from typing import cast

import httpx

from task_astar_island.models import AuthConfig, normalize_json_value

DEFAULT_BASE_URL = "https://api.ainm.no/astar-island"


class AstarIslandClient:
    def __init__(
        self,
        auth: AuthConfig,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Accept": "application/json", **auth.headers()},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> AstarIslandClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_json(
        self, method: str, path: str, json_body: dict[str, object] | None = None
    ) -> object:
        response = await self._client.request(method, path, json=json_body)
        response.raise_for_status()
        return normalize_json_value(cast(object, response.json()))

    async def get_budget(self) -> object:
        return await self._request_json("GET", "/budget")

    async def get_rounds(self) -> object:
        return await self._request_json("GET", "/rounds")

    async def simulate(self, payload: dict[str, object] | None = None) -> object:
        return await self._request_json("POST", "/simulate", payload or {})

    async def submit(self, payload: dict[str, object]) -> object:
        return await self._request_json("POST", "/submit", payload)
