from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import cast

from .models import AuthConfig, normalize_json_value

DEFAULT_BASE_URL = "https://api.ainm.no/astar-island"


class AstarIslandClient:
    def __init__(
        self,
        auth: AuthConfig,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Accept": "application/json", **auth.headers()}
        self._timeout = timeout

    async def __aenter__(self) -> AstarIslandClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    async def close(self) -> None:
        return None

    async def _request_json(
        self, method: str, path: str, json_body: dict[str, object] | None = None
    ) -> object:
        return await asyncio.to_thread(
            self._request_json_sync, method, path, json_body or None
        )

    def _request_json_sync(
        self, method: str, path: str, json_body: dict[str, object] | None = None
    ) -> object:
        data: bytes | None = None
        headers = dict(self._headers)
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url=f"{self._base_url}{path}",
            method=method,
            headers=headers,
            data=data,
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return normalize_json_value(cast(object, payload))

    async def get_budget(self) -> object:
        return await self._request_json("GET", "/budget")

    async def get_rounds(self) -> object:
        return await self._request_json("GET", "/rounds")

    async def simulate(self, payload: dict[str, object] | None = None) -> object:
        return await self._request_json("POST", "/simulate", payload or {})

    async def submit(self, payload: dict[str, object]) -> object:
        return await self._request_json("POST", "/submit", payload)


__all__ = ["AstarIslandClient", "DEFAULT_BASE_URL"]
