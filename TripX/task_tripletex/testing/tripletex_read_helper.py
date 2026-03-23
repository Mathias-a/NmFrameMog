from __future__ import annotations

import base64
import json
from typing import Protocol, cast
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from task_tripletex.models import TripletexCredentials


class _UrlopenResponse(Protocol):
    status: int

    def read(self) -> bytes: ...

    def close(self) -> None: ...


def _normalize_json_value(raw: object) -> object:
    if raw is None or isinstance(raw, (str, int, float, bool)):
        return raw
    if isinstance(raw, list):
        return [_normalize_json_value(item) for item in raw]
    if isinstance(raw, dict):
        normalized: dict[str, object] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings.")
            normalized[key] = _normalize_json_value(value)
        return normalized
    raise ValueError(f"Unsupported JSON value type: {type(raw).__name__}")


def _load_json_document(text: str) -> object:
    return _normalize_json_value(cast(object, json.loads(text)))


def _build_basic_auth_header(session_token: str) -> str:
    raw = f"0:{session_token}".encode()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"Basic {encoded}"


def _prepare_query_params(
    query: dict[str, str | int | float | bool | list[str | int | float | bool]],
) -> dict[str, str | list[str]]:
    params: dict[str, str | list[str]] = {}
    for key, value in query.items():
        if isinstance(value, list):
            params[key] = [_stringify_scalar(item) for item in value]
        else:
            params[key] = _stringify_scalar(value)
    return params


def _stringify_scalar(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class TripletexReadHelper:
    def __init__(
        self,
        credentials: TripletexCredentials,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._credentials = credentials
        self._timeout_seconds = timeout_seconds
        self._headers = {
            "Accept": "application/json",
            "Authorization": _build_basic_auth_header(credentials.session_token),
        }

    async def __aenter__(self) -> TripletexReadHelper:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        await self.close()

    async def close(self) -> None:
        return None

    async def fetch_json(
        self,
        path: str,
        query: dict[str, str | int | float | bool | list[str | int | float | bool]]
        | None = None,
    ) -> object:
        base_url = f"{self._credentials.base_url.rstrip('/')}{path}"
        query_string = urllib_parse.urlencode(
            _prepare_query_params(query or {}), doseq=True
        )
        url = f"{base_url}?{query_string}" if query_string else base_url
        request = urllib_request.Request(url, headers=self._headers, method="GET")
        try:
            response = cast(
                _UrlopenResponse,
                urllib_request.urlopen(request, timeout=self._timeout_seconds),
            )
            try:
                if response.status >= 400:
                    raise ValueError(
                        "Verification read failed: "
                        f"GET {path} returned {response.status}."
                    )
                response_text = response.read().decode("utf-8", errors="replace")
            finally:
                response.close()
        except urllib_error.HTTPError as exc:
            raise ValueError(
                f"Verification read failed: GET {path} returned {exc.code}."
            ) from exc
        return _load_json_document(response_text)

    async def fetch_values(
        self,
        path: str,
        query: dict[str, str | int | float | bool | list[str | int | float | bool]]
        | None = None,
    ) -> list[object]:
        document = await self.fetch_json(path, query)
        if not isinstance(document, dict):
            raise ValueError(f"Expected object response for list read at {path}.")
        values = document.get("values")
        if not isinstance(values, list):
            raise ValueError(
                f"Expected Tripletex list wrapper with 'values' at {path}."
            )
        return values
