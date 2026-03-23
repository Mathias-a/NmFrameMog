from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from typing import cast

import httpx

from task_tripletex.models import (
    ExecutionPlan,
    OperationResult,
    StructuredOperation,
    TripletexCredentials,
)


class TripletexExecutionError(RuntimeError):
    """Raised when an operation in the execution plan fails."""


@dataclass(frozen=True)
class ExecutedOperation:
    method: str
    path: str
    status_code: int
    response_body: object | None = None

    def to_result(self) -> OperationResult:
        return {
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "response_body": self.response_body,
        }


def _basic_auth_header(session_token: str) -> str:
    raw = f"0:{session_token}".encode()
    encoded = b64encode(raw).decode("ascii")
    return f"Basic {encoded}"


def _stringify_query_scalar(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _prepare_query_params(
    query: dict[str, str | int | float | bool | list[str | int | float | bool]],
) -> dict[str, str | list[str]]:
    params: dict[str, str | list[str]] = {}
    for key, value in query.items():
        if isinstance(value, list):
            params[key] = [_stringify_query_scalar(item) for item in value]
        else:
            params[key] = _stringify_query_scalar(value)
    return params


class TripletexClient:
    def __init__(
        self,
        credentials: TripletexCredentials,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=credentials.base_url.rstrip("/"),
            headers={
                "Accept": "application/json",
                "Authorization": _basic_auth_header(credentials.session_token),
            },
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> TripletexClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def execute_operation(
        self, operation: StructuredOperation
    ) -> ExecutedOperation:
        response = await self._client.request(
            operation.method,
            operation.path,
            params=_prepare_query_params(operation.query),
            json=operation.body,
        )

        response_body: object | None = None
        try:
            response_body = cast(object, response.json())
        except ValueError:
            response_body = response.text if response.text else None

        if response.status_code >= 400 and not operation.allow_failure:
            raise TripletexExecutionError(
                f"Tripletex operation failed: {operation.method} {operation.path} "
                f"returned {response.status_code}. Response: {response_body}"
            )

        return ExecutedOperation(
            method=operation.method,
            path=operation.path,
            status_code=response.status_code,
            response_body=response_body,
        )

    async def execute_plan(self, plan: ExecutionPlan) -> list[ExecutedOperation]:
        return [
            await self.execute_operation(operation) for operation in plan.operations
        ]
