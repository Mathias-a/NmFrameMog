from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass
from typing import Literal, TypedDict


@dataclass(frozen=True)
class TripletexCredentials:
    base_url: str
    session_token: str


@dataclass(frozen=True)
class SolveFile:
    filename: str
    content_base64: str
    mime_type: str | None = None

    def decoded_content(self) -> bytes:
        return b64decode(self.content_base64, validate=True)


@dataclass(frozen=True)
class SolveRequest:
    prompt: str
    files: list[SolveFile]
    tripletex_credentials: TripletexCredentials


@dataclass(frozen=True)
class SolveResponse:
    status: Literal["completed"]


class OperationResult(TypedDict):
    method: str
    path: str
    status_code: int


@dataclass(frozen=True)
class StructuredOperation:
    method: Literal["DELETE", "GET", "PATCH", "POST", "PUT"]
    path: str
    query: dict[str, str | int | float | bool | list[str | int | float | bool]]
    body: object | None
    allow_failure: bool = False


@dataclass(frozen=True)
class ExecutionPlan:
    source: str
    operations: list[StructuredOperation]
