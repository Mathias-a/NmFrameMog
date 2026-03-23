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
class RequestBudget:
    endpoint_timeout_seconds: int
    reserved_headroom_seconds: int
    execution_budget_seconds: int
    max_model_turns: int
    max_tool_calls: int

    def as_log_detail(self) -> dict[str, int]:
        return {
            "endpoint_timeout_seconds": self.endpoint_timeout_seconds,
            "reserved_headroom_seconds": self.reserved_headroom_seconds,
            "execution_budget_seconds": self.execution_budget_seconds,
            "max_model_turns": self.max_model_turns,
            "max_tool_calls": self.max_tool_calls,
        }


@dataclass(frozen=True)
class RequestContext:
    current_date_iso: str
    budget: RequestBudget
    guardrails: tuple[str, ...]
    execution_mode: Literal["synchronous"] = "synchronous"
    solve_response_contract: Literal['{"status":"completed"}'] = (
        '{"status":"completed"}'
    )

    def as_log_detail(self) -> dict[str, object]:
        return {
            "current_date_iso": self.current_date_iso,
            "budget": self.budget.as_log_detail(),
            "guardrails": list(self.guardrails),
            "execution_mode": self.execution_mode,
            "solve_response_contract": self.solve_response_contract,
        }


@dataclass(frozen=True)
class SolveResponse:
    status: Literal["completed"]


@dataclass(frozen=True)
class SolveExecutionOutcome:
    status: Literal["completed", "incomplete"]
    reason: str


class OperationResult(TypedDict):
    method: str
    path: str
    status_code: int
    response_body: object | None


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
