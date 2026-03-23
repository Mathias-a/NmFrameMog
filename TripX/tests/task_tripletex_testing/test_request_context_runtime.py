from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]
from fastapi.testclient import TestClient  # pyright: ignore[reportMissingImports]

import task_tripletex.agent as agent_module
import task_tripletex.service as service_module
from task_tripletex.client import TripletexClient
from task_tripletex.models import (
    SolveExecutionOutcome,
    SolveRequest,
    TripletexCredentials,
)
from task_tripletex.task_log import task_logger


def _build_request(prompt: str) -> SolveRequest:
    return SolveRequest(
        prompt=prompt,
        files=[],
        tripletex_credentials=TripletexCredentials(
            base_url="https://example.invalid/v2",
            session_token="test-session-token",
        ),
    )


class _RuntimeFunctionCall:
    def __init__(
        self,
        *,
        method: str,
        path: str,
        query: dict[str, object] | None = None,
        body: dict[str, object] | None = None,
    ) -> None:
        self.name = "execute_tripletex_api"
        self.args: dict[str, object] = {"method": method, "path": path}
        if query is not None:
            self.args["query"] = query
        if body is not None:
            self.args["body"] = body


class _RuntimeFakeResponse:
    def __init__(
        self, *, text: str = "", function_calls: list[object] | None = None
    ) -> None:
        self.text = text
        self.function_calls = function_calls or []


class _RuntimeFakeChat:
    def __init__(self, responses: list[_RuntimeFakeResponse]) -> None:
        self._responses = iter(responses)
        self.messages: list[list[object]] = []

    async def send_message(self, contents: list[object]) -> _RuntimeFakeResponse:
        self.messages.append(list(contents))
        return next(self._responses)


def _install_fake_genai_client(
    monkeypatch: pytest.MonkeyPatch,
    chat: _RuntimeFakeChat,
) -> None:
    class _FakeChats:
        def create(self, *, model: str, config: object) -> _RuntimeFakeChat:
            del model, config
            return chat

    class _FakeAio:
        chats = _FakeChats()

    class _FakeGenAIClient:
        def __init__(self, *, api_key: str) -> None:
            del api_key
            self.aio = _FakeAio()

    monkeypatch.setattr(agent_module.genai, "Client", _FakeGenAIClient)


def test_run_agent_injects_current_iso_date_into_model_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    request = _build_request("Opprett en kunde som heter Acme AS.")
    request_context = agent_module.build_request_context(
        now=datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    )

    class _FakeResponse:
        text = "TASK COMPLETED"
        function_calls: list[object] = []

    class _FakeChat:
        async def send_message(self, contents: list[object]) -> _FakeResponse:
            captured["contents"] = list(contents)
            return _FakeResponse()

    class _FakeChats:
        def create(self, *, model: str, config: object) -> _FakeChat:
            captured["model"] = model
            captured["config"] = config
            return _FakeChat()

    class _FakeAio:
        chats = _FakeChats()

    class _FakeGenAIClient:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key
            self.aio = _FakeAio()

    monkeypatch.setattr(agent_module.genai, "Client", _FakeGenAIClient)

    task_logger.start_task(request.prompt, request_context={})
    asyncio.run(
        agent_module.run_agent(
            request,
            client=cast(TripletexClient, object()),
            request_context=request_context,
        )
    )
    task_logger.finish_task()

    config = cast(Any, captured["config"])
    system_instruction = "\n".join(cast(list[str], config.system_instruction))

    assert captured["contents"] == [request.prompt]
    assert captured["model"] == agent_module.MODEL_NAME
    assert "Current request date (UTC): 2026-03-21" in system_instruction
    assert (
        "Use this injected date only when the user did not provide an explicit date."
        in system_instruction
    )
    assert "max_model_turns=24" in system_instruction
    assert "max_tool_calls=48" in system_instruction
    assert config.temperature == agent_module.MODEL_TEMPERATURE
    assert config.top_p == agent_module.MODEL_TOP_P
    assert config.candidate_count == agent_module.MODEL_CANDIDATE_COUNT
    assert config.seed == agent_module.MODEL_RANDOM_SEED

    snapshot = task_logger.snapshot()
    assert snapshot is not None
    assert snapshot["request_context"] == {}
    entries = cast(list[dict[str, object]], snapshot["entries"])
    model_context_entry = next(
        entry for entry in entries if entry["event"] == "model_context_injected"
    )
    assert model_context_entry["detail"] == request_context.as_log_detail()
    request_context_decision_entry = next(
        entry for entry in entries if entry["event"] == "request_context_decision"
    )
    assert request_context_decision_entry["detail"] == {
        "decision": "inject_current_date_when_needed",
        "current_date_iso": "2026-03-21",
        "preserve_explicit_user_dates": True,
        "execution_mode": "synchronous",
        "prompt_length": len(request.prompt),
        "file_count": 0,
    }
    runtime_config_entry = next(
        entry for entry in entries if entry["event"] == "model_runtime_configured"
    )
    assert runtime_config_entry["detail"] == {
        "model": agent_module.MODEL_NAME,
        "temperature": agent_module.MODEL_TEMPERATURE,
        "top_p": agent_module.MODEL_TOP_P,
        "candidate_count": agent_module.MODEL_CANDIDATE_COUNT,
        "seed": agent_module.MODEL_RANDOM_SEED,
        "thinking_level": "HIGH",
        "include_thoughts": True,
        "endpoint_timeout_seconds": request_context.budget.endpoint_timeout_seconds,
        "reserved_headroom_seconds": request_context.budget.reserved_headroom_seconds,
        "execution_budget_seconds": request_context.budget.execution_budget_seconds,
        "max_model_turns": request_context.budget.max_model_turns,
        "max_tool_calls": request_context.budget.max_tool_calls,
    }


def test_build_generate_content_config_sets_explicit_deterministic_runtime_controls() -> (
    None
):
    request_context = agent_module.build_request_context(
        now=datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    )

    config = agent_module._build_generate_content_config(request_context)

    assert config.temperature == 0.0
    assert config.top_p == 1.0
    assert config.candidate_count == 1
    assert config.seed == 0
    assert config.system_instruction == [
        agent_module.SYSTEM_PROMPT,
        agent_module._render_request_context_instruction(request_context),
    ]
    assert request_context.budget.execution_budget_seconds == (
        request_context.budget.endpoint_timeout_seconds
        - request_context.budget.reserved_headroom_seconds
    )
    assert request_context.budget.execution_budget_seconds == 270
    assert request_context.budget.endpoint_timeout_seconds == 300
    assert request_context.budget.reserved_headroom_seconds == 30


def test_run_agent_preserves_explicit_user_date_in_prompt_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    explicit_prompt = "Opprett et prosjekt med startdato 2027-04-10 for kunden Acme AS."
    request = _build_request(explicit_prompt)
    request_context = agent_module.build_request_context(
        now=datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    )

    class _FakeResponse:
        text = "TASK COMPLETED"
        function_calls: list[object] = []

    class _FakeChat:
        async def send_message(self, contents: list[object]) -> _FakeResponse:
            captured["contents"] = list(contents)
            return _FakeResponse()

    class _FakeChats:
        def create(self, *, model: str, config: object) -> _FakeChat:
            captured["config"] = config
            return _FakeChat()

    class _FakeAio:
        chats = _FakeChats()

    class _FakeGenAIClient:
        def __init__(self, *, api_key: str) -> None:
            del api_key
            self.aio = _FakeAio()

    monkeypatch.setattr(agent_module.genai, "Client", _FakeGenAIClient)

    asyncio.run(
        agent_module.run_agent(
            request,
            client=cast(TripletexClient, object()),
            request_context=request_context,
        )
    )

    config = cast(Any, captured["config"])
    system_instruction = "\n".join(cast(list[str], config.system_instruction))
    prompt_contents = cast(list[str], captured["contents"])

    assert prompt_contents == [explicit_prompt]
    assert "2027-04-10" in prompt_contents[0]
    assert "Current request date (UTC): 2026-03-21" in system_instruction
    assert (
        "Preserve any explicit user-provided dates exactly as written."
        in system_instruction
    )


def test_solve_preserves_contract_and_exposes_request_context_markers_in_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fixed_request_context = agent_module.build_request_context(
        now=datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    )

    async def _fake_plan_executor(
        request: SolveRequest, request_context: agent_module.RequestContext
    ) -> SolveExecutionOutcome:
        captured["prompt"] = request.prompt
        captured["request_context"] = request_context.as_log_detail()
        task_logger.log(
            "executor_request_context_seen",
            request_context.as_log_detail(),
        )
        return SolveExecutionOutcome(status="completed", reason="test_completed")

    monkeypatch.setattr(
        service_module,
        "build_request_context",
        lambda: fixed_request_context,
    )
    monkeypatch.setattr(service_module, "_default_plan_executor", _fake_plan_executor)

    client = TestClient(service_module.create_app())
    response = client.post(
        "/solve",
        json={
            "prompt": "Opprett en kunde som heter Acme AS.",
            "files": [],
            "tripletex_credentials": {
                "base_url": "https://example.invalid/v2",
                "session_token": "test-session-token",
            },
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.text == '{"status":"completed"}'
    assert response.json() == {"status": "completed"}
    assert captured["prompt"] == "Opprett en kunde som heter Acme AS."
    assert captured["request_context"] == fixed_request_context.as_log_detail()

    logs_response = client.get("/logs")
    assert logs_response.status_code == 200
    logs_payload = logs_response.json()
    assert logs_payload["status"] == "ok"

    trace = cast(dict[str, object], logs_payload["trace"])
    request_context_detail = cast(dict[str, object], trace["request_context"])
    assert request_context_detail["current_date_iso"] == "2026-03-21"
    assert (
        request_context_detail["budget"]
        == fixed_request_context.as_log_detail()["budget"]
    )
    assert (
        request_context_detail["guardrails"]
        == fixed_request_context.as_log_detail()["guardrails"]
    )
    assert request_context_detail["execution_mode"] == "synchronous"
    assert request_context_detail["solve_response_contract"] == '{"status":"completed"}'
    assert isinstance(request_context_detail["request_id"], str)
    assert request_context_detail["request_id"]
    assert trace["status"] == "completed"
    assert trace["final_reason"] == "test_completed"
    entries = cast(list[dict[str, object]], trace["entries"])
    request_context_entry = next(
        entry for entry in entries if entry["event"] == "request_context_initialized"
    )
    assert request_context_entry["detail"] == request_context_detail


def test_solve_preserves_external_contract_when_internal_outcome_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_plan_executor(
        request: SolveRequest, request_context: agent_module.RequestContext
    ) -> SolveExecutionOutcome:
        del request, request_context
        return SolveExecutionOutcome(
            status="incomplete",
            reason="max_tool_calls_reached",
        )

    monkeypatch.setattr(service_module, "_default_plan_executor", _fake_plan_executor)

    client = TestClient(service_module.create_app())
    response = client.post(
        "/solve",
        json={
            "prompt": "Opprett en kunde som heter Acme AS.",
            "files": [],
            "tripletex_credentials": {
                "base_url": "https://example.invalid/v2",
                "session_token": "test-session-token",
            },
        },
    )

    assert response.status_code == 200
    assert response.text == '{"status":"completed"}'
    logs_payload = client.get("/logs").json()
    trace = cast(dict[str, object], logs_payload["trace"])
    assert trace["status"] == "incomplete"
    assert trace["final_reason"] == "max_tool_calls_reached"
    solve_contract_entry = next(
        entry
        for entry in cast(list[dict[str, object]], trace["entries"])
        if entry["event"] == "solve_contract_response"
    )
    assert solve_contract_entry["detail"] == {
        "request_id": trace["request_id"],
        "external_status": "completed",
        "internal_status": "incomplete",
        "internal_reason": "max_tool_calls_reached",
    }


def test_run_agent_retries_once_for_structured_422_and_logs_retry_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[object]] = []
    request = _build_request("Opprett en ansatt med navn Ola Nordmann.")

    class _FunctionCall:
        def __init__(self, *, method: str, path: str, body: dict[str, object]) -> None:
            self.name = "execute_tripletex_api"
            self.args = {"method": method, "path": path, "body": body}

    class _FakeResponse:
        def __init__(
            self, *, text: str = "", function_calls: list[object] | None = None
        ) -> None:
            self.text = text
            self.function_calls = function_calls or []

    class _FakeChat:
        def __init__(self) -> None:
            self._responses = iter(
                [
                    _FakeResponse(
                        function_calls=[
                            _FunctionCall(
                                method="POST",
                                path="/employee",
                                body={"firstName": "Ola"},
                            )
                        ]
                    ),
                    _FakeResponse(
                        function_calls=[
                            _FunctionCall(
                                method="POST",
                                path="/employee",
                                body={
                                    "firstName": "Ola",
                                    "userType": 2,
                                    "email": "ola@example.org",
                                },
                            )
                        ]
                    ),
                    _FakeResponse(text="TASK COMPLETED"),
                ]
            )

        async def send_message(self, contents: list[object]) -> _FakeResponse:
            captured_messages.append(list(contents))
            return next(self._responses)

    class _FakeChats:
        def create(self, *, model: str, config: object) -> _FakeChat:
            del model, config
            return _FakeChat()

    class _FakeAio:
        chats = _FakeChats()

    class _FakeGenAIClient:
        def __init__(self, *, api_key: str) -> None:
            del api_key
            self.aio = _FakeAio()

    class _Executed:
        def __init__(self, *, status_code: int, response_body: object) -> None:
            self.status_code = status_code
            self.response_body = response_body

    class _FakeTripletexClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, object | None]] = []

        async def execute_operation(self, operation: object) -> _Executed:
            method = cast(str, getattr(operation, "method"))
            path = cast(str, getattr(operation, "path"))
            body = getattr(operation, "body")
            self.calls.append((method, path, body))
            if len(self.calls) == 1:
                return _Executed(
                    status_code=422,
                    response_body={
                        "validationMessages": [
                            {
                                "field": "userType",
                                "message": "Brukertype kan ikke være '0' eller tom",
                            },
                            {
                                "field": "email",
                                "message": "Må angis for Tripletex-brukere",
                            },
                        ],
                        "developerMessage": "Validation failed for request body",
                    },
                )
            return _Executed(
                status_code=201,
                response_body={"value": {"id": 123, "email": "ola@example.org"}},
            )

    fake_client = _FakeTripletexClient()
    monkeypatch.setattr(agent_module.genai, "Client", _FakeGenAIClient)

    task_logger.start_task(request.prompt, request_context={})
    asyncio.run(
        agent_module.run_agent(request, client=cast(TripletexClient, fake_client))
    )
    snapshot = task_logger.snapshot()
    task_logger.finish_task()

    assert len(fake_client.calls) == 2
    assert fake_client.calls[0][0:2] == ("POST", "/employee")
    assert fake_client.calls[1][0:2] == ("POST", "/employee")
    assert len(captured_messages) == 3
    repair_turn = captured_messages[1]
    assert len(repair_turn) == 2
    repair_instruction = cast(str, repair_turn[1])
    assert (
        "Repair the last failed Tripletex objective exactly once: POST /employee."
        in repair_instruction
    )
    assert "- userType: Brukertype kan ikke være '0' eller tom" in repair_instruction
    assert "- email: Må angis for Tripletex-brukere" in repair_instruction

    assert snapshot is not None
    entries = cast(list[dict[str, object]], snapshot["entries"])
    decision_entry = next(
        entry for entry in entries if entry["event"] == "client_error_decision"
    )
    detail = cast(dict[str, object], decision_entry["detail"])
    assert detail["decision"] == "retry"
    assert detail["reason"] == "recoverable_validation_422"
    assert detail["objective"] == "POST /employee"
    assert detail["status_code"] == 422
    assert detail["recoverable"] is True
    assert detail["validation_message_count"] == 2
    retry_summary_entry = next(
        entry for entry in entries if entry["event"] == "retry_decision_summary"
    )
    assert retry_summary_entry["detail"] == {
        "step": 0,
        "decision": "retry",
        "reason": "recoverable_validation_422",
        "objective": "POST /employee",
        "status_code": 422,
        "repair_turn_used": False,
    }
    response_shaping_entries = [
        cast(dict[str, object], entry["detail"])
        for entry in entries
        if entry["event"] == "response_shaping_summary"
    ]
    assert response_shaping_entries == [
        {
            "method": "POST",
            "path": "/employee",
            "status_code": 422,
            "response_kind": "error",
            "validation_message_count": 2,
            "error_keys": [
                "developerMessage",
                "status_code",
            ],
        },
        {
            "method": "POST",
            "path": "/employee",
            "status_code": 201,
            "response_kind": "value",
            "value_keys": ["email", "id"],
        },
    ]


def test_run_agent_fails_fast_after_second_4xx_on_same_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[object]] = []
    request = _build_request("Opprett en ansatt med navn Ola Nordmann.")

    class _FunctionCall:
        def __init__(self, *, method: str, path: str, body: dict[str, object]) -> None:
            self.name = "execute_tripletex_api"
            self.args = {"method": method, "path": path, "body": body}

    class _FakeResponse:
        def __init__(
            self, *, text: str = "", function_calls: list[object] | None = None
        ) -> None:
            self.text = text
            self.function_calls = function_calls or []

    class _FakeChat:
        def __init__(self) -> None:
            self._responses = iter(
                [
                    _FakeResponse(
                        function_calls=[
                            _FunctionCall(
                                method="POST",
                                path="/employee",
                                body={"firstName": "Ola"},
                            )
                        ]
                    ),
                    _FakeResponse(
                        function_calls=[
                            _FunctionCall(
                                method="POST",
                                path="/employee",
                                body={
                                    "firstName": "Ola",
                                    "userType": 2,
                                },
                            )
                        ]
                    ),
                ]
            )

        async def send_message(self, contents: list[object]) -> _FakeResponse:
            captured_messages.append(list(contents))
            return next(self._responses)

    class _FakeChats:
        def create(self, *, model: str, config: object) -> _FakeChat:
            del model, config
            return _FakeChat()

    class _FakeAio:
        chats = _FakeChats()

    class _FakeGenAIClient:
        def __init__(self, *, api_key: str) -> None:
            del api_key
            self.aio = _FakeAio()

    class _Executed:
        def __init__(self, *, status_code: int, response_body: object) -> None:
            self.status_code = status_code
            self.response_body = response_body

    class _FakeTripletexClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def execute_operation(self, operation: object) -> _Executed:
            method = cast(str, getattr(operation, "method"))
            path = cast(str, getattr(operation, "path"))
            self.calls.append((method, path))
            return _Executed(
                status_code=422,
                response_body={
                    "validationMessages": [
                        {
                            "field": "email",
                            "message": "Må angis for Tripletex-brukere",
                        }
                    ]
                },
            )

    fake_client = _FakeTripletexClient()
    monkeypatch.setattr(agent_module.genai, "Client", _FakeGenAIClient)

    task_logger.start_task(request.prompt, request_context={})
    asyncio.run(
        agent_module.run_agent(request, client=cast(TripletexClient, fake_client))
    )
    snapshot = task_logger.snapshot()
    task_logger.finish_task()

    assert len(fake_client.calls) == 2
    assert len(captured_messages) == 2

    assert snapshot is not None
    entries = cast(list[dict[str, object]], snapshot["entries"])
    decision_entries = [
        cast(dict[str, object], entry["detail"])
        for entry in entries
        if entry["event"] == "client_error_decision"
    ]
    assert [entry["decision"] for entry in decision_entries] == ["retry", "fail_fast"]
    assert decision_entries[1]["reason"] == "second_client_error_same_objective"
    assert decision_entries[1]["objective"] == "POST /employee"
    assert decision_entries[1]["status_code"] == 422


def test_run_agent_fails_fast_for_employee_standard_time_422_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[object]] = []
    request = _build_request("Opprett en ansatt med navn Ola Nordmann.")

    class _FunctionCall:
        def __init__(self, *, method: str, path: str, body: dict[str, object]) -> None:
            self.name = "execute_tripletex_api"
            self.args = {"method": method, "path": path, "body": body}

    class _FakeResponse:
        def __init__(
            self, *, text: str = "", function_calls: list[object] | None = None
        ) -> None:
            self.text = text
            self.function_calls = function_calls or []

    class _FakeChat:
        def __init__(self) -> None:
            self._responses = iter(
                [
                    _FakeResponse(
                        function_calls=[
                            _FunctionCall(
                                method="POST",
                                path="/employee/standardTime",
                                body={"employee": {"id": 1}, "thursday": 7.5},
                            )
                        ]
                    )
                ]
            )

        async def send_message(self, contents: list[object]) -> _FakeResponse:
            captured_messages.append(list(contents))
            return next(self._responses)

    class _FakeChats:
        def create(self, *, model: str, config: object) -> _FakeChat:
            del model, config
            return _FakeChat()

    class _FakeAio:
        chats = _FakeChats()

    class _FakeGenAIClient:
        def __init__(self, *, api_key: str) -> None:
            del api_key
            self.aio = _FakeAio()

    class _Executed:
        def __init__(self, *, status_code: int, response_body: object) -> None:
            self.status_code = status_code
            self.response_body = response_body

    class _FakeTripletexClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def execute_operation(self, operation: object) -> _Executed:
            method = cast(str, getattr(operation, "method"))
            path = cast(str, getattr(operation, "path"))
            self.calls.append((method, path))
            return _Executed(
                status_code=422,
                response_body={
                    "validationMessages": [
                        {
                            "field": "thursday",
                            "message": "Feltet eksisterer ikke i objektet.",
                        }
                    ]
                },
            )

    fake_client = _FakeTripletexClient()
    monkeypatch.setattr(agent_module.genai, "Client", _FakeGenAIClient)

    task_logger.start_task(request.prompt, request_context={})
    outcome = asyncio.run(
        agent_module.run_agent(request, client=cast(TripletexClient, fake_client))
    )
    snapshot = task_logger.snapshot()
    task_logger.finish_task(status=outcome.status, final_reason=outcome.reason)

    assert fake_client.calls == [("POST", "/employee/standardTime")]
    assert len(captured_messages) == 1
    assert outcome == SolveExecutionOutcome(
        status="incomplete",
        reason="employee_standardtime_non_recoverable",
    )
    assert snapshot is not None
    decision_entry = next(
        entry
        for entry in cast(list[dict[str, object]], snapshot["entries"])
        if entry["event"] == "client_error_decision"
    )
    assert decision_entry["detail"] == {
        "objective": "POST /employee/standardTime",
        "status_code": 422,
        "recoverable": False,
        "validation_message_count": 1,
        "client_error_count_in_turn": 1,
        "decision": "fail_fast",
        "reason": "employee_standardtime_non_recoverable",
    }


def test_run_agent_fails_fast_for_non_recoverable_404_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[object]] = []
    request = _build_request("Finn prosjekt 999 og oppdater det.")

    class _FunctionCall:
        def __init__(self, *, method: str, path: str) -> None:
            self.name = "execute_tripletex_api"
            self.args = {"method": method, "path": path}

    class _FakeResponse:
        def __init__(
            self, *, text: str = "", function_calls: list[object] | None = None
        ) -> None:
            self.text = text
            self.function_calls = function_calls or []

    class _FakeChat:
        def __init__(self) -> None:
            self._responses = iter(
                [
                    _FakeResponse(
                        function_calls=[
                            _FunctionCall(method="GET", path="/project/999")
                        ]
                    )
                ]
            )

        async def send_message(self, contents: list[object]) -> _FakeResponse:
            captured_messages.append(list(contents))
            return next(self._responses)

    class _FakeChats:
        def create(self, *, model: str, config: object) -> _FakeChat:
            del model, config
            return _FakeChat()

    class _FakeAio:
        chats = _FakeChats()

    class _FakeGenAIClient:
        def __init__(self, *, api_key: str) -> None:
            del api_key
            self.aio = _FakeAio()

    class _Executed:
        def __init__(self, *, status_code: int, response_body: object) -> None:
            self.status_code = status_code
            self.response_body = response_body

    class _FakeTripletexClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def execute_operation(self, operation: object) -> _Executed:
            method = cast(str, getattr(operation, "method"))
            path = cast(str, getattr(operation, "path"))
            self.calls.append((method, path))
            return _Executed(status_code=404, response_body={"message": "Not found"})

    fake_client = _FakeTripletexClient()
    monkeypatch.setattr(agent_module.genai, "Client", _FakeGenAIClient)

    task_logger.start_task(request.prompt, request_context={})
    asyncio.run(
        agent_module.run_agent(request, client=cast(TripletexClient, fake_client))
    )
    snapshot = task_logger.snapshot()
    task_logger.finish_task()

    assert fake_client.calls == [("GET", "/project/999")]
    assert len(captured_messages) == 1

    assert snapshot is not None
    entries = cast(list[dict[str, object]], snapshot["entries"])
    decision_entry = next(
        entry for entry in entries if entry["event"] == "client_error_decision"
    )
    detail = cast(dict[str, object], decision_entry["detail"])
    assert detail["decision"] == "fail_fast"
    assert detail["reason"] == "non_recoverable_client_error"
    assert detail["objective"] == "GET /project/999"
    assert detail["status_code"] == 404
    assert detail["recoverable"] is False
    retry_summary_entry = next(
        entry for entry in entries if entry["event"] == "retry_decision_summary"
    )
    assert (
        cast(dict[str, object], retry_summary_entry["detail"])["decision"]
        == "fail_fast"
    )


def test_logs_snapshot_redacts_sensitive_details_but_keeps_trace_breadcrumbs() -> None:
    task_logger.start_task(
        "Sensitive prompt",
        request_context={
            "current_date_iso": "2026-03-21",
            "session_token": "top-secret-token",
        },
    )
    task_logger.log(
        "custom_sensitive_event",
        {
            "session_token": "top-secret-token",
            "content_base64": "c2VjcmV0LWZpbGUtYnl0ZXM=",
            "filename": "invoice.pdf",
            "decision": "accepted_for_inline_model_input",
        },
    )

    snapshot = task_logger.snapshot()
    task_logger.finish_task()

    assert snapshot is not None
    request_context = cast(dict[str, object], snapshot["request_context"])
    assert request_context["current_date_iso"] == "2026-03-21"
    assert request_context["session_token"] == "[redacted]"
    custom_entry = next(
        entry
        for entry in cast(list[dict[str, object]], snapshot["entries"])
        if entry["event"] == "custom_sensitive_event"
    )
    detail = cast(dict[str, object], custom_entry["detail"])
    assert detail["session_token"] == "[redacted]"
    assert detail["content_base64"] == "[redacted]"
    assert detail["filename"] == "invoice.pdf"
    assert detail["decision"] == "accepted_for_inline_model_input"


def test_run_agent_reuses_identical_gets_with_exact_request_local_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _build_request("Finn ansatte to ganger.")
    chat = _RuntimeFakeChat(
        [
            _RuntimeFakeResponse(
                function_calls=[
                    _RuntimeFunctionCall(
                        method="GET",
                        path="/employee",
                        query={"fields": "id,name", "count": 1},
                    ),
                    _RuntimeFunctionCall(
                        method="GET",
                        path="/employee",
                        query={"count": 1, "fields": "id,name"},
                    ),
                ]
            ),
            _RuntimeFakeResponse(text="TASK COMPLETED"),
        ]
    )
    _install_fake_genai_client(monkeypatch, chat)

    class _Executed:
        def __init__(self, *, status_code: int, response_body: object) -> None:
            self.status_code = status_code
            self.response_body = response_body

    class _FakeTripletexClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        async def execute_operation(self, operation: object) -> _Executed:
            method = cast(str, getattr(operation, "method"))
            path = cast(str, getattr(operation, "path"))
            query = cast(dict[str, object], getattr(operation, "query"))
            self.calls.append((method, path, dict(query)))
            return _Executed(
                status_code=200,
                response_body={"values": [{"id": 1, "name": "Ola"}], "count": 1},
            )

    fake_client = _FakeTripletexClient()

    task_logger.start_task(request.prompt, request_context={})
    asyncio.run(
        agent_module.run_agent(request, client=cast(TripletexClient, fake_client))
    )
    snapshot = task_logger.snapshot()
    task_logger.finish_task()

    assert fake_client.calls == [
        ("GET", "/employee", {"fields": "id,name", "count": 1})
    ]
    assert len(chat.messages) == 2
    assert snapshot is not None
    entries = cast(list[dict[str, object]], snapshot["entries"])
    miss_entry = next(
        entry for entry in entries if entry["event"] == "request_cache_miss"
    )
    hit_entry = next(
        entry for entry in entries if entry["event"] == "request_cache_hit"
    )
    response_entries = [
        cast(dict[str, object], entry["detail"])
        for entry in entries
        if entry["event"] == "api_response"
    ]
    assert cast(dict[str, object], miss_entry["detail"])["path"] == "/employee"
    assert cast(dict[str, object], hit_entry["detail"])["path"] == "/employee"
    assert [entry["response_source"] for entry in response_entries] == [
        "upstream",
        "request_cache",
    ]
    assert cast(dict[str, object], miss_entry["detail"])["serialized_query"] == (
        '{"count":"1","fields":"id,name"}'
    )
    assert cast(dict[str, object], hit_entry["detail"])["serialized_query"] == (
        '{"count":"1","fields":"id,name"}'
    )


def test_run_agent_logs_budget_state_when_max_tool_calls_guardrail_stops_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _build_request("Bruk opp verktøykvoten.")
    request_context = agent_module.RequestContext(
        current_date_iso="2026-03-21",
        budget=agent_module.RequestBudget(
            endpoint_timeout_seconds=300,
            reserved_headroom_seconds=30,
            execution_budget_seconds=270,
            max_model_turns=24,
            max_tool_calls=1,
        ),
        guardrails=("enforce_request_budget_caps",),
    )
    chat = _RuntimeFakeChat(
        [
            _RuntimeFakeResponse(
                function_calls=[
                    _RuntimeFunctionCall(method="GET", path="/employee"),
                    _RuntimeFunctionCall(method="GET", path="/department"),
                ]
            ),
            _RuntimeFakeResponse(text="TASK COMPLETED"),
        ]
    )
    _install_fake_genai_client(monkeypatch, chat)

    class _Executed:
        def __init__(self, *, status_code: int, response_body: object) -> None:
            self.status_code = status_code
            self.response_body = response_body

    class _FakeTripletexClient:
        async def execute_operation(self, operation: object) -> _Executed:
            del operation
            return _Executed(status_code=200, response_body={"values": [], "count": 0})

    task_logger.start_task(request.prompt, request_context={})
    asyncio.run(
        agent_module.run_agent(
            request,
            client=cast(TripletexClient, _FakeTripletexClient()),
            request_context=request_context,
        )
    )
    snapshot = task_logger.snapshot()
    task_logger.finish_task()

    assert snapshot is not None
    entries = cast(list[dict[str, object]], snapshot["entries"])
    guardrail_entry = next(
        entry for entry in entries if entry["event"] == "request_budget_guardrail"
    )
    task_completed_entry = next(
        entry for entry in entries if entry["event"] == "task_completed"
    )
    assert cast(dict[str, object], guardrail_entry["detail"])["budget_state"] == (
        "guardrail_applied"
    )
    assert cast(dict[str, object], task_completed_entry["detail"])["budget_state"] == (
        "completed_with_budget_remaining"
    )


def test_run_agent_invalidates_cached_gets_after_write_in_same_resource_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _build_request(
        "Finn ansatte, opprett en ansatt, og finn ansatte på nytt."
    )
    chat = _RuntimeFakeChat(
        [
            _RuntimeFakeResponse(
                function_calls=[
                    _RuntimeFunctionCall(
                        method="GET",
                        path="/employee",
                        query={"fields": "id,name", "count": 1},
                    ),
                    _RuntimeFunctionCall(
                        method="POST",
                        path="/employee",
                        body={
                            "firstName": "Ola",
                            "lastName": "Nordmann",
                            "email": "ola@example.org",
                        },
                    ),
                    _RuntimeFunctionCall(
                        method="GET",
                        path="/employee",
                        query={"fields": "id,name", "count": 1},
                    ),
                ]
            ),
            _RuntimeFakeResponse(text="TASK COMPLETED"),
        ]
    )
    _install_fake_genai_client(monkeypatch, chat)

    class _Executed:
        def __init__(self, *, status_code: int, response_body: object) -> None:
            self.status_code = status_code
            self.response_body = response_body

    class _FakeTripletexClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object], object | None]] = []

        async def execute_operation(self, operation: object) -> _Executed:
            method = cast(str, getattr(operation, "method"))
            path = cast(str, getattr(operation, "path"))
            query = cast(dict[str, object], getattr(operation, "query"))
            body = getattr(operation, "body")
            self.calls.append((method, path, dict(query), body))
            if method == "POST":
                return _Executed(status_code=201, response_body={"value": {"id": 2}})
            return _Executed(
                status_code=200,
                response_body={"values": [{"id": len(self.calls)}], "count": 1},
            )

    fake_client = _FakeTripletexClient()

    task_logger.start_task(request.prompt, request_context={})
    asyncio.run(
        agent_module.run_agent(request, client=cast(TripletexClient, fake_client))
    )
    snapshot = task_logger.snapshot()
    task_logger.finish_task()

    assert [call[0:2] for call in fake_client.calls] == [
        ("GET", "/employee"),
        ("POST", "/employee"),
        ("GET", "/employee"),
    ]
    assert snapshot is not None
    entries = cast(list[dict[str, object]], snapshot["entries"])
    miss_entries = [
        cast(dict[str, object], entry["detail"])
        for entry in entries
        if entry["event"] == "request_cache_miss"
    ]
    invalidate_entry = next(
        cast(dict[str, object], entry["detail"])
        for entry in entries
        if entry["event"] == "request_cache_invalidate"
    )
    response_entries = [
        cast(dict[str, object], entry["detail"])
        for entry in entries
        if entry["event"] == "api_response"
    ]
    assert len(miss_entries) == 2
    assert all(entry["path"] == "/employee" for entry in miss_entries)
    assert invalidate_entry["resource_family"] == "/employee"
    assert invalidate_entry["invalidated_entry_count"] == 1
    assert [entry["response_source"] for entry in response_entries] == [
        "upstream",
        "upstream",
        "upstream",
    ]
