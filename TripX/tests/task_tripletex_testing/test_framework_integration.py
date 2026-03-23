from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]
from fastapi.testclient import TestClient  # pyright: ignore[reportMissingImports]

import task_tripletex.agent as agent_module
import task_tripletex.service as service_module
from task_tripletex.client import TripletexClient
from task_tripletex.models import SolveFile, SolveRequest, TripletexCredentials
from task_tripletex.task_log import task_logger
from task_tripletex.testing.endpoint_runner import run_solve_endpoint
from task_tripletex.testing.fixture_loader import (
    build_solve_request,
    load_packaged_case_fixture,
)
from task_tripletex.testing.reverse_proxy_recorder import ReverseProxyRecorder
from task_tripletex.testing.scoring import compute_score
from task_tripletex.testing.verifier import verify_case
from tests.task_tripletex_testing.helpers import (
    LocalHTTPServer,
    SolveHandler,
    SolveState,
    UpstreamState,
    UpstreamTripletexHandler,
)


class _OptimalEmployeeSolveHandler(SolveHandler):
    def do_POST(self) -> None:
        if self.path != "/solve":
            self._send_json(404, {"error": "not found"})
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = cast(object, json.loads(raw.decode("utf-8")))
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object.")

        state = cast(Any, self.server).state
        assert isinstance(state, SolveState)
        with state.lock:
            state.last_payload = cast(dict[str, object], payload)

        credentials = payload["tripletex_credentials"]
        if not isinstance(credentials, dict):
            raise ValueError("Expected tripletex_credentials object.")
        base_url = credentials["base_url"]
        token = credentials["session_token"]
        if not isinstance(base_url, str) or not isinstance(token, str):
            raise ValueError("Invalid credentials.")

        auth_header = self._basic_auth_header(token)
        self._send_request(
            f"{base_url}/employee?firstName=Ola&lastName=Nordmann&fields=id,firstName,lastName,email&count=50",
            method="GET",
            headers={"Authorization": auth_header},
        )
        self._send_request(
            f"{base_url}/employee",
            method="POST",
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/json",
            },
            payload={
                "firstName": "Ola",
                "lastName": "Nordmann",
                "email": "ola.admin.test@example.org",
                "userType": 2,
                "department": {"id": 1},
            },
        )

        self._send_json(200, {"status": "completed"})


def test_system_prompt_covers_tier_2_and_3_endpoint_families() -> None:
    prompt = agent_module.SYSTEM_PROMPT

    assert "### POST /supplierInvoice" in prompt
    assert "### POST /purchaseOrder" in prompt
    assert "### POST /timesheet" in prompt
    assert "### Payroll" in prompt
    assert "### Voucher import / batch vouchers" in prompt
    assert "### Bank statement import and reconciliation" in prompt
    assert "### Attachments / document endpoints" in prompt
    assert "### Batch and import endpoints" in prompt


def test_system_prompt_keeps_existing_invoice_and_employee_guidance() -> None:
    prompt = agent_module.SYSTEM_PROMPT

    assert "GET /department → POST /employee (userType: 2)" in prompt
    assert "GET requires: `orderDateFrom` AND `orderDateTo`" in prompt
    assert "GET requires: `invoiceDateFrom` AND `invoiceDateTo`" in prompt
    assert "Include `orderLines` inline in the order POST" in prompt


def test_shape_tripletex_tool_response_keeps_write_ids_and_followup_fields() -> None:
    shaped = agent_module._shape_tripletex_tool_response(
        method="POST",
        status_code=201,
        response_body={
            "value": {
                "id": 321,
                "version": 7,
                "employeeNumber": 42,
                "department": {"id": 9, "name": "Accounting"},
                "userType": 2,
            }
        },
    )

    assert shaped == {
        "status_code": 201,
        "value": {
            "id": 321,
            "version": 7,
            "employeeNumber": 42,
            "department": {"id": 9, "name": "Accounting"},
            "userType": 2,
        },
    }
    assert cast(dict[str, object], shaped["value"])["id"] == 321
    assert cast(dict[str, object], shaped["value"])["department"] == {
        "id": 9,
        "name": "Accounting",
    }


def test_shape_tripletex_tool_response_caps_large_lists_and_preserves_counts() -> None:
    shaped = agent_module._shape_tripletex_tool_response(
        method="GET",
        status_code=200,
        response_body={
            "count": 100,
            "from": 20,
            "fullResultSize": 37,
            "values": [
                {"id": index, "name": f"Customer {index}"} for index in range(15)
            ],
        },
    )

    values = cast(list[dict[str, object]], shaped["values"])
    assert shaped["status_code"] == 200
    assert shaped["count"] == 100
    assert shaped["from"] == 20
    assert shaped["fullResultSize"] == 37
    assert shaped["returned_value_count"] == agent_module.MAX_SHAPED_LIST_VALUES
    assert shaped["omitted_value_count"] == 5
    assert shaped["values_truncated"] is True
    assert len(values) == agent_module.MAX_SHAPED_LIST_VALUES
    assert values[0] == {"id": 0, "name": "Customer 0"}
    assert values[-1] == {"id": 9, "name": "Customer 9"}


def test_shape_tripletex_tool_response_keeps_422_validation_messages_verbatim() -> None:
    validation_messages = [
        {
            "field": "postings[0].customer",
            "message": "Kunde mangler",
            "code": "MISSING_CUSTOMER",
        },
        {
            "field": "date",
            "message": "Ugyldig dato",
            "code": "INVALID_DATE",
        },
    ]

    shaped = agent_module._shape_tripletex_tool_response(
        method="POST",
        status_code=422,
        response_body={
            "validationMessages": validation_messages,
            "developerMessage": "Validation failed for request body",
            "requestId": "req-123",
        },
    )

    assert shaped["status_code"] == 422
    assert shaped["validationMessages"] == validation_messages
    assert shaped["developerMessage"] == "Validation failed for request body"
    assert shaped["requestId"] == "req-123"


def test_live_framework_rewrites_proxy_and_counts_writes_and_4xx() -> None:
    case = load_packaged_case_fixture("create_employee_admin")
    upstream_state = UpstreamState()
    solve_state = SolveState()
    credentials = TripletexCredentials(
        base_url="http://placeholder.invalid/v2",
        session_token="test-session-token",
    )

    with LocalHTTPServer(UpstreamTripletexHandler, upstream_state) as upstream:
        with LocalHTTPServer(SolveHandler, solve_state) as solve_server:
            real_credentials = TripletexCredentials(
                base_url=f"{upstream.base_url}/v2",
                session_token=credentials.session_token,
            )
            request = build_solve_request(case, real_credentials)
            with ReverseProxyRecorder(
                real_credentials.base_url,
                real_credentials.session_token,
            ) as recorder:
                endpoint_run = asyncio.run(
                    run_solve_endpoint(
                        f"{solve_server.base_url}/solve",
                        request,
                        proxy_base_url=recorder.advertised_base_url,
                    )
                )
                proxy_metrics = recorder.summarize(
                    rewritten_base_url=endpoint_run.rewritten_request.tripletex_credentials.base_url,
                    expected_min_proxy_calls=case.expected_min_proxy_calls,
                )

            verification = asyncio.run(verify_case(case, real_credentials))
            score = compute_score(
                case,
                verification,
                proxy_metrics,
                endpoint_run.contract,
                enforce_https=False,
            )

    payload = solve_state.last_payload
    assert payload is not None
    solve_credentials = payload["tripletex_credentials"]
    assert isinstance(solve_credentials, dict)
    solve_base_url = solve_credentials.get("base_url")
    assert isinstance(solve_base_url, str)
    assert solve_base_url == endpoint_run.proxy_base_url
    assert endpoint_run.contract.exact_success_response is True
    assert endpoint_run.contract.request_content_type == "application/json"
    assert proxy_metrics.used_proxy is True
    assert proxy_metrics.base_url_rewritten is True
    assert proxy_metrics.all_calls_used_expected_basic_auth is True
    assert proxy_metrics.all_calls_forwarded_to_upstream_base_url is True
    assert proxy_metrics.write_calls == 2
    assert proxy_metrics.client_error_calls == 1
    assert verification.correctness == 1.0
    assert score.total_score == 1.5


def test_employee_admin_baseline_stays_perfect_with_two_calls_and_zero_4xx() -> None:
    case = load_packaged_case_fixture("create_employee_admin")
    upstream_state = UpstreamState()
    solve_state = SolveState()
    credentials = TripletexCredentials(
        base_url="http://placeholder.invalid/v2",
        session_token="test-session-token",
    )

    with LocalHTTPServer(UpstreamTripletexHandler, upstream_state) as upstream:
        with LocalHTTPServer(_OptimalEmployeeSolveHandler, solve_state) as solve_server:
            real_credentials = TripletexCredentials(
                base_url=f"{upstream.base_url}/v2",
                session_token=credentials.session_token,
            )
            request = build_solve_request(case, real_credentials)
            with ReverseProxyRecorder(
                real_credentials.base_url,
                real_credentials.session_token,
            ) as recorder:
                endpoint_run = asyncio.run(
                    run_solve_endpoint(
                        f"{solve_server.base_url}/solve",
                        request,
                        proxy_base_url=recorder.advertised_base_url,
                    )
                )
                proxy_metrics = recorder.summarize(
                    rewritten_base_url=endpoint_run.rewritten_request.tripletex_credentials.base_url,
                    expected_min_proxy_calls=case.expected_min_proxy_calls,
                )

            verification = asyncio.run(verify_case(case, real_credentials))
            score = compute_score(
                case,
                verification,
                proxy_metrics,
                endpoint_run.contract,
                enforce_https=False,
            )

    assert endpoint_run.contract.exact_success_response is True
    assert verification.correctness == 1.0
    assert proxy_metrics.total_calls == 2
    assert proxy_metrics.write_calls == 1
    assert proxy_metrics.client_error_calls == 0
    assert score.total_score == 2.0


def test_endpoint_runner_captures_non_200_contract_failures() -> None:
    upstream_state = UpstreamState()
    solve_state = SolveState()
    credentials = TripletexCredentials(
        base_url="http://placeholder.invalid/v2",
        session_token="test-session-token",
    )

    with LocalHTTPServer(UpstreamTripletexHandler, upstream_state) as upstream:
        with LocalHTTPServer(SolveHandler, solve_state) as solve_server:
            request = build_solve_request(
                load_packaged_case_fixture("create_employee_admin"),
                TripletexCredentials(
                    base_url=f"{upstream.base_url}/v2",
                    session_token=credentials.session_token,
                ),
            )
            endpoint_run = asyncio.run(
                run_solve_endpoint(
                    f"{solve_server.base_url}/not-solve",
                    request,
                    proxy_base_url=f"{upstream.base_url}/v2",
                )
            )

    assert endpoint_run.contract.response_status_code == 404
    assert endpoint_run.contract.exact_success_response is False
    assert "Solve URL is not HTTPS." in endpoint_run.contract.errors
    assert "Solve URL does not target /solve." in endpoint_run.contract.errors
    assert "Solve endpoint returned HTTP 404." in endpoint_run.contract.errors


def test_supplier_invoice_fixture_files_are_forwarded_to_agent_multimodal_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_packaged_case_fixture("supplier_invoice_basic")
    request = build_solve_request(
        case,
        TripletexCredentials(
            base_url="https://example.invalid/v2",
            session_token="test-session-token",
        ),
    )
    captured_contents: list[object] = []

    class _FakeResponse:
        text = "TASK COMPLETED"
        function_calls: list[object] = []

    class _FakeChat:
        async def send_message(self, contents: list[object]) -> _FakeResponse:
            captured_contents.extend(contents)
            return _FakeResponse()

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

    monkeypatch.setattr(agent_module.genai, "Client", _FakeGenAIClient)

    asyncio.run(agent_module.run_agent(request, client=cast(TripletexClient, object())))

    assert captured_contents[0] == case.prompt
    assert len(captured_contents) == 2
    multimodal_part = cast(Any, captured_contents[1])
    assert multimodal_part.inline_data.mime_type == "application/pdf"
    assert multimodal_part.inline_data.data == case.files[0].decoded_content()


def test_run_agent_rejects_unsupported_attachment_mime_type_with_explicit_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SolveRequest(
        prompt="Registrer innholdet i vedlagt fil.",
        files=[
            SolveFile(
                filename="bank.csv",
                content_base64="Y29sMSxjb2wyCg==",
                mime_type="text/csv",
            )
        ],
        tripletex_credentials=TripletexCredentials(
            base_url="https://example.invalid/v2",
            session_token="test-session-token",
        ),
    )

    class _UnexpectedClient:
        def __init__(self, *, api_key: str) -> None:
            raise AssertionError(f"Gemini client should not be created: {api_key}")

    monkeypatch.setattr(agent_module.genai, "Client", _UnexpectedClient)

    task_logger.start_task(request.prompt, request_context={})
    with pytest.raises(ValueError, match=r"Unsupported file MIME type"):
        asyncio.run(
            agent_module.run_agent(request, client=cast(TripletexClient, object()))
        )
    snapshot = task_logger.snapshot()
    task_logger.finish_task(error="invalid file attachment")

    assert snapshot is not None
    rejection_entry = next(
        entry
        for entry in cast(list[dict[str, object]], snapshot["entries"])
        if entry["event"] == "file_attachment_rejected"
    )
    detail = cast(dict[str, object], rejection_entry["detail"])
    assert detail["filename"] == "bank.csv"
    assert detail["mime_type"] == "text/csv"
    assert detail["reason"] == "unsupported_mime_type"
    assert "content_base64" not in detail


def test_run_agent_rejects_oversized_attachment_with_explicit_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SolveRequest(
        prompt="Les vedlagt PDF.",
        files=[
            SolveFile(
                filename="invoice.pdf",
                content_base64="YQ==",
                mime_type="application/pdf",
            )
        ],
        tripletex_credentials=TripletexCredentials(
            base_url="https://example.invalid/v2",
            session_token="test-session-token",
        ),
    )

    class _UnexpectedClient:
        def __init__(self, *, api_key: str) -> None:
            raise AssertionError(f"Gemini client should not be created: {api_key}")

    monkeypatch.setattr(agent_module.genai, "Client", _UnexpectedClient)
    monkeypatch.setattr(agent_module, "MAX_INLINE_FILE_BYTES", 0)

    task_logger.start_task(request.prompt, request_context={})
    with pytest.raises(ValueError, match=r"exceeds inline attachment size limit"):
        asyncio.run(
            agent_module.run_agent(request, client=cast(TripletexClient, object()))
        )
    snapshot = task_logger.snapshot()
    task_logger.finish_task(error="invalid file attachment")

    assert snapshot is not None
    rejection_entry = next(
        entry
        for entry in cast(list[dict[str, object]], snapshot["entries"])
        if entry["event"] == "file_attachment_rejected"
    )
    detail = cast(dict[str, object], rejection_entry["detail"])
    assert detail["filename"] == "invoice.pdf"
    assert detail["mime_type"] == "application/pdf"
    assert detail["reason"] == "file_too_large"
    assert detail["size_bytes"] == 1
    assert detail["max_inline_file_bytes"] == 0


def test_solve_returns_422_for_invalid_file_policy_with_logged_reason() -> None:
    client = TestClient(service_module.create_app())

    response = client.post(
        "/solve",
        json={
            "prompt": "Les vedlagt bankfil.",
            "files": [
                {
                    "filename": "bank.csv",
                    "content_base64": "Y29sMSxjb2wyCg==",
                    "mime_type": "text/csv",
                }
            ],
            "tripletex_credentials": {
                "base_url": "https://example.invalid/v2",
                "session_token": "test-session-token",
            },
        },
    )

    assert response.status_code == 422
    assert "Unsupported file MIME type" in response.json()["detail"]

    logs_response = client.get("/logs")
    assert logs_response.status_code == 200
    logs_payload = logs_response.json()
    assert logs_payload["status"] == "ok"
    trace = cast(dict[str, object], logs_payload["trace"])
    assert trace["status"] == "error"
    assert trace["error"] is not None
    events = [
        cast(str, entry["event"])
        for entry in cast(list[dict[str, object]], trace["entries"])
    ]
    assert events[:3] == [
        "request_context_initialized",
        "request_context_decision",
        "file_attachment_rejected",
    ]
    request_context_decision_entry = next(
        entry
        for entry in cast(list[dict[str, object]], trace["entries"])
        if entry["event"] == "request_context_decision"
    )
    request_context_decision_detail = cast(
        dict[str, object], request_context_decision_entry["detail"]
    )
    assert request_context_decision_detail["decision"] == (
        "inject_current_date_when_needed"
    )
    assert request_context_decision_detail["file_count"] == 1
    rejection_entry = next(
        entry
        for entry in cast(list[dict[str, object]], trace["entries"])
        if entry["event"] == "file_attachment_rejected"
    )
    detail = cast(dict[str, object], rejection_entry["detail"])
    assert detail["filename"] == "bank.csv"
    assert detail["reason"] == "unsupported_mime_type"
    assert "content_base64" not in detail
    serialized_trace = json.dumps(trace)
    assert "test-session-token" not in serialized_trace
    assert "Y29sMSxjb2wyCg==" not in serialized_trace
