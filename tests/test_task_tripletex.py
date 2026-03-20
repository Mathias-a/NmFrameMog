from __future__ import annotations

import asyncio
from base64 import b64encode

import httpx
from fastapi.testclient import TestClient

from task_tripletex.client import TripletexClient
from task_tripletex.models import (
    ExecutionPlan,
    SolveFile,
    SolveRequest,
    StructuredOperation,
    TripletexCredentials,
)
from task_tripletex.planning import extract_execution_plan
from task_tripletex.service import create_app


def test_extract_execution_plan_from_json_file() -> None:
    request = SolveRequest(
        prompt="Use the attached plan.",
        files=[
            SolveFile(
                filename="tripletex_operations.json",
                content_base64=b64encode(
                    b'{"operations": [{"method": "GET", "path": "/company"}]}'
                ).decode("ascii"),
                mime_type="application/json",
            )
        ],
        tripletex_credentials=TripletexCredentials(
            base_url="https://example.invalid",
            session_token="secret-token",
        ),
    )

    plan = extract_execution_plan(request)

    assert plan.source == "file:tripletex_operations.json"
    assert [(operation.method, operation.path) for operation in plan.operations] == [
        ("GET", "/company")
    ]


def test_tripletex_client_uses_required_basic_auth() -> None:
    captured_headers: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.append(request.headers["Authorization"])
        return httpx.Response(status_code=200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    credentials = TripletexCredentials(
        base_url="https://tripletex.example", session_token="abc123"
    )
    plan = ExecutionPlan(
        source="test",
        operations=[
            StructuredOperation(method="GET", path="/company", query={}, body=None)
        ],
    )

    async def run_client() -> None:
        async with TripletexClient(credentials, transport=transport) as client:
            await client.execute_plan(plan)

    asyncio.run(run_client())

    expected = f"Basic {b64encode(b'0:abc123').decode('ascii')}"
    assert captured_headers == [expected]


def test_solve_endpoint_returns_completed_for_structured_plan() -> None:
    captured: list[tuple[str, list[str]]] = []

    async def fake_executor(
        credentials: TripletexCredentials, plan: ExecutionPlan
    ) -> None:
        captured.append(
            (credentials.base_url, [operation.path for operation in plan.operations])
        )

    app = create_app(executor=fake_executor)
    client = TestClient(app)

    response = client.post(
        "/solve",
        json={
            "prompt": (
                '```json\n{"operations": [{"method": "GET", "path": "/company"}]}\n```'
            ),
            "files": [
                {
                    "filename": "ignored.txt",
                    "content_base64": b64encode(b"ignored").decode("ascii"),
                    "mime_type": "text/plain",
                }
            ],
            "tripletex_credentials": {
                "base_url": "https://tripletex.example",
                "session_token": "abc123",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "completed"}
    assert captured == [("https://tripletex.example", ["/company"])]


def test_solve_endpoint_rejects_invalid_base64_file_payload() -> None:
    app = create_app(executor=lambda _credentials, _plan: asyncio.sleep(0))
    client = TestClient(app)
    prompt = '```json\n{"operations": [{"method": "GET", "path": "/company"}]}\n```'

    response = client.post(
        "/solve",
        json={
            "prompt": prompt,
            "files": [
                {
                    "filename": "plan.json",
                    "content_base64": "@@not-base64@@",
                    "mime_type": "application/json",
                }
            ],
            "tripletex_credentials": {
                "base_url": "https://tripletex.example",
                "session_token": "abc123",
            },
        },
    )

    assert response.status_code == 422


def test_prompt_plan_is_used_when_json_attachment_is_not_a_plan() -> None:
    captured: list[str] = []
    prompt = '```json\n{"operations": [{"method": "GET", "path": "/company"}]}\n```'

    async def fake_executor(
        _credentials: TripletexCredentials, plan: ExecutionPlan
    ) -> None:
        captured.extend(operation.path for operation in plan.operations)

    app = create_app(executor=fake_executor)
    client = TestClient(app)

    response = client.post(
        "/solve",
        json={
            "prompt": prompt,
            "files": [
                {
                    "filename": "metadata.json",
                    "content_base64": b64encode(b'{"note":"not a plan"}').decode(
                        "ascii"
                    ),
                    "mime_type": "application/json",
                }
            ],
            "tripletex_credentials": {
                "base_url": "https://tripletex.example",
                "session_token": "abc123",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "completed"}
    assert captured == ["/company"]
