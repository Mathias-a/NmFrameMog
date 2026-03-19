"""Local test harness for the Tripletex AI Accounting Agent.

Simulates what the competition evaluator does:
1. Sends a POST to /solve with a task prompt + credentials
2. Checks the Tripletex sandbox API to verify the agent's work

Run with: uv run --package ai-accounting-agent pytest tests/ -v
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest


@dataclass
class SandboxConfig:
    """Tripletex sandbox credentials — fill these in from the competition site."""

    base_url: str = "https://tx-proxy.ainm.no/v2"
    session_token: str = ""  # Get from competition site: "Get Sandbox Account"
    agent_url: str = "http://localhost:8080"  # Your local agent endpoint


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def config() -> SandboxConfig:
    cfg = SandboxConfig()
    if not cfg.session_token:
        pytest.skip(
            "Set session_token in SandboxConfig or TRIPLETEX_SESSION_TOKEN env var"
        )
    return cfg


@pytest.fixture
def tripletex_client(config: SandboxConfig) -> httpx.Client:
    """Authenticated client for verifying agent actions against the sandbox."""
    return httpx.Client(
        base_url=config.base_url,
        auth=("0", config.session_token),
        timeout=30.0,
    )


@pytest.fixture
def agent_client(config: SandboxConfig) -> httpx.Client:
    """Client for sending tasks to our local agent endpoint."""
    return httpx.Client(base_url=config.agent_url, timeout=120.0)


# ── Helper ────────────────────────────────────────────────────────────


def send_task(
    agent: httpx.Client,
    config: SandboxConfig,
    prompt: str,
    files: list[dict[str, str]] | None = None,
) -> httpx.Response:
    """Send a task to the agent, mimicking the evaluator's request format."""
    payload: dict[str, str | dict[str, str] | list[dict[str, str]]] = {
        "prompt": prompt,
        "tripletex_credentials": {
            "base_url": config.base_url,
            "session_token": config.session_token,
        },
    }
    if files:
        payload["files"] = files
    return agent.post("/solve", json=payload)


# ── Tests ─────────────────────────────────────────────────────────────


class TestEndpointContract:
    """Tests that the /solve endpoint responds correctly."""

    def test_endpoint_returns_200(
        self, agent_client: httpx.Client, config: SandboxConfig
    ) -> None:
        """Agent must return 200 for a valid task request."""
        resp = send_task(
            agent_client, config, "Opprett en ny ansatt med navn Test Person"
        )
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

    def test_endpoint_returns_json(
        self, agent_client: httpx.Client, config: SandboxConfig
    ) -> None:
        """Agent must return valid JSON."""
        resp = send_task(
            agent_client, config, "Opprett en ny kunde med navn TestKunde AS"
        )
        assert resp.headers.get("content-type", "").startswith("application/json"), (
            f"Expected JSON content-type, got {resp.headers.get('content-type')}"
        )
        json.loads(resp.text)  # Should not raise


class TestCreateEmployee:
    """Verify agent can create an employee and it shows up in the Tripletex API."""

    def test_create_employee(
        self,
        agent_client: httpx.Client,
        tripletex_client: httpx.Client,
        config: SandboxConfig,
    ) -> None:
        """Send a 'create employee' task and verify via Tripletex API."""
        first_name = "TestLocal"
        last_name = "Harness"

        resp = send_task(
            agent_client,
            config,
            f"Opprett en ny ansatt med fornavn {first_name} og etternavn {last_name}",
        )
        assert resp.status_code == 200

        # Verify: query the Tripletex API for the employee
        search = tripletex_client.get(
            "/employee", params={"firstName": first_name, "lastName": last_name}
        )
        assert search.status_code == 200
        data = search.json()
        employees = data.get("values", [])
        assert len(employees) >= 1, (
            f"Employee {first_name} {last_name} not found in Tripletex after agent ran"
        )
        emp = employees[0]
        assert emp["firstName"] == first_name
        assert emp["lastName"] == last_name


class TestCreateCustomer:
    """Verify agent can create a customer."""

    def test_create_customer(
        self,
        agent_client: httpx.Client,
        tripletex_client: httpx.Client,
        config: SandboxConfig,
    ) -> None:
        """Send a 'create customer' task and verify via Tripletex API."""
        customer_name = "LocalTest AS"

        resp = send_task(
            agent_client,
            config,
            f"Opprett en ny kunde med navn {customer_name}",
        )
        assert resp.status_code == 200

        search = tripletex_client.get("/customer", params={"name": customer_name})
        assert search.status_code == 200
        data = search.json()
        customers = data.get("values", [])
        assert len(customers) >= 1, (
            f"Customer '{customer_name}' not found in Tripletex after agent ran"
        )


class TestMultiLanguage:
    """Verify agent handles multiple languages."""

    @pytest.mark.parametrize(
        "lang,prompt",
        [
            ("no", "Opprett en ny ansatt med navn Språktest Person"),
            ("en", "Create a new employee named Langtest Person"),
            ("de", "Erstellen Sie einen neuen Mitarbeiter namens Sprachtest Person"),
        ],
    )
    def test_multilang_returns_200(
        self,
        agent_client: httpx.Client,
        config: SandboxConfig,
        lang: str,
        prompt: str,
    ) -> None:
        """Agent should handle tasks in different languages."""
        resp = send_task(agent_client, config, prompt)
        assert resp.status_code == 200, f"[{lang}] {resp.status_code}: {resp.text}"
