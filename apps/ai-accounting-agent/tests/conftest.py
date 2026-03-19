"""Shared fixtures for the AI Accounting Agent test suite."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from ai_accounting_agent.task_library import ALL_TASKS, TaskDefinition
from ai_accounting_agent.tripletex_client import TripletexClient

sandbox_skip = pytest.mark.skipif(
    not os.environ.get("TRIPLETEX_SESSION_TOKEN"),
    reason="TRIPLETEX_SESSION_TOKEN not set",
)


@pytest.fixture
def session_token() -> str:
    """Return the sandbox session token, skipping if not set."""
    token = os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    if not token:
        pytest.skip("TRIPLETEX_SESSION_TOKEN not set")
    return token


@pytest.fixture
def base_url() -> str:
    """Return the Tripletex proxy base URL."""
    return os.environ.get("TRIPLETEX_BASE_URL", "https://tx-proxy.ainm.no/v2")


@pytest.fixture
def tripletex_client(session_token: str, base_url: str) -> Generator[TripletexClient]:
    """Yield an authenticated TripletexClient, closing it after the test."""
    client = TripletexClient(base_url=base_url, session_token=session_token)
    yield client
    client.close()


@pytest.fixture
def sample_tasks() -> list[TaskDefinition]:
    """Return the first 5 tasks from the task library."""
    return ALL_TASKS[:5]
