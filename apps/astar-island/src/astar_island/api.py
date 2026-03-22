"""API client for the A* Island competition server.

Handles round discovery, round detail fetching, prediction submission,
and historical analysis retrieval.

Base URL: https://api.ainm.no/astar-island
Auth: Bearer token from ASTAR_API_TOKEN environment variable or .env file.

All outbound requests use httpx with retry/backoff on HTTP 429.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

API_BASE: str = "https://api.ainm.no/astar-island"

# Retry configuration
_MAX_RETRIES_SUBMIT: int = 5
_MAX_RETRIES_DEFAULT: int = 3


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def _resolve_token() -> str:
    """Resolve API token from environment or .env file.

    Checks ASTAR_API_TOKEN env var first, then falls back to parsing
    the .env file at the project root (4 levels up from this file).

    Returns:
        The JWT token string.

    Raises:
        ValueError: If no token is found.
    """
    token = os.environ.get("ASTAR_API_TOKEN", "")
    if token:
        return token

    # Fall back to .env at project root
    env_path = Path(__file__).resolve().parents[4] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("ASTAR_API_TOKEN="):
                token = stripped.split("=", 1)[1].strip().strip("\"'")
                if token:
                    return token

    msg = (
        "ASTAR_API_TOKEN not found. Set it as an environment variable "
        "or add ASTAR_API_TOKEN=<token> to the .env file at the project root."
    )
    raise ValueError(msg)


def make_client(token: str | None = None) -> httpx.Client:
    """Create an authenticated httpx.Client.

    Args:
        token: Optional JWT token. If None, resolves from env / .env.

    Returns:
        Configured httpx.Client with Bearer auth, 60s timeout, transport retries.
    """
    resolved_token = token or _resolve_token()
    return httpx.Client(
        headers={"Authorization": f"Bearer {resolved_token}"},
        timeout=60.0,
        transport=httpx.HTTPTransport(retries=3),
    )


# ---------------------------------------------------------------------------
# Request helper with 429 retry/backoff
# ---------------------------------------------------------------------------


def _request(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    json_body: dict[str, object] | None = None,
    max_retries: int = _MAX_RETRIES_DEFAULT,
    timeout: float | None = None,
) -> httpx.Response:
    """Send an HTTP request with retry on 429.

    Args:
        client: Authenticated httpx.Client.
        method: HTTP method ("GET" or "POST").
        path: URL path relative to API_BASE (e.g. "/rounds").
        json_body: Optional JSON payload for POST requests.
        max_retries: Maximum retry attempts on 429.
        timeout: Per-request timeout override.

    Returns:
        The httpx.Response (status already checked via raise_for_status).

    Raises:
        httpx.HTTPStatusError: On non-2xx after retries exhausted.
        RuntimeError: If retries exhausted on 429.
    """
    url = f"{API_BASE}{path}"
    kwargs: dict[str, Any] = {}
    if json_body is not None:
        kwargs["json"] = json_body
    if timeout is not None:
        kwargs["timeout"] = timeout

    for attempt in range(max_retries + 1):
        if method.upper() == "GET":
            resp = client.get(url, **kwargs)
        else:
            resp = client.post(url, **kwargs)

        if resp.status_code == 429:
            if attempt < max_retries:
                wait = 3.0 * (attempt + 1)
                logger.warning(
                    "Rate limited (429) on %s %s, attempt %d/%d, retrying in %.1fs",
                    method,
                    path,
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue
            msg = f"Rate limited after {max_retries} retries on {method} {path}"
            raise RuntimeError(msg)

        resp.raise_for_status()
        return resp

    # Unreachable — satisfies type checker
    msg = f"_request: exhausted retries on {method} {path}"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Round discovery
# ---------------------------------------------------------------------------


def get_rounds(client: httpx.Client) -> list[dict[str, object]]:
    """Fetch all rounds from the API.

    Returns:
        List of round summary dicts. Each contains at minimum:
        'id', 'round_number', 'status', 'map_width', 'map_height'.
    """
    resp = _request(client, "GET", "/rounds")
    raw: object = resp.json()
    if not isinstance(raw, list):
        logger.warning("Unexpected /rounds response type: %s", type(raw))
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def get_active_round(client: httpx.Client) -> dict[str, object] | None:
    """Find the currently active round, if any.

    Returns:
        The active round dict, or None if no round is active.
    """
    rounds = get_rounds(client)
    for r in rounds:
        status = r.get("status")
        if isinstance(status, str) and status == "active":
            return r
    return None


def get_round_detail(client: httpx.Client, round_id: str) -> dict[str, object]:
    """Fetch detailed information for a specific round.

    Args:
        round_id: The UUID string of the round.

    Returns:
        Round detail dict with 'id', 'map_width', 'map_height',
        'seeds_count', 'initial_states'.

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
        TypeError: If response is not a dict.
    """
    resp = _request(client, "GET", f"/rounds/{round_id}")
    raw: object = resp.json()
    if not isinstance(raw, dict):
        msg = f"Unexpected response type from /rounds/{round_id}: {type(raw)}"
        raise TypeError(msg)
    return dict(raw)


# ---------------------------------------------------------------------------
# Prediction submission
# ---------------------------------------------------------------------------


def submit_prediction(
    client: httpx.Client,
    round_id: str,
    seed_index: int,
    prediction: list[list[list[float]]],
) -> dict[str, object]:
    """Submit a prediction tensor to the competition API.

    Args:
        client: Authenticated httpx.Client.
        round_id: The UUID of the active round.
        seed_index: Seed index (0-based).
        prediction: Nested list of shape (H, W, 6) with probability values.
            Each cell must sum to ~1.0 and contain no zeros/negatives.

    Returns:
        The submission response dict (typically {"status": "accepted", ...}).

    Raises:
        RuntimeError: If retries exhausted.
        httpx.HTTPStatusError: On 4xx/5xx (validation failure, etc.).
    """
    body: dict[str, object] = {
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction,
    }
    resp = _request(
        client,
        "POST",
        "/submit",
        json_body=body,
        max_retries=_MAX_RETRIES_SUBMIT,
        timeout=60.0,
    )
    raw: object = resp.json()
    if not isinstance(raw, dict):
        return {"raw": raw}
    return dict(raw)


# ---------------------------------------------------------------------------
# Analysis (historical ground truth)
# ---------------------------------------------------------------------------


def get_analysis(
    client: httpx.Client,
    round_id: str,
    seed_index: int,
) -> dict[str, object] | None:
    """Fetch analysis data (ground truth) for a completed round seed.

    Args:
        client: Authenticated httpx.Client.
        round_id: The UUID of the round.
        seed_index: Seed index (0-based).

    Returns:
        Analysis dict with 'ground_truth', 'initial_grid', 'score', etc.
        Returns None if the analysis is unavailable (404/409).
    """
    try:
        resp = _request(client, "GET", f"/analysis/{round_id}/{seed_index}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (404, 409):
            logger.debug(
                "Analysis unavailable for round %s seed %d (HTTP %d)",
                round_id,
                seed_index,
                exc.response.status_code,
            )
            return None
        raise

    raw: object = resp.json()
    if not isinstance(raw, dict):
        logger.warning(
            "Unexpected analysis response type for %s/%d: %s",
            round_id,
            seed_index,
            type(raw),
        )
        return None
    return dict(raw)


# ---------------------------------------------------------------------------
# Simulation queries
# ---------------------------------------------------------------------------


def simulate_query(
    client: httpx.Client,
    round_id: str,
    seed_index: int,
    vx: int,
    vy: int,
    viewport_size: int = 15,
) -> dict[str, object]:
    """Send a viewport simulation query to the API.

    Each query runs an independent 50-year simulation and returns the
    terrain state for the requested viewport region.

    Args:
        client: Authenticated httpx.Client.
        round_id: The UUID of the round.
        seed_index: Seed index (0-based).
        vx: Viewport top-left x coordinate.
        vy: Viewport top-left y coordinate.
        viewport_size: Width and height of the viewport (default 15, max 15).

    Returns:
        Response dict with 'grid' (list of rows), 'viewport' (x,y,w,h),
        'settlements' (list), 'queries_used', 'queries_max'.

    Raises:
        httpx.HTTPStatusError: On non-2xx after retries.
        RuntimeError: If retries exhausted on 429.
    """
    body: dict[str, object] = {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": vx,
        "viewport_y": vy,
        "viewport_w": viewport_size,
        "viewport_h": viewport_size,
    }
    resp = _request(
        client,
        "POST",
        "/simulate",
        json_body=body,
        max_retries=_MAX_RETRIES_DEFAULT,
    )
    raw: object = resp.json()
    if not isinstance(raw, dict):
        msg = f"Unexpected simulate response type: {type(raw)}"
        raise TypeError(msg)
    return dict(raw)
