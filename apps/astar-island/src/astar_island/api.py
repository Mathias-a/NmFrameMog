"""API client for the A* Island competition server.

Handles round discovery, historical data fetching with caching,
simulation queries, and local prediction saving.

Uses httpx for HTTP requests. Fetches the /rounds listing to discover
completed rounds (with UUID IDs), then downloads analysis data for each.
Caches raw JSON to data/analysis/ for offline use.

Note: Predictions are saved locally only — no submission to live endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray

from astar_island.types import N_SEEDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://api.ainm.no/astar-island"
NUM_SEEDS: int = N_SEEDS
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "analysis"
PREDICTIONS_DIR = Path(__file__).resolve().parents[2] / "data" / "predictions"

# Type alias: round_number -> list of (initial_grid, ground_truth) per seed
RoundData = dict[int, list[tuple[NDArray[np.int_], NDArray[np.float64]]]]


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def _get_token() -> str:
    """Read API token from environment variable or .env file at project root.

    Reads ASTAR_API_TOKEN from the environment first.
    Falls back to parsing the .env file at parents[4] from this file.

    Returns:
        The API token string.

    Raises:
        ValueError: If no token is found in either location.
    """
    token = os.environ.get("ASTAR_API_TOKEN", "")
    if not token:
        env_path = Path(__file__).resolve().parents[4] / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("ASTAR_API_TOKEN="):
                    token = stripped.split("=", 1)[1].strip().strip("\"'")
                    break
    if not token:
        msg = (
            "ASTAR_API_TOKEN not found. Set it as an environment variable "
            "or add ASTAR_API_TOKEN=<token> to the .env file at the project root."
        )
        raise ValueError(msg)
    return token


def _make_client() -> httpx.Client:
    """Create an authenticated httpx.Client with retry transport.

    Returns:
        Configured httpx.Client with Bearer token auth, 60s timeout, 3 retries.
    """
    token = _get_token()
    return httpx.Client(
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0,
        transport=httpx.HTTPTransport(retries=3),
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(round_number: int, seed_index: int) -> Path:
    """Return the cache file path for a given round number and seed index.

    Args:
        round_number: Integer round number (not UUID).
        seed_index: Seed index (0 to NUM_SEEDS-1).

    Returns:
        Path to the JSON cache file.
    """
    return CACHE_DIR / f"round_{round_number}_seed_{seed_index}.json"


# ---------------------------------------------------------------------------
# Round discovery
# ---------------------------------------------------------------------------


def fetch_completed_rounds(client: httpx.Client) -> list[tuple[int, str]]:
    """Fetch the rounds listing and return completed rounds.

    Args:
        client: Authenticated httpx.Client.

    Returns:
        List of (round_number, uuid) tuples sorted ascending by round_number.
    """
    url = f"{API_BASE}/rounds"
    resp = client.get(url)
    resp.raise_for_status()
    raw: object = resp.json()
    if not isinstance(raw, list):
        logger.warning("Unexpected /rounds response type: %s", type(raw))
        return []

    completed: list[tuple[int, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        status: object = item.get("status")
        if not isinstance(status, str) or status != "completed":
            continue
        round_number: object = item.get("round_number")
        round_id: object = item.get("id")
        if not isinstance(round_number, int) or not isinstance(round_id, str):
            logger.warning("Skipping malformed round entry: %s", item)
            continue
        completed.append((round_number, round_id))

    completed.sort(key=lambda x: x[0])
    logger.info("Found %d completed rounds from API", len(completed))
    return completed


def get_active_round(client: httpx.Client) -> dict[str, object] | None:
    """Fetch the currently active round, if any.

    Args:
        client: Authenticated httpx.Client.

    Returns:
        The active round dict from the API, or None if no active round exists.
    """
    url = f"{API_BASE}/rounds"
    resp = client.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    raw: object = resp.json()
    if not isinstance(raw, list):
        return None

    for item in raw:
        if not isinstance(item, dict):
            continue
        status: object = item.get("status")
        if isinstance(status, str) and status == "active":
            result: dict[str, object] = dict(item)
            return result

    return None


def get_round_detail(client: httpx.Client, round_id: str) -> dict[str, object]:
    """Fetch detailed information for a specific round by UUID.

    Args:
        client: Authenticated httpx.Client.
        round_id: The UUID string of the round.

    Returns:
        The round detail dict from the API.

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
    """
    url = f"{API_BASE}/rounds/{round_id}"
    resp = client.get(url)
    resp.raise_for_status()
    raw: object = resp.json()
    if not isinstance(raw, dict):
        msg = f"Unexpected response type from /rounds/{round_id}: {type(raw)}"
        raise TypeError(msg)
    result: dict[str, object] = dict(raw)
    return result


# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------


def _fetch_seed(
    client: httpx.Client,
    round_uuid: str,
    seed_index: int,
    round_number: int,
) -> dict[str, object] | None:
    """Fetch analysis data for one seed from the API.

    Args:
        client: Authenticated httpx.Client.
        round_uuid: The UUID of the round.
        seed_index: The seed index to fetch.
        round_number: The integer round number (for logging only).

    Returns:
        The analysis dict, or None on 404 or other HTTP errors.
    """
    url = f"{API_BASE}/analysis/{round_uuid}/{seed_index}"
    resp = client.get(url)
    if resp.status_code == 404:
        logger.debug("404 for round %d seed %d — not found", round_number, seed_index)
        return None
    if resp.status_code >= 400:
        logger.warning(
            "HTTP %d for round %d (seed %d), skipping",
            resp.status_code,
            round_number,
            seed_index,
        )
        return None
    raw: object = resp.json()
    if not isinstance(raw, dict):
        logger.warning(
            "Unexpected analysis response type for round %d seed %d: %s",
            round_number,
            seed_index,
            type(raw),
        )
        return None
    result: Mapping[str, object] = raw
    return dict(result)


def download_all(*, force: bool = False) -> None:
    """Download and cache all available historical round data.

    Fetches the /rounds listing to discover completed rounds (with UUID
    IDs), then downloads analysis data for each round/seed combination.
    Caches raw JSON to CACHE_DIR using round_number (not UUID) in filenames.

    Args:
        force: If True, re-download even if already cached.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with _make_client() as client:
        completed = fetch_completed_rounds(client)

        for round_number, round_uuid in completed:
            all_cached = all(
                _cache_path(round_number, s).exists() for s in range(NUM_SEEDS)
            )
            if not force and all_cached:
                logger.debug("Round %d already fully cached, skipping", round_number)
                continue

            for seed_idx in range(NUM_SEEDS):
                cache_file = _cache_path(round_number, seed_idx)
                if not force and cache_file.exists():
                    logger.debug(
                        "Round %d seed %d already cached", round_number, seed_idx
                    )
                    continue

                data = _fetch_seed(client, round_uuid, seed_idx, round_number)
                if data is not None:
                    cache_file.write_text(json.dumps(data))
                    logger.info("Downloaded round %d seed %d", round_number, seed_idx)

    logger.info("Download complete. Cache dir: %s", CACHE_DIR)


# ---------------------------------------------------------------------------
# Load from cache
# ---------------------------------------------------------------------------


def load_all_rounds() -> RoundData:
    """Load all cached round data into memory.

    Discovers round numbers from cached filenames, then loads each
    (initial_grid, ground_truth) pair per seed into numpy arrays.

    Returns:
        Dict mapping round_number -> list of (initial_grid, ground_truth)
        tuples, one per seed. Rounds with no valid seeds are omitted.
    """
    if not CACHE_DIR.exists():
        logger.warning(
            "Cache dir %s does not exist. Run download_all() first.", CACHE_DIR
        )
        return {}

    # Discover round numbers from filenames: round_{id}_seed_{idx}.json
    round_numbers: set[int] = set()
    for path in sorted(CACHE_DIR.glob("round_*_seed_*.json")):
        parts = path.stem.split("_")
        if len(parts) == 4 and parts[0] == "round" and parts[2] == "seed":  # noqa: PLR2004
            try:
                round_numbers.add(int(parts[1]))
            except ValueError:
                logger.debug("Skipping unrecognised cache file: %s", path)

    rounds: RoundData = {}
    for rnum in sorted(round_numbers):
        seeds: list[tuple[NDArray[np.int_], NDArray[np.float64]]] = []
        for seed_idx in range(NUM_SEEDS):
            cache_file = _cache_path(rnum, seed_idx)
            if not cache_file.exists():
                continue
            with cache_file.open() as fh:
                raw_data: object = json.load(fh)
            if not isinstance(raw_data, dict):
                logger.warning("Unexpected cache format in %s, skipping", cache_file)
                continue
            data: Mapping[str, object] = raw_data
            initial_grid: NDArray[np.int_] = np.array(
                data["initial_grid"], dtype=np.int_
            )
            ground_truth: NDArray[np.float64] = np.array(
                data["ground_truth"], dtype=np.float64
            )
            seeds.append((initial_grid, ground_truth))

        if seeds:
            rounds[rnum] = seeds

    logger.info("Loaded %d rounds from cache", len(rounds))
    return rounds


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

    POSTs to {API_BASE}/simulate with viewport coordinates.
    Retries up to 3 times with exponential backoff on HTTP 429.

    Args:
        client: Authenticated httpx.Client.
        round_id: The UUID of the round.
        seed_index: The seed index to simulate.
        vx: Viewport top-left x (column) coordinate.
        vy: Viewport top-left y (row) coordinate.
        viewport_size: Width and height of the viewport (default: 15).

    Returns:
        The simulation response dict with 'grid', 'viewport', 'settlements'.

    Raises:
        httpx.HTTPStatusError: After exhausting retries or on non-429/non-2xx.
    """
    url = f"{API_BASE}/simulate"
    body: dict[str, object] = {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": vx,
        "viewport_y": vy,
        "viewport_w": viewport_size,
        "viewport_h": viewport_size,
    }

    max_retries = 3
    for attempt in range(max_retries + 1):
        resp = client.post(url, json=body)
        if resp.status_code == 429:  # noqa: PLR2004
            if attempt < max_retries:
                wait = 2.0**attempt
                logger.warning(
                    "Rate limited (429) on simulate attempt %d/%d, retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue
            # Exhausted retries
            resp.raise_for_status()
        if resp.status_code == 404:  # noqa: PLR2004
            logger.warning("404 on simulate for round %s seed %d", round_id, seed_index)
            resp.raise_for_status()
        resp.raise_for_status()
        raw: object = resp.json()
        if not isinstance(raw, dict):
            msg = f"Unexpected simulate response type: {type(raw)}"
            raise TypeError(msg)
        result: dict[str, object] = dict(raw)
        return result

    # Unreachable — loop always returns or raises; satisfies type checker
    msg = "simulate_query: exhausted retries without returning"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Prediction saving and submission
# ---------------------------------------------------------------------------


def save_prediction(
    round_id: str,
    seed_index: int,
    prediction_tensor: NDArray[np.float64],
) -> Path:
    """Save a prediction tensor to local disk as JSON.

    Writes the prediction to data/predictions/{round_id}_seed{seed_index}.json
    as a nested list (JSON-serialisable). Does NOT submit to any API endpoint.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{round_id}_seed{seed_index}.json"
    nested: list[object] = prediction_tensor.tolist()
    out_path.write_text(json.dumps(nested))
    logger.info("Saved prediction to %s", out_path)
    return out_path


def submit_prediction(
    client: httpx.Client,
    round_id: str,
    seed_index: int,
    prediction_tensor: NDArray[np.float64],
) -> dict[str, object]:
    """Submit a prediction tensor to the competition API.

    POST /astar-island/submit with {round_id, seed_index, prediction}.
    Retries up to 5 times with linear backoff on HTTP 429.
    """
    pred_list: list[list[list[float]]] = prediction_tensor.tolist()
    body: dict[str, object] = {
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": pred_list,
    }

    url = f"{API_BASE}/submit"
    max_retries = 5
    for attempt in range(max_retries + 1):
        resp = client.post(url, json=body, timeout=60.0)
        if resp.status_code == 429:  # noqa: PLR2004
            if attempt < max_retries:
                wait = 3.0 * (attempt + 1)
                logger.warning(
                    "Submit rate limited (429) attempt %d/%d, retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
        resp.raise_for_status()
        raw: object = resp.json()
        if not isinstance(raw, dict):
            msg = f"Unexpected submit response type: {type(raw)}"
            raise TypeError(msg)
        result: dict[str, object] = dict(raw)
        logger.info(
            "Submitted prediction for round %s seed %d: %s",
            round_id,
            seed_index,
            result,
        )
        return result

    msg = "submit_prediction: exhausted retries without returning"
    raise RuntimeError(msg)
