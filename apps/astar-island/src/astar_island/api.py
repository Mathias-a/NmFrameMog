"""Download and cache historical round data from the A* Island API.

Uses httpx for HTTP requests. Fetches the /rounds listing to discover
completed rounds (with UUID IDs), then downloads analysis data for each.
Caches raw JSON to data/analysis/ for offline use.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = "https://api.ainm.no/astar-island"
NUM_SEEDS = 5
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "analysis"

# Type alias: round_id -> list of (initial_grid, ground_truth) per seed
RoundData = dict[int, list[tuple[NDArray[np.int_], NDArray[np.float64]]]]


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------


def _get_token() -> str:
    """Read API token from environment or .env file."""
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
            "or in the .env file at the project root."
        )
        raise ValueError(msg)
    return token


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _cache_path(round_id: int, seed_index: int) -> Path:
    """Return cache file path for a given round/seed."""
    return CACHE_DIR / f"round_{round_id}_seed_{seed_index}.json"


def _fetch_seed(
    client: httpx.Client,
    round_uuid: str,
    seed_index: int,
    round_number: int,
) -> dict[str, object] | None:
    """Fetch analysis data for one seed, returning None on 404/error."""
    url = f"{API_BASE}/analysis/{round_uuid}/{seed_index}"
    resp = client.get(url)
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        logger.warning(
            "HTTP %d for round %d (seed %d), skipping",
            resp.status_code,
            round_number,
            seed_index,
        )
        return None
    raw_json: object = resp.json()
    assert isinstance(raw_json, dict)
    result: Mapping[str, object] = raw_json
    return dict(result)


def _fetch_completed_rounds(
    client: httpx.Client,
) -> list[tuple[int, str]]:
    """Fetch the rounds listing and return completed rounds as (number, uuid).

    Returns list sorted by round_number ascending.
    """
    url = f"{API_BASE}/rounds"
    resp = client.get(url)
    resp.raise_for_status()
    raw: object = resp.json()
    assert isinstance(raw, list)

    completed: list[tuple[int, str]] = []
    for item in raw:
        assert isinstance(item, dict)
        status: object = item.get("status")
        if not isinstance(status, str) or status != "completed":
            continue
        round_number: object = item.get("round_number")
        round_id: object = item.get("id")
        assert isinstance(round_number, int)
        assert isinstance(round_id, str)
        completed.append((round_number, round_id))

    completed.sort(key=lambda x: x[0])
    logger.info("Found %d completed rounds from API", len(completed))
    return completed


def download_all(*, force: bool = False) -> None:
    """Download and cache all available historical round data.

    Fetches the /rounds listing to discover completed rounds (with UUID
    IDs), then downloads analysis data for each round/seed combination.

    Args:
        force: If True, re-download even if cached.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    token = _get_token()
    headers = {"Authorization": f"Bearer {token}"}

    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(
        headers=headers,
        timeout=60.0,
        transport=transport,
    ) as client:
        completed = _fetch_completed_rounds(client)

        for round_number, round_uuid in completed:
            all_cached = all(
                _cache_path(round_number, s).exists() for s in range(NUM_SEEDS)
            )
            if not force and all_cached:
                logger.debug("Round %d already cached, skipping", round_number)
                continue

            for seed_idx in range(NUM_SEEDS):
                cache_s = _cache_path(round_number, seed_idx)
                if not force and cache_s.exists():
                    logger.debug(
                        "Round %d seed %d already cached", round_number, seed_idx
                    )
                    continue
                data_s = _fetch_seed(client, round_uuid, seed_idx, round_number)
                if data_s is not None:
                    cache_s.write_text(json.dumps(data_s))
                    logger.info("Downloaded round %d seed %d", round_number, seed_idx)

    logger.info("Download complete. Cache dir: %s", CACHE_DIR)


# ---------------------------------------------------------------------------
# Active round
# ---------------------------------------------------------------------------


def fetch_active_round() -> tuple[str, int, list[NDArray[np.int_]]]:
    """Fetch the currently active round and its initial grids.

    Returns:
        (round_uuid, round_number, grids) where grids is a list of 5
        initial_grid arrays, one per seed.

    Raises:
        RuntimeError: If no active round is found.
    """
    token = _get_token()
    headers = {"Authorization": f"Bearer {token}"}

    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(
        headers=headers,
        timeout=60.0,
        transport=transport,
    ) as client:
        # Find the active round
        resp = client.get(f"{API_BASE}/rounds")
        resp.raise_for_status()
        raw_rounds: object = resp.json()
        assert isinstance(raw_rounds, list)

        active_uuid: str | None = None
        active_number: int = -1
        for item in raw_rounds:
            assert isinstance(item, dict)
            status: object = item.get("status")
            if isinstance(status, str) and status == "active":
                uuid_val: object = item.get("id")
                num_val: object = item.get("round_number")
                assert isinstance(uuid_val, str)
                assert isinstance(num_val, int)
                active_uuid = uuid_val
                active_number = num_val
                break

        if active_uuid is None:
            msg = "No active round found."
            raise RuntimeError(msg)

        logger.info("Active round: %d (uuid=%s)", active_number, active_uuid)

        # Fetch round details to get initial_states
        resp2 = client.get(f"{API_BASE}/rounds/{active_uuid}")
        resp2.raise_for_status()
        raw_detail: object = resp2.json()
        assert isinstance(raw_detail, dict)

        raw_states: object = raw_detail.get("initial_states")
        assert isinstance(raw_states, list)

        grids: list[NDArray[np.int_]] = []
        for state in raw_states:
            assert isinstance(state, dict)
            raw_grid: object = state.get("grid")
            assert isinstance(raw_grid, list)
            grid = np.array(raw_grid, dtype=np.int_)
            grids.append(grid)

        logger.info("Fetched %d seed grids for round %d", len(grids), active_number)

    return active_uuid, active_number, grids


# ---------------------------------------------------------------------------
# Submit prediction
# ---------------------------------------------------------------------------


def submit_prediction(
    round_uuid: str,
    seed_index: int,
    prediction: NDArray[np.float64],
) -> float | None:
    """Submit a prediction for a specific round/seed.

    Args:
        round_uuid: UUID of the round.
        seed_index: Seed index (0-4).
        prediction: (H, W, 6) probability tensor.

    Returns:
        Score if returned by the API, else None.

    Raises:
        RuntimeError: If submission fails.
    """
    token = _get_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Convert numpy array to nested list for JSON serialization
    raw_list: object = prediction.tolist()
    assert isinstance(raw_list, list)
    pred_list: list[list[list[float]]] = []
    for row_raw in raw_list:
        assert isinstance(row_raw, list)
        row_out: list[list[float]] = []
        for cell_raw in row_raw:
            assert isinstance(cell_raw, list)
            row_out.append([float(v) for v in cell_raw])
        pred_list.append(row_out)

    body: dict[str, object] = {
        "round_id": round_uuid,
        "seed_index": seed_index,
        "prediction": pred_list,
    }

    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(
        headers=headers,
        timeout=120.0,
        transport=transport,
    ) as client:
        resp = client.post(f"{API_BASE}/submit", json=body)
        if resp.status_code >= 400:
            detail: str = resp.text
            msg = f"Submission failed (HTTP {resp.status_code}): {detail}"
            raise RuntimeError(msg)

        raw_resp: object = resp.json()
        logger.info("Submitted seed %d: %s", seed_index, raw_resp)

        if isinstance(raw_resp, dict):
            score_val: object = raw_resp.get("score")
            if isinstance(score_val, (float, int)):
                return float(score_val)

    return None


# ---------------------------------------------------------------------------
# Simulation queries
# ---------------------------------------------------------------------------

VP_SIZE = 15


def plan_viewports(
    grid_h: int = 40,
    grid_w: int = 40,
    vp_size: int = VP_SIZE,
) -> list[tuple[int, int]]:
    """Generate 3x3 tiling positions covering the dynamic interior.

    Starts at (1,1) to skip ocean border. Tiles without gaps, allowing
    overlap at edges to ensure full coverage.
    """
    positions: list[tuple[int, int]] = []
    y = 1
    while y + vp_size <= grid_h:
        x = 1
        while x + vp_size <= grid_w:
            positions.append((x, y))
            x += vp_size
        if x < grid_w - 1:
            positions.append((grid_w - vp_size, y))
        y += vp_size
    if y < grid_h - 1:
        x = 1
        while x + vp_size <= grid_w:
            positions.append((x, grid_h - vp_size))
            x += vp_size
        if x < grid_w - 1:
            positions.append((grid_w - vp_size, grid_h - vp_size))
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int]] = []
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            deduped.append(pos)
    return deduped


def _accumulate_viewport(
    vp_grid: NDArray[np.int_],
    vx: int,
    vy: int,
    accum: NDArray[np.float64],
    counts: NDArray[np.int_],
    grid_h: int,
    grid_w: int,
) -> None:
    """Accumulate one-hot observations from a viewport into full-grid arrays."""
    from astar_island.prob import NUM_CLASSES, TERRAIN_TO_CLASS

    vp_h: int = int(vp_grid.shape[0])
    vp_w: int = int(vp_grid.shape[1])
    ey: int = min(vy + vp_h, grid_h)
    ex: int = min(vx + vp_w, grid_w)
    actual_h: int = ey - vy
    actual_w: int = ex - vx

    patch: NDArray[np.int_] = vp_grid[:actual_h, :actual_w]
    flat_codes: NDArray[np.int_] = patch.ravel()

    max_code: int = max(
        int(np.max(flat_codes).item()) + 1, max(TERRAIN_TO_CLASS.keys()) + 1
    )
    lookup: NDArray[np.int_] = np.zeros(max_code, dtype=np.int_)
    for code, cls in TERRAIN_TO_CLASS.items():
        if code < max_code:
            lookup[code] = np.int_(cls)

    unknown_mask: NDArray[np.bool_] = np.ones(max_code, dtype=np.bool_)
    for code in TERRAIN_TO_CLASS:
        if code < max_code:
            unknown_mask[code] = False

    has_unknown: bool = bool(np.any(unknown_mask[flat_codes]))
    if has_unknown:
        unique_codes: NDArray[np.int_] = np.unique(flat_codes).astype(np.int_)
        for idx in range(int(unique_codes.shape[0])):
            c: int = int(unique_codes.item(idx))
            if c >= max_code or bool(unknown_mask.item(c)):
                logger.warning("Unknown terrain code %d in viewport", c)

    class_indices: NDArray[np.int_] = lookup[flat_codes]
    onehot: NDArray[np.float64] = np.eye(NUM_CLASSES, dtype=np.float64)[class_indices]
    onehot_3d: NDArray[np.float64] = onehot.reshape(actual_h, actual_w, NUM_CLASSES)

    accum[vy:ey, vx:ex, :] += onehot_3d
    counts[vy:ey, vx:ex] += np.int_(1)


def _score_viewport(
    vx: int,
    vy: int,
    pred_entropy: NDArray[np.float64],
    vp_size: int = VP_SIZE,
) -> float:
    """Score a viewport by sum of predicted entropy in its cells."""
    ey = min(vy + vp_size, pred_entropy.shape[0])
    ex = min(vx + vp_size, pred_entropy.shape[1])
    return float(np.sum(pred_entropy[vy:ey, vx:ex]))


def run_queries_for_seed(
    client: httpx.Client,
    round_uuid: str,
    seed_index: int,
    viewports: list[tuple[int, int]],
    grid_h: int = 40,
    grid_w: int = 40,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Run simulation queries for one seed on specified viewports.

    Returns:
        (accum, counts) where:
        - accum: (H, W, 6) accumulated one-hot observations
        - counts: (H, W) number of observations per cell
    """
    from astar_island.prob import NUM_CLASSES

    accum = np.zeros((grid_h, grid_w, NUM_CLASSES), dtype=np.float64)
    counts = np.zeros((grid_h, grid_w), dtype=np.int_)

    for q, (vx, vy) in enumerate(viewports):
        body: dict[str, object] = {
            "round_id": round_uuid,
            "seed_index": seed_index,
            "viewport_x": vx,
            "viewport_y": vy,
            "viewport_w": VP_SIZE,
            "viewport_h": VP_SIZE,
        }
        try:
            resp = client.post(f"{API_BASE}/simulate", json=body)
        except httpx.HTTPError:
            logger.warning("Simulate query %d: HTTP error, skipping", q)
            continue
        if resp.status_code >= 400:
            logger.warning(
                "Simulate query %d failed (HTTP %d), skipping",
                q,
                resp.status_code,
            )
            continue

        try:
            raw: object = resp.json()
        except (ValueError, KeyError):
            logger.warning("Simulate query %d: malformed JSON, skipping", q)
            continue
        assert isinstance(raw, dict)
        raw_grid: object = raw.get("grid")
        assert isinstance(raw_grid, list)
        vp_grid = np.array(raw_grid, dtype=np.int_)
        _accumulate_viewport(vp_grid, vx, vy, accum, counts, grid_h, grid_w)

        used: object = raw.get("queries_used", "?")
        logger.debug(
            "Query %d/%d: seed %d viewport (%d,%d), budget %s",
            q + 1,
            len(viewports),
            seed_index,
            vx,
            vy,
            used,
        )

    observed = int(np.sum(counts > 0))
    logger.info(
        "Seed %d: %d queries, %d cells observed (%.0f%% coverage)",
        seed_index,
        len(viewports),
        observed,
        100.0 * observed / (grid_h * grid_w),
    )
    return accum, counts


def query_all_seeds(
    round_uuid: str,
    grids: list[NDArray[np.int_]],
    predictions: list[NDArray[np.float64]],
    total_budget: int = 50,
) -> list[tuple[NDArray[np.float64], NDArray[np.int_]]]:
    """Run simulation queries across all seeds using entropy-ranked viewports.

    Ranks all candidate viewports (9 per seed × 5 seeds = 45) by predicted
    entropy and selects the top `total_budget` windows.
    """
    from astar_island.prob import NUM_CLASSES, entropy

    num_seeds = len(grids)
    viewports_per_seed = plan_viewports()

    scored: list[tuple[float, int, int, int]] = []
    for seed_idx in range(num_seeds):
        pred_ent = entropy(predictions[seed_idx])
        for vp_idx, (vx, vy) in enumerate(viewports_per_seed):
            score = _score_viewport(vx, vy, pred_ent)
            scored.append((score, seed_idx, vp_idx, vp_idx))

    scored.sort(key=lambda t: t[0], reverse=True)
    selected = scored[:total_budget]

    seed_viewports: dict[int, list[tuple[int, int]]] = {i: [] for i in range(num_seeds)}
    for _, seed_idx, vp_idx, _ in selected:
        seed_viewports[seed_idx].append(viewports_per_seed[vp_idx])

    for seed_idx, vps in seed_viewports.items():
        logger.info("Seed %d: %d viewports selected", seed_idx, len(vps))

    token = _get_token()
    headers = {"Authorization": f"Bearer {token}"}
    transport = httpx.HTTPTransport(retries=3)

    results: list[tuple[NDArray[np.float64], NDArray[np.int_]]] = []
    with httpx.Client(
        headers=headers,
        timeout=60.0,
        transport=transport,
    ) as client:
        for seed_idx in range(num_seeds):
            vps = seed_viewports[seed_idx]
            if not vps:
                g_shape = grids[seed_idx].shape
                accum_empty = np.zeros(
                    (g_shape[0], g_shape[1], NUM_CLASSES), dtype=np.float64
                )
                counts_empty = np.zeros((g_shape[0], g_shape[1]), dtype=np.int_)
                results.append((accum_empty, counts_empty))
                continue
            accum, counts = run_queries_for_seed(client, round_uuid, seed_idx, vps)
            results.append((accum, counts))

    return results


# ---------------------------------------------------------------------------
# Load from cache
# ---------------------------------------------------------------------------


def load_all_rounds() -> RoundData:
    """Load all cached rounds into memory.

    Returns:
        Dict mapping round_id to list of (initial_grid, ground_truth) tuples.
    """
    if not CACHE_DIR.exists():
        logger.warning("Cache dir %s does not exist. Run download first.", CACHE_DIR)
        return {}

    rounds: RoundData = {}
    # Discover round IDs from filenames
    round_ids: set[int] = set()
    for path in sorted(CACHE_DIR.glob("round_*_seed_*.json")):
        parts = path.stem.split("_")
        # round_{id}_seed_{idx}
        if len(parts) == 4 and parts[0] == "round" and parts[2] == "seed":
            round_ids.add(int(parts[1]))

    for rid in sorted(round_ids):
        seeds: list[tuple[NDArray[np.int_], NDArray[np.float64]]] = []
        for seed_idx in range(NUM_SEEDS):
            cache_file = _cache_path(rid, seed_idx)
            if not cache_file.exists():
                continue
            with open(cache_file) as f:
                raw_data = json.load(f)
            assert isinstance(raw_data, dict)
            data: Mapping[str, object] = raw_data

            initial_grid = np.array(data["initial_grid"], dtype=np.int_)
            ground_truth = np.array(data["ground_truth"], dtype=np.float64)
            seeds.append((initial_grid, ground_truth))

        if seeds:
            rounds[rid] = seeds

    logger.info("Loaded %d rounds from cache", len(rounds))
    return rounds
