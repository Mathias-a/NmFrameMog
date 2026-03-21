#!/usr/bin/env python3
"""Fetch all completed/scoring Astar Island rounds from the live API and save
them as RoundFixture JSON files under benchmark/data/rounds/{round_id}/.

Usage (from the benchmark/ directory):
    uv run python scripts/fetch_real_rounds.py
    uv run python scripts/fetch_real_rounds.py --prior-spread 0.5

Environment:
    ACCESS_TOKEN or .env file at repo root (worktree-2/.env) with:
        access_token=<jwt>

The script will:
  1. List all rounds and filter to completed/scoring.
  2. For each: fetch initial_states from GET /astar-island/rounds/{id}.
  3. For each seed: try GET /astar-island/analysis/{id}/{seed} for ground truths.
  4. Fall back to compute_and_attach_ground_truths if analysis is unavailable.
  5. Write fixture via write_fixture() to data/rounds/{round_id}/round_detail.json.

Use ``--prior-spread`` to control ``DEFAULT_PRIOR`` hyperparameter sampling for
local fallback ground-truth generation when API analysis is unavailable. The
default is ``1.0``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo paths before any local imports so the script can be run from
# anywhere inside the worktree.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parent.parent  # worktree-2/
DATA_DIR = BENCHMARK_DIR / "data" / "rounds"

# Add the src package to sys.path (uv run already handles this via pyproject,
# but keep the explicit path for direct invocations).
sys.path.insert(0, str(BENCHMARK_DIR / "src"))

import httpx  # noqa: E402

from astar_twin.contracts.api_models import AnalysisResponse, RoundSummary  # noqa: E402
from astar_twin.data.models import GroundTruthSource, ParamsSource, RoundFixture  # noqa: E402
from astar_twin.fixture_prep.ground_truth import compute_and_attach_ground_truths  # noqa: E402
from astar_twin.fixture_prep.writer import write_fixture  # noqa: E402
from astar_twin.params import SimulationParams  # noqa: E402

BASE_URL = "https://api.ainm.no"
FETCHABLE_STATUSES = {"completed", "scoring"}
GROUND_TRUTH_N_RUNS = 200  # MC runs for fallback ground truth generation
GROUND_TRUTH_BASE_SEED = 42


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def _load_token() -> str:
    """Return the API access token.

    Priority:
      1. ACCESS_TOKEN environment variable
      2. .env file at repo root (worktree-2/.env)
    """
    token = os.environ.get("ACCESS_TOKEN", "").strip()
    if token:
        return token

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("access_token="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token

    raise RuntimeError(
        "No API token found. Set ACCESS_TOKEN env var or ensure "
        f"{env_path} contains 'access_token=<jwt>'."
    )


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _list_rounds(client: httpx.Client) -> list[RoundSummary]:
    resp = client.get(f"{BASE_URL}/astar-island/rounds")
    resp.raise_for_status()
    raw: list[dict[str, object]] = resp.json()
    return [RoundSummary.model_validate(r) for r in raw]


def _get_round_detail(client: httpx.Client, round_id: str) -> dict[str, object]:
    """Return the raw round-detail JSON dict.

    The API response includes extra fields (event_date, prediction_window_minutes,
    started_at, closes_at) that the strict RoundDetail model rejects, so we
    return the raw dict and pick the fields we need at call sites.
    """
    resp = client.get(f"{BASE_URL}/astar-island/rounds/{round_id}")
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _get_analysis(
    client: httpx.Client,
    round_id: str,
    seed_index: int,
    token: str,
) -> AnalysisResponse | None:
    """Fetch analysis for one seed.  Returns None if unavailable (403, 404, etc.)."""
    try:
        resp = client.get(
            f"{BASE_URL}/astar-island/analysis/{round_id}/{seed_index}",
            headers=_auth_headers(token),
        )
        if resp.status_code in (403, 404):
            return None
        resp.raise_for_status()
        return AnalysisResponse.model_validate(resp.json())
    except httpx.HTTPStatusError as exc:
        print(f"    [warn] analysis/{seed_index} HTTP {exc.response.status_code} — skipping")
        return None


# ---------------------------------------------------------------------------
# Per-round processing
# ---------------------------------------------------------------------------


def _fetch_and_save_round(
    client: httpx.Client,
    summary: RoundSummary,
    token: str,
    prior_spread: float,
) -> Path:
    """Fetch one round, build a RoundFixture, save it, return the path."""
    print(f"  Fetching detail for round {summary.round_number} ({summary.id}) …")
    # The /rounds/{id} endpoint returns extra fields (event_date,
    # prediction_window_minutes, started_at, closes_at) that the strict
    # RoundDetail model rejects, so we work with the raw dict.
    raw = _get_round_detail(client, summary.id)

    round_id: str = str(raw["id"])
    round_number: int = int(raw["round_number"])  # type: ignore[arg-type]
    status: str = str(raw["status"])
    map_width: int = int(raw["map_width"])  # type: ignore[arg-type]
    map_height: int = int(raw["map_height"])  # type: ignore[arg-type]
    seeds_count: int = int(raw["seeds_count"])  # type: ignore[arg-type]

    # Parse initial_states via the strict model to get proper typed objects.
    from astar_twin.contracts.api_models import InitialSettlement, InitialState  # noqa: PLC0415

    initial_states: list[InitialState] = [
        InitialState(
            grid=s["grid"],  # type: ignore[arg-type]
            settlements=[
                InitialSettlement(**st)  # type: ignore[arg-type]
                for st in s["settlements"]  # type: ignore[index]
            ],
        )
        for s in raw["initial_states"]  # type: ignore[union-attr]
    ]

    # Attempt to pull ground truths from the analysis endpoint (real server MC).
    ground_truths: list[list[list[list[float]]]] | None = None
    analysis_available = True

    print("  Fetching ground truths from analysis endpoint …")
    seed_gts: list[list[list[list[float]]]] = []
    for seed_idx in range(seeds_count):
        analysis = _get_analysis(client, round_id, seed_idx, token)
        if analysis is None:
            analysis_available = False
            print(f"    seed {seed_idx}: analysis unavailable")
            break
        seed_gts.append(analysis.ground_truth)  # shape: H x W x 6
        print(
            f"    seed {seed_idx}: got ground truth "
            f"({len(analysis.ground_truth)}x"
            f"{len(analysis.ground_truth[0])}x"
            f"{len(analysis.ground_truth[0][0])})"
        )

    if analysis_available and len(seed_gts) == seeds_count:
        ground_truths = seed_gts
        print(f"  Ground truths from API: {seeds_count} seeds ✓")
    else:
        print(f"  Analysis unavailable — will compute locally ({GROUND_TRUTH_N_RUNS} MC runs) …")

    gt_source = (
        GroundTruthSource.API_ANALYSIS if ground_truths is not None else GroundTruthSource.UNKNOWN
    )

    fixture = RoundFixture(
        id=round_id,
        round_number=round_number,
        status=status,
        map_width=map_width,
        map_height=map_height,
        seeds_count=seeds_count,
        initial_states=initial_states,
        ground_truths=ground_truths,
        simulation_params=SimulationParams(),
        params_source=ParamsSource.DEFAULT_PRIOR,
        ground_truth_source=gt_source,
        event_date=summary.event_date,
        round_weight=summary.round_weight,
    )

    if ground_truths is None:
        print("  Computing local ground truths …")
        fixture = compute_and_attach_ground_truths(
            fixture,
            n_runs=GROUND_TRUTH_N_RUNS,
            base_seed=GROUND_TRUTH_BASE_SEED,
            prior_spread=prior_spread,
        )
        print("  Local ground truths computed ✓")

    out_path = DATA_DIR / round_id / "round_detail.json"
    write_fixture(fixture, out_path)
    print(f"  Saved → {out_path.relative_to(BENCHMARK_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch completed real rounds into benchmark fixtures"
    )
    parser.add_argument(
        "--prior-spread",
        type=float,
        default=1.0,
        help=(
            "Spread for DEFAULT_PRIOR hyperparameter sampling during local "
            "fallback ground-truth generation (default: 1.0)."
        ),
    )
    args = parser.parse_args()

    token = _load_token()
    print(f"Token loaded (first 20 chars): {token[:20]}…\n")

    with httpx.Client(timeout=60.0) as client:
        print("Listing rounds …")
        all_rounds = _list_rounds(client)
        print(f"  {len(all_rounds)} rounds found total")

        target_rounds = [r for r in all_rounds if r.status in FETCHABLE_STATUSES]
        print(f"  {len(target_rounds)} rounds with status in {FETCHABLE_STATUSES}\n")

        if not target_rounds:
            print("Nothing to fetch. Exiting.")
            return

        saved_paths: list[Path] = []
        for summary in sorted(target_rounds, key=lambda r: r.round_number):
            print(f"Round {summary.round_number} — {summary.status} — {summary.event_date}")
            try:
                path = _fetch_and_save_round(client, summary, token, args.prior_spread)
                saved_paths.append(path)
            except Exception as exc:
                print(f"  [ERROR] {exc}")
            print()

    if saved_paths:
        # Report the most recent round (highest round_number among saved)
        most_recent_id = sorted(target_rounds, key=lambda r: r.round_number)[-1].id
        print(f"Done. {len(saved_paths)} fixture(s) saved.")
        print(f"Most recent real round ID: {most_recent_id}")
        print("\nUpdate AGENTS.md benchmark example to use:")
        print(f"    load_fixture(Path('data/rounds/{most_recent_id}'))")
    else:
        print("No fixtures saved.")


if __name__ == "__main__":
    main()
