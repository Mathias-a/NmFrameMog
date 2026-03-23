#!/usr/bin/env python3
"""Run the best benchmark strategy against the live Astar Island API.

Uses CalibratedMCZones (the top-performing benchmark strategy) to produce
predictions locally via the digital twin, then submits them to the real API.

Usage (from the benchmark/ directory):
    uv run python scripts/run_best_strategy.py                      # auto-detect active round
    uv run python scripts/run_best_strategy.py --round-id <uuid>    # specify round
    uv run python scripts/run_best_strategy.py --dry-run             # predict but don't submit

Environment:
    ACCESS_TOKEN or .env file with: access_token=<jwt>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo paths for direct invocations outside uv run.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR / "src"))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from astar_twin.contracts.api_models import InitialState  # noqa: E402
from astar_twin.harness.budget import Budget  # noqa: E402
from astar_twin.scoring import safe_prediction  # noqa: E402
from astar_twin.solver.adapters.prod import (  # noqa: E402
    DEFAULT_BASE_URL,
    ProdAdapter,
    _resolve_token,
)
from astar_twin.strategies.calibrated_mc.strategy import CalibratedMCStrategy  # noqa: E402


def _find_active_round_id(base_url: str) -> str:
    """Query the rounds list and return the first active round's ID."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{base_url}/astar-island/rounds")
        resp.raise_for_status()
        rounds: list[dict[str, object]] = resp.json()

    active = [r for r in rounds if r.get("status") == "active"]
    if not active:
        print("No active round found. Available rounds:")
        for r in sorted(rounds, key=lambda x: x.get("round_number", 0), reverse=True)[:5]:  # type: ignore[arg-type,return-value]
            print(f"  Round {r.get('round_number')} — {r.get('status')} — {r.get('id')}")
        sys.exit(1)

    round_id = str(active[0]["id"])
    round_number = active[0].get("round_number", "?")
    print(f"Active round: #{round_number} ({round_id})")
    return round_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run best benchmark strategy against production API"
    )
    parser.add_argument(
        "--round-id",
        type=str,
        default=None,
        help="Round ID to solve. If omitted, auto-detects the active round.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate predictions but do NOT submit them.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=50,
        help="Monte Carlo simulation runs per seed (default: 50).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for deterministic execution (default: 42).",
    )
    args = parser.parse_args()

    # Resolve auth token
    token = _resolve_token()
    print(f"Token loaded (first 20 chars): {token[:20]}...")

    # Find the active round
    round_id = args.round_id or _find_active_round_id(DEFAULT_BASE_URL)

    # Create the best strategy: CalibratedMCZones
    # (zones=True, adaptive=False, variance=False — benchmark winner)
    strategy = CalibratedMCStrategy(
        use_settlement_zones=True,
        use_adaptive_blend=False,
        use_mc_variance=False,
        n_runs=args.n_runs,
    )
    print(f"Strategy: {strategy.name} (n_runs={args.n_runs})")

    with ProdAdapter(token=token) as adapter:
        # Fetch round detail (public endpoint, but adapter handles it)
        print("Fetching round detail...")
        detail = adapter.get_round_detail(round_id)
        height = detail.map_height
        width = detail.map_width
        n_seeds = detail.seeds_count
        print(f"Map: {width}x{height}, {n_seeds} seeds")

        # Generate predictions locally using the digital twin
        budget = Budget(total=50)  # Not consumed — strategy uses local simulator
        tensors: list[np.ndarray[object, np.dtype[np.float64]]] = []

        print(f"\nGenerating predictions for {n_seeds} seeds...")
        t_start = time.monotonic()

        for seed_idx in range(n_seeds):
            initial_state: InitialState = detail.initial_states[seed_idx]
            t_seed = time.monotonic()
            raw_tensor = strategy.predict(initial_state, budget, args.base_seed)
            safe_tensor = safe_prediction(raw_tensor)
            tensors.append(safe_tensor)

            min_prob = float(np.min(safe_tensor))
            max_prob = float(np.max(safe_tensor))
            elapsed = time.monotonic() - t_seed
            print(
                f"  Seed {seed_idx}: shape={safe_tensor.shape}, "
                f"min_prob={min_prob:.4f}, max_prob={max_prob:.4f}, "
                f"time={elapsed:.1f}s"
            )

        total_time = time.monotonic() - t_start
        print(f"\nAll {n_seeds} predictions generated in {total_time:.1f}s")

        if args.dry_run:
            print("\n--dry-run: skipping submission.")
            return

        # Submit predictions
        print("\nSubmitting predictions...")
        for seed_idx, tensor in enumerate(tensors):
            resp = adapter.submit(round_id, seed_idx, tensor)
            print(f"  Seed {seed_idx}: {resp.status}")
            time.sleep(0.6)  # Respect 2 req/sec rate limit

        print(f"\nDone! All {n_seeds} seeds submitted for round {round_id}.")


if __name__ == "__main__":
    main()
