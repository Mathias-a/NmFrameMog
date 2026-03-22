#!/usr/bin/env python3
"""Run the solver pipeline against the live Astar Island API.

Usage (from the benchmark/ directory):
    uv run python scripts/run_prod.py                       # auto-detect active round
    uv run python scripts/run_prod.py --round-id <uuid>     # specify round
    uv run python scripts/run_prod.py --dry-run              # solve but don't submit

Environment:
    ACCESS_TOKEN or .env file at repo root (worktree-2/.env) with:
        access_token=<jwt>

The script will:
  1. Connect to the production API via ProdAdapter.
  2. Determine the active round (or use --round-id).
  3. Run the solver pipeline to produce prediction tensors.
  4. Apply safe_prediction() to floor zeros and renormalize.
  5. Submit all 5 seed tensors (unless --dry-run).
  6. Print a summary of queries used and runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo paths for direct invocations outside uv run.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR / "src"))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from astar_twin.scoring import safe_prediction  # noqa: E402
from astar_twin.solver.adapters.prod import (  # noqa: E402
    DEFAULT_BASE_URL,
    ProdAdapter,
    _resolve_token,
)
from astar_twin.solver.pipeline import solve  # noqa: E402


def _find_active_round_id(token: str, base_url: str) -> str:
    """Query the rounds list and return the first active round's ID."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{base_url}/astar-island/rounds")
        resp.raise_for_status()
        rounds: list[dict[str, object]] = resp.json()

    active = [r for r in rounds if r.get("status") == "active"]
    if not active:
        print("No active round found. Available rounds:")
        for r in rounds:
            print(f"  Round {r.get('round_number')} — {r.get('status')} — {r.get('id')}")
        sys.exit(1)

    round_id = str(active[0]["id"])
    round_number = active[0].get("round_number", "?")
    print(f"Active round: #{round_number} ({round_id})")
    return round_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run solver against production API")
    parser.add_argument(
        "--round-id",
        type=str,
        default=None,
        help="Round ID to solve. If omitted, auto-detects the active round.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the solver but do NOT submit predictions.",
    )
    parser.add_argument(
        "--sims-per-seed",
        type=int,
        default=64,
        help="Monte Carlo simulations per seed for final prediction (default: 64).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for deterministic execution (default: 42).",
    )
    args = parser.parse_args()

    token = _resolve_token()
    print(f"Token loaded (first 20 chars): {token[:20]}...")

    round_id = args.round_id or _find_active_round_id(token, DEFAULT_BASE_URL)

    with ProdAdapter(token=token) as adapter:
        # Check budget before starting
        used, max_q = adapter.get_budget(round_id)
        print(f"Budget: {used}/{max_q} queries used")

        if used >= max_q:
            print("Budget exhausted — cannot run solver.")
            sys.exit(1)

        print(
            f"\nRunning solver (sims_per_seed={args.sims_per_seed}, base_seed={args.base_seed})..."
        )
        result = solve(
            adapter,
            round_id,
            sims_per_seed=args.sims_per_seed,
            base_seed=args.base_seed,
        )

        print(f"\nSolver finished in {result.runtime_seconds:.1f}s")
        print(f"Queries used: {result.total_queries_used}")
        print(f"Final ESS: {result.final_ess:.2f}")
        print(f"Contradiction triggered: {result.contradiction_triggered}")
        print(f"Tensors produced: {len(result.tensors)}")

        if args.dry_run:
            print("\n--dry-run: skipping submission.")
            for i, tensor in enumerate(result.tensors):
                safe_t = safe_prediction(tensor)
                min_prob = float(np.min(safe_t))
                print(f"  Seed {i}: shape={safe_t.shape}, min_prob={min_prob:.4f}")
            return

        print("\nSubmitting predictions...")
        for seed_idx, tensor in enumerate(result.tensors):
            safe_t = safe_prediction(tensor)
            min_prob = float(np.min(safe_t))
            print(f"  Seed {seed_idx}: shape={safe_t.shape}, min_prob={min_prob:.4f}", end="")
            resp = adapter.submit(round_id, seed_idx, safe_t)
            print(f" — {resp.status}")

        print(f"\nDone. All {len(result.tensors)} seeds submitted for round {round_id}.")


if __name__ == "__main__":
    main()
