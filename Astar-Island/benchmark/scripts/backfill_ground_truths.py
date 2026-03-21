#!/usr/bin/env python3
"""Recompute and persist ground truth tensors for all real-round fixtures.

Run this once (or whenever you want higher-quality ground truths) to backfill
all fixture JSON files with tensors derived from many Monte Carlo simulations.
Fixtures that already have ground_truths will be recomputed and overwritten if
``--force`` is passed; otherwise they are skipped.

Usage (from the benchmark/ directory)::

    uv run python scripts/backfill_ground_truths.py
    uv run python scripts/backfill_ground_truths.py --n-runs 500 --force

The script writes each fixture back to its original ``round_detail.json`` file
via ``write_fixture()`` so that ``run_benchmark_suite`` and
``run_multi_fixture_suite`` will pick up the cached tensors automatically on
subsequent runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo paths before any local imports so the script can be run from
# anywhere inside the worktree.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DATA_DIR = BENCHMARK_DIR / "data" / "rounds"

# Add the src package to sys.path (uv run already handles this via pyproject,
# but keep the explicit path for direct invocations).
sys.path.insert(0, str(BENCHMARK_DIR / "src"))

from astar_twin.data.loaders import list_fixtures  # noqa: E402
from astar_twin.fixture_prep.ground_truth import compute_and_attach_ground_truths  # noqa: E402
from astar_twin.fixture_prep.writer import write_fixture  # noqa: E402

DEFAULT_N_RUNS = 300
DEFAULT_BASE_SEED = 42


def _is_real_round(fixture_id: str) -> bool:
    return not fixture_id.startswith("test-")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill ground truth tensors for all real-round fixtures."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help=f"Directory containing rounds/ subdirectory (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help=f"Number of MC simulation runs per seed (default: {DEFAULT_N_RUNS}).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=f"Base random seed for MC runs (default: {DEFAULT_BASE_SEED}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if ground_truths are already present in the fixture.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    # list_fixtures expects the parent of the rounds/ directory
    parent_dir = data_dir.parent if data_dir.name == "rounds" else data_dir

    all_fixtures = list_fixtures(parent_dir)
    real_fixtures = [f for f in all_fixtures if _is_real_round(f.id)]
    real_fixtures.sort(key=lambda f: f.round_number)

    if not real_fixtures:
        print(f"No real-round fixtures found under {parent_dir}")
        sys.exit(1)

    print(f"Found {len(real_fixtures)} real-round fixtures.")

    skipped = 0
    updated = 0

    for fixture in real_fixtures:
        fixture_path = parent_dir / "rounds" / fixture.id / "round_detail.json"

        if fixture.ground_truths is not None and not args.force:
            print(
                f"  Round {fixture.round_number:>2} ({fixture.id[:8]}…): "
                f"already has ground_truths ({len(fixture.ground_truths)} seeds) — skipping. "
                f"Use --force to recompute."
            )
            skipped += 1
            continue

        action = "Would recompute" if args.dry_run else "Recomputing"
        print(
            f"  Round {fixture.round_number:>2} ({fixture.id[:8]}…): "
            f"{action} ground_truths with {args.n_runs} MC runs per seed…",
            flush=True,
        )

        if args.dry_run:
            continue

        if fixture.simulation_params is None:
            print(
                f"    WARNING: fixture has no simulation_params — cannot compute ground truths. Skipping."
            )
            skipped += 1
            continue

        updated_fixture = compute_and_attach_ground_truths(
            fixture,
            n_runs=args.n_runs,
            base_seed=args.base_seed,
        )
        write_fixture(updated_fixture, fixture_path)
        print(f"    Written to {fixture_path}")
        updated += 1

    print()
    if args.dry_run:
        print(f"Dry run complete. Would update {len(real_fixtures) - skipped} fixtures.")
    else:
        print(f"Done. Updated {updated} fixtures, skipped {skipped}.")


if __name__ == "__main__":
    main()
