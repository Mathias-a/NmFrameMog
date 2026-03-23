"""Compare calibrated solver against strategy baselines on real round fixtures.

Usage:
    cd Astar-Island/benchmark
    uv run python scripts/benchmark_calibration.py [--fast]

Runs:
  1. Four registered strategies (uniform, initial_prior, filter_baseline, mc_oracle)
  2. The calibrated solver pipeline (with improved likelihood/posterior)
  3. Scores all outputs against the same ground truths and prints comparison.

Flags:
  --fast   Use very small parameters for quick feasibility check (~3-5min).
           Default mode uses moderate parameters (~15-20min).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import RoundFixture
from astar_twin.harness.budget import Budget
from astar_twin.scoring import compute_score, safe_prediction
from astar_twin.solver.adapters.benchmark import BenchmarkAdapter
from astar_twin.solver.eval.run_benchmark_suite import load_or_compute_ground_truths
from astar_twin.solver.pipeline import solve
from astar_twin.strategies import REGISTRY


def log(msg: str) -> None:
    """Flush-safe logging for piped output."""
    print(msg, flush=True)


def run_strategies(
    fixture: RoundFixture,
    ground_truths: list[NDArray[np.float64]],
    base_seed: int = 42,
) -> dict[str, dict[str, float | list[float]]]:
    """Run all registered strategies and score them."""
    results: dict[str, dict[str, float | list[float]]] = {}

    for name, cls in REGISTRY.items():
        t0 = time.monotonic()
        strategy = cls()
        budget = Budget(total=50)
        seed_scores: list[float] = []

        for seed_idx in range(fixture.seeds_count):
            initial_state = fixture.initial_states[seed_idx]
            raw = strategy.predict(
                initial_state=initial_state,
                budget=budget,
                base_seed=base_seed,
            )
            pred = safe_prediction(raw)
            score = float(compute_score(ground_truths[seed_idx], pred))
            seed_scores.append(score)

        elapsed = time.monotonic() - t0
        mean = float(np.mean(seed_scores))
        results[name] = {
            "mean": mean,
            "per_seed": seed_scores,
            "runtime_s": round(elapsed, 2),
        }

    return results


def run_solver_single(
    fixture: RoundFixture,
    ground_truths: list[NDArray[np.float64]],
    n_particles: int,
    n_inner_runs: int,
    sims_per_seed: int,
    base_seed: int = 0,
) -> dict[str, float | list[float]]:
    """Run the calibrated solver pipeline ONCE and score against ground truths."""
    warnings.filterwarnings("ignore")

    adapter = BenchmarkAdapter(fixture, n_mc_runs=3, sim_seed_offset=0)
    t0 = time.monotonic()
    result = solve(
        adapter,
        fixture.id,
        n_particles=n_particles,
        n_inner_runs=n_inner_runs,
        sims_per_seed=sims_per_seed,
        base_seed=base_seed,
    )
    elapsed = time.monotonic() - t0

    per_seed = [
        float(compute_score(gt, safe_prediction(t))) for gt, t in zip(ground_truths, result.tensors)
    ]
    mean = float(np.mean(per_seed))

    log(
        f"  Solver done: mean={mean:.2f}, queries={result.total_queries_used}, "
        f"ess={result.final_ess:.1f}, time={elapsed:.1f}s"
    )

    return {
        "mean": mean,
        "per_seed": per_seed,
        "runtime_s": round(elapsed, 2),
        "queries": result.total_queries_used,
        "ess": result.final_ess,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark calibrated solver vs baselines")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Minimal params for quick feasibility check",
    )
    parser.add_argument(
        "--round-id",
        default="cc5442dd-bc5d-418b-911b-7eb960cb0390",
        help="Round fixture ID",
    )
    args = parser.parse_args()

    if args.fast:
        n_particles = 4
        n_inner_runs = 2
        sims_per_seed = 8
        log("=== FAST MODE (4p/2ir/8sps) ===\n")
    else:
        n_particles = 8
        n_inner_runs = 3
        sims_per_seed = 24
        log("=== STANDARD MODE (8p/3ir/24sps) ===\n")

    fixture_path = Path(f"data/rounds/{args.round_id}")
    if not fixture_path.exists():
        rounds_dir = Path("data/rounds")
        available = [
            d for d in sorted(rounds_dir.iterdir()) if d.is_dir() and d.name != "test-round-001"
        ]
        if not available:
            log("No real round fixtures found.")
            sys.exit(1)
        fixture_path = available[-1]
        log(f"Using fallback fixture: {fixture_path.name}")

    log(f"Loading fixture: {fixture_path.name}")
    fixture = load_fixture(fixture_path)
    log(
        f"  Map: {fixture.map_width}x{fixture.map_height}, "
        f"Seeds: {fixture.seeds_count}, Params: {fixture.params_source}"
    )

    # Ground truths (cached in fixture)
    log("Loading ground truths...")
    t0 = time.monotonic()
    ground_truths = load_or_compute_ground_truths(fixture, n_mc_runs=50, base_seed=0)
    log(f"  Ready ({time.monotonic() - t0:.1f}s)")

    # Strategies
    log("\n--- Strategy Baselines ---")
    strategy_results = run_strategies(fixture, ground_truths)
    for name, res in strategy_results.items():
        per_seed = res["per_seed"]
        assert isinstance(per_seed, list)
        seeds_str = ", ".join(f"{s:.1f}" for s in per_seed)
        log(f"  {name:20s}: mean={res['mean']:6.2f}  [{seeds_str}]  {res['runtime_s']}s")

    # Solver
    log(f"\n--- Calibrated Solver ({n_particles}p/{n_inner_runs}ir/{sims_per_seed}sps) ---")
    solver_result = run_solver_single(
        fixture,
        ground_truths,
        n_particles=n_particles,
        n_inner_runs=n_inner_runs,
        sims_per_seed=sims_per_seed,
    )

    # Comparison table
    log("\n" + "=" * 72)
    log(f"{'Strategy':<25s} {'Mean Score':>10s} {'Runtime':>10s}")
    log("-" * 72)
    for name, res in strategy_results.items():
        log(f"  {name:<23s} {res['mean']:>10.2f} {res['runtime_s']:>9.1f}s")

    solver_mean = solver_result["mean"]
    solver_time = solver_result["runtime_s"]
    assert isinstance(solver_mean, float)
    assert isinstance(solver_time, float)
    log(f"  {'calibrated_solver':<23s} {solver_mean:>10.2f} {solver_time:>9.1f}s")
    log("=" * 72)

    per_seed = solver_result["per_seed"]
    assert isinstance(per_seed, list)
    log(f"\nSolver per-seed: [{', '.join(f'{s:.1f}' for s in per_seed)}]")


if __name__ == "__main__":
    main()
