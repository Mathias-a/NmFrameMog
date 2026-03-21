"""Multi-fixture benchmark suite.

Evaluates the solver across all known real rounds and aggregates results.
Each fixture is evaluated independently with a fresh 50-query budget per run
(shared across all 5 seeds within a run, matching the real competition rule).

Usage::

    cd Astar-Island/benchmark
    uv run python -m astar_twin.solver.eval.run_multi_fixture_suite \\
        --data-dir data/rounds \\
        --output results/multi_suite.json \\
        --repeats 5
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from astar_twin.data.loaders import list_fixtures
from astar_twin.data.models import RoundFixture
from astar_twin.solver.eval.run_benchmark_suite import SuiteResult, run_suite


@dataclass
class FixtureResult:
    """Benchmark result for a single fixture (round)."""

    round_id: str
    round_number: int
    round_weight: float
    suite: SuiteResult

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "round_number": self.round_number,
            "round_weight": self.round_weight,
            "suite": self.suite.to_dict(),
        }


@dataclass
class MultiFixtureSuiteResult:
    """Aggregated benchmark result across all real rounds.

    Scores are weighted by ``round_weight`` to match competition scoring,
    plus an unweighted mean for easy comparison.
    """

    fixture_results: list[FixtureResult]
    overall_weighted_mean: float
    overall_unweighted_mean: float
    per_round_means: dict[int, float]
    total_runtime_seconds: float
    rounds_evaluated: int
    repeats_per_round: int

    def to_dict(self) -> dict:
        return {
            "rounds_evaluated": self.rounds_evaluated,
            "repeats_per_round": self.repeats_per_round,
            "overall_weighted_mean": self.overall_weighted_mean,
            "overall_unweighted_mean": self.overall_unweighted_mean,
            "per_round_means": {str(k): v for k, v in self.per_round_means.items()},
            "total_runtime_seconds": self.total_runtime_seconds,
            "fixtures": [fr.to_dict() for fr in self.fixture_results],
        }

    def print_summary(self) -> None:
        print(
            f"Multi-fixture suite: {self.rounds_evaluated} rounds, {self.repeats_per_round} repeats each"
        )
        print(f"  Weighted mean:   {self.overall_weighted_mean:.2f}")
        print(f"  Unweighted mean: {self.overall_unweighted_mean:.2f}")
        print(f"  Runtime:         {self.total_runtime_seconds:.1f}s")
        print()
        for fr in sorted(self.fixture_results, key=lambda r: r.round_number):
            s = fr.suite
            print(
                f"  Round {fr.round_number:>2} (w={fr.round_weight:.3f}): "
                f"mean={s.candidate_mean:.2f} "
                f"std={s.candidate_std:.2f} "
                f"[{s.candidate_min:.2f}–{s.candidate_max:.2f}] "
                f"vs uniform={s.uniform_mean:.2f} fc={s.fixed_coverage_mean:.2f}"
            )


def _is_real_round(fixture: RoundFixture) -> bool:
    return not fixture.id.startswith("test-")


def run_multi_fixture_suite(
    data_dir: Path,
    repeats: int = 5,
    n_particles: int = 24,
    n_inner_runs: int = 6,
    sims_per_seed: int = 64,
    fc_mc_runs: int = 200,
) -> MultiFixtureSuiteResult:
    """Evaluate the solver across all known real rounds.

    Args:
        data_dir: Directory containing ``rounds/*/round_detail.json`` fixtures.
        repeats: Solver runs per fixture for variance estimation.
        n_particles: Particle count for the solver.
        n_inner_runs: Inner MC runs per likelihood update.
        sims_per_seed: Final MC runs for prediction.
        fc_mc_runs: MC runs for the fixed-coverage baseline.

    Returns:
        MultiFixtureSuiteResult with per-fixture and aggregated statistics.
    """
    t_start = time.monotonic()

    all_fixtures = list_fixtures(data_dir)
    real_fixtures = [f for f in all_fixtures if _is_real_round(f)]
    real_fixtures.sort(key=lambda f: f.round_number)

    if not real_fixtures:
        raise ValueError(f"No real-round fixtures found under {data_dir}")

    fixture_results: list[FixtureResult] = []
    for fixture in real_fixtures:
        fixture_path = data_dir / "rounds" / fixture.id / "round_detail.json"
        suite = run_suite(
            fixture_path,
            repeats=repeats,
            n_particles=n_particles,
            n_inner_runs=n_inner_runs,
            sims_per_seed=sims_per_seed,
            fc_mc_runs=fc_mc_runs,
        )
        fixture_results.append(
            FixtureResult(
                round_id=fixture.id,
                round_number=fixture.round_number,
                round_weight=fixture.round_weight,
                suite=suite,
            )
        )

    candidate_means = [fr.suite.candidate_mean for fr in fixture_results]
    weights = [fr.round_weight for fr in fixture_results]
    total_weight = sum(weights)

    weighted_mean = (
        float(sum(m * w for m, w in zip(candidate_means, weights)) / total_weight)
        if total_weight > 0
        else 0.0
    )
    unweighted_mean = float(np.mean(candidate_means))
    per_round_means = {fr.round_number: fr.suite.candidate_mean for fr in fixture_results}

    return MultiFixtureSuiteResult(
        fixture_results=fixture_results,
        overall_weighted_mean=weighted_mean,
        overall_unweighted_mean=unweighted_mean,
        per_round_means=per_round_means,
        total_runtime_seconds=time.monotonic() - t_start,
        rounds_evaluated=len(fixture_results),
        repeats_per_round=repeats,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run multi-fixture benchmark suite across all real rounds"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory containing rounds/ subdirectory (default: data).",
    )
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    parser.add_argument(
        "--repeats", type=int, default=5, help="Solver runs per fixture (default 5)."
    )
    parser.add_argument("--particles", type=int, default=24)
    parser.add_argument("--inner-runs", type=int, default=6)
    parser.add_argument("--sims-per-seed", type=int, default=64)
    parser.add_argument("--fc-mc-runs", type=int, default=200)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    result = run_multi_fixture_suite(
        data_dir,
        repeats=args.repeats,
        n_particles=args.particles,
        n_inner_runs=args.inner_runs,
        sims_per_seed=args.sims_per_seed,
        fc_mc_runs=args.fc_mc_runs,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2))
    result.print_summary()
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
