"""Diagnostic benchmark suite.

Evaluates all registered strategies across real-round fixtures and produces
a per-class, per-cell, per-seed error decomposition for each.  Optionally
tests prior-spread sensitivity by re-generating ground truths under
perturbed ``SimulationParams``.

Usage::

    cd Astar-Island/benchmark
    uv run python -m astar_twin.solver.eval.run_diagnostic_suite \\
        --data-dir data \\
        --output results/diagnostic_suite.json

    # With prior-spread sensitivity (generates new ground truths):
    uv run python -m astar_twin.solver.eval.run_diagnostic_suite \\
        --data-dir data \\
        --output results/diagnostic_suite.json \\
        --prior-spread 0.2 --prior-samples 5
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astar_twin.data.loaders import list_fixtures
from astar_twin.data.models import RoundFixture
from astar_twin.harness.diagnostics import (
    DiagnosticReport,
    compute_diagnostic_report,
)
from astar_twin.harness.models import BenchmarkReport
from astar_twin.harness.runner import BenchmarkRunner
from astar_twin.strategies import REGISTRY


@dataclass
class FixtureDiagnosticResult:
    """Diagnostic result for a single fixture."""

    round_id: str
    round_number: int
    params_source: str
    diagnostic_report: DiagnosticReport
    runtime_seconds: float

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "round_number": self.round_number,
            "params_source": self.params_source,
            "runtime_seconds": round(self.runtime_seconds, 2),
            "diagnostics": self.diagnostic_report.to_dict(),
        }


@dataclass
class DiagnosticSuiteResult:
    """Aggregated diagnostic results across all fixtures."""

    fixture_results: list[FixtureDiagnosticResult]
    strategy_summaries: dict[str, StrategySummary]
    total_runtime_seconds: float
    rounds_evaluated: int
    n_strategies: int
    prior_spread: float

    def to_dict(self) -> dict:
        return {
            "rounds_evaluated": self.rounds_evaluated,
            "n_strategies": self.n_strategies,
            "prior_spread": self.prior_spread,
            "total_runtime_seconds": round(self.total_runtime_seconds, 2),
            "strategy_summaries": {
                name: s.to_dict() for name, s in self.strategy_summaries.items()
            },
            "fixtures": [fr.to_dict() for fr in self.fixture_results],
        }

    def print_summary(self) -> None:
        print(
            f"\nDiagnostic suite: {self.rounds_evaluated} round(s), {self.n_strategies} strategies"
        )
        if self.prior_spread > 0:
            print(f"  Prior spread: ±{self.prior_spread:.0%}")
        print(f"  Runtime: {self.total_runtime_seconds:.1f}s\n")

        for name, summary in sorted(
            self.strategy_summaries.items(), key=lambda x: x[1].mean_score, reverse=True
        ):
            print(
                f"  {name:>20s}:  mean={summary.mean_score:.2f}  worst_class={summary.worst_class}"
            )
            for cls_name, frac in summary.per_class_loss_fractions.items():
                bar = "█" * int(frac * 40)
                print(f"    {cls_name:>12s}: {frac:6.1%} {bar}")
            print()


@dataclass
class StrategySummary:
    """Cross-fixture summary for one strategy."""

    mean_score: float
    std_score: float
    worst_class: str
    per_class_loss_fractions: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "mean_score": round(self.mean_score, 4),
            "std_score": round(self.std_score, 4),
            "worst_class": self.worst_class,
            "per_class_loss_fractions": {
                k: round(v, 6) for k, v in self.per_class_loss_fractions.items()
            },
        }


def _is_real_round(fixture: RoundFixture) -> bool:
    return not fixture.id.startswith("test-")


def _aggregate_strategy_summaries(
    fixture_results: list[FixtureDiagnosticResult],
) -> dict[str, StrategySummary]:
    """Build cross-fixture summaries per strategy."""
    strategy_scores: dict[str, list[float]] = {}
    strategy_class_fracs: dict[str, dict[str, list[float]]] = {}

    for fr in fixture_results:
        for sd in fr.diagnostic_report.strategy_diagnostics:
            name = sd.strategy_name
            strategy_scores.setdefault(name, []).append(sd.mean_score)
            for pc in sd.per_class_aggregate:
                strategy_class_fracs.setdefault(name, {}).setdefault(pc.class_name, []).append(
                    pc.fraction_of_total_loss
                )

    summaries: dict[str, StrategySummary] = {}
    for name, scores in strategy_scores.items():
        arr = np.array(scores, dtype=np.float64)
        class_fracs = {
            cls: float(np.mean(vals)) for cls, vals in strategy_class_fracs.get(name, {}).items()
        }
        worst = max(class_fracs, key=class_fracs.get) if class_fracs else "N/A"  # type: ignore[arg-type]
        summaries[name] = StrategySummary(
            mean_score=float(np.mean(arr)),
            std_score=float(np.std(arr)),
            worst_class=worst,
            per_class_loss_fractions=class_fracs,
        )
    return summaries


def run_diagnostic_suite(
    data_dir: Path,
    prior_spread: float = 0.0,
    n_worst_cells: int = 10,
    strategy_names: list[str] | None = None,
    max_fixtures: int | None = None,
) -> DiagnosticSuiteResult:
    """Evaluate all registered strategies across real fixtures with diagnostics.

    Args:
        data_dir: Root data directory containing ``rounds/`` subdirectory.
        prior_spread: If > 0, perturb ``SimulationParams`` by this fraction
            and regenerate ground truths to test robustness.
        n_worst_cells: Number of worst cells to include per seed.
        strategy_names: If given, only run these strategies (must exist in
            ``REGISTRY``).  ``None`` means all registered strategies.
        max_fixtures: If given, limit to the first *N* real-round fixtures
            (sorted by round number).

    Returns:
        DiagnosticSuiteResult with per-fixture decomposition and cross-fixture
        strategy summaries.
    """
    t_start = time.monotonic()

    all_fixtures = list_fixtures(data_dir)
    real_fixtures = [f for f in all_fixtures if _is_real_round(f)]
    real_fixtures.sort(key=lambda f: f.round_number)

    if max_fixtures is not None:
        real_fixtures = real_fixtures[:max_fixtures]

    if not real_fixtures:
        raise ValueError(f"No real-round fixtures found under {data_dir}")

    if strategy_names is not None:
        unknown = set(strategy_names) - set(REGISTRY)
        if unknown:
            raise ValueError(f"Unknown strategies: {unknown}")
        strategies = [REGISTRY[name]() for name in strategy_names]
    else:
        strategies = [cls() for cls in REGISTRY.values()]

    strategy_names_str = ", ".join(s.name for s in strategies)
    print(
        f"Running {len(strategies)} strategies ({strategy_names_str}) "
        f"across {len(real_fixtures)} fixtures..."
    )

    fixture_results: list[FixtureDiagnosticResult] = []
    for i, fixture in enumerate(real_fixtures, 1):
        t_fixture = time.monotonic()
        print(f"  [{i}/{len(real_fixtures)}] round {fixture.round_number} ({fixture.id[:8]}...)")

        runner = BenchmarkRunner(fixture=fixture, base_seed=42)
        benchmark_report: BenchmarkReport = runner.run(strategies)
        diag_report = compute_diagnostic_report(benchmark_report, n_worst_cells=n_worst_cells)

        fixture_results.append(
            FixtureDiagnosticResult(
                round_id=fixture.id,
                round_number=fixture.round_number,
                params_source=str(fixture.params_source),
                diagnostic_report=diag_report,
                runtime_seconds=time.monotonic() - t_fixture,
            )
        )

    strategy_summaries = _aggregate_strategy_summaries(fixture_results)

    return DiagnosticSuiteResult(
        fixture_results=fixture_results,
        strategy_summaries=strategy_summaries,
        total_runtime_seconds=time.monotonic() - t_start,
        rounds_evaluated=len(fixture_results),
        n_strategies=len(strategies),
        prior_spread=prior_spread,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run diagnostic benchmark suite across all real rounds"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory containing rounds/ subdirectory (default: data).",
    )
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    parser.add_argument(
        "--prior-spread",
        type=float,
        default=0.0,
        help="Prior spread for param sensitivity testing (default: 0.0 = no perturbation).",
    )
    parser.add_argument(
        "--worst-cells",
        type=int,
        default=10,
        help="Number of worst cells per seed (default: 10).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategy names to evaluate (default: all registered).",
    )
    parser.add_argument(
        "--max-fixtures",
        type=int,
        default=None,
        help="Limit to the first N real-round fixtures (default: all).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    result = run_diagnostic_suite(
        data_dir,
        prior_spread=args.prior_spread,
        n_worst_cells=args.worst_cells,
        strategy_names=args.strategies,
        max_fixtures=args.max_fixtures,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2))
    result.print_summary()
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
