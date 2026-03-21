"""Strategy tournament: evaluate multiple strategies across multiple fixtures.

Runs every registered (or supplied) strategy against a filtered set of
real-round fixtures, with repeated evaluations for variance estimation.
Produces a ranking with confidence signals so you can distinguish genuine
performance differences from noise.

Usage::

    cd Astar-Island/benchmark
    uv run python -m astar_twin.harness.tournament \\
        --data-dir data \\
        --repeats 3 \\
        --output results/tournament.json

    # Only use fixtures with API-sourced ground truths:
    uv run python -m astar_twin.harness.tournament \\
        --data-dir data \\
        --provenance api_analysis \\
        --repeats 5 \\
        --output results/tournament_api.json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astar_twin.data.loaders import list_fixtures
from astar_twin.data.models import GroundTruthSource, RoundFixture
from astar_twin.harness.protocol import Strategy
from astar_twin.harness.runner import BenchmarkRunner


def _is_real_round(fixture: RoundFixture) -> bool:
    return not fixture.id.startswith("test-")


def filter_fixtures(
    fixtures: list[RoundFixture],
    provenance: GroundTruthSource | None = None,
) -> list[RoundFixture]:
    """Return real-round fixtures, optionally filtered by ground-truth provenance.

    Fixtures without cached ground_truths are always excluded: the tournament
    requires pre-computed ground truths for reproducible, fast evaluation.
    """
    result = [f for f in fixtures if _is_real_round(f) and f.ground_truths is not None]
    if provenance is not None:
        result = [f for f in result if f.ground_truth_source == provenance]
    return sorted(result, key=lambda f: f.round_number)


@dataclass(frozen=True)
class FixtureStrategyResult:
    """One strategy's scores on one fixture across all repeats."""

    fixture_id: str
    round_number: int
    round_weight: float
    ground_truth_source: GroundTruthSource
    repeat_mean_scores: tuple[float, ...]
    overall_mean: float
    overall_std: float

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "round_number": self.round_number,
            "round_weight": self.round_weight,
            "ground_truth_source": str(self.ground_truth_source),
            "repeat_mean_scores": list(self.repeat_mean_scores),
            "overall_mean": self.overall_mean,
            "overall_std": self.overall_std,
        }


@dataclass(frozen=True)
class StrategyTournamentResult:
    """One strategy's aggregated performance across all fixtures and repeats."""

    strategy_name: str
    fixture_results: tuple[FixtureStrategyResult, ...]
    weighted_mean: float
    unweighted_mean: float
    weighted_std: float
    repeat_weighted_means: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_name": self.strategy_name,
            "weighted_mean": self.weighted_mean,
            "unweighted_mean": self.unweighted_mean,
            "weighted_std": self.weighted_std,
            "repeat_weighted_means": list(self.repeat_weighted_means),
            "fixtures": [fr.to_dict() for fr in self.fixture_results],
        }


@dataclass(frozen=True)
class TournamentReport:
    """Full tournament output: rankings, per-fixture breakdowns, stability."""

    strategy_results: tuple[StrategyTournamentResult, ...]
    fixtures_evaluated: int
    repeats: int
    provenance_filter: str | None
    total_runtime_seconds: float

    def ranked(self) -> list[StrategyTournamentResult]:
        return sorted(
            self.strategy_results,
            key=lambda s: s.weighted_mean,
            reverse=True,
        )

    def is_top_rank_stable(self) -> bool:
        """True if the #1 strategy wins in every repeat."""
        if len(self.strategy_results) < 2:
            return True
        ranked = self.ranked()
        top = ranked[0]
        runner_up = ranked[1]
        for r_idx in range(len(top.repeat_weighted_means)):
            if top.repeat_weighted_means[r_idx] < runner_up.repeat_weighted_means[r_idx]:
                return False
        return True

    def print_summary(self) -> None:
        ranked = self.ranked()
        prov = self.provenance_filter or "all"
        print(
            f"\nTournament: {self.fixtures_evaluated} fixtures, "
            f"{self.repeats} repeats, provenance={prov}"
        )
        print(f"Runtime: {self.total_runtime_seconds:.1f}s")
        print(f"Top-rank stable: {self.is_top_rank_stable()}\n")
        print("Ranking (by weighted mean score):")
        for rank, sr in enumerate(ranked, 1):
            gap_str = ""
            if rank > 1:
                gap = ranked[rank - 2].weighted_mean - sr.weighted_mean
                gap_str = f"  gap={gap:+.2f}"
            print(
                f"  #{rank:<2} {sr.strategy_name:<24} "
                f"weighted={sr.weighted_mean:6.2f} \u00b1{sr.weighted_std:.2f}  "
                f"unweighted={sr.unweighted_mean:6.2f}{gap_str}"
            )
        print("\nPer-fixture breakdown:")
        fixture_lists: list[list[FixtureStrategyResult]] = [
            list(sr.fixture_results) for sr in ranked
        ]
        n_fixtures = len(fixture_lists[0]) if fixture_lists else 0
        for fix_idx in range(n_fixtures):
            rn = fixture_lists[0][fix_idx].round_number
            w = fixture_lists[0][fix_idx].round_weight
            scores = "  ".join(
                f"{ranked[i].strategy_name}={fixture_lists[i][fix_idx].overall_mean:.1f}"
                for i in range(len(ranked))
            )
            print(f"  Round {rn:>2} (w={w:.3f}): {scores}")
        print()

    def to_dict(self) -> dict[str, object]:
        return {
            "fixtures_evaluated": self.fixtures_evaluated,
            "repeats": self.repeats,
            "provenance_filter": self.provenance_filter,
            "total_runtime_seconds": self.total_runtime_seconds,
            "top_rank_stable": self.is_top_rank_stable(),
            "ranking": [sr.to_dict() for sr in self.ranked()],
        }


@dataclass
class TournamentRunner:
    """Evaluate strategies across fixtures with repeated runs.

    Each repeat creates a fresh ``BenchmarkRunner`` per fixture with a
    different ``base_seed``, ensuring within-repeat fairness (all strategies
    scored against the same ground truths) while varying seeds across
    repeats to measure robustness.
    """

    fixtures: list[RoundFixture]
    repeats: int = 3
    base_seed: int = 42
    n_mc_runs: int = 200
    provenance_filter: GroundTruthSource | None = None

    def run(self, strategies: list[Strategy]) -> TournamentReport:
        t_start = time.monotonic()

        selected = filter_fixtures(self.fixtures, self.provenance_filter)
        if not selected:
            raise ValueError(
                f"No fixtures match filter (provenance={self.provenance_filter}). "
                f"Available: {len(self.fixtures)} total, "
                f"{sum(1 for f in self.fixtures if _is_real_round(f))} real."
            )

        strategy_names = [s.name for s in strategies]
        n_strategies = len(strategies)
        n_fixtures = len(selected)
        n_repeats = self.repeats

        # scores[strategy_idx][fixture_idx][repeat_idx] = mean_score
        scores: list[list[list[float]]] = [
            [[] for _ in range(n_fixtures)] for _ in range(n_strategies)
        ]

        for repeat_idx in range(n_repeats):
            seed = self.base_seed + repeat_idx * 1000
            for fix_idx, fixture in enumerate(selected):
                runner = BenchmarkRunner(
                    fixture=fixture,
                    base_seed=seed,
                    n_mc_runs=self.n_mc_runs,
                )
                report = runner.run(strategies)
                for strat_idx, sr in enumerate(report.strategy_reports):
                    scores[strat_idx][fix_idx].append(sr.mean_score)

        weights = [f.round_weight for f in selected]
        total_weight = sum(weights)

        strategy_results: list[StrategyTournamentResult] = []
        for strat_idx in range(n_strategies):
            fixture_results: list[FixtureStrategyResult] = []
            for fix_idx, fixture in enumerate(selected):
                repeat_scores = tuple(scores[strat_idx][fix_idx])
                fixture_results.append(
                    FixtureStrategyResult(
                        fixture_id=fixture.id,
                        round_number=fixture.round_number,
                        round_weight=fixture.round_weight,
                        ground_truth_source=fixture.ground_truth_source,
                        repeat_mean_scores=repeat_scores,
                        overall_mean=float(np.mean(repeat_scores)),
                        overall_std=float(np.std(repeat_scores)),
                    )
                )

            # Per-repeat weighted means (for ranking stability)
            repeat_weighted_means: list[float] = []
            for r_idx in range(n_repeats):
                wm = (
                    sum(
                        scores[strat_idx][f_idx][r_idx] * weights[f_idx]
                        for f_idx in range(n_fixtures)
                    )
                    / total_weight
                )
                repeat_weighted_means.append(wm)

            fixture_means = [fr.overall_mean for fr in fixture_results]
            weighted_mean = (
                sum(m * w for m, w in zip(fixture_means, weights, strict=True)) / total_weight
            )
            unweighted_mean = float(np.mean(fixture_means))
            weighted_std = float(np.std(repeat_weighted_means))

            strategy_results.append(
                StrategyTournamentResult(
                    strategy_name=strategy_names[strat_idx],
                    fixture_results=tuple(fixture_results),
                    weighted_mean=weighted_mean,
                    unweighted_mean=unweighted_mean,
                    weighted_std=weighted_std,
                    repeat_weighted_means=tuple(repeat_weighted_means),
                )
            )

        return TournamentReport(
            strategy_results=tuple(strategy_results),
            fixtures_evaluated=n_fixtures,
            repeats=n_repeats,
            provenance_filter=str(self.provenance_filter) if self.provenance_filter else None,
            total_runtime_seconds=time.monotonic() - t_start,
        )


def main() -> None:
    import argparse

    from astar_twin.strategies import REGISTRY

    parser = argparse.ArgumentParser(
        description="Run strategy tournament across real-round fixtures"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory containing rounds/ subdirectory (default: data).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Evaluation repeats per fixture (default: 3).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    parser.add_argument(
        "--provenance",
        choices=["api_analysis", "local_mc", "unknown"],
        default=None,
        help="Filter fixtures by ground-truth provenance (default: all).",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Strategy names to include (default: all registered).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (optional).",
    )
    args = parser.parse_args()

    data_dir_str: str = args.data_dir
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    provenance_str: str | None = args.provenance
    provenance = GroundTruthSource(provenance_str) if provenance_str else None

    strategy_names: list[str] | None = args.strategies
    if strategy_names is not None:
        unknown = set(strategy_names) - set(REGISTRY.keys())
        if unknown:
            print(f"Unknown strategies: {unknown}.  Available: {sorted(REGISTRY.keys())}")
            sys.exit(1)
        strategies = [REGISTRY[name]() for name in strategy_names]
    else:
        strategies = [cls() for cls in REGISTRY.values()]

    repeats: int = args.repeats
    base_seed: int = args.base_seed
    all_fixtures = list_fixtures(data_dir)
    runner = TournamentRunner(
        fixtures=all_fixtures,
        repeats=repeats,
        base_seed=base_seed,
        provenance_filter=provenance,
    )

    print(
        f"Running tournament: {len(strategies)} strategies, "
        f"{repeats} repeats, provenance={provenance_str or 'all'}"
    )
    report = runner.run(strategies)
    report.print_summary()

    output_str: str | None = args.output
    if output_str:
        output_path = Path(output_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
