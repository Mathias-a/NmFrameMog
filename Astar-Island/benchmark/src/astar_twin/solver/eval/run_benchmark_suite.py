"""Repeated benchmark evaluation suite.

Runs the solver multiple times against the local twin and records:
  - mean, min, max, std scores
  - per-seed averages
  - runtime per run
  - hedge activation count
  - baseline comparisons
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import ParamsSource, RoundFixture
from astar_twin.params import sample_default_prior_params
from astar_twin.scoring import compute_score, safe_prediction
from astar_twin.solver.adapters.benchmark import BenchmarkAdapter
from astar_twin.solver.baselines import (
    fixed_coverage_baseline,
    uniform_baseline,
)
from astar_twin.solver.high_value_bidirectional import solve_high_value_bidirectional
from astar_twin.solver.pipeline import solve
from astar_twin.solver.predict.hedge import apply_hedge, should_hedge


@dataclass
class RunResult:
    """Result from a single benchmark run."""

    run_index: int
    per_seed_scores: list[float]
    mean_score: float
    runtime_seconds: float
    total_queries: int


@dataclass
class SuiteResult:
    """Aggregated result from the full benchmark suite."""

    repeats: int
    # Candidate (particle solver)
    candidate_mean: float
    candidate_min: float
    candidate_max: float
    candidate_std: float
    candidate_per_seed_avg: list[float]
    candidate_runs: list[RunResult]
    # Baselines
    uniform_mean: float
    uniform_per_seed: list[float]
    fixed_coverage_mean: float
    fixed_coverage_per_seed: list[float]
    # Hedge
    hedge_activations: int
    hedged_mean: float | None
    # Timing
    total_runtime_seconds: float
    variant_name: str = "particle"

    def to_dict(self) -> dict[str, object]:
        """Convert to JSON-serializable dict."""
        d = {
            "repeats": self.repeats,
            "candidate": {
                "variant_name": self.variant_name,
                "mean": self.candidate_mean,
                "min": self.candidate_min,
                "max": self.candidate_max,
                "std": self.candidate_std,
                "per_seed_avg": self.candidate_per_seed_avg,
            },
            "baselines": {
                "uniform_mean": self.uniform_mean,
                "uniform_per_seed": self.uniform_per_seed,
                "fixed_coverage_mean": self.fixed_coverage_mean,
                "fixed_coverage_per_seed": self.fixed_coverage_per_seed,
            },
            "hedge": {
                "activations": self.hedge_activations,
                "hedged_mean": self.hedged_mean,
            },
            "total_runtime_seconds": self.total_runtime_seconds,
            "runs": [
                {
                    "run_index": r.run_index,
                    "per_seed_scores": r.per_seed_scores,
                    "mean_score": r.mean_score,
                    "runtime_seconds": r.runtime_seconds,
                    "total_queries": r.total_queries,
                }
                for r in self.candidate_runs
            ],
        }
        return d


def resilient_run_batch(
    simulator: object,
    initial_state: object,
    n_runs: int,
    base_seed: int,
) -> list[Any]:
    """Run MC batch, skipping individual runs that hit simulator NaN bugs.

    This helper is shared by run_benchmark_suite and run_replay_validation.
    """
    from astar_twin.contracts.api_models import InitialState
    from astar_twin.engine import Simulator
    from astar_twin.state.round_state import RoundState

    assert isinstance(simulator, Simulator)
    assert isinstance(initial_state, InitialState)

    runs: list[RoundState] = []
    for i in range(n_runs):
        try:
            state = simulator.run(initial_state=initial_state, sim_seed=base_seed + i)
            runs.append(state)
        except (ValueError, FloatingPointError):
            # Skip runs that hit stochastic NaN issues in the engine
            pass
    if not runs:
        # Fallback: at least one run needed; try with a very different seed
        state = simulator.run(initial_state=initial_state, sim_seed=base_seed + 99999)
        runs.append(state)
    return runs


def load_or_compute_ground_truths(
    fixture: RoundFixture,
    n_mc_runs: int,
    base_seed: int,
    prior_spread: float = 1.0,
) -> list[NDArray[np.float64]]:
    """Load pre-computed ground truths from fixture or compute them on demand.

    If ``fixture.ground_truths`` is already populated (i.e. was pre-computed
    and persisted in the round_detail.json), the cached tensors are returned
    directly without running any simulations.  This keeps benchmark runs fast
    and reproducible.

    Otherwise, ground truths are derived by running ``n_mc_runs`` resilient
    Monte Carlo simulations per seed using the fixture's simulation_params.
    """
    if fixture.ground_truths is not None:
        return [np.array(gt, dtype=np.float64) for gt in fixture.ground_truths]

    from astar_twin.engine import Simulator
    from astar_twin.mc import aggregate_runs

    simulation_params = fixture.simulation_params
    if fixture.params_source == ParamsSource.DEFAULT_PRIOR:
        simulation_params = sample_default_prior_params(
            seed=base_seed,
            defaults=fixture.simulation_params,
            spread=prior_spread,
        )

    simulator = Simulator(simulation_params)
    ground_truths: list[NDArray[np.float64]] = []
    for seed_idx, initial_state in enumerate(fixture.initial_states):
        runs = resilient_run_batch(
            simulator,
            initial_state,
            n_runs=n_mc_runs,
            base_seed=base_seed + seed_idx * n_mc_runs,
        )
        gt = safe_prediction(aggregate_runs(runs, fixture.map_height, fixture.map_width))
        ground_truths.append(gt)
    return ground_truths


def run_suite(
    fixture_path: Path,
    repeats: int = 10,
    n_particles: int = 24,
    n_inner_runs: int = 6,
    sims_per_seed: int = 64,
    fc_mc_runs: int = 200,
    gt_mc_runs: int = 200,
    gt_base_seed: int = 0,
    gt_prior_spread: float = 1.0,
    variant: str = "particle",
) -> SuiteResult:
    """Run the full benchmark suite.

    Args:
        fixture_path: Path to round_detail.json fixture.
        repeats: Number of solver runs (default 10).
        n_particles: Particles for solver.
        n_inner_runs: Inner MC runs per likelihood.
        sims_per_seed: Final prediction MC runs.
        fc_mc_runs: MC runs for fixed_coverage baseline.
        gt_mc_runs: MC runs for ground truth (only used when fixture has no
            cached ground_truths).
        gt_base_seed: Base seed for ground truth derivation.
        gt_prior_spread: Spread for DEFAULT_PRIOR sampling when deriving
            uncached ground truths. Ignored when cached ground truths already
            exist.
        variant: Solver variant to benchmark.

    Returns:
        SuiteResult with all metrics.
    """
    t_total_start = time.monotonic()
    fixture = load_fixture(fixture_path)
    height = fixture.map_height
    width = fixture.map_width
    initial_states = fixture.initial_states
    n_seeds = len(initial_states)

    # Ground truth: use cached tensors when available, otherwise compute.
    ground_truths = load_or_compute_ground_truths(
        fixture,
        gt_mc_runs,
        gt_base_seed,
        prior_spread=gt_prior_spread,
    )

    # Compute baselines
    uniform = uniform_baseline(height, width)
    uniform_per_seed = [float(compute_score(gt, uniform)) for gt in ground_truths]
    uniform_mean = float(np.mean(uniform_per_seed))

    fc_tensors = fixed_coverage_baseline(
        initial_states, height, width, n_mc_runs=fc_mc_runs, base_seed=42
    )
    fc_per_seed = [
        float(compute_score(gt, t)) for gt, t in zip(ground_truths, fc_tensors, strict=True)
    ]
    fc_mean = float(np.mean(fc_per_seed))

    solve_fn = solve if variant == "particle" else solve_high_value_bidirectional

    # Run solver N times
    candidate_runs: list[RunResult] = []
    all_scores: list[float] = []
    per_seed_accum: list[list[float]] = [[] for _ in range(n_seeds)]
    all_tensors: list[list[NDArray[np.float64]]] = []

    for run_idx in range(repeats):
        adapter = BenchmarkAdapter(fixture, n_mc_runs=5, sim_seed_offset=run_idx * 100)
        t_run_start = time.monotonic()
        result = solve_fn(
            adapter,
            fixture.id,
            n_particles=n_particles,
            n_inner_runs=n_inner_runs,
            sims_per_seed=sims_per_seed,
            base_seed=run_idx * 10000,
        )
        runtime = time.monotonic() - t_run_start

        # Score against ground truth — never use adapter.get_analysis() here
        # to keep solver blind to its own scoring during the run.
        per_seed_scores = [
            float(compute_score(gt, t))
            for gt, t in zip(ground_truths, result.tensors, strict=True)
        ]
        mean_score = float(np.mean(per_seed_scores))

        candidate_runs.append(
            RunResult(
                run_index=run_idx,
                per_seed_scores=per_seed_scores,
                mean_score=mean_score,
                runtime_seconds=runtime,
                total_queries=result.total_queries_used,
            )
        )
        all_scores.append(mean_score)
        for s_idx in range(n_seeds):
            per_seed_accum[s_idx].append(per_seed_scores[s_idx])
        all_tensors.append(result.tensors)

    candidate_mean = float(np.mean(all_scores))
    candidate_min = float(np.min(all_scores))
    candidate_max = float(np.max(all_scores))
    candidate_std = float(np.std(all_scores))
    candidate_per_seed_avg = [float(np.mean(s)) for s in per_seed_accum]

    # Check hedge gating
    hedge_activations = 0
    hedged_scores: list[float] = []
    for run_idx, run_tensors in enumerate(all_tensors):
        if should_hedge(candidate_runs[run_idx].mean_score, fc_mean):
            hedge_activations += 1
            hedged = apply_hedge(run_tensors, fc_tensors, initial_states, height, width)
            hedged_per_seed = [
                float(compute_score(gt, t)) for gt, t in zip(ground_truths, hedged, strict=True)
            ]
            hedged_scores.append(float(np.mean(hedged_per_seed)))

    hedged_mean = float(np.mean(hedged_scores)) if hedged_scores else None

    total_runtime = time.monotonic() - t_total_start

    return SuiteResult(
        repeats=repeats,
        candidate_mean=candidate_mean,
        candidate_min=candidate_min,
        candidate_max=candidate_max,
        candidate_std=candidate_std,
        candidate_per_seed_avg=candidate_per_seed_avg,
        candidate_runs=candidate_runs,
        uniform_mean=uniform_mean,
        uniform_per_seed=uniform_per_seed,
        fixed_coverage_mean=fc_mean,
        fixed_coverage_per_seed=fc_per_seed,
        hedge_activations=hedge_activations,
        hedged_mean=hedged_mean,
        total_runtime_seconds=total_runtime,
        variant_name=variant,
    )


def main() -> None:
    """CLI entry point for running the benchmark suite.

    ``--gt-prior-spread`` only affects fixtures without cached ground truths.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--particles", type=int, default=24)
    parser.add_argument("--inner-runs", type=int, default=6)
    parser.add_argument("--sims-per-seed", type=int, default=64)
    parser.add_argument(
        "--variant",
        choices=("particle", "high_value_bidirectional"),
        default="particle",
        help="Solver variant to benchmark.",
    )
    parser.add_argument(
        "--gt-mc-runs",
        type=int,
        default=200,
        help="MC runs for ground truth when not cached in fixture (default 200).",
    )
    parser.add_argument(
        "--gt-prior-spread",
        type=float,
        default=1.0,
        help=(
            "Spread for DEFAULT_PRIOR sampling when deriving uncached ground truths (default 1.0)."
        ),
    )
    args = parser.parse_args()

    fixture_path = Path(f"data/rounds/{args.round_id}/round_detail.json")
    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}")
        sys.exit(1)

    result = run_suite(
        fixture_path,
        repeats=args.repeats,
        n_particles=args.particles,
        n_inner_runs=args.inner_runs,
        sims_per_seed=args.sims_per_seed,
        gt_mc_runs=args.gt_mc_runs,
        variant=args.variant,
        gt_prior_spread=args.gt_prior_spread,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"Suite complete: {result.repeats} runs")
    print(
        f"  Candidate ({result.variant_name}): mean={result.candidate_mean:.2f}, "
        f"std={result.candidate_std:.2f}"
    )
    print(f"  Uniform:   mean={result.uniform_mean:.2f}")
    print(f"  FixedCov:  mean={result.fixed_coverage_mean:.2f}")
    print(f"  Hedge:     activations={result.hedge_activations}")
    print(f"  Output:    {output_path}")


if __name__ == "__main__":
    main()
