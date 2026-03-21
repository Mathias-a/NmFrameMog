"""Replay validation with ablations.

Evaluates the final solver against:
  - Uniform baseline
  - Fixed-coverage baseline
  - Particle solver without hedge
  - Particle solver with hedge enabled when gated

Selects the winner as the highest-mean configuration.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astar_twin.data.loaders import load_fixture
from astar_twin.scoring import compute_score
from astar_twin.solver.adapters.benchmark import BenchmarkAdapter
from astar_twin.solver.baselines import fixed_coverage_baseline, uniform_baseline
from astar_twin.solver.eval.run_benchmark_suite import load_or_compute_ground_truths
from astar_twin.solver.pipeline import solve
from astar_twin.solver.predict.hedge import apply_hedge, should_hedge


@dataclass
class VariantResult:
    """Result for a single solver variant."""

    name: str
    per_seed_scores: list[float]
    mean_score: float


@dataclass
class ReplayResult:
    """Full replay validation result."""

    variants: list[VariantResult]
    winner_name: str
    winner_mean: float
    hedge_activated: bool
    calibration_disagreements: list[float]

    def to_dict(self) -> dict:
        return {
            "variants": [
                {
                    "name": v.name,
                    "per_seed_scores": v.per_seed_scores,
                    "mean_score": v.mean_score,
                }
                for v in self.variants
            ],
            "winner": {
                "name": self.winner_name,
                "mean_score": self.winner_mean,
            },
            "hedge_activated": self.hedge_activated,
            "calibration_disagreements": self.calibration_disagreements,
        }


def run_replay_validation(
    fixture_path: Path,
    n_particles: int = 24,
    n_inner_runs: int = 6,
    sims_per_seed: int = 64,
    fc_mc_runs: int = 200,
    gt_mc_runs: int = 200,
    gt_base_seed: int = 0,
    gt_prior_spread: float = 1.0,
) -> ReplayResult:
    """Run replay validation comparing all variants.

    Args:
        fixture_path: Path to round_detail.json.
        n_particles: Particles for solver.
        n_inner_runs: Inner MC runs.
        sims_per_seed: Final prediction MC runs.
        fc_mc_runs: MC runs for fixed_coverage baseline.
        gt_mc_runs: MC runs for ground truth (only used when fixture has no
            cached ground_truths).
        gt_base_seed: Base seed for ground truth derivation.
        gt_prior_spread: Spread for DEFAULT_PRIOR sampling when deriving
            uncached ground truths. Ignored when cached ground truths already
            exist.

    Returns:
        ReplayResult with all variant scores and selected winner.
    """
    fixture = load_fixture(fixture_path)
    height = fixture.map_height
    width = fixture.map_width
    initial_states = fixture.initial_states
    n_seeds = len(initial_states)

    # Ground truth: prefer cached tensors, otherwise compute.
    ground_truths = load_or_compute_ground_truths(
        fixture,
        gt_mc_runs,
        gt_base_seed,
        prior_spread=gt_prior_spread,
    )

    variants: list[VariantResult] = []

    # Variant 1: Uniform baseline
    uniform = uniform_baseline(height, width)
    uniform_scores = [float(compute_score(gt, uniform)) for gt in ground_truths]
    variants.append(
        VariantResult(
            name="uniform",
            per_seed_scores=uniform_scores,
            mean_score=float(np.mean(uniform_scores)),
        )
    )

    # Variant 2: Fixed-coverage baseline
    fc_tensors = fixed_coverage_baseline(
        initial_states,
        height,
        width,
        n_mc_runs=fc_mc_runs,
        base_seed=42,
    )
    fc_scores = [
        float(compute_score(gt, t)) for gt, t in zip(ground_truths, fc_tensors, strict=True)
    ]
    variants.append(
        VariantResult(
            name="fixed_coverage",
            per_seed_scores=fc_scores,
            mean_score=float(np.mean(fc_scores)),
        )
    )
    fc_mean = float(np.mean(fc_scores))

    # Variant 3: Particle solver (no hedge)
    adapter = BenchmarkAdapter(fixture, n_mc_runs=5, sim_seed_offset=0)
    result = solve(
        adapter,
        fixture.id,
        n_particles=n_particles,
        n_inner_runs=n_inner_runs,
        sims_per_seed=sims_per_seed,
        base_seed=42,
    )
    particle_scores = [
        float(compute_score(gt, t)) for gt, t in zip(ground_truths, result.tensors, strict=True)
    ]
    particle_mean = float(np.mean(particle_scores))
    variants.append(
        VariantResult(
            name="particle_no_hedge",
            per_seed_scores=particle_scores,
            mean_score=particle_mean,
        )
    )

    # Compute calibration disagreements per seed
    calibration_disagreements: list[float] = []
    for seed_idx in range(n_seeds):
        # Disagreement = absolute difference in score between particle and fc
        disagree = abs(particle_scores[seed_idx] - fc_scores[seed_idx]) / 100.0
        calibration_disagreements.append(disagree)

    # Variant 4: Particle solver with hedge (if gate triggers)
    hedge_activated = should_hedge(
        particle_mean,
        fc_mean,
        calibration_disagreements,
    )
    if hedge_activated:
        hedged = apply_hedge(
            result.tensors,
            fc_tensors,
            initial_states,
            height,
            width,
        )
        hedged_scores = [
            float(compute_score(gt, t)) for gt, t in zip(ground_truths, hedged, strict=True)
        ]
        variants.append(
            VariantResult(
                name="particle_hedged",
                per_seed_scores=hedged_scores,
                mean_score=float(np.mean(hedged_scores)),
            )
        )

    # Select winner
    winner = max(variants, key=lambda v: v.mean_score)

    return ReplayResult(
        variants=variants,
        winner_name=winner.name,
        winner_mean=winner.mean_score,
        hedge_activated=hedge_activated,
        calibration_disagreements=calibration_disagreements,
    )


def main() -> None:
    """CLI entry point for replay validation.

    ``--gt-prior-spread`` only affects fixtures without cached ground truths.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run replay validation")
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--particles", type=int, default=24)
    parser.add_argument("--inner-runs", type=int, default=6)
    parser.add_argument("--sims-per-seed", type=int, default=64)
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

    result = run_replay_validation(
        fixture_path,
        n_particles=args.particles,
        n_inner_runs=args.inner_runs,
        sims_per_seed=args.sims_per_seed,
        gt_prior_spread=args.gt_prior_spread,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"Replay validation complete. Winner: {result.winner_name} ({result.winner_mean:.2f})")
    for v in result.variants:
        print(f"  {v.name}: {v.mean_score:.2f}")
    print(f"  Hedge activated: {result.hedge_activated}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
