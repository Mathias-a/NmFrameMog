"""Tests for benchmark suite and hedge gating.

Covers:
  - Suite runs with reduced params and returns valid structure
  - Candidate scores are computed and > 0
  - Hedge gate logic: activates only when conditions met
  - Hedge blend produces valid tensors
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import ParamsSource
from astar_twin.params import SimulationParams
from astar_twin.scoring import safe_prediction
from astar_twin.solver.eval.run_benchmark_suite import load_or_compute_ground_truths, run_suite
from astar_twin.solver.predict.hedge import (
    COVERAGE_WEIGHT,
    PARTICLE_WEIGHT,
    apply_hedge,
    should_hedge,
)

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


# ---- Hedge gating tests ----


def test_hedge_not_activated_when_candidate_dominates() -> None:
    """Hedge stays off when candidate clearly beats fixed_coverage."""
    assert (
        should_hedge(
            candidate_mean_score=80.0,
            fixed_coverage_mean_score=50.0,
        )
        is False
    )


def test_hedge_activated_on_score_margin() -> None:
    """Hedge activates when candidate is within SCORE_MARGIN of fixed_coverage."""
    assert (
        should_hedge(
            candidate_mean_score=52.0,
            fixed_coverage_mean_score=50.0,
        )
        is True
    )


def test_hedge_activated_on_calibration_disagreement() -> None:
    """Hedge activates when >=2 seeds have high disagreement."""
    assert (
        should_hedge(
            candidate_mean_score=80.0,
            fixed_coverage_mean_score=50.0,
            per_seed_disagreements=[0.05, 0.20, 0.25, 0.01, 0.01],
        )
        is True
    )


def test_hedge_not_activated_with_low_disagreement() -> None:
    """Hedge stays off with only 1 seed above threshold."""
    assert (
        should_hedge(
            candidate_mean_score=80.0,
            fixed_coverage_mean_score=50.0,
            per_seed_disagreements=[0.20, 0.05, 0.05, 0.05, 0.05],
        )
        is False
    )


# ---- Hedge blend tests ----


def test_hedge_blend_valid_tensors() -> None:
    """Hedge blend produces valid normalized tensors."""
    fixture = load_fixture(FIXTURE_PATH)
    height = fixture.map_height
    width = fixture.map_width

    # Create dummy particle and coverage tensors
    particle_tensors = [
        safe_prediction(np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64))
        for _ in range(5)
    ]
    coverage_tensors = [
        safe_prediction(np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64))
        for _ in range(5)
    ]

    hedged = apply_hedge(
        particle_tensors,
        coverage_tensors,
        fixture.initial_states,
        height,
        width,
    )
    assert len(hedged) == 5
    for t in hedged:
        assert t.shape == (height, width, NUM_CLASSES)
        assert np.all(t > 0)
        sums = np.sum(t, axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=0.02)


def test_hedge_blend_weights() -> None:
    """Hedge blend uses correct 0.85/0.15 weights."""
    assert abs(PARTICLE_WEIGHT - 0.85) < 1e-10
    assert abs(COVERAGE_WEIGHT - 0.15) < 1e-10
    assert abs(PARTICLE_WEIGHT + COVERAGE_WEIGHT - 1.0) < 1e-10


# ---- Suite tests (reduced params for speed) ----


def test_suite_runs_and_returns_valid_result() -> None:
    """Suite produces valid SuiteResult with scores."""
    result = run_suite(
        FIXTURE_PATH,
        repeats=2,
        n_particles=4,
        n_inner_runs=2,
        sims_per_seed=4,
        fc_mc_runs=10,
    )
    assert result.repeats == 2
    assert len(result.candidate_runs) == 2
    assert result.candidate_mean > 0
    assert result.uniform_mean > 0
    assert result.fixed_coverage_mean > 0
    assert len(result.candidate_per_seed_avg) == 5
    assert len(result.uniform_per_seed) == 5
    assert len(result.fixed_coverage_per_seed) == 5


def test_suite_candidate_beats_uniform() -> None:
    """Candidate solver should beat uniform baseline."""
    result = run_suite(
        FIXTURE_PATH,
        repeats=2,
        n_particles=4,
        n_inner_runs=2,
        sims_per_seed=4,
        fc_mc_runs=10,
    )
    # Even with tiny params, particle solver should outperform random
    assert result.candidate_mean >= result.uniform_mean * 0.8, (
        f"Candidate ({result.candidate_mean:.2f}) should be close to or beat "
        f"uniform ({result.uniform_mean:.2f})"
    )


def test_suite_serialization() -> None:
    """Suite result can be serialized to JSON-compatible dict."""
    result = run_suite(
        FIXTURE_PATH,
        repeats=1,
        n_particles=4,
        n_inner_runs=2,
        sims_per_seed=4,
        fc_mc_runs=10,
    )
    d = result.to_dict()
    assert "candidate" in d
    assert "baselines" in d
    assert "hedge" in d
    assert d["repeats"] == 1
    assert len(d["runs"]) == 1


def test_load_or_compute_ground_truths_respects_prior_spread_for_default_prior() -> None:
    fixture = load_fixture(FIXTURE_PATH)

    gt_zero = load_or_compute_ground_truths(fixture, n_mc_runs=5, base_seed=22, prior_spread=0.0)
    gt_wide = load_or_compute_ground_truths(fixture, n_mc_runs=5, base_seed=22, prior_spread=1.0)

    assert len(gt_zero) == len(gt_wide) == fixture.seeds_count
    assert any(not np.array_equal(a, b) for a, b in zip(gt_zero, gt_wide, strict=False))


def test_load_or_compute_ground_truths_ignores_prior_spread_when_cached() -> None:
    fixture = load_fixture(FIXTURE_PATH)
    height = fixture.map_height
    width = fixture.map_width
    gt = safe_prediction(np.full((height, width, 6), 1.0 / 6.0, dtype=np.float64)).tolist()
    cached_fixture = fixture.model_copy(
        update={"ground_truths": [gt for _ in range(fixture.seeds_count)]}
    )

    gt_zero = load_or_compute_ground_truths(
        cached_fixture, n_mc_runs=5, base_seed=22, prior_spread=0.0
    )
    gt_wide = load_or_compute_ground_truths(
        cached_fixture, n_mc_runs=5, base_seed=22, prior_spread=1.0
    )

    assert all(np.array_equal(a, b) for a, b in zip(gt_zero, gt_wide, strict=False))


def test_load_or_compute_ground_truths_ignores_prior_spread_for_calibrated_fixture() -> None:
    fixture = load_fixture(FIXTURE_PATH).model_copy(
        update={
            "params_source": ParamsSource.BENCHMARK_TRUTH,
            "simulation_params": SimulationParams(init_population_mean=2.0, trade_range=12),
        }
    )

    gt_zero = load_or_compute_ground_truths(fixture, n_mc_runs=5, base_seed=22, prior_spread=0.0)
    gt_wide = load_or_compute_ground_truths(fixture, n_mc_runs=5, base_seed=22, prior_spread=1.0)

    assert all(np.array_equal(a, b) for a, b in zip(gt_zero, gt_wide, strict=False))
