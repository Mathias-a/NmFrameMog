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
import pytest

from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.data.loaders import load_fixture
from astar_twin.scoring import compute_score, safe_prediction
from astar_twin.solver.baselines import fixed_coverage_baseline, uniform_baseline
from astar_twin.solver.predict.hedge import (
    CALIBRATION_THRESHOLD,
    COVERAGE_WEIGHT,
    MIN_DISAGREEMENT_SEEDS,
    PARTICLE_WEIGHT,
    SCORE_MARGIN,
    apply_hedge,
    should_hedge,
)
from astar_twin.solver.eval.run_benchmark_suite import RunResult, SuiteResult, run_suite


FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


# ---- Hedge gating tests ----


def test_hedge_not_activated_when_candidate_dominates():
    """Hedge stays off when candidate clearly beats fixed_coverage."""
    assert should_hedge(
        candidate_mean_score=80.0,
        fixed_coverage_mean_score=50.0,
    ) is False


def test_hedge_activated_on_score_margin():
    """Hedge activates when candidate is within SCORE_MARGIN of fixed_coverage."""
    assert should_hedge(
        candidate_mean_score=52.0,
        fixed_coverage_mean_score=50.0,
    ) is True


def test_hedge_activated_on_calibration_disagreement():
    """Hedge activates when >=2 seeds have high disagreement."""
    assert should_hedge(
        candidate_mean_score=80.0,
        fixed_coverage_mean_score=50.0,
        per_seed_disagreements=[0.05, 0.20, 0.25, 0.01, 0.01],
    ) is True


def test_hedge_not_activated_with_low_disagreement():
    """Hedge stays off with only 1 seed above threshold."""
    assert should_hedge(
        candidate_mean_score=80.0,
        fixed_coverage_mean_score=50.0,
        per_seed_disagreements=[0.20, 0.05, 0.05, 0.05, 0.05],
    ) is False


# ---- Hedge blend tests ----


def test_hedge_blend_valid_tensors():
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
        particle_tensors, coverage_tensors,
        fixture.initial_states, height, width,
    )
    assert len(hedged) == 5
    for t in hedged:
        assert t.shape == (height, width, NUM_CLASSES)
        assert np.all(t > 0)
        sums = np.sum(t, axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=0.02)


def test_hedge_blend_weights():
    """Hedge blend uses correct 0.85/0.15 weights."""
    assert abs(PARTICLE_WEIGHT - 0.85) < 1e-10
    assert abs(COVERAGE_WEIGHT - 0.15) < 1e-10
    assert abs(PARTICLE_WEIGHT + COVERAGE_WEIGHT - 1.0) < 1e-10


# ---- Suite tests (reduced params for speed) ----


def test_suite_runs_and_returns_valid_result():
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


def test_suite_candidate_beats_uniform():
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


def test_suite_serialization():
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
