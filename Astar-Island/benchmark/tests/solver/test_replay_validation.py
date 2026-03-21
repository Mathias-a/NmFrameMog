"""Tests for replay validation and ablations.

Covers:
  - Replay validation produces all variants
  - Winner is the highest-mean variant
  - Calibration disagreements are computed per seed
  - Result is JSON-serializable
"""

from __future__ import annotations

from pathlib import Path

from astar_twin.solver.eval.run_replay_validation import (
    run_replay_validation,
)

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


def test_replay_produces_at_least_3_variants() -> None:
    """Replay validation produces uniform, fixed_coverage, and particle_no_hedge."""
    result = run_replay_validation(
        FIXTURE_PATH,
        n_particles=4,
        n_inner_runs=2,
        sims_per_seed=4,
        fc_mc_runs=10,
    )
    variant_names = {v.name for v in result.variants}
    assert "uniform" in variant_names
    assert "fixed_coverage" in variant_names
    assert "particle_no_hedge" in variant_names
    assert len(result.variants) >= 3


def test_replay_winner_is_highest_mean() -> None:
    """Winner should be the variant with highest mean score."""
    result = run_replay_validation(
        FIXTURE_PATH,
        n_particles=4,
        n_inner_runs=2,
        sims_per_seed=4,
        fc_mc_runs=10,
    )
    best = max(result.variants, key=lambda v: v.mean_score)
    assert result.winner_name == best.name
    assert abs(result.winner_mean - best.mean_score) < 1e-6


def test_replay_calibration_disagreements() -> None:
    """Calibration disagreements are computed for 5 seeds."""
    result = run_replay_validation(
        FIXTURE_PATH,
        n_particles=4,
        n_inner_runs=2,
        sims_per_seed=4,
        fc_mc_runs=10,
    )
    assert len(result.calibration_disagreements) == 5
    assert all(d >= 0.0 for d in result.calibration_disagreements)


def test_replay_serialization() -> None:
    """Result serializes to JSON-compatible dict."""
    result = run_replay_validation(
        FIXTURE_PATH,
        n_particles=4,
        n_inner_runs=2,
        sims_per_seed=4,
        fc_mc_runs=10,
    )
    d = result.to_dict()
    assert "variants" in d
    assert "winner" in d
    assert "hedge_activated" in d
    assert "calibration_disagreements" in d
    assert d["winner"]["name"] == result.winner_name


def test_replay_per_seed_scores_present() -> None:
    """Each variant has per_seed_scores for all 5 seeds."""
    result = run_replay_validation(
        FIXTURE_PATH,
        n_particles=4,
        n_inner_runs=2,
        sims_per_seed=4,
        fc_mc_runs=10,
    )
    for v in result.variants:
        assert len(v.per_seed_scores) == 5
        assert all(0 <= s <= 100 for s in v.per_seed_scores)
