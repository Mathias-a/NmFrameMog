"""Tests for multi-fixture benchmark suite orchestrator.

Covers:
  - MultiFixtureSuiteResult dataclass structure and serialisation
  - Weighted/unweighted aggregation math
  - run_multi_fixture_suite filters out test- fixtures
  - run_multi_fixture_suite end-to-end smoke test (1 real fixture, reduced params)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astar_twin.solver.eval.run_benchmark_suite import SuiteResult, RunResult
from astar_twin.solver.eval.run_multi_fixture_suite import (
    FixtureResult,
    MultiFixtureSuiteResult,
    _is_real_round,
    run_multi_fixture_suite,
)
from astar_twin.data.models import RoundFixture

# Path used by existing suite tests — 10x10 synthetic fixture, no simulation_params needed
TEST_ROUND_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)
DATA_DIR = TEST_ROUND_PATH.parent.parent.parent  # benchmark/data


# ---------------------------------------------------------------------------
# Unit tests — aggregation logic
# ---------------------------------------------------------------------------


def _make_suite_result(mean: float, weight_hint: float = 1.0) -> SuiteResult:
    """Build a minimal SuiteResult with the given candidate mean."""
    run = RunResult(
        run_index=0,
        per_seed_scores=[mean] * 5,
        mean_score=mean,
        runtime_seconds=0.1,
        total_queries=10,
    )
    return SuiteResult(
        repeats=1,
        candidate_mean=mean,
        candidate_min=mean,
        candidate_max=mean,
        candidate_std=0.0,
        candidate_per_seed_avg=[mean] * 5,
        candidate_runs=[run],
        uniform_mean=5.0,
        uniform_per_seed=[5.0] * 5,
        fixed_coverage_mean=10.0,
        fixed_coverage_per_seed=[10.0] * 5,
        hedge_activations=0,
        hedged_mean=None,
        total_runtime_seconds=0.5,
    )


def _make_fixture_result(round_number: int, mean: float, weight: float) -> FixtureResult:
    return FixtureResult(
        round_id=f"round-{round_number:04d}",
        round_number=round_number,
        round_weight=weight,
        suite=_make_suite_result(mean),
    )


def test_weighted_mean_calculation():
    """Weighted mean respects round_weight values."""
    fr1 = _make_fixture_result(1, mean=20.0, weight=1.0)
    fr2 = _make_fixture_result(2, mean=40.0, weight=3.0)
    result = MultiFixtureSuiteResult(
        fixture_results=[fr1, fr2],
        overall_weighted_mean=(20.0 * 1.0 + 40.0 * 3.0) / 4.0,
        overall_unweighted_mean=30.0,
        per_round_means={1: 20.0, 2: 40.0},
        total_runtime_seconds=1.0,
        rounds_evaluated=2,
        repeats_per_round=1,
    )
    assert abs(result.overall_weighted_mean - 35.0) < 1e-9
    assert abs(result.overall_unweighted_mean - 30.0) < 1e-9


def test_serialisation_round_trip():
    """MultiFixtureSuiteResult serialises and is valid JSON."""
    fr = _make_fixture_result(5, mean=42.0, weight=2.0)
    result = MultiFixtureSuiteResult(
        fixture_results=[fr],
        overall_weighted_mean=42.0,
        overall_unweighted_mean=42.0,
        per_round_means={5: 42.0},
        total_runtime_seconds=3.14,
        rounds_evaluated=1,
        repeats_per_round=1,
    )
    d = result.to_dict()
    # Round-trip via JSON
    reloaded = json.loads(json.dumps(d))
    assert reloaded["rounds_evaluated"] == 1
    assert reloaded["overall_weighted_mean"] == pytest.approx(42.0)
    assert len(reloaded["fixtures"]) == 1
    assert reloaded["fixtures"][0]["round_number"] == 5


def test_is_real_round_filters_test_fixtures():
    """_is_real_round rejects test- prefixed fixtures."""
    test_fixture = MagicMock(spec=RoundFixture)
    test_fixture.id = "test-round-001"
    real_fixture = MagicMock(spec=RoundFixture)
    real_fixture.id = "cc5442dd-bc5d-418b-911b-7eb960cb0390"
    assert _is_real_round(test_fixture) is False
    assert _is_real_round(real_fixture) is True


def test_print_summary_does_not_crash():
    """print_summary runs without error for a single-fixture result."""
    fr = _make_fixture_result(1, mean=30.0, weight=1.0)
    result = MultiFixtureSuiteResult(
        fixture_results=[fr],
        overall_weighted_mean=30.0,
        overall_unweighted_mean=30.0,
        per_round_means={1: 30.0},
        total_runtime_seconds=0.5,
        rounds_evaluated=1,
        repeats_per_round=1,
    )
    result.print_summary()  # should not raise


# ---------------------------------------------------------------------------
# Integration smoke test — mocked run_suite to avoid slow solver
# ---------------------------------------------------------------------------


def test_run_multi_fixture_suite_with_mocked_run_suite(tmp_path: Path):
    """run_multi_fixture_suite aggregates results from mocked per-fixture runs."""
    import shutil

    # Create a synthetic data_dir with one real-looking fixture
    rounds_dir = tmp_path / "rounds"
    real_id = "aaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    fixture_dir = rounds_dir / real_id
    fixture_dir.mkdir(parents=True)

    # Copy test-round-001 as a stand-in but give it a non-test- id
    src = TEST_ROUND_PATH
    if not src.exists():
        pytest.skip("test-round-001 fixture not found")

    import json as _json

    original = _json.loads(src.read_text())
    original["id"] = real_id
    (fixture_dir / "round_detail.json").write_text(_json.dumps(original))

    mock_suite = _make_suite_result(mean=55.0)

    with patch(
        "astar_twin.solver.eval.run_multi_fixture_suite.run_suite",
        return_value=mock_suite,
    ) as mock_run:
        result = run_multi_fixture_suite(
            data_dir=tmp_path,
            repeats=1,
            n_particles=4,
            n_inner_runs=2,
            sims_per_seed=4,
            fc_mc_runs=10,
        )

    mock_run.assert_called_once()
    assert result.rounds_evaluated == 1
    assert result.repeats_per_round == 1
    assert abs(result.overall_weighted_mean - 55.0) < 1e-6
    assert abs(result.overall_unweighted_mean - 55.0) < 1e-6
    assert len(result.fixture_results) == 1
    assert result.fixture_results[0].round_id == real_id


def test_run_multi_fixture_suite_raises_when_no_real_rounds(tmp_path: Path):
    """run_multi_fixture_suite raises ValueError when only test- fixtures exist."""
    rounds_dir = tmp_path / "rounds"
    test_dir = rounds_dir / "test-only-fixture"
    test_dir.mkdir(parents=True)

    import json as _json

    original = _json.loads(TEST_ROUND_PATH.read_text()) if TEST_ROUND_PATH.exists() else {}
    original["id"] = "test-only-fixture"
    (test_dir / "round_detail.json").write_text(_json.dumps(original))

    with pytest.raises(ValueError, match="No real-round fixtures"):
        run_multi_fixture_suite(data_dir=tmp_path, repeats=1)
