from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import ParamsSource
from astar_twin.fixture_prep.ground_truth import compute_and_attach_ground_truths
from astar_twin.params import SimulationParams

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def base_fixture() -> object:
    return load_fixture(FIXTURE_PATH)


def test_returns_new_fixture_with_ground_truths_populated(base_fixture: object) -> None:
    result = compute_and_attach_ground_truths(base_fixture, n_runs=5, base_seed=0)  # type: ignore[arg-type]
    assert result.ground_truths is not None


def test_ground_truths_shape_matches_fixture(base_fixture: object) -> None:
    result = compute_and_attach_ground_truths(base_fixture, n_runs=5, base_seed=0)  # type: ignore[arg-type]
    gt = result.ground_truths
    assert gt is not None
    assert len(gt) == base_fixture.seeds_count  # type: ignore[union-attr]
    for seed_gt in gt:
        arr = np.array(seed_gt)
        assert arr.shape == (base_fixture.map_height, base_fixture.map_width, 6)  # type: ignore[union-attr]


def test_ground_truths_probabilities_sum_to_one(base_fixture: object) -> None:
    result = compute_and_attach_ground_truths(base_fixture, n_runs=5, base_seed=0)  # type: ignore[arg-type]
    assert result.ground_truths is not None
    for seed_gt in result.ground_truths:
        arr = np.array(seed_gt, dtype=np.float64)
        sums = arr.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_original_fixture_is_unchanged(base_fixture: object) -> None:
    original_gt = base_fixture.ground_truths  # type: ignore[union-attr]
    compute_and_attach_ground_truths(base_fixture, n_runs=5, base_seed=0)  # type: ignore[arg-type]
    assert base_fixture.ground_truths == original_gt  # type: ignore[union-attr]


def test_different_seeds_produce_different_results(base_fixture: object) -> None:
    result_a = compute_and_attach_ground_truths(base_fixture, n_runs=5, base_seed=0)  # type: ignore[arg-type]
    result_b = compute_and_attach_ground_truths(base_fixture, n_runs=5, base_seed=999)  # type: ignore[arg-type]
    # With very few runs, distributions should differ when seeds differ
    assert result_a.ground_truths is not None
    assert result_b.ground_truths is not None
    arr_a = np.array(result_a.ground_truths[0])
    arr_b = np.array(result_b.ground_truths[0])
    assert not np.allclose(arr_a, arr_b)


def test_default_prior_fixture_samples_reproducible_benchmark_params(base_fixture: object) -> None:
    result_a = compute_and_attach_ground_truths(base_fixture, n_runs=5, base_seed=123)  # type: ignore[arg-type]
    result_b = compute_and_attach_ground_truths(base_fixture, n_runs=5, base_seed=123)  # type: ignore[arg-type]

    assert result_a.simulation_params == result_b.simulation_params  # type: ignore[union-attr]
    assert result_a.simulation_params != SimulationParams()  # type: ignore[union-attr]


def test_default_prior_fixture_sampling_changes_with_seed(base_fixture: object) -> None:
    result_a = compute_and_attach_ground_truths(base_fixture, n_runs=5, base_seed=123)  # type: ignore[arg-type]
    result_b = compute_and_attach_ground_truths(base_fixture, n_runs=5, base_seed=124)  # type: ignore[arg-type]

    assert result_a.simulation_params != result_b.simulation_params  # type: ignore[union-attr]


def test_calibrated_fixture_keeps_explicit_simulation_params(base_fixture: object) -> None:
    calibrated_fixture = base_fixture.model_copy(  # type: ignore[union-attr]
        update={
            "params_source": ParamsSource.BENCHMARK_TRUTH,
            "simulation_params": SimulationParams(init_population_mean=2.1, trade_range=11),
        }
    )

    result = compute_and_attach_ground_truths(calibrated_fixture, n_runs=5, base_seed=123)

    assert result.simulation_params == calibrated_fixture.simulation_params


def test_zero_prior_spread_keeps_numeric_defaults(base_fixture: object) -> None:
    result = compute_and_attach_ground_truths(  # type: ignore[arg-type]
        base_fixture,
        n_runs=5,
        base_seed=123,
        prior_spread=0.0,
    )

    assert result.simulation_params.init_population_mean == SimulationParams().init_population_mean  # type: ignore[union-attr]
    assert result.simulation_params.trade_range == SimulationParams().trade_range  # type: ignore[union-attr]


def test_prior_spread_ignored_for_calibrated_fixture(base_fixture: object) -> None:
    calibrated_fixture = base_fixture.model_copy(  # type: ignore[union-attr]
        update={
            "params_source": ParamsSource.BENCHMARK_TRUTH,
            "simulation_params": SimulationParams(init_population_mean=2.1, trade_range=11),
        }
    )

    result_a = compute_and_attach_ground_truths(
        calibrated_fixture,
        n_runs=5,
        base_seed=123,
        prior_spread=0.0,
    )
    result_b = compute_and_attach_ground_truths(
        calibrated_fixture,
        n_runs=5,
        base_seed=123,
        prior_spread=1.0,
    )

    assert result_a.simulation_params == calibrated_fixture.simulation_params
    assert result_b.simulation_params == calibrated_fixture.simulation_params
