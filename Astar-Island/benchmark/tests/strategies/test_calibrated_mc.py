from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialSettlement, InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.data.loaders import load_fixture
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.strategies import REGISTRY
from astar_twin.strategies.calibrated_mc.strategy import (
    CalibratedMCStrategy,
    Zone,
    _apply_temperature_scaling,
    _build_coastal_mask,
    _build_static_mask,
    _build_template_prior,
    _fill_static_cells,
)

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


def _baseline_reference_prediction(
    strategy: CalibratedMCStrategy,
    initial_state: InitialState,
    base_seed: int,
) -> np.ndarray:
    grid = initial_state.grid
    height = len(grid)
    width = len(grid[0])
    simulator = Simulator(params=strategy._params)
    runner = MCRunner(simulator)
    runs = runner.run_batch(initial_state, strategy._n_runs, base_seed=base_seed)
    mc_tensor = aggregate_runs(runs, height, width)

    is_static = _build_static_mask(grid, height, width)
    is_coastal = _build_coastal_mask(grid, height, width)
    mc_scaled = _apply_temperature_scaling(mc_tensor, is_static, strategy._temperature)
    prior = _build_template_prior(grid, height, width, is_static, is_coastal)

    result = np.empty((height, width, NUM_CLASSES), dtype=np.float64)
    dynamic = ~is_static
    result[dynamic] = (
        strategy._prior_weight * prior[dynamic]
        + (1.0 - strategy._prior_weight) * mc_scaled[dynamic]
    )
    _fill_static_cells(
        result,
        grid,
        height,
        width,
        is_static,
        strategy._static_confidence,
    )
    sums = np.sum(result, axis=2, keepdims=True)
    sums = np.maximum(sums, 1e-10)
    return result / sums


@pytest.fixture
def first_initial_state() -> InitialState:
    fixture = load_fixture(FIXTURE_PATH)
    return fixture.initial_states[0]


@pytest.fixture
def strategy() -> CalibratedMCStrategy:
    return CalibratedMCStrategy(n_runs=20)


@pytest.fixture
def baseline_strategy() -> CalibratedMCStrategy:
    return CalibratedMCStrategy(
        n_runs=20,
        use_settlement_zones=False,
        use_adaptive_blend=False,
        use_mc_variance=False,
    )


@pytest.fixture
def port_initial_state() -> InitialState:
    return InitialState(
        grid=[
            [10, 10, 10, 10, 10, 10],
            [10, 11, 11, 11, 11, 10],
            [10, 11, 1, 11, 11, 10],
            [10, 11, 11, 11, 11, 10],
            [10, 11, 11, 11, 11, 10],
            [5, 5, 5, 5, 5, 5],
        ],
        settlements=[InitialSettlement(x=2, y=2, has_port=True, alive=True)],
    )


def test_name(strategy: CalibratedMCStrategy, baseline_strategy: CalibratedMCStrategy) -> None:
    assert strategy.name == "calibrated_mc"
    assert baseline_strategy.name == "calibrated_mc_v1"


def test_output_shape(strategy: CalibratedMCStrategy, first_initial_state: InitialState) -> None:
    height = len(first_initial_state.grid)
    width = len(first_initial_state.grid[0])
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.shape == (height, width, NUM_CLASSES)


def test_probabilities_sum_to_one(
    strategy: CalibratedMCStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_output_dtype_is_float64(
    strategy: CalibratedMCStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.dtype == np.float64


def test_deterministic_with_same_seed(
    strategy: CalibratedMCStrategy, first_initial_state: InitialState
) -> None:
    r1 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    r2 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_different_seeds_produce_different_results(
    strategy: CalibratedMCStrategy, first_initial_state: InitialState
) -> None:
    r1 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    r2 = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=12345)
    assert not np.allclose(r1, r2)


def test_all_probabilities_in_unit_interval(
    strategy: CalibratedMCStrategy, first_initial_state: InitialState
) -> None:
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_no_zero_probabilities_after_safe_prediction(
    strategy: CalibratedMCStrategy, first_initial_state: InitialState
) -> None:
    from astar_twin.scoring import safe_prediction

    raw = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)
    safe = safe_prediction(raw)
    assert float(np.min(safe)) >= 0.01 - 1e-9


def test_static_cells_have_high_confidence(first_initial_state: InitialState) -> None:
    strategy = CalibratedMCStrategy(n_runs=20, static_confidence=0.97)
    result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=0)

    grid = first_initial_state.grid
    for y, row in enumerate(grid):
        for x, code in enumerate(row):
            if code == TerrainCode.OCEAN:
                assert result[y, x, ClassIndex.EMPTY] >= 0.90
            elif code == TerrainCode.MOUNTAIN:
                assert result[y, x, ClassIndex.MOUNTAIN] >= 0.90


def test_baseline_equivalence(
    baseline_strategy: CalibratedMCStrategy,
    first_initial_state: InitialState,
) -> None:
    reference = _baseline_reference_prediction(baseline_strategy, first_initial_state, base_seed=7)
    actual = baseline_strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=7)
    np.testing.assert_array_equal(actual, reference)


def test_zone_map_shape_and_values(
    strategy: CalibratedMCStrategy,
    first_initial_state: InitialState,
) -> None:
    grid = first_initial_state.grid
    height = len(grid)
    width = len(grid[0])
    is_static = _build_static_mask(grid, height, width)
    is_coastal = _build_coastal_mask(grid, height, width)
    zone_map = strategy._build_zone_map(first_initial_state, height, width, is_static, is_coastal)

    assert zone_map.shape == (height, width)
    valid_values = {int(zone) for zone in Zone}
    assert set(np.unique(zone_map).tolist()).issubset(valid_values)


def test_zone_map_core_near_settlements(
    strategy: CalibratedMCStrategy,
    first_initial_state: InitialState,
) -> None:
    grid = first_initial_state.grid
    height = len(grid)
    width = len(grid[0])
    is_static = _build_static_mask(grid, height, width)
    is_coastal = _build_coastal_mask(grid, height, width)
    zone_map = strategy._build_zone_map(first_initial_state, height, width, is_static, is_coastal)

    for settlement in first_initial_state.settlements:
        if settlement.alive:
            assert zone_map[settlement.y, settlement.x] == Zone.CORE


def test_zone_map_marks_coastal_hub_near_port_settlement(
    strategy: CalibratedMCStrategy,
    port_initial_state: InitialState,
) -> None:
    grid = port_initial_state.grid
    height = len(grid)
    width = len(grid[0])
    is_static = _build_static_mask(grid, height, width)
    is_coastal = _build_coastal_mask(grid, height, width)
    zone_map = strategy._build_zone_map(port_initial_state, height, width, is_static, is_coastal)

    assert zone_map[1, 2] == Zone.COASTAL_HUB


def test_zone_prior_shape(
    strategy: CalibratedMCStrategy,
    first_initial_state: InitialState,
) -> None:
    grid = first_initial_state.grid
    height = len(grid)
    width = len(grid[0])
    is_static = _build_static_mask(grid, height, width)
    is_coastal = _build_coastal_mask(grid, height, width)
    zone_map = strategy._build_zone_map(first_initial_state, height, width, is_static, is_coastal)
    prior = strategy._build_zone_prior(zone_map, height, width, is_static)

    assert prior.shape == (height, width, NUM_CLASSES)
    np.testing.assert_allclose(prior.sum(axis=-1), 1.0, atol=1e-9)


def test_subbatch_variance_shape(first_initial_state: InitialState) -> None:
    strategy = CalibratedMCStrategy(n_runs=20, n_subbatches=5, use_mc_variance=True)
    grid = first_initial_state.grid
    height = len(grid)
    width = len(grid[0])
    mc_tensor, variance = strategy._run_mc_with_variance(
        first_initial_state,
        height,
        width,
        base_seed=0,
    )

    assert mc_tensor.shape == (height, width, NUM_CLASSES)
    assert variance.shape == (height, width, NUM_CLASSES)
    assert np.all(variance >= 0.0)


def test_adaptive_weights_shape(
    strategy: CalibratedMCStrategy,
    first_initial_state: InitialState,
) -> None:
    grid = first_initial_state.grid
    height = len(grid)
    width = len(grid[0])
    is_static = _build_static_mask(grid, height, width)
    is_coastal = _build_coastal_mask(grid, height, width)
    zone_map = strategy._build_zone_map(first_initial_state, height, width, is_static, is_coastal)
    _, variance = strategy._run_mc_with_variance(first_initial_state, height, width, base_seed=0)
    weights = strategy._compute_adaptive_weights(zone_map, height, width, variance)

    assert weights.shape == (height, width)
    assert float(weights.min()) >= 0.1
    assert float(weights.max()) <= 0.8


def test_all_variants_produce_valid_output(first_initial_state: InitialState) -> None:
    height = len(first_initial_state.grid)
    width = len(first_initial_state.grid[0])
    calibrated_variants = {
        name: cls for name, cls in REGISTRY.items() if name.startswith("calibrated_mc")
    }

    assert len(calibrated_variants) >= 5

    for cls in calibrated_variants.values():
        strategy = cls()
        result = strategy.predict(first_initial_state, budget=Budget(total=5), base_seed=11)
        assert result.shape == (height, width, NUM_CLASSES)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_feature_flags_change_output(first_initial_state: InitialState) -> None:
    baseline = CalibratedMCStrategy(
        n_runs=20,
        use_settlement_zones=False,
        use_adaptive_blend=False,
        use_mc_variance=False,
    )
    improved = CalibratedMCStrategy(n_runs=20)

    baseline_result = baseline.predict(first_initial_state, budget=Budget(total=5), base_seed=3)
    improved_result = improved.predict(first_initial_state, budget=Budget(total=5), base_seed=3)

    assert not np.allclose(baseline_result, improved_result)
