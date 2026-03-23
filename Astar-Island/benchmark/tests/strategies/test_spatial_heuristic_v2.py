from __future__ import annotations

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialSettlement, InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.harness.budget import Budget
from astar_twin.strategies import REGISTRY
from astar_twin.strategies.spatial_heuristic_v2.strategy import SpatialHeuristicV2Strategy


def _make_state(
    grid: list[list[int]],
    settlements: list[InitialSettlement] | None = None,
) -> InitialState:
    return InitialState(grid=grid, settlements=settlements or [])


@pytest.fixture
def strategy() -> SpatialHeuristicV2Strategy:
    return SpatialHeuristicV2Strategy()


def test_name(strategy: SpatialHeuristicV2Strategy) -> None:
    assert strategy.name == "spatial_heuristic_v2"


def test_output_shape_and_dtype(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state([[10, 11, 4], [11, 1, 11]])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result.shape == (2, 3, NUM_CLASSES)
    assert result.dtype == np.float64


def test_probabilities_sum_to_one(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state([[10, 11, 5], [4, 0, 3]])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-9)


def test_ocean_and_mountain_deterministic(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state([[TerrainCode.OCEAN, TerrainCode.MOUNTAIN]])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    np.testing.assert_array_equal(
        result[0, 0], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.EMPTY]
    )
    np.testing.assert_array_equal(
        result[0, 1], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.MOUNTAIN]
    )


def test_settlement_cell_favours_settlement_and_ruin(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state(
        [
            [TerrainCode.FOREST, TerrainCode.FOREST, TerrainCode.FOREST],
            [TerrainCode.FOREST, TerrainCode.SETTLEMENT, TerrainCode.FOREST],
            [TerrainCode.FOREST, TerrainCode.FOREST, TerrainCode.FOREST],
        ],
        settlements=[InitialSettlement(x=1, y=1, has_port=False, alive=True)],
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    cell = result[1, 1]
    dynamic_mass = cell[ClassIndex.SETTLEMENT] + cell[ClassIndex.PORT] + cell[ClassIndex.RUIN]
    assert dynamic_mass > 0.60


def test_coastal_settlement_has_port_probability(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state(
        [[TerrainCode.OCEAN, TerrainCode.SETTLEMENT, TerrainCode.PLAINS]],
        settlements=[InitialSettlement(x=1, y=0, has_port=False, alive=True)],
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result[0, 1, ClassIndex.PORT] > 0.0


def test_inland_settlement_has_no_port(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state(
        [
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.SETTLEMENT, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
        ],
        settlements=[InitialSettlement(x=1, y=1, has_port=False, alive=True)],
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result[1, 1, ClassIndex.PORT] == 0.0


def test_empty_near_settlement_has_expansion_probability(
    strategy: SpatialHeuristicV2Strategy,
) -> None:
    state = _make_state(
        [
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.SETTLEMENT, TerrainCode.PLAINS],
            [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
        ],
        settlements=[InitialSettlement(x=1, y=1, has_port=False, alive=True)],
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result[0, 1, ClassIndex.SETTLEMENT] > result[0, 1, ClassIndex.MOUNTAIN]


def test_empty_far_from_settlement_favours_empty(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state([[TerrainCode.PLAINS] * 10] * 10, settlements=[])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result[5, 5, ClassIndex.EMPTY] > 0.80


def test_forest_mostly_stays_forest(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state([[TerrainCode.FOREST] * 5] * 5, settlements=[])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result[2, 2, ClassIndex.FOREST] > 0.70


def test_has_port_flag_increases_port_probability(strategy: SpatialHeuristicV2Strategy) -> None:
    no_port_state = _make_state(
        [[TerrainCode.OCEAN, TerrainCode.SETTLEMENT, TerrainCode.PLAINS]],
        settlements=[InitialSettlement(x=1, y=0, has_port=False, alive=True)],
    )
    has_port_state = _make_state(
        [[TerrainCode.OCEAN, TerrainCode.SETTLEMENT, TerrainCode.PLAINS]],
        settlements=[InitialSettlement(x=1, y=0, has_port=True, alive=True)],
    )
    no_port_result = strategy.predict(no_port_state, budget=Budget(total=50), base_seed=0)
    has_port_result = strategy.predict(has_port_state, budget=Budget(total=50), base_seed=0)
    assert has_port_result[0, 1, ClassIndex.PORT] > no_port_result[0, 1, ClassIndex.PORT]


def test_deterministic_across_seeds(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state([[10, 11, 4], [11, 5, 3]])
    r1 = strategy.predict(state, budget=Budget(total=10), base_seed=0)
    r2 = strategy.predict(state, budget=Budget(total=50), base_seed=999)
    np.testing.assert_array_equal(r1, r2)


def test_all_probabilities_in_unit_interval(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state(
        [[10, 11, 4, 5], [0, 1, 2, 3]],
        settlements=[
            InitialSettlement(x=1, y=1, has_port=False, alive=True),
            InitialSettlement(x=2, y=1, has_port=True, alive=True),
        ],
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_registry_contains_spatial_heuristic_v2() -> None:
    assert REGISTRY["spatial_heuristic_v2"] is SpatialHeuristicV2Strategy


def test_does_not_consume_budget(strategy: SpatialHeuristicV2Strategy) -> None:
    state = _make_state([[TerrainCode.PLAINS]])
    budget = Budget(total=50)
    strategy.predict(state, budget=budget, base_seed=0)
    assert budget.remaining == 50
