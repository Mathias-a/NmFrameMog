from __future__ import annotations

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialSettlement, InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.harness.budget import Budget
from astar_twin.strategies import REGISTRY
from astar_twin.strategies.spatial_heuristic.strategy import SpatialHeuristicStrategy


def _make_state(
    grid: list[list[int]],
    settlements: list[InitialSettlement] | None = None,
) -> InitialState:
    return InitialState(grid=grid, settlements=settlements or [])


@pytest.fixture
def strategy() -> SpatialHeuristicStrategy:
    return SpatialHeuristicStrategy()


def test_name(strategy: SpatialHeuristicStrategy) -> None:
    assert strategy.name == "spatial_heuristic"


def test_output_shape_and_dtype(strategy: SpatialHeuristicStrategy) -> None:
    state = _make_state([[10, 11, 4], [11, 1, 11]])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result.shape == (2, 3, NUM_CLASSES)
    assert result.dtype == np.float64


def test_probabilities_sum_to_one(strategy: SpatialHeuristicStrategy) -> None:
    state = _make_state([[10, 11, 5], [4, 0, 3]])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-9)


def test_ocean_and_mountain_deterministic(strategy: SpatialHeuristicStrategy) -> None:
    state = _make_state([[TerrainCode.OCEAN, TerrainCode.MOUNTAIN]])
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    np.testing.assert_array_equal(
        result[0, 0], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.EMPTY]
    )
    np.testing.assert_array_equal(
        result[0, 1], np.eye(NUM_CLASSES, dtype=np.float64)[ClassIndex.MOUNTAIN]
    )


def test_settlement_cell_favours_settlement_and_ruin(
    strategy: SpatialHeuristicStrategy,
) -> None:
    """A settlement cell should have high Settlement+Port+Ruin probability."""
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
    # Settlement + Ruin should be a large fraction
    dynamic_mass = cell[ClassIndex.SETTLEMENT] + cell[ClassIndex.PORT] + cell[ClassIndex.RUIN]
    assert dynamic_mass > 0.60


def test_coastal_settlement_has_port_probability(
    strategy: SpatialHeuristicStrategy,
) -> None:
    state = _make_state(
        [
            [TerrainCode.OCEAN, TerrainCode.SETTLEMENT, TerrainCode.PLAINS],
        ],
        settlements=[InitialSettlement(x=1, y=0, has_port=False, alive=True)],
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result[0, 1, ClassIndex.PORT] > 0.0


def test_inland_settlement_has_no_port(strategy: SpatialHeuristicStrategy) -> None:
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
    strategy: SpatialHeuristicStrategy,
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
    # Adjacent cell should have settlement expansion probability
    assert result[0, 1, ClassIndex.SETTLEMENT] > result[0, 1, ClassIndex.MOUNTAIN]


def test_empty_far_from_settlement_favours_empty(
    strategy: SpatialHeuristicStrategy,
) -> None:
    """A plains cell with no settlements nearby should strongly predict Empty."""
    state = _make_state(
        [[TerrainCode.PLAINS] * 10] * 10,
        settlements=[],
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    # With no settlements at all, every cell should be mostly empty
    assert result[5, 5, ClassIndex.EMPTY] > 0.80


def test_forest_mostly_stays_forest(strategy: SpatialHeuristicStrategy) -> None:
    state = _make_state(
        [[TerrainCode.FOREST] * 5] * 5,
        settlements=[],
    )
    result = strategy.predict(state, budget=Budget(total=50), base_seed=0)
    assert result[2, 2, ClassIndex.FOREST] > 0.70


def test_deterministic_across_seeds(strategy: SpatialHeuristicStrategy) -> None:
    state = _make_state([[10, 11, 4], [11, 5, 3]])
    r1 = strategy.predict(state, budget=Budget(total=10), base_seed=0)
    r2 = strategy.predict(state, budget=Budget(total=50), base_seed=999)
    np.testing.assert_array_equal(r1, r2)


def test_all_probabilities_in_unit_interval(strategy: SpatialHeuristicStrategy) -> None:
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


def test_registry_contains_spatial_heuristic() -> None:
    assert REGISTRY["spatial_heuristic"] is SpatialHeuristicStrategy


def test_does_not_consume_budget(strategy: SpatialHeuristicStrategy) -> None:
    state = _make_state([[TerrainCode.PLAINS]])
    budget = Budget(total=50)
    strategy.predict(state, budget=budget, base_seed=0)
    assert budget.remaining == 50
