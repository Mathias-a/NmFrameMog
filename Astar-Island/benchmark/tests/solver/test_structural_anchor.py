"""Tests for the structural-anchor tensor builder."""

from __future__ import annotations

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import NUM_CLASSES, ClassIndex, TerrainCode
from astar_twin.solver.predict.structural_anchor import (
    COASTAL_TEMPLATE,
    INLAND_TEMPLATE,
    build_structural_anchor,
    build_structural_anchors,
)


def _make_state(grid: list[list[int]]) -> InitialState:
    return InitialState(grid=grid, settlements=[])


class TestBuildStructuralAnchor:
    """Unit tests for build_structural_anchor."""

    def test_output_shape_and_dtype(self) -> None:
        state = _make_state([[10, 11, 4], [11, 1, 11]])
        result = build_structural_anchor(state, height=2, width=3)
        assert result.shape == (2, 3, NUM_CLASSES)
        assert result.dtype == np.float64

    def test_probabilities_sum_to_one(self) -> None:
        state = _make_state([[10, 11, 5], [4, 0, 3]])
        result = build_structural_anchor(state, height=2, width=3)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-9)

    def test_ocean_maps_to_empty(self) -> None:
        state = _make_state([[TerrainCode.OCEAN]])
        result = build_structural_anchor(state, height=1, width=1)
        expected = np.zeros(NUM_CLASSES, dtype=np.float64)
        expected[ClassIndex.EMPTY] = 1.0
        np.testing.assert_array_equal(result[0, 0], expected)

    def test_mountain_maps_to_mountain(self) -> None:
        state = _make_state([[TerrainCode.MOUNTAIN]])
        result = build_structural_anchor(state, height=1, width=1)
        expected = np.zeros(NUM_CLASSES, dtype=np.float64)
        expected[ClassIndex.MOUNTAIN] = 1.0
        np.testing.assert_array_equal(result[0, 0], expected)

    def test_coastal_cell_uses_coastal_template(self) -> None:
        # Cell (0,1) is next to ocean at (0,0) → coastal
        state = _make_state([[TerrainCode.OCEAN, TerrainCode.PLAINS, TerrainCode.PLAINS]])
        result = build_structural_anchor(state, height=1, width=3)
        np.testing.assert_allclose(result[0, 1], COASTAL_TEMPLATE)

    def test_inland_cell_uses_inland_template(self) -> None:
        # Cell (1,1) surrounded by plains only → inland
        state = _make_state(
            [
                [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
                [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
                [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
            ]
        )
        result = build_structural_anchor(state, height=3, width=3)
        np.testing.assert_allclose(result[1, 1], INLAND_TEMPLATE)

    def test_port_probability_zero_for_inland(self) -> None:
        state = _make_state(
            [
                [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
                [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
            ]
        )
        result = build_structural_anchor(state, height=2, width=3)
        assert result[1, 1, ClassIndex.PORT] == 0.0

    def test_port_probability_positive_for_coastal(self) -> None:
        state = _make_state(
            [
                [TerrainCode.OCEAN, TerrainCode.PLAINS],
            ]
        )
        result = build_structural_anchor(state, height=1, width=2)
        assert result[0, 1, ClassIndex.PORT] > 0.0

    def test_diagonal_ocean_does_not_make_coastal(self) -> None:
        state = _make_state(
            [
                [TerrainCode.OCEAN, TerrainCode.PLAINS],
                [TerrainCode.PLAINS, TerrainCode.PLAINS],
            ]
        )
        result = build_structural_anchor(state, height=2, width=2)
        # (1,1) is diagonally adjacent to ocean but not 4-connected
        np.testing.assert_allclose(result[1, 1], INLAND_TEMPLATE)

    @pytest.mark.parametrize(
        "code",
        [
            TerrainCode.EMPTY,
            TerrainCode.PLAINS,
            TerrainCode.FOREST,
            TerrainCode.SETTLEMENT,
            TerrainCode.PORT,
            TerrainCode.RUIN,
        ],
    )
    def test_non_static_inland_cells_all_use_inland_template(self, code: int) -> None:
        state = _make_state(
            [
                [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
                [TerrainCode.PLAINS, code, TerrainCode.PLAINS],
                [TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.PLAINS],
            ]
        )
        result = build_structural_anchor(state, height=3, width=3)
        np.testing.assert_allclose(result[1, 1], INLAND_TEMPLATE)


class TestBuildStructuralAnchors:
    """Tests for build_structural_anchors (multi-seed batch)."""

    def test_returns_list_of_correct_length(self) -> None:
        states = [
            _make_state([[TerrainCode.OCEAN]]),
            _make_state([[TerrainCode.PLAINS]]),
            _make_state([[TerrainCode.MOUNTAIN]]),
        ]
        result = build_structural_anchors(states, height=1, width=1)
        assert len(result) == 3

    def test_each_anchor_matches_single_build(self) -> None:
        states = [
            _make_state([[TerrainCode.OCEAN, TerrainCode.PLAINS]]),
            _make_state([[TerrainCode.MOUNTAIN, TerrainCode.FOREST]]),
        ]
        batch = build_structural_anchors(states, height=1, width=2)
        for i, state in enumerate(states):
            single = build_structural_anchor(state, height=1, width=2)
            np.testing.assert_array_equal(batch[i], single)

    def test_empty_list(self) -> None:
        result = build_structural_anchors([], height=1, width=1)
        assert result == []
