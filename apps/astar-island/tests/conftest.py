"""Shared pytest fixtures for astar_island tests."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from astar_island.terrain import NUM_PREDICTION_CLASSES, TerrainType

# ── Grid size used by most unit tests (small for speed) ──
TEST_W: int = 5
TEST_H: int = 5
K: int = NUM_PREDICTION_CLASSES  # 6


@pytest.fixture()
def small_raw_grid() -> list[list[int]]:
    """5x5 raw terrain grid (grid[x][y]) with varied terrain codes.

    Layout (x=col, y=row):
      x=0: Ocean  column  (code 10)
      x=1: Plains column  (code 11)
      x=2: Mixed land     (Forest, Mountain, Settlement, Ruin, Empty)
      x=3: Forest column  (code 4)
      x=4: Mountain col   (code 5)
    """
    return [
        [10, 10, 10, 10, 10],  # x=0: all ocean
        [11, 11, 11, 11, 11],  # x=1: all plains (→ class 0)
        [0, 1, 2, 3, 4],  # x=2: empty, settlement, port, ruin, forest
        [4, 4, 4, 4, 4],  # x=3: all forest
        [5, 5, 5, 5, 5],  # x=4: all mountain
    ]


@pytest.fixture()
def uniform_prediction() -> NDArray[np.float64]:
    """5x5x6 uniform prediction (1/6 each class)."""
    pred = np.full((TEST_W, TEST_H, K), 1.0 / K, dtype=np.float64)
    return pred


@pytest.fixture()
def one_hot_ground_truth() -> NDArray[np.float64]:
    """5x5x6 ground truth with deterministic (one-hot) cells.

    Each cell is one-hot for class = (x + y) % K.
    """
    gt = np.zeros((TEST_W, TEST_H, K), dtype=np.float64)
    for x in range(TEST_W):
        for y in range(TEST_H):
            gt[x, y, (x + y) % K] = 1.0
    return gt


@pytest.fixture()
def mixed_ground_truth() -> NDArray[np.float64]:
    """5x5x6 ground truth with a mix of deterministic and stochastic cells.

    Even cells: one-hot for class 0.
    Odd cells: 50% class 0, 30% class 1, 20% class 2.
    """
    gt = np.zeros((TEST_W, TEST_H, K), dtype=np.float64)
    for x in range(TEST_W):
        for y in range(TEST_H):
            if (x + y) % 2 == 0:
                gt[x, y, 0] = 1.0
            else:
                gt[x, y, 0] = 0.5
                gt[x, y, 1] = 0.3
                gt[x, y, 2] = 0.2
    return gt


@pytest.fixture()
def full_size_raw_grid() -> list[list[int]]:
    """40x40 raw terrain grid for full-size tests.

    Pattern: outer 2 rows/cols ocean, inner land with mixed terrain.
    """
    grid: list[list[int]] = []
    for x in range(40):
        col: list[int] = []
        for y in range(40):
            if x < 2 or x >= 38 or y < 2 or y >= 38:
                col.append(int(TerrainType.OCEAN))
            else:
                code = [0, 1, 2, 3, 4, 5, 11][(x + y) % 7]
                col.append(code)
        grid.append(col)
    return grid


@pytest.fixture()
def full_size_uniform_prediction() -> NDArray[np.float64]:
    """40x40x6 uniform prediction."""
    return np.full((40, 40, K), 1.0 / K, dtype=np.float64)


@pytest.fixture()
def mock_round_list() -> list[dict[str, object]]:
    """Sample /rounds API response."""
    return [
        {
            "id": "aaaa-bbbb-cccc-0001",
            "round_number": 18,
            "status": "completed",
            "map_width": 40,
            "map_height": 40,
        },
        {
            "id": "aaaa-bbbb-cccc-0002",
            "round_number": 19,
            "status": "active",
            "map_width": 40,
            "map_height": 40,
        },
    ]


@pytest.fixture()
def mock_round_detail() -> dict[str, object]:
    """Sample /rounds/{id} API response with 2 seeds."""
    # Minimal 3x3 grid for testing (server uses 40x40 but this suffices for unit tests)
    small_grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    return {
        "id": "aaaa-bbbb-cccc-0002",
        "round_number": 19,
        "map_width": 40,
        "map_height": 40,
        "seeds_count": 2,
        "initial_states": [
            {"grid": small_grid, "settlements": []},
            {"grid": small_grid, "settlements": []},
        ],
    }


@pytest.fixture()
def mock_simulate_response() -> dict[str, object]:
    """Sample /simulate API response for a 3x3 viewport."""
    return {
        "grid": [[0, 1, 2], [3, 4, 5], [10, 11, 0]],
        "viewport": {"x": 0, "y": 0, "w": 3, "h": 3},
        "settlements": [],
        "queries_used": 1,
        "queries_max": 50,
    }


@pytest.fixture()
def mock_submit_response() -> dict[str, object]:
    """Sample /submit API response."""
    return {
        "status": "accepted",
        "round_id": "aaaa-bbbb-cccc-0002",
        "seed_index": 0,
    }
