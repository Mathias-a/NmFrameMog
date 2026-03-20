from __future__ import annotations

from random import Random

from round_8_implementation.solver.proxy_simulator import (
    InitialSettlement,
    build_ground_truth_tensor,
    run_proxy_simulation,
)


def test_proxy_simulation_preserves_static_terrain() -> None:
    initial_grid = [
        [10, 10, 10, 10, 10],
        [10, 11, 11, 11, 10],
        [10, 11, 1, 11, 10],
        [10, 11, 11, 5, 10],
        [10, 10, 10, 10, 10],
    ]
    settlements = (InitialSettlement(x=2, y=2, has_port=False, alive=True, owner_id=1),)

    result = run_proxy_simulation(initial_grid, settlements, rng=Random(7), years=5)

    for y, row in enumerate(initial_grid):
        for x, terrain_code in enumerate(row):
            if terrain_code in {5, 10}:
                assert result.grid[y][x] == terrain_code


def test_proxy_ports_remain_coastal() -> None:
    initial_grid = [
        [10, 10, 10, 10, 10, 10],
        [10, 11, 11, 11, 11, 10],
        [10, 11, 2, 11, 11, 10],
        [10, 11, 11, 11, 11, 10],
        [10, 10, 10, 10, 10, 10],
    ]
    settlements = (InitialSettlement(x=2, y=2, has_port=True, alive=True, owner_id=1),)

    result = run_proxy_simulation(initial_grid, settlements, rng=Random(11), years=12)

    for settlement in result.settlements:
        if not settlement.has_port:
            continue
        neighbors = []
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                if delta_x == 0 and delta_y == 0:
                    continue
                next_x = settlement.x + delta_x
                next_y = settlement.y + delta_y
                if 0 <= next_x < len(result.grid[0]) and 0 <= next_y < len(result.grid):
                    neighbors.append(result.grid[next_y][next_x])
        assert 10 in neighbors


def test_ground_truth_tensor_is_normalized() -> None:
    initial_grid = [
        [10, 10, 10, 10, 10],
        [10, 11, 11, 11, 10],
        [10, 11, 1, 11, 10],
        [10, 11, 4, 11, 10],
        [10, 10, 10, 10, 10],
    ]
    settlements = (InitialSettlement(x=2, y=2, has_port=False, alive=True, owner_id=1),)

    tensor = build_ground_truth_tensor(
        initial_grid,
        settlements,
        rollout_count=16,
        base_seed=31,
        years=10,
    )

    assert len(tensor) == 5
    assert len(tensor[0]) == 5
    for row in tensor:
        for cell in row:
            assert len(cell) == 6
            assert abs(sum(cell) - 1.0) < 1e-9
            assert all(probability >= 0.0 for probability in cell)
