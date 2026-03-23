from __future__ import annotations

import pytest

from astar_twin.contracts.api_models import SimSettlement, SimulateResponse, ViewportBounds
from astar_twin.contracts.types import ClassIndex, TerrainCode
from astar_twin.solver.observe.features import extract_features
from astar_twin.solver.observe.ledger import (
    ViewportObservation,
    create_ledger,
    record_observation,
)


def _make_settlement(
    x: int = 0,
    y: int = 0,
    population: float = 2.0,
    food: float = 1.0,
    wealth: float = 1.0,
    defense: float = 0.5,
    has_port: bool = False,
    alive: bool = True,
    owner_id: int = 0,
) -> SimSettlement:
    return SimSettlement(
        x=x,
        y=y,
        population=population,
        food=food,
        wealth=wealth,
        defense=defense,
        has_port=has_port,
        alive=alive,
        owner_id=owner_id,
    )


def _make_observation(
    seed_index: int = 0,
    viewport_x: int = 0,
    viewport_y: int = 0,
    viewport_w: int = 5,
    viewport_h: int = 5,
    grid: list[list[int]] | None = None,
    settlements: list[SimSettlement] | None = None,
) -> ViewportObservation:
    if grid is None:
        grid = [[0] * viewport_w for _ in range(viewport_h)]
    if settlements is None:
        settlements = []
    response = SimulateResponse(
        grid=grid,
        settlements=settlements,
        viewport=ViewportBounds(x=viewport_x, y=viewport_y, w=viewport_w, h=viewport_h),
        width=40,
        height=40,
        queries_used=1,
        queries_max=50,
    )
    features = extract_features(response)
    return ViewportObservation(
        seed_index=seed_index,
        phase="bootstrap",
        viewport_x=viewport_x,
        viewport_y=viewport_y,
        viewport_w=viewport_w,
        viewport_h=viewport_h,
        grid=grid,
        settlements=settlements,
        features=features,
    )


def test_create_ledger_shapes() -> None:
    ledger = create_ledger(n_seeds=5, height=40, width=40)

    assert len(ledger.per_seed_class_counts) == 5
    assert len(ledger.per_seed_visit_counts) == 5
    for seed_idx in range(5):
        assert ledger.per_seed_class_counts[seed_idx].shape == (40, 40, 6)
        assert ledger.per_seed_visit_counts[seed_idx].shape == (40, 40)


def test_record_observation_updates_counts_and_visits() -> None:
    ledger = create_ledger(n_seeds=5, height=40, width=40)
    grid = [
        [TerrainCode.PLAINS, TerrainCode.FOREST],
        [TerrainCode.PORT, TerrainCode.MOUNTAIN],
    ]
    settlements = [_make_settlement(has_port=True)]

    record_observation(
        ledger,
        _make_observation(
            seed_index=2,
            viewport_x=3,
            viewport_y=4,
            viewport_w=2,
            viewport_h=2,
            grid=grid,
            settlements=settlements,
        ),
    )

    seed_counts = ledger.per_seed_class_counts[2]
    seed_visits = ledger.per_seed_visit_counts[2]
    assert seed_visits[4, 3] == 1.0
    assert seed_visits[4, 4] == 1.0
    assert seed_visits[5, 3] == 1.0
    assert seed_visits[5, 4] == 1.0
    assert seed_counts[4, 3, ClassIndex.EMPTY] == 1.0
    assert seed_counts[4, 4, ClassIndex.FOREST] == 1.0
    assert seed_counts[5, 3, ClassIndex.PORT] == 1.0
    assert seed_counts[5, 4, ClassIndex.MOUNTAIN] == 1.0


def test_multiple_observations_accumulate_overlap() -> None:
    ledger = create_ledger(n_seeds=5, height=40, width=40)
    first = _make_observation(
        seed_index=0,
        viewport_x=0,
        viewport_y=0,
        viewport_w=2,
        viewport_h=2,
        grid=[[TerrainCode.PLAINS, TerrainCode.PLAINS], [TerrainCode.PLAINS, TerrainCode.PLAINS]],
    )
    second = _make_observation(
        seed_index=0,
        viewport_x=1,
        viewport_y=1,
        viewport_w=2,
        viewport_h=2,
        grid=[[TerrainCode.FOREST, TerrainCode.FOREST], [TerrainCode.FOREST, TerrainCode.FOREST]],
    )

    record_observation(ledger, first)
    record_observation(ledger, second)

    assert ledger.per_seed_visit_counts[0][1, 1] == 2.0
    assert ledger.per_seed_class_counts[0][1, 1, ClassIndex.EMPTY] == 1.0
    assert ledger.per_seed_class_counts[0][1, 1, ClassIndex.FOREST] == 1.0
    assert ledger.n_observations == 2


def test_pooled_stats_accumulate_total_cells_across_seeds() -> None:
    ledger = create_ledger(n_seeds=5, height=40, width=40)
    obs_a = _make_observation(
        seed_index=0,
        viewport_w=2,
        viewport_h=2,
        grid=[[TerrainCode.PLAINS, TerrainCode.FOREST], [TerrainCode.PORT, TerrainCode.MOUNTAIN]],
        settlements=[_make_settlement(population=4.0, wealth=8.0, has_port=True)],
    )
    obs_b = _make_observation(
        seed_index=1,
        viewport_x=10,
        viewport_y=10,
        viewport_w=3,
        viewport_h=1,
        grid=[[TerrainCode.PLAINS, TerrainCode.PLAINS, TerrainCode.RUIN]],
        settlements=[_make_settlement(x=10, y=10, alive=False)],
    )

    record_observation(ledger, obs_a)
    record_observation(ledger, obs_b)

    pooled = ledger.pooled_stats
    assert pooled.total_cells == 7.0
    assert pooled.class_counts[ClassIndex.EMPTY] == 3.0
    assert pooled.class_counts[ClassIndex.FOREST] == 1.0
    assert pooled.class_counts[ClassIndex.PORT] == 1.0
    assert pooled.class_counts[ClassIndex.MOUNTAIN] == 1.0
    assert pooled.class_counts[ClassIndex.RUIN] == 1.0
    assert pooled.alive_count_sum == 1.0
    assert pooled.dead_count_sum == 1.0
    assert pooled.port_count_sum == 1.0


@pytest.mark.parametrize(
    ("seed_index", "x", "y", "w", "h", "expected"),
    [
        (0, 0, 0, 2, 2, 1.0),
        (0, 1, 1, 2, 2, 0.25),
        (0, 20, 20, 5, 5, 0.0),
    ],
)
def test_mean_visit_count_in_window(
    seed_index: int,
    x: int,
    y: int,
    w: int,
    h: int,
    expected: float,
) -> None:
    ledger = create_ledger(n_seeds=5, height=40, width=40)
    record_observation(
        ledger,
        _make_observation(
            seed_index=0,
            viewport_x=0,
            viewport_y=0,
            viewport_w=2,
            viewport_h=2,
        ),
    )

    assert ledger.mean_visit_count_in_window(seed_index, x, y, w, h) == pytest.approx(expected)


def test_record_observation_handles_edge_viewport_bounds() -> None:
    ledger = create_ledger(n_seeds=5, height=40, width=40)
    grid = [[TerrainCode.PLAINS] * 5 for _ in range(5)]

    record_observation(
        ledger,
        _make_observation(
            seed_index=4,
            viewport_x=38,
            viewport_y=38,
            viewport_w=5,
            viewport_h=5,
            grid=grid,
        ),
    )

    assert ledger.per_seed_visit_counts[4][38, 38] == 1.0
    assert ledger.per_seed_visit_counts[4][39, 39] == 1.0
    assert ledger.per_seed_visit_counts[4].sum() == 4.0
