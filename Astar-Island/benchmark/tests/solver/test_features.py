from __future__ import annotations

from astar_twin.contracts.api_models import SimSettlement, SimulateResponse, ViewportBounds
from astar_twin.solver.observe.features import ObservationFeatures, extract_features


def _make_response(settlements: list[SimSettlement]) -> SimulateResponse:
    return SimulateResponse(
        grid=[[0] * 5 for _ in range(5)],
        settlements=settlements,
        viewport=ViewportBounds(x=0, y=0, w=5, h=5),
        width=10,
        height=10,
        queries_used=1,
        queries_max=50,
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


def test_prosperity_proxy_zero_when_no_settlements() -> None:
    resp = _make_response([])
    features = extract_features(resp)
    assert features.prosperity_proxy_mean == 0.0
    assert features.prosperity_proxy_var == 0.0


def test_prosperity_proxy_computed_correctly() -> None:
    s1 = _make_settlement(population=2.0, wealth=4.0)
    s2 = _make_settlement(x=1, population=1.0, wealth=1.0)
    resp = _make_response([s1, s2])
    features = extract_features(resp)
    expected_mean = (4.0 / 2.0 + 1.0 / 1.0) / 2.0
    assert abs(features.prosperity_proxy_mean - expected_mean) < 1e-9


def test_prosperity_proxy_single_settlement() -> None:
    s = _make_settlement(population=3.0, wealth=6.0)
    resp = _make_response([s])
    features = extract_features(resp)
    assert abs(features.prosperity_proxy_mean - 2.0) < 1e-9
    assert features.prosperity_proxy_var == 0.0


def test_prosperity_proxy_handles_near_zero_population() -> None:
    s = _make_settlement(population=0.0, wealth=1.0)
    resp = _make_response([s])
    features = extract_features(resp)
    assert features.prosperity_proxy_mean == 1.0 / 1e-3


def test_dead_settlements_excluded_from_proxy() -> None:
    alive = _make_settlement(population=2.0, wealth=4.0, alive=True)
    dead = _make_settlement(x=1, population=1.0, wealth=10.0, alive=False)
    resp = _make_response([alive, dead])
    features = extract_features(resp)
    assert abs(features.prosperity_proxy_mean - 2.0) < 1e-9


def test_observation_features_has_proxy_fields() -> None:
    f = ObservationFeatures()
    assert hasattr(f, "prosperity_proxy_mean")
    assert hasattr(f, "prosperity_proxy_var")
