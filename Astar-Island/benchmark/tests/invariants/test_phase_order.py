from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

from astar_twin.data.models import RoundFixture
from astar_twin.engine.simulator import Simulator


def test_phase_order_and_year_count(monkeypatch: MonkeyPatch, fixture: RoundFixture) -> None:
    calls: list[str] = []

    def growth(state: object, params: object, rng: object) -> object:
        calls.append("growth")
        return state

    def conflict(
        state: object,
        params: object,
        rng: object,
        war_registry: object,
        current_year: int,
    ) -> object:
        calls.append("conflict")
        return state

    def trade(
        state: object,
        params: object,
        rng: object,
        war_registry: object,
        current_year: int,
    ) -> object:
        calls.append("trade")
        return state

    def winter(
        state: object, params: object, rng: object, prev_severity: float
    ) -> tuple[object, float]:
        calls.append("winter")
        return state, prev_severity

    def environment(state: object, params: object, rng: object) -> object:
        calls.append("environment")
        return state

    monkeypatch.setattr("astar_twin.engine.simulator.apply_growth", growth)
    monkeypatch.setattr("astar_twin.engine.simulator.apply_conflict", conflict)
    monkeypatch.setattr("astar_twin.engine.simulator.apply_trade", trade)
    monkeypatch.setattr("astar_twin.engine.simulator.apply_winter", winter)
    monkeypatch.setattr("astar_twin.engine.simulator.apply_environment", environment)

    Simulator().run(fixture.initial_states[0], sim_seed=0)

    assert len(calls) == 50 * 5
    for year in range(50):
        start = year * 5
        assert calls[start : start + 5] == ["growth", "conflict", "trade", "winter", "environment"]
