from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from astar_twin.api.app import create_app
from astar_twin.api.store import BudgetStore, RoundStore, SubmissionStore
from astar_twin.contracts.types import TerrainCode
from astar_twin.data.models import RoundFixture
from astar_twin.params import SimulationParams


def test_simulate_success_returns_correct_viewport_shape(client: TestClient) -> None:
    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "viewport_x": 1,
            "viewport_y": 2,
            "viewport_w": 5,
            "viewport_h": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["grid"]) == 5
    assert all(len(row) == 5 for row in payload["grid"])


def test_simulate_settlements_stay_within_viewport_bounds(client: TestClient) -> None:
    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 10,
            "viewport_h": 10,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    viewport = payload["viewport"]
    for settlement in payload["settlements"]:
        assert viewport["x"] <= settlement["x"] < viewport["x"] + viewport["w"]
        assert viewport["y"] <= settlement["y"] < viewport["y"] + viewport["h"]


def test_simulate_clamps_viewport_to_map_bounds(client: TestClient) -> None:
    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "viewport_x": 9,
            "viewport_y": 9,
            "viewport_w": 15,
            "viewport_h": 15,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["viewport"] == {"x": 9, "y": 9, "w": 1, "h": 1}
    assert len(payload["grid"]) == 1
    assert len(payload["grid"][0]) == 1


def test_simulate_404_for_missing_round(client: TestClient) -> None:
    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": "missing-round",
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 5,
            "viewport_h": 5,
        },
    )
    assert response.status_code == 404


def test_simulate_400_for_non_active_round(completed_client: TestClient) -> None:
    response = completed_client.post(
        "/astar-island/simulate",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 5,
            "viewport_h": 5,
        },
    )
    assert response.status_code == 400


def test_simulate_429_when_budget_exhausted(active_fixture: RoundFixture) -> None:
    round_store = RoundStore()
    round_store.add(active_fixture)
    budget_store = BudgetStore()
    for _ in range(50):
        budget_store.increment(active_fixture.id)
    app = create_app(
        round_store=round_store,
        submission_store=SubmissionStore(),
        budget_store=budget_store,
        n_mc_runs=5,
    )
    client = TestClient(app)

    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": "test-round-001",
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 5,
            "viewport_h": 5,
        },
    )
    assert response.status_code == 429


def test_simulate_uses_round_specific_simulation_params(
    active_fixture: RoundFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    captured_means: list[float] = []

    class _RecordingSimulator:
        def __init__(self, params: SimulationParams | None = None) -> None:
            captured_means.append((params or SimulationParams()).init_population_mean)

        def run(self, initial_state: object, sim_seed: int) -> SimpleNamespace:
            del initial_state, sim_seed
            return SimpleNamespace(
                grid=_StubGrid(cells=[[int(TerrainCode.PLAINS)] * 10 for _ in range(10)]),
                settlements=[],
            )

    custom_fixture = active_fixture.model_copy(
        update={
            "id": "custom-round",
            "simulation_params": SimulationParams(init_population_mean=3.0),
        }
    )
    round_store = RoundStore()
    round_store.add(active_fixture)
    round_store.add(custom_fixture)
    monkeypatch.setattr("astar_twin.api.routes.simulate.Simulator", _RecordingSimulator)
    client = TestClient(
        create_app(
            round_store=round_store,
            submission_store=SubmissionStore(),
            budget_store=BudgetStore(),
            n_mc_runs=5,
        )
    )

    default_response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": active_fixture.id,
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 10,
            "viewport_h": 10,
        },
    )
    custom_response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": custom_fixture.id,
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 10,
            "viewport_h": 10,
        },
    )

    assert default_response.status_code == 200
    assert custom_response.status_code == 200
    assert captured_means == [
        active_fixture.simulation_params.init_population_mean,
        custom_fixture.simulation_params.init_population_mean,
    ]


@dataclass
class _StubGrid:
    cells: list[list[int]]

    def viewport(self, vx: int, vy: int, vw: int, vh: int) -> _StubGrid:
        x0 = max(0, vx)
        y0 = max(0, vy)
        x1 = min(len(self.cells[0]), vx + vw)
        y1 = min(len(self.cells), vy + vh)
        return _StubGrid(cells=[row[x0:x1] for row in self.cells[y0:y1]])

    def get(self, y: int, x: int) -> int:
        return self.cells[y][x]

    def to_list(self) -> list[list[int]]:
        return [row[:] for row in self.cells]


@dataclass
class _StubSettlement:
    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    has_port: bool
    alive: bool
    owner_id: int


@dataclass
class _StubRoundState:
    grid: _StubGrid
    settlements: list[_StubSettlement]


class _StubSimulator:
    def __init__(self, params: SimulationParams | None = None) -> None:
        self.params = params or SimulationParams()

    def run(self, initial_state: object, sim_seed: int) -> _StubRoundState:
        del initial_state, sim_seed
        return _StubRoundState(
            grid=_StubGrid(
                cells=[
                    [int(TerrainCode.PLAINS), int(TerrainCode.PLAINS)],
                    [int(TerrainCode.RUIN), int(TerrainCode.PLAINS)],
                ]
            ),
            settlements=[
                _StubSettlement(1, 0, 1.0, 1.0, 1.0, 1.0, False, False, 1),
                _StubSettlement(0, 1, 1.0, 1.0, 1.0, 1.0, False, False, 2),
            ],
        )


def test_simulate_excludes_dead_settlements_after_ruin_decay(
    active_fixture: RoundFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    round_store = RoundStore()
    round_store.add(active_fixture.model_copy(update={"map_width": 2, "map_height": 2}))
    monkeypatch.setattr("astar_twin.api.routes.simulate.Simulator", _StubSimulator)
    client = TestClient(
        create_app(
            round_store=round_store,
            submission_store=SubmissionStore(),
            budget_store=BudgetStore(),
            n_mc_runs=5,
        )
    )

    response = client.post(
        "/astar-island/simulate",
        json={
            "round_id": active_fixture.id,
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 5,
            "viewport_h": 5,
        },
    )

    assert response.status_code == 200
    settlements = response.json()["settlements"]
    assert settlements == [
        {
            "x": 0,
            "y": 1,
            "population": 1.0,
            "food": 1.0,
            "wealth": 1.0,
            "defense": 1.0,
            "has_port": False,
            "alive": False,
            "owner_id": 2,
        }
    ]
