"""Tests for the benchmark adapter — ensures solver never sees hidden params."""

from __future__ import annotations

import numpy as np

from astar_twin.contracts.types import MAX_QUERIES, NUM_CLASSES
from astar_twin.data.models import RoundFixture
from astar_twin.solver.adapters.benchmark import BenchmarkAdapter


def test_adapter_returns_round_detail(fixture: RoundFixture) -> None:
    adapter = BenchmarkAdapter(fixture)
    detail = adapter.get_round_detail(fixture.id)
    assert detail.id == fixture.id
    assert detail.seeds_count == fixture.seeds_count
    assert detail.map_width == fixture.map_width
    assert detail.map_height == fixture.map_height
    assert len(detail.initial_states) == fixture.seeds_count


def test_adapter_simulate_returns_valid_response(fixture: RoundFixture) -> None:
    adapter = BenchmarkAdapter(fixture)
    resp = adapter.simulate(fixture.id, 0, 0, 0, 10, 10)
    assert len(resp.grid) == 10
    assert len(resp.grid[0]) == 10
    assert resp.queries_used == 1
    assert resp.queries_max == MAX_QUERIES


def test_adapter_budget_tracking(fixture: RoundFixture) -> None:
    adapter = BenchmarkAdapter(fixture)
    assert adapter.get_budget(fixture.id) == (0, MAX_QUERIES)
    adapter.simulate(fixture.id, 0, 0, 0, 5, 5)
    assert adapter.get_budget(fixture.id) == (1, MAX_QUERIES)
    adapter.simulate(fixture.id, 1, 0, 0, 5, 5)
    assert adapter.get_budget(fixture.id) == (2, MAX_QUERIES)


def test_simulation_params_not_accessible_from_adapter_outputs(fixture: RoundFixture) -> None:
    """Solver objects returned by adapter must not expose simulation_params."""
    adapter = BenchmarkAdapter(fixture)
    detail = adapter.get_round_detail(fixture.id)
    resp = adapter.simulate(fixture.id, 0, 0, 0, 5, 5)

    # RoundDetail should not have simulation_params
    assert not hasattr(detail, "simulation_params")
    # SimulateResponse should not have simulation_params
    assert not hasattr(resp, "simulation_params")


def test_deterministic_replay_mode(fixture: RoundFixture) -> None:
    """Same seed offset should produce identical observation transcripts."""
    adapter1 = BenchmarkAdapter(fixture, sim_seed_offset=42)
    adapter2 = BenchmarkAdapter(fixture, sim_seed_offset=42)
    resp1 = adapter1.simulate(fixture.id, 0, 0, 0, 10, 10)
    resp2 = adapter2.simulate(fixture.id, 0, 0, 0, 10, 10)
    assert resp1.grid == resp2.grid
    assert len(resp1.settlements) == len(resp2.settlements)


def test_submit_and_analysis(fixture: RoundFixture) -> None:
    adapter = BenchmarkAdapter(fixture)
    prediction = np.full(
        (fixture.map_height, fixture.map_width, NUM_CLASSES),
        1.0 / NUM_CLASSES,
        dtype=np.float64,
    )
    submit_resp = adapter.submit(fixture.id, 0, prediction)
    assert submit_resp.status == "accepted"

    analysis = adapter.get_analysis(fixture.id, 0)
    assert analysis.width == fixture.map_width
    assert analysis.height == fixture.map_height
    assert analysis.ground_truth is not None
    # Score should exist since we submitted
    assert analysis.score is not None


def test_reset_budget(fixture: RoundFixture) -> None:
    adapter = BenchmarkAdapter(fixture)
    adapter.simulate(fixture.id, 0, 0, 0, 5, 5)
    assert adapter.get_budget(fixture.id)[0] == 1
    adapter.reset_budget()
    assert adapter.get_budget(fixture.id)[0] == 0


def test_budget_exhaustion_raises(fixture: RoundFixture) -> None:
    adapter = BenchmarkAdapter(fixture)
    adapter._queries_used = MAX_QUERIES
    try:
        adapter.simulate(fixture.id, 0, 0, 0, 5, 5)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "budget exhausted" in str(e).lower()
