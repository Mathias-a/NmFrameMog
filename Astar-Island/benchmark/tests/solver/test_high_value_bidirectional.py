from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astar_twin.contracts.types import MAX_QUERIES, NUM_CLASSES
from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import RoundFixture
from astar_twin.solver.adapters.benchmark import BenchmarkAdapter
from astar_twin.solver.high_value_bidirectional import solve_high_value_bidirectional

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def fixture() -> RoundFixture:
    return load_fixture(FIXTURE_PATH)


@pytest.fixture
def adapter(fixture: RoundFixture) -> BenchmarkAdapter:
    return BenchmarkAdapter(fixture, n_mc_runs=5, sim_seed_offset=0)


def test_variant_returns_valid_tensors(adapter: BenchmarkAdapter, fixture: RoundFixture) -> None:
    result = solve_high_value_bidirectional(
        adapter,
        fixture.id,
        n_particles=8,
        n_inner_runs=2,
        sims_per_seed=4,
        base_seed=42,
    )
    assert len(result.tensors) == fixture.seeds_count
    for tensor in result.tensors:
        assert tensor.shape == (fixture.map_height, fixture.map_width, NUM_CLASSES)
        assert np.all(tensor > 0)
        np.testing.assert_allclose(np.sum(tensor, axis=2), 1.0, atol=0.02)


def test_variant_budget_and_transcript_consistent(
    adapter: BenchmarkAdapter,
    fixture: RoundFixture,
) -> None:
    result = solve_high_value_bidirectional(
        adapter,
        fixture.id,
        n_particles=8,
        n_inner_runs=2,
        sims_per_seed=4,
        base_seed=42,
    )
    assert result.total_queries_used <= MAX_QUERIES
    assert len(result.transcript) == result.total_queries_used


def test_variant_is_deterministic(fixture: RoundFixture) -> None:
    adapter_a = BenchmarkAdapter(fixture, n_mc_runs=5, sim_seed_offset=0)
    adapter_b = BenchmarkAdapter(fixture, n_mc_runs=5, sim_seed_offset=0)

    result_a = solve_high_value_bidirectional(
        adapter_a,
        fixture.id,
        n_particles=8,
        n_inner_runs=2,
        sims_per_seed=4,
        base_seed=42,
    )
    result_b = solve_high_value_bidirectional(
        adapter_b,
        fixture.id,
        n_particles=8,
        n_inner_runs=2,
        sims_per_seed=4,
        base_seed=42,
    )

    assert result_a.total_queries_used == result_b.total_queries_used
    assert len(result_a.transcript) == len(result_b.transcript)
    for tensor_a, tensor_b in zip(result_a.tensors, result_b.tensors, strict=True):
        np.testing.assert_array_equal(tensor_a, tensor_b)
