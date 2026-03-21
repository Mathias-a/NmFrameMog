from __future__ import annotations

import numpy as np

from astar_twin.contracts.types import (
    DEFAULT_MAP_HEIGHT,
    DEFAULT_MAP_WIDTH,
    MAX_QUERIES,
    MAX_VIEWPORT,
    MIN_VIEWPORT,
    NUM_CLASSES,
    TERRAIN_TO_CLASS,
    TerrainCode,
)
from astar_twin.data.models import RoundFixture
from astar_twin.scoring import safe_prediction
from astar_twin.solver.baselines import fixed_coverage_baseline, uniform_baseline


def test_benchmark_contract_constants() -> None:
    assert NUM_CLASSES == 6
    assert MAX_QUERIES == 50
    assert MIN_VIEWPORT == 5
    assert MAX_VIEWPORT == 15
    assert DEFAULT_MAP_HEIGHT == 40
    assert DEFAULT_MAP_WIDTH == 40


def test_terrain_to_class_mapping_complete() -> None:
    for code in TerrainCode:
        assert int(code) in TERRAIN_TO_CLASS, f"Missing TERRAIN_TO_CLASS entry for {code}"
        cls_idx = TERRAIN_TO_CLASS[int(code)]
        assert 0 <= cls_idx < NUM_CLASSES, f"Class index {cls_idx} out of range for {code}"


def test_uniform_baseline_shape_and_normalization(fixture: RoundFixture) -> None:
    tensor = uniform_baseline(fixture.map_height, fixture.map_width)
    assert tensor.shape == (fixture.map_height, fixture.map_width, NUM_CLASSES)
    sums = np.sum(tensor, axis=2)
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert np.all(tensor > 0)


def test_fixed_coverage_baseline_valid_tensors(fixture: RoundFixture) -> None:
    tensors = fixed_coverage_baseline(
        fixture.initial_states, fixture.map_height, fixture.map_width, n_mc_runs=5, base_seed=42
    )
    assert len(tensors) == fixture.seeds_count
    for tensor in tensors:
        assert tensor.shape == (fixture.map_height, fixture.map_width, NUM_CLASSES)
        sums = np.sum(tensor, axis=2)
        assert np.allclose(sums, 1.0, atol=1e-6)
        assert np.all(tensor > 0)


def test_uniform_baseline_all_seeds(fixture: RoundFixture) -> None:
    t1 = uniform_baseline(fixture.map_height, fixture.map_width)
    t2 = uniform_baseline(fixture.map_height, fixture.map_width)
    assert np.array_equal(t1, t2), "Uniform baseline should be deterministic"


def test_unsafe_tensor_before_floor_has_zeros() -> None:
    raw = np.zeros((2, 2, 6), dtype=np.float64)
    assert np.any(raw == 0.0), "Raw zero tensor should contain zeros"
    safe = safe_prediction(raw)
    assert np.all(safe > 0.0), "safe_prediction must remove all zeros"
    sums = np.sum(safe, axis=2)
    assert np.allclose(sums, 1.0, atol=1e-6)
