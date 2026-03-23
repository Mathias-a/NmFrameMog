from __future__ import annotations

from pathlib import Path

import numpy as np

from astar_twin.data.loaders import load_fixture
from astar_twin.harness.budget import Budget
from astar_twin.scoring import safe_prediction
from astar_twin.strategies import REGISTRY
from astar_twin.strategies.learned_calibrator.model import (
    DEFAULT_ZONE_WEIGHTS,
    blend_predictions,
)
from astar_twin.strategies.learned_calibrator.strategy import LearnedCalibratorStrategy
from astar_twin.strategies.learned_calibrator.training import (
    HistoricalSeedExample,
    fit_zone_weights,
)

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


def test_registry_contains_learned_calibrator() -> None:
    assert REGISTRY["learned_calibrator"] is LearnedCalibratorStrategy


def test_learned_calibrator_output_properties() -> None:
    fixture = load_fixture(FIXTURE_PATH)
    state = fixture.initial_states[0]
    strategy = LearnedCalibratorStrategy()
    result = strategy.predict(state, budget=Budget(total=50), base_seed=7)

    assert result.shape == (fixture.map_height, fixture.map_width, 6)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_learned_calibrator_is_deterministic() -> None:
    fixture = load_fixture(FIXTURE_PATH)
    state = fixture.initial_states[0]
    strategy = LearnedCalibratorStrategy()

    r1 = strategy.predict(state, budget=Budget(total=50), base_seed=11)
    r2 = strategy.predict(state, budget=Budget(total=50), base_seed=11)
    np.testing.assert_array_equal(r1, r2)


def test_blend_predictions_respects_weights() -> None:
    base = np.zeros((1, 1, 6), dtype=np.float64)
    base[0, 0, 0] = 1.0
    fallback = np.zeros((1, 1, 6), dtype=np.float64)
    fallback[0, 0, 1] = 1.0
    zone_map = np.array([[0]], dtype=np.int8)
    is_static = np.array([[False]])

    blended = blend_predictions(
        base,
        fallback,
        zone_map,
        is_static,
        {**DEFAULT_ZONE_WEIGHTS, "core": 1.0},
    )
    np.testing.assert_allclose(blended[0, 0], fallback[0, 0])


def test_fit_zone_weights_prefers_fallback_when_ground_truth_matches_it() -> None:
    base = np.zeros((1, 1, 6), dtype=np.float64)
    base[0, 0, 0] = 1.0
    fallback = np.zeros((1, 1, 6), dtype=np.float64)
    fallback[0, 0, 1] = 1.0
    ground_truth = safe_prediction(fallback)

    example = HistoricalSeedExample(
        round_id="round-a",
        round_number=1,
        round_weight=1.0,
        seed_index=0,
        base_prediction=base,
        fallback_prediction=fallback,
        ground_truth=ground_truth,
        zone_map=np.array([[0]], dtype=np.int8),
        is_static=np.array([[False]]),
    )

    weights = fit_zone_weights([example], weight_grid=[0.0, 0.5, 1.0], n_passes=2)
    assert weights["core"] == 1.0
