"""Tests for hotspot generation and observation features."""
from __future__ import annotations

import numpy as np

from astar_twin.contracts.api_models import (
    InitialSettlement,
    InitialState,
    SimSettlement,
    SimulateResponse,
    ViewportBounds,
)
from astar_twin.contracts.types import (
    MAX_VIEWPORT,
    MIN_VIEWPORT,
    NUM_CLASSES,
    TerrainCode,
)
from astar_twin.data.models import RoundFixture
from astar_twin.solver.observe.features import ObservationFeatures, extract_features
from astar_twin.solver.policy.hotspots import ViewportCandidate, generate_hotspots


def test_hotspots_from_fixture_seed(fixture: RoundFixture) -> None:
    """Every fixture seed should yield at least two bootstrap candidates."""
    for initial_state in fixture.initial_states:
        candidates = generate_hotspots(
            initial_state, fixture.map_height, fixture.map_width
        )
        assert len(candidates) >= 2, "Each seed must produce at least 2 candidates"


def test_all_candidates_respect_viewport_bounds(fixture: RoundFixture) -> None:
    """All generated candidates must have valid viewport dimensions."""
    for initial_state in fixture.initial_states:
        candidates = generate_hotspots(
            initial_state, fixture.map_height, fixture.map_width
        )
        for c in candidates:
            assert MIN_VIEWPORT <= c.w <= MAX_VIEWPORT, f"Width {c.w} out of bounds"
            assert MIN_VIEWPORT <= c.h <= MAX_VIEWPORT, f"Height {c.h} out of bounds"
            assert c.x >= 0
            assert c.y >= 0
            assert c.x + c.w <= fixture.map_width, f"x={c.x} + w={c.w} > {fixture.map_width}"
            assert c.y + c.h <= fixture.map_height, f"y={c.y} + h={c.h} > {fixture.map_height}"


def test_hotspot_categories_present(fixture: RoundFixture) -> None:
    """Expected hotspot categories should appear when map features exist."""
    initial_state = fixture.initial_states[0]
    candidates = generate_hotspots(
        initial_state, fixture.map_height, fixture.map_width
    )
    categories = {c.category for c in candidates}
    # At minimum, some categories should be present
    assert len(categories) >= 1


def test_degenerate_empty_seed_gets_fallback() -> None:
    """A completely empty/ocean seed should still get valid fallback candidates."""
    # All ocean grid — no settlements, no forest, no ruins
    grid = [[TerrainCode.OCEAN] * 20 for _ in range(20)]
    initial_state = InitialState(grid=grid, settlements=[])
    candidates = generate_hotspots(initial_state, 20, 20)
    assert len(candidates) >= 2, "Fallback candidates should be generated"
    for c in candidates:
        assert c.category == "fallback"
        assert MIN_VIEWPORT <= c.w <= MAX_VIEWPORT
        assert MIN_VIEWPORT <= c.h <= MAX_VIEWPORT


def test_overlap_fraction() -> None:
    c1 = ViewportCandidate(x=0, y=0, w=10, h=10, category="test")
    c2 = ViewportCandidate(x=5, y=5, w=10, h=10, category="test")
    overlap = c1.overlap_fraction(c2)
    # Overlap area = 5*5 = 25, c1 area = 100
    assert abs(overlap - 0.25) < 1e-6

    c3 = ViewportCandidate(x=0, y=0, w=10, h=10, category="test")
    assert abs(c1.overlap_fraction(c3) - 1.0) < 1e-6  # identical


def test_extract_features_with_settlements() -> None:
    """Feature extraction should capture settlement stats."""
    response = SimulateResponse(
        grid=[[0, 0], [0, 0]],
        settlements=[
            SimSettlement(
                x=0, y=0, population=2.0, food=1.5, wealth=0.5,
                defense=0.3, has_port=True, alive=True, owner_id=0,
            ),
            SimSettlement(
                x=1, y=1, population=3.0, food=2.0, wealth=0.8,
                defense=0.6, has_port=False, alive=True, owner_id=1,
            ),
        ],
        viewport=ViewportBounds(x=0, y=0, w=2, h=2),
        width=10,
        height=10,
        queries_used=1,
        queries_max=50,
    )
    features = extract_features(response)
    assert features.alive_count == 2
    assert features.dead_count == 0
    assert features.port_count == 1
    assert features.total_cells == 4
    assert features.population_mean > 0
    assert np.isfinite(features.population_var)


def test_extract_features_empty_viewport() -> None:
    """Feature extraction should handle empty viewports gracefully."""
    response = SimulateResponse(
        grid=[[TerrainCode.OCEAN, TerrainCode.OCEAN]],
        settlements=[],
        viewport=ViewportBounds(x=0, y=0, w=2, h=1),
        width=10,
        height=10,
        queries_used=1,
        queries_max=50,
    )
    features = extract_features(response)
    assert features.alive_count == 0
    assert features.dead_count == 0
    assert features.port_count == 0
    assert features.population_mean == 0.0


def test_extract_features_class_counts() -> None:
    """Class counts should correctly map terrain codes."""
    response = SimulateResponse(
        grid=[
            [TerrainCode.FOREST, TerrainCode.MOUNTAIN],
            [TerrainCode.OCEAN, TerrainCode.SETTLEMENT],
        ],
        settlements=[],
        viewport=ViewportBounds(x=0, y=0, w=2, h=2),
        width=10,
        height=10,
        queries_used=1,
        queries_max=50,
    )
    features = extract_features(response)
    assert features.total_cells == 4
    # Forest=4, Mountain=5, Ocean→Empty=0, Settlement=1
    assert features.class_counts[4] == 1  # Forest
    assert features.class_counts[5] == 1  # Mountain
    assert features.class_counts[0] == 1  # Ocean→Empty
    assert features.class_counts[1] == 1  # Settlement
