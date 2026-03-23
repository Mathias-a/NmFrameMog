from __future__ import annotations

import numpy as np
import pytest

from astar_twin.contracts.api_models import InitialState
from astar_twin.strategies import REGISTRY, Strategy

# ── helpers ──────────────────────────────────────────────────────────────────


def _minimal_initial_state(height: int = 5, width: int = 5) -> InitialState:
    """Build a tiny InitialState for quick testing.

    IMPORTANT: No Port cells (code=2) unless adjacent to Ocean — the simulator's
    check_invariants() enforces coastal adjacency and will raise InvariantViolation.
    Grid has 6 columns so callers requesting width=6 get a valid result.
    """
    # Terrain codes: Ocean=10, Plains=11, Forest=4, Mountain=5,
    #                Empty=0, Settlement=1, Ruin=3  (Port=2 deliberately omitted)
    grid = [
        [10, 10, 11, 4, 5, 11],
        [10, 11, 11, 4, 5, 11],
        [11, 11, 0, 1, 5, 11],
        [11, 4, 4, 3, 5, 11],  # Ruin(3) — not Port(2), which would fail coastal check
        [10, 10, 11, 3, 5, 11],
    ][:height]
    grid = [row[:width] for row in grid]
    return InitialState(grid=grid, settlements=[])


# ── protocol conformance ─────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy_cls", list(REGISTRY.values()))
def test_strategy_satisfies_protocol(strategy_cls: type[Strategy]) -> None:
    """Every registered class must satisfy the Strategy runtime-checkable Protocol."""
    instance = strategy_cls()
    assert isinstance(instance, Strategy)


@pytest.mark.parametrize("strategy_cls", list(REGISTRY.values()))
def test_strategy_name_is_non_empty_string(strategy_cls: type[Strategy]) -> None:
    instance = strategy_cls()
    assert isinstance(instance.name, str)
    assert instance.name  # non-empty


@pytest.mark.parametrize("strategy_cls", list(REGISTRY.values()))
def test_strategy_name_matches_registry_key(strategy_cls: type[Strategy]) -> None:
    """REGISTRY key must match the strategy's own name."""
    instance = strategy_cls()
    matching_key = next(k for k, v in REGISTRY.items() if v is strategy_cls)
    assert matching_key == instance.name


# ── output shape and dtype ────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy_cls", list(REGISTRY.values()))
def test_predict_output_shape(strategy_cls: type[Strategy]) -> None:
    """Output must be (H, W, 6) where H, W match the initial state."""
    instance = strategy_cls()
    state = _minimal_initial_state(height=4, width=6)
    result = instance.predict(initial_state=state, budget=50, base_seed=0)
    assert result.shape == (4, 6, 6), f"Expected (4, 6, 6), got {result.shape}"


@pytest.mark.parametrize("strategy_cls", list(REGISTRY.values()))
def test_predict_output_dtype_is_float64(strategy_cls: type[Strategy]) -> None:
    instance = strategy_cls()
    state = _minimal_initial_state()
    result = instance.predict(initial_state=state, budget=50, base_seed=0)
    assert result.dtype == np.float64


@pytest.mark.parametrize("strategy_cls", list(REGISTRY.values()))
def test_predict_output_probabilities_non_negative(strategy_cls: type[Strategy]) -> None:
    instance = strategy_cls()
    state = _minimal_initial_state()
    result = instance.predict(initial_state=state, budget=50, base_seed=0)
    assert np.all(result >= 0.0)


@pytest.mark.parametrize("strategy_cls", list(REGISTRY.values()))
def test_predict_output_rows_sum_to_one(strategy_cls: type[Strategy]) -> None:
    """Each cell's 6 probabilities must sum to 1.0 (within tolerance)."""
    instance = strategy_cls()
    state = _minimal_initial_state()
    result = instance.predict(initial_state=state, budget=50, base_seed=0)
    cell_sums = result.sum(axis=2)
    assert np.allclose(cell_sums, 1.0, atol=1e-9), (
        f"Cells don't sum to 1.0: min={cell_sums.min():.6f} max={cell_sums.max():.6f}"
    )


# ── determinism ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy_cls", list(REGISTRY.values()))
def test_predict_is_deterministic_same_seed(strategy_cls: type[Strategy]) -> None:
    """Same base_seed must produce identical output."""
    instance = strategy_cls()
    state = _minimal_initial_state()
    result_a = instance.predict(initial_state=state, budget=50, base_seed=7)
    result_b = instance.predict(initial_state=state, budget=50, base_seed=7)
    np.testing.assert_array_equal(result_a, result_b)


@pytest.mark.parametrize("strategy_cls", list(REGISTRY.values()))
def test_predict_different_seeds_may_differ(strategy_cls: type[Strategy]) -> None:
    """Different seeds should (for stochastic strategies) produce different output.

    Naive/deterministic strategies will produce the same output regardless of seed —
    that's acceptable, so we only check that the call succeeds.
    """
    instance = strategy_cls()
    state = _minimal_initial_state()
    result_a = instance.predict(initial_state=state, budget=50, base_seed=1)
    result_b = instance.predict(initial_state=state, budget=50, base_seed=2)
    # Both must be valid shapes — we can't assert they differ for deterministic strategies.
    assert result_a.shape == result_b.shape
