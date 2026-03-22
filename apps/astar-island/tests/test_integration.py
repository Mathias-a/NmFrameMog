"""Integration tests for the HybridSolver full solve pipeline.

Mocks astar_island.solver.simulate_query and astar_island.solver.save_prediction
to exercise the full solve_round pipeline without any real HTTP calls or
loading a saved model from disk.
"""

import numpy as np
from unittest.mock import MagicMock, patch, call

from astar_island.config import Config
from astar_island.solver import HybridSolver
from astar_island.types import H, W, K, N_SEEDS, VIEWPORT


# ---------------------------------------------------------------------------
# Constants derived from default Config
# ---------------------------------------------------------------------------

_SEED_CAPS = (15, 12, 10, 8, 5)  # matches Config.seed_caps default
_TOTAL_QUERIES = sum(_SEED_CAPS)  # 50


# ---------------------------------------------------------------------------
# Mock response helper
# ---------------------------------------------------------------------------


def _mock_simulate_response(round_id, seed_index, vx, vy, viewport_size):
    """Return a valid simulation response with random terrain codes."""
    # Return a 15×15 grid of random terrain codes from {0, 1, 3, 4, 11}
    # (exclude 2=Port, 5=Mountain, 10=Ocean to keep it simple)
    rng = np.random.default_rng(hash((seed_index, vx, vy)) % 2**32)
    codes = rng.choice([0, 1, 3, 4, 11], size=(viewport_size, viewport_size))
    return {
        "grid": codes.tolist(),
        "viewport": {"x": vx, "y": vy, "w": viewport_size, "h": viewport_size},
    }


# ---------------------------------------------------------------------------
# Synthetic training helper
# ---------------------------------------------------------------------------


def _build_trained_solver():
    """Build and return a HybridSolver trained on synthetic data.

    Uses LightGBM with minimal estimators for speed in tests.
    """
    config = Config(
        prior_model="lightgbm",
        lgb_n_estimators=5,
        lgb_num_leaves=4,
        lgb_max_depth=3,
        lgb_min_data_in_leaf=1,
    )
    solver = HybridSolver(config)

    # Create synthetic training data: 3 fake rounds × 5 seeds
    rounds = {}
    rng = np.random.default_rng(42)
    for r in range(1, 4):
        seeds = []
        for s in range(5):
            grid = np.full((40, 40), 11, dtype=np.int32)  # plains
            grid[0, :] = 10  # ocean border top
            grid[:, 0] = 10  # ocean border left
            grid[20, 20] = 5  # mountain
            grid[15, 15] = 1  # settlement

            gt = rng.dirichlet(np.ones(6), size=(40, 40))
            # Make static cells deterministic
            gt[grid == 10] = [1, 0, 0, 0, 0, 0]  # ocean → class 0
            gt[grid == 5] = [0, 0, 0, 0, 0, 1]  # mountain → class 5
            seeds.append((grid.astype(np.int_), gt))
        rounds[r] = seeds

    solver.train_offline(rounds)
    return solver


def _make_initial_grids():
    """Create 5 initial grids matching the synthetic training pattern."""
    grids = []
    for _ in range(N_SEEDS):
        grid = np.full((H, W), 11, dtype=np.int32)  # plains
        grid[0, :] = 10  # ocean border top
        grid[:, 0] = 10  # ocean border left
        grid[20, 20] = 5  # mountain
        grid[15, 15] = 1  # settlement
        grids.append(grid)
    return grids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_solve_round_query_count():
    """solve_round should call simulate_query exactly 50 times total (sum of seed_caps)."""
    solver = _build_trained_solver()
    mock_client = MagicMock()
    initial_grids = _make_initial_grids()
    round_id = "test-round-001"

    with (
        patch("astar_island.solver.simulate_query") as mock_sim,
        patch("astar_island.solver.save_prediction"),
    ):
        mock_sim.side_effect = lambda client, rid, seed_index, vx, vy, vs: (
            _mock_simulate_response(rid, seed_index, vx, vy, vs)
        )

        solver.solve_round(mock_client, round_id, initial_grids)

    assert mock_sim.call_count == _TOTAL_QUERIES, (
        f"Expected {_TOTAL_QUERIES} simulate_query calls, got {mock_sim.call_count}"
    )


def test_solve_round_prediction_shape():
    """Each returned prediction must have shape (40, 40, 6)."""
    solver = _build_trained_solver()
    mock_client = MagicMock()
    initial_grids = _make_initial_grids()
    round_id = "test-round-002"

    with (
        patch("astar_island.solver.simulate_query") as mock_sim,
        patch("astar_island.solver.save_prediction"),
    ):
        mock_sim.side_effect = lambda client, rid, seed_index, vx, vy, vs: (
            _mock_simulate_response(rid, seed_index, vx, vy, vs)
        )

        predictions = solver.solve_round(mock_client, round_id, initial_grids)

    assert len(predictions) == N_SEEDS, (
        f"Expected {N_SEEDS} predictions, got {len(predictions)}"
    )
    for s, pred in enumerate(predictions):
        assert pred.shape == (H, W, K), (
            f"Seed {s}: expected shape ({H}, {W}, {K}), got {pred.shape}"
        )


def test_solve_round_prediction_valid():
    """Predictions must be valid probability distributions with no NaN/Inf."""
    solver = _build_trained_solver()
    mock_client = MagicMock()
    initial_grids = _make_initial_grids()
    round_id = "test-round-003"

    with (
        patch("astar_island.solver.simulate_query") as mock_sim,
        patch("astar_island.solver.save_prediction"),
    ):
        mock_sim.side_effect = lambda client, rid, seed_index, vx, vy, vs: (
            _mock_simulate_response(rid, seed_index, vx, vy, vs)
        )

        predictions = solver.solve_round(mock_client, round_id, initial_grids)

    for s, (pred, grid) in enumerate(zip(predictions, initial_grids)):
        # No NaN or Inf
        assert not np.any(np.isnan(pred)), f"Seed {s}: NaN values in prediction"
        assert not np.any(np.isinf(pred)), f"Seed {s}: Inf values in prediction"

        # All values >= 0
        assert np.all(pred >= 0), f"Seed {s}: negative probability values"

        # Each cell sums to ~1.0
        cell_sums = pred.sum(axis=-1)  # (H, W)
        np.testing.assert_allclose(
            cell_sums,
            np.ones((H, W)),
            atol=1e-6,
            err_msg=f"Seed {s}: cell probabilities don't sum to 1.0",
        )

        # Static cells (ocean=10, mountain=5) must be one-hot
        ocean_mask = grid == 10
        mountain_mask = grid == 5

        # Ocean cells: class 0 should be 1.0, others 0.0
        if ocean_mask.any():
            ocean_preds = pred[ocean_mask]  # (n_ocean, K)
            expected_ocean = np.zeros(K)
            expected_ocean[0] = 1.0
            np.testing.assert_allclose(
                ocean_preds,
                np.tile(expected_ocean, (ocean_preds.shape[0], 1)),
                atol=1e-6,
                err_msg=f"Seed {s}: ocean cells are not one-hot at class 0",
            )

        # Mountain cells: class 5 should be 1.0, others 0.0
        if mountain_mask.any():
            mountain_preds = pred[mountain_mask]  # (n_mountain, K)
            expected_mountain = np.zeros(K)
            expected_mountain[5] = 1.0
            np.testing.assert_allclose(
                mountain_preds,
                np.tile(expected_mountain, (mountain_preds.shape[0], 1)),
                atol=1e-6,
                err_msg=f"Seed {s}: mountain cells are not one-hot at class 5",
            )


def test_solve_round_budget_per_seed():
    """Each seed should use exactly its seed_cap queries: (15, 12, 10, 8, 5)."""
    solver = _build_trained_solver()
    mock_client = MagicMock()
    initial_grids = _make_initial_grids()
    round_id = "test-round-004"

    # Track call counts per seed_index
    calls_per_seed = {i: 0 for i in range(N_SEEDS)}

    def tracking_simulate(client, rid, seed_index, vx, vy, viewport_size):
        calls_per_seed[seed_index] += 1
        return _mock_simulate_response(rid, seed_index, vx, vy, viewport_size)

    with (
        patch("astar_island.solver.simulate_query") as mock_sim,
        patch("astar_island.solver.save_prediction"),
    ):
        mock_sim.side_effect = tracking_simulate

        solver.solve_round(mock_client, round_id, initial_grids)

    expected_caps = {i: cap for i, cap in enumerate(_SEED_CAPS)}
    for seed_idx, expected in expected_caps.items():
        actual = calls_per_seed[seed_idx]
        assert actual == expected, (
            f"Seed {seed_idx}: expected {expected} queries, got {actual}"
        )
