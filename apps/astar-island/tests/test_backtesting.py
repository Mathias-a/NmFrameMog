"""Tests for the backtesting harness."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astar_island.backtesting.client import AstarClient
from astar_island.backtesting.fixtures import (
    GroundTruthFixture,
    list_fixtures,
    load_fixture,
    save_fixture,
)
from astar_island.backtesting.runner import (
    backtest_strategy,
)
from astar_island.prediction import PredictionTensor, make_uniform_prediction
from astar_island.terrain import NUM_PREDICTION_CLASSES

# ── Helpers ──────────────────────────────────────────────────────────


def _make_soft_gt(width: int, height: int, seed: int = 42) -> PredictionTensor:
    """Create a soft ground-truth tensor with Dirichlet samples."""
    rng = np.random.default_rng(seed)
    gt: PredictionTensor = rng.dirichlet(
        np.full(NUM_PREDICTION_CLASSES, 0.5), size=(width, height)
    )
    return gt


def _uniform_strategy(
    grid: list[list[int]], width: int, height: int
) -> PredictionTensor:
    """Baseline strategy: uniform prediction for every cell."""
    return make_uniform_prediction(width, height)


# ── Fixture save/load round-trip ─────────────────────────────────────


class TestFixturePersistence:
    """Test fixture serialization preserves data fidelity."""

    def test_save_load_roundtrip_preserves_float64(self, tmp_path: Path) -> None:
        """Saving and loading a fixture preserves float64 precision."""
        gt = _make_soft_gt(10, 8, seed=99)
        initial_grid = [[0, 1, 2] for _ in range(10)]

        fixture = GroundTruthFixture(
            round_id="round-abc",
            seed_index=3,
            ground_truth=gt,
            initial_grid=initial_grid,
            official_score=72.5,
        )

        fixture_dir = tmp_path / "data" / "fixtures"
        with patch("astar_island.backtesting.fixtures.FIXTURE_DIR", fixture_dir):
            path = save_fixture(fixture)
            assert path.exists()

            loaded = load_fixture("round-abc", 3)

        assert loaded.round_id == fixture.round_id
        assert loaded.seed_index == fixture.seed_index
        assert loaded.official_score == fixture.official_score
        assert loaded.initial_grid == fixture.initial_grid
        assert loaded.ground_truth.dtype == np.float64
        np.testing.assert_array_almost_equal(
            loaded.ground_truth, fixture.ground_truth, decimal=10
        )

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        """save_fixture creates missing parent directories."""
        gt = _make_soft_gt(4, 4)
        fixture = GroundTruthFixture(
            round_id="r1",
            seed_index=0,
            ground_truth=gt,
            initial_grid=[[0] * 4 for _ in range(4)],
            official_score=50.0,
        )
        deep_dir = tmp_path / "deep" / "nested" / "fixtures"
        with patch("astar_island.backtesting.fixtures.FIXTURE_DIR", deep_dir):
            path = save_fixture(fixture)
        assert path.exists()


# ── list_fixtures ────────────────────────────────────────────────────


class TestListFixtures:
    """Test fixture discovery."""

    def test_list_fixtures_finds_saved_files(self, tmp_path: Path) -> None:
        """list_fixtures returns IDs matching saved fixtures."""
        gt = _make_soft_gt(4, 4)
        fixture_dir = tmp_path / "data" / "fixtures"

        with patch("astar_island.backtesting.fixtures.FIXTURE_DIR", fixture_dir):
            for seed_idx in range(3):
                save_fixture(
                    GroundTruthFixture(
                        round_id="round-x",
                        seed_index=seed_idx,
                        ground_truth=gt,
                        initial_grid=[[0] * 4 for _ in range(4)],
                        official_score=60.0,
                    )
                )

            found = list_fixtures()

        assert len(found) == 3
        assert ("round-x", 0) in found
        assert ("round-x", 1) in found
        assert ("round-x", 2) in found

    def test_list_fixtures_empty_dir(self, tmp_path: Path) -> None:
        """list_fixtures returns empty list when no fixtures exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("astar_island.backtesting.fixtures.FIXTURE_DIR", empty_dir):
            assert list_fixtures() == []

    def test_list_fixtures_nonexistent_dir(self, tmp_path: Path) -> None:
        """list_fixtures returns empty list when directory doesn't exist."""
        with patch(
            "astar_island.backtesting.fixtures.FIXTURE_DIR",
            tmp_path / "nope",
        ):
            assert list_fixtures() == []


# ── Backtest runner ──────────────────────────────────────────────────


class TestBacktestRunner:
    """Test the backtest runner with uniform strategy."""

    def test_uniform_strategy_scores_finite_and_in_range(self, tmp_path: Path) -> None:
        """Uniform strategy produces scores in [0, 100]."""
        width, height = 10, 10
        gt = _make_soft_gt(width, height, seed=77)
        fixture_dir = tmp_path / "fixtures"

        with patch("astar_island.backtesting.fixtures.FIXTURE_DIR", fixture_dir):
            save_fixture(
                GroundTruthFixture(
                    round_id="test-round",
                    seed_index=0,
                    ground_truth=gt,
                    initial_grid=[[0] * height for _ in range(width)],
                    official_score=55.0,
                )
            )

            summary = backtest_strategy(_uniform_strategy)

        assert len(summary.results) == 1
        result = summary.results[0]

        assert 0.0 <= result.local_score <= 100.0
        assert np.isfinite(result.local_score)
        assert np.isfinite(result.weighted_kl_value)
        assert result.weighted_kl_value >= 0.0

    def test_score_delta_computed_correctly(self, tmp_path: Path) -> None:
        """score_delta = local_score - official_score."""
        width, height = 6, 6
        gt = _make_soft_gt(width, height, seed=88)
        official = 42.0
        fixture_dir = tmp_path / "fixtures"

        with patch("astar_island.backtesting.fixtures.FIXTURE_DIR", fixture_dir):
            save_fixture(
                GroundTruthFixture(
                    round_id="delta-round",
                    seed_index=0,
                    ground_truth=gt,
                    initial_grid=[[0] * height for _ in range(width)],
                    official_score=official,
                )
            )

            summary = backtest_strategy(_uniform_strategy)

        result = summary.results[0]
        expected_delta = result.local_score - official
        assert abs(result.score_delta - expected_delta) < 1e-12

    def test_backtest_summary_means(self, tmp_path: Path) -> None:
        """Summary means are correctly computed from results."""
        width, height = 5, 5
        fixture_dir = tmp_path / "fixtures"

        with patch("astar_island.backtesting.fixtures.FIXTURE_DIR", fixture_dir):
            for seed_idx in range(3):
                gt = _make_soft_gt(width, height, seed=seed_idx)
                save_fixture(
                    GroundTruthFixture(
                        round_id="multi",
                        seed_index=seed_idx,
                        ground_truth=gt,
                        initial_grid=[[0] * height for _ in range(width)],
                        official_score=50.0 + seed_idx * 10,
                    )
                )

            summary = backtest_strategy(_uniform_strategy)

        assert len(summary.results) == 3

        scores = [r.local_score for r in summary.results]
        assert abs(summary.mean_local_score - sum(scores) / 3) < 1e-12

    def test_backtest_no_fixtures_raises(self, tmp_path: Path) -> None:
        """backtest_strategy raises ValueError when no fixtures exist."""
        with patch(
            "astar_island.backtesting.fixtures.FIXTURE_DIR",
            tmp_path / "empty",
        ):
            with pytest.raises(ValueError, match="No fixtures"):
                backtest_strategy(_uniform_strategy)

    def test_backtest_filter_by_round_id(self, tmp_path: Path) -> None:
        """round_ids filter selects only matching fixtures."""
        width, height = 5, 5
        fixture_dir = tmp_path / "fixtures"

        with patch("astar_island.backtesting.fixtures.FIXTURE_DIR", fixture_dir):
            for rid in ("keep", "skip"):
                gt = _make_soft_gt(width, height, seed=hash(rid) % 1000)
                save_fixture(
                    GroundTruthFixture(
                        round_id=rid,
                        seed_index=0,
                        ground_truth=gt,
                        initial_grid=[[0] * height for _ in range(width)],
                        official_score=60.0,
                    )
                )

            summary = backtest_strategy(_uniform_strategy, round_ids=["keep"])

        assert len(summary.results) == 1
        assert summary.results[0].round_id == "keep"


# ── Client construction ──────────────────────────────────────────────


class TestAstarClient:
    """Test AstarClient construction without network calls."""

    def test_client_construction_with_token(self) -> None:
        """Client constructs with explicit token."""
        client = AstarClient(token="test-token-123")
        assert client.token == "test-token-123"
        assert client.base_url == "https://api.ainm.no"
        client.close()

    def test_client_construction_default(self) -> None:
        """Client constructs with defaults."""
        with patch.dict("os.environ", {}, clear=False):
            client = AstarClient()
            assert client.base_url == "https://api.ainm.no"
            client.close()

    def test_client_context_manager(self) -> None:
        """Client works as context manager."""
        with AstarClient(token="ctx-token") as client:
            assert client.token == "ctx-token"

    def test_client_env_token(self) -> None:
        """Client picks up token from environment."""
        with patch.dict("os.environ", {"ASTAR_API_TOKEN": "env-tok"}):
            client = AstarClient()
            # The token is resolved internally, just verify it doesn't crash
            client.close()
