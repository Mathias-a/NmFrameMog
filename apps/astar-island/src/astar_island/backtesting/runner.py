"""Backtest runner for scoring prediction strategies against fixtures.

Loads frozen ground-truth fixtures and evaluates a strategy function
to produce comparable scores and metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean

from astar_island.backtesting.fixtures import list_fixtures, load_fixture
from astar_island.prediction import (
    PredictionTensor,
    apply_probability_floor,
)
from astar_island.scoring import competition_score, weighted_kl

PredictionStrategy = Callable[[list[list[int]], int, int], PredictionTensor]
"""A strategy takes (initial_grid, width, height) and returns a prediction tensor."""


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Result of backtesting one strategy against one fixture."""

    round_id: str
    seed_index: int
    local_score: float
    official_score: float
    score_delta: float
    weighted_kl_value: float


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    """Aggregate results across all fixtures."""

    results: tuple[BacktestResult, ...]
    mean_local_score: float
    mean_official_score: float
    mean_score_delta: float
    mean_weighted_kl: float


def backtest_strategy(
    strategy: PredictionStrategy,
    round_ids: list[str] | None = None,
    floor: float = 0.01,
) -> BacktestSummary:
    """Run a prediction strategy against all available fixtures.

    Args:
        strategy: Function (initial_grid, width, height) -> PredictionTensor.
        round_ids: If provided, only backtest against these rounds.
        floor: Probability floor applied to predictions before scoring.

    Returns:
        Summary with per-fixture results and aggregate metrics.

    Raises:
        ValueError: If no fixtures are available for the given round_ids.
    """
    available = list_fixtures()

    if round_ids is not None:
        round_id_set = set(round_ids)
        available = [(rid, sid) for rid, sid in available if rid in round_id_set]

    if not available:
        msg = "No fixtures available for backtesting"
        raise ValueError(msg)

    results: list[BacktestResult] = []

    for round_id, seed_index in available:
        fixture = load_fixture(round_id, seed_index)
        gt = fixture.ground_truth

        width = len(fixture.initial_grid)
        height = len(fixture.initial_grid[0]) if fixture.initial_grid else 0

        raw_prediction = strategy(fixture.initial_grid, width, height)
        safe_prediction = apply_probability_floor(raw_prediction, floor=floor)

        local_score = competition_score(gt, safe_prediction)
        wkl = weighted_kl(gt, safe_prediction)
        delta = local_score - fixture.official_score

        results.append(
            BacktestResult(
                round_id=round_id,
                seed_index=seed_index,
                local_score=local_score,
                official_score=fixture.official_score,
                score_delta=delta,
                weighted_kl_value=wkl,
            )
        )

    return BacktestSummary(
        results=tuple(results),
        mean_local_score=mean(r.local_score for r in results),
        mean_official_score=mean(r.official_score for r in results),
        mean_score_delta=mean(r.score_delta for r in results),
        mean_weighted_kl=mean(r.weighted_kl_value for r in results),
    )
