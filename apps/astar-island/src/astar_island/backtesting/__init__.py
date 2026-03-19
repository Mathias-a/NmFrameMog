"""Backtesting harness for Astar Island prediction strategies.

Fetches historical rounds, freezes ground truth as local fixtures,
and scores prediction strategies offline against known outcomes.
"""

from __future__ import annotations

from astar_island.backtesting.client import (
    AnalysisResult,
    AstarClient,
    InitialState,
    RoundData,
    RoundInfo,
)
from astar_island.backtesting.fixtures import (
    GroundTruthFixture,
    fetch_and_freeze_all,
    list_fixtures,
    load_fixture,
    save_fixture,
)
from astar_island.backtesting.runner import (
    BacktestResult,
    BacktestSummary,
    PredictionStrategy,
    backtest_strategy,
)

__all__ = [
    "AnalysisResult",
    "AstarClient",
    "BacktestResult",
    "BacktestSummary",
    "GroundTruthFixture",
    "InitialState",
    "PredictionStrategy",
    "RoundData",
    "RoundInfo",
    "backtest_strategy",
    "fetch_and_freeze_all",
    "list_fixtures",
    "load_fixture",
    "save_fixture",
]
