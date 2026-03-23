from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedResult:
    """Score for one strategy on one seed."""

    strategy_name: str
    seed_index: int
    score: float

    def __post_init__(self) -> None:
        if not self.strategy_name:
            raise ValueError("strategy_name must be non-empty.")
        if self.seed_index < 0:
            raise ValueError("seed_index must be non-negative.")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite.")


@dataclass(frozen=True)
class StrategyReport:
    """Aggregated results for one strategy across all seeds."""

    strategy_name: str
    scores: tuple[float, ...]
    mean_score: float
    min_score: float
    max_score: float

    def __post_init__(self) -> None:
        if not self.strategy_name:
            raise ValueError("strategy_name must be non-empty.")
        if not self.scores:
            raise ValueError("scores must be non-empty.")
        if not all(math.isfinite(s) for s in self.scores):
            raise ValueError("All scores must be finite.")
        if not math.isfinite(self.mean_score):
            raise ValueError("mean_score must be finite.")

    @classmethod
    def from_seed_results(
        cls, strategy_name: str, seed_results: list[SeedResult]
    ) -> StrategyReport:
        if not seed_results:
            raise ValueError("seed_results must be non-empty.")
        scores = tuple(r.score for r in sorted(seed_results, key=lambda r: r.seed_index))
        return cls(
            strategy_name=strategy_name,
            scores=scores,
            mean_score=math.fsum(scores) / len(scores),
            min_score=min(scores),
            max_score=max(scores),
        )


@dataclass(frozen=True)
class BenchmarkReport:
    """Full benchmark results across all strategies and seeds."""

    strategy_reports: tuple[StrategyReport, ...]
    fixture_id: str

    def __post_init__(self) -> None:
        if not self.strategy_reports:
            raise ValueError("strategy_reports must be non-empty.")
        if not self.fixture_id:
            raise ValueError("fixture_id must be non-empty.")


__all__ = [
    "BenchmarkReport",
    "SeedResult",
    "StrategyReport",
]
