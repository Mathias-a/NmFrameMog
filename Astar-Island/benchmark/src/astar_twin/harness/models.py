from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SeedResult:
    """Scoring outcome for a single (strategy, fixture, seed_index) triple."""

    seed_index: int
    score: float
    ground_truth: NDArray[np.float64]
    prediction: NDArray[np.float64]


@dataclass(frozen=True)
class StrategyReport:
    """Aggregated results for one strategy evaluated against one fixture."""

    strategy_name: str
    fixture_id: str
    seed_results: tuple[SeedResult, ...]
    scores: tuple[float, ...]
    mean_score: float

    @classmethod
    def from_seed_results(
        cls,
        strategy_name: str,
        fixture_id: str,
        seed_results: list[SeedResult],
    ) -> StrategyReport:
        scores = tuple(sr.score for sr in seed_results)
        mean = float(np.mean(scores)) if scores else 0.0
        return cls(
            strategy_name=strategy_name,
            fixture_id=fixture_id,
            seed_results=tuple(seed_results),
            scores=scores,
            mean_score=mean,
        )


@dataclass(frozen=True)
class BenchmarkReport:
    """Collection of per-strategy results for a single benchmark run."""

    strategy_reports: tuple[StrategyReport, ...]
    fixture_id: str
    fixture_ids: tuple[str, ...] = field(default_factory=tuple)

    def ranked(self) -> list[StrategyReport]:
        """Return strategy reports sorted by mean_score descending."""
        return sorted(self.strategy_reports, key=lambda r: r.mean_score, reverse=True)
