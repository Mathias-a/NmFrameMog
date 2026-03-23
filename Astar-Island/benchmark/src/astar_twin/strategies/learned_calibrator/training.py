from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astar_twin.data.loaders import list_fixtures
from astar_twin.data.models import RoundFixture
from astar_twin.harness.budget import Budget
from astar_twin.scoring import compute_score, safe_prediction
from astar_twin.strategies.calibrated_mc.strategy import (
    CalibratedMCStrategy,
    _build_coastal_mask,
    _build_static_mask,
)
from astar_twin.strategies.filter_baseline.strategy import FilterBaselineStrategy
from astar_twin.strategies.learned_calibrator.model import (
    DEFAULT_ZONE_WEIGHTS,
    ZONE_NAMES,
    blend_predictions,
    normalize_zone_weights,
)


@dataclass(frozen=True)
class HistoricalSeedExample:
    round_id: str
    round_number: int
    round_weight: float
    seed_index: int
    base_prediction: NDArray[np.float64]
    fallback_prediction: NDArray[np.float64]
    ground_truth: NDArray[np.float64]
    zone_map: NDArray[np.int8]
    is_static: NDArray[np.bool_]


@dataclass(frozen=True)
class FoldBenchmarkResult:
    held_out_round_id: str
    held_out_round_number: int
    round_weight: float
    learned_weights: dict[str, float]
    learned_scores: list[float]
    base_scores: list[float]
    fallback_scores: list[float]

    @property
    def learned_mean(self) -> float:
        return float(np.mean(self.learned_scores))

    @property
    def base_mean(self) -> float:
        return float(np.mean(self.base_scores))

    @property
    def fallback_mean(self) -> float:
        return float(np.mean(self.fallback_scores))

    def to_dict(self) -> dict[str, object]:
        return {
            "held_out_round_id": self.held_out_round_id,
            "held_out_round_number": self.held_out_round_number,
            "round_weight": self.round_weight,
            "learned_weights": dict(self.learned_weights),
            "learned_scores": list(self.learned_scores),
            "learned_mean": self.learned_mean,
            "base_scores": list(self.base_scores),
            "base_mean": self.base_mean,
            "fallback_scores": list(self.fallback_scores),
            "fallback_mean": self.fallback_mean,
        }


@dataclass(frozen=True)
class CrossValidationReport:
    folds: tuple[FoldBenchmarkResult, ...]

    def _weighted_mean(self, attr: str) -> float:
        total_weight = sum(fold.round_weight for fold in self.folds)
        if total_weight <= 0.0:
            return 0.0
        numerator = sum(getattr(fold, attr) * fold.round_weight for fold in self.folds)
        return float(numerator / total_weight)

    @property
    def learned_weighted_mean(self) -> float:
        return self._weighted_mean("learned_mean")

    @property
    def base_weighted_mean(self) -> float:
        return self._weighted_mean("base_mean")

    @property
    def fallback_weighted_mean(self) -> float:
        return self._weighted_mean("fallback_mean")

    def mean_learned_weights(self) -> dict[str, float]:
        if not self.folds:
            return dict(DEFAULT_ZONE_WEIGHTS)
        weight_sum = {zone_name: 0.0 for zone_name in ZONE_NAMES}
        for fold in self.folds:
            for zone_name in ZONE_NAMES:
                weight_sum[zone_name] += fold.learned_weights[zone_name]
        fold_count = float(len(self.folds))
        return {zone_name: weight_sum[zone_name] / fold_count for zone_name in ZONE_NAMES}

    def to_dict(self) -> dict[str, object]:
        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "learned_weighted_mean": self.learned_weighted_mean,
            "base_weighted_mean": self.base_weighted_mean,
            "fallback_weighted_mean": self.fallback_weighted_mean,
            "mean_learned_weights": self.mean_learned_weights(),
        }


def _is_real_fixture(fixture: RoundFixture) -> bool:
    return not fixture.id.startswith("test-") and fixture.ground_truths is not None


def build_historical_examples(
    data_dir: Path,
    n_runs: int = 25,
) -> list[HistoricalSeedExample]:
    fixtures = [fixture for fixture in list_fixtures(data_dir) if _is_real_fixture(fixture)]
    fixtures.sort(key=lambda fixture: fixture.round_number)
    if not fixtures:
        raise ValueError(f"No real fixtures with cached ground truths found under {data_dir}")

    base_strategy = CalibratedMCStrategy(n_runs=n_runs)
    zone_helper = CalibratedMCStrategy(n_runs=5)
    fallback_strategy = FilterBaselineStrategy()

    examples: list[HistoricalSeedExample] = []
    for fixture in fixtures:
        assert fixture.ground_truths is not None
        for seed_index, initial_state in enumerate(fixture.initial_states):
            budget = Budget(total=50)
            base_prediction = base_strategy.predict(
                initial_state,
                budget=budget,
                base_seed=fixture.round_number * 1000 + seed_index,
            )
            fallback_prediction = fallback_strategy.predict(
                initial_state,
                budget=budget,
                base_seed=0,
            )
            grid = initial_state.grid
            height = len(grid)
            width = len(grid[0])
            is_static = _build_static_mask(grid, height, width)
            is_coastal = _build_coastal_mask(grid, height, width)
            zone_map = zone_helper._build_zone_map(
                initial_state,
                height,
                width,
                is_static,
                is_coastal,
            )
            examples.append(
                HistoricalSeedExample(
                    round_id=fixture.id,
                    round_number=fixture.round_number,
                    round_weight=fixture.round_weight,
                    seed_index=seed_index,
                    base_prediction=np.asarray(base_prediction, dtype=np.float64),
                    fallback_prediction=np.asarray(fallback_prediction, dtype=np.float64),
                    ground_truth=np.asarray(fixture.ground_truths[seed_index], dtype=np.float64),
                    zone_map=zone_map,
                    is_static=is_static,
                )
            )
    return examples


def score_examples(
    examples: Iterable[HistoricalSeedExample],
    zone_weights: dict[str, float],
) -> tuple[list[float], float]:
    scores: list[float] = []
    weighted_total = 0.0
    weight_sum = 0.0
    for example in examples:
        blended = blend_predictions(
            example.base_prediction,
            example.fallback_prediction,
            example.zone_map,
            example.is_static,
            zone_weights,
        )
        score = float(compute_score(example.ground_truth, safe_prediction(blended)))
        scores.append(score)
        weighted_total += score * example.round_weight
        weight_sum += example.round_weight
    weighted_mean = weighted_total / weight_sum if weight_sum > 0.0 else 0.0
    return scores, float(weighted_mean)


def fit_zone_weights(
    examples: Sequence[HistoricalSeedExample],
    weight_grid: Sequence[float] | None = None,
    n_passes: int = 3,
) -> dict[str, float]:
    if not examples:
        raise ValueError("fit_zone_weights requires at least one training example")

    candidate_weights = tuple(weight_grid or np.linspace(0.0, 0.4, 9).tolist())
    learned = dict(DEFAULT_ZONE_WEIGHTS)
    _, best_score = score_examples(examples, learned)

    for _ in range(n_passes):
        improved = False
        for zone_name in ZONE_NAMES:
            zone_best_weight = learned[zone_name]
            zone_best_score = best_score
            for candidate in candidate_weights:
                trial = dict(learned)
                trial[zone_name] = float(candidate)
                _, trial_score = score_examples(examples, trial)
                if trial_score > zone_best_score + 1e-9:
                    zone_best_score = trial_score
                    zone_best_weight = float(candidate)
            if abs(zone_best_weight - learned[zone_name]) > 1e-12:
                learned[zone_name] = zone_best_weight
                best_score = zone_best_score
                improved = True
        if not improved:
            break

    return normalize_zone_weights(learned)


def cross_validate_zone_calibrator(
    data_dir: Path,
    n_runs: int = 25,
    weight_grid: Sequence[float] | None = None,
) -> CrossValidationReport:
    examples = build_historical_examples(data_dir, n_runs=n_runs)
    round_ids = sorted({example.round_id for example in examples})
    folds: list[FoldBenchmarkResult] = []

    for held_out_round_id in round_ids:
        train_examples = [example for example in examples if example.round_id != held_out_round_id]
        test_examples = [example for example in examples if example.round_id == held_out_round_id]
        if not test_examples:
            continue
        learned_weights = fit_zone_weights(train_examples, weight_grid=weight_grid)
        learned_scores, _ = score_examples(test_examples, learned_weights)
        base_scores, _ = score_examples(test_examples, {zone_name: 0.0 for zone_name in ZONE_NAMES})
        fallback_scores, _ = score_examples(
            test_examples, {zone_name: 1.0 for zone_name in ZONE_NAMES}
        )
        held_out_round = test_examples[0]
        folds.append(
            FoldBenchmarkResult(
                held_out_round_id=held_out_round.round_id,
                held_out_round_number=held_out_round.round_number,
                round_weight=held_out_round.round_weight,
                learned_weights=learned_weights,
                learned_scores=learned_scores,
                base_scores=base_scores,
                fallback_scores=fallback_scores,
            )
        )

    return CrossValidationReport(
        folds=tuple(sorted(folds, key=lambda fold: fold.held_out_round_number))
    )
