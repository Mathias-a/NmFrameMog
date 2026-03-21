from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev

from .baseline import floor_and_normalize
from .contract import CLASS_COUNT, PROBABILITY_FLOOR
from .emulator import DEFAULT_FIXTURE_PATHS
from .pipeline import parse_round_detail_payload
from .proxy_simulator import (
    DEFAULT_SIMULATION_YEARS,
    InitialSettlement,
    build_ground_truth_tensor,
)
from .validator import entropy_weighted_kl_score, validate_prediction_tensor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    predict: Callable[[list[list[int]]], list[list[list[float]]]]


@dataclass(frozen=True)
class BenchmarkConfig:
    rollout_count: int = 128
    years: int = DEFAULT_SIMULATION_YEARS
    base_seed: int = 20260320
    seed_indices: tuple[int, ...] | None = None

    @classmethod
    def quick(cls) -> BenchmarkConfig:
        return cls(rollout_count=16)

    @classmethod
    def full(cls) -> BenchmarkConfig:
        return cls(rollout_count=256)


@dataclass(frozen=True)
class SeedResult:
    round_id: str
    seed_index: int
    score: float
    prediction_time_ms: float
    error: str | None = None


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    mean_score: float
    median_score: float
    stdev_score: float
    min_score: float
    max_score: float
    total_time_ms: float
    seed_results: tuple[SeedResult, ...]


@dataclass(frozen=True)
class BenchmarkReport:
    config: BenchmarkConfig
    model_results: tuple[ModelResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": {
                "rollout_count": self.config.rollout_count,
                "years": self.config.years,
                "base_seed": self.config.base_seed,
                "seed_indices": list(self.config.seed_indices)
                if self.config.seed_indices is not None
                else None,
                "class_count": CLASS_COUNT,
                "probability_floor": PROBABILITY_FLOOR,
            },
            "model_results": [
                {
                    "model_name": model_result.model_name,
                    "mean_score": model_result.mean_score,
                    "median_score": model_result.median_score,
                    "stdev_score": model_result.stdev_score,
                    "min_score": model_result.min_score,
                    "max_score": model_result.max_score,
                    "total_time_ms": model_result.total_time_ms,
                    "seed_results": [
                        {
                            "round_id": seed_result.round_id,
                            "seed_index": seed_result.seed_index,
                            "score": seed_result.score,
                            "prediction_time_ms": seed_result.prediction_time_ms,
                            "error": seed_result.error,
                        }
                        for seed_result in model_result.seed_results
                    ],
                }
                for model_result in self.model_results
            ],
        }

    def format_table(self) -> str:
        headers = (
            "Model",
            "Mean",
            "Median",
            "StDev",
            "Min",
            "Max",
            "Time(ms)",
        )
        sorted_results = sorted(
            self.model_results,
            key=lambda result: (-result.mean_score, result.model_name),
        )
        row_values = [
            (
                result.model_name,
                f"{result.mean_score:.2f}",
                f"{result.median_score:.2f}",
                f"{result.stdev_score:.2f}",
                f"{result.min_score:.2f}",
                f"{result.max_score:.2f}",
                f"{result.total_time_ms:.0f}",
            )
            for result in sorted_results
        ]
        widths = [len(header) for header in headers]
        for row in row_values:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def format_row(row: tuple[str, ...]) -> str:
            cells: list[str] = []
            for index, cell in enumerate(row):
                if index == 0:
                    cells.append(cell.ljust(widths[index]))
                    continue
                cells.append(cell.rjust(widths[index]))
            return "| " + " | ".join(cells) + " |"

        separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
        lines = [format_row(headers), separator]
        lines.extend(format_row(row) for row in row_values)
        return "\n".join(lines)


@dataclass(frozen=True)
class _LoadedBenchmarkRound:
    round_id: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_grids: tuple[list[list[int]], ...]
    initial_settlements: tuple[tuple[InitialSettlement, ...], ...]


class BenchmarkRunner:
    def __init__(self, rounds: Sequence[_LoadedBenchmarkRound]) -> None:
        if not rounds:
            raise ValueError("At least one benchmark round is required.")
        round_ids = {round_data.round_id for round_data in rounds}
        if len(round_ids) != len(rounds):
            raise ValueError("Duplicate round IDs are not supported.")
        self._rounds = tuple(rounds)
        self._ground_truth_cache: dict[tuple[str, int], list[list[list[float]]]] = {}

    @classmethod
    def from_fixture_paths(
        cls,
        fixture_paths: Sequence[Path] | None = None,
    ) -> BenchmarkRunner:
        paths = tuple(fixture_paths or DEFAULT_FIXTURE_PATHS)
        loaded_rounds = tuple(
            _load_benchmark_round(_resolve_fixture_path(path)) for path in paths
        )
        return cls(loaded_rounds)

    def compare(
        self,
        models: Sequence[ModelSpec],
        config: BenchmarkConfig | None = None,
    ) -> BenchmarkReport:
        if not models:
            raise ValueError("At least one model must be provided.")
        benchmark_config = config or BenchmarkConfig()
        round_seed_pairs = self._round_seed_pairs(benchmark_config)
        if not round_seed_pairs:
            raise ValueError("Benchmark configuration selected no seeds.")

        for round_data, seed_index in round_seed_pairs:
            self._get_ground_truth(
                round_data, seed_index=seed_index, config=benchmark_config
            )

        model_results = tuple(
            self._evaluate_model(
                model=model,
                config=benchmark_config,
                round_seed_pairs=round_seed_pairs,
            )
            for model in models
        )
        return BenchmarkReport(config=benchmark_config, model_results=model_results)

    def _round_seed_pairs(
        self,
        config: BenchmarkConfig,
    ) -> tuple[tuple[_LoadedBenchmarkRound, int], ...]:
        round_seed_pairs: list[tuple[_LoadedBenchmarkRound, int]] = []
        for round_data in self._rounds:
            if config.seed_indices is None:
                selected_seed_indices = tuple(range(round_data.seeds_count))
            else:
                _validate_seed_indices(
                    config.seed_indices, seeds_count=round_data.seeds_count
                )
                selected_seed_indices = config.seed_indices
            for seed_index in selected_seed_indices:
                round_seed_pairs.append((round_data, seed_index))
        return tuple(round_seed_pairs)

    def _get_ground_truth(
        self,
        round_data: _LoadedBenchmarkRound,
        *,
        seed_index: int,
        config: BenchmarkConfig,
    ) -> list[list[list[float]]]:
        cache_key = (round_data.round_id, seed_index)
        cached = self._ground_truth_cache.get(cache_key)
        if cached is not None:
            return cached

        ground_truth = build_ground_truth_tensor(
            round_data.initial_grids[seed_index],
            round_data.initial_settlements[seed_index],
            rollout_count=config.rollout_count,
            base_seed=_analysis_seed(config.base_seed, round_data.round_id, seed_index),
            years=config.years,
        )
        normalized_ground_truth = [
            [floor_and_normalize(cell, probability_floor=0.0) for cell in row]
            for row in ground_truth
        ]
        self._ground_truth_cache[cache_key] = normalized_ground_truth
        return normalized_ground_truth

    def _evaluate_model(
        self,
        *,
        model: ModelSpec,
        config: BenchmarkConfig,
        round_seed_pairs: Sequence[tuple[_LoadedBenchmarkRound, int]],
    ) -> ModelResult:
        seed_results: list[SeedResult] = []
        for round_data, seed_index in round_seed_pairs:
            seed_results.append(
                self._score_seed(
                    model=model,
                    round_data=round_data,
                    seed_index=seed_index,
                    config=config,
                )
            )

        scores = [seed_result.score for seed_result in seed_results]
        total_time_ms = sum(
            seed_result.prediction_time_ms for seed_result in seed_results
        )
        stdev_score = stdev(scores) if len(scores) > 1 else 0.0
        return ModelResult(
            model_name=model.name,
            mean_score=mean(scores),
            median_score=median(scores),
            stdev_score=stdev_score,
            min_score=min(scores),
            max_score=max(scores),
            total_time_ms=total_time_ms,
            seed_results=tuple(seed_results),
        )

    def _score_seed(
        self,
        *,
        model: ModelSpec,
        round_data: _LoadedBenchmarkRound,
        seed_index: int,
        config: BenchmarkConfig,
    ) -> SeedResult:
        initial_grid = _copy_grid(round_data.initial_grids[seed_index])
        started_at_ns = time.perf_counter_ns()
        try:
            prediction = model.predict(initial_grid)
            elapsed_ms = (time.perf_counter_ns() - started_at_ns) / 1_000_000
            validate_prediction_tensor(
                prediction,
                width=round_data.map_width,
                height=round_data.map_height,
            )
            score = entropy_weighted_kl_score(
                prediction,
                self._get_ground_truth(
                    round_data, seed_index=seed_index, config=config
                ),
            )
            return SeedResult(
                round_id=round_data.round_id,
                seed_index=seed_index,
                score=score,
                prediction_time_ms=elapsed_ms,
            )
        except Exception as error:
            elapsed_ms = (time.perf_counter_ns() - started_at_ns) / 1_000_000
            return SeedResult(
                round_id=round_data.round_id,
                seed_index=seed_index,
                score=0.0,
                prediction_time_ms=elapsed_ms,
                error=str(error),
            )


def _load_benchmark_round(path: Path) -> _LoadedBenchmarkRound:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    round_detail = parse_round_detail_payload(payload)
    initial_settlements = tuple(
        _extract_initial_settlements_from_grid(grid)
        for grid in round_detail.initial_states
    )
    return _LoadedBenchmarkRound(
        round_id=round_detail.round_id,
        map_width=round_detail.map_width,
        map_height=round_detail.map_height,
        seeds_count=round_detail.seeds_count,
        initial_grids=round_detail.initial_states,
        initial_settlements=initial_settlements,
    )


def _resolve_fixture_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / path).resolve()


def _analysis_seed(base_seed: int, round_id: str, seed_index: int) -> int:
    round_component = sum(ord(character) for character in round_id)
    return base_seed + round_component * 101 + seed_index * 10_000


def _extract_initial_settlements_from_grid(
    grid: list[list[int]],
) -> tuple[InitialSettlement, ...]:
    settlements: list[InitialSettlement] = []
    next_owner_id = 1
    for y, row in enumerate(grid):
        for x, terrain_code in enumerate(row):
            if terrain_code not in {1, 2}:
                continue
            settlements.append(
                InitialSettlement(
                    x=x,
                    y=y,
                    has_port=terrain_code == 2,
                    alive=True,
                    owner_id=next_owner_id,
                )
            )
            next_owner_id += 1
    return tuple(settlements)


def _validate_seed_indices(seed_indices: tuple[int, ...], *, seeds_count: int) -> None:
    for seed_index in seed_indices:
        if not 0 <= seed_index < seeds_count:
            raise ValueError(
                "Invalid seed_index "
                f"{seed_index}; expected 0 <= seed_index < {seeds_count}"
            )


def _copy_grid(grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in grid]
