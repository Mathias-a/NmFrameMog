"""Benchmark tests simulating active-round performance.

Two benchmark modes:
1. Holdout simulation — for each round, train on all others, predict and score
   the held-out round.  This mirrors what happens when a new round goes live:
   the model has never seen this round's map.

2. Production artifact — use the actual saved model artifact to predict every
   fixture and score against ground truth.  Catches regressions from model
   retraining or pipeline changes.

Both are marked ``pytest.mark.benchmark`` so normal ``pytest`` skips them.
Run explicitly with ``pytest -m benchmark -v``.
"""

from __future__ import annotations

import numpy as np
import pytest

from astar_island.calibration import (
    calibrate_prediction,
    find_optimal_temperature,
)
from astar_island.features import (
    extract_cell_features,
    extract_cell_targets,
    raw_grid_to_class_grid,
)
from astar_island.fixtures import Fixture, group_by_round, load_all_fixtures
from astar_island.models import GBTHyperparams, GBTSoftClassifier
from astar_island.prediction import validate_prediction
from astar_island.scoring import competition_score
from astar_island.solver import DEFAULT_ARTIFACT, predict_grid

pytestmark = pytest.mark.benchmark

MINIMUM_MEAN_SCORE = 45.0  # Lower for fast benchmark config (80 trees vs 200)
MINIMUM_SEED_SCORE = 20.0
MINIMUM_ARTIFACT_MEAN_SCORE = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_and_predict_fixture(
    train_fixtures: list[Fixture],
    test_fixture: Fixture,
    temperature: float,
) -> np.ndarray:
    all_x = []
    all_y = []
    for f in train_fixtures:
        cg = raw_grid_to_class_grid(f.initial_grid)
        all_x.append(extract_cell_features(cg, f.initial_grid))
        all_y.append(extract_cell_targets(f.ground_truth))

    model = GBTSoftClassifier()
    model.fit(np.vstack(all_x), np.vstack(all_y))

    raw_probs = model.predict_grid(test_fixture.initial_grid)
    return calibrate_prediction(raw_probs, test_fixture.initial_grid, temperature)


def _find_fold_temperature(
    model: GBTSoftClassifier,
    val_fixtures: list[Fixture],
) -> float:
    preds = []
    gts = []
    for f in val_fixtures:
        preds.append(model.predict_grid(f.initial_grid))
        gts.append(f.ground_truth)
    return find_optimal_temperature(
        np.concatenate(preds, axis=0),
        np.concatenate(gts, axis=0),
    )


# ---------------------------------------------------------------------------
# 1. Holdout simulation — leave-one-round-out
# ---------------------------------------------------------------------------


class TestHoldoutSimulation:
    """Train on N-1 rounds, predict held-out round, score per seed.

    Mirrors the exact situation during a live active round.
    Uses a fast GBT config (fewer trees) and subsamples rounds
    to keep runtime under ~60s while still catching regressions.
    """

    @pytest.fixture(scope="class")
    def holdout_results(self) -> list[dict]:
        fixtures = load_all_fixtures()
        rounds = group_by_round(fixtures)
        round_ids = sorted(rounds.keys())

        # Subsample: test every 4th round (≈4-5 folds instead of 18)
        test_round_ids = round_ids[::4]

        results: list[dict] = []

        for held_out_id in test_round_ids:
            train_fixtures = [
                f for rid in round_ids if rid != held_out_id for f in rounds[rid]
            ]
            test_fixtures = rounds[held_out_id]

            all_x = []
            all_y = []
            for f in train_fixtures:
                cg = raw_grid_to_class_grid(f.initial_grid)
                all_x.append(extract_cell_features(cg, f.initial_grid))
                all_y.append(extract_cell_targets(f.ground_truth))

            fast_params = GBTHyperparams(max_iter=80, max_depth=4, learning_rate=0.1)
            model = GBTSoftClassifier(params=fast_params)
            model.fit(np.vstack(all_x), np.vstack(all_y))

            other_rounds = [rid for rid in round_ids if rid != held_out_id]
            held_idx = round_ids.index(held_out_id)
            val_round = min(
                other_rounds,
                key=lambda rid: abs(round_ids.index(rid) - held_idx),
            )
            temperature = _find_fold_temperature(model, rounds[val_round])

            seed_scores: list[float] = []
            for f in test_fixtures:
                raw_probs = model.predict_grid(f.initial_grid)
                calibrated = calibrate_prediction(
                    raw_probs, f.initial_grid, temperature
                )

                errors = validate_prediction(calibrated)
                assert errors == [], (
                    f"Validation failed for {held_out_id} seed {f.seed_index}: {errors}"
                )

                score = competition_score(f.ground_truth, calibrated)
                seed_scores.append(score)

            results.append(
                {
                    "round_id": held_out_id,
                    "temperature": temperature,
                    "seed_scores": seed_scores,
                    "mean_score": float(np.mean(seed_scores)),
                }
            )

        return results

    def test_all_predictions_are_valid(self, holdout_results: list[dict]) -> None:
        assert len(holdout_results) > 0

    def test_mean_score_above_threshold(self, holdout_results: list[dict]) -> None:
        all_means = [r["mean_score"] for r in holdout_results]
        overall_mean = float(np.mean(all_means))
        print(f"\nHoldout overall mean score: {overall_mean:.2f}")
        for r in holdout_results:
            print(
                f"  Round {r['round_id'][:8]}...: "
                f"mean={r['mean_score']:.2f} T={r['temperature']:.3f} "
                f"seeds={[f'{s:.1f}' for s in r['seed_scores']]}"
            )
        assert overall_mean >= MINIMUM_MEAN_SCORE, (
            f"Overall mean {overall_mean:.2f} below threshold {MINIMUM_MEAN_SCORE}"
        )

    def test_no_seed_below_floor(self, holdout_results: list[dict]) -> None:
        for r in holdout_results:
            for i, s in enumerate(r["seed_scores"]):
                assert s >= MINIMUM_SEED_SCORE, (
                    f"Round {r['round_id'][:8]} seed {i} scored {s:.2f}, "
                    f"below floor {MINIMUM_SEED_SCORE}"
                )

    def test_score_variance_is_bounded(self, holdout_results: list[dict]) -> None:
        all_means = [r["mean_score"] for r in holdout_results]
        std = float(np.std(all_means))
        print(f"\nHoldout cross-round std: {std:.2f}")
        assert std < 25.0, (
            f"Cross-round score std {std:.2f} is too high — model is unstable"
        )

    def test_temperature_is_reasonable(self, holdout_results: list[dict]) -> None:
        for r in holdout_results:
            t = r["temperature"]
            assert 0.5 <= t <= 3.0, (
                f"Round {r['round_id'][:8]} temperature {t:.3f} out of bounds"
            )


# ---------------------------------------------------------------------------
# 2. Production artifact — score the trained model on all fixtures
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DEFAULT_ARTIFACT.exists(),
    reason=f"Model artifact not found at {DEFAULT_ARTIFACT}",
)
class TestProductionArtifact:
    """Use the saved model to predict every fixture and check scores.

    This catches regressions from retraining the model or changing the
    prediction/calibration pipeline.
    """

    @pytest.fixture(scope="class")
    def artifact_results(self) -> list[dict]:
        fixtures = load_all_fixtures()
        rounds = group_by_round(fixtures)

        results: list[dict] = []
        for round_id in sorted(rounds.keys()):
            seed_scores: list[float] = []
            for f in rounds[round_id]:
                pred = predict_grid(f.initial_grid)
                errors = validate_prediction(pred)
                assert errors == [], (
                    f"Validation failed for {round_id} seed {f.seed_index}: {errors}"
                )
                score = competition_score(f.ground_truth, pred)
                seed_scores.append(score)

            results.append(
                {
                    "round_id": round_id,
                    "seed_scores": seed_scores,
                    "mean_score": float(np.mean(seed_scores)),
                }
            )
        return results

    def test_all_predictions_valid(self, artifact_results: list[dict]) -> None:
        assert len(artifact_results) > 0

    def test_mean_score_above_threshold(self, artifact_results: list[dict]) -> None:
        all_means = [r["mean_score"] for r in artifact_results]
        overall_mean = float(np.mean(all_means))
        print(f"\nArtifact overall mean score: {overall_mean:.2f}")
        for r in artifact_results:
            print(
                f"  Round {r['round_id'][:8]}...: "
                f"mean={r['mean_score']:.2f} "
                f"seeds={[f'{s:.1f}' for s in r['seed_scores']]}"
            )
        assert overall_mean >= MINIMUM_ARTIFACT_MEAN_SCORE, (
            f"Artifact mean {overall_mean:.2f} below threshold "
            f"{MINIMUM_ARTIFACT_MEAN_SCORE}"
        )

    def test_no_seed_below_floor(self, artifact_results: list[dict]) -> None:
        for r in artifact_results:
            for i, s in enumerate(r["seed_scores"]):
                assert s >= MINIMUM_SEED_SCORE, (
                    f"Round {r['round_id'][:8]} seed {i} scored {s:.2f}, "
                    f"below floor {MINIMUM_SEED_SCORE}"
                )

    def test_artifact_beats_uniform_baseline(
        self, artifact_results: list[dict]
    ) -> None:
        fixtures = load_all_fixtures()
        uniform_scores: list[float] = []
        for f in fixtures:
            w, h = len(f.initial_grid), len(f.initial_grid[0])
            k = f.ground_truth.shape[-1]
            uniform = np.full((w, h, k), 1.0 / k)
            uniform_scores.append(competition_score(f.ground_truth, uniform))

        uniform_mean = float(np.mean(uniform_scores))
        artifact_mean = float(np.mean([r["mean_score"] for r in artifact_results]))
        print(
            f"\nArtifact mean: {artifact_mean:.2f} vs Uniform baseline: {uniform_mean:.2f}"
        )
        assert artifact_mean > uniform_mean + 5.0, (
            f"Model ({artifact_mean:.2f}) does not convincingly beat "
            f"uniform baseline ({uniform_mean:.2f})"
        )

    def test_per_round_consistency(self, artifact_results: list[dict]) -> None:
        for r in artifact_results:
            scores = r["seed_scores"]
            if len(scores) < 2:
                continue
            seed_std = float(np.std(scores))
            assert seed_std < 20.0, (
                f"Round {r['round_id'][:8]} has high within-round variance: "
                f"std={seed_std:.2f}, seeds={scores}"
            )
