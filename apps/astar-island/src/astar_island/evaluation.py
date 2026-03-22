"""Leave-one-round-out cross-validation and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astar_island.calibration import calibrate_prediction, find_optimal_temperature
from astar_island.features import (
    extract_cell_features,
    extract_cell_targets,
    raw_grid_to_class_grid,
)
from astar_island.fixtures import Fixture, group_by_round
from astar_island.models import GBTHyperparams, GBTSoftClassifier
from astar_island.scoring import (
    competition_score,
    entropy_per_cell,
    kl_divergence_per_cell,
)
from astar_island.terrain import NUM_PREDICTION_CLASSES


@dataclass(frozen=True)
class FoldResult:
    round_id: str
    scores: list[float]
    mean_score: float
    temperature: float


@dataclass(frozen=True)
class LOROCVResult:
    folds: list[FoldResult]
    mean_score: float
    std_score: float
    min_score: float
    max_score: float


def _prepare_features_and_targets(
    fixtures: list[Fixture],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    all_x: list[NDArray[np.float64]] = []
    all_y: list[NDArray[np.float64]] = []
    for f in fixtures:
        class_grid = raw_grid_to_class_grid(f.initial_grid)
        all_x.append(extract_cell_features(class_grid, f.initial_grid))
        all_y.append(extract_cell_targets(f.ground_truth))
    return np.vstack(all_x), np.vstack(all_y)


def run_lorocv(
    fixtures: list[Fixture],
    params: GBTHyperparams | None = None,
    verbose: bool = True,
) -> LOROCVResult:
    """Leave-one-round-out cross-validation.

    For each round, trains on all other rounds and evaluates on the held-out round.
    Temperature is optimized per fold on the training data (last round used as
    temperature validation set).

    Args:
        fixtures: All loaded fixtures.
        params: GBT hyperparameters (uses defaults if None).
        verbose: Print per-fold results.

    Returns:
        LOROCVResult with per-fold and aggregate scores.
    """
    rounds = group_by_round(fixtures)
    round_ids = sorted(rounds.keys())
    folds: list[FoldResult] = []

    for held_out_id in round_ids:
        train_fixtures = [
            f for rid in round_ids if rid != held_out_id for f in rounds[rid]
        ]
        test_fixtures = rounds[held_out_id]

        x_train, y_train = _prepare_features_and_targets(train_fixtures)

        model = GBTSoftClassifier(params)
        model.fit(x_train, y_train)

        temp = _find_temperature_for_fold(model, rounds, held_out_id, round_ids)

        fold_scores: list[float] = []
        for f in test_fixtures:
            raw_probs = model.predict_grid(f.initial_grid)
            calibrated = calibrate_prediction(raw_probs, f.initial_grid, temp)
            score = competition_score(f.ground_truth, calibrated)
            fold_scores.append(score)

        fold = FoldResult(
            round_id=held_out_id,
            scores=fold_scores,
            mean_score=float(np.mean(fold_scores)),
            temperature=temp,
        )
        folds.append(fold)

        if verbose:
            print(
                f"  Round {held_out_id[:8]}...: "
                f"score={fold.mean_score:.2f} (T={temp:.3f}, n={len(fold_scores)})"
            )

    all_means = [f.mean_score for f in folds]
    result = LOROCVResult(
        folds=folds,
        mean_score=float(np.mean(all_means)),
        std_score=float(np.std(all_means)),
        min_score=float(np.min(all_means)),
        max_score=float(np.max(all_means)),
    )

    if verbose:
        print(
            f"\nLOROCV: mean={result.mean_score:.2f} "
            f"std={result.std_score:.2f} "
            f"range=[{result.min_score:.2f}, {result.max_score:.2f}]"
        )

    return result


def _find_temperature_for_fold(
    model: GBTSoftClassifier,
    rounds: dict[str, list[Fixture]],
    held_out_id: str,
    round_ids: list[str],
) -> float:
    """Find optimal temperature using a per-fold validation round.

    Selects the training round whose sorted position is closest to the
    held-out round's position, avoiding systematic bias from always
    picking the same validation round.
    """
    other_rounds = [rid for rid in round_ids if rid != held_out_id]
    if len(other_rounds) < 2:
        return 1.0

    held_out_idx = round_ids.index(held_out_id)
    best_val_round = min(
        other_rounds,
        key=lambda rid: abs(round_ids.index(rid) - held_out_idx),
    )
    val_fixtures = rounds[best_val_round]

    val_preds: list[NDArray[np.float64]] = []
    val_gts: list[NDArray[np.float64]] = []
    for f in val_fixtures:
        raw_probs = model.predict_grid(f.initial_grid)
        val_preds.append(raw_probs)
        val_gts.append(f.ground_truth)

    preds_concat = np.concatenate(val_preds, axis=0)
    gts_concat = np.concatenate(val_gts, axis=0)

    return find_optimal_temperature(preds_concat, gts_concat)


def per_class_error_analysis(
    predictions: NDArray[np.float64],
    ground_truth: NDArray[np.float64],
    class_grid: NDArray[np.int32],
) -> dict[int, dict[str, float]]:
    """Break down prediction error by initial terrain class.

    Returns dict of class_index -> {count, mean_kl, mean_entropy, mean_weighted_kl}.
    """
    h, w = class_grid.shape
    kl = kl_divergence_per_cell(ground_truth, predictions)
    ent = entropy_per_cell(ground_truth)

    summary: dict[int, dict[str, float]] = {}
    for k in range(NUM_PREDICTION_CLASSES):
        mask = class_grid == k
        count = int(mask.sum())
        if count == 0:
            continue
        summary[k] = {
            "count": float(count),
            "mean_kl": float(kl[mask].mean()),
            "mean_entropy": float(ent[mask].mean()),
            "mean_weighted_kl": float((ent[mask] * kl[mask]).mean()),
        }

    return summary
