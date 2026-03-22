"""Offline evaluation: leave-one-round-out CV and temperature tuning.

CLI entry points for validating the prior model and tuning calibration
temperature without touching the live API.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray

from astar_island.api import RoundData, load_all_rounds
from astar_island.config import Config
from astar_island.features import extract_features
from astar_island.posterior import apply_floors
from astar_island.prior import PriorModel, prepare_training_data
from astar_island.solver import HybridSolver
from astar_island.types import K
from astar_island.utils import kl_divergence, normalized_entropy

logger = logging.getLogger(__name__)

# class 0 (ocean/plains/empty) maps to code 11 (plains) for sampling,
# since CODE_TO_CLASS has three codes→0 and we need one canonical reverse code
_CLASS_TO_CODE: dict[int, int] = {0: 11, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def _cast_rounds_int32(
    rounds: RoundData,
) -> dict[int, list[tuple[NDArray[np.int32], NDArray[np.float64]]]]:
    """Cast RoundData grids from np.int_ to np.int32."""
    result: dict[int, list[tuple[NDArray[np.int32], NDArray[np.float64]]]] = {}
    for rnum, seeds in rounds.items():
        cast: list[tuple[NDArray[np.int32], NDArray[np.float64]]] = []
        for grid, gt in seeds:
            cast.append((np.asarray(grid, dtype=np.int32), gt))
        result[rnum] = cast
    return result


def leave_one_round_out_cv(
    rounds: RoundData | None = None,
    config: Config | None = None,
) -> dict[int, float]:
    """Leave-one-round-out cross-validation for the prior model.

    For each round R, trains on all other rounds, predicts on R's seeds,
    and computes weighted KL divergence (the competition scoring metric).

    Returns:
        Dict mapping round_number -> average weighted KL score.
    """
    if config is None:
        config = Config()
    if rounds is None:
        rounds = load_all_rounds()

    if len(rounds) < 2:  # noqa: PLR2004
        msg = f"Need at least 2 rounds for CV, got {len(rounds)}"
        raise ValueError(msg)

    round_nums = sorted(rounds.keys())
    scores: dict[int, float] = {}

    for held_out in round_nums:
        # Build training data from all rounds except held_out
        train_rounds: RoundData = {
            rnum: pairs for rnum, pairs in rounds.items() if rnum != held_out
        }
        train_i32 = _cast_rounds_int32(train_rounds)
        features, labels, round_ids, weights = prepare_training_data(
            train_i32,
            config,
        )

        model = PriorModel(config)
        model.train(features, labels, round_ids, weights)

        # Evaluate on held-out round's seeds
        seed_scores: list[float] = []
        for grid, gt in rounds[held_out]:
            grid_i32 = np.asarray(grid, dtype=np.int32)
            feats = extract_features(grid_i32, config)
            pred = model.predict_grid(feats, grid_i32)
            pred = apply_floors(pred, grid_i32, config)
            score = _weighted_kl_score(gt, pred, config.eps)
            seed_scores.append(score)

        avg_score = float(np.mean(seed_scores)) if seed_scores else 0.0
        scores[held_out] = avg_score
        logger.info(
            "Round %d: %.4f weighted KL (avg over %d seeds)",
            held_out,
            avg_score,
            len(seed_scores),
        )

    overall = float(np.mean(list(scores.values())))
    logger.info("Overall LOOCV weighted KL: %.4f", overall)
    return scores


def tune_temperature(
    temperatures: list[float] | None = None,
    rounds: RoundData | None = None,
    config: Config | None = None,
) -> tuple[float, dict[float, float]]:
    """Grid search over calibration temperatures using LOOCV.

    For each temperature, runs full LOOCV and records the average
    score.  Returns the best temperature and the full results dict.

    Args:
        temperatures: Temperatures to try. Defaults to [0.5..3.0].
        rounds: Historical round data. Loaded from cache if None.
        config: Base config (temperature overridden per trial).

    Returns:
        (best_temperature, {temperature: avg_weighted_kl})
    """
    if config is None:
        config = Config()
    if rounds is None:
        rounds = load_all_rounds()
    if temperatures is None:
        temperatures = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0,
            2.5,
            3.0,
        ]

    results: dict[float, float] = {}
    for temp in temperatures:
        logger.info("Testing temperature=%.2f", temp)
        trial_config = Config(
            prior_model=config.prior_model,
            lgb_max_depth=config.lgb_max_depth,
            lgb_num_leaves=config.lgb_num_leaves,
            lgb_min_data_in_leaf=config.lgb_min_data_in_leaf,
            lgb_n_estimators=config.lgb_n_estimators,
            lgb_learning_rate=config.lgb_learning_rate,
            temperature=temp,
            feature_radii=config.feature_radii,
            window_sizes=config.window_sizes,
            c_base=config.c_base,
            ess_min=config.ess_min,
            ess_max=config.ess_max,
            ess_confidence_weight=config.ess_confidence_weight,
            lambda_prior=config.lambda_prior,
            eps=config.eps,
            floor_impossible=config.floor_impossible,
            floor_standard=config.floor_standard,
        )
        cv_scores = leave_one_round_out_cv(rounds, trial_config)
        avg = float(np.mean(list(cv_scores.values())))
        results[temp] = avg
        logger.info(
            "Temperature=%.2f -> avg weighted KL=%.4f",
            temp,
            avg,
        )

    best_temp = min(results, key=results.get)  # type: ignore[arg-type]
    logger.info(
        "Best temperature: %.2f (score=%.4f)",
        best_temp,
        results[best_temp],
    )
    return best_temp, results


def benchmark_prior_only(
    rounds: RoundData | None = None,
    config: Config | None = None,
) -> dict[int, list[float]]:
    if config is None:
        config = Config()
    if rounds is None:
        rounds = load_all_rounds()

    if len(rounds) < 2:  # noqa: PLR2004
        msg = f"Need at least 2 rounds for benchmark, got {len(rounds)}"
        raise ValueError(msg)

    round_nums = sorted(rounds.keys())
    results: dict[int, list[float]] = {}

    for held_out in round_nums:
        train_rounds: RoundData = {
            rnum: pairs for rnum, pairs in rounds.items() if rnum != held_out
        }
        train_i32 = _cast_rounds_int32(train_rounds)
        features, labels, round_ids, weights = prepare_training_data(
            train_i32,
            config,
        )
        model = PriorModel(config)
        model.train(features, labels, round_ids, weights)

        seed_scores: list[float] = []
        for grid, gt in rounds[held_out]:
            grid_i32 = np.asarray(grid, dtype=np.int32)
            feats = extract_features(grid_i32, config)
            pred = model.predict_grid(feats, grid_i32)
            pred = apply_floors(pred, grid_i32, config)
            seed_scores.append(_weighted_kl_score(gt, pred, config.eps))

        results[held_out] = seed_scores
        avg = float(np.mean(seed_scores)) if seed_scores else 0.0
        logger.info(
            "Prior-only benchmark round %d: %.2f avg score (%d seeds)",
            held_out,
            avg,
            len(seed_scores),
        )

    overall = float(np.mean([s for scores in results.values() for s in scores]))
    logger.info("Prior-only benchmark overall average score: %.2f", overall)
    return results


def _gt_simulate_response(
    ground_truth: NDArray[np.float64],
    vx: int,
    vy: int,
    viewport_size: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    h, w = ground_truth.shape[:2]
    rows: list[list[int]] = []
    for r in range(vy, min(vy + viewport_size, h)):
        row: list[int] = []
        for c in range(vx, min(vx + viewport_size, w)):
            class_probs = ground_truth[r, c, :]
            class_idx = int(rng.choice(K, p=class_probs))
            row.append(_CLASS_TO_CODE[class_idx])
        rows.append(row)
    return {
        "grid": rows,
        "viewport": {"x": vx, "y": vy, "w": viewport_size, "h": viewport_size},
    }


def benchmark_scores(
    rounds: RoundData | None = None,
    config: Config | None = None,
    seed: int = 42,
) -> dict[int, list[float]]:
    if config is None:
        config = Config()
    if rounds is None:
        rounds = load_all_rounds()

    if len(rounds) < 2:  # noqa: PLR2004
        msg = f"Need at least 2 rounds for benchmark, got {len(rounds)}"
        raise ValueError(msg)

    round_nums = sorted(rounds.keys())
    results: dict[int, list[float]] = {}
    rng = np.random.default_rng(seed)

    for held_out in round_nums:
        train_rounds: RoundData = {
            rnum: pairs for rnum, pairs in rounds.items() if rnum != held_out
        }

        solver = HybridSolver(config)
        solver.train_offline(train_rounds)

        held_seeds = rounds[held_out]
        initial_grids = [np.asarray(grid, dtype=np.int32) for grid, _ in held_seeds]
        ground_truths = [gt for _, gt in held_seeds]

        round_id = f"benchmark-round-{held_out}"

        def make_simulate_fn(
            gts: list[NDArray[np.float64]],
            _rng: np.random.Generator,
        ):
            def _simulate(
                client: object,
                rid: str,
                seed_index: int,
                vx: int,
                vy: int,
                viewport_size: int,
            ) -> dict[str, object]:
                gt = gts[seed_index]
                return _gt_simulate_response(gt, vx, vy, viewport_size, _rng)

            return _simulate

        simulate_fn = make_simulate_fn(ground_truths, rng)
        mock_client = MagicMock()

        with (
            patch("astar_island.solver.simulate_query") as mock_sim,
            patch("astar_island.solver.save_prediction"),
        ):
            mock_sim.side_effect = simulate_fn
            predictions = solver.solve_round(mock_client, round_id, initial_grids)

        seed_scores: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            seed_scores.append(_weighted_kl_score(gt, pred, config.eps))

        results[held_out] = seed_scores
        avg = float(np.mean(seed_scores)) if seed_scores else 0.0
        logger.info(
            "Full-pipeline benchmark round %d: %.2f avg score (%d seeds)",
            held_out,
            avg,
            len(seed_scores),
        )

    overall = float(np.mean([s for scores in results.values() for s in scores]))
    logger.info("Full-pipeline benchmark overall average score: %.2f", overall)
    return results


def sensitivity_sweep(
    rounds: RoundData | None = None,
    seed: int = 42,
    held_out_rounds: list[int] | None = None,
    max_seeds: int = 2,
) -> dict[str, float]:
    """Quick sensitivity analysis of key hyperparameters.

    Tests combinations of lambda_prior and c_base on a subset of rounds
    using the full pipeline benchmark.  Returns {config_label: avg_score}.

    Optimised for speed:
      - Trains the LightGBM model ONCE per held-out round (the sweep
        parameters only affect posterior merging, not the prior).
      - Evaluates only *max_seeds* seeds per round (default 2).
      - Uses 3 held-out rounds by default.
    """
    from dataclasses import replace

    from astar_island.calibration import (
        RoundCalibrator,
        compute_archetype_frequencies,
    )

    if rounds is None:
        rounds = load_all_rounds()

    if held_out_rounds is None:
        all_rounds = sorted(rounds.keys())
        step = max(1, len(all_rounds) // 3)
        held_out_rounds = all_rounds[::step][:3]
        if all_rounds[-1] not in held_out_rounds:
            held_out_rounds[-1] = all_rounds[-1]

    logger.info(
        "Sensitivity sweep on %d held-out rounds (max %d seeds each): %s",
        len(held_out_rounds),
        max_seeds,
        held_out_rounds,
    )

    base = Config()
    configs: dict[str, Config] = {}

    for lp in [0.5, 0.6, 0.7, 0.8, 0.9]:
        configs[f"lp={lp}"] = replace(base, lambda_prior=lp)

    for cb in [1.0, 2.0, 3.0, 5.0]:
        configs[f"cb={cb}"] = replace(base, c_base=cb)

    configs["lp=0.6,cb=3.0"] = replace(base, lambda_prior=0.6, c_base=3.0)
    configs["lp=0.8,cb=1.5"] = replace(
        base,
        lambda_prior=0.8,
        c_base=1.5,
        ess_max=3.0,
    )

    # Sweep params (lambda_prior, c_base, ess_*) don't affect LightGBM
    # training or feature extraction — train once per fold, replay with each config.
    cached: dict[int, dict] = {}

    for held_out in held_out_rounds:
        if held_out not in rounds:
            continue

        train_rounds: RoundData = {
            rnum: pairs for rnum, pairs in rounds.items() if rnum != held_out
        }

        train_i32 = _cast_rounds_int32(train_rounds)
        features, labels, round_ids, weights = prepare_training_data(
            train_i32,
            base,
        )
        model = PriorModel(base)
        model.train(features, labels, round_ids, weights)

        theta_hist, hist_round_ids = compute_archetype_frequencies(rounds)

        held_seeds = rounds[held_out][:max_seeds]
        initial_grids = [np.asarray(g, dtype=np.int32) for g, _ in held_seeds]
        ground_truths = [gt for _, gt in held_seeds]

        p_priors = []
        for grid_i32 in initial_grids:
            feats = extract_features(grid_i32, base)
            p_prior = model.predict_grid(feats, grid_i32)
            p_priors.append(p_prior)

        cached[held_out] = {
            "model": model,
            "theta_hist": theta_hist,
            "hist_round_ids": hist_round_ids,
            "initial_grids": initial_grids,
            "ground_truths": ground_truths,
            "p_priors": p_priors,
        }
        logger.info("Cached model + features for held-out round %d", held_out)

    results: dict[str, float] = {}

    for label, cfg in configs.items():
        logger.info("Testing config: %s", label)
        round_scores: list[float] = []

        for held_out, cache in cached.items():
            initial_grids = cache["initial_grids"]
            ground_truths = cache["ground_truths"]

            solver = HybridSolver(cfg)
            solver.prior = cache["model"]
            solver._theta_hist = cache["theta_hist"]
            solver._round_ids = cache["hist_round_ids"]
            solver._trained = True

            round_id = f"sweep-{held_out}"
            rng = np.random.default_rng(seed)

            def make_simulate_fn(
                gts: list[NDArray[np.float64]],
                _rng: np.random.Generator,
            ):
                def _simulate(
                    client: object,
                    rid: str,
                    seed_index: int,
                    vx: int,
                    vy: int,
                    viewport_size: int,
                ) -> dict[str, object]:
                    gt = gts[seed_index]
                    return _gt_simulate_response(gt, vx, vy, viewport_size, _rng)

                return _simulate

            simulate_fn = make_simulate_fn(ground_truths, rng)
            mock_client = MagicMock()

            with (
                patch("astar_island.solver.simulate_query") as mock_sim,
                patch("astar_island.solver.save_prediction"),
            ):
                mock_sim.side_effect = simulate_fn
                predictions = solver.solve_round(mock_client, round_id, initial_grids)

            for pred, gt in zip(predictions, ground_truths):
                round_scores.append(_weighted_kl_score(gt, pred, cfg.eps))

        avg = float(np.mean(round_scores)) if round_scores else 0.0
        results[label] = avg
        logger.info("Config %s: avg score = %.2f", label, avg)

    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    logger.info("=== Sensitivity Sweep Results ===")
    for rank, (label, score) in enumerate(ranked, 1):
        marker = " <-- BEST" if rank == 1 else ""
        logger.info("#%d  %-25s  %.2f%s", rank, label, score, marker)

    return results


def _weighted_kl_score(
    ground_truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
    eps: float,
) -> float:
    """Compute entropy-weighted KL divergence (competition metric).

    score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
    """
    gt_flat = ground_truth.reshape(-1, K)
    pred_flat = prediction.reshape(-1, K)

    # Per-cell normalized entropy of ground truth
    h_norm = normalized_entropy(ground_truth, eps)
    h_flat = h_norm.ravel()

    # Per-cell KL divergence
    kl_vals = kl_divergence(gt_flat, pred_flat, eps)

    # Weighted average
    total_weight = float(np.sum(h_flat))
    if total_weight < eps:
        return 0.0

    weighted_kl = float(np.sum(h_flat * kl_vals) / total_weight)
    score = max(
        0.0,
        min(100.0, 100.0 * float(np.exp(-3.0 * weighted_kl))),
    )
    return score


def main() -> None:
    """CLI entry point for offline evaluation."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Offline evaluation tools",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("cv", help="Run leave-one-round-out CV")
    sub.add_parser("sweep", help="Quick hyperparameter sensitivity sweep")
    temp_parser = sub.add_parser(
        "tune-temperature",
        help="Grid search temperatures",
    )
    temp_parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=None,
        help=("Temperatures to try (default: 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0)"),
    )

    args = parser.parse_args()

    if args.command == "cv":
        leave_one_round_out_cv()
    elif args.command == "sweep":
        sensitivity_sweep()
    elif args.command == "tune-temperature":
        temps: list[float] | None = args.temps
        tune_temperature(temps)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
