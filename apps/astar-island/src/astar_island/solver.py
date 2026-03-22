"""CLI entry point: download / evaluate / predict / submit subcommands."""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from numpy.typing import NDArray

from astar_island.api import (
    download_all,
    fetch_active_round,
    load_all_rounds,
    query_all_seeds,
    submit_prediction,
)
from astar_island.evaluate import benchmark_live_pipeline, evaluate_lorocv
from astar_island.model import PerCellGBDT, prepare_training_data, tta_predict
from astar_island.prob import apply_floors, bayesian_blend

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Typed argument namespaces
# ---------------------------------------------------------------------------


class _MainArgs(argparse.Namespace):
    command: str | None
    verbose: bool


class _DownloadArgs(_MainArgs):
    force: bool


class _EvaluateArgs(_MainArgs):
    pass


class _PredictArgs(_MainArgs):
    round_id: int
    seed_index: int


class _SubmitArgs(_MainArgs):
    dry_run: bool


class _BenchmarkArgs(_MainArgs):
    budget: int
    alpha: float
    trials: int
    folds: int


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_download(args: _DownloadArgs) -> None:
    """Download and cache all historical round data."""
    verbose: bool = args.verbose
    force: bool = args.force
    _setup_logging(verbose=verbose)
    logger.info("Downloading historical round data...")
    download_all(force=force)
    rounds = load_all_rounds()
    logger.info("Done. %d rounds cached.", len(rounds))


def cmd_evaluate(args: _EvaluateArgs) -> None:
    """Run LOROCV evaluation."""
    verbose: bool = args.verbose
    _setup_logging(verbose=verbose)
    rounds = load_all_rounds()
    if not rounds:
        logger.error("No rounds found. Run 'download' first.")
        sys.exit(1)
    logger.info("Running LOROCV on %d rounds...", len(rounds))
    evaluate_lorocv(rounds)


def cmd_benchmark(args: _BenchmarkArgs) -> None:
    """Benchmark full live pipeline: GBDT + TTA + simulated queries + blend."""
    verbose: bool = args.verbose
    _setup_logging(verbose=verbose)
    rounds = load_all_rounds()
    if not rounds:
        logger.error("No rounds found. Run 'download' first.")
        sys.exit(1)
    budget: int = args.budget
    alpha: float = args.alpha
    trials: int = args.trials
    folds: int = args.folds
    logger.info(
        "Benchmarking on %d rounds (budget=%d, alpha=%.1f, trials=%d, folds=%s)...",
        len(rounds),
        budget,
        alpha,
        trials,
        folds if folds > 0 else "all",
    )
    benchmark_live_pipeline(
        rounds, total_budget=budget, alpha=alpha, n_trials=trials, max_folds=folds
    )


def cmd_predict(args: _PredictArgs) -> None:
    """Train on all data and predict for a specific round/seed.

    Saves the prediction as a .npy file.
    """
    verbose: bool = args.verbose
    _setup_logging(verbose=verbose)
    rounds = load_all_rounds()
    if not rounds:
        logger.error("No rounds found. Run 'download' first.")
        sys.exit(1)

    round_id: int = args.round_id
    seed_index: int = args.seed_index

    # Find the initial grid for this round/seed
    if round_id not in rounds:
        logger.error("Round %d not found in cache.", round_id)
        sys.exit(1)

    seeds = rounds[round_id]
    if seed_index >= len(seeds):
        logger.error(
            "Seed %d not found for round %d (have %d seeds).",
            seed_index,
            round_id,
            len(seeds),
        )
        sys.exit(1)

    initial_grid, _ = seeds[seed_index]

    # Train GBDT on all rounds (including the target round for production use)
    logger.info("Preparing training data...")
    x_train, y_train, _ = prepare_training_data(rounds, augment=True)

    logger.info("Training GBDT model...")
    gbdt = PerCellGBDT(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
    )
    gbdt.fit(x_train, y_train)

    logger.info("Generating prediction...")
    prediction = gbdt.predict_grid(initial_grid)

    out_path = f"prediction_r{round_id}_s{seed_index}.npy"
    np.save(out_path, prediction)
    logger.info("Prediction saved to %s (shape: %s)", out_path, prediction.shape)


def cmd_submit(args: _SubmitArgs) -> None:
    """Train GBDT, predict, query, blend with observations, and submit."""
    verbose: bool = args.verbose
    dry_run: bool = args.dry_run
    _setup_logging(verbose=verbose)

    logger.info("Step 1/6: Downloading latest historical data...")
    download_all(force=False)
    rounds = load_all_rounds()
    if not rounds:
        logger.error("No rounds found after download. Cannot train.")
        sys.exit(1)
    logger.info("Loaded %d historical rounds.", len(rounds))

    logger.info("Step 2/6: Training GBDT on historical data...")
    x_train, y_train, _ = prepare_training_data(rounds, augment=True)
    logger.info("Training data: X=%s, Y=%s", x_train.shape, y_train.shape)
    gbdt = PerCellGBDT(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
    )
    gbdt.fit(x_train, y_train)

    logger.info("Step 3/6: Fetching active round grids...")
    round_uuid, round_number, grids = fetch_active_round()
    logger.info(
        "Active round %d (uuid=%s), %d seeds",
        round_number,
        round_uuid,
        len(grids),
    )
    if len(grids) != 5:
        logger.error("Expected 5 seeds but got %d. Aborting submission.", len(grids))
        sys.exit(1)

    logger.info("Step 4/6: Generating GBDT predictions (TTA, unfloored)...")
    raw_predictions: list[NDArray[np.float64]] = []
    for seed_idx, initial_grid in enumerate(grids):
        pred = tta_predict(initial_grid, gbdt.predict_grid_raw)
        raw_predictions.append(pred)
        logger.info("  Seed %d: predicted (shape=%s)", seed_idx, pred.shape)

    logger.info("Step 5/6: Running simulation queries...")
    query_results = query_all_seeds(
        round_uuid, grids=grids, predictions=raw_predictions
    )

    logger.info("Step 6/6: Blending, flooring, and submitting...")
    scores: list[tuple[int, float | None]] = []
    for seed_idx, initial_grid in enumerate(grids):
        accum, counts = query_results[seed_idx]
        prediction = bayesian_blend(raw_predictions[seed_idx], accum, counts, alpha=5.0)
        prediction = apply_floors(prediction, initial_grid)

        observed_cells = int(np.sum(counts > 0))
        logger.info(
            "  Seed %d: blended (%d observed cells), floored",
            seed_idx,
            observed_cells,
        )

        if dry_run:
            out_path = f"prediction_r{round_number}_s{seed_idx}.npy"
            np.save(out_path, prediction)
            logger.info(
                "  Seed %d: DRY RUN — saved to %s (shape: %s)",
                seed_idx,
                out_path,
                prediction.shape,
            )
            scores.append((seed_idx, None))
        else:
            score = submit_prediction(round_uuid, seed_idx, prediction)
            scores.append((seed_idx, score))
            logger.info("  Seed %d: submitted (score=%s)", seed_idx, score)

    logger.info("--- Submission Summary ---")
    logger.info("Round: %d (uuid=%s)", round_number, round_uuid)
    for seed_idx, score in scores:
        status = f"score={score}" if score is not None else "submitted (no score)"
        if dry_run:
            status = "DRY RUN (not submitted)"
        logger.info("  Seed %d: %s", seed_idx, status)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="astar-island",
        description="A* Island supervised prediction pipeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # download
    dl = subparsers.add_parser("download", help="Download historical round data")
    dl.add_argument("--force", action="store_true", help="Re-download even if cached")
    dl.set_defaults(func=cmd_download, _ns_cls=_DownloadArgs)

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Run LOROCV evaluation")
    ev.set_defaults(func=cmd_evaluate, _ns_cls=_EvaluateArgs)

    # predict
    pr = subparsers.add_parser("predict", help="Generate prediction for a round/seed")
    pr.add_argument("round_id", type=int, help="Round ID")
    pr.add_argument("seed_index", type=int, help="Seed index (0-4)")
    pr.set_defaults(func=cmd_predict, _ns_cls=_PredictArgs)

    # submit
    sb = subparsers.add_parser(
        "submit",
        help="Train, predict, and submit to active round",
    )
    sb.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate predictions but don't submit (saves .npy files instead)",
    )
    sb.set_defaults(func=cmd_submit, _ns_cls=_SubmitArgs)

    # benchmark
    bm = subparsers.add_parser(
        "benchmark",
        help="Benchmark full pipeline (GBDT + queries + blend) via LOROCV",
    )
    bm.add_argument(
        "--budget",
        type=int,
        default=50,
        help="Simulated query budget per round (default: 50)",
    )
    bm.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="Bayesian blend prior strength (default: 5.0)",
    )
    bm.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Independent stochastic trials to average (default: 5)",
    )
    bm.add_argument(
        "--folds",
        type=int,
        default=0,
        help="Max CV folds to run, 0=all (default: 0). Evenly spaced.",
    )
    bm.set_defaults(func=cmd_benchmark, _ns_cls=_BenchmarkArgs)

    pre_ns = _MainArgs()
    parser.parse_args(namespace=pre_ns)
    command: str | None = pre_ns.command

    if command is None:
        parser.print_help()
        sys.exit(0)

    if command == "download":
        dl_ns = _DownloadArgs()
        parser.parse_args(namespace=dl_ns)
        cmd_download(dl_ns)
    elif command == "evaluate":
        ev_ns = _EvaluateArgs()
        parser.parse_args(namespace=ev_ns)
        cmd_evaluate(ev_ns)
    elif command == "predict":
        pr_ns = _PredictArgs()
        parser.parse_args(namespace=pr_ns)
        cmd_predict(pr_ns)
    elif command == "submit":
        sb_ns = _SubmitArgs()
        parser.parse_args(namespace=sb_ns)
        cmd_submit(sb_ns)
    elif command == "benchmark":
        bm_ns = _BenchmarkArgs()
        parser.parse_args(namespace=bm_ns)
        cmd_benchmark(bm_ns)


if __name__ == "__main__":
    main()
