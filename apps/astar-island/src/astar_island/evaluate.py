"""Leave-one-round-out cross-validation (LOROCV) harness.

Trains both ShrunkenArchetype and PerCellGBDT on all rounds except one,
predicts the held-out round, and scores. Reports per-round and aggregate
results with a paired comparison.

``benchmark_live_pipeline`` extends this to simulate the full submission
pipeline (TTA predict → entropy-ranked viewport queries → Bayesian blend
→ floor → score) using ground truth as a stand-in for the live API.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from numpy.typing import NDArray

from astar_island.api import RoundData, _score_viewport, plan_viewports
from astar_island.model import (
    PerCellGBDT,
    ShrunkenArchetype,
    prepare_training_data,
    tta_predict,
)
from astar_island.prob import (
    NUM_CLASSES,
    apply_floors,
    bayesian_blend,
    entropy,
    score_prediction,
)

logger = logging.getLogger(__name__)


def evaluate_lorocv(rounds: RoundData) -> dict[str, object]:
    """Run leave-one-round-out cross-validation on both models.

    For each round:
        1. Train both models on all other rounds (with D4 augmentation)
        2. Predict the held-out round's grids
        3. Score predictions against ground truth

    Returns:
        Dict with per-round scores, means, and paired comparison.
    """
    round_ids = sorted(rounds.keys())
    n_rounds = len(round_ids)

    if n_rounds < 2:
        logger.error("Need at least 2 rounds for LOROCV, got %d", n_rounds)
        return {"error": "insufficient rounds"}

    baseline_scores: list[float] = []
    gbdt_scores: list[float] = []
    round_results: list[dict[str, object]] = []

    total_t0 = time.time()

    for fold_idx, held_out_id in enumerate(round_ids):
        fold_t0 = time.time()
        logger.info(
            "=== LOROCV fold %d/%d: holding out round %d ===",
            fold_idx + 1,
            n_rounds,
            held_out_id,
        )

        # Prepare training data (all rounds except held-out)
        x_train, y_train, _ = prepare_training_data(
            rounds, exclude_round=held_out_id, augment=True
        )

        # Train baseline
        baseline = ShrunkenArchetype(k=10, alpha=0.7)
        baseline.fit(x_train, y_train)

        # Train GBDT
        gbdt = PerCellGBDT(
            max_iter=200,
            max_depth=5,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
        )
        gbdt.fit(x_train, y_train)

        # Score on held-out round (all seeds)
        held_out_seeds = rounds[held_out_id]
        b_seed_scores: list[float] = []
        g_seed_scores: list[float] = []

        for seed_idx, (ig, gt) in enumerate(held_out_seeds):
            b_pred = baseline.predict_grid(ig)
            g_pred = gbdt.predict_grid(ig)

            b_score = score_prediction(gt, b_pred)
            g_score = score_prediction(gt, g_pred)

            b_seed_scores.append(b_score)
            g_seed_scores.append(g_score)

            logger.info(
                "  Round %d seed %d: baseline=%.2f, GBDT=%.2f",
                held_out_id,
                seed_idx,
                b_score,
                g_score,
            )

        b_arr_fold: NDArray[np.float64] = np.fromiter(b_seed_scores, dtype=np.float64)
        b_fold_sum: np.float64 = np.sum(b_arr_fold, dtype=np.float64)
        b_mean: float = float(b_fold_sum) / float(len(b_seed_scores))
        g_arr_fold: NDArray[np.float64] = np.fromiter(g_seed_scores, dtype=np.float64)
        g_fold_sum: np.float64 = np.sum(g_arr_fold, dtype=np.float64)
        g_mean: float = float(g_fold_sum) / float(len(g_seed_scores))
        baseline_scores.append(b_mean)
        gbdt_scores.append(g_mean)

        fold_time = time.time() - fold_t0
        round_results.append(
            {
                "round_id": held_out_id,
                "baseline_mean": b_mean,
                "gbdt_mean": g_mean,
                "baseline_seeds": b_seed_scores,
                "gbdt_seeds": g_seed_scores,
                "time_seconds": fold_time,
            }
        )
        logger.info(
            "  Round %d mean: baseline=%.2f, GBDT=%.2f (%.1fs)",
            held_out_id,
            b_mean,
            g_mean,
            fold_time,
        )

    total_time = time.time() - total_t0

    # Aggregate
    b_arr: NDArray[np.float64] = np.fromiter(baseline_scores, dtype=np.float64)
    g_arr: NDArray[np.float64] = np.fromiter(gbdt_scores, dtype=np.float64)
    diff: NDArray[np.float64] = np.asarray(g_arr - b_arr, dtype=np.float64)

    # Paired t-test
    n = len(diff)
    n_f: float = float(n)
    diff_sum: np.float64 = np.sum(diff, dtype=np.float64)
    diff_mean: float = float(diff_sum) / n_f
    diff_arr: NDArray[np.float64] = diff - diff_mean
    diff_sq: NDArray[np.float64] = diff_arr * diff_arr
    diff_sq_sum: np.float64 = np.sum(diff_sq, dtype=np.float64)
    diff_std: float = float(np.sqrt(diff_sq_sum / float(n - 1))) if n > 1 else 0.0
    n_sqrt: float = float(np.sqrt(float(n)))
    t_stat: float = diff_mean / (diff_std / n_sqrt) if diff_std > 0 else float("inf")

    b_total: np.float64 = np.sum(b_arr, dtype=np.float64)
    b_mean_agg: float = float(b_total) / n_f
    b_diff: NDArray[np.float64] = b_arr - b_mean_agg
    b_sq_sum: np.float64 = np.sum(b_diff * b_diff, dtype=np.float64)
    b_std_agg: float = float(np.sqrt(b_sq_sum / float(n - 1))) if n > 1 else 0.0

    g_total: np.float64 = np.sum(g_arr, dtype=np.float64)
    g_mean_agg: float = float(g_total) / n_f
    g_diff: NDArray[np.float64] = g_arr - g_mean_agg
    g_sq_sum: np.float64 = np.sum(g_diff * g_diff, dtype=np.float64)
    g_std_agg: float = float(np.sqrt(g_sq_sum / float(n - 1))) if n > 1 else 0.0

    diff_pos_raw: object = diff > 0
    assert isinstance(diff_pos_raw, np.ndarray)
    diff_pos: NDArray[np.bool_] = np.asarray(diff_pos_raw, dtype=np.bool_)
    gbdt_wins: int = int(np.count_nonzero(diff_pos))

    result: dict[str, object] = {
        "rounds": round_results,
        "baseline_mean": b_mean_agg,
        "baseline_std": b_std_agg,
        "gbdt_mean": g_mean_agg,
        "gbdt_std": g_std_agg,
        "gbdt_wins": gbdt_wins,
        "total_rounds": n,
        "mean_improvement": diff_mean,
        "paired_t_stat": t_stat,
        "total_time_seconds": total_time,
    }

    # Print summary table
    _print_summary(round_results, result)

    return result


def _print_summary(
    round_results: list[dict[str, object]],
    aggregate: dict[str, object],
) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 70)
    print("LOROCV Results")
    print("=" * 70)
    print(f"{'Round':>8} {'Baseline':>10} {'GBDT':>10} {'Diff':>10} {'Winner':>10}")
    print("-" * 70)

    for r in round_results:
        rid = r["round_id"]
        b = r["baseline_mean"]
        g = r["gbdt_mean"]
        assert isinstance(rid, int)
        assert isinstance(b, float)
        assert isinstance(g, float)
        d = g - b
        winner = "GBDT" if d > 0 else "Baseline" if d < 0 else "Tie"
        print(f"{rid:>8} {b:>10.2f} {g:>10.2f} {d:>+10.2f} {winner:>10}")

    print("-" * 70)
    b_mean = aggregate["baseline_mean"]
    g_mean = aggregate["gbdt_mean"]
    improvement = aggregate["mean_improvement"]
    wins = aggregate["gbdt_wins"]
    total = aggregate["total_rounds"]
    t_stat = aggregate["paired_t_stat"]
    total_time = aggregate["total_time_seconds"]

    assert isinstance(b_mean, float)
    assert isinstance(g_mean, float)
    assert isinstance(improvement, float)
    assert isinstance(wins, int)
    assert isinstance(total, int)
    assert isinstance(t_stat, float)
    assert isinstance(total_time, float)

    print(f"{'Mean':>8} {b_mean:>10.2f} {g_mean:>10.2f} {improvement:>+10.2f}")
    print(f"\nGBDT wins: {wins}/{total} rounds")
    print(f"Paired t-statistic: {t_stat:.3f}")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Simulated query benchmark
# ---------------------------------------------------------------------------

VP_SIZE = 15


def _sample_viewport_from_gt(
    ground_truth: NDArray[np.float64],
    vx: int,
    vy: int,
    vp_size: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Sample a single simulation outcome for a viewport from the GT distribution.

    For each cell in the viewport, draw one class from the multinomial
    defined by ground_truth[y, x, :].  Returns a (vp_h, vp_w) grid of
    terrain codes (class indices 0-5, not API codes).

    Fully vectorized — uses cumulative-sum inverse-CDF sampling.
    """
    h, w = ground_truth.shape[:2]
    ey = min(vy + vp_size, int(h))
    ex = min(vx + vp_size, int(w))
    vp_h: int = ey - vy
    vp_w: int = ex - vx

    gt_patch: NDArray[np.float64] = ground_truth[vy:ey, vx:ex, :]
    flat_probs: NDArray[np.float64] = gt_patch.reshape(-1, NUM_CLASSES)

    # Normalise rows (zero-sum rows stay zero → will sample class 0)
    row_sums: NDArray[np.float64] = flat_probs.sum(axis=1, keepdims=True)
    safe_sums: NDArray[np.float64] = np.where(
        row_sums < 1e-12, np.float64(1.0), row_sums
    )
    normed: NDArray[np.float64] = flat_probs / safe_sums

    # Inverse-CDF: draw uniform, find bucket via cumsum + searchsorted
    cdf: NDArray[np.float64] = np.cumsum(normed, axis=1)
    n_cells: int = int(flat_probs.shape[0])
    u: NDArray[np.float64] = rng.random(n_cells).astype(np.float64)
    indices: list[int] = []
    for i in range(n_cells):
        row_cdf: NDArray[np.float64] = np.asarray(
            cdf[i : i + 1].ravel(), dtype=np.float64
        )
        val: float = float(u.item(i))
        indices.append(int(np.searchsorted(row_cdf, val)))
    sampled: NDArray[np.int_] = np.array(indices, dtype=np.int_)
    # Clamp to valid class range (floating-point edge case)
    np.clip(sampled, 0, NUM_CLASSES - 1, out=sampled)

    return sampled.reshape(vp_h, vp_w)


def _accumulate_class_viewport(
    vp_classes: NDArray[np.int_],
    vx: int,
    vy: int,
    accum: NDArray[np.float64],
    counts: NDArray[np.int_],
    grid_h: int,
    grid_w: int,
) -> None:
    """Accumulate one-hot observations from sampled class indices."""
    vp_h: int = int(vp_classes.shape[0])
    vp_w: int = int(vp_classes.shape[1])
    ey: int = min(vy + vp_h, grid_h)
    ex: int = min(vx + vp_w, grid_w)
    actual_h: int = ey - vy
    actual_w: int = ex - vx

    patch: NDArray[np.int_] = vp_classes[:actual_h, :actual_w]
    onehot: NDArray[np.float64] = np.eye(NUM_CLASSES, dtype=np.float64)[patch.ravel()]
    onehot_3d: NDArray[np.float64] = onehot.reshape(actual_h, actual_w, NUM_CLASSES)

    accum[vy:ey, vx:ex, :] += onehot_3d
    counts[vy:ey, vx:ex] += np.int_(1)


def simulate_queries_from_gt(
    ground_truths: list[NDArray[np.float64]],
    predictions: list[NDArray[np.float64]],
    total_budget: int = 50,
    rng_seed: int = 0,
) -> list[tuple[NDArray[np.float64], NDArray[np.int_]]]:
    """Simulate the full query pipeline using ground truth as the oracle.

    Mirrors ``query_all_seeds``: rank viewports by prediction entropy,
    select top ``total_budget``, and for each selected viewport sample
    one outcome from the ground truth distribution.

    Args:
        ground_truths: Per-seed (H, W, 6) ground truth tensors.
        predictions: Per-seed (H, W, 6) unfloored GBDT predictions.
        total_budget: Number of simulated queries to run.
        rng_seed: Seed for the sampling RNG (reproducibility).

    Returns:
        Per-seed (accum, counts) tuples identical in format to
        ``query_all_seeds``.
    """
    rng = np.random.default_rng(rng_seed)
    num_seeds = len(ground_truths)
    viewports_per_seed = plan_viewports()

    scored: list[tuple[float, int, int]] = []
    for seed_idx in range(num_seeds):
        pred_ent = entropy(predictions[seed_idx])
        for vp_idx, (vx, vy) in enumerate(viewports_per_seed):
            score = _score_viewport(vx, vy, pred_ent)
            scored.append((score, seed_idx, vp_idx))

    scored.sort(key=lambda t: t[0], reverse=True)
    selected = scored[:total_budget]

    seed_viewports: dict[int, list[tuple[int, int]]] = {i: [] for i in range(num_seeds)}
    for _, seed_idx, vp_idx in selected:
        seed_viewports[seed_idx].append(viewports_per_seed[vp_idx])

    results: list[tuple[NDArray[np.float64], NDArray[np.int_]]] = []
    for seed_idx in range(num_seeds):
        gt = ground_truths[seed_idx]
        grid_h, grid_w = gt.shape[:2]
        accum = np.zeros((grid_h, grid_w, NUM_CLASSES), dtype=np.float64)
        counts = np.zeros((grid_h, grid_w), dtype=np.int_)

        vps = seed_viewports[seed_idx]
        for vx, vy in vps:
            vp_classes = _sample_viewport_from_gt(gt, vx, vy, VP_SIZE, rng)
            _accumulate_class_viewport(
                vp_classes, vx, vy, accum, counts, grid_h, grid_w
            )

        observed = int(np.sum(counts > 0))
        logger.debug(
            "Seed %d: %d simulated queries, %d cells observed",
            seed_idx,
            len(vps),
            observed,
        )
        results.append((accum, counts))

    return results


def benchmark_live_pipeline(
    rounds: RoundData,
    total_budget: int = 50,
    alpha: float = 5.0,
    n_trials: int = 5,
    max_folds: int = 0,
) -> dict[str, object]:
    """Benchmark the full submission pipeline via LOROCV.

    For each held-out round, runs ``n_trials`` independent stochastic
    simulations of the query pipeline and averages the blended scores.
    Compares GBDT-only (no queries) vs GBDT+queries (blended).

    Args:
        rounds: Historical round data.
        total_budget: Simulated query budget per trial.
        alpha: Bayesian blend strength.
        n_trials: Independent query simulations to average over.
        max_folds: Maximum folds to run (0 = all rounds). Evenly spaced.

    Returns:
        Dict with per-round scores, aggregate means, and improvement
        statistics.
    """
    round_ids = sorted(rounds.keys())
    n_rounds = len(round_ids)

    if n_rounds < 2:
        logger.error("Need at least 2 rounds for benchmark, got %d", n_rounds)
        return {"error": "insufficient rounds"}

    if 0 < max_folds < n_rounds:
        step: float = n_rounds / max_folds
        selected_indices: list[int] = [int(i * step) for i in range(max_folds)]
        round_ids = [round_ids[i] for i in selected_indices]
        logger.info("Using %d/%d folds (rounds: %s)", max_folds, n_rounds, round_ids)

    if n_rounds < 2:
        logger.error("Need at least 2 rounds for benchmark, got %d", n_rounds)
        return {"error": "insufficient rounds"}

    gbdt_only_scores: list[float] = []
    blended_scores: list[float] = []
    round_results: list[dict[str, object]] = []

    total_t0 = time.time()

    for fold_idx, held_out_id in enumerate(round_ids):
        fold_t0 = time.time()
        logger.info(
            "=== Benchmark fold %d/%d: holding out round %d ===",
            fold_idx + 1,
            n_rounds,
            held_out_id,
        )

        x_train, y_train, _ = prepare_training_data(
            rounds, exclude_round=held_out_id, augment=True
        )

        gbdt = PerCellGBDT(
            max_iter=200,
            max_depth=5,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
        )
        gbdt.fit(x_train, y_train)

        held_out_seeds = rounds[held_out_id]
        go_seed_scores: list[float] = []
        bl_seed_scores: list[float] = []

        initial_grids: list[NDArray[np.int_]] = []
        ground_truths: list[NDArray[np.float64]] = []
        raw_predictions: list[NDArray[np.float64]] = []

        for _seed_idx, (ig, gt) in enumerate(held_out_seeds):
            initial_grids.append(ig)
            ground_truths.append(gt)
            raw_pred = tta_predict(ig, gbdt.predict_grid_raw)
            raw_predictions.append(raw_pred)

            floored = apply_floors(raw_pred, ig)
            go_score = score_prediction(gt, floored)
            go_seed_scores.append(go_score)

        trial_seed_totals: list[list[float]] = [[] for _ in range(len(held_out_seeds))]

        for trial in range(n_trials):
            query_results = simulate_queries_from_gt(
                ground_truths,
                raw_predictions,
                total_budget=total_budget,
                rng_seed=fold_idx * 1000 + trial,
            )
            for seed_idx, (ig, gt) in enumerate(held_out_seeds):
                accum, counts = query_results[seed_idx]
                blended = bayesian_blend(
                    raw_predictions[seed_idx], accum, counts, alpha=alpha
                )
                blended = apply_floors(blended, ig)
                bl_score = score_prediction(gt, blended)
                trial_seed_totals[seed_idx].append(bl_score)

        for seed_idx, (_ig, _gt) in enumerate(held_out_seeds):
            scores_list: list[float] = trial_seed_totals[seed_idx]
            bl_mean_seed: float = sum(scores_list) / max(len(scores_list), 1)
            bl_seed_scores.append(bl_mean_seed)

            logger.info(
                "  Round %d seed %d: GBDT-only=%.2f, blended=%.2f (avg %d trials)",
                held_out_id,
                seed_idx,
                go_seed_scores[seed_idx],
                bl_mean_seed,
                n_trials,
            )

        go_arr: NDArray[np.float64] = np.fromiter(go_seed_scores, dtype=np.float64)
        go_mean: float = float(np.sum(go_arr, dtype=np.float64)) / float(
            len(go_seed_scores)
        )
        bl_arr: NDArray[np.float64] = np.fromiter(bl_seed_scores, dtype=np.float64)
        bl_mean: float = float(np.sum(bl_arr, dtype=np.float64)) / float(
            len(bl_seed_scores)
        )

        gbdt_only_scores.append(go_mean)
        blended_scores.append(bl_mean)

        fold_time = time.time() - fold_t0
        round_results.append(
            {
                "round_id": held_out_id,
                "gbdt_only_mean": go_mean,
                "blended_mean": bl_mean,
                "gbdt_only_seeds": go_seed_scores,
                "blended_seeds": bl_seed_scores,
                "time_seconds": fold_time,
            }
        )
        logger.info(
            "  Round %d mean: GBDT-only=%.2f, blended=%.2f (%.1fs)",
            held_out_id,
            go_mean,
            bl_mean,
            fold_time,
        )

    total_time = time.time() - total_t0

    go_all: NDArray[np.float64] = np.fromiter(gbdt_only_scores, dtype=np.float64)
    bl_all: NDArray[np.float64] = np.fromiter(blended_scores, dtype=np.float64)
    diff: NDArray[np.float64] = np.asarray(bl_all - go_all, dtype=np.float64)

    n = len(diff)
    n_f: float = float(n)
    diff_sum: np.float64 = np.sum(diff, dtype=np.float64)
    diff_mean: float = float(diff_sum) / n_f
    diff_arr: NDArray[np.float64] = diff - diff_mean
    diff_sq: NDArray[np.float64] = diff_arr * diff_arr
    diff_sq_sum: np.float64 = np.sum(diff_sq, dtype=np.float64)
    diff_std: float = float(np.sqrt(diff_sq_sum / float(n - 1))) if n > 1 else 0.0
    n_sqrt: float = float(np.sqrt(float(n)))
    t_stat: float = diff_mean / (diff_std / n_sqrt) if diff_std > 0 else float("inf")

    go_total: np.float64 = np.sum(go_all, dtype=np.float64)
    go_mean_agg: float = float(go_total) / n_f
    bl_total: np.float64 = np.sum(bl_all, dtype=np.float64)
    bl_mean_agg: float = float(bl_total) / n_f

    diff_pos_raw: object = diff > 0
    assert isinstance(diff_pos_raw, np.ndarray)
    diff_pos: NDArray[np.bool_] = np.asarray(diff_pos_raw, dtype=np.bool_)
    blend_wins: int = int(np.count_nonzero(diff_pos))

    result: dict[str, object] = {
        "rounds": round_results,
        "gbdt_only_mean": go_mean_agg,
        "blended_mean": bl_mean_agg,
        "blend_wins": blend_wins,
        "total_rounds": n,
        "mean_improvement": diff_mean,
        "paired_t_stat": t_stat,
        "total_budget": total_budget,
        "alpha": alpha,
        "n_trials": n_trials,
        "total_time_seconds": total_time,
    }

    _print_benchmark_summary(round_results, result)
    return result


def _print_benchmark_summary(
    round_results: list[dict[str, object]],
    aggregate: dict[str, object],
) -> None:
    print("\n" + "=" * 78)
    print("Live Pipeline Benchmark (LOROCV + simulated queries)")
    print("=" * 78)
    budget = aggregate["total_budget"]
    alpha_val = aggregate["alpha"]
    n_trials_val = aggregate["n_trials"]
    assert isinstance(budget, int)
    assert isinstance(alpha_val, float)
    assert isinstance(n_trials_val, int)
    print(f"  Budget: {budget} queries | Alpha: {alpha_val} | Trials: {n_trials_val}")
    print("-" * 78)
    print(f"{'Round':>8} {'GBDT-only':>12} {'Blended':>12} {'Δ':>10} {'Winner':>10}")
    print("-" * 78)

    for r in round_results:
        rid = r["round_id"]
        go = r["gbdt_only_mean"]
        bl = r["blended_mean"]
        assert isinstance(rid, int)
        assert isinstance(go, float)
        assert isinstance(bl, float)
        d = bl - go
        winner = "Blended" if d > 0.01 else "GBDT" if d < -0.01 else "~Tie"
        print(f"{rid:>8} {go:>12.2f} {bl:>12.2f} {d:>+10.2f} {winner:>10}")

    print("-" * 78)
    go_mean = aggregate["gbdt_only_mean"]
    bl_mean = aggregate["blended_mean"]
    improvement = aggregate["mean_improvement"]
    wins = aggregate["blend_wins"]
    total = aggregate["total_rounds"]
    t_stat_val = aggregate["paired_t_stat"]
    total_time = aggregate["total_time_seconds"]

    assert isinstance(go_mean, float)
    assert isinstance(bl_mean, float)
    assert isinstance(improvement, float)
    assert isinstance(wins, int)
    assert isinstance(total, int)
    assert isinstance(t_stat_val, float)
    assert isinstance(total_time, float)

    print(f"{'Mean':>8} {go_mean:>12.2f} {bl_mean:>12.2f} {improvement:>+10.2f}")
    print(f"\nBlended wins: {wins}/{total} rounds")
    print(f"Paired t-statistic: {t_stat_val:.3f}")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 78 + "\n")
