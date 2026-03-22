#!/usr/bin/env python3
"""Fetch Round #20 analysis from API and perform detailed error analysis.

Downloads per-seed ground truth + scores, then compares against our
current model's predictions to identify systematic weaknesses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add package to path
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "apps" / "astar-island" / "src")
)

from astar_island.api import make_client, get_analysis, get_round_detail
from astar_island.scoring import (
    competition_score,
    kl_divergence_per_cell,
    entropy_per_cell,
)
from astar_island.solver import predict_grid
from astar_island.prediction import apply_probability_floor
from astar_island.terrain import TERRAIN_CODE_TO_CLASS, NUM_PREDICTION_CLASSES

ROUND_20_ID = "fd82f643-15e2-40e7-9866-8d8f5157081c"
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
OUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "round20"


def fetch_all_seeds(round_id: str) -> list[dict]:
    """Fetch analysis for all seeds."""
    results = []
    with make_client() as client:
        for seed in range(5):
            print(f"Fetching seed {seed}...", end=" ", flush=True)
            analysis = get_analysis(client, round_id, seed)
            if analysis is None:
                print("unavailable")
                continue
            score = analysis.get("score", "?")
            print(f"score={score}")
            results.append({"seed": seed, **analysis})
    return results


def analyze_single_seed(seed_data: dict, repredict: bool = True) -> dict:
    """Analyze errors for a single seed."""
    seed_idx = seed_data["seed"]
    gt_raw = seed_data["ground_truth"]
    initial_grid = seed_data["initial_grid"]
    submitted_pred_raw = seed_data.get("prediction")
    official_score = seed_data.get("score", 0.0)

    gt = np.array(gt_raw, dtype=np.float64)
    submitted_pred = (
        np.array(submitted_pred_raw, dtype=np.float64) if submitted_pred_raw else None
    )

    # Per-class ground truth distribution (mean over all cells)
    gt_class_means = gt.mean(axis=(0, 1))

    # Identify dynamic vs static cells
    grid_arr = np.array(initial_grid, dtype=np.int32)
    is_ocean = grid_arr == 10
    is_mountain = grid_arr == 5
    is_dynamic = ~(is_ocean | is_mountain)
    n_dynamic = int(is_dynamic.sum())
    n_total = grid_arr.size

    # GT class means on dynamic cells only
    gt_dynamic_means = gt[is_dynamic].mean(axis=0)

    result = {
        "seed": seed_idx,
        "official_score": official_score,
        "n_dynamic": n_dynamic,
        "n_total": n_total,
        "gt_class_means_all": gt_class_means.tolist(),
        "gt_class_means_dynamic": gt_dynamic_means.tolist(),
    }

    # Score of submitted prediction
    if submitted_pred is not None:
        sub_score = competition_score(gt, submitted_pred)
        result["submitted_local_score"] = sub_score

        # Per-class KL analysis on submitted prediction
        kl_cells = kl_divergence_per_cell(gt, submitted_pred)  # (H, W)
        ent_cells = entropy_per_cell(gt)  # (H, W)

        # Worst cells
        worst_indices = np.unravel_index(
            np.argsort(kl_cells.ravel())[-10:], kl_cells.shape
        )
        worst_kls = kl_cells[worst_indices]

        result["submitted_kl_mean"] = float(kl_cells[is_dynamic].mean())
        result["submitted_kl_max"] = float(kl_cells.max())
        result["submitted_kl_p95"] = float(np.percentile(kl_cells[is_dynamic], 95))

        # Per-class error analysis: for each class, look at cells where GT has > 0.5 prob
        per_class_errors = {}
        for k in range(NUM_PREDICTION_CLASSES):
            dominant = gt[:, :, k] > 0.3  # cells where this class is significant
            if dominant.sum() == 0:
                continue
            # Mean predicted prob for this class on cells where GT says it's dominant
            pred_mean = float(submitted_pred[:, :, k][dominant].mean())
            gt_mean = float(gt[:, :, k][dominant].mean())
            # Mean KL on these cells
            kl_mean = float(kl_cells[dominant].mean()) if dominant.sum() > 0 else 0
            per_class_errors[CLASS_NAMES[k]] = {
                "n_cells": int(dominant.sum()),
                "gt_mean_prob": round(gt_mean, 4),
                "pred_mean_prob": round(pred_mean, 4),
                "error": round(pred_mean - gt_mean, 4),
                "kl_mean": round(kl_mean, 4),
            }
        result["submitted_per_class_errors"] = per_class_errors

    # Re-predict with current model
    if repredict:
        print(
            f"  Re-predicting seed {seed_idx} with current model...",
            end=" ",
            flush=True,
        )
        new_pred = predict_grid(initial_grid)
        new_safe = apply_probability_floor(new_pred, initial_grid)
        new_score = competition_score(gt, new_safe)
        print(f"score={new_score:.2f}")

        result["new_model_score"] = new_score

        # Difference from official
        if official_score > 0:
            result["new_vs_official"] = round(new_score - official_score, 2)

        # Per-class error analysis for new model
        kl_new = kl_divergence_per_cell(gt, new_safe)
        per_class_new = {}
        for k in range(NUM_PREDICTION_CLASSES):
            dominant = gt[:, :, k] > 0.3
            if dominant.sum() == 0:
                continue
            pred_mean = float(new_safe[:, :, k][dominant].mean())
            gt_mean = float(gt[:, :, k][dominant].mean())
            kl_mean = float(kl_new[dominant].mean()) if dominant.sum() > 0 else 0
            per_class_new[CLASS_NAMES[k]] = {
                "n_cells": int(dominant.sum()),
                "gt_mean_prob": round(gt_mean, 4),
                "pred_mean_prob": round(pred_mean, 4),
                "error": round(pred_mean - gt_mean, 4),
                "kl_mean": round(kl_mean, 4),
            }
        result["new_model_per_class_errors"] = per_class_new

        # Settlement bias analysis (key failure mode)
        settlement_gt = gt[:, :, 1]  # class 1 = settlement
        settlement_pred_old = (
            submitted_pred[:, :, 1] if submitted_pred is not None else None
        )
        settlement_pred_new = new_safe[:, :, 1]

        result["settlement_analysis"] = {
            "gt_mean_all": round(float(settlement_gt.mean()), 4),
            "gt_mean_dynamic": round(float(settlement_gt[is_dynamic].mean()), 4),
            "new_pred_mean_all": round(float(settlement_pred_new.mean()), 4),
            "new_pred_mean_dynamic": round(
                float(settlement_pred_new[is_dynamic].mean()), 4
            ),
        }
        if settlement_pred_old is not None:
            result["settlement_analysis"]["old_pred_mean_all"] = round(
                float(settlement_pred_old.mean()), 4
            )
            result["settlement_analysis"]["old_pred_mean_dynamic"] = round(
                float(settlement_pred_old[is_dynamic].mean()), 4
            )

    return result


def print_summary(analyses: list[dict]) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("ROUND #20 ANALYSIS SUMMARY")
    print("=" * 70)

    # Per-seed scores
    print("\n--- Per-Seed Scores ---")
    print(
        f"{'Seed':<6} {'Official':<12} {'Submitted(local)':<18} {'New Model':<12} {'Delta':<8}"
    )
    print("-" * 56)
    official_scores = []
    new_scores = []
    for a in analyses:
        seed = a["seed"]
        off = a.get("official_score", 0)
        sub = a.get("submitted_local_score", 0)
        new = a.get("new_model_score", 0)
        delta = a.get("new_vs_official", 0)
        print(f"{seed:<6} {off:<12.2f} {sub:<18.2f} {new:<12.2f} {delta:+.2f}")
        official_scores.append(off)
        new_scores.append(new)

    if official_scores:
        print("-" * 56)
        off_mean = np.mean(official_scores)
        new_mean = np.mean(new_scores)
        print(
            f"{'MEAN':<6} {off_mean:<12.2f} {'':18} {new_mean:<12.2f} {new_mean - off_mean:+.2f}"
        )

    # Settlement bias per seed
    print("\n--- Settlement Bias (Key Failure Mode) ---")
    print(
        f"{'Seed':<6} {'GT Mean':<10} {'Old Pred':<10} {'New Pred':<10} {'Old Err':<10} {'New Err':<10}"
    )
    print("-" * 56)
    for a in analyses:
        sa = a.get("settlement_analysis", {})
        gt = sa.get("gt_mean_dynamic", 0)
        old = sa.get("old_pred_mean_dynamic", 0)
        new = sa.get("new_pred_mean_dynamic", 0)
        old_err = old - gt if old else 0
        new_err = new - gt
        print(
            f"{a['seed']:<6} {gt:<10.4f} {old:<10.4f} {new:<10.4f} {old_err:+.4f}   {new_err:+.4f}"
        )

    # Per-class error breakdown (new model)
    print("\n--- Per-Class Error Analysis (New Model, Dynamic Cells, GT > 0.3) ---")
    for a in analyses:
        if "new_model_per_class_errors" not in a:
            continue
        print(f"\nSeed {a['seed']} (new_score={a.get('new_model_score', 0):.2f}):")
        errors = a["new_model_per_class_errors"]
        print(
            f"  {'Class':<12} {'N Cells':<10} {'GT Mean':<10} {'Pred Mean':<10} {'Error':<10} {'KL Mean':<10}"
        )
        for cls_name, info in errors.items():
            print(
                f"  {cls_name:<12} {info['n_cells']:<10} {info['gt_mean_prob']:<10.4f} {info['pred_mean_prob']:<10.4f} {info['error']:+.4f}    {info['kl_mean']:<10.4f}"
            )

    # Ground truth class distribution per seed
    print("\n--- GT Class Distribution (Dynamic Cells) ---")
    print(f"{'Seed':<6}", end="")
    for name in CLASS_NAMES:
        print(f" {name:<10}", end="")
    print()
    print("-" * 66)
    for a in analyses:
        print(f"{a['seed']:<6}", end="")
        for v in a.get("gt_class_means_dynamic", []):
            print(f" {v:<10.4f}", end="")
        print()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching Round #20 analysis from API...")
    seed_data = fetch_all_seeds(ROUND_20_ID)

    if not seed_data:
        print("ERROR: No analysis data available for Round #20")
        return

    # Save raw API data
    for sd in seed_data:
        path = OUT_DIR / f"seed{sd['seed']}_raw.json"
        path.write_text(json.dumps(sd, indent=2))
        print(f"Saved raw data: {path}")

    # Analyze each seed
    print("\nAnalyzing per-seed errors...")
    analyses = []
    for sd in seed_data:
        analysis = analyze_single_seed(sd)
        analyses.append(analysis)

    # Save analysis results
    analysis_path = OUT_DIR / "analysis_summary.json"
    analysis_path.write_text(json.dumps(analyses, indent=2))
    print(f"\nSaved analysis: {analysis_path}")

    # Print summary
    print_summary(analyses)


if __name__ == "__main__":
    main()
