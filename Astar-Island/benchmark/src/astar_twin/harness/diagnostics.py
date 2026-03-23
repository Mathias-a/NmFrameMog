"""Diagnostic decomposition for benchmark results.

Breaks down overall scores into per-class, per-cell, and per-seed
contributions so that solver changes can be traced to specific
mechanics or map regions.  All functions are pure and operate on
the ground-truth and prediction tensors stored in ``SeedResult``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.types import NUM_CLASSES, ClassIndex
from astar_twin.harness.models import BenchmarkReport, SeedResult, StrategyReport

CLASS_NAMES: dict[int, str] = {
    ClassIndex.EMPTY: "Empty",
    ClassIndex.SETTLEMENT: "Settlement",
    ClassIndex.PORT: "Port",
    ClassIndex.RUIN: "Ruin",
    ClassIndex.FOREST: "Forest",
    ClassIndex.MOUNTAIN: "Mountain",
}

_ENTROPY_THRESHOLD = 1e-10


@dataclass(frozen=True)
class CellDiagnostic:
    """Single cell's diagnostic data."""

    row: int
    col: int
    kl: float
    entropy: float
    weighted_kl: float  # entropy * kl
    dominant_gt_class: int
    dominant_pred_class: int


@dataclass(frozen=True)
class PerClassMetrics:
    """Aggregated metrics for one terrain class across all dynamic cells."""

    class_index: int
    class_name: str
    mean_gt_prob: float
    mean_pred_prob: float
    mean_kl_contribution: float
    total_weighted_kl: float
    fraction_of_total_loss: float


@dataclass(frozen=True)
class SeedDiagnostics:
    """Full diagnostic breakdown for a single (strategy, fixture, seed)."""

    seed_index: int
    score: float
    weighted_kl: float
    n_dynamic_cells: int
    n_static_cells: int
    mean_entropy: float
    per_class: tuple[PerClassMetrics, ...]
    worst_cells: tuple[CellDiagnostic, ...]  # top-N highest weighted KL

    def worst_class(self) -> PerClassMetrics:
        """Return the class contributing most to total loss."""
        return max(self.per_class, key=lambda c: c.fraction_of_total_loss)


@dataclass(frozen=True)
class StrategyDiagnostics:
    """Diagnostic breakdown aggregated across seeds for one strategy."""

    strategy_name: str
    fixture_id: str
    mean_score: float
    seed_diagnostics: tuple[SeedDiagnostics, ...]
    per_class_aggregate: tuple[PerClassMetrics, ...]
    worst_seed_index: int
    best_seed_index: int

    def to_dict(self) -> dict:
        """JSON-serialisable summary."""
        return {
            "strategy_name": self.strategy_name,
            "fixture_id": self.fixture_id,
            "mean_score": round(self.mean_score, 4),
            "worst_seed": self.worst_seed_index,
            "best_seed": self.best_seed_index,
            "per_class": [
                {
                    "class": pc.class_name,
                    "mean_gt_prob": round(pc.mean_gt_prob, 6),
                    "mean_pred_prob": round(pc.mean_pred_prob, 6),
                    "fraction_of_loss": round(pc.fraction_of_total_loss, 4),
                }
                for pc in self.per_class_aggregate
            ],
            "seeds": [
                {
                    "seed_index": sd.seed_index,
                    "score": round(sd.score, 4),
                    "weighted_kl": round(sd.weighted_kl, 6),
                    "n_dynamic_cells": sd.n_dynamic_cells,
                    "mean_entropy": round(sd.mean_entropy, 6),
                    "worst_class": sd.worst_class().class_name,
                    "top_5_worst_cells": [
                        {
                            "row": c.row,
                            "col": c.col,
                            "weighted_kl": round(c.weighted_kl, 6),
                            "gt_class": CLASS_NAMES.get(
                                c.dominant_gt_class, str(c.dominant_gt_class)
                            ),
                            "pred_class": CLASS_NAMES.get(
                                c.dominant_pred_class, str(c.dominant_pred_class)
                            ),
                        }
                        for c in sd.worst_cells[:5]
                    ],
                }
                for sd in self.seed_diagnostics
            ],
        }


@dataclass(frozen=True)
class DiagnosticReport:
    """Diagnostics for a full benchmark run (multiple strategies, one fixture)."""

    fixture_id: str
    strategy_diagnostics: tuple[StrategyDiagnostics, ...]

    def to_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "strategies": [sd.to_dict() for sd in self.strategy_diagnostics],
        }


# ---------------------------------------------------------------------------
# Computation functions
# ---------------------------------------------------------------------------


def _per_cell_kl_and_entropy(
    gt: NDArray[np.float64],
    pred: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-cell KL divergence and entropy arrays (H, W)."""
    eps = 1e-15
    gt_safe = np.clip(gt, eps, None)
    pred_safe = np.clip(pred, eps, None)

    entropy: NDArray[np.float64] = -np.sum(np.where(gt > 0, gt * np.log(gt_safe), 0.0), axis=2)
    kl: NDArray[np.float64] = np.sum(
        np.where(gt > 0, gt * np.log(gt_safe / pred_safe), 0.0), axis=2
    )
    return kl, entropy


def _per_class_kl_contributions(
    gt: NDArray[np.float64],
    pred: NDArray[np.float64],
    entropy: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> list[PerClassMetrics]:
    """Break down weighted KL contribution by terrain class."""
    eps = 1e-15
    gt_safe = np.clip(gt, eps, None)
    pred_safe = np.clip(pred, eps, None)
    total_entropy = float(np.sum(entropy[mask]))

    results: list[PerClassMetrics] = []
    for c in range(NUM_CLASSES):
        gt_c = gt[:, :, c]
        pred_c = pred[:, :, c]

        # per-cell KL contribution from this class:  p_c * log(p_c / q_c)
        kl_c = np.where(gt_c > 0, gt_c * np.log(gt_safe[:, :, c] / pred_safe[:, :, c]), 0.0)
        weighted_kl_c = entropy * kl_c
        total_wkl_c = float(np.sum(weighted_kl_c[mask]))

        mean_gt = float(np.mean(gt_c[mask])) if np.any(mask) else 0.0
        mean_pred = float(np.mean(pred_c[mask])) if np.any(mask) else 0.0
        mean_kl = float(np.mean(kl_c[mask])) if np.any(mask) else 0.0
        frac = total_wkl_c / total_entropy if total_entropy > 0 else 0.0

        results.append(
            PerClassMetrics(
                class_index=c,
                class_name=CLASS_NAMES.get(c, str(c)),
                mean_gt_prob=mean_gt,
                mean_pred_prob=mean_pred,
                mean_kl_contribution=mean_kl,
                total_weighted_kl=total_wkl_c,
                fraction_of_total_loss=frac,
            )
        )
    return results


def compute_seed_diagnostics(
    seed_result: SeedResult,
    n_worst_cells: int = 10,
) -> SeedDiagnostics:
    """Compute full diagnostic breakdown for a single seed."""
    gt = seed_result.ground_truth
    pred = seed_result.prediction

    kl, entropy = _per_cell_kl_and_entropy(gt, pred)
    mask = entropy >= _ENTROPY_THRESHOLD
    n_dynamic = int(np.sum(mask))
    n_static = int(mask.size - n_dynamic)
    mean_ent = float(np.mean(entropy[mask])) if n_dynamic > 0 else 0.0

    total_entropy = float(np.sum(entropy[mask]))
    weighted_kl_map = entropy * kl
    weighted_kl = float(np.sum(weighted_kl_map[mask]) / total_entropy) if total_entropy > 0 else 0.0

    per_class = _per_class_kl_contributions(gt, pred, entropy, mask)

    # Find worst cells (highest weighted KL)
    wkl_flat = weighted_kl_map.ravel()
    n_take = min(n_worst_cells, len(wkl_flat))
    worst_indices = np.argpartition(wkl_flat, -n_take)[-n_take:]
    worst_indices = worst_indices[np.argsort(wkl_flat[worst_indices])[::-1]]

    h, w = kl.shape
    worst_cells: list[CellDiagnostic] = []
    for idx in worst_indices:
        r, c = divmod(int(idx), w)
        worst_cells.append(
            CellDiagnostic(
                row=r,
                col=c,
                kl=float(kl[r, c]),
                entropy=float(entropy[r, c]),
                weighted_kl=float(weighted_kl_map[r, c]),
                dominant_gt_class=int(np.argmax(gt[r, c])),
                dominant_pred_class=int(np.argmax(pred[r, c])),
            )
        )

    return SeedDiagnostics(
        seed_index=seed_result.seed_index,
        score=seed_result.score,
        weighted_kl=weighted_kl,
        n_dynamic_cells=n_dynamic,
        n_static_cells=n_static,
        mean_entropy=mean_ent,
        per_class=tuple(per_class),
        worst_cells=tuple(worst_cells),
    )


def compute_strategy_diagnostics(
    report: StrategyReport,
    n_worst_cells: int = 10,
) -> StrategyDiagnostics:
    """Compute diagnostics for all seeds of one strategy."""
    seed_diags = tuple(
        compute_seed_diagnostics(sr, n_worst_cells=n_worst_cells) for sr in report.seed_results
    )

    # Aggregate per-class across seeds
    n_classes = NUM_CLASSES
    agg_gt = [0.0] * n_classes
    agg_pred = [0.0] * n_classes
    agg_kl = [0.0] * n_classes
    agg_wkl = [0.0] * n_classes
    agg_frac = [0.0] * n_classes
    n_seeds = len(seed_diags)

    for sd in seed_diags:
        for pc in sd.per_class:
            agg_gt[pc.class_index] += pc.mean_gt_prob
            agg_pred[pc.class_index] += pc.mean_pred_prob
            agg_kl[pc.class_index] += pc.mean_kl_contribution
            agg_wkl[pc.class_index] += pc.total_weighted_kl
            agg_frac[pc.class_index] += pc.fraction_of_total_loss

    per_class_agg = tuple(
        PerClassMetrics(
            class_index=c,
            class_name=CLASS_NAMES.get(c, str(c)),
            mean_gt_prob=agg_gt[c] / n_seeds if n_seeds > 0 else 0.0,
            mean_pred_prob=agg_pred[c] / n_seeds if n_seeds > 0 else 0.0,
            mean_kl_contribution=agg_kl[c] / n_seeds if n_seeds > 0 else 0.0,
            total_weighted_kl=agg_wkl[c] / n_seeds if n_seeds > 0 else 0.0,
            fraction_of_total_loss=agg_frac[c] / n_seeds if n_seeds > 0 else 0.0,
        )
        for c in range(n_classes)
    )

    scores = [sd.score for sd in seed_diags]
    worst_idx = int(np.argmin(scores)) if scores else 0
    best_idx = int(np.argmax(scores)) if scores else 0

    return StrategyDiagnostics(
        strategy_name=report.strategy_name,
        fixture_id=report.fixture_id,
        mean_score=report.mean_score,
        seed_diagnostics=seed_diags,
        per_class_aggregate=per_class_agg,
        worst_seed_index=seed_diags[worst_idx].seed_index if seed_diags else 0,
        best_seed_index=seed_diags[best_idx].seed_index if seed_diags else 0,
    )


def compute_diagnostic_report(
    benchmark_report: BenchmarkReport,
    n_worst_cells: int = 10,
) -> DiagnosticReport:
    """Compute full diagnostics from a BenchmarkReport."""
    return DiagnosticReport(
        fixture_id=benchmark_report.fixture_id,
        strategy_diagnostics=tuple(
            compute_strategy_diagnostics(sr, n_worst_cells=n_worst_cells)
            for sr in benchmark_report.strategy_reports
        ),
    )
