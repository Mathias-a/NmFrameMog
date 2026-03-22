"""Three-phase query allocation strategy for the astar_island hybrid solver.

Phase 1 (calibration):  farthest-point sampling over dynamic cells.
Phase 2 (refinement):   greedy expected information gain (EIG).
Phase 3 (exploitation): focus on high-entropy, low-observation cells.

All viewport coordinates are (vx, vy) top-left corners clamped to
[0, W-VIEWPORT] × [0, H-VIEWPORT]  →  [0, 25] × [0, 25].
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from astar_island.config import Config
from astar_island.types import (
    N_SEEDS,
    VIEWPORT,
    AlphaTensor,
    BoolMask,
    Grid,
    H,
    K,
    ProbTensor,
    W,
)
from astar_island.utils import entropy

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
_MAX_VX: int = W - VIEWPORT  # 25  (valid vx ∈ [0, 25])
_MAX_VY: int = H - VIEWPORT  # 25  (valid vy ∈ [0, 25])

# Pre-build arrays of all valid (vx, vy) anchor positions — shape (676, 2)
_VX_ALL: NDArray[np.int32] = np.arange(_MAX_VX + 1, dtype=np.int32)
_VY_ALL: NDArray[np.int32] = np.arange(_MAX_VY + 1, dtype=np.int32)
# meshgrid → shape (26, 26); stack → (676, 2) with columns [vx, vy]
_VX_GRID, _VY_GRID = np.meshgrid(_VX_ALL, _VY_ALL, indexing="xy")
_ANCHORS: NDArray[np.int32] = np.stack(
    [_VX_GRID.ravel(), _VY_GRID.ravel()], axis=1
).astype(np.int32)  # (676, 2)
_N_ANCHORS: int = _ANCHORS.shape[0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _viewport_sums(field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute sum of *field* over every valid viewport using 2-D prefix sums.

    Parameters
    ----------
    field:
        2-D array of shape (H, W).

    Returns
    -------
    NDArray of shape (MAX_VY+1, MAX_VX+1) = (26, 26) with viewport sums.
    """
    # 2-D cumulative sum (prefix sum)
    cs: NDArray[np.float64] = np.asarray(
        np.cumsum(np.cumsum(field, axis=0), axis=1), dtype=np.float64
    )
    # Pad with zeros on the top and left for easy rectangular-sum queries
    padded: NDArray[np.float64] = np.zeros((H + 1, W + 1), dtype=np.float64)
    padded[1:, 1:] = cs

    # For viewport anchored at (vy, vx), the window covers [vy:vy+V, vx:vx+V].
    # Rectangle sum = P[vy+V, vx+V] - P[vy, vx+V] - P[vy+V, vx] + P[vy, vx]
    vy_idx: NDArray[np.int32] = np.arange(_MAX_VY + 1, dtype=np.int32)  # [0..25]
    vx_idx: NDArray[np.int32] = np.arange(_MAX_VX + 1, dtype=np.int32)  # [0..25]

    vy_end: NDArray[np.int32] = vy_idx + VIEWPORT  # [15..40]
    vx_end: NDArray[np.int32] = vx_idx + VIEWPORT  # [15..40]

    sums: NDArray[np.float64] = (
        padded[vy_end[:, np.newaxis], vx_end[np.newaxis, :]]
        - padded[vy_idx[:, np.newaxis], vx_end[np.newaxis, :]]
        - padded[vy_end[:, np.newaxis], vx_idx[np.newaxis, :]]
        + padded[vy_idx[:, np.newaxis], vx_idx[np.newaxis, :]]
    )
    return sums  # shape (26, 26)


def _eig_map(
    pred: ProbTensor,
    alpha: AlphaTensor,
    dynamic: BoolMask,
) -> NDArray[np.float64]:
    """Per-cell EIG, vectorised over K classes.

    EIG[h,w] = H(pred[h,w]) - Σ_k pred[h,w,k] · H(α[h,w]+e_k) / (Σα+1)

    Static cells are zeroed out.
    """
    h_current: NDArray[np.float64] = entropy(pred)  # (H, W)

    alpha_sum: NDArray[np.float64] = np.asarray(
        np.sum(alpha, axis=-1), dtype=np.float64
    )  # (H, W)
    new_sum: NDArray[np.float64] = (alpha_sum + 1.0)[..., np.newaxis]  # (H,W,1)

    # Stack all K posterior alphas at once: (K, H, W, K)
    eye_k: NDArray[np.float64] = np.eye(K, dtype=np.float64)  # (K, K)
    # alpha_post[k] = alpha + e_k  — broadcast: (H,W,K) + (K,) → (K,H,W,K)
    alpha_post: NDArray[np.float64] = (
        alpha[np.newaxis, ...] + eye_k[:, np.newaxis, np.newaxis, :]
    )  # (K, H, W, K)

    # Normalise each posterior
    pred_post: NDArray[np.float64] = np.asarray(
        alpha_post / new_sum[np.newaxis, ...], dtype=np.float64
    )  # (K, H, W, K)

    # Entropy of each posterior: (K, H, W)
    h_post: NDArray[np.float64] = entropy(pred_post)  # (K, H, W)

    # Weight by current pred probability: pred[h,w,k] → (K, H, W)
    pred_t: NDArray[np.float64] = np.asarray(
        np.transpose(pred, (2, 0, 1)), dtype=np.float64
    )  # (K, H, W)

    e_h_post: NDArray[np.float64] = np.asarray(
        np.sum(pred_t * h_post, axis=0), dtype=np.float64
    )  # (H, W)

    eig: NDArray[np.float64] = np.maximum(h_current - e_h_post, 0.0)
    eig[~dynamic] = 0.0
    return eig


# ---------------------------------------------------------------------------
# Phase 1: Farthest-point calibration
# ---------------------------------------------------------------------------


def plan_phase1(
    initial_grid: Grid,
    dynamic: BoolMask,
    n_seeds: int,
    queries_per_seed: int = 2,
) -> list[list[tuple[int, int]]]:
    """Plan Phase 1 calibration viewports using farthest-point sampling.

    Selects viewport top-left corners (vx, vy) to maximise coverage of
    dynamic cells with no overlap between viewports within a seed.

    Different seeds receive slightly shifted starting positions to increase
    cross-seed information diversity.

    Returns
    -------
    list of n_seeds lists, each containing queries_per_seed (vx, vy) tuples.
    Each (vx, vy) is the top-left corner of a VIEWPORT×VIEWPORT window.
    """
    del initial_grid  # not needed once dynamic mask is provided

    dyn_field: NDArray[np.float64] = np.asarray(dynamic, dtype=np.float64)
    dyn_sums: NDArray[np.float64] = _viewport_sums(dyn_field)  # (26, 26)

    result: list[list[tuple[int, int]]] = []

    for seed_i in range(n_seeds):
        seed_viewports: list[tuple[int, int]] = []

        # Bias the starting search by a small offset per seed (mod grid)
        # so that different seeds probe different parts of the map.
        offset_y: int = (seed_i * 5) % (_MAX_VY + 1)
        offset_x: int = (seed_i * 7) % (_MAX_VX + 1)

        # Shifted scores: roll the dyn_sums so we start near a different peak
        shifted: NDArray[np.float64] = np.roll(
            np.roll(dyn_sums, offset_y, axis=0), offset_x, axis=1
        )

        # min-distance tracker in viewport-anchor space
        min_dist: NDArray[np.float64] = np.full(
            (_MAX_VY + 1, _MAX_VX + 1), np.inf, dtype=np.float64
        )

        for q in range(queries_per_seed):
            if q == 0:
                # First viewport: highest coverage in shifted space
                flat_best: int = int(np.argmax(shifted))
                best_vy_s, best_vx_s = divmod(flat_best, _MAX_VX + 1)
                # Unshift back to original coordinates
                best_vy: int = (best_vy_s - offset_y) % (_MAX_VY + 1)
                best_vx: int = (best_vx_s - offset_x) % (_MAX_VX + 1)
                chosen: tuple[int, int] = (best_vx, best_vy)
            else:
                # Subsequent: farthest (Manhattan on centers) from all chosen
                last_vx, last_vy = seed_viewports[-1]
                last_cx: int = last_vx + VIEWPORT // 2
                last_cy: int = last_vy + VIEWPORT // 2

                # Update min-Manhattan-distance from anchor centres
                vy_idx: NDArray[np.int32] = np.arange(_MAX_VY + 1, dtype=np.int32)
                vx_idx: NDArray[np.int32] = np.arange(_MAX_VX + 1, dtype=np.int32)
                cy_all: NDArray[np.int32] = vy_idx + VIEWPORT // 2  # (26,)
                cx_all: NDArray[np.int32] = vx_idx + VIEWPORT // 2  # (26,)

                dist_to_last: NDArray[np.float64] = np.asarray(
                    np.abs(cy_all[:, np.newaxis] - last_cy)
                    + np.abs(cx_all[np.newaxis, :] - last_cx),
                    dtype=np.float64,
                )  # (26, 26)
                min_dist = np.minimum(min_dist, dist_to_last)

                # Enforce no-overlap: zero score for overlapping anchors
                overlap_mask: NDArray[np.bool_] = min_dist < VIEWPORT
                scores: NDArray[np.float64] = min_dist * dyn_sums
                scores[overlap_mask] = -np.inf

                flat_idx: int = int(np.argmax(scores))
                sel_vy, sel_vx = divmod(flat_idx, _MAX_VX + 1)
                chosen = (sel_vx, sel_vy)

            seed_viewports.append(chosen)

        result.append(seed_viewports)

    log.debug("Phase 1 plan: %s", result)
    return result


# ---------------------------------------------------------------------------
# Phase 2: Greedy EIG refinement
# ---------------------------------------------------------------------------


def select_eig_viewport(
    pred: ProbTensor,
    alpha: AlphaTensor,
    n_obs: NDArray[np.int32],
    dynamic: BoolMask,
) -> tuple[int, int]:
    """Select next viewport maximising expected information gain.

    For each cell: EIG = H[current] - E_k[H[posterior if observe class k]]
    Viewport score = Σ EIG[h,w] / (1 + n_obs[h,w]) over viewport cells.

    Returns
    -------
    (vx, vy) top-left corner of the best viewport.
    """
    eig: NDArray[np.float64] = _eig_map(pred, alpha, dynamic)  # (H, W)

    # Overlap discount: weight each cell by 1/(1+n_obs)
    discount: NDArray[np.float64] = 1.0 / (
        1.0 + np.asarray(n_obs, dtype=np.float64)
    )  # (H, W)
    discounted_eig: NDArray[np.float64] = eig * discount  # (H, W)

    scores: NDArray[np.float64] = _viewport_sums(discounted_eig)  # (26, 26)

    flat_best: int = int(np.argmax(scores))
    best_vy, best_vx = divmod(flat_best, _MAX_VX + 1)
    log.debug(
        "EIG viewport: vx=%d vy=%d score=%.4f",
        best_vx,
        best_vy,
        float(scores[best_vy, best_vx]),
    )
    return (best_vx, best_vy)


# ---------------------------------------------------------------------------
# Phase 3: Exploitation
# ---------------------------------------------------------------------------


def select_exploitation_viewport(
    pred: ProbTensor,
    n_obs: NDArray[np.int32],
    dynamic: BoolMask,
) -> tuple[int, int]:
    """Select viewport targeting highest remaining uncertainty.

    Score per cell = entropy(pred[h,w]) × (1 if n_obs[h,w] < 3 else 0.5)
    Focuses on high-entropy dynamic cells with few observations.

    Returns
    -------
    (vx, vy) top-left corner.
    """
    ent: NDArray[np.float64] = entropy(pred)  # (H, W)
    ent[~dynamic] = 0.0

    obs_weight: NDArray[np.float64] = np.where(
        np.asarray(n_obs, dtype=np.int32) < 3, 1.0, 0.5
    ).astype(np.float64)  # (H, W)

    weighted: NDArray[np.float64] = ent * obs_weight  # (H, W)
    scores: NDArray[np.float64] = _viewport_sums(weighted)  # (26, 26)

    flat_best: int = int(np.argmax(scores))
    best_vy, best_vx = divmod(flat_best, _MAX_VX + 1)
    log.debug(
        "Exploitation viewport: vx=%d vy=%d score=%.4f",
        best_vx,
        best_vy,
        float(scores[best_vy, best_vx]),
    )
    return (best_vx, best_vy)


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------


def allocate_budget(
    config: Config,
) -> tuple[list[int], list[int], list[int]]:
    """Allocate query budget across phases and seeds.

    Returns
    -------
    phase1_per_seed, phase2_per_seed, phase3_per_seed
        Each is a list of N_SEEDS ints.  Totals across all seeds/phases
        equal config.n_queries_total (50).

    Strategy
    --------
    - Phase 1: 2 queries per seed = 10 total.
    - Remaining per seed = seed_cap - 2.
    - Phase 3 gets min(remaining // 4, 2); Phase 2 gets the rest.
    """
    p1_each: int = 2
    p1: list[int] = [p1_each] * N_SEEDS

    p2: list[int] = []
    p3: list[int] = []

    for cap in config.seed_caps:
        remaining: int = cap - p1_each
        p3_n: int = min(remaining // 4, 2)
        p2_n: int = remaining - p3_n
        p2.append(p2_n)
        p3.append(p3_n)

    log.debug(
        "Budget — p1=%s p2=%s p3=%s total=%d",
        p1,
        p2,
        p3,
        sum(p1) + sum(p2) + sum(p3),
    )
    return p1, p2, p3


# ---------------------------------------------------------------------------
# Seed selection
# ---------------------------------------------------------------------------


def select_seed(
    queries_used: list[int],
    seed_caps: tuple[int, ...],
    pred_per_seed: list[ProbTensor],
    alpha_per_seed: list[AlphaTensor],
    dynamic: BoolMask,
) -> int:
    """Select which seed to query next.

    Picks the seed with the highest total remaining EIG that has not yet
    exhausted its per-seed budget.

    Parameters
    ----------
    queries_used:
        Number of queries already used for each seed.
    seed_caps:
        Maximum queries per seed (from config).
    pred_per_seed, alpha_per_seed:
        Current belief state per seed.
    dynamic:
        Shared dynamic-cell mask.

    Returns
    -------
    Index of the chosen seed.
    """
    best_seed: int = 0
    best_score: float = -1.0

    for i, (used, cap) in enumerate(zip(queries_used, seed_caps, strict=False)):
        if used >= cap:
            continue  # budget exhausted

        eig: NDArray[np.float64] = _eig_map(
            pred_per_seed[i], alpha_per_seed[i], dynamic
        )
        total_eig: float = float(np.sum(eig))

        if total_eig > best_score:
            best_score = total_eig
            best_seed = i

    log.debug("Selected seed %d (total EIG=%.4f)", best_seed, best_score)
    return best_seed


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def plan_next_query(
    phase: int,
    seed_index: int,
    pred: ProbTensor,
    alpha: AlphaTensor,
    n_obs: NDArray[np.int32],
    dynamic: BoolMask,
    phase1_plan: list[list[tuple[int, int]]] | None = None,
    phase1_idx: int = 0,
) -> tuple[int, int]:
    """Select next viewport based on current phase.

    Parameters
    ----------
    phase:
        1, 2, or 3.
    seed_index:
        Which seed this query is for (used to look up phase1_plan).
    pred:
        Current probability tensor for this seed (H, W, K).
    alpha:
        Current Dirichlet alpha for this seed (H, W, K).
    n_obs:
        Per-cell observation counts for this seed (H, W).
    dynamic:
        Shared dynamic-cell mask (H, W).
    phase1_plan:
        Pre-computed Phase 1 viewport list (from plan_phase1).
        Required when phase==1.
    phase1_idx:
        Which entry in phase1_plan[seed_index] to return (0-based).

    Returns
    -------
    (vx, vy) top-left corner of the selected viewport.
    """
    if phase == 1:
        if phase1_plan is None:
            log.warning("phase1_plan not provided for Phase 1; falling back to EIG.")
            return select_eig_viewport(pred, alpha, n_obs, dynamic)
        viewports: list[tuple[int, int]] = phase1_plan[seed_index]
        idx: int = min(phase1_idx, len(viewports) - 1)
        chosen_p1: tuple[int, int] = viewports[idx]
        log.debug(
            "Phase 1 seed=%d idx=%d → vx=%d vy=%d",
            seed_index,
            idx,
            chosen_p1[0],
            chosen_p1[1],
        )
        return chosen_p1

    if phase == 2:
        return select_eig_viewport(pred, alpha, n_obs, dynamic)

    if phase == 3:
        return select_exploitation_viewport(pred, n_obs, dynamic)

    msg: str = f"Unknown phase {phase!r}; expected 1, 2, or 3."
    raise ValueError(msg)
