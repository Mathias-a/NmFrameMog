"""Cross-seed round calibration via archetype-based empirical Bayes.

Identifies the current round's hidden parameter regime by matching
pooled cross-seed observations against historical archetype frequency tables.

Archetypes:
    ARCHETYPE_COASTAL = 0           non-ocean cells adjacent to ocean (code 10)
    ARCHETYPE_SETTLEMENT_ADJ = 1    within Chebyshev-2 of settlement,
                                        not coastal
    ARCHETYPE_INLAND = 2            all other cells

Pipeline:
    1. Precompute theta_hist offline via compute_archetype_frequencies.
    2. At each calibration checkpoint, call aggregate_observations_by_archetype
       to pool counts from all seeds, then compute_round_weights to obtain
       posterior weights over historical rounds.
    3. Build per-seed probability templates via build_archetype_template.
    4. RoundCalibrator wraps the stateful online workflow.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
from numpy.typing import NDArray

from astar_island.archetypes import (
    COASTAL,
    INLAND_NATURAL,
    SETTLEMENT_ADJACENT,
)
from astar_island.archetypes import (
    classify_archetypes as _classify_archetypes,
)
from astar_island.config import Config
from astar_island.types import CountTensor, Grid, K, ProbTensor
from astar_island.utils import clip_normalize, softmax

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Archetype constants (re-exported for callers that import from calibration)
# ---------------------------------------------------------------------------

ARCHETYPE_COASTAL: Final[int] = COASTAL
ARCHETYPE_SETTLEMENT_ADJ: Final[int] = SETTLEMENT_ADJACENT
ARCHETYPE_INLAND: Final[int] = INLAND_NATURAL
N_ARCHETYPES: Final[int] = 3

# ---------------------------------------------------------------------------
# Archetype classification (thin wrapper around archetypes module)
# ---------------------------------------------------------------------------


def classify_archetypes(initial_grid: Grid) -> NDArray[np.int32]:
    """Classify each cell into archetype.

    0 = coastal   — non-ocean cell adjacent to ocean (code 10) in Chebyshev-1
    1 = settlement_adjacent — within Chebyshev distance 2 of settlement code 1,
                               not coastal
    2 = inland_natural — neither

    Args:
        initial_grid: shape (H, W) terrain codes.

    Returns:
        shape (H, W) int32 array with values in {0, 1, 2}.
    """
    return _classify_archetypes(initial_grid)


# ---------------------------------------------------------------------------
# Offline precomputation
# ---------------------------------------------------------------------------


def compute_archetype_frequencies(
    rounds: dict[int, list[tuple[NDArray[np.int_], NDArray[np.float64]]]],
) -> tuple[NDArray[np.float64], list[int]]:
    """Compute per-round, per-archetype class frequency tables from historical data.

    For each historical round r, for each archetype a, computes the average
    ground-truth probability vector across all cells of that archetype,
    averaged across all seeds within that round.

    Args:
        rounds: mapping from round_id to a list of (initial_grid, gt_probs) pairs,
                one per seed.  initial_grid: (H, W) int32; gt_probs: (H, W, K) float64.

    Returns:
        theta_hist: shape (n_rounds, N_ARCHETYPES, K) — class frequency tables.
        round_ids: list of round numbers corresponding to axis-0 of theta_hist.
    """
    round_ids: list[int] = sorted(rounds.keys())
    n_rounds = len(round_ids)
    theta_hist: NDArray[np.float64] = np.zeros(
        (n_rounds, N_ARCHETYPES, K),
        dtype=np.float64,
    )

    for r_idx, rid in enumerate(round_ids):
        seeds = rounds[rid]
        # Accumulate numerator (sum of per-archetype mean vectors across seeds)
        accum: NDArray[np.float64] = np.zeros((N_ARCHETYPES, K), dtype=np.float64)
        seed_count: NDArray[np.int64] = np.zeros(N_ARCHETYPES, dtype=np.int64)

        for grid, gt in seeds:
            grid_i32: NDArray[np.int32] = np.asarray(grid, dtype=np.int32)
            arch_map = _classify_archetypes(grid_i32)
            static_mask: NDArray[np.bool_] = np.asarray(
                (grid_i32 == 10) | (grid_i32 == 5),
                dtype=np.bool_,
            )

            for a in range(N_ARCHETYPES):
                cell_mask: NDArray[np.bool_] = np.asarray(
                    (arch_map == a) & ~static_mask,
                    dtype=np.bool_,
                )
                n_cells = int(np.count_nonzero(cell_mask))
                if n_cells == 0:
                    continue
                # Mean gt probability over matching cells: (K,)
                mean_probs: NDArray[np.float64] = np.asarray(
                    gt[cell_mask].mean(axis=0),
                    dtype=np.float64,
                )
                accum[a] += mean_probs
                seed_count[a] += 1

        for a in range(N_ARCHETYPES):
            if seed_count[a] == 0:
                theta_hist[r_idx, a] = 1.0 / K  # uniform fallback
            else:
                theta_hist[r_idx, a] = accum[a] / float(seed_count[a])

    return theta_hist, round_ids


# ---------------------------------------------------------------------------
# Online observation aggregation
# ---------------------------------------------------------------------------


def aggregate_observations_by_archetype(
    archetype_maps: list[NDArray[np.int32]],
    counts_per_seed: list[CountTensor],
) -> NDArray[np.float64]:
    """Aggregate observed counts by archetype across all seeds.

    Args:
        archetype_maps: list of N_SEEDS archetype maps, each shape (H, W).
        counts_per_seed: list of N_SEEDS count tensors, each shape (H, W, K).

    Returns:
        m_obs: shape (N_ARCHETYPES, K) — total observed counts per archetype,
               as float64 for direct use in log-likelihood.
    """
    m_obs: NDArray[np.float64] = np.zeros((N_ARCHETYPES, K), dtype=np.float64)

    for arch_map, counts in zip(archetype_maps, counts_per_seed, strict=False):
        counts_f: NDArray[np.float64] = np.asarray(counts, dtype=np.float64)
        for a in range(N_ARCHETYPES):
            mask: NDArray[np.bool_] = np.asarray(arch_map == a, dtype=np.bool_)
            # Sum over matching cells: counts_f[mask] has shape (n_cells, K)
            m_obs[a] += np.asarray(counts_f[mask].sum(axis=0), dtype=np.float64)

    return m_obs


# ---------------------------------------------------------------------------
# Empirical Bayes round weights
# ---------------------------------------------------------------------------


def compute_round_weights(
    m_obs: NDArray[np.float64],
    theta_hist: NDArray[np.float64],
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """Compute posterior weights over historical rounds given online observations.

    Uses a multinomial log-likelihood: for each historical round r,
        log_w[r] = sum_{a,k} m_obs[a,k] * log(clip(theta_hist[r,a,k], eps))

    Then applies softmax (with log-sum-exp trick) to get normalized weights.

    Args:
        m_obs: shape (N_ARCHETYPES, K) — total observed counts per archetype.
        theta_hist: shape (n_rounds, N_ARCHETYPES, K) — historical frequency tables.
        eps: probability floor applied before taking log.

    Returns:
        shape (n_rounds,) — normalized weights summing to 1.0.
        Returns uniform weights when total observations are zero.
    """
    total_obs = float(np.sum(m_obs))
    n_rounds = int(theta_hist.shape[0])

    if total_obs < 1e-12:
        log.debug("compute_round_weights: no observations — returning uniform weights")
        return np.full(n_rounds, 1.0 / n_rounds, dtype=np.float64)

    # log_theta: (n_rounds, N_ARCHETYPES, K)
    log_theta: NDArray[np.float64] = np.log(
        np.clip(theta_hist, eps, 1.0).astype(np.float64),
    )

    # log_w[r] = sum_{a,k} m_obs[a,k] * log_theta[r,a,k]
    # m_obs shape: (N_ARCHETYPES, K);  log_theta shape: (n_rounds, N_ARCHETYPES, K)
    # einsum: "ak, rak -> r"
    log_w: NDArray[np.float64] = np.einsum("ak,rak->r", m_obs, log_theta)

    weights = softmax(log_w)

    # Diagnostic logging
    best_idx = int(np.argmax(weights))
    log.debug(
        "compute_round_weights: top round idx=%d weight=%.4f (total_obs=%.0f)",
        best_idx,
        float(weights[best_idx]),
        total_obs,
    )

    return weights


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------


def build_archetype_template(
    w_hist: NDArray[np.float64],
    theta_hist: NDArray[np.float64],
    archetype_map: NDArray[np.int32],
    eps: float = 1e-4,
) -> ProbTensor:
    """Build per-cell probability template from weighted historical rounds.

    For each cell (h, w) with archetype a:
        p_template[h,w,k] = sum_r  w_hist[r] * theta_hist[r, a, k]

    Clips to eps floor and normalizes to sum to 1 along class axis.

    Args:
        w_hist: shape (n_rounds,) — posterior round weights.
        theta_hist: shape (n_rounds, N_ARCHETYPES, K) — historical frequency tables.
        archetype_map: shape (H, W) — archetype IDs for the current seed.
        eps: probability floor.

    Returns:
        ProbTensor of shape (H, W, K).
    """
    # Weighted archetype frequencies: (N_ARCHETYPES, K)
    # einsum "r, rak -> ak"
    weighted_theta: NDArray[np.float64] = np.einsum(
        "r,rak->ak",
        w_hist,
        theta_hist,
    )

    # Map archetype frequencies to each cell: (H, W, K)
    # weighted_theta[archetype_map] expands via advanced indexing
    template: NDArray[np.float64] = np.asarray(
        weighted_theta[archetype_map],
        dtype=np.float64,
    )

    return clip_normalize(template, eps)


# ---------------------------------------------------------------------------
# Stateful calibrator
# ---------------------------------------------------------------------------


class RoundCalibrator:
    """Manages cross-seed round calibration state.

    Holds:
      - Precomputed historical frequency tables (theta_hist).
      - Per-seed archetype maps (computed once from initial grids).
      - Current posterior round weights (updated at calibration checkpoints).

    Usage::

        calibrator = RoundCalibrator(theta_hist, round_ids, archetype_maps, config)
        calibrator.update(counts_per_seed)          # at each checkpoint
        template = calibrator.get_template(seed_idx)  # feed into posterior
    """

    def __init__(
        self,
        theta_hist: NDArray[np.float64],
        round_ids: list[int],
        archetype_maps: list[NDArray[np.int32]],
        config: Config,
    ) -> None:
        """Initialize with precomputed historical frequencies and seed archetype maps.

        Args:
            theta_hist: shape (n_rounds, N_ARCHETYPES, K).
            round_ids: list of round numbers (len == n_rounds).
            archetype_maps: list of N_SEEDS (H, W) archetype ID arrays.
            config: solver configuration (uses config.eps).
        """
        n_rounds = int(theta_hist.shape[0])
        if len(round_ids) != n_rounds:
            msg = f"round_ids length {len(round_ids)} != theta_hist.shape[0] {n_rounds}"
            raise ValueError(msg)

        self._theta_hist: NDArray[np.float64] = np.clip(
            np.asarray(theta_hist, dtype=np.float64),
            config.eps,
            1.0,
        )
        self._round_ids: list[int] = list(round_ids)
        self._archetype_maps: list[NDArray[np.int32]] = list(archetype_maps)
        self._config = config
        self._n_rounds = n_rounds

        # Start with uniform weights
        self._weights: NDArray[np.float64] = np.full(
            n_rounds,
            1.0 / n_rounds,
            dtype=np.float64,
        )

        log.debug(
            "RoundCalibrator initialized: %d historical rounds, %d seeds",
            n_rounds,
            len(archetype_maps),
        )

    def update(self, counts_per_seed: list[CountTensor]) -> None:
        """Recompute round weights from current observation counts across all seeds.

        Args:
            counts_per_seed: list of N_SEEDS (H, W, K) count tensors.
        """
        m_obs = aggregate_observations_by_archetype(
            self._archetype_maps,
            counts_per_seed,
        )
        self._weights = compute_round_weights(
            m_obs,
            self._theta_hist,
            eps=self._config.eps,
        )

        rid, wt = self.top_round_match
        log.info(
            "RoundCalibrator.update: top match round=%d weight=%.4f",
            rid,
            wt,
        )

    def get_template(self, seed_index: int) -> ProbTensor:
        """Get the calibrated probability template for a specific seed.

        Args:
            seed_index: index into the archetype_maps list (0-based).

        Returns:
            ProbTensor of shape (H, W, K).
        """
        arch_map = self._archetype_maps[seed_index]
        return build_archetype_template(
            self._weights,
            self._theta_hist,
            arch_map,
            eps=self._config.eps,
        )

    @property
    def round_weights(self) -> NDArray[np.float64]:
        """Current posterior weights over historical rounds. Shape (n_rounds,)."""
        return self._weights.copy()

    @property
    def top_round_match(self) -> tuple[int, float]:
        """(round_id, weight) of the best-matching historical round."""
        best_idx = int(np.argmax(self._weights))
        return self._round_ids[best_idx], float(self._weights[best_idx])
