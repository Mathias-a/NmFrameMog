"""Cell archetype classification and historical archetype frequency tables.

Archetypes group cells by spatial context for round-level calibration:
  0 (COASTAL):              Chebyshev distance to ocean <= 1
  1 (SETTLEMENT_ADJACENT):  distance to settlement <= 2, not coastal
  2 (INLAND_NATURAL):       everything else
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_cdt

from astar_island.types import H, K, W

# Archetype IDs
COASTAL: Final[int] = 0
SETTLEMENT_ADJACENT: Final[int] = 1
INLAND_NATURAL: Final[int] = 2
N_ARCHETYPES: Final[int] = 3


def classify_archetypes(
    initial_grid: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Assign each cell an archetype ID.

    Returns: (H, W) int32 array with values in {0, 1, 2}.
    """
    ocean_mask: NDArray[np.bool_] = np.asarray(initial_grid == 10, dtype=np.bool_)
    settlement_mask: NDArray[np.bool_] = np.asarray(
        (initial_grid == 1) | (initial_grid == 2), dtype=np.bool_,
    )

    dist_coast: NDArray[np.float64] = np.asarray(
        distance_transform_cdt(~ocean_mask), dtype=np.float64,
    )
    dist_settlement: NDArray[np.float64] = (
        np.asarray(distance_transform_cdt(~settlement_mask), dtype=np.float64)
        if bool(settlement_mask.any())
        else np.full((H, W), 99.0, dtype=np.float64)
    )

    archetypes: NDArray[np.int32] = np.full((H, W), INLAND_NATURAL, dtype=np.int32)
    archetypes[dist_coast <= 1] = COASTAL
    # Settlement-adjacent overrides only inland cells, not coastal
    settlement_adj: NDArray[np.bool_] = np.asarray(
        (dist_settlement <= 2) & (dist_coast > 1), dtype=np.bool_,
    )
    archetypes[settlement_adj] = SETTLEMENT_ADJACENT

    return archetypes


def build_historical_tables(
    historical_rounds: list[tuple[NDArray[np.int32], NDArray[np.float64]]],
) -> NDArray[np.float64]:
    """Build archetype frequency tables from historical rounds.

    Args:
        historical_rounds: list of (initial_grid, ground_truth) pairs.

    Returns:
        theta_hist: (n_rounds, N_ARCHETYPES, K) array where
        theta_hist[r, a, k] = mean ground_truth probability of class k
        across all dynamic cells of archetype a in round r.
    """
    n_rounds = len(historical_rounds)
    theta: NDArray[np.float64] = np.zeros(
        (n_rounds, N_ARCHETYPES, K), dtype=np.float64,
    )

    for r, (grid, gt) in enumerate(historical_rounds):
        archetypes = classify_archetypes(grid)
        static_mask: NDArray[np.bool_] = np.asarray(
            (grid == 10) | (grid == 5), dtype=np.bool_,
        )

        for a in range(N_ARCHETYPES):
            mask: NDArray[np.bool_] = np.asarray(
                (archetypes == a) & ~static_mask, dtype=np.bool_,
            )
            n_cells = int(np.sum(mask))
            if n_cells == 0:
                theta[r, a] = 1.0 / K
            else:
                mean_val: NDArray[np.float64] = np.mean(gt[mask], axis=0)
                theta[r, a] = mean_val

    return theta
