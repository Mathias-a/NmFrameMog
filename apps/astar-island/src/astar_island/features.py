"""Feature engineering for the LightGBM prior model.

Extracts ~46 features per cell from a 40×40 initial_grid:
  - 8  terrain one-hot
  - 5  distance features
  - 15 neighbour counts (5 types × 3 radii)
  - 12 terrain fractions in windows (6 groups × 2 windows)
  - 3  archetype flags
  - 3  position features
  = 46 features total (column count depends on config.feature_radii / window_sizes)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, uniform_filter

from astar_island.config import Config
from astar_island.types import STATIC_CODES, BoolMask, Grid, H, W

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered list of terrain codes for one-hot encoding (8 codes)
_ONE_HOT_CODES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 10, 11)

# Diagonal of a 40×40 grid — used as "no cell found" sentinel
_MAX_DIST: float = float(np.sqrt(H**2 + W**2))  # ≈ 56.57

# Row/col index grids (shape H×W) — computed once at import time
_ROW_IDX: NDArray[np.float64] = np.arange(H, dtype=np.float64)[:, np.newaxis] * np.ones(
    (H, W), dtype=np.float64
)
_COL_IDX: NDArray[np.float64] = np.arange(W, dtype=np.float64)[np.newaxis, :] * np.ones(
    (H, W), dtype=np.float64
)

# ---------------------------------------------------------------------------
# Mask helpers (exported)
# ---------------------------------------------------------------------------


def dynamic_mask(initial_grid: Grid) -> BoolMask:
    """Return True for cells that are dynamic (not ocean=10, not mountain=5).

    Shape (H, W).
    """
    result: BoolMask = np.ones((H, W), dtype=np.bool_)
    for code in STATIC_CODES:
        result &= initial_grid != code
    return result


def coastal_mask(initial_grid: Grid) -> BoolMask:
    """Return True for non-ocean cells within Chebyshev distance 1 of ocean.

    Uses 8-connected adjacency (includes diagonals) to match the coastal
    definition used throughout the pipeline (prior, posterior, archetypes).

    Shape (H, W).
    """
    from scipy.ndimage import distance_transform_cdt

    ocean: NDArray[np.bool_] = initial_grid == 10
    dist_coast: NDArray[np.float64] = np.asarray(
        distance_transform_cdt(~ocean),
        dtype=np.float64,
    )
    result: BoolMask = np.asarray(dist_coast <= 1.0, dtype=np.bool_) & ~ocean
    return result


# ---------------------------------------------------------------------------
# Archetype classification (exported)
# ---------------------------------------------------------------------------


def classify_archetypes(initial_grid: Grid) -> NDArray[np.int32]:
    """Classify each cell into archetype.

    0 = coastal (adjacent to ocean, non-ocean cell)
    1 = settlement_adjacent (within Chebyshev distance 2 of a settlement)
    2 = inland_natural (neither coastal nor settlement-adjacent)

    Returns shape (H, W).
    """
    coast: BoolMask = coastal_mask(initial_grid)

    # Chebyshev distance ≤ 2 from any settlement (code 1)
    settlement_cells: NDArray[np.bool_] = initial_grid == 1
    settle_adj: NDArray[np.bool_] = np.zeros((H, W), dtype=np.bool_)
    for dh in range(-2, 3):
        for dw in range(-2, 3):
            if dh == 0 and dw == 0:
                continue
            shifted: NDArray[np.bool_] = np.roll(
                np.roll(settlement_cells, dh, axis=0), dw, axis=1
            )
            # Mask out wrap-around edges
            if dh > 0:
                shifted[:dh, :] = False
            elif dh < 0:
                shifted[dh:, :] = False
            if dw > 0:
                shifted[:, :dw] = False
            elif dw < 0:
                shifted[:, dw:] = False
            settle_adj |= shifted
    # Include the settlement cells themselves
    settle_adj |= settlement_cells

    archetypes: NDArray[np.int32] = np.full((H, W), 2, dtype=np.int32)  # inland_natural
    archetypes[settle_adj] = 1  # settlement_adjacent
    archetypes[coast] = 0  # coastal wins
    return archetypes


# ---------------------------------------------------------------------------
# Internal feature block builders
# ---------------------------------------------------------------------------


def _terrain_onehot(initial_grid: Grid) -> NDArray[np.float64]:
    """8 one-hot features for terrain code at each cell. Shape (H*W, 8)."""
    flat = initial_grid.ravel()
    result = np.zeros((H * W, len(_ONE_HOT_CODES)), dtype=np.float64)
    for i, code in enumerate(_ONE_HOT_CODES):
        result[:, i] = (flat == code).astype(np.float64)
    return result


def _distance_features(initial_grid: Grid) -> NDArray[np.float64]:
    """5 Euclidean distance features (normalised to [0,1]). Shape (H*W, 5)."""
    out = np.empty((H * W, 5), dtype=np.float64)

    def _edt_to_code(code: int) -> NDArray[np.float64]:
        mask: NDArray[np.bool_] = initial_grid == code
        if not mask.any():
            return np.full((H, W), _MAX_DIST, dtype=np.float64)
        raw = distance_transform_edt(~mask)
        dist: NDArray[np.float64] = np.asarray(raw, dtype=np.float64)
        return dist

    # 0: distance to ocean
    out[:, 0] = _edt_to_code(10).ravel() / _MAX_DIST

    # 1: distance to settlement (code 1)
    out[:, 1] = _edt_to_code(1).ravel() / _MAX_DIST

    # 2: distance to port (code 2)
    out[:, 2] = _edt_to_code(2).ravel() / _MAX_DIST

    # 3: distance to ruin (code 3)
    out[:, 3] = _edt_to_code(3).ravel() / _MAX_DIST

    # 4: distance to forest edge — forest cell adjacent to non-forest
    forest_mask: NDArray[np.bool_] = initial_grid == 4
    non_forest_mask: NDArray[np.bool_] = ~forest_mask
    forest_edge: NDArray[np.bool_] = np.zeros((H, W), dtype=np.bool_)
    forest_edge[1:, :] |= non_forest_mask[:-1, :] & forest_mask[1:, :]
    forest_edge[:-1, :] |= non_forest_mask[1:, :] & forest_mask[:-1, :]
    forest_edge[:, 1:] |= non_forest_mask[:, :-1] & forest_mask[:, 1:]
    forest_edge[:, :-1] |= non_forest_mask[:, 1:] & forest_mask[:, :-1]
    if forest_edge.any():
        edge_raw = distance_transform_edt(~forest_edge)
        edge_dist: NDArray[np.float64] = np.asarray(edge_raw, dtype=np.float64)
    else:
        edge_dist = np.full((H, W), _MAX_DIST, dtype=np.float64)
    out[:, 4] = edge_dist.ravel() / _MAX_DIST

    return out


def _neighbour_counts(
    initial_grid: Grid, radii: tuple[int, ...]
) -> NDArray[np.float64]:
    """Chebyshev-neighbour counts for 5 terrain types × len(radii) radii.

    Terrain types: forest(4), mountain(5), settlement(1), ruin(3), ocean(10).
    Returns shape (H*W, 5*len(radii)).
    """
    count_codes: tuple[int, ...] = (4, 5, 1, 3, 10)
    n_types = len(count_codes)
    n_radii = len(radii)
    out = np.empty((H * W, n_types * n_radii), dtype=np.float64)

    col = 0
    for radius in radii:
        size = 2 * radius + 1
        for code in count_codes:
            binary: NDArray[np.float64] = (initial_grid == code).astype(np.float64)
            # uniform_filter computes the mean; multiply by window area for count
            windowed: NDArray[np.float64] = uniform_filter(
                binary, size=size, mode="constant"
            )
            # Subtract self to get neighbour-only count
            neighbour_sum: NDArray[np.float64] = windowed * (size**2) - binary
            out[:, col] = neighbour_sum.ravel()
            col += 1

    return out


def _terrain_window_fractions(
    initial_grid: Grid, window_sizes: tuple[int, ...]
) -> NDArray[np.float64]:
    """Terrain fractions in local windows.

    6 terrain groups × len(window_sizes) windows.

    Groups:
      0 → empty/plains (codes 0, 11)
      1 → settlement (code 1)
      2 → port (code 2)
      3 → ruin (code 3)
      4 → forest (code 4)
      5 → mountain (code 5)

    Returns shape (H*W, 6*len(window_sizes)).
    """
    n_groups = 6
    n_windows = len(window_sizes)
    out = np.empty((H * W, n_groups * n_windows), dtype=np.float64)

    group_masks: list[NDArray[np.float64]] = [
        ((initial_grid == 0) | (initial_grid == 11)).astype(np.float64),
        (initial_grid == 1).astype(np.float64),
        (initial_grid == 2).astype(np.float64),
        (initial_grid == 3).astype(np.float64),
        (initial_grid == 4).astype(np.float64),
        (initial_grid == 5).astype(np.float64),
    ]

    col = 0
    for wsize in window_sizes:
        for gmask in group_masks:
            frac: NDArray[np.float64] = uniform_filter(
                gmask, size=wsize, mode="constant"
            )
            out[:, col] = frac.ravel()
            col += 1

    return out


def _archetype_flags(initial_grid: Grid) -> NDArray[np.float64]:
    """3 archetype flag features. Shape (H*W, 3)."""
    archetypes: NDArray[np.int32] = classify_archetypes(initial_grid)
    coastal_flag: NDArray[np.float64] = (archetypes == 0).astype(np.float64)
    settle_adj_flag: NDArray[np.float64] = (archetypes == 1).astype(np.float64)
    inland_flag: NDArray[np.float64] = (archetypes == 2).astype(np.float64)
    return np.stack(
        [coastal_flag.ravel(), settle_adj_flag.ravel(), inland_flag.ravel()], axis=1
    )


def _position_features() -> NDArray[np.float64]:
    """Position features: norm_row, norm_col, dist_from_center.

    Returns: shape (H*W, 3)."""
    center_h = (H - 1) / 2.0
    center_w = (W - 1) / 2.0
    norm_dist_max = float(np.sqrt(center_h**2 + center_w**2))

    norm_row: NDArray[np.float64] = (_ROW_IDX / (H - 1)).ravel()
    norm_col: NDArray[np.float64] = (_COL_IDX / (W - 1)).ravel()
    dist_from_center: NDArray[np.float64] = np.sqrt(
        (_ROW_IDX - center_h) ** 2 + (_COL_IDX - center_w) ** 2
    ).ravel()
    norm_dist: NDArray[np.float64] = dist_from_center / norm_dist_max

    return np.stack([norm_row, norm_col, norm_dist], axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(initial_grid: Grid, config: Config) -> NDArray[np.float64]:
    """Extract features for all cells.

    Returns shape (H*W, n_features) — 46 features with default config.
    """
    blocks: list[NDArray[np.float64]] = [
        _terrain_onehot(initial_grid),  # 8
        _distance_features(initial_grid),  # 5
        _neighbour_counts(initial_grid, config.feature_radii),  # 15
        _terrain_window_fractions(initial_grid, config.window_sizes),  # 12
        _archetype_flags(initial_grid),  # 3
        _position_features(),  # 3
    ]
    return np.concatenate(blocks, axis=1)


def feature_names(config: Config) -> list[str]:
    """Return ordered feature names matching extract_features columns."""
    names: list[str] = []

    # 8 one-hot terrain features
    for code in _ONE_HOT_CODES:
        names.append(f"terrain_code_{code}")

    # 5 distance features
    names += [
        "dist_ocean",
        "dist_settlement",
        "dist_port",
        "dist_ruin",
        "dist_forest_edge",
    ]

    # neighbour count features (5 types × len(radii))
    count_code_names: dict[int, str] = {
        4: "forest",
        5: "mountain",
        1: "settlement",
        3: "ruin",
        10: "ocean",
    }
    for radius in config.feature_radii:
        for code in (4, 5, 1, 3, 10):
            names.append(f"nbr_r{radius}_{count_code_names[code]}")

    # terrain fraction features (6 groups × len(window_sizes))
    group_names: list[str] = [
        "empty_plains",
        "settlement",
        "port",
        "ruin",
        "forest",
        "mountain",
    ]
    for wsize in config.window_sizes:
        for gname in group_names:
            names.append(f"frac_w{wsize}_{gname}")

    # 3 archetype flags
    names += ["flag_coastal", "flag_settlement_adj", "flag_inland"]

    # 3 position features
    names += ["pos_norm_row", "pos_norm_col", "pos_dist_center"]

    return names
