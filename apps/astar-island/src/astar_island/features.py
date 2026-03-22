"""Per-cell feature extraction from initial grids.

All features are computed from the initial_grid (40x40 terrain codes).
No absolute position (x, y) features — only relative/structural features.
All computation is vectorized with NumPy/SciPy — no Python loops over cells.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, uniform_filter

from astar_island.prob import (
    NUM_CLASSES,
    initial_grid_to_classes,
    make_coastal_mask,
    make_mountain_mask,
    make_ocean_mask,
    make_static_mask,
)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(initial_grid: NDArray[np.int_]) -> NDArray[np.float64]:
    """Extract per-cell features from the initial grid.

    Returns (H, W, F) feature tensor. All features are structural/relational,
    never absolute position.

    Feature groups (20):
        1. is_settlement — binary, initial terrain code 1
        2. is_port — binary, initial terrain code 2
        3. terrain one-hot — 6 indicators for prediction classes
        4. distance to nearest settlement — normalized EDT
        5. coastal adjacency — binary, is cell adjacent to ocean
        6. 3x3 neighbor class counts — 6 features (uniform_filter)
        7. 5x5 neighbor class counts — 6 features
        8. 7x7 neighbor class counts — 6 features
        9. distance to nearest ocean cell — normalized EDT
       10. distance to nearest forest cell — normalized EDT
       10b. is_plains — binary, terrain code 11
       10c. distance to nearest plains cell — normalized EDT
       10d. plains neighbor frequency at 3x3, 5x5, 7x7 — 3 features
       11. settlement count in 7x7 neighborhood
       12. terrain diversity in 5x5 neighborhood
       13. distance to grid edge — normalized min distance
       14. is_static — binary
       15. mountain neighbor count in 3x3
    """
    h_raw, w_raw = initial_grid.shape
    h: int = int(h_raw)
    w: int = int(w_raw)
    max_dim: float = float(h if h > w else w)
    classes = initial_grid_to_classes(initial_grid)

    features: list[NDArray[np.float64]] = []

    # --- 1. is_settlement ---
    is_settle: NDArray[np.bool_] = np.asarray(np.equal(initial_grid, 1), dtype=np.bool_)
    features.append(np.asarray(is_settle, dtype=np.float64))

    # --- 2. is_port ---
    is_port: NDArray[np.bool_] = np.asarray(np.equal(initial_grid, 2), dtype=np.bool_)
    features.append(np.asarray(is_port, dtype=np.float64))

    # --- 3. terrain one-hot (6 features) ---
    for c in range(NUM_CLASSES):
        class_mask: NDArray[np.bool_] = np.asarray(np.equal(classes, c), dtype=np.bool_)
        features.append(np.asarray(class_mask, dtype=np.float64))

    # --- 4. distance to nearest settlement ---
    settle_1: NDArray[np.bool_] = np.asarray(np.equal(initial_grid, 1), dtype=np.bool_)
    settle_2: NDArray[np.bool_] = np.asarray(np.equal(initial_grid, 2), dtype=np.bool_)
    settlement_mask: NDArray[np.bool_] = np.asarray(settle_1 | settle_2, dtype=np.bool_)
    if settlement_mask.any():
        _dt_raw: object = distance_transform_edt(~settlement_mask)
        assert isinstance(_dt_raw, np.ndarray)
        dt: NDArray[np.float64] = np.asarray(_dt_raw, dtype=np.float64)
        features.append(dt / max_dim)
    else:
        features.append(np.full((h, w), 1.0, dtype=np.float64))

    # --- 5. coastal adjacency ---
    coastal = make_coastal_mask(initial_grid)
    features.append(np.asarray(coastal, dtype=np.float64))

    # --- 6/7/8. neighbor class frequencies at 3x3, 5x5, 7x7 ---
    for size in [3, 5, 7]:
        for c in range(NUM_CLASSES):
            class_mask_c: NDArray[np.bool_] = np.asarray(
                np.equal(classes, c), dtype=np.bool_
            )
            indicator: NDArray[np.float64] = np.asarray(class_mask_c, dtype=np.float64)
            _uf_raw: object = uniform_filter(
                indicator, size=size, mode="constant", cval=0.0
            )
            assert isinstance(_uf_raw, np.ndarray)
            freq: NDArray[np.float64] = np.asarray(_uf_raw, dtype=np.float64)
            features.append(freq)

    # --- 9. distance to nearest ocean ---
    ocean = make_ocean_mask(initial_grid)
    if ocean.any():
        _dt_ocean_raw: object = distance_transform_edt(~ocean)
        assert isinstance(_dt_ocean_raw, np.ndarray)
        dt_ocean: NDArray[np.float64] = np.asarray(_dt_ocean_raw, dtype=np.float64)
        features.append(dt_ocean / max_dim)
    else:
        features.append(np.full((h, w), 1.0, dtype=np.float64))

    # --- 10. distance to nearest forest ---
    forest: NDArray[np.bool_] = np.asarray(np.equal(initial_grid, 4), dtype=np.bool_)
    if forest.any():
        _dt_forest_raw: object = distance_transform_edt(~forest)
        assert isinstance(_dt_forest_raw, np.ndarray)
        dt_forest: NDArray[np.float64] = np.asarray(_dt_forest_raw, dtype=np.float64)
        features.append(dt_forest / max_dim)
    else:
        features.append(np.full((h, w), 1.0, dtype=np.float64))

    # --- 10b. is_plains ---
    plains: NDArray[np.bool_] = np.asarray(np.equal(initial_grid, 11), dtype=np.bool_)
    features.append(np.asarray(plains, dtype=np.float64))

    # --- 10c. distance to nearest plains ---
    if plains.any():
        _dt_plains_raw: object = distance_transform_edt(~plains)
        assert isinstance(_dt_plains_raw, np.ndarray)
        dt_plains: NDArray[np.float64] = np.asarray(_dt_plains_raw, dtype=np.float64)
        features.append(dt_plains / max_dim)
    else:
        features.append(np.full((h, w), 1.0, dtype=np.float64))

    # --- 10d. plains neighbor frequency at 3x3, 5x5, 7x7 ---
    plains_indicator: NDArray[np.float64] = np.asarray(plains, dtype=np.float64)
    for size in [3, 5, 7]:
        _pf_raw: object = uniform_filter(
            plains_indicator, size=size, mode="constant", cval=0.0
        )
        assert isinstance(_pf_raw, np.ndarray)
        plains_freq: NDArray[np.float64] = np.asarray(_pf_raw, dtype=np.float64)
        features.append(plains_freq)

    # --- 11. settlement count in 7x7 ---
    settle_f: NDArray[np.float64] = np.asarray(settlement_mask, dtype=np.float64)
    _sc_raw: object = uniform_filter(settle_f, size=7, mode="constant", cval=0.0)
    assert isinstance(_sc_raw, np.ndarray)
    settle_count: NDArray[np.float64] = np.asarray(_sc_raw, dtype=np.float64)
    # uniform_filter gives mean; multiply by 49 to get approximate count
    features.append(settle_count * 49.0)

    # --- 12. terrain diversity in 5x5 ---
    # Count number of distinct classes present in 5x5 neighborhood
    # Approximation: sum of (class_present_fraction > 0) indicators
    diversity = np.zeros((h, w), dtype=np.float64)
    for c in range(NUM_CLASSES):
        class_mask_d: NDArray[np.bool_] = np.asarray(
            np.equal(classes, c), dtype=np.bool_
        )
        indicator_d: NDArray[np.float64] = np.asarray(class_mask_d, dtype=np.float64)
        _pres_raw: object = uniform_filter(
            indicator_d, size=5, mode="constant", cval=0.0
        )
        assert isinstance(_pres_raw, np.ndarray)
        present: NDArray[np.float64] = np.asarray(_pres_raw, dtype=np.float64)
        present_bool: NDArray[np.bool_] = np.asarray(present > 0, dtype=np.bool_)
        diversity += np.asarray(present_bool, dtype=np.float64)
    features.append(diversity / NUM_CLASSES)

    # --- 13. distance to grid edge ---
    grid_idx: NDArray[np.float64] = np.asarray(np.mgrid[0:h, 0:w], dtype=np.float64)
    yy: NDArray[np.float64] = grid_idx[0]
    xx: NDArray[np.float64] = grid_idx[1]
    yy_inv: NDArray[np.float64] = float(h - 1) - yy
    xx_inv: NDArray[np.float64] = float(w - 1) - xx
    min_y: NDArray[np.float64] = np.asarray(np.minimum(yy, yy_inv), dtype=np.float64)
    min_x: NDArray[np.float64] = np.asarray(np.minimum(xx, xx_inv), dtype=np.float64)
    dist_to_edge: NDArray[np.float64] = np.asarray(
        np.minimum(min_y, min_x), dtype=np.float64
    )
    features.append(dist_to_edge / max_dim)

    # --- 14. is_static ---
    static = make_static_mask(initial_grid)
    features.append(np.asarray(static, dtype=np.float64))

    # --- 15. mountain neighbor count in 3x3 ---
    mountain = make_mountain_mask(initial_grid)
    mtn_f: NDArray[np.float64] = np.asarray(mountain, dtype=np.float64)
    _mtn_raw: object = uniform_filter(mtn_f, size=3, mode="constant", cval=0.0)
    assert isinstance(_mtn_raw, np.ndarray)
    mtn_neighbors: NDArray[np.float64] = np.asarray(_mtn_raw, dtype=np.float64)
    features.append(mtn_neighbors * 9.0)

    return np.stack(features, axis=-1)


def feature_count() -> int:
    """Return the number of features produced by extract_features.

    2 (is_settlement, is_port) + 6 (one-hot) + 1 (dist_settlement)
    + 1 (coastal) + 18 (3 sizes x 6 classes) + 1 (dist_ocean) + 1 (dist_forest)
    + 1 (is_plains) + 1 (dist_plains) + 3 (plains_freq 3/5/7)
    + 1 (settle_count) + 1 (diversity) + 1 (dist_edge) + 1 (is_static)
    + 1 (mtn_neighbors) = 40
    """
    return 40
