"""Per-cell feature extraction for spatial grids.

Extracts a feature vector per cell with these groups:
  - One-hot class encoding (6)
  - Neighbor counts per class in 3x3 window (6)
  - Distance to nearest cell of each class (6)
  - Local class entropy in 5x5 window (1)
  - Positional features: x, y, dist_center, dist_edge (4)
  - Global grid statistics: class fractions + grid entropy (8)
  - Connected-component features per class: count, mean_size, max_size (18)
  - Boundary features: is_boundary, boundary_fraction_3x3, unique_neighbors (3)
  - Patch shape features: local_same_fraction, same_terrain_component_size (2)
  - Raw terrain: is_ocean, is_plains, ocean_neighbor_frac, plains_neighbor_frac (4)
  - Larger-radius neighbor counts: per-class count within radius 7 (6)
Total: 64 dimensions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve, distance_transform_edt, label

from astar_island.terrain import (
    GRID_SIZE,
    NUM_PREDICTION_CLASSES,
    TERRAIN_CODE_TO_CLASS,
)

_NEIGHBOR_KERNEL = np.ones((3, 3), dtype=np.float64)
_NEIGHBOR_KERNEL[1, 1] = 0.0


def _one_hot(grid: NDArray[np.int32]) -> NDArray[np.float64]:
    result = np.zeros(
        (grid.shape[0], grid.shape[1], NUM_PREDICTION_CLASSES), dtype=np.float64
    )
    for k in range(NUM_PREDICTION_CLASSES):
        result[:, :, k] = (grid == k).astype(np.float64)
    return result


def _neighbor_counts(one_hot: NDArray[np.float64]) -> NDArray[np.float64]:
    counts = np.zeros_like(one_hot)
    for k in range(NUM_PREDICTION_CLASSES):
        counts[:, :, k] = convolve(
            one_hot[:, :, k], _NEIGHBOR_KERNEL, mode="constant", cval=0.0
        )
    return counts


def _distance_transforms(one_hot: NDArray[np.float64]) -> NDArray[np.float64]:
    h, w, c = one_hot.shape
    distances = np.zeros((h, w, c), dtype=np.float64)
    for k in range(c):
        mask = one_hot[:, :, k] > 0.5
        if mask.any():
            distances[:, :, k] = distance_transform_edt(~mask)
        else:
            distances[:, :, k] = float(GRID_SIZE)
    return distances / GRID_SIZE


def _local_entropy(
    one_hot: NDArray[np.float64],
    window: int = 5,
) -> NDArray[np.float64]:
    kernel = np.ones((window, window), dtype=np.float64) / (window * window)
    h, w, c = one_hot.shape
    local_probs = np.zeros((h, w, c), dtype=np.float64)
    for k in range(c):
        local_probs[:, :, k] = convolve(
            one_hot[:, :, k], kernel, mode="constant", cval=0.0
        )

    safe = np.where(local_probs > 0, local_probs, 1.0)
    terms = np.where(local_probs > 0, -local_probs * np.log(safe), 0.0)
    return terms.sum(axis=-1)


def _positional_features(h: int = GRID_SIZE, w: int = GRID_SIZE) -> NDArray[np.float64]:
    xs = np.arange(h, dtype=np.float64) / max(h - 1, 1)
    ys = np.arange(w, dtype=np.float64) / max(w - 1, 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    dist_center = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    dist_edge = np.minimum(np.minimum(xx, 1 - xx), np.minimum(yy, 1 - yy))

    return np.stack([xx, yy, dist_center, dist_edge], axis=-1)


def _global_features(one_hot: NDArray[np.float64]) -> NDArray[np.float64]:
    total = one_hot.shape[0] * one_hot.shape[1]
    class_fractions = one_hot.sum(axis=(0, 1)) / total

    ocean_fraction = class_fractions[0]
    safe = class_fractions[class_fractions > 0]
    grid_entropy = float(-np.sum(safe * np.log(safe)))

    return np.concatenate([class_fractions, [ocean_fraction, grid_entropy]])


def _connected_component_features(
    one_hot: NDArray[np.float64],
) -> tuple[NDArray[np.float64], list[NDArray[np.int32]]]:
    """Per-class connected-component statistics broadcast to every cell.

    Returns:
        features: (H, W, 18) — for each class: num_components, mean_size, max_size
            (all normalized by total cells).
        labeled_per_class: list of 6 (H, W) int32 arrays with component labels per class.
    """
    h, w, c = one_hot.shape
    total = h * w
    features = np.zeros((h, w, c * 3), dtype=np.float64)
    labeled_per_class: list[NDArray[np.int32]] = []

    for k in range(c):
        mask = one_hot[:, :, k] > 0.5
        if not mask.any():
            labeled_per_class.append(np.zeros((h, w), dtype=np.int32))
            continue
        labeled, num_components = label(mask)
        labeled_per_class.append(labeled.astype(np.int32))
        sizes = np.bincount(labeled.ravel())[1:]

        features[:, :, k * 3] = num_components / total
        features[:, :, k * 3 + 1] = sizes.mean() / total if len(sizes) > 0 else 0.0
        features[:, :, k * 3 + 2] = sizes.max() / total if len(sizes) > 0 else 0.0

    return features, labeled_per_class


def _boundary_features(
    class_grid: NDArray[np.int32],
    one_hot: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Boundary/edge features per cell.

    Returns (H, W, 3):
      - is_boundary: 1 if any 4-connected neighbor has different class
      - boundary_fraction_3x3: fraction of 8-neighbors with different class
      - unique_neighbor_classes: number of distinct classes in 3x3 window / NUM_CLASSES
    """
    h, w = class_grid.shape
    result = np.zeros((h, w, 3), dtype=np.float64)

    # Pad with -1 sentinel so edge cells have no false matches
    padded = np.pad(class_grid, pad_width=1, mode="constant", constant_values=-1)

    is_boundary = np.zeros((h, w), dtype=np.float64)
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor = padded[1 + di : h + 1 + di, 1 + dj : w + 1 + dj]
        valid = neighbor != -1
        is_boundary = np.maximum(
            is_boundary,
            (valid & (neighbor != class_grid)).astype(np.float64),
        )
    result[:, :, 0] = is_boundary

    differ_count = np.zeros((h, w), dtype=np.float64)
    neighbor_count = np.zeros((h, w), dtype=np.float64)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            neighbor = padded[1 + di : h + 1 + di, 1 + dj : w + 1 + dj]
            valid = neighbor != -1
            neighbor_count += valid.astype(np.float64)
            differ_count += (valid & (neighbor != class_grid)).astype(np.float64)
    safe_neighbor_count = np.maximum(neighbor_count, 1.0)
    result[:, :, 1] = differ_count / safe_neighbor_count

    unique_counts = np.zeros((h, w), dtype=np.float64)
    for k in range(NUM_PREDICTION_CLASSES):
        class_present = (one_hot[:, :, k] > 0.5).astype(np.float64)
        kernel = np.ones((3, 3), dtype=np.float64)
        has_class_nearby = convolve(class_present, kernel, mode="constant", cval=0.0)
        unique_counts += (has_class_nearby > 0).astype(np.float64)
    result[:, :, 2] = unique_counts / NUM_PREDICTION_CLASSES

    return result


def _patch_shape_features(
    class_grid: NDArray[np.int32],
    one_hot: NDArray[np.float64],
    labeled_per_class: list[NDArray[np.int32]],
) -> NDArray[np.float64]:
    """Patch-level shape features per cell.

    Returns (H, W, 2):
      - local_same_fraction: fraction of cells in 5x5 window with same class
      - component_size_normalized: size of connected component this cell belongs to / total
    """
    h, w = class_grid.shape
    total = h * w
    result = np.zeros((h, w, 2), dtype=np.float64)

    kernel_5x5 = np.ones((5, 5), dtype=np.float64) / 25.0
    for k in range(NUM_PREDICTION_CLASSES):
        mask = one_hot[:, :, k]
        local_frac = convolve(mask, kernel_5x5, mode="constant", cval=0.0)
        cells = class_grid == k
        result[cells, 0] = local_frac[cells]

    for k in range(NUM_PREDICTION_CLASSES):
        labeled = labeled_per_class[k]
        if not labeled.any():
            continue
        sizes = np.bincount(labeled.ravel())
        cell_component_sizes = sizes[labeled]
        cells = class_grid == k
        result[cells, 1] = cell_component_sizes[cells] / total

    return result


def _raw_terrain_features(raw_arr: NDArray[np.int32]) -> NDArray[np.float64]:
    h, w = raw_arr.shape
    result = np.zeros((h, w, 4), dtype=np.float64)

    is_ocean = (raw_arr == 10).astype(np.float64)
    is_plains = (raw_arr == 11).astype(np.float64)
    result[:, :, 0] = is_ocean
    result[:, :, 1] = is_plains

    result[:, :, 2] = (
        convolve(is_ocean, _NEIGHBOR_KERNEL, mode="constant", cval=0.0) / 8.0
    )
    result[:, :, 3] = (
        convolve(is_plains, _NEIGHBOR_KERNEL, mode="constant", cval=0.0) / 8.0
    )

    return result


def _large_radius_neighbor_counts(
    one_hot: NDArray[np.float64], radius: int = 7
) -> NDArray[np.float64]:
    h, w, k = one_hot.shape
    y_idx, x_idx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    circle_mask = (x_idx**2 + y_idx**2) <= radius**2
    kernel = circle_mask.astype(np.float64)
    kernel[radius, radius] = 0.0
    kernel_sum = max(kernel.sum(), 1.0)

    result = np.zeros((h, w, k), dtype=np.float64)
    for c in range(k):
        result[:, :, c] = (
            convolve(one_hot[:, :, c], kernel, mode="constant", cval=0.0) / kernel_sum
        )

    return result


def raw_grid_to_class_grid(raw_grid: list[list[int]]) -> NDArray[np.int32]:
    arr = np.array(raw_grid, dtype=np.int32)
    valid_codes = set(TERRAIN_CODE_TO_CLASS.keys())
    unique_codes = set(int(v) for v in np.unique(arr))
    unknown = unique_codes - valid_codes
    if unknown:
        msg = f"Unknown terrain codes: {sorted(unknown)}. Valid: {sorted(valid_codes)}"
        raise ValueError(msg)
    mapped = np.vectorize(TERRAIN_CODE_TO_CLASS.__getitem__)(arr)
    return mapped.astype(np.int32)


def extract_cell_features(
    class_grid: NDArray[np.int32],
    raw_grid: list[list[int]] | None = None,
) -> NDArray[np.float64]:
    h, w = class_grid.shape
    oh = _one_hot(class_grid)
    nbr = _neighbor_counts(oh) / 8.0
    dist = _distance_transforms(oh)
    ent = _local_entropy(oh)
    pos = _positional_features(h, w)
    glob = _global_features(oh)
    glob_broadcast = np.tile(glob, (h, w, 1))
    cc_feats, labeled_per_class = _connected_component_features(oh)
    boundary = _boundary_features(class_grid, oh)
    patch = _patch_shape_features(class_grid, oh, labeled_per_class)

    parts = [
        oh,  # 6: one-hot
        nbr,  # 6: neighbor counts
        dist,  # 6: distance transforms
        ent[:, :, np.newaxis],  # 1: local entropy
        pos,  # 4: positional
        glob_broadcast,  # 8: global stats
        cc_feats,  # 18: connected components
        boundary,  # 3: boundary features
        patch,  # 2: patch shape
    ]

    if raw_grid is not None:
        raw_arr = np.array(raw_grid, dtype=np.int32)
        parts.append(_raw_terrain_features(raw_arr))  # 4: ocean/plains
        parts.append(_large_radius_neighbor_counts(oh, 7))  # 6: radius-7 counts

    features = np.concatenate(parts, axis=-1)
    return features.reshape(h * w, -1)


def extract_cell_targets(ground_truth: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flatten ground truth tensor to per-cell target matrix.

    Args:
        ground_truth: (H, W, 6) probability tensor.

    Returns:
        (H*W, 6) target matrix.
    """
    return ground_truth.reshape(-1, NUM_PREDICTION_CLASSES)
