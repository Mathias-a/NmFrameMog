from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.strategies.calibrated_mc.strategy import Zone

ZONE_SEQUENCE: tuple[Zone, ...] = (
    Zone.CORE,
    Zone.EXPANSION_RING,
    Zone.COASTAL_HUB,
    Zone.REMOTE_COASTAL,
    Zone.REMOTE_INLAND,
)

ZONE_NAMES: tuple[str, ...] = tuple(zone.name.lower() for zone in ZONE_SEQUENCE)

DEFAULT_ZONE_WEIGHTS: dict[str, float] = {
    "core": 0.10,
    "expansion_ring": 0.16,
    "coastal_hub": 0.20,
    "remote_coastal": 0.26,
    "remote_inland": 0.18,
}


def normalize_zone_weights(zone_weights: Mapping[str, float] | None = None) -> dict[str, float]:
    merged = dict(DEFAULT_ZONE_WEIGHTS)
    if zone_weights is not None:
        for zone_name, raw_weight in zone_weights.items():
            if zone_name not in DEFAULT_ZONE_WEIGHTS:
                raise ValueError(f"Unknown zone name: {zone_name}")
            merged[zone_name] = float(np.clip(raw_weight, 0.0, 1.0))
    return merged


def blend_predictions(
    base_prediction: NDArray[np.float64],
    fallback_prediction: NDArray[np.float64],
    zone_map: NDArray[np.int8],
    is_static: NDArray[np.bool_],
    zone_weights: Mapping[str, float] | None = None,
) -> NDArray[np.float64]:
    if base_prediction.shape != fallback_prediction.shape:
        raise ValueError(
            "base_prediction and fallback_prediction must have identical shapes, got "
            f"{base_prediction.shape} and {fallback_prediction.shape}"
        )
    if base_prediction.ndim != 3 or base_prediction.shape[2] != NUM_CLASSES:
        raise ValueError(
            f"prediction tensors must have shape (H, W, 6), got {base_prediction.shape}"
        )

    H, W, _ = base_prediction.shape
    if zone_map.shape != (H, W):
        raise ValueError(f"zone_map must have shape {(H, W)}, got {zone_map.shape}")
    if is_static.shape != (H, W):
        raise ValueError(f"is_static must have shape {(H, W)}, got {is_static.shape}")

    weights = normalize_zone_weights(zone_weights)
    blended = np.array(base_prediction, dtype=np.float64, copy=True)
    dynamic_mask = ~is_static
    fallback = np.asarray(fallback_prediction, dtype=np.float64)

    for zone in ZONE_SEQUENCE:
        zone_name = zone.name.lower()
        weight = weights[zone_name]
        zone_mask = dynamic_mask & (zone_map == int(zone))
        if not np.any(zone_mask):
            continue
        blended[zone_mask] = (1.0 - weight) * blended[zone_mask] + weight * fallback[zone_mask]

    sums: NDArray[np.float64] = np.sum(blended, axis=2, keepdims=True)
    sums = np.maximum(sums, 1e-12)
    normalized: NDArray[np.float64] = blended / sums
    return normalized
