from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def safe_prediction(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    result = tensor.astype(np.float64, copy=True)
    sums = np.sum(result, axis=2, keepdims=True)
    zero_mask = sums <= 0.0
    if np.any(zero_mask):
        result[zero_mask.repeat(result.shape[2], axis=2)] = 1.0 / result.shape[2]

    for _ in range(10):
        result = np.maximum(result, 0.01)
        sums = np.sum(result, axis=2, keepdims=True)
        result = result / sums
        if float(np.min(result)) >= 0.01 - 1e-9:
            break

    return result
