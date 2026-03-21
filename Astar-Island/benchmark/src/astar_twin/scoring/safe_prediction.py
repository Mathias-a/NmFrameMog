from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def safe_prediction(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    floored = np.maximum(tensor.astype(np.float64, copy=True), 0.01)
    sums = np.sum(floored, axis=2, keepdims=True)
    return floored / sums
