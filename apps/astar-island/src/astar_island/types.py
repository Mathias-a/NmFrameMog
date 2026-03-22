"""Type aliases and constants for the astar_island package."""

from typing import Final

import numpy as np
from numpy.typing import NDArray

# Map dimensions
H: Final[int] = 40
W: Final[int] = 40
K: Final[int] = 6  # terrain classes
N_SEEDS: Final[int] = 5
VIEWPORT: Final[int] = 15

# Type aliases for numpy arrays with specific semantics
type Grid = NDArray[np.int32]  # (H, W) terrain codes 0-11
type ProbTensor = NDArray[np.float64]  # (H, W, K) probability distributions
type AlphaTensor = NDArray[np.float64]  # (H, W, K) Dirichlet parameters
type CountTensor = NDArray[np.int32]  # (H, W, K) observation counts
type ESSMap = NDArray[np.float64]  # (H, W) effective sample sizes
type BoolMask = NDArray[np.bool_]  # (H, W) cell masks

# Terrain code → prediction class mapping
# Ocean (10), Plains (11), Empty (0) all map to prediction class 0
CODE_TO_CLASS: Final[dict[int, int]] = {
    0: 0,  # Empty → class 0
    10: 0,  # Ocean → class 0
    11: 0,  # Plains → class 0
    1: 1,  # Settlement
    2: 2,  # Port
    3: 3,  # Ruin
    4: 4,  # Forest
    5: 5,  # Mountain
}

# Static terrain codes (excluded from scoring — zero entropy)
STATIC_CODES: Final[frozenset[int]] = frozenset({5, 10})
