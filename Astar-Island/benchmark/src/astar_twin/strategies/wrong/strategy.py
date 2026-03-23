from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from astar_twin.data.models import InitialState

class WrongStrategy:
    @property
    def name(self) -> str:
        return "wrong"

    def predict(self, initial_state: InitialState, budget: int, base_seed: int) -> NDArray[np.float64]:
        H = len(initial_state.grid)
        W = len(initial_state.grid[0])
        pred = np.zeros((H, W, 6), dtype=np.float64)
        pred[:, :, 5] = 1.0
        return pred
