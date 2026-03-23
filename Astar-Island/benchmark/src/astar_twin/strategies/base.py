from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState


@runtime_checkable
class Strategy(Protocol):
    """Protocol that every prediction strategy must satisfy.

    Rules:
    - Output shape MUST be (H, W, 6) with H, W from initial_state.grid.
    - Output dtype MUST be np.float64.
    - Each cell's 6 probabilities MUST sum to 1.0 (the harness applies
      safe_prediction() before scoring — do NOT floor zeros yourself).
    - Same base_seed MUST produce identical output (determinism required).
    - DO NOT import from astar_twin.phases, astar_twin.api, or astar_twin.data.
    - DO NOT modify SimulationParams default field values.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier shown in benchmark reports."""
        ...

    def predict(
        self,
        initial_state: InitialState,
        budget: int,
        base_seed: int,
    ) -> NDArray[np.float64]:
        """Produce an H×W×6 probability tensor for the given initial state.

        Args:
            initial_state: The initial terrain grid and settlements for one seed.
            budget: Total query budget available for the full round (metadata
                    only — strategies do NOT allocate API queries here).
            base_seed: RNG seed for reproducibility. Same seed → identical output.

        Returns:
            Float64 array of shape (H, W, 6). The harness calls safe_prediction()
            before scoring, so raw probabilities are fine — just keep them ≥ 0.
        """
        ...
