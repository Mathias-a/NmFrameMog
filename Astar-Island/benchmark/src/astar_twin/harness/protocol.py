from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.harness.budget import Budget


class Strategy(Protocol):
    """Protocol that all benchmark strategies must satisfy.

    Implementations must be deterministic: identical ``initial_state``,
    ``budget``, and ``base_seed`` inputs must always produce the same output.

    The harness applies ``safe_prediction()`` before scoring, so strategies
    must NOT floor probabilities themselves.

    Forbidden imports inside strategy implementations (AGENTS.md):
      - astar_twin.phases
      - astar_twin.api
      - astar_twin.data
    """

    @property
    def name(self) -> str:
        """Human-readable strategy identifier used in reports."""
        ...

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        """Return a probability tensor of shape ``(H, W, 6)``.

        Args:
            initial_state: The starting grid and settlements for this map.
                           ``H = len(initial_state.grid)``,
                           ``W = len(initial_state.grid[0])``.
            budget:        Shared query budget for this round.  The same
                           ``Budget`` instance is passed into *every* per-seed
                           ``predict`` call, so spending a query on seed 0
                           reduces what is available for seeds 1-4.  This
                           mirrors the real challenge where 50 queries are
                           shared across all 5 seeds.  Use
                           ``budget.remaining`` to check availability and
                           ``budget.consume()`` to spend a query.
            base_seed:     Integer seed that must make the prediction fully
                           reproducible — same seed → same output.

        Returns:
            Float64 array of shape ``(H, W, 6)`` where axis-2 holds class
            probabilities.  Values must be non-negative; they need not sum to
            exactly 1.0 (``safe_prediction`` normalises them).
        """
        ...
