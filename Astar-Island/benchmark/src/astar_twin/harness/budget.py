"""Shared query-budget object passed into Strategy.predict.

A single ``Budget`` instance is created once per strategy evaluation run and
shared across *all* per-seed ``predict`` calls.  This mirrors the real
challenge, where the 50-query pool is global — spending a query on seed 0
reduces what remains for seeds 1-4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from astar_twin.contracts.api_models import SimulateResponse


@dataclass
class Budget:
    """Mutable shared query budget.

    Args:
        total: Maximum number of queries allowed across all seeds in a round
               (mirrors ``MAX_QUERIES`` = 50 in the real challenge).

    Example::

        budget = Budget(total=50)
        while budget.remaining > 0:
            budget.consume()          # costs 1 query
        budget.consume()              # raises RuntimeError — budget exhausted
    """

    total: int
    _used: int = field(default=0, init=False, repr=False)
    _query_callback: Callable[[int, int, int, int, int], SimulateResponse] | None = field(
        default=None, repr=False
    )

    @property
    def remaining(self) -> int:
        """Number of queries still available."""
        return self.total - self._used

    @property
    def used(self) -> int:
        """Number of queries already spent."""
        return self._used

    def consume(self, n: int = 1) -> None:
        """Mark *n* queries as spent.

        Raises:
            RuntimeError: if consuming *n* queries would exceed the total budget.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if self._used + n > self.total:
            raise RuntimeError(
                f"Query budget exhausted: tried to consume {n} but only "
                f"{self.remaining} remain ({self._used}/{self.total} used)"
            )
        self._used += n

    def observe(
        self,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> SimulateResponse:
        """Spend 1 query to observe a viewport on the real/twin simulator.

        Raises:
            RuntimeError: If the budget is exhausted.
            NotImplementedError: If this budget instance does not support active sensing.
        """
        if self._query_callback is None:
            raise NotImplementedError(
                "Active sensing is not supported by this budget instance. "
                "Ensure you run via BenchmarkRunner or a capable harness."
            )
        self.consume(1)
        return self._query_callback(seed_index, viewport_x, viewport_y, viewport_w, viewport_h)

    def __int__(self) -> int:
        """Return remaining queries as an int for backwards-compatible usage."""
        return self.remaining

    def __repr__(self) -> str:
        return f"Budget(used={self._used}, total={self.total}, remaining={self.remaining})"
