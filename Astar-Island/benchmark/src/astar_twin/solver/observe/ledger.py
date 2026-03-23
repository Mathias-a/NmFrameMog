"""Round-scoped observation ledger for viewport evidence.

Stores every viewport observation and maintains two accumulators:
  - Per-seed observed-cell class counts (H×W×6) + visit counts (H×W)
    for local empirical correction during final prediction.
  - Pooled round-level summary statistics aggregated from
    ObservationFeatures across all seeds/queries for cross-seed
    hidden-parameter calibration.

Usage:
    ledger = create_ledger(n_seeds=5, height=40, width=40)
    # After each query response:
    obs = ViewportObservation(...)
    record_observation(ledger, obs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import SimSettlement
from astar_twin.contracts.types import NUM_CLASSES, TERRAIN_TO_CLASS
from astar_twin.solver.observe.features import ObservationFeatures


@dataclass
class ViewportObservation:
    """Full record of one viewport query result."""

    seed_index: int
    phase: str  # "bootstrap" | "adaptive" | "reserve"
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    grid: list[list[int]]
    settlements: list[SimSettlement]
    features: ObservationFeatures
    utility_score: float = 0.0
    ess_after: float = 0.0


@dataclass
class PooledObservationStats:
    """Weighted summary statistics pooled across all seeds and queries.

    Cell-level features are weighted by total_cells.
    Settlement-level features are weighted by alive_count.
    Only summary stats should transfer across seeds, never raw cell data.
    """

    class_counts: list[float] = field(default_factory=lambda: [0.0] * NUM_CLASSES)
    total_cells: float = 0.0

    alive_weight: float = 0.0
    alive_count_sum: float = 0.0
    dead_count_sum: float = 0.0
    port_count_sum: float = 0.0

    # Weighted sums (divide by alive_weight to get weighted mean)
    population_mean_wsum: float = 0.0
    food_mean_wsum: float = 0.0
    wealth_mean_wsum: float = 0.0
    defense_mean_wsum: float = 0.0
    prosperity_proxy_mean_wsum: float = 0.0

    @property
    def n_observations(self) -> int:
        """Number of observations that contributed to pooled stats."""
        return int(self.total_cells > 0) if self.total_cells > 0 else 0

    def weighted_population_mean(self) -> float:
        """Weighted mean of settlement population across all observations."""
        return self.population_mean_wsum / self.alive_weight if self.alive_weight > 0 else 0.0

    def weighted_food_mean(self) -> float:
        """Weighted mean of settlement food across all observations."""
        return self.food_mean_wsum / self.alive_weight if self.alive_weight > 0 else 0.0

    def weighted_wealth_mean(self) -> float:
        """Weighted mean of settlement wealth across all observations."""
        return self.wealth_mean_wsum / self.alive_weight if self.alive_weight > 0 else 0.0

    def weighted_defense_mean(self) -> float:
        """Weighted mean of settlement defense across all observations."""
        return self.defense_mean_wsum / self.alive_weight if self.alive_weight > 0 else 0.0

    def weighted_prosperity_proxy_mean(self) -> float:
        """Weighted mean of prosperity proxy across all observations."""
        return self.prosperity_proxy_mean_wsum / self.alive_weight if self.alive_weight > 0 else 0.0


@dataclass
class ObservationLedger:
    """Round-scoped observation store with per-seed and pooled accumulators."""

    observations: list[ViewportObservation] = field(default_factory=list)

    # Per-seed observed-cell accumulators (seed_index → array)
    per_seed_class_counts: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    per_seed_visit_counts: dict[int, NDArray[np.float64]] = field(default_factory=dict)

    # Round-level pooled summary stats
    pooled_stats: PooledObservationStats = field(default_factory=PooledObservationStats)

    @property
    def n_observations(self) -> int:
        """Total number of recorded observations."""
        return len(self.observations)

    def observations_for_seed(self, seed_index: int) -> list[ViewportObservation]:
        """Return all observations for a specific seed."""
        return [obs for obs in self.observations if obs.seed_index == seed_index]

    def mean_visit_count_in_window(
        self,
        seed_index: int,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> float:
        """Mean visit count within a viewport window for a specific seed."""
        visits = self.per_seed_visit_counts.get(seed_index)
        if visits is None:
            return 0.0
        map_h = int(visits.shape[0])
        map_w = int(visits.shape[1])
        y_end = min(y + h, map_h)
        x_end = min(x + w, map_w)
        y_start = max(0, y)
        x_start = max(0, x)
        if y_end <= y_start or x_end <= x_start:
            return 0.0
        window = cast(NDArray[np.float64], visits[y_start:y_end, x_start:x_end])
        return float(cast(np.float64, np.mean(window)))


def create_ledger(
    n_seeds: int,
    height: int,
    width: int,
) -> ObservationLedger:
    """Create a fresh observation ledger with zero-initialized accumulators.

    Args:
        n_seeds: Number of seeds in the round.
        height: Map height.
        width: Map width.

    Returns:
        Empty ObservationLedger ready to record observations.
    """
    ledger = ObservationLedger()
    for seed_idx in range(n_seeds):
        ledger.per_seed_class_counts[seed_idx] = np.zeros(
            (height, width, NUM_CLASSES), dtype=np.float64
        )
        ledger.per_seed_visit_counts[seed_idx] = np.zeros((height, width), dtype=np.float64)
    return ledger


def record_observation(
    ledger: ObservationLedger,
    observation: ViewportObservation,
) -> None:
    """Record a viewport observation and update all accumulators.

    Updates:
      - observations list
      - per-seed cell class counts and visit counts (using actual viewport bounds)
      - pooled round-level summary statistics

    Args:
        ledger: The round-scoped ledger to update.
        observation: The viewport observation to record.
    """
    ledger.observations.append(observation)

    # --- Per-seed cell accumulator ---
    seed_counts = ledger.per_seed_class_counts.get(observation.seed_index)
    seed_visits = ledger.per_seed_visit_counts.get(observation.seed_index)

    if seed_counts is not None and seed_visits is not None:
        max_y = int(seed_counts.shape[0])
        max_x = int(seed_counts.shape[1])
        for local_y, row in enumerate(observation.grid):
            for local_x, code in enumerate(row):
                abs_y = observation.viewport_y + local_y
                abs_x = observation.viewport_x + local_x
                # Bounds check against accumulator shape
                if 0 <= abs_y < max_y and 0 <= abs_x < max_x:
                    cls_idx = TERRAIN_TO_CLASS.get(code, 0)
                    seed_counts[abs_y, abs_x, cls_idx] = (
                        float(cast(np.float64, seed_counts[abs_y, abs_x, cls_idx])) + 1.0
                    )
                    seed_visits[abs_y, abs_x] = (
                        float(cast(np.float64, seed_visits[abs_y, abs_x])) + 1.0
                    )

    # --- Pooled round-level stats ---
    f = observation.features
    pooled = ledger.pooled_stats

    pooled.total_cells += f.total_cells
    for i in range(NUM_CLASSES):
        pooled.class_counts[i] += f.class_counts[i]

    alive_weight = float(f.alive_count)
    pooled.alive_weight += alive_weight
    pooled.alive_count_sum += f.alive_count
    pooled.dead_count_sum += f.dead_count
    pooled.port_count_sum += f.port_count

    # Settlement stat weighted sums (weight by alive count)
    pooled.population_mean_wsum += alive_weight * f.population_mean
    pooled.food_mean_wsum += alive_weight * f.food_mean
    pooled.wealth_mean_wsum += alive_weight * f.wealth_mean
    pooled.defense_mean_wsum += alive_weight * f.defense_mean
    pooled.prosperity_proxy_mean_wsum += alive_weight * f.prosperity_proxy_mean
