"""Observation feature extraction from viewport responses.

Produces summary features used by the likelihood model:
  - cell-class counts in viewport
  - alive/dead settlement counts
  - port count
  - mean and variance of settlement population, food, wealth, defense
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from astar_twin.contracts.api_models import SimulateResponse
from astar_twin.contracts.types import NUM_CLASSES, TERRAIN_TO_CLASS


@dataclass
class ObservationFeatures:
    """Summary features extracted from one viewport observation."""

    # Cell-class distribution in viewport
    class_counts: list[int] = field(default_factory=lambda: [0] * NUM_CLASSES)
    total_cells: int = 0

    # Settlement summary
    alive_count: int = 0
    dead_count: int = 0
    port_count: int = 0

    # Settlement stat summaries (mean, variance pairs)
    population_mean: float = 0.0
    population_var: float = 0.0
    food_mean: float = 0.0
    food_var: float = 0.0
    wealth_mean: float = 0.0
    wealth_var: float = 0.0
    defense_mean: float = 0.0
    defense_var: float = 0.0


def extract_features(response: SimulateResponse) -> ObservationFeatures:
    """Extract observation features from a simulate response."""
    features = ObservationFeatures()

    # Count cell classes in viewport grid
    class_counts = [0] * NUM_CLASSES
    for row in response.grid:
        for code in row:
            cls_idx = TERRAIN_TO_CLASS.get(code, 0)
            class_counts[cls_idx] += 1
    features.class_counts = class_counts
    features.total_cells = sum(class_counts)

    # Settlement statistics
    alive_settlements = [s for s in response.settlements if s.alive]
    dead_settlements = [s for s in response.settlements if not s.alive]
    features.alive_count = len(alive_settlements)
    features.dead_count = len(dead_settlements)
    features.port_count = sum(1 for s in alive_settlements if s.has_port)

    if alive_settlements:
        pops = [s.population for s in alive_settlements]
        foods = [s.food for s in alive_settlements]
        wealths = [s.wealth for s in alive_settlements]
        defenses = [s.defense for s in alive_settlements]

        features.population_mean = float(np.mean(pops))
        features.population_var = float(np.var(pops)) if len(pops) > 1 else 0.0
        features.food_mean = float(np.mean(foods))
        features.food_var = float(np.var(foods)) if len(foods) > 1 else 0.0
        features.wealth_mean = float(np.mean(wealths))
        features.wealth_var = float(np.var(wealths)) if len(wealths) > 1 else 0.0
        features.defense_mean = float(np.mean(defenses))
        features.defense_var = float(np.var(defenses)) if len(defenses) > 1 else 0.0

    return features
