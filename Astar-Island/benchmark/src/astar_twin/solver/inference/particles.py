"""Particle schema, priors, and v1 parameter subset for inference.

A particle is a subset of SimulationParams (17 inferred fields) plus a log-weight.
All other params are frozen at SimulationParams() defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator, default_rng

from astar_twin.params.simulation_params import (
    AdjacencyMode,
    DistanceMetric,
    SimulationParams,
    UpdateOrderMode,
)

# The extended parameter subset inferred by the particle filter (28 fields)
INFERRED_PARAMS: list[str] = [
    "adjacency_mode",
    "distance_metric",
    "update_order_mode",
    "prosperity_threshold_port",
    "prosperity_threshold_longship",
    "prosperity_threshold_expand",
    "expansion_rate",
    "expansion_radius",
    "raid_base_prob",
    "raid_success_scale",
    "raid_range_base",
    "raid_range_longship_bonus",
    "raid_damage_frac",
    "raid_capture_threshold",
    "trade_range",
    "trade_value_scale",
    "winter_severity_mean",
    "winter_severity_concentration",
    "winter_severity_autocorr",
    "winter_food_loss_per_population",
    "collapse_threshold",
    "collapse_softness",
    "collapse_food_floor",
    "collapse_raid_stress_weight",
    "collapse_winter_severity_weight",
    "reclaim_threshold",
    "reclaim_rate",
    "ruin_forest_rate",
]

# Default values from SimulationParams() for computing perturbation ranges
_DEFAULTS = SimulationParams()

# Enum choices
_ADJACENCY_CHOICES = list(AdjacencyMode)
_DISTANCE_CHOICES = list(DistanceMetric)
_UPDATE_ORDER_CHOICES = list(UpdateOrderMode)

# Continuous parameter ranges (name -> (min, max, default))
_CONTINUOUS_RANGES: dict[str, tuple[float, float, float]] = {
    "prosperity_threshold_port": (0.5, 3.0, _DEFAULTS.prosperity_threshold_port),
    "prosperity_threshold_longship": (1.0, 4.0, _DEFAULTS.prosperity_threshold_longship),
    "prosperity_threshold_expand": (0.5, 3.0, _DEFAULTS.prosperity_threshold_expand),
    "expansion_rate": (0.05, 0.50, _DEFAULTS.expansion_rate),
    "expansion_radius": (1, 6, _DEFAULTS.expansion_radius),
    "raid_base_prob": (0.01, 0.20, _DEFAULTS.raid_base_prob),
    "raid_success_scale": (0.3, 1.5, _DEFAULTS.raid_success_scale),
    "raid_range_base": (2, 10, _DEFAULTS.raid_range_base),
    "raid_range_longship_bonus": (2, 14, _DEFAULTS.raid_range_longship_bonus),
    "raid_damage_frac": (0.05, 0.50, _DEFAULTS.raid_damage_frac),
    "raid_capture_threshold": (0.5, 3.0, _DEFAULTS.raid_capture_threshold),
    "trade_range": (3, 15, _DEFAULTS.trade_range),
    "trade_value_scale": (0.05, 0.50, _DEFAULTS.trade_value_scale),
    "winter_severity_mean": (0.1, 0.9, _DEFAULTS.winter_severity_mean),
    "winter_severity_concentration": (2.0, 50.0, _DEFAULTS.winter_severity_concentration),
    "winter_severity_autocorr": (0.0, 0.8, _DEFAULTS.winter_severity_autocorr),
    "winter_food_loss_per_population": (0.02, 0.30, _DEFAULTS.winter_food_loss_per_population),
    "collapse_threshold": (0.3, 2.5, _DEFAULTS.collapse_threshold),
    "collapse_softness": (0.05, 0.80, _DEFAULTS.collapse_softness),
    "collapse_food_floor": (0.0, 0.5, _DEFAULTS.collapse_food_floor),
    "collapse_raid_stress_weight": (0.2, 3.0, _DEFAULTS.collapse_raid_stress_weight),
    "collapse_winter_severity_weight": (0.1, 2.0, _DEFAULTS.collapse_winter_severity_weight),
    "reclaim_threshold": (0.5, 3.5, _DEFAULTS.reclaim_threshold),
    "reclaim_rate": (0.01, 0.40, _DEFAULTS.reclaim_rate),
    "ruin_forest_rate": (0.02, 0.30, _DEFAULTS.ruin_forest_rate),
}


@dataclass
class Particle:
    """One hypothesis about the hidden simulation parameters."""

    params: dict[str, Any]
    log_weight: float = 0.0

    def to_simulation_params(self) -> SimulationParams:
        """Convert particle params to a full SimulationParams object.

        Non-inferred params are frozen at defaults.
        """
        defaults = SimulationParams()
        kwargs: dict[str, Any] = {}
        for f in defaults.__dataclass_fields__:
            if f in self.params:
                kwargs[f] = self.params[f]
            else:
                kwargs[f] = getattr(defaults, f)
        return SimulationParams(**kwargs)

    def normalized_weight(self, log_normalizer: float) -> float:
        """Compute normalized weight given log normalizer."""
        return float(np.exp(self.log_weight - log_normalizer))


def _sample_enum_param(name: str, rng: Generator) -> Any:
    """Sample a random value for an enum parameter."""
    if name == "adjacency_mode":
        return rng.choice(_ADJACENCY_CHOICES)
    elif name == "distance_metric":
        return rng.choice(_DISTANCE_CHOICES)
    elif name == "update_order_mode":
        return rng.choice(_UPDATE_ORDER_CHOICES)
    raise ValueError(f"Unknown enum param: {name}")


def _sample_continuous_param(name: str, rng: Generator, spread: float = 0.3) -> float:
    """Sample a continuous param around its default with bounded perturbation."""
    lo, hi, default = _CONTINUOUS_RANGES[name]
    # Perturb around default with Gaussian noise, clamp to range
    sigma = (hi - lo) * spread
    value = rng.normal(default, sigma)
    return float(np.clip(value, lo, hi))


_INTEGER_PARAMS: dict[str, float] = {
    "expansion_radius": 1.0,
    "trade_range": 2.0,
    "raid_range_base": 1.5,
    "raid_range_longship_bonus": 2.0,
}


def initialize_particles(
    n_particles: int = 24,
    seed: int = 0,
) -> list[Particle]:
    """Initialize particles around defaults with controlled perturbation.

    Args:
        n_particles: Number of particles to create (default: 24).
        seed: RNG seed for deterministic initialization.

    Returns:
        List of Particles with uniform log-weights.
    """
    rng = default_rng(seed)
    particles: list[Particle] = []

    # First particle: exact defaults (anchor)
    default_params: dict[str, Any] = {}
    for name in INFERRED_PARAMS:
        default_params[name] = getattr(_DEFAULTS, name)
    particles.append(Particle(params=default_params.copy(), log_weight=0.0))

    # Remaining particles: perturbations
    for _ in range(n_particles - 1):
        params: dict[str, Any] = {}
        for name in INFERRED_PARAMS:
            if name in ("adjacency_mode", "distance_metric", "update_order_mode"):
                if rng.random() < 0.7:
                    params[name] = getattr(_DEFAULTS, name)
                else:
                    params[name] = _sample_enum_param(name, rng)
            elif name in _INTEGER_PARAMS:
                lo, hi, default = _CONTINUOUS_RANGES[name]
                sigma = _INTEGER_PARAMS[name]
                params[name] = int(np.clip(rng.normal(default, sigma), lo, hi))
            else:
                params[name] = _sample_continuous_param(name, rng)
        particles.append(Particle(params=params, log_weight=0.0))

    return particles


def validate_particle(particle: Particle) -> list[str]:
    """Validate a particle's params against known ranges. Returns list of errors."""
    errors: list[str] = []
    for name in INFERRED_PARAMS:
        if name not in particle.params:
            errors.append(f"Missing param: {name}")
            continue

        value = particle.params[name]

        if name == "adjacency_mode":
            if value not in _ADJACENCY_CHOICES:
                errors.append(f"Invalid adjacency_mode: {value}")
        elif name == "distance_metric":
            if value not in _DISTANCE_CHOICES:
                errors.append(f"Invalid distance_metric: {value}")
        elif name == "update_order_mode":
            if value not in _UPDATE_ORDER_CHOICES:
                errors.append(f"Invalid update_order_mode: {value}")
        elif name in _CONTINUOUS_RANGES:
            lo, hi, _ = _CONTINUOUS_RANGES[name]
            if not (lo <= float(value) <= hi):
                errors.append(f"{name}={value} out of range [{lo}, {hi}]")

    return errors
