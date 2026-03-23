"""Prior parameter sampling for fair benchmark comparison.

When a ``RoundFixture`` carries ``ParamsSource.DEFAULT_PRIOR``, the
ground truths were generated with ``SimulationParams()`` defaults.
To test whether a solver's advantage is robust to reasonable
parameter variation, this module samples ``SimulationParams``
instances from a distribution centred on the defaults.

Usage::

    from astar_twin.params.prior_sampling import PriorConfig, sample_params

    config = PriorConfig(spread=0.2, seed=42)
    sampled = sample_params(config)
    # sampled is a SimulationParams with numeric fields perturbed by ±20%
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import numpy as np

from astar_twin.params.simulation_params import SimulationParams

# Fields that are discrete enums — never perturbed.
_ENUM_FIELDS: frozenset[str] = frozenset(["adjacency_mode", "distance_metric", "update_order_mode"])

# Fields that must remain integers after perturbation.
_INT_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(SimulationParams) if f.type in ("int", int)
)


@dataclass(frozen=True)
class PriorConfig:
    """Configuration for parameter sampling.

    Attributes:
        spread: Fractional perturbation around default values.  ``0.2``
                means each numeric parameter is drawn uniformly from
                ``[default * (1 - 0.2), default * (1 + 0.2)]``.
        seed:   RNG seed for reproducibility.
    """

    spread: float = 0.2
    seed: int = 42


DEFAULT_PRIOR_CONFIG = PriorConfig(spread=0.0, seed=0)
"""A config that returns the unperturbed ``SimulationParams()`` defaults."""


def sample_params(config: PriorConfig) -> SimulationParams:
    """Sample a ``SimulationParams`` instance with numeric fields perturbed.

    Enum fields (``adjacency_mode``, ``distance_metric``,
    ``update_order_mode``) are always kept at their default values.

    For ``spread == 0.0`` the result is identical to ``SimulationParams()``.

    Args:
        config: Spread and seed controlling the perturbation.

    Returns:
        A new ``SimulationParams`` with perturbed numeric fields.
    """
    defaults = SimulationParams()
    if config.spread <= 0.0:
        return defaults

    rng = np.random.default_rng(config.seed)
    result = copy.copy(defaults)

    for f in fields(SimulationParams):
        if f.name in _ENUM_FIELDS:
            continue

        default_val = getattr(defaults, f.name)

        if isinstance(default_val, int):
            # Perturb integer fields with a minimum of 1
            low = max(1, int(default_val * (1.0 - config.spread)))
            high = max(low + 1, int(default_val * (1.0 + config.spread)) + 1)
            setattr(result, f.name, int(rng.integers(low, high)))
        elif isinstance(default_val, float):
            if abs(default_val) < 1e-12:
                # Zero-valued defaults: perturb symmetrically around 0
                setattr(result, f.name, float(rng.uniform(-config.spread, config.spread)))
            else:
                low = default_val * (1.0 - config.spread)
                high = default_val * (1.0 + config.spread)
                if low > high:
                    low, high = high, low
                setattr(result, f.name, float(rng.uniform(low, high)))

    return result


def sample_params_batch(
    config: PriorConfig,
    n_samples: int,
) -> list[SimulationParams]:
    """Sample multiple ``SimulationParams`` instances.

    Each sample uses ``config.seed + i`` as the seed for the *i*-th draw,
    guaranteeing deterministic and non-overlapping randomness.
    """
    return [
        sample_params(PriorConfig(spread=config.spread, seed=config.seed + i))
        for i in range(n_samples)
    ]
