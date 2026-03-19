"""Simulation engine for the Norse world simulator."""

from __future__ import annotations

import random

from .mapgen import generate_map
from .models import SimConfig, WorldState
from .phases import (
    phase_conflict,
    phase_environment,
    phase_growth,
    phase_trade,
    phase_winter,
)


def run_simulation(
    seed: int,
    config: SimConfig | None = None,
    num_years: int = 50,
    stochastic_seed: int | None = None,
    width: int = 40,
    height: int = 40,
) -> WorldState:
    """Run a full Norse world simulation.

    Args:
        seed: Seed for deterministic map generation.
        config: Simulation parameters. Uses defaults if None.
        num_years: Number of simulation years to run.
        stochastic_seed: Seed for phase randomness. Uses seed if None.
        width: Map width.
        height: Map height.

    Returns:
        Final world state after all years.
    """
    cfg = config or SimConfig()
    state = generate_map(seed, cfg, width, height)
    rng = random.Random(stochastic_seed if stochastic_seed is not None else seed)

    for year in range(num_years):
        state.year = year + 1
        phase_growth(state, cfg, rng)
        phase_conflict(state, cfg, rng)
        phase_trade(state, cfg, rng)
        phase_winter(state, cfg, rng)
        phase_environment(state, cfg, rng)

    return state
