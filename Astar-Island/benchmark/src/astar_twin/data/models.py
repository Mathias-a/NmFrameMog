from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from astar_twin.contracts.api_models import InitialState
from astar_twin.params import SimulationParams


class ParamsSource(StrEnum):
    """Provenance of the simulation_params stored in a RoundFixture.

    Values:
      benchmark_truth  — params were read from a trusted offline source (e.g.
                         a controlled benchmark round where the hidden params
                         are known).
      inferred         — params were estimated by the particle-filter inference
                         pipeline from real API observations.
      default_prior    — params are the SimulationParams() defaults; the real
                         server values are unknown.  Fixture is suitable for
                         ground-truth generation but NOT for calibration.
    """

    BENCHMARK_TRUTH = "benchmark_truth"
    INFERRED = "inferred"
    DEFAULT_PRIOR = "default_prior"


class RoundFixture(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[InitialState]
    ground_truths: list[list[list[list[float]]]] | None = None
    simulation_params: SimulationParams = SimulationParams()
    params_source: ParamsSource = ParamsSource.DEFAULT_PRIOR
    event_date: str = ""
    round_weight: float = 1.0
