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
      default_prior    — params come from the benchmark prior rooted in
                         SimulationParams() defaults. The real server values are
                         unknown. Fixture is suitable for synthetic
                         ground-truth generation but NOT for calibration.
    """

    BENCHMARK_TRUTH = "benchmark_truth"
    INFERRED = "inferred"
    DEFAULT_PRIOR = "default_prior"


class GroundTruthSource(StrEnum):
    """Provenance of the ground_truths stored in a RoundFixture.

    Values:
      api_analysis — ground truths were fetched from the real API's
                     ``/analysis`` endpoint (server-side MC).  These are the
                     most trustworthy reference for offline evaluation.
      local_mc     — ground truths were computed locally via the digital twin
                     Monte Carlo pipeline.  Useful for development but
                     reflects twin fidelity, not live server behaviour.
      unknown      — provenance is not recorded.  Treat as local_mc for
                     safety when making selection decisions.
    """

    API_ANALYSIS = "api_analysis"
    LOCAL_MC = "local_mc"
    UNKNOWN = "unknown"


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
    ground_truth_source: GroundTruthSource = GroundTruthSource.UNKNOWN
    event_date: str = ""
    round_weight: float = 1.0
