"""HTTP client for the Astar Island competition API.

Fetches round metadata, initial states, and ground-truth analysis
for use in offline backtesting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import TracebackType

import httpx
import numpy as np
from numpy.typing import NDArray

# JSON dict type from httpx response.json() — we cast explicitly at boundaries.
type _JsonDict = dict[str, object]


@dataclass(frozen=True, slots=True)
class RoundData:
    """Summary metadata for a single competition round."""

    round_id: str
    round_number: int
    status: str
    map_width: int
    map_height: int


@dataclass(frozen=True, slots=True)
class InitialState:
    """Initial grid and settlement data for one seed within a round."""

    grid: list[list[int]]
    settlements: list[dict[str, int | bool]]


@dataclass(frozen=True, slots=True)
class RoundInfo:
    """Detailed round information including initial states."""

    round_id: str
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[InitialState]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Ground-truth analysis for one seed of a round."""

    round_id: str
    seed_index: int
    ground_truth: NDArray[np.float64]
    prediction: NDArray[np.float64] | None
    official_score: float
    initial_grid: list[list[int]]


def _int_field(data: _JsonDict, key: str) -> int:
    """Extract an integer field from a JSON dict."""
    return int(str(data[key]))


def _str_field(data: _JsonDict, key: str) -> str:
    """Extract a string field from a JSON dict."""
    return str(data[key])


def _float_field(data: _JsonDict, key: str) -> float:
    """Extract a float field from a JSON dict."""
    return float(str(data[key]))


def _parse_round_data(data: _JsonDict) -> RoundData:
    """Parse a round data dict from the API response."""
    return RoundData(
        round_id=_str_field(data, "round_id"),
        round_number=_int_field(data, "round_number"),
        status=_str_field(data, "status"),
        map_width=_int_field(data, "map_width"),
        map_height=_int_field(data, "map_height"),
    )


def _parse_initial_state(data: _JsonDict) -> InitialState:
    """Parse an initial state dict from the API response."""
    grid: list[list[int]] = []
    settlements: list[dict[str, int | bool]] = []

    raw_grid = data.get("grid")
    if isinstance(raw_grid, list):
        grid = raw_grid

    raw_settlements = data.get("settlements")
    if isinstance(raw_settlements, list):
        settlements = raw_settlements

    return InitialState(grid=grid, settlements=settlements)


def _parse_round_info(data: _JsonDict) -> RoundInfo:
    """Parse a round info dict from the API response."""
    raw_states = data.get("initial_states")
    initial_states: list[InitialState] = []
    if isinstance(raw_states, list):
        initial_states = [_parse_initial_state(s) for s in raw_states]
    return RoundInfo(
        round_id=_str_field(data, "round_id"),
        status=_str_field(data, "status"),
        map_width=_int_field(data, "map_width"),
        map_height=_int_field(data, "map_height"),
        seeds_count=_int_field(data, "seeds_count"),
        initial_states=initial_states,
    )


def _parse_analysis_result(
    data: _JsonDict, round_id: str, seed_index: int
) -> AnalysisResult:
    """Parse an analysis result dict from the API response."""
    gt_raw = data["ground_truth"]
    gt_array: NDArray[np.float64] = np.array(gt_raw, dtype=np.float64)

    pred_raw = data.get("prediction")
    pred_array: NDArray[np.float64] | None = None
    if pred_raw is not None:
        pred_array = np.array(pred_raw, dtype=np.float64)

    raw_grid = data.get("initial_grid")
    initial_grid: list[list[int]] = []
    if isinstance(raw_grid, list):
        initial_grid = raw_grid

    return AnalysisResult(
        round_id=round_id,
        seed_index=seed_index,
        ground_truth=gt_array,
        prediction=pred_array,
        official_score=_float_field(data, "official_score"),
        initial_grid=initial_grid,
    )


@dataclass(slots=True)
class AstarClient:
    """HTTP client for the Astar Island competition API.

    Supports context-manager usage for automatic cleanup.

    Usage::

        with AstarClient(token="my-token") as client:
            rounds = client.get_rounds()
    """

    base_url: str = "https://api.ainm.no"
    token: str | None = None
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        resolved_token = self.token or os.environ.get("ASTAR_API_TOKEN")
        headers: dict[str, str] = {}
        if resolved_token:
            headers["Authorization"] = f"Bearer {resolved_token}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0,
        )

    def get_rounds(self) -> list[RoundData]:
        """Fetch all available rounds."""
        response = self._client.get("/rounds")
        response.raise_for_status()
        raw: list[_JsonDict] = response.json()
        return [_parse_round_data(r) for r in raw]

    def get_round(self, round_id: str) -> RoundInfo:
        """Fetch detailed information for a specific round."""
        response = self._client.get(f"/rounds/{round_id}")
        response.raise_for_status()
        data: _JsonDict = response.json()
        return _parse_round_info(data)

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResult:
        """Fetch ground-truth analysis for a specific seed."""
        response = self._client.get(f"/rounds/{round_id}/analysis/{seed_index}")
        response.raise_for_status()
        data: _JsonDict = response.json()
        return _parse_analysis_result(data, round_id, seed_index)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> AstarClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
