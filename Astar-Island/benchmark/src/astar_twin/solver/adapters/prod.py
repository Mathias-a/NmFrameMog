"""Production adapter — drives the real Astar Island competition API.

Implements the SolverAdapter protocol so the solver pipeline can run
against the live server at https://api.ainm.no without any code changes.

Token resolution follows the same convention as fetch_real_rounds.py:
  1. ``token`` constructor argument (explicit)
  2. ``ACCESS_TOKEN`` environment variable
  3. ``access_token=<jwt>`` in worktree-2/.env
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import cast

import httpx
import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import (
    AnalysisResponse,
    BudgetResponse,
    InitialSettlement,
    InitialState,
    RoundDetail,
    SimulateResponse,
    SubmitResponse,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.ainm.no"
_ENV_FILE_SEARCH_LIMIT = 10  # max parent dirs to search for .env


def _resolve_token(explicit_token: str | None = None) -> str:
    """Resolve an API access token.

    Priority:
      1. Explicit value passed to constructor.
      2. ``ACCESS_TOKEN`` environment variable.
      3. ``access_token=<jwt>`` line in the nearest ``.env`` file
         (walks up to 5 parent directories from this file).

    Raises:
        RuntimeError: if no token can be found.
    """
    if explicit_token:
        return explicit_token

    env_token = os.environ.get("ACCESS_TOKEN", "").strip()
    if env_token:
        return env_token

    # Walk upward looking for a .env file (matches fetch_real_rounds.py behaviour)
    search_dir = Path(__file__).resolve().parent
    for _ in range(_ENV_FILE_SEARCH_LIMIT):
        search_dir = search_dir.parent
        candidate = search_dir / ".env"
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("access_token="):
                    value = stripped.split("=", 1)[1].strip()
                    if value:
                        return value

    raise RuntimeError(
        "No API token found. Pass token= to ProdAdapter, set ACCESS_TOKEN env var, "
        "or create a .env file with 'access_token=<jwt>'."
    )


def _parse_round_detail(raw: dict[str, object]) -> RoundDetail:
    """Build a typed ``RoundDetail`` from the raw API JSON.

    The ``/rounds/{id}`` endpoint returns extra fields (event_date,
    prediction_window_minutes, started_at, closes_at) that the strict
    Pydantic model rejects, so we pick only the fields we need.
    """
    raw_states = cast(list[dict[str, object]], raw["initial_states"])
    initial_states: list[InitialState] = [
        InitialState(
            grid=cast(list[list[int]], s["grid"]),
            settlements=[
                InitialSettlement.model_validate(st) for st in cast(list[object], s["settlements"])
            ],
        )
        for s in raw_states
    ]
    return RoundDetail(
        id=str(raw["id"]),
        round_number=cast(int, raw["round_number"]),
        status=str(raw["status"]),
        map_width=cast(int, raw["map_width"]),
        map_height=cast(int, raw["map_height"]),
        seeds_count=cast(int, raw["seeds_count"]),
        initial_states=initial_states,
    )


class ProdAdapter:
    """Adapter that routes solver calls to the live competition API.

    Satisfies the ``SolverAdapter`` protocol — the solver pipeline can use
    this as a drop-in replacement for ``BenchmarkAdapter``.

    Args:
        token: API access token.  If ``None``, resolved via env var / .env file.
        base_url: API base URL (default: ``https://api.ainm.no``).
        timeout: HTTP request timeout in seconds (default: 60).
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self._token = _resolve_token(token)
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._auth_headers = {"Authorization": f"Bearer {self._token}"}

    # ------------------------------------------------------------------
    # SolverAdapter protocol methods
    # ------------------------------------------------------------------

    def get_round_detail(self, round_id: str) -> RoundDetail:
        """Fetch round metadata and initial states for all seeds."""
        resp = self._client.get(f"{self._base_url}/astar-island/rounds/{round_id}")
        resp.raise_for_status()
        raw: dict[str, object] = resp.json()
        return _parse_round_detail(raw)

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> SimulateResponse:
        """Execute one viewport query (costs 1 query from budget)."""
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        }
        resp = self._client.post(
            f"{self._base_url}/astar-island/simulate",
            json=payload,
            headers=self._auth_headers,
        )
        if resp.status_code == 429:
            raise RuntimeError(f"Query budget exhausted (HTTP 429): {resp.text}")
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        return SimulateResponse.model_validate(data)

    def submit(
        self,
        round_id: str,
        seed_index: int,
        prediction: NDArray[np.float64],
    ) -> SubmitResponse:
        """Submit a H x W x 6 prediction tensor for one seed."""
        pred_list = cast(list[list[list[float]]], prediction.tolist())
        payload: dict[str, str | int | list[list[list[float]]]] = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": pred_list,
        }
        resp = self._client.post(
            f"{self._base_url}/astar-island/submit",
            json=payload,
            headers=self._auth_headers,
        )
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        return SubmitResponse.model_validate(data)

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResponse:
        """Fetch post-round ground truth and score (only after round completes)."""
        resp = self._client.get(
            f"{self._base_url}/astar-island/analysis/{round_id}/{seed_index}",
            headers=self._auth_headers,
        )
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        return AnalysisResponse.model_validate(data)

    def get_budget(self, round_id: str) -> tuple[int, int]:
        """Return (queries_used, queries_max) for the active round."""
        resp = self._client.get(
            f"{self._base_url}/astar-island/budget",
            headers=self._auth_headers,
        )
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        budget = BudgetResponse.model_validate(data)
        return (budget.queries_used, budget.queries_max)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> ProdAdapter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
