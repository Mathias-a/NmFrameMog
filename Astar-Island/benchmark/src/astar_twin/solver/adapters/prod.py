from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import (
    AnalysisResponse,
    BudgetResponse,
    InitialSettlement,
    InitialState,
    RoundDetail,
    SimulateRequest,
    SimulateResponse,
    SubmitRequest,
    SubmitResponse,
)

_DEFAULT_BASE_URL = "https://api.ainm.no"
_DEFAULT_TIMEOUT = 30.0
_TOKEN_ENV_VARS = ("AINM_ACCESS_TOKEN", "ACCESS_TOKEN")


class ProdAdapterError(RuntimeError):
    pass


class QueryBudgetExhaustedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProdAdapterConfig:
    base_url: str = _DEFAULT_BASE_URL
    timeout_seconds: float = _DEFAULT_TIMEOUT
    token: str | None = None
    submit_enabled: bool = False


def _normalize_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/astar-island"):
        return stripped[: -len("/astar-island")]
    return stripped


def _load_token_from_environment() -> str | None:
    for env_var in _TOKEN_ENV_VARS:
        token = os.environ.get(env_var, "").strip()
        if token:
            return token

    repo_root = Path(__file__).resolve().parents[5]
    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("access_token="):
                token = stripped.split("=", 1)[1].strip()
                if token:
                    return token

    return None


def _auth_headers(token: str | None) -> dict[str, str]:
    if token is None:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _require_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Round detail field '{key}' must be an integer")
    return value


def _require_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Round detail field '{key}' must be a string")
    return value


def _project_round_detail(payload: dict[str, object]) -> RoundDetail:
    initial_states_payload = payload.get("initial_states")
    if not isinstance(initial_states_payload, list):
        raise ValueError("Round detail payload missing initial_states list")

    initial_states: list[InitialState] = []
    for state in initial_states_payload:
        if not isinstance(state, dict):
            raise ValueError("Initial state payload must be an object")
        settlements_payload = state.get("settlements")
        if not isinstance(settlements_payload, list):
            raise ValueError("Initial state settlements must be a list")
        settlements = [InitialSettlement.model_validate(item) for item in settlements_payload]
        initial_states.append(
            InitialState(
                grid=state["grid"],
                settlements=settlements,
            )
        )

    return RoundDetail(
        id=_require_str(payload, "id"),
        round_number=_require_int(payload, "round_number"),
        status=_require_str(payload, "status"),
        map_width=_require_int(payload, "map_width"),
        map_height=_require_int(payload, "map_height"),
        seeds_count=_require_int(payload, "seeds_count"),
        initial_states=initial_states,
    )


class ProdAdapter:
    def __init__(
        self,
        config: ProdAdapterConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = ProdAdapterConfig(
            base_url=_normalize_base_url(config.base_url),
            timeout_seconds=config.timeout_seconds,
            token=config.token,
            submit_enabled=config.submit_enabled,
        )
        self._client = client or httpx.Client(timeout=self._config.timeout_seconds)

    @classmethod
    def from_environment(
        cls,
        *,
        base_url: str | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
        submit_enabled: bool = False,
        client: httpx.Client | None = None,
    ) -> ProdAdapter:
        resolved_base_url = base_url or os.environ.get("AINM_BASE_URL", _DEFAULT_BASE_URL)
        token = _load_token_from_environment()
        return cls(
            ProdAdapterConfig(
                base_url=resolved_base_url,
                timeout_seconds=timeout_seconds,
                token=token,
                submit_enabled=submit_enabled,
            ),
            client=client,
        )

    def close(self) -> None:
        self._client.close()

    def _url(self, path: str) -> str:
        return f"{self._config.base_url}{path}"

    def _require_token(self) -> str:
        token = self._config.token
        if token is None:
            raise ProdAdapterError(
                "No API token configured. Set AINM_ACCESS_TOKEN or ACCESS_TOKEN, "
                "or provide a token explicitly."
            )
        return token

    def _request(
        self,
        method: str,
        path: str,
        *,
        auth_required: bool,
        json_body: dict[str, object] | None = None,
    ) -> httpx.Response:
        headers = _auth_headers(self._require_token()) if auth_required else {}
        response = self._client.request(method, self._url(path), headers=headers, json=json_body)
        if response.status_code == 429:
            raise QueryBudgetExhaustedError(response.text)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProdAdapterError(
                f"{method} {path} failed with {exc.response.status_code}: {exc.response.text}"
            ) from exc
        return response

    def get_round_detail(self, round_id: str) -> RoundDetail:
        response = self._request("GET", f"/astar-island/rounds/{round_id}", auth_required=False)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Round detail endpoint returned non-object payload")
        return _project_round_detail(payload)

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> SimulateResponse:
        request = SimulateRequest(
            round_id=round_id,
            seed_index=seed_index,
            viewport_x=viewport_x,
            viewport_y=viewport_y,
            viewport_w=viewport_w,
            viewport_h=viewport_h,
        )
        response = self._request(
            "POST",
            "/astar-island/simulate",
            auth_required=True,
            json_body=request.model_dump(),
        )
        return SimulateResponse.model_validate(response.json())

    def submit(
        self,
        round_id: str,
        seed_index: int,
        prediction: NDArray[np.float64],
    ) -> SubmitResponse:
        if not self._config.submit_enabled:
            raise RuntimeError("Submission disabled. Re-run with explicit submit opt-in.")
        request = SubmitRequest(
            round_id=round_id,
            seed_index=seed_index,
            prediction=prediction.tolist(),
        )
        response = self._request(
            "POST",
            "/astar-island/submit",
            auth_required=True,
            json_body=request.model_dump(),
        )
        return SubmitResponse.model_validate(response.json())

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResponse:
        response = self._request(
            "GET",
            f"/astar-island/analysis/{round_id}/{seed_index}",
            auth_required=True,
        )
        return AnalysisResponse.model_validate(response.json())

    def get_budget(self, round_id: str) -> tuple[int, int]:
        del round_id
        response = self._request("GET", "/astar-island/budget", auth_required=True)
        budget = BudgetResponse.model_validate(response.json())
        return (budget.queries_used, budget.queries_max)
