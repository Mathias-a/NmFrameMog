from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from task_tripletex.models import SolveFile, SolveRequest


@dataclass(frozen=True)
class ReadDefinition:
    name: str
    path: str
    query: dict[str, str | int | float | bool | list[str | int | float | bool]]
    mode: Literal["list_values", "object"]


@dataclass(frozen=True)
class CheckDefinition:
    name: str
    points: float
    read_name: str
    kind: Literal["entity_exists", "field_equals"]
    selector: dict[str, object]
    field_path: str | None = None
    expected: object | None = None


@dataclass(frozen=True)
class EfficiencyPolicy:
    best_write_calls: int
    max_write_calls: int
    max_4xx_errors: int
    write_weight: float
    error_weight: float


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    description: str
    tier: Literal[1, 2, 3]
    prompt: str
    files: list[SolveFile]
    reads: list[ReadDefinition]
    checks: list[CheckDefinition]
    efficiency_policy: EfficiencyPolicy
    expected_min_proxy_calls: int = 0


@dataclass(frozen=True)
class EndpointContractResult:
    url_is_https: bool
    url_targets_solve: bool
    request_content_type: str
    response_status_code: int
    response_content_type: str | None
    response_json: object | None
    response_text: str
    exact_success_response: bool
    errors: list[str]


@dataclass(frozen=True)
class EndpointRunResult:
    original_request: SolveRequest
    rewritten_request: SolveRequest
    sent_headers: dict[str, str]
    proxy_base_url: str
    elapsed_seconds: float
    contract: EndpointContractResult


@dataclass(frozen=True)
class RecordedTripletexCall:
    method: str
    path: str
    query_string: str
    forwarded_url: str
    request_headers: dict[str, str]
    request_body_text: str | None
    response_status_code: int
    response_body_text: str
    used_expected_basic_auth: bool
    write_call: bool
    client_error: bool


@dataclass(frozen=True)
class ProxyMetrics:
    total_calls: int
    write_calls: int
    client_error_calls: int
    used_proxy: bool
    base_url_rewritten: bool
    all_calls_used_expected_basic_auth: bool
    all_calls_forwarded_to_upstream_base_url: bool
    invalid_auth_paths: list[str]
    invalid_forward_paths: list[str]
    calls: list[RecordedTripletexCall]


@dataclass(frozen=True)
class CheckResult:
    name: str
    points_awarded: float
    max_points: float
    passed: bool
    details: str


@dataclass(frozen=True)
class VerificationResult:
    points_earned: float
    max_points: float
    correctness: float
    checks: list[CheckResult]
    snapshots: dict[str, object]


@dataclass(frozen=True)
class ScoreResult:
    base_score: float
    efficiency_bonus: float
    total_score: float
    write_efficiency: float
    error_efficiency: float
    combined_efficiency: float
    efficiency_bonus_applied: bool
    contract_valid: bool
    proxy_valid: bool
    disqualification_reasons: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    case: EvaluationCase
    endpoint_run: EndpointRunResult
    proxy_metrics: ProxyMetrics
    verification: VerificationResult
    score: ScoreResult
