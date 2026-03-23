from __future__ import annotations

from task_tripletex.testing.models import (
    EndpointContractResult,
    EvaluationCase,
    ProxyMetrics,
    ScoreResult,
    VerificationResult,
)


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _write_efficiency(
    best_write_calls: int, max_write_calls: int, observed: int
) -> float:
    if observed <= best_write_calls:
        return 1.0
    if max_write_calls == best_write_calls:
        return 0.0
    return _clamp((max_write_calls - observed) / (max_write_calls - best_write_calls))


def _error_efficiency(max_4xx_errors: int, observed: int) -> float:
    if max_4xx_errors == 0:
        return 1.0 if observed == 0 else 0.0
    return _clamp((max_4xx_errors - observed) / max_4xx_errors)


def _check_contract(
    contract: EndpointContractResult, *, enforce_https: bool
) -> list[str]:
    reasons: list[str] = []
    if enforce_https and not contract.url_is_https:
        reasons.append("Solve URL is not HTTPS.")
    if not contract.url_targets_solve:
        reasons.append("Solve URL does not target /solve.")
    if not contract.exact_success_response:
        reasons.append(
            f"Solve response was not HTTP 200 with "
            f'{{"status": "completed"}} (got HTTP {contract.response_status_code}).'
        )
    return reasons


def _check_proxy(proxy_metrics: ProxyMetrics) -> list[str]:
    reasons: list[str] = []
    if not proxy_metrics.used_proxy:
        reasons.append("Solution did not route traffic through the recording proxy.")
    if not proxy_metrics.base_url_rewritten:
        reasons.append("Solution did not use the rewritten proxy base URL.")
    if not proxy_metrics.all_calls_used_expected_basic_auth:
        reasons.append(
            f"Invalid Basic Auth on: {', '.join(proxy_metrics.invalid_auth_paths)}."
        )
    if not proxy_metrics.all_calls_forwarded_to_upstream_base_url:
        reasons.append(
            f"Invalid forwarding on: {', '.join(proxy_metrics.invalid_forward_paths)}."
        )
    return reasons


def compute_score(
    case: EvaluationCase,
    verification: VerificationResult,
    proxy_metrics: ProxyMetrics,
    contract: EndpointContractResult,
    *,
    enforce_https: bool = True,
) -> ScoreResult:
    """Compute the final score for an evaluation case.

    Contract and proxy violations are score-blocking: any violation
    zeroes the entire score regardless of correctness.

    Set ``enforce_https=False`` for local testing with ``http://127.0.0.1``
    solve URLs (the HTTPS contract check is skipped).
    """
    contract_reasons = _check_contract(contract, enforce_https=enforce_https)
    proxy_reasons = _check_proxy(proxy_metrics)
    disqualification_reasons = contract_reasons + proxy_reasons
    contract_valid = len(contract_reasons) == 0
    proxy_valid = len(proxy_reasons) == 0

    if disqualification_reasons:
        return ScoreResult(
            base_score=0.0,
            efficiency_bonus=0.0,
            total_score=0.0,
            write_efficiency=0.0,
            error_efficiency=0.0,
            combined_efficiency=0.0,
            efficiency_bonus_applied=False,
            contract_valid=contract_valid,
            proxy_valid=proxy_valid,
            disqualification_reasons=disqualification_reasons,
        )

    base_score = verification.correctness * float(case.tier)
    if verification.correctness < 1.0:
        return ScoreResult(
            base_score=base_score,
            efficiency_bonus=0.0,
            total_score=base_score,
            write_efficiency=0.0,
            error_efficiency=0.0,
            combined_efficiency=0.0,
            efficiency_bonus_applied=False,
            contract_valid=True,
            proxy_valid=True,
            disqualification_reasons=[],
        )

    policy = case.efficiency_policy
    write_efficiency = _write_efficiency(
        policy.best_write_calls,
        policy.max_write_calls,
        proxy_metrics.write_calls,
    )
    error_efficiency = _error_efficiency(
        policy.max_4xx_errors,
        proxy_metrics.client_error_calls,
    )
    weight_total = policy.write_weight + policy.error_weight
    combined_efficiency = (
        (write_efficiency * policy.write_weight)
        + (error_efficiency * policy.error_weight)
    ) / weight_total
    efficiency_bonus = float(case.tier) * combined_efficiency
    return ScoreResult(
        base_score=base_score,
        efficiency_bonus=efficiency_bonus,
        total_score=base_score + efficiency_bonus,
        write_efficiency=write_efficiency,
        error_efficiency=error_efficiency,
        combined_efficiency=combined_efficiency,
        efficiency_bonus_applied=True,
        contract_valid=True,
        proxy_valid=True,
        disqualification_reasons=[],
    )
