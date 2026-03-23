from __future__ import annotations

from task_tripletex.testing.fixture_loader import load_packaged_case_fixture
from task_tripletex.testing.models import (
    EndpointContractResult,
    ProxyMetrics,
    ScoreResult,
    VerificationResult,
)
from task_tripletex.testing.scoring import compute_score


def _proxy_metrics(write_calls: int, client_error_calls: int) -> ProxyMetrics:
    return ProxyMetrics(
        total_calls=write_calls,
        write_calls=write_calls,
        client_error_calls=client_error_calls,
        used_proxy=True,
        base_url_rewritten=True,
        all_calls_used_expected_basic_auth=True,
        all_calls_forwarded_to_upstream_base_url=True,
        invalid_auth_paths=[],
        invalid_forward_paths=[],
        calls=[],
    )


def _valid_contract() -> EndpointContractResult:
    return EndpointContractResult(
        url_is_https=True,
        url_targets_solve=True,
        request_content_type="application/json",
        response_status_code=200,
        response_content_type="application/json",
        response_json={"status": "completed"},
        response_text='{"status": "completed"}',
        exact_success_response=True,
        errors=[],
    )


def test_compute_score_awards_full_efficiency_bonus_for_best_run() -> None:
    case = load_packaged_case_fixture("create_employee_admin")
    verification = VerificationResult(
        points_earned=10.0,
        max_points=10.0,
        correctness=1.0,
        checks=[],
        snapshots={},
    )

    score = compute_score(
        case,
        verification,
        _proxy_metrics(write_calls=1, client_error_calls=0),
        _valid_contract(),
    )

    assert isinstance(score, ScoreResult)
    assert score.base_score == 1.0
    assert score.efficiency_bonus == 1.0
    assert score.total_score == 2.0
    assert score.contract_valid is True
    assert score.proxy_valid is True
    assert score.disqualification_reasons == []


def test_compute_score_disables_efficiency_for_imperfect_correctness() -> None:
    case = load_packaged_case_fixture("create_employee_admin")
    verification = VerificationResult(
        points_earned=8.0,
        max_points=10.0,
        correctness=0.8,
        checks=[],
        snapshots={},
    )

    score = compute_score(
        case,
        verification,
        _proxy_metrics(write_calls=5, client_error_calls=2),
        _valid_contract(),
    )

    assert score.base_score == 0.8
    assert score.efficiency_bonus == 0.0
    assert score.total_score == 0.8
    assert score.disqualification_reasons == []


def test_compute_score_zeroes_on_contract_violation() -> None:
    case = load_packaged_case_fixture("create_employee_admin")
    verification = VerificationResult(
        points_earned=10.0,
        max_points=10.0,
        correctness=1.0,
        checks=[],
        snapshots={},
    )
    bad_contract = EndpointContractResult(
        url_is_https=False,
        url_targets_solve=True,
        request_content_type="application/json",
        response_status_code=200,
        response_content_type="application/json",
        response_json={"status": "completed"},
        response_text='{"status": "completed"}',
        exact_success_response=True,
        errors=["Solve URL is not HTTPS."],
    )

    score = compute_score(
        case,
        verification,
        _proxy_metrics(write_calls=1, client_error_calls=0),
        bad_contract,
    )

    assert score.total_score == 0.0
    assert score.contract_valid is False
    assert score.proxy_valid is True
    assert len(score.disqualification_reasons) > 0
    assert "HTTPS" in score.disqualification_reasons[0]


def test_compute_score_zeroes_on_proxy_violation() -> None:
    case = load_packaged_case_fixture("create_employee_admin")
    verification = VerificationResult(
        points_earned=10.0,
        max_points=10.0,
        correctness=1.0,
        checks=[],
        snapshots={},
    )
    bad_proxy = ProxyMetrics(
        total_calls=0,
        write_calls=0,
        client_error_calls=0,
        used_proxy=False,
        base_url_rewritten=False,
        all_calls_used_expected_basic_auth=True,
        all_calls_forwarded_to_upstream_base_url=True,
        invalid_auth_paths=[],
        invalid_forward_paths=[],
        calls=[],
    )

    score = compute_score(
        case,
        verification,
        bad_proxy,
        _valid_contract(),
    )

    assert score.total_score == 0.0
    assert score.contract_valid is True
    assert score.proxy_valid is False
    assert len(score.disqualification_reasons) > 0
    assert "proxy" in score.disqualification_reasons[0].lower()
