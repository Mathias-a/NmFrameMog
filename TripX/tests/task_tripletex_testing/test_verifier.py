from __future__ import annotations

import asyncio

from task_tripletex.models import TripletexCredentials
from task_tripletex.testing.fixture_loader import load_packaged_case_fixture
from task_tripletex.testing.verifier import verify_case
from tests.task_tripletex_testing.helpers import (
    LocalHTTPServer,
    UpstreamState,
    UpstreamTripletexHandler,
)


def test_verify_case_scores_field_by_field() -> None:
    case = load_packaged_case_fixture("create_employee_admin")
    state = UpstreamState()
    state.create_employee(
        {
            "firstName": "Ola",
            "lastName": "Nordmann",
            "email": "ola.admin.test@example.org",
            "isAdministrator": False,
        }
    )

    with LocalHTTPServer(UpstreamTripletexHandler, state) as upstream:
        credentials = TripletexCredentials(
            base_url=f"{upstream.base_url}/v2",
            session_token="test-token",
        )
        result = asyncio.run(verify_case(case, credentials))

    assert result.points_earned == 5.0
    assert result.max_points == 5.0
    assert result.correctness == 1.0
    assert [check.passed for check in result.checks] == [True, True, True, True]


def test_verify_case_scores_customer_field_by_field() -> None:
    case = load_packaged_case_fixture("create_customer")
    state = UpstreamState()
    state.create_customer(
        {
            "name": "Nordlys Testkunde Create Customer AS",
            "email": "kunde.create_customer@example.org",
            "phoneNumber": "40010001",
        }
    )

    with LocalHTTPServer(UpstreamTripletexHandler, state) as upstream:
        credentials = TripletexCredentials(
            base_url=f"{upstream.base_url}/v2",
            session_token="test-token",
        )
        result = asyncio.run(verify_case(case, credentials))

    assert result.points_earned == 5.0
    assert result.max_points == 5.0
    assert result.correctness == 1.0
    assert [check.passed for check in result.checks] == [True, True, True, True]


def test_verify_case_scores_product_field_by_field() -> None:
    case = load_packaged_case_fixture("create_product")
    state = UpstreamState()
    state.create_product(
        {
            "name": "Reisetid Konsulentpakke Create Product",
            "number": "PROD-CREATE-PRODUCT-001",
            "priceExcludingVatCurrency": 1500.0,
        }
    )

    with LocalHTTPServer(UpstreamTripletexHandler, state) as upstream:
        credentials = TripletexCredentials(
            base_url=f"{upstream.base_url}/v2",
            session_token="test-token",
        )
        result = asyncio.run(verify_case(case, credentials))

    assert result.points_earned == 5.0
    assert result.max_points == 5.0
    assert result.correctness == 1.0
    assert [check.passed for check in result.checks] == [True, True, True, True]


def test_verify_case_matches_project_nested_manager_selector() -> None:
    case = load_packaged_case_fixture("create_project_basic")
    state = UpstreamState()
    state.create_project(
        {
            "name": "Fjordbro Forprosjekt",
            "number": "PROJ-BASIC-001",
            "startDate": "2026-03-21",
            "projectManager": {
                "id": 77,
                "email": "kari.project.basic@example.org",
                "firstName": "Kari",
                "lastName": "Prosjektleder",
            },
        }
    )

    with LocalHTTPServer(UpstreamTripletexHandler, state) as upstream:
        credentials = TripletexCredentials(
            base_url=f"{upstream.base_url}/v2",
            session_token="test-token",
        )
        result = asyncio.run(verify_case(case, credentials))

    assert result.points_earned == 5.0
    assert result.max_points == 5.0
    assert result.correctness == 1.0
    assert [check.passed for check in result.checks] == [True, True, True, True]


def test_verify_case_scores_journal_entry_with_posting_paths() -> None:
    case = load_packaged_case_fixture("create_journal_entry_basic")
    state = UpstreamState()
    state.create_voucher(
        {
            "description": "JOURNAL-BASIC-001",
            "date": "2026-03-21",
            "postings": [
                {
                    "amountGross": 1500,
                    "amountGrossCurrency": 1500,
                    "account": {"number": 1920},
                },
                {
                    "amountGross": -1500,
                    "amountGrossCurrency": -1500,
                    "account": {"number": 3000},
                },
            ],
        }
    )

    with LocalHTTPServer(UpstreamTripletexHandler, state) as upstream:
        credentials = TripletexCredentials(
            base_url=f"{upstream.base_url}/v2",
            session_token="test-token",
        )
        result = asyncio.run(verify_case(case, credentials))

    assert result.points_earned == 5.0
    assert result.max_points == 5.0
    assert result.correctness == 1.0
    assert [check.passed for check in result.checks] == [True, True, True, True]


def test_verify_case_requires_travel_expense_employee_linkage() -> None:
    case = load_packaged_case_fixture("create_travel_expense_basic")
    state = UpstreamState()
    state.create_travel_expense(
        {
            "title": "TRAVEL-EXPENSE-BASIC-001",
            "employee": {
                "id": 91,
                "email": "wrong.employee@example.org",
                "firstName": "Wrong",
                "lastName": "Person",
            },
        }
    )

    with LocalHTTPServer(UpstreamTripletexHandler, state) as upstream:
        credentials = TripletexCredentials(
            base_url=f"{upstream.base_url}/v2",
            session_token="test-token",
        )
        result = asyncio.run(verify_case(case, credentials))

    assert result.points_earned == 0.0
    assert result.max_points == 4.0
    assert result.correctness == 0.0
    assert [check.passed for check in result.checks] == [False, False, False]
    assert result.checks[0].details == "No entity matched the selector."
    assert result.checks[1].details == "No entity matched the selector."
    assert (
        result.checks[2].details
        == "Observed 'wrong.employee@example.org'; expected 'kari.travel.basic@example.org'."
    )
