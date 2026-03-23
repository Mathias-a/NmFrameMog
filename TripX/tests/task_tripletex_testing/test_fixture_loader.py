from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from task_tripletex.testing.fixture_loader import (
    load_case_fixture,
    load_packaged_case_fixture,
)


SUPPORTED_INLINE_FILE_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}
MAX_INLINE_FILE_BYTES = 20 * 1024 * 1024


PACKAGED_CASE_NAMES = [
    "create_employee_admin",
    "create_customer",
    "create_product",
    "create_invoice_basic",
    "create_project_basic",
    "create_journal_entry_basic",
    "create_travel_expense_basic",
    "supplier_invoice_basic",
    "bank_reconciliation_file",
    "invoice_with_payment",
]


def _build_minimal_fixture_document(
    *, files: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "case_id": "fixture_with_file_policy_expectations",
        "description": "Minimal fixture document for file policy expectation tests.",
        "tier": 2,
        "prompt": "Registrer fixture_with_file_policy_expectations.",
        "files": files,
        "verification": {
            "reads": [
                {
                    "name": "entities",
                    "path": "/supplierInvoice",
                    "query": {
                        "invoiceNumber": "fixture-with-file-policy-expectations",
                        "count": 1,
                    },
                    "mode": "list_values",
                }
            ],
            "checks": [
                {
                    "name": "Fixture entity exists",
                    "points": 1,
                    "read_name": "entities",
                    "kind": "entity_exists",
                    "selector": {
                        "invoiceNumber": "fixture-with-file-policy-expectations"
                    },
                }
            ],
        },
        "efficiency": {
            "best_write_calls": 1,
            "max_write_calls": 2,
            "max_4xx_errors": 0,
            "write_weight": 0.7,
            "error_weight": 0.3,
        },
    }


def _write_fixture(path: Path, *, files: list[dict[str, object]]) -> Path:
    fixture_path = path / "fixture_with_file_policy_expectations.json"
    fixture_path.write_text(
        json.dumps(_build_minimal_fixture_document(files=files)), encoding="utf-8"
    )
    return fixture_path


def _assert_future_multimodal_file_policy(
    *, filename: str, mime_type: str | None, content_base64: str
) -> None:
    if mime_type not in SUPPORTED_INLINE_FILE_MIME_TYPES:
        raise ValueError(
            f"Unsupported file policy expectation for {filename}: {mime_type!r}."
        )

    decoded_size = len(base64.b64decode(content_base64, validate=True))
    if decoded_size > MAX_INLINE_FILE_BYTES:
        raise ValueError(
            f"Oversized file policy expectation for {filename}: {decoded_size} bytes."
        )


@pytest.mark.parametrize("case_name", PACKAGED_CASE_NAMES)
def test_load_packaged_case_fixture(case_name: str) -> None:
    case = load_packaged_case_fixture(case_name)

    assert case.case_id == case_name
    assert case.prompt
    assert case.reads
    assert case.checks
    assert (
        case.efficiency_policy.max_write_calls
        >= case.efficiency_policy.best_write_calls
    )


def test_create_employee_admin_fixture_keeps_existing_expectations() -> None:
    case = load_packaged_case_fixture("create_employee_admin")

    assert case.tier == 1
    assert case.expected_min_proxy_calls == 1
    assert len(case.reads) == 1
    assert len(case.checks) == 4
    assert case.efficiency_policy.best_write_calls == 1


def test_linked_entity_and_ledger_fixtures_encode_breadth_matrix_expectations() -> None:
    project_case = load_packaged_case_fixture("create_project_basic")
    journal_case = load_packaged_case_fixture("create_journal_entry_basic")
    travel_case = load_packaged_case_fixture("create_travel_expense_basic")

    assert project_case.expected_min_proxy_calls == 2
    assert project_case.reads[0].path == "/project"
    assert project_case.reads[0].query["fields"] == (
        "id,name,number,startDate,projectManager.id,projectManager.email,"
        "projectManager.firstName,projectManager.lastName"
    )
    assert project_case.checks[0].selector["projectManager.email"] == (
        "kari.project.basic@example.org"
    )
    assert project_case.checks[1].field_path == "number"

    assert journal_case.expected_min_proxy_calls == 2
    assert journal_case.reads[0].path == "/ledger/voucher"
    assert journal_case.reads[0].query["dateFrom"] == "2026-03-01"
    assert journal_case.reads[0].query["dateTo"] == "2026-03-31"
    assert journal_case.checks[2].field_path == "postings.0.amountGrossCurrency"
    assert journal_case.checks[2].expected == 1500
    assert journal_case.checks[3].field_path == "postings.1.amountGrossCurrency"
    assert journal_case.checks[3].expected == -1500

    assert travel_case.expected_min_proxy_calls == 2
    assert travel_case.reads[0].path == "/travelExpense"
    assert travel_case.reads[0].query["fields"] == (
        "id,title,employee.id,employee.email,employee.firstName,employee.lastName"
    )
    assert travel_case.checks[0].selector["employee.email"] == (
        "kari.travel.basic@example.org"
    )
    assert travel_case.checks[1].field_path == "title"
    assert travel_case.checks[2].field_path == "employee.email"


def test_create_customer_fixture_has_explicit_tier1_field_checks() -> None:
    case = load_packaged_case_fixture("create_customer")

    assert case.tier == 1
    assert case.expected_min_proxy_calls == 1
    assert len(case.reads) == 1
    assert case.reads[0].path == "/customer"
    assert case.reads[0].query["name"] == "Nordlys Testkunde Create Customer AS"
    assert case.reads[0].query["fields"] == "id,name,email,phoneNumber"
    assert [check.kind for check in case.checks] == [
        "entity_exists",
        "field_equals",
        "field_equals",
        "field_equals",
    ]
    assert case.checks[0].selector == {
        "name": "Nordlys Testkunde Create Customer AS",
        "email": "kunde.create_customer@example.org",
        "phoneNumber": "40010001",
    }
    assert [check.field_path for check in case.checks[1:]] == [
        "name",
        "email",
        "phoneNumber",
    ]
    assert case.efficiency_policy.best_write_calls == 1
    assert case.efficiency_policy.max_write_calls == 1
    assert case.efficiency_policy.max_4xx_errors == 0


def test_create_product_fixture_has_explicit_tier1_field_checks() -> None:
    case = load_packaged_case_fixture("create_product")

    assert case.tier == 1
    assert case.expected_min_proxy_calls == 1
    assert len(case.reads) == 1
    assert case.reads[0].path == "/product"
    assert case.reads[0].query["name"] == "Reisetid Konsulentpakke Create Product"
    assert case.reads[0].query["fields"] == "id,name,number,priceExcludingVatCurrency"
    assert [check.kind for check in case.checks] == [
        "entity_exists",
        "field_equals",
        "field_equals",
        "field_equals",
    ]
    assert case.checks[0].selector == {
        "name": "Reisetid Konsulentpakke Create Product",
        "number": "PROD-CREATE-PRODUCT-001",
    }
    assert [check.field_path for check in case.checks[1:]] == [
        "name",
        "number",
        "priceExcludingVatCurrency",
    ]
    assert case.checks[3].expected == 1500.0
    assert case.efficiency_policy.best_write_calls == 1
    assert case.efficiency_policy.max_write_calls == 1
    assert case.efficiency_policy.max_4xx_errors == 0


def test_bank_reconciliation_fixture_includes_placeholder_file() -> None:
    case = load_packaged_case_fixture("bank_reconciliation_file")

    assert len(case.files) == 1
    assert case.files[0].filename == "bank_reconciliation_file_placeholder.csv"
    assert case.files[0].mime_type == "text/csv"
    assert (
        case.files[0]
        .decoded_content()
        .decode("utf-8")
        .startswith("BANK-RECONCILIATION-FILE-001")
    )
    assert "BANK-RECONCILIATION-FILE-001" in case.prompt
    assert case.reads[0].query["description"] == "BANK-RECONCILIATION-FILE-001"


def test_supplier_invoice_fixture_carries_embedded_pdf_and_stable_anchors() -> None:
    case = load_packaged_case_fixture("supplier_invoice_basic")

    assert len(case.files) == 1
    assert case.files[0].filename == "supplier_invoice_basic_placeholder.pdf"
    assert case.files[0].mime_type == "application/pdf"
    assert case.files[0].decoded_content().startswith(b"%PDF-1.4")
    assert case.reads[0].query["invoiceNumber"] == "SUPPLIER-INVOICE-BASIC-001"
    assert case.checks[-1].field_path == "supplier.name"
    assert case.checks[-1].expected == "Leverandor PDF AS"
    _assert_future_multimodal_file_policy(
        filename=case.files[0].filename,
        mime_type=case.files[0].mime_type,
        content_base64=case.files[0].content_base64,
    )


def test_unsupported_mime_policy_expectation_is_explicit(tmp_path: Path) -> None:
    fixture_path = _write_fixture(
        tmp_path,
        files=[
            {
                "filename": "fixture_payload.bin",
                "content_base64": base64.b64encode(b"fixture-payload").decode("ascii"),
                "mime_type": "application/x-msdownload",
            }
        ],
    )
    case = load_case_fixture(fixture_path)

    with pytest.raises(ValueError, match=r"Unsupported file policy expectation"):
        _assert_future_multimodal_file_policy(
            filename=case.files[0].filename,
            mime_type=case.files[0].mime_type,
            content_base64=case.files[0].content_base64,
        )


def test_oversized_file_policy_expectation_is_explicit(tmp_path: Path) -> None:
    fixture_path = _write_fixture(
        tmp_path,
        files=[
            {
                "filename": "fixture_payload.pdf",
                "content_base64": base64.b64encode(
                    b"a" * (MAX_INLINE_FILE_BYTES + 1)
                ).decode("ascii"),
                "mime_type": "application/pdf",
            }
        ],
    )
    case = load_case_fixture(fixture_path)

    with pytest.raises(ValueError, match=r"Oversized file policy expectation"):
        _assert_future_multimodal_file_policy(
            filename=case.files[0].filename,
            mime_type=case.files[0].mime_type,
            content_base64=case.files[0].content_base64,
        )


def test_create_invoice_basic_fixture_uses_date_ranged_order_and_invoice_reads() -> (
    None
):
    case = load_packaged_case_fixture("create_invoice_basic")

    assert case.expected_min_proxy_calls == 3
    assert case.efficiency_policy.best_write_calls == 3
    assert case.efficiency_policy.max_4xx_errors == 0

    reads_by_name = {read.name: read for read in case.reads}
    assert set(reads_by_name) == {"customers", "orders", "invoices"}
    assert reads_by_name["orders"].query["orderDateFrom"] == "2024-01-01"
    assert reads_by_name["orders"].query["orderDateTo"] == "2026-12-31"
    assert reads_by_name["invoices"].query["invoiceDateFrom"] == "2024-01-01"
    assert reads_by_name["invoices"].query["invoiceDateTo"] == "2026-12-31"
    assert "orderLines" in str(reads_by_name["orders"].query["fields"])

    check_names = {check.name for check in case.checks}
    assert "Invoice for linked customer found" in check_names
    assert "Invoice links the intended order" in check_names


def test_invoice_with_payment_fixture_tracks_payment_flow_with_date_ranges() -> None:
    case = load_packaged_case_fixture("invoice_with_payment")

    assert case.expected_min_proxy_calls == 4
    assert case.efficiency_policy.best_write_calls == 4
    assert case.efficiency_policy.max_4xx_errors == 0

    reads_by_name = {read.name: read for read in case.reads}
    assert set(reads_by_name) == {"customers", "orders", "invoices"}
    assert reads_by_name["orders"].query["orderDateFrom"] == "2024-01-01"
    assert reads_by_name["orders"].query["orderDateTo"] == "2026-12-31"
    assert reads_by_name["invoices"].query["invoiceDateFrom"] == "2024-01-01"
    assert reads_by_name["invoices"].query["invoiceDateTo"] == "2026-12-31"
    assert reads_by_name["invoices"].query["fields"] == (
        "id,invoiceNumber,invoiceDate,paymentDate,amountDue,paidAmount,isPaid,"
        "customer.id,customer.name,orders"
    )

    payment_check = next(
        check
        for check in case.checks
        if check.name == "Invoice payment marks invoice as paid"
    )
    assert payment_check.kind == "field_equals"
    assert payment_check.field_path == "isPaid"
    assert payment_check.expected is True


def test_load_case_fixture_raises_value_error_for_malformed_fixture(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "malformed_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "case_id": "malformed_fixture",
                "description": "Missing checks should fail loader validation.",
                "tier": 1,
                "prompt": "Opprett noe ugyldig.",
                "files": [],
                "verification": {
                    "reads": [
                        {
                            "name": "entities",
                            "path": "/customer",
                            "query": {"name": "Malformed Fixture AS"},
                            "mode": "list_values",
                        }
                    ],
                    "checks": [],
                },
                "efficiency": {
                    "best_write_calls": 1,
                    "max_write_calls": 2,
                    "max_4xx_errors": 0,
                    "write_weight": 0.7,
                    "error_weight": 0.3,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match=r"verification\.checks must be a non-empty list"
    ):
        load_case_fixture(fixture_path)
