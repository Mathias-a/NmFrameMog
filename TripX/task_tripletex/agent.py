# mypy: disable-error-code="any, explicit-any, attr-defined, unused-ignore, arg-type, no-any-unimported, union-attr"
from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, cast

from google import genai  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from google.genai import types  # pyright: ignore[reportMissingImports]

from task_tripletex.client import TripletexClient
from task_tripletex.models import (
    RequestBudget,
    RequestContext,
    SolveExecutionOutcome,
    SolveFile,
    SolveRequest,
    StructuredOperation,
)
from task_tripletex.task_log import task_logger

execute_api_func = {
    "name": "execute_tripletex_api",
    "description": (
        "Calls the Tripletex v2 REST API. Paths are RELATIVE to a pre-configured base URL "
        "(e.g. use '/employee' NOT 'https://...' and NOT '/v2/employee'). "
        "Authentication is pre-configured. Wait for response to extract IDs before dependent calls."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "method": {
                "type": "STRING",
                "description": "HTTP method: GET, POST, PUT, DELETE",
            },
            "path": {
                "type": "STRING",
                "description": "RELATIVE path only, e.g. '/employee', '/customer/123', '/order/orderline'. Never use absolute URLs or '/v2/' prefix.",
            },
            "query": {
                "type": "OBJECT",
                "description": 'Query parameters as key-value pairs. E.g. {"fields": "id,name", "count": 100}',
            },
            "body": {
                "type": "OBJECT",
                "description": "JSON payload for POST/PUT requests.",
            },
        },
        "required": ["method", "path"],
    },
}

API_KEY = "AIzaSyDXWbMGEpkYrJUpf-qArxglOZrN56GTnr8"
ENDPOINT_TIMEOUT_SECONDS = 300
RESERVED_HEADROOM_SECONDS = 30
EXECUTION_BUDGET_SECONDS = ENDPOINT_TIMEOUT_SECONDS - RESERVED_HEADROOM_SECONDS
MAX_MODEL_TURNS = 24
MAX_TOOL_CALLS = 48
MAX_SHAPED_LIST_VALUES = 10
MAX_REPAIR_HINT_VALIDATION_MESSAGES = 3
MAX_REPAIRS_PER_OBJECTIVE = 2
MODEL_NAME = "gemini-3.1-pro-preview"
MODEL_TEMPERATURE = 0.0
MODEL_TOP_P = 1.0
MODEL_CANDIDATE_COUNT = 1
MODEL_RANDOM_SEED = 0
SUPPORTED_INLINE_FILE_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/webp",
        "text/csv",
        "text/plain",
    }
)
_TEXT_MIME_TYPES = frozenset({"text/csv", "text/plain"})
MAX_INLINE_FILE_BYTES = 20 * 1024 * 1024

# ---------------------------------------------------------------------------
# Task classification & profiling
# ---------------------------------------------------------------------------

_SEND_MAX_RETRIES = 2
_SEND_RETRY_BASE_DELAY_SECONDS = 2.0
_TRANSIENT_ERROR_PATTERNS = re.compile(
    r"429|500|503|quota|rate|resource.?exhausted|timeout|deadline|overloaded|unavailable",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TaskProfile:
    task_type: str
    max_writes: int
    max_turns: int
    max_tool_calls: int
    allowed_write_prefixes: frozenset[str]
    task_preamble: str


TASK_PROFILES: dict[str, TaskProfile] = {
    "employee": TaskProfile(
        task_type="employee",
        max_writes=3,
        max_turns=12,
        max_tool_calls=8,
        allowed_write_prefixes=frozenset({"/employee", "/department"}),
        task_preamble=(
            "This is an EMPLOYEE creation task. Follow this exact pattern:\n"
            "1. GET /department?fields=id,name&count=1 to find an existing department.\n"
            "2. POST /employee with ALL fields from the prompt (firstName, lastName, email, "
            "userType, dateOfBirth, department ref, etc.).\n"
            "3. Say TASK COMPLETED. Do NOT create anything else."
        ),
    ),
    "customer": TaskProfile(
        task_type="customer",
        max_writes=1,
        max_turns=6,
        max_tool_calls=4,
        allowed_write_prefixes=frozenset({"/customer"}),
        task_preamble=(
            "This is a CUSTOMER creation task. Follow this exact pattern:\n"
            "1. POST /customer with ALL fields mentioned in the prompt (name, email, "
            "phoneNumber, organizationNumber, etc.).\n"
            "2. Say TASK COMPLETED. Do NOT create contacts, orders, invoices, or anything else."
        ),
    ),
    "product": TaskProfile(
        task_type="product",
        max_writes=1,
        max_turns=6,
        max_tool_calls=4,
        allowed_write_prefixes=frozenset({"/product"}),
        task_preamble=(
            "This is a PRODUCT creation task. Follow this exact pattern:\n"
            "1. POST /product with ALL fields from the prompt (name, price, vatType, etc.).\n"
            "2. Say TASK COMPLETED. Do NOT create anything else."
        ),
    ),
    "project": TaskProfile(
        task_type="project",
        max_writes=5,
        max_turns=14,
        max_tool_calls=12,
        allowed_write_prefixes=frozenset({"/project", "/employee"}),
        task_preamble=(
            "This is a PROJECT creation task. Follow this exact pattern:\n"
            "1. GET existing employee for project manager (if needed).\n"
            "2. GET /employee/entitlement?fields=customer(id)&count=1 to find company customer ID.\n"
            "3. POST /employee/entitlement with entitlementId 45 (AUTH_CREATE_PROJECT).\n"
            "4. POST /employee/entitlement with entitlementId 10 (AUTH_PROJECT_MANAGER).\n"
            "5. POST /project with name, projectManager, startDate, and other fields from prompt.\n"
            "6. Say TASK COMPLETED."
        ),
    ),
    "invoice": TaskProfile(
        task_type="invoice",
        max_writes=6,
        max_turns=16,
        max_tool_calls=14,
        allowed_write_prefixes=frozenset(
            {"/customer", "/order", "/invoice", "/ledger"}
        ),
        task_preamble=(
            "This is an INVOICE creation task. Follow this exact pattern:\n"
            "1. POST /customer (if needed) with ALL fields from the prompt.\n"
            "2. POST /order with inline orderLines (include vatType on each line).\n"
            "3. GET /ledger/account?number=1920 to check bank account — PUT if empty.\n"
            "4. POST /invoice with invoiceDate, invoiceDueDate, orders ref.\n"
            "5. Say TASK COMPLETED. Do NOT create unrelated entities."
        ),
    ),
    "invoice_with_payment": TaskProfile(
        task_type="invoice_with_payment",
        max_writes=7,
        max_turns=18,
        max_tool_calls=16,
        allowed_write_prefixes=frozenset(
            {"/customer", "/order", "/invoice", "/ledger"}
        ),
        task_preamble=(
            "This is an INVOICE WITH PAYMENT task. Follow this exact pattern:\n"
            "1. POST /customer (if needed).\n"
            "2. POST /order with inline orderLines (include vatType on each line).\n"
            "3. GET /ledger/account?number=1920 — PUT if bankAccountNumber empty.\n"
            "4. GET /invoice/paymentType to find 'Betalt til bank' payment type ID.\n"
            "5. POST /invoice.\n"
            "6. PUT /invoice/{id}/:payment with paymentDate, paymentTypeId, paidAmount.\n"
            "7. Say TASK COMPLETED."
        ),
    ),
    "journal_entry": TaskProfile(
        task_type="journal_entry",
        max_writes=2,
        max_turns=10,
        max_tool_calls=8,
        allowed_write_prefixes=frozenset({"/ledger"}),
        task_preamble=(
            "This is a JOURNAL ENTRY / VOUCHER task. Follow this exact pattern:\n"
            "1. GET /ledger/account with ALL needed account numbers in ONE call "
            "(e.g. ?number=5000,5200,2930&fields=id,number&count=10).\n"
            "2. POST /ledger/voucher with ONE voucher containing ALL postings "
            "(row starting at 1, balanced debits and credits).\n"
            "3. Say TASK COMPLETED."
        ),
    ),
    "travel_expense": TaskProfile(
        task_type="travel_expense",
        max_writes=4,
        max_turns=14,
        max_tool_calls=12,
        allowed_write_prefixes=frozenset(
            {"/travelExpense", "/employee", "/department"}
        ),
        task_preamble=(
            "This is a TRAVEL EXPENSE task. Follow this exact pattern:\n"
            "1. GET /employee?email=...&fields=id,firstName,lastName,email&count=1 to find the employee.\n"
            "2. If not found, GET /department?fields=id,name&count=1 to find a department, then "
            "POST /employee with firstName, lastName, email, userType=1, dateOfBirth='1990-01-15', "
            "department ref.\n"
            "3. POST /travelExpense with employee ref and title from the prompt.\n"
            "4. Say TASK COMPLETED. Do NOT create anything else."
        ),
    ),
    "supplier_invoice": TaskProfile(
        task_type="supplier_invoice",
        max_writes=4,
        max_turns=14,
        max_tool_calls=12,
        allowed_write_prefixes=frozenset({"/supplier", "/supplierInvoice", "/ledger"}),
        task_preamble=(
            "This is a SUPPLIER INVOICE task. Follow this exact pattern:\n"
            "1. Check if supplier exists: GET /supplier?organizationNumber=...&fields=id,name.\n"
            "2. POST /supplier if not found.\n"
            "3. GET /ledger/account for expense (e.g. 4000) and AP (2400) accounts in ONE call.\n"
            "4. POST /supplierInvoice with nested voucher structure.\n"
            "5. Say TASK COMPLETED."
        ),
    ),
    "bank_reconciliation": TaskProfile(
        task_type="bank_reconciliation",
        max_writes=8,
        max_turns=18,
        max_tool_calls=20,
        allowed_write_prefixes=frozenset({"/ledger"}),
        task_preamble=(
            "This is a BANK RECONCILIATION task.\n"
            "1. Parse the attached file for transactions.\n"
            "2. GET /ledger/account for bank (1920) and contra accounts.\n"
            "3. POST /ledger/voucher for each transaction group (balanced postings).\n"
            "4. Say TASK COMPLETED."
        ),
    ),
    "generic": TaskProfile(
        task_type="generic",
        max_writes=10,
        max_turns=MAX_MODEL_TURNS,
        max_tool_calls=MAX_TOOL_CALLS,
        allowed_write_prefixes=frozenset(),  # empty = allow all
        task_preamble="",
    ),
}


_TASK_KEYWORDS: list[tuple[str, list[str]]] = [
    ("invoice_with_payment", ["faktura", "betaling"]),
    ("invoice_with_payment", ["invoice", "payment"]),
    ("supplier_invoice", ["leverandør", "faktura"]),
    ("supplier_invoice", ["supplier", "invoice"]),
    ("supplier_invoice", ["leverandørfaktura"]),
    ("bank_reconciliation", ["bank", "avstem"]),
    ("bank_reconciliation", ["bank", "reconcil"]),
    ("journal_entry", ["bilag"]),
    ("journal_entry", ["journal", "entry"]),
    ("journal_entry", ["bokfør", "lønn"]),
    ("journal_entry", ["voucher"]),
    ("travel_expense", ["reiseregning"]),
    ("travel_expense", ["reise", "regning"]),
    ("travel_expense", ["travel", "expense"]),
    ("invoice", ["faktura"]),
    ("invoice", ["invoice"]),
    ("project", ["prosjekt"]),
    ("project", ["project"]),
    ("employee", ["ansatt"]),
    ("employee", ["employee"]),
    ("customer", ["kunde"]),
    ("customer", ["customer"]),
    ("product", ["produkt"]),
    ("product", ["product"]),
]


def classify_task(prompt: str) -> TaskProfile:
    lower_prompt = prompt.lower()
    for task_type, keywords in _TASK_KEYWORDS:
        if all(kw in lower_prompt for kw in keywords):
            return TASK_PROFILES[task_type]
    return TASK_PROFILES["generic"]


def _is_write_method(method: str) -> bool:
    return method in {"POST", "PUT", "PATCH", "DELETE"}


def _is_write_allowed(path: str, allowed_prefixes: frozenset[str]) -> bool:
    # Empty allowed_prefixes = all writes allowed (generic profile)
    if not allowed_prefixes:
        return True
    normalized = path.split("?", 1)[0]
    return any(normalized.startswith(prefix) for prefix in allowed_prefixes)


SYSTEM_PROMPT = """\
You are an AI Accounting Agent for Tripletex. You receive a natural language prompt \
(in Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French) and must \
execute the correct Tripletex v2 REST API calls to complete the accounting task.

## CRITICAL RULES

1. **Paths are RELATIVE.** Use `/employee`, `/customer`, etc. NEVER use absolute URLs, \
NEVER prefix with `/v2/`. The base URL is pre-configured.
2. **Authentication is pre-configured.** Do NOT set auth headers or tokens yourself.
3. **Plan before calling.** Analyze the prompt fully, determine ALL required API calls \
and their order, then execute. Avoid trial-and-error.
4. **Every 4xx error hurts your score.** Get it right the first time.
5. **Minimize API calls.** Don't fetch entities you don't need. After creating something, \
you already have its ID from the response — don't GET it again.
6. **Say 'TASK COMPLETED' when done.** This exact phrase signals success.
7. **ALWAYS include vatType on order lines.** Every order line MUST have a `vatType` reference. \
Default to `"vatType": {"id": 6}` (no VAT) unless the prompt explicitly requires VAT. If the prompt \
requires 25% MVA, use `{"id": 3}` but fall back to `{"id": 6}` if it's rejected. NEVER omit vatType.
8. **ALWAYS include row on voucher postings.** Every posting needs a `"row"` field starting \
at 1. Row 0 is system-reserved. Also include `vatType` on VAT-locked accounts.
9. **Do NOT duplicate inline order lines.** If you included `orderLines` in the POST /order \
body, do NOT also POST /order/orderline — those lines already exist.
10. **Before creating an invoice, ensure bank account is configured.** Check ledger account 1920 \
for bankAccountNumber. If empty, set it via PUT /ledger/account before POST /invoice.
11. **Before creating a project, ensure PM has entitlements.** Grant entitlements 45 then 10 \
via POST /employee/entitlement before POST /project.
12. **Include ALL fields mentioned in the prompt.** Every detail in the prompt (name, email, phone, \
address, organization number, dates, etc.) MUST appear in your API payload. Omitting mentioned fields \
causes scoring failures even if the entity is created successfully.
13. **STOP when the requested task is done.** Do NOT create additional entities beyond what the prompt \
asks for. If asked to create a customer, create ONLY the customer — do NOT also create departments, \
employees, contacts, or anything else. Re-read the prompt before each API call and ask yourself: \
"Did the user ask for this?" If no, STOP and say TASK COMPLETED.
14. **Never use `fields=*` in queries.** Always specify the exact fields you need, e.g. \
`fields=id,name,number`. Using `fields=*` wastes bandwidth and processing time.
15. **Batch account lookups.** When you need multiple ledger accounts, fetch them in ONE call: \
`GET /ledger/account?number=5000,5200,2930&fields=id,number&count=10` — NOT separate calls per account.
16. **One voucher, multiple postings.** For payroll journal entries, manual journals, and any task \
requiring multiple debit/credit lines, create ONE voucher with ALL postings in a single POST — \
NOT multiple vouchers with one posting each.
17. **Don't create departments unless explicitly asked.** Only create departments if the prompt \
specifically requests it, or if employee creation fails due to a missing department and no departments exist.

## RESPONSE FORMAT

- Single entity: `{"value": {...}}`
- List: `{"count": N, "from": 0, "values": [...]}`
- Error: `{"status": 422, "validationMessages": [{"field": "...", "message": "..."}]}`

## QUERY PARAMETERS

- `fields=id,name,email` to select only the fields you need (NEVER use `fields=*`)
- `count=100&from=0` for pagination
- Search filters vary by endpoint (see below)

## ENTITY REFERENCES

Use `{"id": 123}` for relationships. Examples:
- `"customer": {"id": 456}` — link to customer
- `"department": {"id": 789}` — link to department
- `"projectManager": {"id": 101}` — link to employee

---

## ENDPOINT REFERENCE (verified against live OpenAPI spec v2.74.00)

### POST /employee
Required: `firstName`, `lastName`, `userType` (integer), `email`, `department` (ref)
- `userType`: 1 = standard user, 2 = account administrator
- You MUST get an existing department first: `GET /department?fields=id,name&count=1`
- `dateOfBirth` ("YYYY-MM-DD"): required for Norwegian employees (A-melding backend)
- Optional: `phoneNumberMobile`, `nationalIdentityNumber` (11 digits), `address` (object with \
`addressLine1`, `postalCode`, `city`, `country` ref)

**IMPORTANT:** There is NO `isAdministrator` field on Employee. To make someone admin, use \
`userType: 2`. For more granular entitlements: \
`PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=X&template=ADMIN`

Example — create standard employee:
```json
{"firstName": "Ola", "lastName": "Nordmann", "email": "ola@example.org", "userType": 1, \
"dateOfBirth": "1990-01-15", "department": {"id": DEPT_ID}}
```

Example — create administrator:
```json
{"firstName": "Ola", "lastName": "Nordmann", "email": "ola@example.org", "userType": 2, \
"dateOfBirth": "1990-01-15", "department": {"id": DEPT_ID}}
```

GET search params: `firstName`, `lastName`, `email`, `fields`, `count`, `from`

### POST /customer
Required: `name`
Common: `email`, `phoneNumber`, `organizationNumber`, `language` ("NO"/"EN"), \
`invoiceSendMethod` ("EMAIL"/"EHF"/"PAPER"), `isPrivateIndividual` (boolean)

**CRITICAL — Include ALL fields from the prompt:** When the prompt mentions an email address, \
phone number, organization number, or any other customer detail, you MUST include ALL of them \
in the POST body. Do NOT omit fields that appear in the prompt — every detail is scored.

**IMPORTANT:** If `isPrivateIndividual=false` AND `invoiceSendMethod=EHF`, then `postalAddress` \
is required. Set `invoiceSendMethod` to "EMAIL" to avoid this requirement.

`isCustomer` is **readOnly** on the `/customer` endpoint — do NOT send it.
`isSupplier` can be set to make the customer also act as a supplier.

Example:
```json
{"name": "Acme AS", "email": "post@acme.no", "phoneNumber": "99887766"}
```

GET search params: `name`, `email`, `organizationNumber`, `customerNumber`, `fields`, `count`

### POST /supplier
Required: `name`
Common: `email`, `phoneNumber`, `organizationNumber`, `isPrivateIndividual` (boolean)

`isSupplier` is **readOnly** on `/supplier` — do NOT send it.

### Creating supplier invoices — POST /supplierInvoice

**Use `POST /supplierInvoice` to create supplier invoices.** This endpoint requires a nested \
`voucher` object with `description` and balanced double-entry `postings`. Do NOT use \
`POST /incomingInvoice` — that BETA endpoint returns 403 Forbidden on most sandboxes.

**CRITICAL — When registering a supplier invoice from an attached PDF:**
1. Read the PDF carefully and extract: supplier name, org number, invoice number, invoice date, \
   due date, total amount (including VAT), and any account references (e.g. account 4000).
2. First check if the supplier exists: `GET /supplier?organizationNumber=...&fields=id,name&count=5` \
   (or by name if no org number).
3. If the supplier doesn't exist, create it: `POST /supplier` with `name` (and optionally \
   `organizationNumber` if the PDF shows it).
4. Look up the expense account ID: `GET /ledger/account?number=4000&fields=id,number&count=1`
5. Look up the AP (accounts payable) account ID: `GET /ledger/account?number=2400&fields=id,number&count=1`
6. Then create the supplier invoice: `POST /supplierInvoice` with the correct nested structure.

**POST /supplierInvoice body structure:**
```json
{
  "invoiceNumber": "SINV-001",
  "invoiceDate": "2026-03-15",
  "invoiceDueDate": "2026-04-15",
  "supplier": {"id": SUPPLIER_ID},
  "voucher": {
    "date": "2026-03-15",
    "description": "Leverandorfaktura SINV-001",
    "postings": [
      {
        "date": "2026-03-15",
        "account": {"id": EXPENSE_ACCOUNT_ID},
        "supplier": {"id": SUPPLIER_ID},
        "amountGross": 8500.0,
        "amountGrossCurrency": 8500.0
      },
      {
        "date": "2026-03-15",
        "account": {"id": AP_ACCOUNT_2400_ID},
        "supplier": {"id": SUPPLIER_ID},
        "amountGross": -8500.0,
        "amountGrossCurrency": -8500.0
      }
    ]
  }
}
```

**Field reference for `POST /supplierInvoice`:**
- `invoiceNumber`: string — the supplier's invoice number (REQUIRED)
- `invoiceDate`: "YYYY-MM-DD" — the invoice date (REQUIRED)
- `invoiceDueDate`: "YYYY-MM-DD" — the due/forfall date (optional)
- `supplier`: object ref `{"id": SUPPLIER_ID}` — the supplier reference (REQUIRED)
- `voucher`: nested object (REQUIRED) containing:
  - `date`: "YYYY-MM-DD" — voucher date (same as invoiceDate)
  - `description`: string — voucher description (REQUIRED, cannot be null)
  - `postings`: array of posting objects (REQUIRED, cannot be null). Must be balanced (sum to zero).

**Field reference for `voucher.postings` entries:**
- `date`: "YYYY-MM-DD" — posting date
- `account`: object ref `{"id": ACCOUNT_ID}` — the ledger account
- `supplier`: object ref `{"id": SUPPLIER_ID}` — link posting to the supplier
- `amountGross`: number — gross amount (positive for debit/expense, negative for credit/AP)
- `amountGrossCurrency`: number — same as amountGross for NOK transactions

**Double-entry pattern:**
- Debit line: expense account (e.g. 4000 Varekjøp) with positive `amountGross` = total incl. VAT
- Credit line: AP account (2400 Leverandørgjeld) with negative `amountGross` = -(total incl. VAT)
- The two postings MUST sum to zero.

**IMPORTANT — Field name is `invoiceDueDate`, NOT `dueDate`.** Sending `dueDate` causes 422 \
"Feltet eksisterer ikke i objektet". Use `invoiceDueDate` for the due/forfall date.

**IMPORTANT — Do NOT send `amount` or `amountCurrency`.** These are read-only computed fields. \
Sending them causes 500 server errors.

**After creating**, verify with `GET /supplierInvoice?invoiceDateFrom=...&invoiceDateTo=...` — \
the supplier invoice will appear there.

Guidance:
- If the prompt gives a supplier by name/org number and you do not already have its ID, find or
  create the supplier first; otherwise reuse the supplied ID/reference directly.
- The voucher postings MUST include the `supplier` ref on BOTH lines (debit and credit) to properly \
  link the invoice to the supplier in the ledger.

### POST /purchaseOrder
Use purchase orders for procurement workflows before any supplier invoice or goods receipt step.

Guidance:
- Prefer purchase orders when the prompt is about ordering from a supplier, awaiting delivery, or
  matching later incoming invoices against a prior purchase document.
- As with customer orders, prefer complete request bodies with inline lines/details when the endpoint
  supports them instead of creating the header first and line items afterward.
- Do not misuse customer sales order endpoints for supplier purchasing tasks.

### POST /timesheet
Use for time registration. Common relationships are `employee` (ref), `project` (ref), `activity`
(ref), a work date, and hours/count.

Guidance:
- If the task is about logging hours, use timesheet endpoints, not vouchers.
- Resolve employee/project/activity only when needed; if the prompt clearly identifies an existing
  target and provides IDs, use them directly.
- For many rows in one request, prefer any available timesheet batch/import endpoint instead of many
  single-entry writes.

### Payroll
Use payroll/salary endpoints for payroll runs, payslips, salary lines, and employer-reporting style
tasks. Do not emulate payroll by posting general ledger vouchers unless the prompt explicitly asks
for a manual journal entry.

Guidance:
- For payroll preparation, first ensure required employees and periods are identified, then use the
  payroll-specific endpoint family.
- Prefer payroll batch/run endpoints when the prompt concerns multiple employees in the same pay run.
- Only fall back to vouchers if the user explicitly asks for manual accounting entries instead of a
  payroll workflow.
- **Manual payroll journal entry (when prompt says "bokfør lønn" or "post payroll manually"):**
  Batch-fetch ALL needed accounts in ONE call: \
  `GET /ledger/account?number=5000,5200,2930,2770,2600&fields=id,number&count=10`
  Then create ONE voucher with ALL postings:
  ```json
  {"date": "2026-03-01", "description": "Lønn mars 2026", "postings": [
    {"row": 1, "account": {"id": SALARY_ID}, "amountGross": 50000},
    {"row": 2, "account": {"id": TAX_ID}, "amountGross": -15000},
    {"row": 3, "account": {"id": NET_PAY_ID}, "amountGross": -35000}
  ]}
  ```
  Do NOT look up occupationCode or employee entitlements for voucher-based payroll tasks.
  Do NOT create separate vouchers per posting — always combine into one.

### POST /contact
Required: at least one of `firstName`, `lastName`, or `email`
Common: `customer` (ref), `phoneNumberMobile`, `phoneNumberWork`

### POST /product
Required: `name`
Common: `priceExcludingVatCurrency` (number), `vatType` (ref, id=3 for 25% MVA), \
`description`, `productUnit` (ref), `costExcludingVatCurrency`, `number` (string, your product code)

Example:
```json
{"name": "Consulting", "priceExcludingVatCurrency": 1500.0, "vatType": {"id": 3}}
```

GET search params: `name`, `number`, `fields`, `count`

### POST /order
Required: `customer` (ref), `orderDate` ("YYYY-MM-DD"), `deliveryDate` ("YYYY-MM-DD")
Common: `orderLines` (inline array), `receiverEmail`, `isClosed`, `isPrioritizeAmountsIncludingVat` (boolean), \
`number` (string — the order number shown on invoices), `reference` (string — free-text reference)

**IMPORTANT — Order number vs reference:** When the prompt mentions "ordrereferanse" (order reference), \
set BOTH the `number` field AND the `reference` field to the given value. The `number` field is what \
appears on invoices and is used for matching. If you only set `reference`, the order gets an auto-generated \
`number` which won't match the intended reference.

**VAT PRICE RULE:** If `isPrioritizeAmountsIncludingVat=true`, use `unitPriceIncludingVatCurrency` \
on order lines. If false (default), use `unitPriceExcludingVatCurrency`. Mixing causes HTTP 422.

**CRITICAL — vatType on order lines:** Every order line MUST include `vatType`. Use `{"id": 3}` for \
25% MVA if the company has the MVA module, or `{"id": 6}` (no VAT) if unsure. If you get "Ugyldig \
mva-kode" with vatType 3, immediately retry with `{"id": 6}` — the company's VAT module is not \
activated. To avoid the error entirely, default to `{"id": 6}` unless the prompt explicitly \
requires VAT calculation.

**CRITICAL — Recovery from 422 on order creation:** If `POST /order` returns a 422 error, read \
the `validationMessages` carefully. Common fixes:
- "Ugyldig mva-kode" → Change `vatType` from `{"id": 3}` to `{"id": 6}` on order lines
- "isPrioritizeAmountsIncludingVat" mismatch → If using `unitPriceExcludingVatCurrency`, ensure \
  `isPrioritizeAmountsIncludingVat` is either omitted or set to `false`
- Missing `deliveryDate` → Add `deliveryDate` (same as `orderDate` is fine)
- "Nummeret er allerede i bruk" / "number already in use" → The order number already exists from \
  a prior run. **Remove the `number` field entirely** from the payload and let Tripletex auto-generate \
  it. Keep the `reference` field set to the prompt's reference value. The auto-generated number will \
  not affect scoring — only the `reference` field matters for matching. Do NOT try to append suffixes \
  or query existing orders — just omit `number`.
Do NOT retry with the exact same payload. Fix the specific field mentioned in the error.

**CRITICAL — Do NOT duplicate order lines:** If you included `orderLines` inline when creating the \
order (POST /order), those lines are already created. Do NOT also POST separate /order/orderline calls \
for the same lines — this creates duplicates and wastes an API call.

Example (with inline order lines — preferred):
```json
{"customer": {"id": CUST_ID}, "orderDate": "2026-03-21", "deliveryDate": "2026-03-21", \
"orderLines": [{"description": "Service", "count": 1, "unitPriceExcludingVatCurrency": 1000.0, \
"vatType": {"id": 6}}]}
```

GET requires: `orderDateFrom` AND `orderDateTo` (both mandatory for listing)

To invoice an order: `PUT /order/{id}/:invoice?invoiceDate=YYYY-MM-DD`

### POST /order/orderline
Required: `order` (ref)
Common: `description`, `count` (quantity, supports decimals), `unitPriceExcludingVatCurrency`, \
`vatType` (ref), `product` (ref)

Tip: You can also include `orderLines` inline when creating the order — this saves an API call.

### POST /invoice
Required: `invoiceDate` ("YYYY-MM-DD"), `invoiceDueDate` ("YYYY-MM-DD"), `orders` (non-empty list of refs)
Optional: `comment`, `sendToCustomer` (boolean, defaults true — set false to suppress auto-send)

**Note:** Only ONE order per invoice is currently supported.

**CRITICAL — Bank account required:** `POST /invoice` will fail with "Faktura kan ikke opprettes \
før selskapet har registrert et bankkontonummer" if the company has no bank account registered. \
To fix this:
1. `GET /ledger/account?number=1920&fields=id,version,number,name,bankAccountNumber` to find the bank account
2. If `bankAccountNumber` is empty, set it with `PUT /ledger/account/{id}` including the `id`, \
`version`, `number`, `name`, and `bankAccountNumber` (use a valid Norwegian bank account number, \
e.g. "12345678903" — 11 digits).
3. Then retry `POST /invoice`.

Do this PROACTIVELY before creating the invoice if you suspect the bank account is not configured.

Example:
```json
{"invoiceDate": "2026-03-21", "invoiceDueDate": "2026-04-04", "orders": [{"id": ORDER_ID}]}
```

GET requires: `invoiceDateFrom` AND `invoiceDateTo` (both mandatory for listing)

Credit note: `PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD`
Payment: `PUT /invoice/{id}/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=X&paidAmount=Y`

**Invoice payment details:** To register a payment on an invoice:
- `paymentDate`: ISO date of the payment (use today if not specified)
- `paymentTypeId`: Payment type ID. Use `GET /invoice/paymentType?count=10` to find available types. \
  Look for "Betalt til bank" (bank payment) in the results — use that ID. \
  Do NOT guess IDs like 0 or 1 — these will cause 500 errors. Always look up the real ID first.
- `paidAmount`: Amount paid. For full payment, use the invoice's `amount` (total including VAT).
- `paidAmountCurrency`: Same as paidAmount for NOK invoices.
Example: `PUT /invoice/12345/:payment?paymentDate=2026-03-21&paymentTypeId=LOOKED_UP_ID&paidAmount=2400.0&paidAmountCurrency=2400.0`

### POST /project
Required: `name`, `projectManager` (ref to employee), `startDate` ("YYYY-MM-DD")
Common: `customer` (ref), `endDate`, `isInternal` (boolean), `isFixedPrice` (boolean), \
`fixedprice` (number), `description`, `number` (auto-generated if null)

**CRITICAL — Project manager access:** The employee referenced as `projectManager` MUST have \
project manager entitlements in Tripletex. If `POST /project` returns a 422 saying the project manager \
"har ikke fått tilgang som prosjektleder", you MUST grant PM entitlements via `POST /employee/entitlement` \
BEFORE retrying. The entitlements have a dependency chain — you must grant them in order:

**Step 1 — Grant AUTH_CREATE_PROJECT (entitlementId 45):**
```json
POST /employee/entitlement
{"employee": {"id": EMPLOYEE_ID}, "entitlementId": 45, "customer": {"id": COMPANY_CUSTOMER_ID}}
```

**Step 2 — Grant AUTH_PROJECT_MANAGER (entitlementId 10):**
```json
POST /employee/entitlement
{"employee": {"id": EMPLOYEE_ID}, "entitlementId": 10, "customer": {"id": COMPANY_CUSTOMER_ID}}
```

**IMPORTANT:** You must grant entitlementId 45 FIRST, then 10. Granting 10 without 45 returns a 422 \
saying "Tilgangen 'Bruker kan være prosjektleder' krever tilgang til å opprette nye prosjekter."

**Finding the company customer ID:** Use `GET /employee/entitlement?fields=customer(id)&count=1` \
and use the `customer.id` from any returned entitlement. Do NOT use `GET /company` — it's not available.

**Proactive approach:** When the task involves creating a project with a specific employee as PM, \
ALWAYS grant entitlements (45 then 10) BEFORE attempting `POST /project`. Don't wait for a 422 error.

Example:
```json
{"name": "Web Redesign", "projectManager": {"id": EMP_ID}, "startDate": "2026-03-21"}
```

Project participants: `POST /project/participant` with `employee` ref + `adminAccess` boolean
Project activities: `POST /project/projectActivity` with `activity` ref + `project` ref

GET search params: `name`, `projectManagerId`, `customerId`, `isClosed`, `fields`, `count`

### POST /employee/entitlement
Grants a specific entitlement (permission) to an employee. Required fields: `employee` (ref), \
`entitlementId` (integer), `customer` (ref — the company's own customer ID).

**Finding the company customer ID:** Call `GET /employee/entitlement?fields=customer(id)&count=1` \
and use the `customer.id` from any returned entitlement. This is the company's internal customer \
reference, NOT a regular customer. Do NOT use `GET /company` — that endpoint is not available.

**Common entitlementIds:**
- 45 = AUTH_CREATE_PROJECT (can create projects)
- 10 = AUTH_PROJECT_MANAGER (can be assigned as project manager)
- 92 = AUTH_PROJECT_INFO (read-only project access)

Example:
```json
POST /employee/entitlement
{"employee": {"id": EMP_ID}, "entitlementId": 45, "customer": {"id": COMPANY_CUSTOMER_ID}}
```

### POST /department
Required: `name`
Common: `departmentNumber`, `departmentManager` (ref to employee)

**Note:** If department functionality has been activated, ALL new employees must have a `department`.

### POST /travelExpense
Required: `employee` (ref)
Common: `title`, `project` (ref), `department` (ref)

Sub-objects (per diem, mileage) are separate endpoints:
- Per diem rates: `GET /travelExpense/rate`
- Create voucher from expense: `PUT /travelExpense/:createVouchers?date=YYYY-MM-DD`

Example:
```json
{"employee": {"id": EMP_ID}, "title": "Bergen trip"}
```

DELETE: `DELETE /travelExpense/{id}`

### GET /ledger/account
Query: `number` (account number, e.g. 1920), `fields`, `count`
No POST — accounts are pre-defined in the chart of accounts.

**Ledger types:** Accounts with `ledgerType=CUSTOMER` require `customer` in voucher postings. \
`VENDOR` requires `supplier`. `EMPLOYEE` requires `employee`. Missing these causes "mangler" errors.

### POST /ledger/voucher
Required: `description`, `date` ("YYYY-MM-DD"), `postings` (non-empty list that MUST balance)
Each posting: `row` (integer, **MUST start at 1**), `amountGross` (positive=debit, negative=credit), \
`amountGrossCurrency` (same value for NOK), `account` (ref), optional `vatType` (ref)

**CRITICAL — ROW FIELD:** Every posting MUST include a `row` field. Number rows starting from 1 \
(row 0 is reserved for system-generated VAT postings and will cause a 422 error if used or omitted).
**CRITICAL — VAT-LOCKED ACCOUNTS:** Some accounts (e.g. 3000 Salgsinntekt) are locked to a specific \
VAT code (e.g. vatType id 3 for 25% MVA). When posting to such accounts, you MUST include the \
`vatType` reference. If unsure whether an account requires a vatType, fetch it with `fields=*` and \
check its `vatType` field. Omitting vatType on a locked account causes "Kontoen X er låst til mva-kode Y".
**IMPORTANT:** Use `amountGross` and `amountGrossCurrency`, NOT `amount`. Amounts auto-round to 2 decimals.
**IMPORTANT:** For customer ledger accounts, include `customer` ref in posting. For vendor accounts, \
include `supplier` ref. For employee accounts, include `employee` ref.

Example:
```json
{
  "description": "Manual journal entry",
  "date": "2026-03-21",
  "postings": [
    {"row": 1, "amountGross": 1000.0, "amountGrossCurrency": 1000.0, "account": {"id": DEBIT_ACCT_ID}},
    {"row": 2, "amountGross": -1000.0, "amountGrossCurrency": -1000.0, "account": {"id": CREDIT_ACCT_ID}, "vatType": {"id": 3}}
  ]
}
```

Reverse voucher: `PUT /ledger/voucher/{id}/:reverse?date=YYYY-MM-DD`

### Voucher import / batch vouchers
Use voucher import endpoints when the task is to ingest many accounting rows from a file, bank
statement, or structured import source.

Guidance:
- Prefer import/batch voucher endpoints over many `POST /ledger/voucher` calls when the prompt is
  clearly about importing a file or many transactions at once.
- Use single `POST /ledger/voucher` only for one-off or very small manual journal tasks.
- Preserve source-level grouping when importing so related postings stay together in the same batch
  or voucher document.

### Bank statement import and reconciliation
When the prompt asks to reconcile a bank file or import bank transactions, the simplest and most \
reliable approach is to create manual journal vouchers that record each transaction.

**Approach — voucher-based reconciliation:**
1. Parse the attached CSV/file to extract transaction rows (date, description, amount).
2. Identify the bank account (usually 1920 Bankinnskudd) via `GET /ledger/account?number=1920`.
3. For each transaction or group, create a `POST /ledger/voucher` with:
   - `description` set to the file/batch reference from the prompt (e.g. the reconciliation ID)
   - `date` set to the transaction date
   - Debit posting to bank account 1920 (positive amount for incoming) or credit (negative for outgoing)
   - Contra posting to the appropriate income/expense account (e.g. 3000 for sales income, \
     6800 for bank fees, 7770 for other expenses)
   - Both postings MUST balance to zero

**CRITICAL:** Use the reference/ID from the prompt as the voucher `description` so the verifier \
can find it. If the prompt says "BANK-RECONCILIATION-FILE-001", use exactly that string.

Example for an incoming payment of 1250.00:
```json
{
  "description": "BANK-RECONCILIATION-FILE-001",
  "date": "2024-01-15",
  "postings": [
    {"row": 1, "amountGross": 1250.0, "amountGrossCurrency": 1250.0, "account": {"id": BANK_ACCT_ID}},
    {"row": 2, "amountGross": -1250.0, "amountGrossCurrency": -1250.0, "account": {"id": INCOME_ACCT_ID}}
  ]
}
```

### Attachments / document endpoints
Use attachment/document endpoints when the task says to upload, link, or preserve files on a
supplier invoice, voucher, order, or other Tripletex entity.

Guidance:
- Treat attached files as source material for interpretation first, but if the user also asks to
  store the file in Tripletex, call the relevant attachment endpoint family after the main entity is
  created or identified.
- Prefer attaching the file directly to the target entity instead of creating unrelated placeholder
  records.
- If the prompt only asks to read the file and register data from it, do not add extra attachment
  API calls unless persistence is explicitly requested.

### Batch and import endpoints
Tripletex includes dedicated batch/import endpoint families for several domains. Use them whenever
the prompt is about many rows, a file import, or repeated entities in one operation.

Guidance:
- Prefer batch/import endpoints when they let you complete the task with fewer write calls and the
  prompt naturally describes bulk work.
- Prefer single-record endpoints for one entity when batch adds unnecessary setup.
- Do not split a clear bulk-import task into many single POST/PUT calls unless no batch/import
  endpoint fits the requested workflow.

### POST /activity
Required: `name`
Common: `activityType` ("GENERAL_ACTIVITY"/"PROJECT_GENERAL_ACTIVITY"), `isChargeable`, `rate`

For project-specific activities: use `POST /project/projectActivity` instead.

### GET /supplierInvoice (read-only)
Use `GET /supplierInvoice` to list/search existing supplier invoices. \
To CREATE a new supplier invoice, use `POST /supplierInvoice` (see above).

GET requires: `invoiceDateFrom` AND `invoiceDateTo` (both mandatory for listing)
Search params: `invoiceNumber`, `fields`, `count`

### POST /supplier
Required: `name`
Common: `email`, `phoneNumber`, `organizationNumber` (org.nr), `accountNumber` (bank account), \
`supplierNumber`

Example:
```json
{"name": "Leverandor AS", "organizationNumber": "999888777"}
```

GET search params: `name`, `organizationNumber`, `fields`, `count`

### GET /currency
Query: `code` ("NOK", "SEK", "EUR", "USD"), `fields`, `count`
NOK = id 1, SEK = id 2, DKK = id 3, EUR = id 4
Read-only — currencies cannot be created.

### GET /ledger/vatType
Common VAT types (Norway):
- id 0 = no VAT (0%)
- id 3 = 25% outgoing MVA (høy sats)
- id 5 = 15% outgoing MVA (food/næringsmidler)
- id 31 = 12% outgoing MVA (transport, etc.)

---

## COMMON TASK PATTERNS

| Task | API Flow |
|------|----------|
| Create employee | GET /department → POST /employee |
| Create employee (admin) | GET /department → POST /employee (userType: 2) |
| Create customer | POST /customer |
| Create invoice | POST /customer → POST /order (with inline orderLines, MUST include vatType on each line) → check bank account on ledger/account 1920 (set if empty) → POST /invoice |
| Create product | POST /product |
| Create project | GET /employee (find manager) → POST /employee/entitlement (entitlementId 45, then 10) → POST /project |
| Register travel expense | GET /employee → POST /travelExpense |
| Delete travel expense | GET /travelExpense → DELETE /travelExpense/{id} |
| Create department | POST /department |
| Journal entry | GET /ledger/account (find accounts) → POST /ledger/voucher (MUST include `row` starting at 1 on each posting, and `vatType` on VAT-locked accounts) |
| Create supplier | POST /supplier |
| Register supplier invoice | GET/POST /supplier → GET /ledger/account (find expense account 4000 + AP account 2400) → POST /supplierInvoice (nested: invoiceNumber, invoiceDate, invoiceDueDate, supplier ref, voucher with description + balanced postings using amountGross/amountGrossCurrency) |
| Reconcile bank file | Parse CSV/file → POST /ledger/voucher per transaction (description=file reference, postings: bank account 1920 vs expense/income account) |
| Create purchase order | GET/POST /supplier → POST /purchaseOrder |
| Import vouchers | Voucher import/batch endpoint → only use single voucher POST for one-off journals |
| Register timesheet | GET /employee and optional GET /project/activity → POST /timesheet |
| Run payroll for many employees | Payroll batch/run endpoint family |
| Manual payroll journal entry | GET /ledger/account?number=5000,5200,...&fields=id,number&count=10 (ONE call for ALL accounts) → POST /ledger/voucher (ONE voucher with ALL debit/credit postings, row starting at 1) |
| Create contact | POST /contact |
| Invoice with payment | POST /customer → POST /order (inline orderLines, vatType on each line) → GET /invoice/paymentType (find "Betalt til bank" ID) → POST /invoice → PUT /invoice/{id}/:payment (paymentDate, paymentTypeId=LOOKED_UP_ID, paidAmount, paidAmountCurrency) |
| Credit note | GET /invoice → PUT /invoice/{id}/:createCreditNote |
| Create activity | POST /activity |

## EFFICIENCY TIPS

- Include `orderLines` inline in the order POST — saves separate POST /order/orderline calls. \
Do NOT also POST /order/orderline afterward — the lines already exist from the inline creation.
- Don't GET an entity you just created — the POST response already has the full object with ID.
- Don't search for entities unless the prompt asks you to find existing ones.
- Prefer fewer, complete requests over many small ones.
- For imports, bank files, payroll runs, and other bulk jobs, prefer the relevant batch/import
  endpoint family over many single-record calls.
- If a file is only evidence for extracting data, read it and post the resulting business object;
  only call attachment endpoints when the user also wants the file stored in Tripletex.
- **Scope discipline: ONLY do what the prompt asks.** Before each API call, re-read the original \
prompt and verify it requires this call. Creating unrequested entities wastes calls AND can cause errors.
- **Batch GET requests.** When you need multiple entities of the same type, use query params to fetch \
them in ONE call (e.g. `GET /ledger/account?number=5000,5200,2930&count=10`) instead of one call per entity.
- **Combine postings into ONE voucher.** For journal entries with multiple debit/credit lines, \
create a single voucher with all postings — never create separate vouchers per line.
- **Use specific field selectors.** Always use `fields=id,name,number` (only what you need) — \
never `fields=*`. This reduces payload size and processing time.

## DATE FORMAT

Always use ISO format: "YYYY-MM-DD" (e.g. "2026-03-21"). Use today's date when the \
prompt doesn't specify one. Today is the date when you receive the prompt.

## ERROR HANDLING

If an API call returns a 422 error, the `validationMessages` array tells you exactly what's wrong. \
Read it carefully and fix the specific field. Do NOT retry with the same payload.

If you get "Brukertype kan ikke være '0' eller tom" on employee creation, you forgot `userType`.
If you get "Må angis for Tripletex-brukere" on employee, you forgot `email`.
If you get "Kunde mangler" on a voucher posting, the account requires a `customer` ref in the posting.
If you get "systemgenererte og kan ikke opprettes" on voucher postings, you are missing the `row` field \
(must start at 1) or accidentally used row 0 (reserved for system VAT postings).
If you get "Kontoen X er låst til mva-kode Y", the account requires a specific `vatType` in the posting.
If you get "Ugyldig mva-kode" on order lines, you forgot to include `vatType` (e.g. `{"id": 3}` for 25% MVA). \
If the first attempt with vatType 3 fails, retry with vatType 6 (no VAT).
If order creation fails with a 422, you MUST fix the specific validation error and retry — \
do not abandon the entire workflow. The downstream invoice and payment steps depend on the order \
being created. Read the 422 error message carefully, fix the field, and retry once.
If you get "Faktura kan ikke opprettes før selskapet har registrert et bankkontonummer", the company \
needs a bank account: GET /ledger/account?number=1920, then PUT to set bankAccountNumber (11 digits), then retry.
If you get "har ikke fått tilgang som prosjektleder" on project creation, grant entitlements 45 then \
10 via POST /employee/entitlement (see endpoint reference above), then retry POST /project.
If you get "krever tilgang til å opprette nye prosjekter" when granting entitlement 10, you must \
grant entitlement 45 (AUTH_CREATE_PROJECT) FIRST, then grant 10 (AUTH_PROJECT_MANAGER).
If an API call returns a 500 error (Internal Server Error), do NOT retry with the same or similar \
payload more than once. A 500 usually means you are using the wrong endpoint or the wrong field \
structure. Try an alternative approach instead of repeating the same call.
"""


def build_request_context(*, now: datetime | None = None) -> RequestContext:
    request_now = now or datetime.now(timezone.utc)
    return RequestContext(
        current_date_iso=request_now.date().isoformat(),
        budget=RequestBudget(
            endpoint_timeout_seconds=ENDPOINT_TIMEOUT_SECONDS,
            reserved_headroom_seconds=RESERVED_HEADROOM_SECONDS,
            execution_budget_seconds=EXECUTION_BUDGET_SECONDS,
            max_model_turns=MAX_MODEL_TURNS,
            max_tool_calls=MAX_TOOL_CALLS,
        ),
        guardrails=(
            "inject_current_iso_date_when_prompt_omits_date",
            "preserve_explicit_user_dates",
            "keep_solve_synchronous",
            "enforce_request_budget_caps",
            "minimize_redundant_api_calls",
        ),
    )


def _render_request_context_instruction(request_context: RequestContext) -> str:
    budget = request_context.budget
    return "\n".join(
        [
            "## RUNTIME REQUEST CONTEXT",
            f"- Current request date (UTC): {request_context.current_date_iso}",
            "- Use this injected date only when the user did not provide an explicit date.",
            "- Preserve any explicit user-provided dates exactly as written.",
            "- This /solve request is synchronous. Do not assume background work, polling, or follow-up callbacks.",
            (
                "- Budget policy: "
                f"endpoint_timeout_seconds={budget.endpoint_timeout_seconds}, "
                f"reserved_headroom_seconds={budget.reserved_headroom_seconds}, "
                f"execution_budget_seconds={budget.execution_budget_seconds}, "
                f"max_model_turns={budget.max_model_turns}, "
                f"max_tool_calls={budget.max_tool_calls}."
            ),
            "- Finish decisively within budget and avoid speculative or redundant API calls.",
            "- End with TASK COMPLETED when the task is done.",
        ]
    )


def _remaining_budget_seconds(deadline_monotonic: float) -> float:
    return round(max(0.0, deadline_monotonic - time.monotonic()), 2)


def _build_generate_content_config(
    request_context: RequestContext,
    task_preamble: str = "",
) -> types.GenerateContentConfig:
    system_parts: list[str] = [
        SYSTEM_PROMPT,
        _render_request_context_instruction(request_context),
    ]
    if task_preamble:
        system_parts.append(f"\n## TASK-SPECIFIC GUIDANCE\n{task_preamble}")
    return types.GenerateContentConfig(
        system_instruction=system_parts,
        tools=[{"function_declarations": [execute_api_func]}],
        temperature=MODEL_TEMPERATURE,
        top_p=MODEL_TOP_P,
        candidate_count=MODEL_CANDIDATE_COUNT,
        seed=MODEL_RANDOM_SEED,
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MEDIUM,
            include_thoughts=False,
        ),
    )


def _resolve_inline_file_mime_type(file: SolveFile) -> str | None:
    if file.mime_type and file.mime_type.strip():
        return file.mime_type.strip().lower()

    guessed_mime_type, _ = mimetypes.guess_type(file.filename)
    if guessed_mime_type:
        return guessed_mime_type.lower()
    return None


def _build_initial_contents(request: SolveRequest) -> list[object]:
    contents: list[object] = [request.prompt]

    for file in request.files:
        mime_type = _resolve_inline_file_mime_type(file)
        if mime_type not in SUPPORTED_INLINE_FILE_MIME_TYPES:
            task_logger.log(
                "file_attachment_rejected",
                {
                    "filename": file.filename,
                    "mime_type": mime_type,
                    "reason": "unsupported_mime_type",
                    "supported_mime_types": sorted(SUPPORTED_INLINE_FILE_MIME_TYPES),
                },
            )
            raise ValueError(
                f"Unsupported file MIME type for {file.filename}: {mime_type!r}. "
                f"Supported types: {', '.join(sorted(SUPPORTED_INLINE_FILE_MIME_TYPES))}."
            )

        decoded_content = file.decoded_content()
        decoded_size = len(decoded_content)
        if decoded_size > MAX_INLINE_FILE_BYTES:
            task_logger.log(
                "file_attachment_rejected",
                {
                    "filename": file.filename,
                    "mime_type": mime_type,
                    "reason": "file_too_large",
                    "size_bytes": decoded_size,
                    "max_inline_file_bytes": MAX_INLINE_FILE_BYTES,
                },
            )
            raise ValueError(
                f"File {file.filename} exceeds inline attachment size limit of "
                f"{MAX_INLINE_FILE_BYTES} bytes: {decoded_size} bytes."
            )

        task_logger.log(
            "file_attachment_prepared",
            {
                "filename": file.filename,
                "mime_type": mime_type,
                "size_bytes": decoded_size,
                "max_inline_file_bytes": MAX_INLINE_FILE_BYTES,
                "decision": "accepted_for_inline_model_input",
            },
        )
        if mime_type in _TEXT_MIME_TYPES:
            text_content = decoded_content.decode("utf-8", errors="replace")
            contents.append(f"[Attached file: {file.filename}]\n{text_content}")
        else:
            contents.append(
                types.Part.from_bytes(data=decoded_content, mime_type=mime_type)
            )

    return contents


def _shape_list_tool_response(response_body: dict[str, Any]) -> dict[str, Any]:
    raw_values = response_body.get("values")
    values = raw_values if isinstance(raw_values, list) else []
    kept_values = values[:MAX_SHAPED_LIST_VALUES]

    shaped: dict[str, Any] = {
        "values": kept_values,
        "returned_value_count": len(kept_values),
        "omitted_value_count": max(0, len(values) - len(kept_values)),
        "values_truncated": len(kept_values) < len(values),
    }
    for key in ("count", "from", "fullResultSize"):
        if key in response_body:
            shaped[key] = response_body[key]
    return shaped


def _shape_error_tool_response(response_body: object | None) -> dict[str, Any]:
    if not isinstance(response_body, dict):
        return {"error": response_body} if response_body is not None else {}

    shaped: dict[str, Any] = {}
    for key in (
        "validationMessages",
        "message",
        "developerMessage",
        "error",
        "code",
        "status",
        "requestId",
    ):
        if key in response_body:
            shaped[key] = response_body[key]

    if shaped:
        return shaped
    return {"body": response_body}


def _shape_tripletex_tool_response(
    *, method: str, status_code: int, response_body: object | None
) -> dict[str, Any]:
    shaped: dict[str, Any] = {"status_code": status_code}

    if status_code >= 400:
        shaped.update(_shape_error_tool_response(response_body))
        return shaped

    if not isinstance(response_body, dict):
        if response_body is not None:
            shaped["body"] = response_body
        return shaped

    if isinstance(response_body.get("values"), list):
        shaped.update(_shape_list_tool_response(response_body))
        return shaped

    if "value" in response_body:
        shaped["value"] = response_body["value"]
        return shaped

    if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        shaped["body"] = response_body
        return shaped

    shaped["body"] = response_body
    return shaped


def _summarize_shaped_tripletex_tool_response(
    *,
    method: str,
    path: str,
    status_code: int,
    shaped_response: dict[str, Any],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "response_kind": "body",
    }

    if status_code >= 400:
        summary["response_kind"] = "error"
        validation_messages = shaped_response.get("validationMessages")
        summary["validation_message_count"] = (
            len(validation_messages) if isinstance(validation_messages, list) else 0
        )
        summary["error_keys"] = sorted(
            key for key in shaped_response.keys() if key != "validationMessages"
        )
        return summary

    if "value" in shaped_response:
        value = shaped_response.get("value")
        summary["response_kind"] = "value"
        summary["value_keys"] = sorted(value.keys()) if isinstance(value, dict) else []
        return summary

    if "values" in shaped_response:
        values = shaped_response.get("values")
        summary["response_kind"] = "list"
        summary["returned_value_count"] = shaped_response.get("returned_value_count", 0)
        summary["omitted_value_count"] = shaped_response.get("omitted_value_count", 0)
        summary["values_truncated"] = shaped_response.get("values_truncated", False)
        if isinstance(values, list) and values and isinstance(values[0], dict):
            summary["sample_value_keys"] = sorted(values[0].keys())
        else:
            summary["sample_value_keys"] = []
        return summary

    body = shaped_response.get("body")
    summary["body_type"] = type(body).__name__ if body is not None else "none"
    if isinstance(body, dict):
        summary["body_keys"] = sorted(body.keys())
    return summary


def _build_client_error_objective(*, method: str, path: str) -> str:
    return f"{method.upper()} {path}"


class _ExecutedOperationLike(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def response_body(self) -> object | None: ...


def _stringify_cache_query_scalar(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _serialize_cache_query_params(
    query: dict[str, str | int | float | bool | list[str | int | float | bool]],
) -> str:
    serialized_query: dict[str, str | list[str]] = {}
    for key, value in query.items():
        if isinstance(value, list):
            serialized_query[key] = [
                _stringify_cache_query_scalar(item) for item in value
            ]
        else:
            serialized_query[key] = _stringify_cache_query_scalar(value)
    return json.dumps(serialized_query, sort_keys=True, separators=(",", ":"))


def _build_request_local_get_cache_key(
    *,
    method: str,
    path: str,
    query: dict[str, str | int | float | bool | list[str | int | float | bool]],
) -> str:
    return "|".join(
        [
            method.upper(),
            path,
            _serialize_cache_query_params(query),
        ]
    )


def _resource_family_for_path(path: str) -> str:
    normalized_path = path.split("?", 1)[0]
    segments = [segment for segment in normalized_path.split("/") if segment]
    if not segments:
        return "/"
    return f"/{segments[0]}"


def _invalidate_request_local_get_cache(
    request_get_cache: dict[str, tuple[str, _ExecutedOperationLike]],
    *,
    step: int,
    tool_call_index: int,
    path: str,
    deadline_monotonic: float,
) -> None:
    resource_family = _resource_family_for_path(path)
    invalidated_keys = [
        cache_key
        for cache_key, cached_entry in request_get_cache.items()
        if _resource_family_for_path(cached_entry[0]) == resource_family
    ]

    for cache_key in invalidated_keys:
        request_get_cache.pop(cache_key, None)

    task_logger.log(
        "request_cache_invalidate",
        {
            "step": step,
            "tool_call_index": tool_call_index,
            "method": "GET",
            "resource_family": resource_family,
            "path": path,
            "invalidated_entry_count": len(invalidated_keys),
            "invalidated_keys": invalidated_keys,
            "remaining_budget_seconds": _remaining_budget_seconds(deadline_monotonic),
        },
    )


def _extract_validation_messages(
    result_content: dict[str, Any],
) -> list[dict[str, str]]:
    raw_validation_messages = result_content.get("validationMessages")
    if not isinstance(raw_validation_messages, list):
        return []

    validation_messages: list[dict[str, str]] = []
    for item in raw_validation_messages:
        if not isinstance(item, dict):
            continue

        field = str(item.get("field") or "").strip()
        message = str(item.get("message") or "").strip()
        if not field and not message:
            continue

        validation_messages.append({"field": field, "message": message})

    return validation_messages


def _build_validation_repair_instruction(
    *, objective: str, validation_messages: list[dict[str, str]]
) -> str:
    rendered_messages: list[str] = []
    for item in validation_messages[:MAX_REPAIR_HINT_VALIDATION_MESSAGES]:
        field = item["field"]
        message = item["message"]
        if field and message:
            rendered_messages.append(f"- {field}: {message}")
        elif message:
            rendered_messages.append(f"- {message}")
        else:
            rendered_messages.append(f"- {field}")

    rendered_hint_block = (
        "\n".join(rendered_messages) or "- validationMessages were present"
    )
    return "\n".join(
        [
            (
                f"Repair the last failed Tripletex objective exactly once: {objective}. "
                "Use the structured validation hints below and change the payload before retrying."
            ),
            rendered_hint_block,
            "Do not repeat the same payload. If the fix is impossible from known data, stop.",
        ]
    )


def _classify_client_error_decision(
    *,
    client_errors: list[dict[str, Any]],
    repair_turn_used: bool,
    repaired_objective: str | None,
    repaired_objective_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    first_error = client_errors[0]
    objective = str(first_error["objective"])
    status_code = int(first_error["status_code"])
    validation_messages = cast(
        list[dict[str, str]], first_error.get("validation_messages") or []
    )
    if (
        status_code == 422
        and "/employee/standardtime" in objective.lower()
        and validation_messages
    ):
        return {
            "objective": objective,
            "status_code": status_code,
            "recoverable": False,
            "validation_message_count": len(validation_messages),
            "client_error_count_in_turn": len(client_errors),
            "decision": "fail_fast",
            "reason": "employee_standardtime_non_recoverable",
        }
    recoverable = status_code == 422 and bool(validation_messages)

    detail: dict[str, Any] = {
        "objective": objective,
        "status_code": status_code,
        "recoverable": recoverable,
        "validation_message_count": len(validation_messages),
        "client_error_count_in_turn": len(client_errors),
    }

    if len(client_errors) > 1:
        detail.update(
            {
                "decision": "fail_fast",
                "reason": "multiple_client_errors_in_single_turn",
            }
        )
        return detail

    if not recoverable:
        detail.update(
            {
                "decision": "fail_fast",
                "reason": "non_recoverable_client_error",
            }
        )
        return detail

    # Per-objective repair: allow up to MAX_REPAIRS_PER_OBJECTIVE repairs per distinct objective
    _counts = repaired_objective_counts or {}
    prior_repairs = _counts.get(objective, 0)
    if prior_repairs >= MAX_REPAIRS_PER_OBJECTIVE:
        detail.update(
            {
                "decision": "fail_fast",
                "reason": "max_repairs_exhausted_for_objective",
                "prior_repair_count": prior_repairs,
                "max_repairs_per_objective": MAX_REPAIRS_PER_OBJECTIVE,
            }
        )
        return detail

    repair_instruction = _build_validation_repair_instruction(
        objective=objective,
        validation_messages=validation_messages,
    )
    detail.update(
        {
            "decision": "retry",
            "reason": "recoverable_validation_422",
            "repair_instruction": repair_instruction,
            "repair_attempt": prior_repairs + 1,
            "max_repairs_per_objective": MAX_REPAIRS_PER_OBJECTIVE,
        }
    )
    return detail


async def _send_with_retry(
    chat: Any,
    contents: list[object],
    *,
    deadline_monotonic: float,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(_SEND_MAX_RETRIES + 1):
        if attempt > 0:
            remaining = deadline_monotonic - time.monotonic()
            delay = _SEND_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            if remaining < delay + 5.0:
                task_logger.log(
                    "send_retry_skipped",
                    {
                        "attempt": attempt,
                        "reason": "insufficient_budget_for_retry",
                        "remaining_seconds": round(remaining, 2),
                    },
                )
                break
            task_logger.log(
                "send_retry_waiting",
                {"attempt": attempt, "delay_seconds": delay},
            )
            await asyncio.sleep(delay)

        try:
            return await chat.send_message(contents)
        except Exception as exc:
            last_exc = exc
            error_str = str(exc)
            if not _TRANSIENT_ERROR_PATTERNS.search(error_str):
                raise
            task_logger.log(
                "send_transient_error",
                {
                    "attempt": attempt,
                    "error": error_str[:300],
                    "will_retry": attempt < _SEND_MAX_RETRIES,
                },
            )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("_send_with_retry exhausted retries without result or exception")


async def run_agent(
    request: SolveRequest,
    client: TripletexClient,
    *,
    request_context: RequestContext | None = None,
) -> SolveExecutionOutcome:
    effective_request_context = request_context or build_request_context()
    deadline_monotonic = (
        time.monotonic() + effective_request_context.budget.execution_budget_seconds
    )

    profile = classify_task(request.prompt)
    task_logger.log(
        "task_classification",
        {
            "task_type": profile.task_type,
            "max_writes": profile.max_writes,
            "max_turns": profile.max_turns,
            "max_tool_calls": profile.max_tool_calls,
            "allowed_write_prefixes": sorted(profile.allowed_write_prefixes),
        },
    )

    task_logger.log(
        "request_context_decision",
        {
            "decision": "inject_current_date_when_needed",
            "current_date_iso": effective_request_context.current_date_iso,
            "preserve_explicit_user_dates": True,
            "execution_mode": effective_request_context.execution_mode,
            "prompt_length": len(request.prompt),
            "file_count": len(request.files),
        },
    )
    contents = _build_initial_contents(request)

    genai_client = genai.Client(api_key=API_KEY)

    config = _build_generate_content_config(
        effective_request_context, task_preamble=profile.task_preamble
    )

    chat = genai_client.aio.chats.create(model=MODEL_NAME, config=config)

    task_logger.log("model_context_injected", effective_request_context.as_log_detail())
    task_logger.log(
        "model_runtime_configured",
        {
            "model": MODEL_NAME,
            "temperature": MODEL_TEMPERATURE,
            "top_p": MODEL_TOP_P,
            "candidate_count": MODEL_CANDIDATE_COUNT,
            "seed": MODEL_RANDOM_SEED,
            "thinking_level": "MEDIUM",
            "include_thoughts": False,
            "endpoint_timeout_seconds": effective_request_context.budget.endpoint_timeout_seconds,
            "reserved_headroom_seconds": effective_request_context.budget.reserved_headroom_seconds,
            "execution_budget_seconds": effective_request_context.budget.execution_budget_seconds,
            "max_model_turns": effective_request_context.budget.max_model_turns,
            "max_tool_calls": effective_request_context.budget.max_tool_calls,
        },
    )
    if time.monotonic() >= deadline_monotonic:
        task_logger.log(
            "request_budget_exhausted",
            {
                "reason": "deadline_reached_before_initial_llm_call",
                "step": -1,
                "tool_calls_completed": 0,
                "budget_state": "deadline_reached_before_initial_llm_call",
                "remaining_budget_seconds": _remaining_budget_seconds(
                    deadline_monotonic
                ),
            },
        )
        return SolveExecutionOutcome(
            status="incomplete",
            reason="deadline_reached_before_initial_llm_call",
        )

    task_logger.log(
        "llm_call",
        {
            "message": "initial_prompt",
            "prompt_length": len(request.prompt),
            "remaining_budget_seconds": _remaining_budget_seconds(deadline_monotonic),
        },
    )
    response = await _send_with_retry(
        chat, contents, deadline_monotonic=deadline_monotonic
    )
    task_logger.log(
        "llm_response",
        {
            "text": (response.text or "")[:500],
            "function_call_count": len(response.function_calls or []),
            "remaining_budget_seconds": _remaining_budget_seconds(deadline_monotonic),
        },
    )

    total_tool_calls = 0
    write_count = 0
    repair_turn_used = False
    repaired_objective: str | None = None
    repaired_objective_counts: dict[str, int] = {}
    request_get_cache: dict[str, tuple[str, _ExecutedOperationLike]] = {}

    for _step in range(profile.max_turns):
        if time.monotonic() >= deadline_monotonic:
            task_logger.log(
                "request_budget_exhausted",
                {
                    "reason": "execution_budget_seconds_reached",
                    "step": _step,
                    "tool_calls_completed": total_tool_calls,
                    "budget_state": "exhausted",
                    "remaining_budget_seconds": _remaining_budget_seconds(
                        deadline_monotonic
                    ),
                },
            )
            return SolveExecutionOutcome(
                status="incomplete",
                reason="execution_budget_seconds_reached",
            )

        if response.text and "TASK COMPLETED" in response.text:
            task_logger.log(
                "task_completed",
                {
                    "step": _step,
                    "tool_calls_completed": total_tool_calls,
                    "budget_state": "completed_with_budget_remaining",
                    "remaining_budget_seconds": _remaining_budget_seconds(
                        deadline_monotonic
                    ),
                },
            )
            return SolveExecutionOutcome(status="completed", reason="task_completed")

        func_calls = response.function_calls
        if not func_calls:
            task_logger.log(
                "no_function_calls",
                {
                    "step": _step,
                    "text": (response.text or "")[:500],
                    "tool_calls_completed": total_tool_calls,
                    "budget_state": "idle_no_tool_calls_requested",
                },
            )
            return SolveExecutionOutcome(
                status="incomplete",
                reason="idle_no_tool_calls_requested",
            )

        remaining_tool_calls = profile.max_tool_calls - total_tool_calls
        if remaining_tool_calls <= 0:
            task_logger.log(
                "request_budget_exhausted",
                {
                    "reason": "max_tool_calls_reached",
                    "step": _step,
                    "tool_calls_completed": total_tool_calls,
                    "budget_state": "max_tool_calls_reached",
                    "remaining_budget_seconds": _remaining_budget_seconds(
                        deadline_monotonic
                    ),
                },
            )
            return SolveExecutionOutcome(
                status="incomplete",
                reason="max_tool_calls_reached",
            )

        capped_func_calls = list(func_calls[:remaining_tool_calls])
        if len(capped_func_calls) < len(func_calls):
            task_logger.log(
                "request_budget_guardrail",
                {
                    "step": _step,
                    "reason": "truncated_function_calls_to_fit_budget",
                    "budget_state": "guardrail_applied",
                    "requested_tool_calls": len(func_calls),
                    "allowed_tool_calls": len(capped_func_calls),
                    "tool_calls_completed": total_tool_calls,
                },
            )

        function_responses: list[Any] = []
        client_errors: list[dict[str, Any]] = []
        deadline_reached_during_tool_execution = False
        for function_call in capped_func_calls:
            if time.monotonic() >= deadline_monotonic:
                task_logger.log(
                    "request_budget_exhausted",
                    {
                        "reason": "execution_budget_seconds_reached_during_tool_execution",
                        "step": _step,
                        "tool_calls_completed": total_tool_calls,
                        "budget_state": "exhausted_during_tool_execution",
                        "remaining_budget_seconds": _remaining_budget_seconds(
                            deadline_monotonic
                        ),
                    },
                )
                deadline_reached_during_tool_execution = True
                break

            if function_call.name == "execute_tripletex_api":
                args = function_call.args
                method_raw = args.get("method", "GET")
                path_raw = args.get("path", "")

                query_raw = dict(args.get("query", {})) if args.get("query") else {}
                body_raw = dict(args.get("body", {})) if args.get("body") else None

                method_upper = str(method_raw).upper()
                if method_upper not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                    method_upper = "GET"
                method = cast(
                    Literal["DELETE", "GET", "PATCH", "POST", "PUT"], method_upper
                )

                operation = StructuredOperation(
                    method=method,
                    path=str(path_raw),
                    query=query_raw,
                    body=body_raw,
                    allow_failure=True,
                )
                total_tool_calls += 1

                if _is_write_method(operation.method):
                    if not _is_write_allowed(
                        operation.path, profile.allowed_write_prefixes
                    ):
                        task_logger.log(
                            "write_blocked_by_profile",
                            {
                                "step": _step,
                                "method": operation.method,
                                "path": operation.path,
                                "task_type": profile.task_type,
                                "allowed_prefixes": sorted(
                                    profile.allowed_write_prefixes
                                ),
                            },
                        )
                        function_responses.append(
                            types.Part.from_function_response(
                                name="execute_tripletex_api",
                                response={
                                    "result": {
                                        "status_code": 403,
                                        "error": (
                                            f"Write to {operation.path} is not allowed "
                                            f"for this task type ({profile.task_type}). "
                                            "Focus on the requested task and say TASK COMPLETED."
                                        ),
                                    }
                                },
                            )
                        )
                        continue

                    if write_count >= profile.max_writes:
                        task_logger.log(
                            "write_budget_exhausted",
                            {
                                "step": _step,
                                "method": operation.method,
                                "path": operation.path,
                                "write_count": write_count,
                                "max_writes": profile.max_writes,
                            },
                        )
                        function_responses.append(
                            types.Part.from_function_response(
                                name="execute_tripletex_api",
                                response={
                                    "result": {
                                        "status_code": 429,
                                        "error": (
                                            f"Write budget exhausted ({write_count}/{profile.max_writes}). "
                                            "All required writes are done. Say TASK COMPLETED now."
                                        ),
                                    }
                                },
                            )
                        )
                        continue

                task_logger.log(
                    "api_call",
                    {
                        "step": _step,
                        "method": operation.method,
                        "path": operation.path,
                        "query": operation.query,
                        "body_keys": list((operation.body or {}).keys())
                        if isinstance(operation.body, dict)
                        else None,
                        "tool_call_index": total_tool_calls,
                        "remaining_budget_seconds": _remaining_budget_seconds(
                            deadline_monotonic
                        ),
                    },
                )

                try:
                    response_source = "upstream"
                    executed: _ExecutedOperationLike
                    if operation.method == "GET":
                        cache_key = _build_request_local_get_cache_key(
                            method=operation.method,
                            path=operation.path,
                            query=operation.query,
                        )
                        cached_entry = request_get_cache.get(cache_key)
                        if cached_entry is not None:
                            response_source = "request_cache"
                            task_logger.log(
                                "request_cache_hit",
                                {
                                    "step": _step,
                                    "tool_call_index": total_tool_calls,
                                    "method": operation.method,
                                    "path": operation.path,
                                    "cache_key": cache_key,
                                    "serialized_query": _serialize_cache_query_params(
                                        operation.query
                                    ),
                                    "remaining_budget_seconds": _remaining_budget_seconds(
                                        deadline_monotonic
                                    ),
                                },
                            )
                            executed = cached_entry[1]
                        else:
                            task_logger.log(
                                "request_cache_miss",
                                {
                                    "step": _step,
                                    "tool_call_index": total_tool_calls,
                                    "method": operation.method,
                                    "path": operation.path,
                                    "cache_key": cache_key,
                                    "serialized_query": _serialize_cache_query_params(
                                        operation.query
                                    ),
                                    "remaining_budget_seconds": _remaining_budget_seconds(
                                        deadline_monotonic
                                    ),
                                },
                            )
                            executed = cast(
                                _ExecutedOperationLike,
                                await client.execute_operation(operation),
                            )
                            request_get_cache[cache_key] = (operation.path, executed)
                    else:
                        executed = cast(
                            _ExecutedOperationLike,
                            await client.execute_operation(operation),
                        )

                    if operation.method in {"POST", "PUT", "PATCH", "DELETE"}:
                        _invalidate_request_local_get_cache(
                            request_get_cache,
                            step=_step,
                            tool_call_index=total_tool_calls,
                            path=operation.path,
                            deadline_monotonic=deadline_monotonic,
                        )

                    result_content = _shape_tripletex_tool_response(
                        method=operation.method,
                        status_code=executed.status_code,
                        response_body=executed.response_body,
                    )
                    task_logger.log(
                        "response_shaping_summary",
                        _summarize_shaped_tripletex_tool_response(
                            method=operation.method,
                            path=operation.path,
                            status_code=executed.status_code,
                            shaped_response=result_content,
                        ),
                    )
                    task_logger.log(
                        "api_response",
                        {
                            "step": _step,
                            "method": operation.method,
                            "path": operation.path,
                            "status_code": executed.status_code,
                            "tool_call_index": total_tool_calls,
                            "response_source": response_source,
                        },
                    )
                    if (
                        _is_write_method(operation.method)
                        and executed.status_code < 400
                    ):
                        write_count += 1
                    if 400 <= executed.status_code < 500 and operation.method != "GET":
                        client_errors.append(
                            {
                                "objective": _build_client_error_objective(
                                    method=operation.method,
                                    path=operation.path,
                                ),
                                "status_code": executed.status_code,
                                "validation_messages": _extract_validation_messages(
                                    result_content
                                ),
                            }
                        )
                except Exception as exc:
                    if operation.method in {"POST", "PUT", "PATCH", "DELETE"}:
                        _invalidate_request_local_get_cache(
                            request_get_cache,
                            step=_step,
                            tool_call_index=total_tool_calls,
                            path=operation.path,
                            deadline_monotonic=deadline_monotonic,
                        )
                    result_content = {"error": str(exc)}
                    task_logger.log(
                        "api_error",
                        {
                            "step": _step,
                            "method": operation.method,
                            "path": operation.path,
                            "error": str(exc)[:500],
                            "tool_call_index": total_tool_calls,
                        },
                    )

                function_responses.append(
                    types.Part.from_function_response(
                        name="execute_tripletex_api",
                        response={"result": result_content},
                    )
                )

        if deadline_reached_during_tool_execution:
            return SolveExecutionOutcome(
                status="incomplete",
                reason="execution_budget_seconds_reached_during_tool_execution",
            )

        if function_responses:
            followup_contents: list[object] = list(function_responses)
            if client_errors:
                client_error_decision = _classify_client_error_decision(
                    client_errors=client_errors,
                    repair_turn_used=repair_turn_used,
                    repaired_objective=repaired_objective,
                    repaired_objective_counts=repaired_objective_counts,
                )
                task_logger.log("client_error_decision", client_error_decision)
                task_logger.log(
                    "retry_decision_summary",
                    {
                        "step": _step,
                        "decision": client_error_decision["decision"],
                        "reason": client_error_decision["reason"],
                        "objective": client_error_decision["objective"],
                        "status_code": client_error_decision["status_code"],
                        "repair_turn_used": repair_turn_used,
                    },
                )

                if client_error_decision["decision"] != "retry":
                    return SolveExecutionOutcome(
                        status="incomplete",
                        reason=cast(str, client_error_decision["reason"]),
                    )

                repair_turn_used = True
                repaired_objective = cast(str, client_error_decision["objective"])
                repaired_objective_counts[repaired_objective] = (
                    repaired_objective_counts.get(repaired_objective, 0) + 1
                )
                followup_contents.append(
                    cast(str, client_error_decision["repair_instruction"])
                )

            if time.monotonic() >= deadline_monotonic:
                task_logger.log(
                    "request_budget_exhausted",
                    {
                        "reason": "execution_budget_seconds_reached_before_followup_llm_call",
                        "step": _step,
                        "tool_calls_completed": total_tool_calls,
                        "budget_state": "exhausted_before_followup_llm_call",
                        "remaining_budget_seconds": _remaining_budget_seconds(
                            deadline_monotonic
                        ),
                    },
                )
                return SolveExecutionOutcome(
                    status="incomplete",
                    reason="execution_budget_seconds_reached_before_followup_llm_call",
                )

            task_logger.log(
                "llm_call",
                {
                    "message": "function_responses",
                    "step": _step,
                    "response_count": len(followup_contents),
                    "tool_calls_completed": total_tool_calls,
                    "remaining_budget_seconds": _remaining_budget_seconds(
                        deadline_monotonic
                    ),
                },
            )
            response = await _send_with_retry(
                chat, followup_contents, deadline_monotonic=deadline_monotonic
            )
            task_logger.log(
                "llm_response",
                {
                    "step": _step,
                    "text": (response.text or "")[:500],
                    "function_call_count": len(response.function_calls or []),
                    "tool_calls_completed": total_tool_calls,
                    "remaining_budget_seconds": _remaining_budget_seconds(
                        deadline_monotonic
                    ),
                },
            )
        else:
            return SolveExecutionOutcome(
                status="incomplete",
                reason="no_function_responses_to_continue",
            )
    else:
        task_logger.log(
            "request_budget_exhausted",
            {
                "reason": "max_model_turns_reached",
                "step": profile.max_turns,
                "tool_calls_completed": total_tool_calls,
                "budget_state": "max_model_turns_reached",
                "remaining_budget_seconds": _remaining_budget_seconds(
                    deadline_monotonic
                ),
            },
        )
        return SolveExecutionOutcome(
            status="incomplete",
            reason="max_model_turns_reached",
        )

    return SolveExecutionOutcome(
        status="incomplete", reason="loop_exited_without_terminal_state"
    )
