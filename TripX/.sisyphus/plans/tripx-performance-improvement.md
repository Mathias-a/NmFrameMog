# TripX Breadth-First Performance Improvement Plan

## TL;DR
> **Summary**: Improve the current TripX agent by expanding measurable task-family coverage first, then hardening runtime behavior for file handling, date correctness, error recovery, and API-efficiency without changing the synchronous Cloud Run `/solve` contract.
> **Deliverables**:
> - file-capable Gemini request path for PDFs/images within the existing FastAPI + Cloud Run service
> - packaged fixture matrix covering core CRUD, multi-step finance, linked-entity, and file-driven Tripletex workflows
> - request-context/date injection, response shaping, bounded 422 recovery, request-local GET caching, and deterministic runtime budgets
> - VM proxy regression wave plus Cloud Run smoke validation against the current deployed service
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: 1 → 2/3/4/5 → 6 → 7/8/9 → 10 → 11/12 → 14 → 15

## Context
### Original Request
- Analyze the task and the current implementation in depth.
- Produce a plan of action for improving performance with the current solution.
- Ensure all steps work with the already initiated Cloud Run instance.
- Convert the prior analysis into an implementation plan.

### Interview Summary
- The current system is a FastAPI service on Cloud Run with synchronous `POST /solve` and `GET /logs` endpoints.
- The agent in `task_tripletex/agent.py` already achieves optimal behavior for `create_employee_admin`, but files are skipped, only one packaged fixture exists, the runtime does not inject the actual date, all API responses are passed back to Gemini verbatim, and every operation is allowed to fail.
- Explore research on Tripletex OpenAPI v2.74.00 identified uncovered endpoint families needed for higher-tier tasks: supplier invoices, bank reconciliation, purchase orders, voucher imports, payroll/timesheets, inventory, attachments, and several batch/import endpoints.
- Oracle review recommended breadth-first optimization: handle more task families correctly before chasing marginal efficiency gains on already-working flows.

### Metis Review (gaps addressed)
- Preserve the external contract exactly: keep `/solve` synchronous, keep `/logs`, and avoid Cloud Tasks, async job orchestration, or platform redesign.
- Use TDD breadth-first sequencing: add/expand packaged fixtures and regression coverage before each runtime behavior change.
- Add explicit guardrails for file size/MIME policy, injected-date rules, response shaping, bounded 422 retry, request budgets, and request-local cache invalidation after writes.
- Keep Cloud Run compatibility fixed to the current service URL and deployment flow in `agents.md:24-29` and `agents.md:39-70`.

## Work Objectives
### Core Objective
Raise leaderboard performance by eliminating zero-score task families first, then improving per-run correctness/efficiency inside the existing Cloud Run deployment and request contract.

### Deliverables
- Expanded packaged fixture suite for the highest-value Tripletex task families.
- Runtime support for multimodal file inputs, injected current date, shaped tool responses, bounded validation recovery, request-local GET caching, and deterministic request budgets.
- Traceable logs and regression commands that prove behavior locally on the VM proxy path and on the current Cloud Run service.

### Definition of Done (verifiable conditions with commands)
- `create_employee_admin` still passes on the VM proxy path with **100% correctness, 2 API calls, 0 4xx errors** using the command pattern in `agents.md:39-58`.
- The packaged-case regression suite for prioritized task families passes via `PYTHONPATH=/home/devstar7073/NmFrameMog/TripX .venv/bin/python -m task_tripletex.testing.cli --packaged-case <case> --solve-url http://127.0.0.1:8080/solve --tripletex-base-url https://kkpqfuj-amager.tripletex.dev/v2 --session-token "<token>" --output json` for every new packaged case added in Tasks 2-5 and 13-14.
- The deployed Cloud Run service at `https://tripletex-124894558027.europe-north1.run.app/solve` returns HTTP 200 and `{"status":"completed"}` for representative text and file-backed smoke prompts using the `curl` pattern in `agents.md:61-66`.
- `https://tripletex-124894558027.europe-north1.run.app/logs` shows the latest run with request-context markers for injected date, retry decisions, and cache usage using the command in `agents.md:68-70`.
- No change introduces background job semantics, non-JSON `/solve` responses, or dependence on infrastructure outside the existing Cloud Run + VM regression workflow.

### Must Have
- Preserve the current FastAPI endpoint contract in `task_tripletex/service.py:85-123`.
- Preserve the generic relative-path Tripletex client flow in `task_tripletex/client.py:61-120`.
- Add fixtures and runtime support for file-driven and Tier 2/3 workflows before advanced optimization.
- Keep every verification step agent-executable with explicit commands and evidence paths.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Must NOT introduce asynchronous `/solve` processing, Cloud Tasks, Pub/Sub, polling contracts, or alternate deployment architecture.
- Must NOT blanket-prefetch reference data before the first model turn; use request-local exact-key cache only.
- Must NOT hide entity IDs, `validationMessages`, or fields needed for follow-up calls when shaping responses.
- Must NOT retry unknown or repeated 4xx responses; only one bounded recovery attempt is allowed for structured, recoverable validation errors.
- Must NOT expand into non-performance security/configuration migrations (for example moving secrets to env vars) in this plan.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: **TDD** using the existing packaged-case CLI, fixture loader unit tests, framework integration tests, VM proxy regression, and Cloud Run smoke checks.
- QA policy: Every task includes at least one happy path and one failure/edge path; no manual UI verification is permitted.
- VM packaged-case commands must reuse the exact `--tripletex-base-url` and `--session-token` values from `agents.md:53-58`; only `--packaged-case` changes unless the fixture itself embeds files.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.json`, `.sisyphus/evidence/task-{N}-{slug}.log`, or `.sisyphus/evidence/task-{N}-{slug}-error.json`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. Extract shared dependencies early so runtime work can proceed with full fixture coverage.

Wave 1: Tasks 1-5 — fixture/test breadth foundation across CRUD, finance, linked-entity, and file-driven workflows

Wave 2: Tasks 6-10 — request context, multimodal file handling, prompt expansion, response shaping, bounded validation recovery

Wave 3: Tasks 11-15 — request-local cache, deterministic budgets, trace enrichment, VM proxy breadth regression, Cloud Run deploy/smoke gate

### Dependency Matrix (full, all tasks)
| Task | Depends On | Enables |
| --- | --- | --- |
| 1 | none | 2, 3, 4, 5, 13 |
| 2 | 1 | 14 |
| 3 | 1 | 14 |
| 4 | 1 | 14 |
| 5 | 1 | 7, 8, 14, 15 |
| 6 | 1 | 7, 8, 9, 10, 12, 13 |
| 7 | 5, 6 | 14, 15 |
| 8 | 5, 6 | 9, 10, 14, 15 |
| 9 | 6, 8 | 10, 11, 14, 15 |
| 10 | 8, 9 | 14, 15 |
| 11 | 9 | 14, 15 |
| 12 | 6, 9, 10 | 14, 15 |
| 13 | 6, 7, 9, 10, 11, 12 | 14, 15 |
| 14 | 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13 | 15 |
| 15 | 14 | Final Verification Wave |

### Agent Dispatch Summary (wave → task count → categories)
| Wave | Task Count | Recommended Categories |
| --- | --- | --- |
| 1 | 5 | quick, unspecified-low, unspecified-high |
| 2 | 5 | unspecified-high, deep |
| 3 | 5 | unspecified-high, deep, quick |

## TODOs
> Implementation + Test = ONE task. Every task includes agent profile, references, acceptance criteria, QA scenarios, and commit guidance.

- [x] 1. Build packaged-fixture expansion scaffold

  **What to do**: Extend the packaged-fixture test scaffold so the repository can safely host multiple new Tripletex cases. Add unit coverage for packaged fixture discovery/loading, define a documented fixture naming convention, and create placeholder failing packaged fixtures for the breadth-first matrix: `create_customer`, `create_product`, `create_invoice_basic`, `create_project_basic`, `create_journal_entry_basic`, `create_travel_expense_basic`, `supplier_invoice_basic`, `bank_reconciliation_file`, and `invoice_with_payment`. Ensure fixture loader tests assert the new cases parse correctly and their read/check structures are valid before any runtime change starts.
  **Must NOT do**: Do not change runtime behavior in `task_tripletex/agent.py`, `service.py`, or `client.py` in this task. Do not mark placeholder fixtures as passing by weakening verifier logic.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: This spans test design, fixture schema, and regression scaffolding across several files.
  - Skills: `[]` — No additional skill is required beyond repository-local test harness understanding.
  - Omitted: `quality-check` — Not needed until runtime code changes begin.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 3, 4, 5 | Blocked By: none

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `task_tripletex/testing/fixture_loader.py:211-258` — canonical fixture loading and `SolveRequest` construction path
  - Pattern: `task_tripletex/testing/models.py:9-140` — allowed fixture/read/check/score schema
  - Pattern: `task_tripletex/testing/fixtures/create_employee_admin.json:1-76` — baseline packaged fixture shape to mirror
  - Test: `tests/task_tripletex_testing/test_fixture_loader.py:1-14` — current packaged-fixture loader assertion style
  - Test: `task_tripletex/testing/cli.py:182-216` — evaluation flow consuming packaged fixtures
  - Docs: `docs/testing_framework.md:51-95` — fixture semantics and deterministic efficiency policy

  **Acceptance Criteria** (agent-executable only):
  - [ ] `pytest tests/task_tripletex_testing/test_fixture_loader.py` passes and asserts all newly added packaged cases load successfully.
  - [ ] Every new placeholder packaged fixture parses via `python -c "from task_tripletex.testing.fixture_loader import load_packaged_case_fixture; [load_packaged_case_fixture(name) for name in ['create_customer','create_product','create_invoice_basic','create_project_basic','create_journal_entry_basic','create_travel_expense_basic','supplier_invoice_basic','bank_reconciliation_file','invoice_with_payment']]"`.
  - [ ] No runtime file outside `task_tripletex/testing/` and `tests/task_tripletex_testing/` changes in this commit.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Breadth fixture scaffold loads end-to-end
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_fixture_loader.py` and capture output.
    Expected: All packaged fixture loader tests pass; no schema/parsing failure for new case files.
    Evidence: .sisyphus/evidence/task-1-fixture-scaffold.json

  Scenario: Invalid placeholder fixture is rejected
    Tool: Bash
    Steps: Temporarily point one test at a deliberately malformed fixture in a new negative test and run the targeted pytest node.
    Expected: Loader raises ValueError with fixture-specific validation message.
    Evidence: .sisyphus/evidence/task-1-fixture-scaffold-error.json
  ```

  **Commit**: YES | Message: `test(tripx): add breadth fixture scaffold` | Files: `task_tripletex/testing/fixtures/*`, `tests/task_tripletex_testing/test_fixture_loader.py`

- [x] 2. Add Tier 1 entity-creation fixtures and verifier assertions

  **What to do**: Create passing packaged fixtures and any necessary verifier-facing selectors for `create_customer` and `create_product`. Each fixture must use unique identifying data, include field-by-field checks for the scored entity, and set realistic efficiency policy bounds that reward a single POST with no 4xx responses. Update fixture-loader and/or integration tests so these cases participate in the standard regression path.
  **Must NOT do**: Do not change agent logic yet. Do not add GET-before-POST expectations unless the Tripletex endpoint truly requires them. Do not use read-only fields like `isCustomer` in customer payload expectations.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Requires designing deterministic verification for real Tripletex entities.
  - Skills: `[]` — Existing test harness is sufficient.
  - Omitted: `quality-check` — Still test-focused, no runtime change yet.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 14 | Blocked By: 1

  **References**:
  - Pattern: `task_tripletex/testing/fixtures/create_employee_admin.json:1-76` — model for unique prompt + read + checks + efficiency policy
  - API/Type: `task_tripletex/agent.py:110-148` — customer/product prompt guidance and payload caveats already documented
  - Test: `task_tripletex/testing/verifier.py:53-129` — selector semantics and `field_equals` behavior
  - Docs: `docs/examples.md:66-109` — customer/product examples and search patterns
  - Docs: `docs/scoring.md:7-55` — correctness + efficiency scoring behavior

  **Acceptance Criteria**:
  - [ ] `python -m task_tripletex.testing.cli --packaged-case create_customer --solve-url http://127.0.0.1:8080/solve --tripletex-base-url https://kkpqfuj-amager.tripletex.dev/v2 --session-token "<token>" --output json` returns correctness `1.0` after implementation tasks are completed.
  - [ ] `python -m task_tripletex.testing.cli --packaged-case create_product --solve-url http://127.0.0.1:8080/solve --tripletex-base-url https://kkpqfuj-amager.tripletex.dev/v2 --session-token "<token>" --output json` returns correctness `1.0` after implementation tasks are completed.
  - [ ] Corresponding fixture loader tests pass locally.

  **QA Scenarios**:
  ```
  Scenario: Customer fixture verifies created entity fields
    Tool: Bash
    Steps: Run packaged-case CLI for `create_customer` against the local VM server after runtime support lands.
    Expected: Customer exists and all field_equals checks pass with zero contract/proxy violations.
    Evidence: .sisyphus/evidence/task-2-create-customer.json

  Scenario: Product fixture detects wrong VAT/price fields
    Tool: Bash
    Steps: Run the packaged-case against a deliberately broken branch or temporary failing payload expectation in a focused test.
    Expected: Verifier reports failed field check with explicit observed vs expected output.
    Evidence: .sisyphus/evidence/task-2-create-product-error.json
  ```

  **Commit**: YES | Message: `test(tripx): add customer and product fixtures` | Files: `task_tripletex/testing/fixtures/create_customer.json`, `task_tripletex/testing/fixtures/create_product.json`, related tests

- [x] 3. Add linked-entity and ledger fixtures for baseline multi-step workflows

  **What to do**: Add packaged fixtures for `create_project_basic`, `create_journal_entry_basic`, and `create_travel_expense_basic`. The project fixture must assume project manager lookup/create behavior, the voucher fixture must verify balanced postings and correct account-linked fields, and the travel expense fixture must verify employee linkage. Extend tests so these fixtures are part of the documented breadth matrix.
  **Must NOT do**: Do not weaken field assertions to “entity exists only.” Do not assume direct account IDs unless they are verified stable in the system prompt or fixture reads. Do not require UI-only verification.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Multi-entity fixture design with accounting-specific verification.
  - Skills: `[]` — No external skill needed.
  - Omitted: `quality-check` — Still fixture-first work.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 14 | Blocked By: 1

  **References**:
  - Pattern: `task_tripletex/agent.py:191-265` — project, travel expense, ledger account, voucher guidance and quirks
  - Pattern: `docs/overview.md:42-54` — competition task categories that justify these fixtures
  - Pattern: `docs/examples.md:120-157` — create-with-linking and error-handling guidance
  - Test: `task_tripletex/testing/tripletex_read_helper.py:88-134` — verification read paths and list wrapper expectations
  - Test: `task_tripletex/testing/verifier.py:17-92` — nested field-path assertions for linked references

  **Acceptance Criteria**:
  - [ ] New packaged fixtures parse and targeted fixture loader tests pass.
  - [ ] After runtime implementation tasks complete, each packaged case returns correctness `1.0` via the CLI.
  - [ ] Voucher fixture explicitly checks account-linked fields or balanced result data, not only entity existence.

  **QA Scenarios**:
  ```
  Scenario: Journal entry fixture validates balanced voucher outcome
    Tool: Bash
    Steps: Run packaged-case CLI for `create_journal_entry_basic` after runtime support lands.
    Expected: Voucher entity exists with expected description/date and no proxy violations.
    Evidence: .sisyphus/evidence/task-3-journal-entry.json

  Scenario: Travel expense fixture fails on missing employee linkage
    Tool: Bash
    Steps: Execute a targeted negative verification test with selector mismatch for employee linkage.
    Expected: Verifier reports "No entity matched the selector" or field mismatch.
    Evidence: .sisyphus/evidence/task-3-travel-expense-error.json
  ```

  **Commit**: YES | Message: `test(tripx): add linked entity and ledger fixtures` | Files: `task_tripletex/testing/fixtures/create_project_basic.json`, `task_tripletex/testing/fixtures/create_journal_entry_basic.json`, `task_tripletex/testing/fixtures/create_travel_expense_basic.json`, related tests

- [x] 4. Add multi-step invoicing fixtures for order-to-invoice flows

  **What to do**: Add packaged fixtures for `create_invoice_basic` and `invoice_with_payment`. The basic invoice fixture must verify customer/order/invoice linkage, and the payment fixture must verify the invoice payment mutation path. Use realistic efficiency policies that assume inline `orderLines` in the order POST and avoid unnecessary post-create GETs.
  **Must NOT do**: Do not split invoice verification into vague manual checks. Do not require separate `POST /order/orderline` if the fixture is intended to reward inline `orderLines`. Do not omit date-range requirements for any invoice/order verification GETs.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: These fixtures encode the most important Tier 2 finance flows.
  - Skills: `[]` — Local harness already supports this.
  - Omitted: `quality-check` — Fixture-first task.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 14 | Blocked By: 1

  **References**:
  - Pattern: `task_tripletex/agent.py:150-189` — order/invoice path rules, inline `orderLines`, payment action, mandatory date ranges
  - Docs: `docs/examples.md:81-99` and `docs/examples.md:149-157` — invoice creation and efficiency guidance
  - Docs: `docs/scoring.md:35-55` — why write-call and 4xx minimization matter for perfect submissions
  - Test: `task_tripletex/testing/reverse_proxy_recorder.py:237-262` — proxy metrics that encode write count and 4xx behavior

  **Acceptance Criteria**:
  - [ ] Packaged cases parse and targeted tests pass.
  - [ ] After runtime implementation, the CLI reports correctness `1.0` for both invoice fixtures.
  - [ ] Invoice/payment verification reads include the required date-range query params.

  **QA Scenarios**:
  ```
  Scenario: Invoice flow rewards inline order lines
    Tool: Bash
    Steps: Run packaged-case CLI for `create_invoice_basic` and inspect proxy call log in JSON output.
    Expected: Order is created with inline order lines and total write count stays within fixture efficiency policy.
    Evidence: .sisyphus/evidence/task-4-create-invoice.json

  Scenario: Payment fixture detects missing invoice date-range read params
    Tool: Bash
    Steps: Run a targeted failing test or broken fixture variant omitting `invoiceDateFrom`/`invoiceDateTo` in verification reads.
    Expected: Verification read fails deterministically with GET path error.
    Evidence: .sisyphus/evidence/task-4-invoice-payment-error.json
  ```

  **Commit**: YES | Message: `test(tripx): add invoice workflow fixtures` | Files: `task_tripletex/testing/fixtures/create_invoice_basic.json`, `task_tripletex/testing/fixtures/invoice_with_payment.json`, related tests

- [x] 5. Add file-driven and higher-tier fixtures for supplier invoices and bank reconciliation

  **What to do**: Add packaged fixtures for `supplier_invoice_basic` and `bank_reconciliation_file`, including small representative base64 files embedded in the fixture payload or referenced test assets encoded into fixture files. The supplier invoice case must verify file presence plus the target Tripletex entity outcome; the bank reconciliation case must at minimum encode a deterministic expected flow and failure semantics even if the full runtime support lands later in the sequence. Add negative fixture coverage for unsupported MIME or oversized file-policy enforcement.
  **Must NOT do**: Do not defer file test design until after runtime changes. Do not create fixtures that depend on manual extraction or UI actions. Do not assume all file-backed tasks use identical MIME types.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: File-backed fixture design crosses multimodal ingress, accounting workflows, and failure-policy definition.
  - Skills: `[]` — No extra skill required.
  - Omitted: `quality-check` — Testing artifact task.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 7, 8, 14, 15 | Blocked By: 1

  **References**:
  - Pattern: `task_tripletex/service.py:48-82` — file ingress and base64 validation behavior already in place
  - Pattern: `task_tripletex/models.py:14-28` — `SolveFile` and `SolveRequest` shape
  - Docs: `docs/endpoint.md:17-42` — file contract on `/solve`
  - Docs: `docs/overview.md:12-18` and `docs/overview.md:21-33` — file attachments are part of the competition surface
  - Research: Explore findings summarized in draft — missing supplier invoice and bank reconciliation endpoint families from OpenAPI v2.74.00

  **Acceptance Criteria**:
  - [ ] File-backed packaged fixtures load through `fixture_loader.py` and preserve `filename`, `mime_type`, and `content_base64`.
  - [ ] Negative tests cover unsupported MIME and oversized-file policy expectations.
  - [ ] The file-backed integration regression for `supplier_invoice_basic` exists and is explicitly marked `xfail` or equivalent with reason `multimodal runtime not yet implemented`, to be removed in Task 7.

  **QA Scenarios**:
  ```
  Scenario: Supplier invoice fixture carries embedded file into solve request
    Tool: Bash
    Steps: Run a focused test that builds the solve request from `supplier_invoice_basic` and inspects serialized file metadata.
    Expected: Request contains non-empty files list with stable filename, mime_type, and valid base64 content.
    Evidence: .sisyphus/evidence/task-5-supplier-invoice-fixture.json

  Scenario: Unsupported MIME fixture fails deterministically
    Tool: Bash
    Steps: Run a negative pytest case for malformed/unsupported file metadata.
    Expected: Test fails with explicit unsupported-file or policy validation message.
    Evidence: .sisyphus/evidence/task-5-file-policy-error.json
  ```

  **Commit**: YES | Message: `test(tripx): add file-driven higher-tier fixtures` | Files: `task_tripletex/testing/fixtures/supplier_invoice_basic.json`, `task_tripletex/testing/fixtures/bank_reconciliation_file.json`, related tests/assets

- [x] 6. Add request-context injection and synchronous budget guardrails

  **What to do**: Modify the runtime so every `/solve` request explicitly injects current ISO date, request budget metadata, and stable execution guardrails into the model context while preserving the exact synchronous `/solve` JSON contract. Define and implement a request-budget policy that reserves Cloud Run headroom under the 300-second endpoint limit, caps model/tool turns, and exposes these markers in logs for traceability.
  **Must NOT do**: Do not change `/solve` request/response shape, do not move work to background jobs, and do not hardcode a stale date value. Do not exceed the current synchronous execution model in `task_tripletex/service.py:88-111`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Runtime logic change with direct effect on all flows and on Cloud Run timeout behavior.
  - Skills: `[]` — No external skill needed.
  - Omitted: `quality-check` — Use after broader runtime changes aggregate, not for a single planning slice.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 7, 8, 9, 10, 12, 13 | Blocked By: 1

  **References**:
  - Pattern: `task_tripletex/service.py:88-111` — current synchronous `/solve` request handling and response contract
  - Pattern: `task_tripletex/agent.py:320-455` — current Gemini config, send_message flow, and 30-step loop
  - Pattern: `task_tripletex/task_log.py:49-97` — trace fields available for budget/date markers
  - Docs: `docs/endpoint.md:7-16` and `docs/endpoint.md:44-86` — `/solve` timeout and exact response requirements
  - Docs: `agents.md:24-29` and `agents.md:61-70` — existing Cloud Run URL and smoke-check commands

  **Acceptance Criteria**:
  - [ ] A targeted unit/integration test proves that omitted-date prompts receive an injected current ISO date in the model context.
  - [ ] A targeted test proves explicit dates in user prompts are not overwritten by the injector.
  - [ ] `/solve` still returns HTTP 200 with exactly `{"status":"completed"}` and remains synchronous.
  - [ ] Logs include explicit request-budget/date markers for the last run.

  **QA Scenarios**:
  ```
  Scenario: Omitted-date prompt gets deterministic date context
    Tool: Bash
    Steps: Run a focused pytest/integration test that exercises an omitted-date prompt and captures the request trace/log output.
    Expected: Trace shows current ISO date injected once; solve response contract remains unchanged.
    Evidence: .sisyphus/evidence/task-6-request-context.json

  Scenario: Explicit user date is preserved
    Tool: Bash
    Steps: Run a focused test with a prompt containing an explicit date and inspect the logged model context.
    Expected: Injector adds context markers but does not replace the explicit task date.
    Evidence: .sisyphus/evidence/task-6-request-context-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): inject request context and budgets` | Files: `task_tripletex/agent.py`, `task_tripletex/service.py`, `task_tripletex/task_log.py`, related tests

- [x] 7. Implement native multimodal file handling in the existing Gemini request path

  **What to do**: Extend `task_tripletex/agent.py` so `SolveFile` attachments are decoded and passed to Gemini as multimodal parts in the initial chat request. Establish deterministic MIME/file-size policy compatible with Cloud Run memory/time constraints: accept supported small PDFs/images, reject unsupported MIME types or oversized payloads with explicit logs/tests, and keep file handling inside the synchronous request flow.
  **Must NOT do**: Do not introduce separate OCR infrastructure, external storage dependencies, or Gemini File API orchestration unless inline multimodal parts are proven insufficient. Do not silently drop files.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Core runtime change across multimodal input, validation policy, and Cloud Run behavior.
  - Skills: `[]` — No extra skill required.
  - Omitted: `quality-check` — Reserve for grouped runtime validation wave.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 14, 15 | Blocked By: 5, 6

  **References**:
  - Pattern: `task_tripletex/agent.py:334-337` — current explicit file-skipping behavior to replace
  - Pattern: `task_tripletex/models.py:14-21` — `SolveFile.decoded_content()` helper for decoding base64 bytes
  - Pattern: `task_tripletex/service.py:56-76` — validated file ingress from `/solve`
  - Docs: `docs/endpoint.md:17-42` — request file contract and MIME examples
  - Known issue: `agents.md:91-96` — file attachments are not yet implemented and are priority next step #1

  **Acceptance Criteria**:
  - [ ] A focused test proves supported PDF/image files are converted into Gemini request parts instead of being omitted.
  - [ ] A focused test proves unsupported MIME or oversized files fail deterministically according to the defined policy.
  - [ ] The file-backed packaged fixture from Task 5 sends files through the solve path and yields trace evidence of file handling.

  **QA Scenarios**:
  ```
  Scenario: Supported PDF reaches Gemini contents
    Tool: Bash
    Steps: Run focused tests for file-backed solve flow and inspect mocked/recorded contents assembly or logs.
    Expected: Contents include prompt plus binary file parts; no file-skipping branch executes.
    Evidence: .sisyphus/evidence/task-7-multimodal-files.json

  Scenario: Oversized or unsupported file is rejected
    Tool: Bash
    Steps: Run the negative file-policy tests added with Task 5.
    Expected: Solve path rejects deterministically with logged reason; no silent omission.
    Evidence: .sisyphus/evidence/task-7-multimodal-files-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): support multimodal file inputs` | Files: `task_tripletex/agent.py`, possibly `task_tripletex/service.py`, related tests

- [x] 8. Expand the system prompt for uncovered Tier 2/3 Tripletex endpoint families

  **What to do**: Extend the system prompt in `task_tripletex/agent.py` with verified rules and example flows for supplier/incoming invoices, bank statement import and reconciliation, purchase orders, voucher import, attachments, timesheets/payroll, and documented batch endpoints. Keep the base prompt coherent and avoid dynamic prompt routing in this phase; instead, add concise, high-signal sections and explicit efficiency guidance for when batch/import endpoints are preferred.
  **Must NOT do**: Do not rewrite the prompt wholesale, do not remove the currently working employee/customer/order/voucher guidance, and do not add unsupported or unverified endpoint details.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: Requires careful prompt engineering grounded in the verified OpenAPI and current working behavior.
  - Skills: `[]` — No external skill needed.
  - Omitted: `quality-check` — Prompt-only plus tests.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 9, 10, 14, 15 | Blocked By: 5, 6

  **References**:
  - Pattern: `task_tripletex/agent.py:44-317` — current system prompt structure and verified endpoint sections
  - Research summary: uncovered endpoint families from Tripletex OpenAPI v2.74.00 include supplier invoices, bank reconciliation, purchase orders, voucher imports, attachments, payroll/timesheets, inventory, and batch/import endpoints — use these as the source list for prompt additions
  - Docs: `docs/overview.md:42-54` — task categories requiring these families
  - Docs: `docs/examples.md:112-157` — general efficiency principles already aligned with prompt style
  - Known issues: `agents.md:78-87` — documented Tripletex quirks that must remain preserved

  **Acceptance Criteria**:
  - [ ] Prompt tests or snapshot assertions prove new Tier 2/3 sections are present without deleting the current working guidance.
  - [ ] New packaged fixture families reference prompt-covered endpoints rather than undocumented blind spots.
  - [ ] `create_employee_admin` regression still passes after the prompt expansion.

  **QA Scenarios**:
  ```
  Scenario: Prompt includes newly covered endpoint families
    Tool: Bash
    Steps: Run focused tests/snapshots against the prompt string and targeted regression cases.
    Expected: Supplier invoice, reconciliation, purchase order, attachment, and batch guidance are present alongside existing sections.
    Evidence: .sisyphus/evidence/task-8-prompt-expansion.json

  Scenario: Prompt regression catches accidental deletion of employee/invoice rules
    Tool: Bash
    Steps: Run create_employee_admin and create_invoice_basic regressions after prompt update.
    Expected: Existing working flows remain correct; failures expose missing prompt sections.
    Evidence: .sisyphus/evidence/task-8-prompt-expansion-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): extend prompt for higher-tier workflows` | Files: `task_tripletex/agent.py`, prompt tests/regressions

- [x] 9. Shape Tripletex tool responses for IDs, counts, and recoverable errors

  **What to do**: Replace raw full-body tool response forwarding with deterministic shaping rules. Preserve created-entity IDs and follow-up fields for successful writes, cap and summarize large list GETs while keeping `count`/`fullResultSize` semantics, and preserve full `validationMessages` plus status code for 4xx responses. Add tests proving the shaper does not remove data needed for subsequent calls.
  **Must NOT do**: Do not hide IDs, relationship refs, or validation details. Do not reduce error bodies to generic text. Do not shape responses inside the underlying HTTP client if it makes verification harder; keep the client generic and apply shaping at the agent/tool-response boundary.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Central runtime behavior affecting every tool turn and error-repair path.
  - Skills: `[]` — No external skill needed.
  - Omitted: `quality-check` — Use within broader regression wave.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 10, 11, 12, 14, 15 | Blocked By: 6, 8

  **References**:
  - Pattern: `task_tripletex/agent.py:402-433` — current raw response forwarding path
  - Pattern: `task_tripletex/client.py:88-115` — generic executed operation result shape to preserve under the hood
  - Docs: `docs/endpoint.md:88-112` — Tripletex list wrapper expectations
  - Docs: `docs/examples.md:149-157` — efficiency guidance motivating reduced context waste
  - Metis directive: response shaping must preserve IDs and `validationMessages`

  **Acceptance Criteria**:
  - [ ] Tests prove successful POST responses retain created entity IDs and necessary follow-up fields.
  - [ ] Tests prove large GET list responses are capped/summarized while still exposing counts and the first required values.
  - [ ] Tests prove 422 responses retain full structured validation payload.

  **QA Scenarios**:
  ```
  Scenario: Successful write response preserves ID for downstream use
    Tool: Bash
    Steps: Run focused tests around response shaping and inspect structured function-response payloads.
    Expected: POST result contains status code plus entity ID/required fields after shaping.
    Evidence: .sisyphus/evidence/task-9-response-shaping.json

  Scenario: 422 validation details survive shaping
    Tool: Bash
    Steps: Run a negative test that feeds a synthetic or recorded 422 body through the shaper.
    Expected: `validationMessages` remain present and verbatim enough for repair logic.
    Evidence: .sisyphus/evidence/task-9-response-shaping-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): shape tool responses for reasoning efficiency` | Files: `task_tripletex/agent.py`, related tests

- [x] 10. Add bounded 422 recovery and fail-fast rules for non-recoverable 4xx responses

  **What to do**: Implement one bounded self-correction turn for recoverable validation failures. Parse structured `validationMessages`, add concise repair hints into the next model turn, and stop after a second 4xx on the same objective or after a non-recoverable client error. Log retry decisions explicitly so `/logs` shows whether the agent retried or failed fast.
  **Must NOT do**: Do not create open-ended retry loops. Do not retry 401/403/404 blindly. Do not let retry logic mask the first failure in logs or evidence.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Retry logic impacts score directly and must be tightly bounded.
  - Skills: `[]` — No extra skill needed.
  - Omitted: `quality-check` — Runtime behavior covered by focused tests and later regression wave.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 14, 15 | Blocked By: 8, 9

  **References**:
  - Pattern: `task_tripletex/agent.py:351-455` — current step loop and function-response turnaround
  - Pattern: `task_tripletex/task_log.py:56-74` — log events to extend for retry decisions
  - Docs: `task_tripletex/agent.py:309-317` — current prompt-side error handling guidance and common validation messages
  - Metis directive: single bounded retry only for recoverable structured validation errors

  **Acceptance Criteria**:
  - [ ] Recoverable 422 test case triggers exactly one retry and then succeeds or stops.
  - [ ] Non-recoverable 4xx test case triggers zero retries.
  - [ ] `/logs` trace contains explicit retry/fail-fast markers.

  **QA Scenarios**:
  ```
  Scenario: Recoverable validation error gets one repair turn
    Tool: Bash
    Steps: Run focused tests or staged regression causing a known 422 (for example missing required field) that can be corrected.
    Expected: Exactly one retry is logged; second attempt uses corrected payload.
    Evidence: .sisyphus/evidence/task-10-bounded-retry.json

  Scenario: Non-recoverable client error stops immediately
    Tool: Bash
    Steps: Run a negative test producing a non-recoverable 4xx condition.
    Expected: No retry occurs; fail-fast marker appears in the trace.
    Evidence: .sisyphus/evidence/task-10-bounded-retry-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add bounded validation recovery` | Files: `task_tripletex/agent.py`, `task_tripletex/task_log.py`, related tests

- [x] 11. Add request-local GET deduplication with post-write invalidation

  **What to do**: Implement an exact-key request-local cache for GET operations inside a single `/solve` invocation. The cache key must include method, path, and serialized query params. Reads may be reused only within the current request, and any write that can affect the resource family must invalidate relevant cached entries conservatively. Add explicit logs for cache hit/miss/invalidate events.
  **Must NOT do**: Do not cache across requests. Do not reuse cached GET responses after writes without invalidation. Do not cache POST/PUT/PATCH/DELETE results as read substitutes.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Requires careful state management in the request loop without changing client semantics.
  - Skills: `[]` — No external skill needed.
  - Omitted: `quality-check` — Focused runtime/test slice.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 14, 15 | Blocked By: 9

  **References**:
  - Pattern: `task_tripletex/client.py:88-115` — current per-operation execution boundary
  - Pattern: `task_tripletex/agent.py:364-445` — where function calls are executed and responses returned
  - Metis directive: request-local exact-key cache only with write-triggered invalidation
  - Docs: `docs/scoring.md:35-55` — fewer unnecessary calls improve efficiency bonus when correctness is perfect

  **Acceptance Criteria**:
  - [ ] Focused tests prove repeated identical GETs within one solve request reuse cached data.
  - [ ] Focused tests prove a write invalidates relevant cached reads before subsequent GETs.
  - [ ] Trace output exposes cache hit/miss/invalidation markers.

  **QA Scenarios**:
  ```
  Scenario: Repeated GET becomes cache hit
    Tool: Bash
    Steps: Run focused tests that request the same GET twice within one synthetic solve flow.
    Expected: First request logs miss, second logs hit, and only one upstream GET is executed.
    Evidence: .sisyphus/evidence/task-11-get-cache.json

  Scenario: Write invalidates affected cache entries
    Tool: Bash
    Steps: Run focused tests with GET -> POST/PUT -> GET on the same resource family.
    Expected: Second GET is not served from stale cache; invalidation marker appears.
    Evidence: .sisyphus/evidence/task-11-get-cache-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add request-local get cache` | Files: `task_tripletex/agent.py`, possibly helper module/tests

- [x] 12. Enforce deterministic model/runtime budgets and preserve current best-case employee performance

  **What to do**: Tune the Gemini/runtime configuration for reproducibility and timeout safety without replacing the model. Add explicit temperature/determinism settings if supported by the SDK, confirm step-budget logic remains within Cloud Run headroom, and add regression coverage proving `create_employee_admin` remains at 100% correctness with the known optimal 2-call/0-4xx profile after all prior runtime changes.
  **Must NOT do**: Do not swap models in this phase. Do not relax the employee regression target. Do not introduce adaptive thinking levels yet.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: This is a global runtime-tuning gate that must preserve the known good baseline exactly.
  - Skills: `[]` — No extra skill required.
  - Omitted: `quality-check` — Use alongside full regression in Task 14.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 14, 15 | Blocked By: 6, 9, 10

  **References**:
  - Pattern: `task_tripletex/agent.py:321-333` — current Gemini config and thinking settings
  - Known baseline: `agents.md:18-23` — verified 100%/2 calls/0 errors on `create_employee_admin`
  - Pattern: `task_tripletex/testing/cli.py:219-274` — JSON/text outputs exposing correctness and proxy metrics
  - Metis directive: define request budget policy under the 5-minute Cloud Run limit

  **Acceptance Criteria**:
  - [ ] `create_employee_admin` regression reaches correctness `1.0`, `total_calls == 2`, `write_calls == 1`, and `client_error_calls == 0` on the VM proxy path.
  - [ ] A focused test or trace check confirms explicit runtime budget markers are logged.
  - [ ] No model replacement or endpoint contract change occurs.

  **QA Scenarios**:
  ```
  Scenario: Employee admin baseline remains optimal
    Tool: Bash
    Steps: Run the packaged-case CLI for `create_employee_admin` using the VM localhost solve URL and capture JSON output.
    Expected: Correctness 1.0, `proxy_metrics.total_calls == 2`, `proxy_metrics.write_calls == 1`, `proxy_metrics.client_error_calls == 0`, exact success response true.
    Evidence: .sisyphus/evidence/task-12-employee-baseline.json

  Scenario: Runtime budget markers appear in logs
    Tool: Bash
    Steps: Call `/solve` with the known employee prompt, then fetch `/logs` from the active service or local run.
    Expected: Trace contains budget markers and no evidence of time-budget overflow.
    Evidence: .sisyphus/evidence/task-12-employee-baseline-error.json
  ```

  **Commit**: YES | Message: `perf(tripx): harden model budgets and baseline` | Files: `task_tripletex/agent.py`, `task_tripletex/task_log.py`, related tests

- [x] 13. Enrich traceability for file, retry, cache, and request-context decisions

  **What to do**: Extend `task_tripletex/task_log.py` usage so `/logs` exposes the decision breadcrumbs needed for debugging new behaviors: injected date, file acceptance/rejection, response shaping summaries, retry reason, cache hit/miss/invalidation, and request budget state. Keep the logger in-memory and request-scoped enough for current needs; do not redesign the logging backend.
  **Must NOT do**: Do not build durable logging infrastructure, external sinks, or observability pipelines. Do not leak sensitive file contents or raw credentials into logs.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: This is a localized observability hardening task once the core behaviors exist.
  - Skills: `[]` — No extra skill needed.
  - Omitted: `quality-check` — Small focused slice.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 14, 15 | Blocked By: 6, 7, 9, 10, 11

  **References**:
  - Pattern: `task_tripletex/task_log.py:19-101` — current trace model and snapshot output
  - Pattern: `task_tripletex/service.py:115-121` — `/logs` endpoint contract to preserve
  - Docs: `agents.md:68-70` — current `curl ... /logs | python3 -m json.tool` inspection path
  - Metis directive: `/logs` should show date/retry/cache markers for agent-executed QA

  **Acceptance Criteria**:
  - [ ] `/logs` snapshot includes event entries for file-policy decisions, retry decisions, cache decisions, and injected date.
  - [ ] No event logs raw file contents or session tokens.
  - [ ] Existing `/logs` JSON structure remains parseable.

  **QA Scenarios**:
  ```
  Scenario: Logs expose new runtime decision markers
    Tool: Bash
    Steps: Execute representative solve flows (text-only and file-backed), then fetch `/logs` and pretty-print JSON.
    Expected: Snapshot includes the new marker events with concise detail dictionaries and no secret leakage.
    Evidence: .sisyphus/evidence/task-13-log-markers.json

  Scenario: Logs stay sanitized
    Tool: Bash
    Steps: Run a targeted test that searches `/logs` snapshot for file bodies or session token strings after a solve.
    Expected: Sensitive payload content is absent; test passes.
    Evidence: .sisyphus/evidence/task-13-log-markers-error.json
  ```

  **Commit**: YES | Message: `chore(tripx): enrich runtime trace markers` | Files: `task_tripletex/task_log.py`, `task_tripletex/agent.py`, related tests

- [x] 14. Run full VM proxy regression wave across prioritized packaged cases

  **What to do**: Use the documented GCE VM workflow to run the packaged-case CLI against the localhost uvicorn service for all prioritized packaged cases: `create_employee_admin`, `create_customer`, `create_product`, `create_project_basic`, `create_journal_entry_basic`, `create_travel_expense_basic`, `create_invoice_basic`, `invoice_with_payment`, `supplier_invoice_basic`, `bank_reconciliation_file`, and any additional file case that is enabled by runtime support. Capture JSON results for each case and classify failures into fixture issue, runtime bug, or sandbox/API constraint.
  **Must NOT do**: Do not test Cloud Run through the local recording proxy — `agents.md:88-89` states Cloud Run cannot reach the localhost proxy. Do not skip the employee baseline run. Do not hand-wave failures without attaching evidence.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: This is the main integrated verification wave across many cases and environments.
  - Skills: `[]` — Existing CLI + VM flow are enough.
  - Omitted: `quality-check` — The key here is live regression evidence.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 15 | Blocked By: 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13

  **References**:
  - Command pattern: `agents.md:39-58` — VM sync, uvicorn launch, and packaged-case CLI execution
  - Pattern: `task_tripletex/testing/cli.py:182-286` — integrated output fields to inspect
  - Pattern: `task_tripletex/testing/reverse_proxy_recorder.py:237-262` — proxy metrics summary used by regression
  - Docs: `docs/testing_framework.md:27-109` — deterministic evaluator semantics and disqualification conditions

  **Acceptance Criteria**:
  - [ ] JSON evidence exists for every prioritized packaged case.
  - [ ] `create_employee_admin` remains 100% correct and optimal on the VM path.
  - [ ] Every failing case has an explicit classification and linked evidence artifact.

  **QA Scenarios**:
  ```
  Scenario: Full prioritized VM regression completes
    Tool: Bash
    Steps: Sync code to VM, restart uvicorn, and run the packaged-case CLI for each prioritized case, saving JSON output per case.
    Expected: Each case yields machine-readable correctness/proxy/score output; employee baseline remains green.
    Evidence: .sisyphus/evidence/task-14-vm-regression.json

  Scenario: Regression failure is classified, not ignored
    Tool: Bash
    Steps: For any failing case, rerun that case once and compare output to classify fixture/runtime/API issue.
    Expected: Evidence bundle includes classification notes and repeated deterministic failure or resolution.
    Evidence: .sisyphus/evidence/task-14-vm-regression-error.json
  ```

  **Commit**: NO | Message: `n/a` | Files: `.sisyphus/evidence/*` only

- [x] 15. Deploy to Cloud Run and run smoke checks against the active service

  **What to do**: After VM proxy regression is green enough for deployment, build and deploy the current codebase to the existing Cloud Run service using the exact `gcloud` flow from `agents.md`. Then run smoke checks against the deployed `/solve` and `/logs` endpoints with representative text-only and file-backed prompts. Confirm the live service returns the exact JSON contract and emits trace markers for the new runtime behaviors.
  **Must NOT do**: Do not change the service name, region, or deployment shape. Do not attempt proxy-based evaluation against Cloud Run. Do not submit to the official leaderboard until the smoke checks are clean.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Live deployment gate with platform-specific validation against the current instance.
  - Skills: `[]` — Existing deployment instructions are sufficient.
  - Omitted: `deploy` — The repository already contains exact gcloud commands in `agents.md`; no broader deployment skill workflow is needed.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Final Verification Wave | Blocked By: 7, 8, 9, 10, 11, 12, 13, 14

  **References**:
  - Command pattern: `agents.md:24-29` — build/deploy command and target service/project/region
  - Command pattern: `agents.md:61-70` — Cloud Run `/solve` smoke and `/logs` inspection commands
  - Docs: `docs/endpoint.md:44-86` — exact success response required by the competition
  - Known issue: `agents.md:88-89` — Cloud Run cannot be exercised through the local recording proxy

  **Acceptance Criteria**:
  - [ ] `gcloud builds submit ... && gcloud run deploy ...` completes successfully against `tripletex` in `europe-north1`.
  - [ ] Text-only smoke prompt returns HTTP 200 and exact success JSON.
  - [ ] File-backed smoke prompt returns HTTP 200 and exact success JSON once file support is live.
  - [ ] `/logs` includes the latest trace with injected date, retry/cache markers as applicable.

  **QA Scenarios**:
  ```
  Scenario: Cloud Run text-only smoke passes
    Tool: Bash
    Steps: Run the existing curl POST against `https://tripletex-124894558027.europe-north1.run.app/solve` with a representative employee/customer prompt.
    Expected: HTTP 200 and body exactly `{"status":"completed"}`.
    Evidence: .sisyphus/evidence/task-15-cloudrun-smoke.json

  Scenario: Cloud Run logs show latest runtime markers
    Tool: Bash
    Steps: Run `curl -s https://tripletex-124894558027.europe-north1.run.app/logs | python3 -m json.tool` after the smoke request.
    Expected: Latest trace reflects the deployed behavior and includes date/retry/cache/file markers without secrets.
    Evidence: .sisyphus/evidence/task-15-cloudrun-smoke-error.json
  ```

  **Commit**: NO | Message: `n/a` | Files: deployment only, `.sisyphus/evidence/*`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Keep commits atomic and revertable in this order: fixture foundation → breadth fixture packs → request context/date → multimodal files → prompt expansion → response shaping → bounded retry → GET cache → deterministic budgets → trace enrichment → regression hardening → deploy gate.
- Do not mix multiple runtime behaviors in one commit.
- Use commit scopes tied to capability slices so Cloud Run regressions can be bisected quickly.

## Success Criteria
- The agent handles file-free and file-backed prompts through the same synchronous `/solve` interface on the current Cloud Run service.
- The packaged fixture set covers the main scoring surfaces instead of only `create_employee_admin`.
- Existing optimal employee behavior is preserved while new task families move from unknown/zero to repeatably correct.
- Runtime efficiency improves through fewer avoidable GETs and fewer cascading 4xx errors, not through infrastructure changes.
