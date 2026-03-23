# TripX C-lite Controller Redesign Plan

## TL;DR
> **Summary**: Replace the monolithic all-endpoint Gemini agent with a single controller that routes high-confidence known workflows through deterministic handlers and everything else through a focused LLM path or the preserved legacy monolith, while keeping the external `/solve` contract unchanged.
> **Deliverables**:
> - controller, routing, attachment-normalization, post-condition, and focused-fallback infrastructure
> - deterministic handlers for customer, product, employee admin, invoice, invoice+payment, project, travel expense, and journal entry flows
> - focused fallback coverage for supplier invoice, bank reconciliation, and unknown task families with local endpoint guardrails
> - full regression evidence from pytest, VM packaged-case runs, Cloud Run smoke checks, and `/logs` telemetry
> **Effort**: XL
> **Parallel**: YES - 3 waves
> **Critical Path**: 1 → 2 → 3/4/5 → 6 → 7/8/9/10/11 → 12 → 13/14 → 15

## Context
### Original Request
- Analyze previous bad scores from `/logs` and identify what is going wrong.
- Evaluate whether a cleaner architecture is better than the current large system prompt.
- Create a plan for the best architecture after checking the competition docs.

### Interview Summary
- Current failure patterns are not only prompt-length issues; they are execution-control issues: the agent stops before terminal state, guesses wrong endpoints, and burns 4xx budget.
- Competition docs confirm 30 task types and 56 prompt variants per task, so pure deterministic coverage is not realistic.
- Oracle recommended Architecture C-lite: one controller, shared primitives, deterministic routing for high-confidence known workflows, and focused LLM fallback for everything else.
- You approved C-lite and selected **test-after** as the implementation strategy.

### Metis Review (gaps addressed)
- Put internal route outcomes and post-condition verification in the controller contract from the start.
- Keep the legacy monolithic `run_agent` reachable during migration so unknown-task behavior does not regress abruptly.
- Never hand control to a second write-capable route after a deterministic path has already issued a write.
- Treat CSV attachment support as phase-1 scope because `bank_reconciliation_file` already ships a `text/csv` attachment.
- Exclude logger hardening, secret migration, and “all 30 tasks deterministic” work from this plan.

## Work Objectives
### Core Objective
Maximize known-task competition score and improve unknown-task safety by replacing the current monolithic prompt-only execution path with a verifier-backed C-lite controller that preserves the exact external endpoint contract.

### Deliverables
- `task_tripletex/controller.py` entrypoint that owns route choice, outcome tracking, fallback policy, and post-condition checks.
- `task_tripletex/routing.py`, `task_tripletex/attachments.py`, `task_tripletex/focused_agent.py`, `task_tripletex/primitives.py`, `task_tripletex/deterministic.py`, and `task_tripletex/postconditions.py`.
- Minimal model additions in `task_tripletex/models.py` for task families, route decisions, and route execution results.
- Focused regression updates in `tests/task_tripletex_testing/` plus VM and Cloud Run evidence.

### Definition of Done (verifiable conditions with commands)
- `pytest tests/task_tripletex_testing/test_request_context_runtime.py tests/task_tripletex_testing/test_framework_integration.py tests/task_tripletex_testing/test_fixture_loader.py tests/task_tripletex_testing/test_cli.py tests/task_tripletex_testing/test_scoring.py tests/task_tripletex_testing/test_verifier.py` passes locally.
- The deterministic fixture set — `create_customer`, `create_product`, `create_employee_admin`, `create_invoice_basic`, `invoice_with_payment`, `create_project_basic`, `create_travel_expense_basic`, `create_journal_entry_basic` — returns correctness `1.0` with no contract/proxy violations on the VM using the `python -m task_tripletex.testing.cli` workflow in `AGENTS.md:39-58`.
- `supplier_invoice_basic` and `bank_reconciliation_file` return correctness `1.0` on the VM, produce `route_decision` and `postcondition_check_finished` telemetry in `/logs`, and do not violate the `/solve` contract or proxy/auth rules.
- `curl -s -X POST https://tripletex-124894558027.europe-north1.run.app/solve -H 'Content-Type: application/json' -d '{...}'` returns HTTP 200 with exact body `{"status":"completed"}` after deployment, and `curl -s https://tripletex-124894558027.europe-north1.run.app/logs | python3 -m json.tool` shows the latest route telemetry.
- No change introduces background work, polling, alternate response bodies, or direct calls to Tripletex outside the provided proxy base URL.

### Must Have
- Preserve the exact service contract enforced by `task_tripletex/testing/endpoint_runner.py:86-163`.
- Preserve the current retry, cache, and budget guardrail behavior from `task_tripletex/agent.py:917-1454` for the legacy path and reuse those mechanics for the focused fallback path.
- Make post-condition GET verification mandatory for deterministic handlers and known focused-fallback routes because write calls and 4xx errors, not GET count, drive efficiency in `task_tripletex/testing/scoring.py:120-135`.
- Keep the legacy `run_agent` path callable until the controller, deterministic handlers, and focused fallback are proven by regression evidence.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Must NOT change `/solve` away from HTTP 200 + exact `{"status":"completed"}` on successful request parsing.
- Must NOT add async job orchestration, Cloud Tasks, Pub/Sub, background queues, or any other platform redesign.
- Must NOT attempt deterministic coverage for all 30 competition tasks in phase 1.
- Must NOT run a second write-capable route after a deterministic or focused route has already performed a write.
- Must NOT guess undocumented write endpoints like `/paymentType` or `/ledger/paymentType`; disallowed operations must be blocked locally before proxy traffic.
- Must NOT include logger hardening, secret migration, or non-score-related refactors in this implementation wave.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: **tests-after** using the existing pytest surface, the packaged-case CLI, VM proxy E2E runs, and Cloud Run smoke checks.
- QA policy: Every task includes one happy-path scenario and one failure/edge scenario with exact commands or targeted pytest nodes.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.json`, `.sisyphus/evidence/task-{N}-{slug}.log`, or `.sisyphus/evidence/task-{N}-{slug}-error.json`
- Contract guardrail: the endpoint contract in `task_tripletex/testing/endpoint_runner.py:125-163` is score-blocking, so contract tests are mandatory in every integration step.

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. Shared routing/attachment/fallback dependencies come first so handler work can proceed without re-deciding architecture.

Wave 1: Tasks 1-5 — controller contract, routing, attachments, focused-agent extraction, post-condition infrastructure

Wave 2: Tasks 6-11 — shared primitives plus deterministic handlers for the known high-confidence families

Wave 3: Tasks 12-15 — focused fallback families, service wiring, regression expansion, VM/Cloud Run validation

### Dependency Matrix (full, all tasks)
| Task | Depends On | Enables |
| --- | --- | --- |
| 1 | none | 2, 4, 5, 13, 14 |
| 2 | 1 | 4, 5, 7, 8, 9, 10, 11, 12 |
| 3 | 1 | 4, 12, 13, 14, 15 |
| 4 | 1, 2, 3 | 7, 8, 9, 10, 11, 12 |
| 5 | 1, 2 | 7, 8, 9, 10, 11, 12, 13, 14 |
| 6 | 1, 2, 5 | 7, 8, 9, 10, 11, 12 |
| 7 | 2, 5, 6 | 15 |
| 8 | 2, 5, 6 | 15 |
| 9 | 2, 5, 6 | 15 |
| 10 | 2, 5, 6 | 15 |
| 11 | 2, 5, 6 | 15 |
| 12 | 2, 3, 4, 5, 6 | 13, 14, 15 |
| 13 | 1, 3, 5, 7, 8, 9, 10, 11, 12 | 14, 15 |
| 14 | 1, 2, 3, 4, 5, 12, 13 | 15 |
| 15 | 7, 8, 9, 10, 11, 12, 13, 14 | Final Verification Wave |

### Agent Dispatch Summary (wave → task count → categories)
| Wave | Task Count | Recommended Categories |
| --- | --- | --- |
| 1 | 5 | unspecified-high, deep |
| 2 | 6 | unspecified-high, quick |
| 3 | 4 | unspecified-high, deep, quick |

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Introduce controller contract and route outcome model

  **What to do**: Add a new controller entrypoint that becomes the single internal execution surface for `/solve`. Create typed route/outcome models for `task_family`, `route_kind`, `confidence`, `write_started`, `postcondition_required`, and final `route_outcome` (`success`, `incomplete`, `fail`). The controller must preserve the external response contract while allowing internal incomplete/fail semantics for logging and route decisions. Keep the legacy `run_agent()` path reachable as one route kind rather than deleting or inlining it.
  **Must NOT do**: Do not change the external `/solve` JSON body, status code, or content type. Do not delete `task_tripletex.agent.run_agent`. Do not move network behavior into the controller yet beyond route orchestration and typed results.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: This is the architectural seam the rest of the migration depends on.
  - Skills: `[]` — No extra skill is required; repository-local architecture and test discipline matter most.
  - Omitted: `quality-check` — Useful later, but not necessary to plan this atomic controller introduction.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 4, 5, 13, 14 | Blocked By: none

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `task_tripletex/service.py:23-130` — current `/solve` orchestration and `/logs` exposure that must remain externally identical
  - Pattern: `task_tripletex/agent.py:1012-1454` — legacy monolithic execution path to preserve behind the controller
  - API/Type: `task_tripletex/models.py:24-93` — current request/response/budget models to extend instead of duplicating
  - Test: `task_tripletex/testing/endpoint_runner.py:86-163` — score-blocking endpoint contract rules
  - Test: `tests/task_tripletex_testing/test_framework_integration.py:300-330` — existing non-200 contract-failure coverage
  - Docs: `docs/endpoint.md:7-87` — mandatory `/solve` contract and proxy/base-url constraints
  - Docs: `docs/scoring.md:35-55` — correctness precedes efficiency bonus, so false-success semantics are unacceptable

  **Acceptance Criteria** (agent-executable only):
  - [ ] `pytest tests/task_tripletex_testing/test_framework_integration.py -k "contract or solve"` passes with new controller wiring in place.
  - [ ] A new focused unit/integration test proves the controller can record an internal non-success route outcome while `/solve` still returns exact `{"status":"completed"}` for successful request parsing.
  - [ ] `task_tripletex/service.py` delegates into the controller instead of calling the legacy path directly, while the legacy path remains importable and callable.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Controller preserves external /solve contract
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_framework_integration.py -k "contract or solve"`.
    Expected: All selected tests pass; response remains HTTP 200 + exact JSON body.
    Evidence: .sisyphus/evidence/task-1-controller-contract.json

  Scenario: Internal incomplete outcome does not leak alternate response body
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_controller_contract.py -k "internal incomplete"` after adding a forced internal non-success route case.
    Expected: Internal telemetry marks non-success route outcome, but HTTP response is still the exact contract body.
    Evidence: .sisyphus/evidence/task-1-controller-contract-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add controller execution contract` | Files: `task_tripletex/controller.py`, `task_tripletex/models.py`, `task_tripletex/service.py`, targeted tests

- [ ] 2. Add rules-first task routing and confidence gating

  **What to do**: Implement a routing layer that classifies requests by final business outcome, not nouns alone. It must recognize at least these families from prompt + attachment metadata: `create_customer`, `create_product`, `create_employee_admin`, `create_invoice_basic`, `invoice_with_payment`, `create_project_basic`, `create_travel_expense_basic`, `create_journal_entry_basic`, `supplier_invoice_basic`, `bank_reconciliation_file`, and `unknown`. Use deterministic rules first across the 7 competition languages; only if rules cannot produce `HIGH` confidence should the route invoke one zero-temperature Gemini classification call that returns strict JSON `{task_family, confidence, endpoint_pack}`. Final routing rule is fixed: `HIGH -> deterministic handler if one exists, else focused fallback pack`; `MEDIUM -> focused fallback pack`; `LOW -> legacy monolith before any write`. Return a route decision object with confidence and allowed endpoint family list.
  **Must NOT do**: Do not dispatch directly to write-capable logic from the classifier. Do not rely on plain English keywords only. Do not classify `invoice_with_payment` as generic invoice when payment intent is explicitly present.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Multilingual routing plus confidence gating is subtle and central to C-lite.
  - Skills: `[]` — No external skill needed.
  - Omitted: `quality-check` — Still architecture-heavy; correctness is covered via targeted tests.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 4, 5, 7, 8, 9, 10, 11, 12 | Blocked By: 1

  **References**:
  - Pattern: `docs/overview.md:23-32` — 30 tasks, 56 variants, 7 languages
  - Pattern: `task_tripletex/testing/fixtures/create_invoice_basic.json:1-110` — invoice intent without payment
  - Pattern: `task_tripletex/testing/fixtures/invoice_with_payment.json:1-108` — invoice intent with explicit payment terminal state
  - Pattern: `task_tripletex/testing/fixtures/bank_reconciliation_file.json:1-59` — file-backed reconciliation family
  - Pattern: `task_tripletex/testing/fixtures/supplier_invoice_basic.json:1-68` — PDF-backed supplier invoice family
  - Test: `tests/task_tripletex_testing/test_fixture_loader.py:302-349` — fixture-level invariants that differentiate invoice vs invoice+payment
  - Docs: `docs/overview.md:42-54` — task category examples to reflect in routing families

  **Acceptance Criteria**:
  - [ ] A new routing test suite covers positive and negative examples for each known family plus explicit ambiguous prompts.
  - [ ] `invoice_with_payment` prompts route to the payment workflow family, not plain invoice.
  - [ ] Low-confidence prompts return a route decision that allows the focused fallback or legacy fallback, never a forced deterministic handler.

  **QA Scenarios**:
  ```
  Scenario: Known prompts map to correct route families
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_routing.py` after adding prompt cases for all 10 known fixtures.
    Expected: Each known fixture prompt resolves to the expected family with confidence metadata present.
    Evidence: .sisyphus/evidence/task-2-routing.json

  Scenario: Ambiguous prompt is downgraded instead of misrouted
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_routing.py -k "ambiguous"` with a deliberately ambiguous multilingual invoice-related prompt.
    Expected: Route confidence is below deterministic threshold and the router selects focused/legacy fallback rather than deterministic invoice handling.
    Evidence: .sisyphus/evidence/task-2-routing-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add task family router` | Files: `task_tripletex/routing.py`, `task_tripletex/models.py`, targeted tests

- [ ] 3. Normalize attachment handling for PDF/image multimodal and CSV/text extraction

  **What to do**: Extract attachment preparation into a dedicated module. Keep existing inline multimodal behavior for supported binary formats, but add a deterministic text path for CSV/text-like attachments used by bank reconciliation. The controller must expose normalized attachment metadata to routing and downstream execution. For CSV/text, decode to UTF-8 text and pass compact extracted content into the focused route rather than rejecting it at request validation time.
  **Must NOT do**: Do not upload files to external storage or introduce asynchronous file processing. Do not remove the existing MIME/size guardrails for PDF/image support. Do not keep rejecting `text/csv` after this task.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: This changes request preprocessing and known file-driven task viability.
  - Skills: `[]` — No external skill needed.
  - Omitted: `quality-check` — Targeted tests provide immediate correctness feedback.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 12, 13, 14, 15 | Blocked By: 1

  **References**:
  - Pattern: `task_tripletex/agent.py:623-685` — current inline file-preparation behavior to extract and generalize
  - Pattern: `task_tripletex/testing/fixtures/supplier_invoice_basic.json:1-68` — PDF fixture that must keep working
  - Pattern: `task_tripletex/testing/fixtures/bank_reconciliation_file.json:1-59` — CSV fixture that currently cannot run through the agent
  - Test: `tests/task_tripletex_testing/test_framework_integration.py:332-376` — current multimodal attachment forwarding assertion
  - Test: `tests/task_tripletex_testing/test_framework_integration.py:378-533` — current rejected-MIME and oversize behavior
  - Test: `tests/task_tripletex_testing/test_fixture_loader.py:225-255` — explicit file-policy expectations for packaged fixtures
  - Docs: `docs/endpoint.md:17-42` — attachments are part of the competition request contract

  **Acceptance Criteria**:
  - [ ] Existing PDF multimodal forwarding tests still pass.
  - [ ] A new test proves `text/csv` attachments are accepted, decoded, and exposed to routing/execution without leaking `content_base64` into logs.
  - [ ] Oversize and unsupported-MIME rejection still works for genuinely unsupported formats.

  **QA Scenarios**:
  ```
  Scenario: PDF and CSV attachments take the correct normalization path
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_framework_integration.py -k "supplier_invoice or invalid_file_policy or attachment"` plus the new CSV acceptance node.
    Expected: PDF remains multimodal, CSV is accepted as text input, and unsupported types still fail explicitly.
    Evidence: .sisyphus/evidence/task-3-attachments.json

  Scenario: CSV normalization keeps secrets redacted in /logs
    Tool: Bash
    Steps: Run the new CSV-focused integration/runtime test and inspect the serialized trace assertion.
    Expected: Filename and policy are logged, but `content_base64` and raw file bytes are absent/redacted.
    Evidence: .sisyphus/evidence/task-3-attachments-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add attachment normalization paths` | Files: `task_tripletex/attachments.py`, `task_tripletex/agent.py` or extracted callers, `task_tripletex/models.py`, targeted tests

- [ ] 4. Build focused prompt slices and conservative fallback execution path

  **What to do**: Create a focused LLM execution path that receives a short, route-specific system instruction instead of the full `SYSTEM_PROMPT`. The fallback must inherit current retry, cache, and budget behavior, but it may only access an endpoint shortlist derived from the route decision. It must support supplier invoice, bank reconciliation, and unknown tasks; for unknown tasks it may read/discover conservatively but must not perform speculative writes to endpoints outside the shortlist.
  **Must NOT do**: Do not feed the full 550-line prompt to the focused fallback. Do not allow unrestricted arbitrary path calls. Do not delete the legacy monolith in this task.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: This is the core unknown-task safety improvement and replaces prompt clutter with scoped context.
  - Skills: `[]` — No external skill needed.
  - Omitted: `quality-check` — Runtime tests are more important at this step.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 12, 13, 14, 15 | Blocked By: 1, 2, 3

  **References**:
  - Pattern: `task_tripletex/agent.py:77-552` — current monolithic prompt content to split into route-scoped slices
  - Pattern: `task_tripletex/agent.py:603-620` — current Gemini config builder to reuse for focused execution
  - Pattern: `task_tripletex/agent.py:917-1454` — retry, cache, and budget logic to preserve
  - Test: `tests/task_tripletex_testing/test_request_context_runtime.py:322-494` — retry behavior expectations
  - Test: `tests/task_tripletex_testing/test_request_context_runtime.py:743-989` — request-local cache and budget guardrail expectations
  - Docs: `docs/scoring.md:37-55` — 4xx cleanliness matters, so speculative writes are forbidden

  **Acceptance Criteria**:
  - [ ] A new focused-fallback test proves only route-allowed endpoint paths can be executed.
  - [ ] Existing retry/cache/budget tests either keep passing as-is or are ported to the focused path without weaker assertions.
  - [ ] Route-scoped prompt builders exist for at least invoice/payment, supplier invoice, bank reconciliation, project, journal entry, and unknown fallback families.

  **QA Scenarios**:
  ```
  Scenario: Focused fallback uses bounded endpoint shortlist
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_focused_agent.py` after adding allowed/disallowed path execution cases.
    Expected: Allowed paths pass through; disallowed write paths are blocked locally before proxy traffic.
    Evidence: .sisyphus/evidence/task-4-focused-fallback.json

  Scenario: Unknown route refuses speculative write to disallowed endpoint
    Tool: Bash
    Steps: Run a targeted test where the model attempts a blocked write path outside the shortlist.
    Expected: Controller/fallback returns internal non-success route outcome and no upstream write call occurs.
    Evidence: .sisyphus/evidence/task-4-focused-fallback-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add focused fallback agent path` | Files: `task_tripletex/focused_agent.py`, extracted prompt slices, targeted tests

- [ ] 5. Add post-condition verification library and controller stop rules

  **What to do**: Implement route-family-specific post-condition checks that verify the requested terminal state before the controller marks a route as successful. Use GET-based checks that mirror the existing packaged-fixture selectors wherever practical: invoice routes must verify invoice/order linkage, payment routes must verify `isPaid`, project routes must verify manager linkage, and so on. Define controller stop rules: deterministic handler success requires post-condition pass; focused fallback success requires post-condition pass for known families and at minimum non-ambiguous objective completion for unknown families; legacy fallback remains the final safety route during migration.
  **Must NOT do**: Do not count a successful POST/PUT as route success by itself. Do not use verifier code directly against the local proxy harness inside production request flow. Do not add extra write calls during post-condition verification.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: This directly fixes the “TASK COMPLETED too early” failure class.
  - Skills: `[]` — No extra skill needed.
  - Omitted: `quality-check` — Focus on correctness and test evidence first.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | Blocked By: 1, 2

  **References**:
  - Pattern: `task_tripletex/testing/verifier.py:17-129` — canonical selector and field-path semantics to mirror
  - Pattern: `task_tripletex/testing/fixtures/create_invoice_basic.json:8-110` — invoice linkage checks
  - Pattern: `task_tripletex/testing/fixtures/invoice_with_payment.json:8-108` — paid-invoice terminal state checks
  - Pattern: `task_tripletex/testing/fixtures/create_project_basic.json:8-78` — project manager linkage checks
  - Pattern: `task_tripletex/testing/fixtures/create_journal_entry_basic.json:8-79` — voucher amount checks
  - Pattern: `task_tripletex/testing/scoring.py:120-135` — GET-based verification is efficiency-safe relative to write/4xx scoring
  - Known issue: Previous logs showed `invoice_with_payment` stopping before `PUT /invoice/{id}/:payment`

  **Acceptance Criteria**:
  - [ ] A new test suite covers post-condition success and post-condition failure for at least invoice, invoice+payment, and project families.
  - [ ] A targeted test proves the controller does not mark success if the write path completed but post-condition verification failed.
  - [ ] `/logs` includes explicit post-condition start/finish entries with route family and result.

  **QA Scenarios**:
  ```
  Scenario: Controller only exits after terminal state is verified
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_postconditions.py -k "invoice or payment"`.
    Expected: Route success occurs only after the expected GET-based check passes.
    Evidence: .sisyphus/evidence/task-5-postconditions.json

  Scenario: False-success path is downgraded when terminal state is missing
    Tool: Bash
    Steps: Run a targeted pytest node that simulates invoice creation without payment completion for the payment family.
    Expected: Controller records internal `incomplete` or `fail`, not `success`.
    Evidence: .sisyphus/evidence/task-5-postconditions-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add postcondition verification` | Files: `task_tripletex/postconditions.py`, `task_tripletex/controller.py`, targeted tests

- [ ] 6. Extract shared Tripletex business primitives and endpoint guardrails

  **What to do**: Create a shared primitive layer used by deterministic handlers and, where safe, the focused fallback. The primitive layer must include the exact known-safe building blocks needed by the known fixtures: `create_customer`, `create_product`, `find_or_create_department_for_employee`, `create_employee`, `ensure_bank_account_1920`, `create_order_with_inline_lines`, `create_invoice_from_order`, `resolve_invoice_payment_type`, `register_invoice_payment`, `ensure_project_manager_entitlements`, `create_project`, `create_travel_expense`, and `create_manual_voucher`. Add per-primitive endpoint allowlists and payload shaping so deterministic handlers never call guessed endpoints.
  **Must NOT do**: Do not hide essential IDs/versions needed for follow-up operations. Do not duplicate the HTTP client. Do not generalize into an unbounded workflow DSL.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Shared primitives are the only sane way to keep C-lite as one controller instead of two architectures.
  - Skills: `[]` — No extra skill required.
  - Omitted: `quality-check` — Atomic primitive extraction should be verified by targeted runtime tests first.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 7, 8, 9, 10, 11, 12 | Blocked By: 1, 2, 5

  **References**:
  - Pattern: `task_tripletex/client.py:61-120` — canonical Tripletex request execution surface
  - Pattern: `task_tripletex/agent.py:128-551` — endpoint-specific payload rules and known quirks to codify as primitives
  - Pattern: `task_tripletex/testing/fixtures/create_invoice_basic.json:1-110` — invoice flow shape
  - Pattern: `task_tripletex/testing/fixtures/invoice_with_payment.json:1-108` — payment flow shape
  - Pattern: `task_tripletex/testing/fixtures/create_project_basic.json:1-78` — project flow shape
  - Pattern: `task_tripletex/testing/fixtures/create_journal_entry_basic.json:1-79` — voucher flow shape
  - Test: `tests/task_tripletex_testing/test_request_context_runtime.py:322-989` — retry/cache expectations that primitives must not violate indirectly

  **Acceptance Criteria**:
  - [ ] New primitive-level tests cover successful payload shaping for each shared building block used by the deterministic routes.
  - [ ] Payment-type resolution primitive explicitly uses `/invoice/paymentType`, not guessed alternatives.
  - [ ] The primitive layer is imported by deterministic handlers; handler code does not re-encode endpoint-specific quirks inline.

  **QA Scenarios**:
  ```
  Scenario: Shared primitives emit known-safe endpoint paths and payloads
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_primitives.py` after adding primitive payload and endpoint-path cases.
    Expected: Each primitive builds the intended endpoint path/payload combination, including paymentType and entitlement order.
    Evidence: .sisyphus/evidence/task-6-primitives.json

  Scenario: Payment primitive blocks wrong endpoint family
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_primitives.py -k "payment type"`.
    Expected: Only `/invoice/paymentType` is allowed for invoice payment type discovery.
    Evidence: .sisyphus/evidence/task-6-primitives-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): extract shared tripletex primitives` | Files: `task_tripletex/primitives.py`, targeted tests, small call-site updates

- [ ] 7. Add deterministic Tier 1 handlers for customer, product, and employee-admin

  **What to do**: Implement deterministic handlers for the three simplest known Tier-1 families using the new primitive layer: `create_customer`, `create_product`, and `create_employee_admin`. The employee-admin flow must continue using the existing safe pattern of department resolution followed by employee creation with `userType: 2`. These handlers should become the highest-confidence deterministic routes and serve as the baseline rollout for the controller.
  **Must NOT do**: Do not add unnecessary preflight GETs for customer or product creation. Do not regress the already-good employee-admin performance. Do not invoke the focused or legacy path for these cases when classification confidence is high.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Each Tier-1 handler is narrow once the controller and primitives exist.
  - Skills: `[]` — Existing tests are enough.
  - Omitted: `quality-check` — VM and pytest verification carry more value here.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 15 | Blocked By: 2, 5, 6

  **References**:
  - Pattern: `task_tripletex/testing/fixtures/create_customer.json:1-75` — customer acceptance contract
  - Pattern: `task_tripletex/testing/fixtures/create_product.json:1-74` — product acceptance contract
  - Pattern: `task_tripletex/testing/fixtures/create_employee_admin.json:1-76` — employee-admin acceptance contract
  - Pattern: `AGENTS.md:18-22` — current verified employee-admin performance baseline
  - Test: `tests/task_tripletex_testing/test_fixture_loader.py:165-220` — Tier-1 fixture invariants
  - Test: `tests/task_tripletex_testing/test_framework_integration.py:251-297` — employee-admin optimal-score regression shape

  **Acceptance Criteria**:
  - [ ] `create_customer`, `create_product`, and `create_employee_admin` each reach correctness `1.0` via packaged-case CLI on the VM.
  - [ ] `create_product` stays within its fixture efficiency rule of one write and zero 4xx errors.
  - [ ] `create_employee_admin` remains at or better than the known-good baseline of one write and zero 4xx errors in proxy metrics.

  **QA Scenarios**:
  ```
  Scenario: Tier-1 deterministic handlers pass fixture and proxy regression
    Tool: Bash
    Steps: Run the three packaged-case CLI commands on the VM for `create_customer`, `create_product`, and `create_employee_admin`.
    Expected: Each case reports correctness 1.0, valid contract/proxy, and expected write/error metrics.
    Evidence: .sisyphus/evidence/task-7-tier1-handlers.json

  Scenario: Employee-admin handler does not regress into extra writes
    Tool: Bash
    Steps: Re-run only `create_employee_admin` and inspect `proxy_metrics.write_calls` and `proxy_metrics.client_error_calls` in the JSON result.
    Expected: One write call, zero 4xx errors.
    Evidence: .sisyphus/evidence/task-7-tier1-handlers-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add tier1 deterministic handlers` | Files: `task_tripletex/deterministic.py`, `task_tripletex/controller.py`, targeted tests

- [ ] 8. Add deterministic invoice and invoice-with-payment handlers

  **What to do**: Implement deterministic handlers for `create_invoice_basic` and `invoice_with_payment`. These handlers must use inline order lines, ensure bank account configuration on ledger account 1920 before invoice creation, create the invoice, and for the payment family explicitly resolve payment type then call `PUT /invoice/{id}/:payment`. Post-condition verification must confirm invoice/order linkage for the plain invoice route and `isPaid == true` for the payment route.
  **Must NOT do**: Do not call `/order/orderline` after inline lines were already provided. Do not stop after invoice creation on the payment family. Do not guess payment endpoints outside `/invoice/paymentType` and `PUT /invoice/{id}/:payment`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: This is the highest-value scoring fix and includes multiple endpoint quirks.
  - Skills: `[]` — No extra skill required.
  - Omitted: `quality-check` — Use fixture, proxy, and log evidence instead.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 15 | Blocked By: 2, 5, 6

  **References**:
  - Pattern: `task_tripletex/testing/fixtures/create_invoice_basic.json:1-110` — plain invoice acceptance contract
  - Pattern: `task_tripletex/testing/fixtures/invoice_with_payment.json:1-108` — paid-invoice acceptance contract
  - Pattern: `task_tripletex/agent.py:239-305` — order/invoice/payment endpoint rules and bank-account requirement
  - Pattern: `task_tripletex/agent.py:498-515` — documented intended task flows
  - Test: `tests/task_tripletex_testing/test_fixture_loader.py:302-349` — date-ranged invoice/payment fixture expectations
  - Docs: `docs/scoring.md:37-55` — zero 4xx matters especially because `invoice_with_payment` has `max_4xx_errors = 0`
  - Known issue: previous logs showed wrong endpoint guesses and missing payment registration

  **Acceptance Criteria**:
  - [ ] `create_invoice_basic` reaches correctness `1.0`, uses at most 3 writes, and records zero 4xx errors on the VM.
  - [ ] `invoice_with_payment` reaches correctness `1.0`, uses at most 4 writes, and records zero 4xx errors on the VM.
  - [ ] `/logs` for `invoice_with_payment` show payment-type resolution and post-condition verification before route success.

  **QA Scenarios**:
  ```
  Scenario: Deterministic invoice families complete end-to-end
    Tool: Bash
    Steps: Run the packaged-case CLI for `create_invoice_basic` and `invoice_with_payment` on the VM.
    Expected: Both cases pass with correctness 1.0, valid proxy use, and no 4xx write errors.
    Evidence: .sisyphus/evidence/task-8-invoice-handlers.json

  Scenario: Payment workflow does not terminate after invoice creation alone
    Tool: Bash
    Steps: Run a targeted pytest or local integration test that simulates invoice creation without payment completion for the payment family.
    Expected: Post-condition rejects the route as incomplete/fail rather than success.
    Evidence: .sisyphus/evidence/task-8-invoice-handlers-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add deterministic invoice handlers` | Files: `task_tripletex/deterministic.py`, `task_tripletex/primitives.py`, targeted tests

- [ ] 9. Add deterministic project handler with entitlement preflight

  **What to do**: Implement the deterministic `create_project_basic` handler. It must ensure the target project manager exists or is created through the established employee flow, then proactively grant entitlement 45 followed by 10 using the company customer reference discovered from `/employee/entitlement`, and only then create the project. Post-condition verification must confirm project number, start date, and linked manager email.
  **Must NOT do**: Do not attempt `POST /project` before entitlements are in place. Do not reverse the entitlement order. Do not route a project prompt into the generic invoice or employee-admin flows.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Entitlement dependency ordering is brittle and easy to regress.
  - Skills: `[]` — No extra skill required.
  - Omitted: `quality-check` — Focus on functional regression against the real fixture.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 15 | Blocked By: 2, 5, 6

  **References**:
  - Pattern: `task_tripletex/testing/fixtures/create_project_basic.json:1-78` — project acceptance contract
  - Pattern: `task_tripletex/agent.py:307-365` — entitlement dependency chain and company-customer lookup
  - Pattern: `task_tripletex/agent.py:500-500` — documented project task flow
  - Test: `tests/task_tripletex_testing/test_fixture_loader.py:128-163` — project fixture invariants
  - Docs: `docs/overview.md:48-54` — projects are a core task category, not optional edge scope

  **Acceptance Criteria**:
  - [ ] `create_project_basic` reaches correctness `1.0` on the VM.
  - [ ] The handler grants entitlement 45 before 10 and avoids project-creation 422 retries for missing PM access.
  - [ ] Post-condition verifies manager email and project number before success.

  **QA Scenarios**:
  ```
  Scenario: Deterministic project flow succeeds with proactive entitlements
    Tool: Bash
    Steps: Run the packaged-case CLI for `create_project_basic` on the VM.
    Expected: Correctness 1.0 with expected manager linkage and no entitlement-order 422s.
    Evidence: .sisyphus/evidence/task-9-project-handler.json

  Scenario: Wrong entitlement order is blocked by tests
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_primitives.py -k "entitlement order"`.
    Expected: Test fails if order is reversed; passing path enforces 45 then 10.
    Evidence: .sisyphus/evidence/task-9-project-handler-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add deterministic project handler` | Files: `task_tripletex/deterministic.py`, `task_tripletex/primitives.py`, targeted tests

- [ ] 10. Add deterministic travel-expense handler

  **What to do**: Implement the deterministic `create_travel_expense_basic` handler. It must ensure the referenced employee exists or is created via the shared employee flow, then create the travel expense with the fixture title and employee reference, and verify the employee linkage afterward. Reuse the route and post-condition telemetry introduced earlier.
  **Must NOT do**: Do not over-engineer sub-objects or voucher creation for the baseline travel-expense fixture. Do not treat travel expense as a generic voucher or attachment task.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Narrow deterministic route once primitives and controller exist.
  - Skills: `[]` — No extra skill required.
  - Omitted: `quality-check` — Regression commands give sufficient proof.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 15 | Blocked By: 2, 5, 6

  **References**:
  - Pattern: `task_tripletex/testing/fixtures/create_travel_expense_basic.json:1-63` — travel-expense acceptance contract
  - Pattern: `task_tripletex/agent.py:373-386` — travel-expense endpoint guidance
  - Test: `tests/task_tripletex_testing/test_fixture_loader.py:153-162` — travel-expense fixture invariants
  - Docs: `docs/overview.md:48-49` — travel expenses are a first-class task category

  **Acceptance Criteria**:
  - [ ] `create_travel_expense_basic` reaches correctness `1.0` on the VM.
  - [ ] The handler uses at most 2 writes and at most 1 4xx error budget, ideally zero.
  - [ ] Post-condition verifies title and employee email before success.

  **QA Scenarios**:
  ```
  Scenario: Deterministic travel-expense flow passes fixture
    Tool: Bash
    Steps: Run the packaged-case CLI for `create_travel_expense_basic` on the VM.
    Expected: Correctness 1.0 with linked employee email and valid proxy metrics.
    Evidence: .sisyphus/evidence/task-10-travel-handler.json

  Scenario: Travel handler does not misroute to voucher flow
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_routing.py -k "travel expense"`.
    Expected: Only travel-expense primitives are used.
    Evidence: .sisyphus/evidence/task-10-travel-handler-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add deterministic travel handler` | Files: `task_tripletex/deterministic.py`, targeted tests

- [ ] 11. Add deterministic journal-entry handler

  **What to do**: Implement the deterministic `create_journal_entry_basic` handler. Resolve the referenced ledger accounts, create a balanced voucher using `amountGross`/`amountGrossCurrency`, assign `row` starting at 1, and include required VAT information when account metadata demands it. Post-condition verification must confirm voucher description/date and the expected debit/credit amounts in the returned postings.
  **Must NOT do**: Do not use `amount`. Do not omit posting rows. Do not assume VAT is irrelevant for all accounts. Do not create multiple vouchers for the single baseline journal-entry fixture.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Voucher payload correctness is sensitive and 422-prone.
  - Skills: `[]` — No extra skill required.
  - Omitted: `quality-check` — Functional coverage is the meaningful gate here.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 15 | Blocked By: 2, 5, 6

  **References**:
  - Pattern: `task_tripletex/testing/fixtures/create_journal_entry_basic.json:1-79` — voucher acceptance contract
  - Pattern: `task_tripletex/agent.py:388-423` — ledger/voucher payload rules, row numbering, and VAT caveats
  - Test: `tests/task_tripletex_testing/test_fixture_loader.py:144-151` — debit/credit fixture invariants
  - Docs: `AGENTS.md:80-87` — known voucher quirks and GET date-range behavior

  **Acceptance Criteria**:
  - [ ] `create_journal_entry_basic` reaches correctness `1.0` on the VM.
  - [ ] The handler uses one voucher write and does not incur avoidable 4xx errors.
  - [ ] Post-condition verifies the debit and credit posting amounts exactly as the fixture expects.

  **QA Scenarios**:
  ```
  Scenario: Deterministic journal-entry flow passes fixture
    Tool: Bash
    Steps: Run the packaged-case CLI for `create_journal_entry_basic` on the VM.
    Expected: Correctness 1.0 and matching debit/credit posting values in verification snapshots.
    Evidence: .sisyphus/evidence/task-11-journal-handler.json

  Scenario: Voucher payload guardrails catch row/amount mistakes
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_primitives.py -k "voucher payload"`.
    Expected: Tests enforce correct voucher payload structure.
    Evidence: .sisyphus/evidence/task-11-journal-handler-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add deterministic journal handler` | Files: `task_tripletex/deterministic.py`, `task_tripletex/primitives.py`, targeted tests

- [ ] 12. Add focused fallback families for supplier invoice, bank reconciliation, and unknown routes

  **What to do**: Implement focused fallback route packs for `supplier_invoice_basic`, `bank_reconciliation_file`, and `unknown`. The supplier-invoice pack must consume PDF context plus prompt anchors and only expose supplier/supplierInvoice/document-related endpoint families. The bank-reconciliation pack must consume normalized CSV text plus prompt anchors and only expose bank/voucher/reconciliation endpoint families. The unknown route must first use a read-only focused planning pass with bounded discovery on its selected endpoint pack; if the plan remains low-confidence before the first write, downgrade to legacy monolith fallback, otherwise continue within the chosen focused pack. No route may transfer control after its first write.
  **Must NOT do**: Do not let the focused fallback issue arbitrary writes to any Tripletex path. Do not send the full monolithic prompt into these routes. Do not chain into legacy fallback after a focused route already performed a write.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: This is the file-driven and unknown-task safety net for the 20 unmodeled task types.
  - Skills: `[]` — No extra skill required.
  - Omitted: `quality-check` — Proxy/fixture evidence is the real gate here.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 13, 14, 15 | Blocked By: 2, 3, 4, 5, 6

  **References**:
  - Pattern: `task_tripletex/testing/fixtures/supplier_invoice_basic.json:1-68` — PDF-backed supplier invoice acceptance contract
  - Pattern: `task_tripletex/testing/fixtures/bank_reconciliation_file.json:1-59` — CSV-backed reconciliation acceptance contract
  - Pattern: `task_tripletex/agent.py:178-188` — supplier-invoice guidance
  - Pattern: `task_tripletex/agent.py:424-469` — bank reconciliation/import guidance
  - Test: `tests/task_tripletex_testing/test_framework_integration.py:332-376` — file forwarding test seam
  - Docs: `docs/overview.md:19-32` — fresh sandbox + files + broad task coverage make fallback necessary
  - Docs: `docs/scoring.md:57-80` — best score per task means conservative unknown-task behavior is acceptable while expanding coverage over time

  **Acceptance Criteria**:
  - [ ] `supplier_invoice_basic` runs through the new controller and reaches correctness `1.0` on the VM without contract/proxy violations.
  - [ ] `bank_reconciliation_file` accepts the CSV attachment through the controller and reaches correctness `1.0` on the VM without MIME rejection or disallowed endpoint speculation.
  - [ ] Unknown-route tests prove bounded discovery reads and no speculative disallowed writes.

  **QA Scenarios**:
  ```
  Scenario: File-driven focused routes execute within endpoint guardrails
    Tool: Bash
    Steps: Run the packaged-case CLI for `supplier_invoice_basic` and `bank_reconciliation_file` on the VM, then run `pytest tests/task_tripletex_testing/test_focused_agent.py -k "supplier or bank"`.
    Expected: No MIME rejection for CSV/PDF, no contract/proxy violations, and route telemetry shows focused fallback usage.
    Evidence: .sisyphus/evidence/task-12-file-fallbacks.json

  Scenario: Unknown route remains conservative under ambiguity
    Tool: Bash
    Steps: Run targeted fallback tests with an unclassified prompt that tempts a guessed write endpoint.
    Expected: Only allowed discovery reads occur, or the controller downgrades to legacy fallback before any write.
    Evidence: .sisyphus/evidence/task-12-file-fallbacks-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): add focused fallback route packs` | Files: `task_tripletex/focused_agent.py`, prompt slices, controller/routing updates, targeted tests

- [ ] 13. Wire the service to the new controller and expand route telemetry

  **What to do**: Replace the direct `_default_plan_executor()` call path in `task_tripletex/service.py` with the new controller. Add log events for `route_decision`, `route_execution_started`, `route_execution_finished`, `postcondition_check_started`, `postcondition_check_finished`, and explicit `fallback_transition_blocked` when a write has already started. Preserve sanitization rules and the current `/logs` schema shape while appending these new events.
  **Must NOT do**: Do not change the `/logs` response envelope. Do not leak session tokens or raw file contents in the new telemetry. Do not remove the existing request-context, retry, cache, and budget events.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Once controller and routes exist, this is narrow service/log wiring.
  - Skills: `[]` — No extra skill required.
  - Omitted: `quality-check` — Contract and log assertions are the important checks.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 14, 15 | Blocked By: 1, 3, 5, 12

  **References**:
  - Pattern: `task_tripletex/service.py:23-130` — current request lifecycle and task logging
  - Pattern: `task_tripletex/task_log.py:19-151` — sanitization and snapshot behavior to preserve
  - Test: `tests/task_tripletex_testing/test_request_context_runtime.py:300-320` — existing `/logs` trace assertions
  - Test: `tests/task_tripletex_testing/test_framework_integration.py:470-533` — sensitive log-redaction expectations

  **Acceptance Criteria**:
  - [ ] Integration tests confirm the service uses the controller and logs new route events.
  - [ ] New route telemetry appears in `/logs` without exposing session tokens or raw `content_base64`.
  - [ ] Existing request-context and file-policy log assertions continue to pass.

  **QA Scenarios**:
  ```
  Scenario: /logs exposes new route telemetry without schema regression
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_framework_integration.py -k "logs or controller"`.
    Expected: Existing log envelope remains valid; new route events are present and sanitized.
    Evidence: .sisyphus/evidence/task-13-service-wiring.json

  Scenario: Fallback transition is blocked after writes begin
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_framework_integration.py -k "fallback_transition"` after adding the blocked-transition case.
    Expected: A `fallback_transition_blocked` event is logged and no second write-capable route is started.
    Evidence: .sisyphus/evidence/task-13-service-wiring-error.json
  ```

  **Commit**: YES | Message: `feat(tripx): wire service through controller` | Files: `task_tripletex/service.py`, `task_tripletex/controller.py`, `task_tripletex/task_log.py` (log events only), targeted tests

- [ ] 14. Expand pytest and packaged-case regression coverage for the C-lite migration

  **What to do**: Add or update tests so the repo has explicit regression coverage for controller routing, focused-fallback guardrails, post-condition semantics, route telemetry, and all 10 packaged fixtures. Extend framework/CLI tests only where necessary; prefer augmenting the existing suites under `tests/task_tripletex_testing/`. Add a documented regression command block to the plan evidence or test output expectations used by later executors.
  **Must NOT do**: Do not create an entirely separate test framework. Do not weaken fixture or scoring assertions to make the migration easier. Do not remove the existing contract-failure coverage.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: This is the confidence gate for a risky architectural migration.
  - Skills: `[]` — Existing repo test infrastructure is enough.
  - Omitted: `quality-check` — The test suite itself is the quality gate.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 15 | Blocked By: 1, 2, 3, 4, 5, 12, 13

  **References**:
  - Test: `tests/task_tripletex_testing/test_framework_integration.py:1-533` — end-to-end framework seams already present
  - Test: `tests/task_tripletex_testing/test_request_context_runtime.py:1-989` — runtime retry/cache/budget/logging assertions
  - Test: `tests/task_tripletex_testing/test_fixture_loader.py:1-390` — all 10 packaged fixtures already anchored here
  - Test: `tests/task_tripletex_testing/test_cli.py:23-49` — CLI smoke surface
  - Test: `tests/task_tripletex_testing/test_scoring.py` — scoring invariants that must remain untouched
  - Test: `tests/task_tripletex_testing/test_verifier.py` — selector semantics mirrored by post-condition checks

  **Acceptance Criteria**:
  - [ ] The full targeted pytest command in Definition of Done passes locally.
  - [ ] New tests explicitly cover controller route selection, blocked speculative writes, post-condition non-success handling, and route telemetry.
  - [ ] No existing scoring/verifier/endpoint contract tests are deleted or weakened.

  **QA Scenarios**:
  ```
  Scenario: Full local regression suite passes after controller migration
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_request_context_runtime.py tests/task_tripletex_testing/test_framework_integration.py tests/task_tripletex_testing/test_fixture_loader.py tests/task_tripletex_testing/test_cli.py tests/task_tripletex_testing/test_scoring.py tests/task_tripletex_testing/test_verifier.py`.
    Expected: Entire targeted suite passes.
    Evidence: .sisyphus/evidence/task-14-regression.json

  Scenario: Regression suite fails if speculative write guardrail is removed
    Tool: Bash
    Steps: Run `pytest tests/task_tripletex_testing/test_focused_agent.py -k "speculative write guardrail"` on a temporary failing branch or with an intentionally broken local patch during development.
    Expected: Test catches the missing guardrail and fails explicitly.
    Evidence: .sisyphus/evidence/task-14-regression-error.json
  ```

  **Commit**: YES | Message: `test(tripx): expand c-lite regression coverage` | Files: `tests/task_tripletex_testing/*.py`

- [ ] 15. Run VM packaged-case wave, deploy to Cloud Run, and capture verification evidence

  **What to do**: Execute the known-fixture validation wave on the GCP VM using the packaged-case CLI, collect evidence for each deterministic family and the two focused file-driven families, then deploy the validated code to Cloud Run using the existing project command from `AGENTS.md`. After deployment, run one text-only smoke prompt and one file-backed smoke prompt against the public service, then fetch `/logs` to confirm controller route telemetry appears in the deployed trace.
  **Must NOT do**: Do not skip the VM proxy wave and jump straight to Cloud Run. Do not deploy code that fails the targeted pytest suite. Do not use Cloud Run for the local proxy-based fixture validations because the recorder cannot be reached from Cloud Run.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: This combines deployment, VM orchestration, fixture evaluation, and evidence capture.
  - Skills: `[]` — Deployment uses repo-local documented commands.
  - Omitted: `quality-check` — The gating command here is the regression suite plus VM/Cloud Run evidence.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Final Verification Wave | Blocked By: 7, 8, 9, 10, 11, 12, 13, 14

  **References**:
  - Pattern: `AGENTS.md:39-58` — authoritative VM packaged-case workflow
  - Pattern: `AGENTS.md:24-30` — authoritative Cloud Run deployment command
  - Pattern: `AGENTS.md:61-70` — Cloud Run smoke and `/logs` commands
  - Test: `task_tripletex/testing/cli.py:182-216` — packaged-case evaluation flow
  - Test: `task_tripletex/testing/scoring.py:69-147` — result interpretation for correctness and efficiency bonus
  - Constraint: `AGENTS.md:88-90` — SSH is flaky, so break VM commands into short steps

  **Acceptance Criteria**:
  - [ ] VM packaged-case runs for `create_customer`, `create_product`, `create_employee_admin`, `create_invoice_basic`, `invoice_with_payment`, `create_project_basic`, `create_travel_expense_basic`, `create_journal_entry_basic`, `supplier_invoice_basic`, and `bank_reconciliation_file` all report correctness `1.0` with valid contract/proxy metrics.
  - [ ] Cloud Run smoke returns HTTP 200 with exact JSON contract and `/logs` includes route telemetry from the deployed revision.
  - [ ] Evidence files are saved for each run under `.sisyphus/evidence/` with enough JSON/log output to inspect correctness, proxy metrics, and score.

  **QA Scenarios**:
  ```
  Scenario: VM packaged-case validation wave confirms known-task behavior
    Tool: Bash
    Steps: Sync `task_tripletex/` to the VM, restart uvicorn, and run the packaged-case CLI once per known case using the command pattern from `AGENTS.md:39-58`.
    Expected: The deterministic families pass with correctness 1.0; file-driven families produce valid controller traces and no contract/proxy violations.
    Evidence: .sisyphus/evidence/task-15-vm-cloudrun.json

  Scenario: Deployed Cloud Run revision preserves contract and logs controller telemetry
    Tool: Bash
    Steps: Run the documented `gcloud builds submit ... && gcloud run deploy ...` command, send one smoke `curl` to `/solve`, then fetch `/logs`.
    Expected: HTTP 200 + exact JSON body from `/solve`; `/logs` shows route decision and post-condition events from the deployed revision.
    Evidence: .sisyphus/evidence/task-15-vm-cloudrun-error.json
  ```

  **Commit**: NO | Message: `n/a` | Files: deployment/evidence only

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Create one atomic commit per numbered task from 1 through 14; Task 15 produces evidence and deployment state, not a source commit.
- Commit ordering must follow the dependency matrix exactly: controller contract first, handler families second, service wiring and regression expansion last.
- Do not amend old commits during this migration unless a pre-commit hook modifies files in the commit that was just created.

## Success Criteria
- **Minimum**: the 8 deterministic known-fixture routes reach correctness `1.0`, `create_employee_admin` keeps its optimal 1 write / 0 4xx behavior, and the remaining 2 known file-driven fixtures run through the new controller without contract or proxy violations.
- **Target**: all 10 known packaged fixtures pass on the VM, `invoice_with_payment` stops using wrong payment endpoints, and `/logs` clearly shows route decision, guardrail decisions, and post-condition results.
- **Stretch**: Cloud Run submission improves rolling leaderboard score without requiring a second architecture rewrite.
