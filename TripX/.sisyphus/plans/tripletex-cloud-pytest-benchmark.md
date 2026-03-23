# TripX Cloud Pytest Benchmark Plan

## TL;DR
> **Summary**: Shift the primary validation strategy from local proxy recording to `pytest`-driven E2E tests executing against the deployed Cloud Run instance, managed by `uv`. Construct a SQLite+JSON benchmark database seeded from current failure logs to measure ongoing performance.
> **Deliverables**:
> - `pyproject.toml` establishing `uv` as the canonical entrypoint
> - `AGENTS.md` update demoting VM/proxy to fallback and defining the cloud-pytest strategy
> - `task_tripletex/testing/benchmark.py` for SQLite ingestion of JSON run artifacts
> - Cloud-targeted `pytest` suite running serialized mutating cases against `/solve` and `/logs`
> - Initial benchmark corpus built from existing `.sisyphus/evidence`
> **Effort**: Medium
> **Parallel**: YES - 2 waves
> **Critical Path**: 1 → 2 → 3 → 4 → 5

## Context
### Original Request
- Change strategy to use `uv` and `pytest` for elaborate test-coverage that runs against our cloud instance.
- Update `AGENTS.md` for this strategy.
- Proceed to build a valid test database of the current log failures.
- Goal: Have a good benchmark to go off for performance.

### Interview Summary
- `uv` is already used via Dockerfile, and `pytest` is used in `tests/`, but `pytest` is missing from `requirements.txt` and there is no `pyproject.toml`.
- Current cloud validation relies on manual curl + ad-hoc `/logs` inspection (`AGENTS.md`). The automated test CLI relies on a local proxy, which Cloud Run cannot reach.
- The repository already contains rich seed data for a benchmark: `.sisyphus/evidence/task-14-vm-regression/` and `task-15-cloud-run-smoke/`.
- User confirmed choices: SQLite+JSON for the benchmark DB, public+staging target support, serialized mutating tests, `pyproject.toml` adoption, and treating the existing VM proxy workflow as a local fallback.

### Metis Review (gaps addressed)
- Ensure existing integration tests aren't broken when transitioning to `pyproject.toml`.
- Avoid concurrent mutating test executions that pollute the shared Tripletex sandbox by enforcing strict serialization.
- Separate product bugs from test/fixture bugs in the benchmark taxonomy to maintain trust in the scores.
- Rely exclusively on `/solve` HTTP contracts and `/logs` traces for cloud validation; do not require a separate remote-control endpoint.

## Work Objectives
### Core Objective
Deploy a robust, `uv`-managed `pytest` suite that hits the live Cloud Run endpoint and automatically archives trace artifacts into a queryable SQLite benchmark database for performance tracking.

### Deliverables
- `pyproject.toml` configured for `uv`.
- Refactored documentation in `AGENTS.md`.
- Benchmark DB ingestion script/schema using SQLite.
- `tests/cloud/` suite targeting deployed endpoints.
- Seeded database tracking historical run evidence.

### Definition of Done (verifiable conditions with commands)
- `uv sync` correctly installs the application and test dependencies.
- `uv run pytest tests/cloud/` executes E2E runs against the live Cloud Run URL (via environment variables).
- The benchmark database is created (`benchmark.db`) and queryable, containing records from previous evidence.
- `AGENTS.md` reflects the new primary test strategy.

### Must Have
- Adopt `pyproject.toml` while keeping `requirements.txt` generation if required by Docker/deployment.
- Ensure all mutating tests run strictly sequentially to prevent Tripletex API sandbox collisions.
- Support both public Cloud Run and staging URLs via standard environment variables.
- Preserve the existing `task_tripletex/testing/cli.py` proxy logic strictly as a local fallback tool.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Must NOT delete the local proxy code; it remains valuable for local optimization.
- Must NOT use a hosted database (Postgres/Cloud SQL) for the benchmark; SQLite + JSON artifacts are required.
- Must NOT execute mutating tests concurrently.
- Must NOT modify the `task_tripletex/agent.py` logic; this plan is strictly for testing infrastructure.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: Verify `uv` environment isolation by running `pytest` in a clean environment. Query the generated SQLite DB to ensure rows exist and map correctly to the JSON artifacts.
- QA policy: Ensure one mutating cloud test and one non-mutating cloud test exist and pass.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.json`

## Execution Strategy
### Parallel Execution Waves
Wave 1: Tasks 1-3 — Base tooling (`pyproject.toml`), docs (`AGENTS.md`), and benchmark DB schema/seeding.
Wave 2: Tasks 4-5 — Cloud `pytest` suite implementation and CI wiring.

## TODOs
> Implementation + Test = ONE task. Every task includes agent profile, references, acceptance criteria, QA scenarios, and commit guidance.

- [ ] 1. Migrate Python Tooling to `uv` and `pyproject.toml`

  **What to do**: Create `pyproject.toml` as the canonical source for both application runtime dependencies and developer/test dependencies (like `pytest`, `pytest-asyncio`). Extract the current `requirements.txt` lists into the `[project]` and `[dependency-groups]` sections. Ensure `uv sync` correctly builds a `.venv` with all tools. Optionally add a `Makefile` or `pyproject.toml` scripts for standard invocation (`uv run pytest`). Keep the `Dockerfile` updated to use the new file.
  **Must NOT do**: Do not drop any required runtime dependencies. Do not break the Cloud Run container build.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Infrastructure migration affecting container and test execution.
  - Skills: `[]`
  
  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4 | Blocked By: none
  
  **References**:
  - `requirements.txt`, `Dockerfile`
  
  **Acceptance Criteria**:
  - [ ] `uv sync` completes successfully and `.venv` contains `pytest`.
  - [ ] `uv run pytest tests/` executes successfully without import errors.
  
  **QA Scenarios**:
  ```
  Scenario: uv runs pytest tests successfully
    Tool: Bash
    Steps: uv sync && uv run pytest tests/
    Expected: Tests pass, proving test dependencies are correctly loaded.
    Evidence: .sisyphus/evidence/task-1-pyproject-migration.txt
  ```

  **Commit**: YES | Message: `chore(tripx): adopt uv and pyproject.toml` | Files: `pyproject.toml`, `requirements.txt`, `Dockerfile`

- [ ] 2. Rewrite `AGENTS.md` for the New Cloud-Targeted Strategy

  **What to do**: Update `AGENTS.md` testing and architecture sections to document the `pytest`-driven, Cloud Run E2E evaluation as the primary verification strategy. Relegate the local `cli.py` and proxy recorder workflow to a "Local Optimization Fallback" section. Provide explicit `uv run pytest ...` invocation commands targeting public and staging URLs. Explain the benchmark SQLite+JSON strategy.
  **Must NOT do**: Do not delete the proxy documentation entirely.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: Documentation task ensuring developer clarity on the new architecture.
  - Skills: `[]`
  
  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: none | Blocked By: none
  
  **References**:
  - `AGENTS.md:37-75` (Testing sections)
  
  **Acceptance Criteria**:
  - [ ] `AGENTS.md` clearly states `uv run pytest tests/cloud/` is the primary way to test.
  
  **QA Scenarios**:
  ```
  Scenario: AGENTS.md reflects the new testing workflow
    Tool: Bash
    Steps: grep "uv run pytest" AGENTS.md
    Expected: Output shows the newly added command examples.
    Evidence: .sisyphus/evidence/task-2-docs.txt
  ```

  **Commit**: YES | Message: `docs(tripx): update agents.md with pytest cloud strategy` | Files: `AGENTS.md`

- [ ] 3. Build the Benchmark SQLite Database Ingestion Pipeline

  **What to do**: Create a lightweight script/module (e.g., `task_tripletex/testing/benchmark.py`) that initializes a SQLite DB (`benchmark.db`) with tables corresponding to the key test metrics: `run_id`, `case_id`, `status`, `total_score`, `write_efficiency`, `error_efficiency`, `timestamp`, and `target_env`. Write an ingestion function that parses JSON artifacts from `.sisyphus/evidence/task-14-vm-regression` and `task-15-cloud-run-smoke` and inserts them as historical seed records. Ensure the JSON blobs themselves are linked or stored alongside.
  **Must NOT do**: Do not over-engineer with ORMs (SQLAlchemy, etc.) unless strictly necessary; raw SQLite/`sqlite3` or simple `pydantic` mapping is preferred. Do not delete the JSON evidence.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: Data modeling for the test benchmark requires careful parsing of historical proxy metrics.
  - Skills: `[]`
  
  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5 | Blocked By: none
  
  **References**:
  - `.sisyphus/evidence/task-14-vm-regression/*.json`
  - `.sisyphus/evidence/task-15-cloud-run-smoke/*.json`
  - `task_tripletex/testing/models.py` (EvaluationResult / ScoreResult schemas)
  
  **Acceptance Criteria**:
  - [ ] Executing the ingestion script populates a new SQLite DB.
  - [ ] A sample `SELECT` query retrieves the correct scores and case classifications.
  
  **QA Scenarios**:
  ```
  Scenario: Database ingestion maps historical JSON correctly
    Tool: Bash
    Steps: uv run python -m task_tripletex.testing.benchmark_ingest && sqlite3 benchmark.db "SELECT count(*) FROM benchmark_runs;"
    Expected: Count equals the number of historical evidence JSON files.
    Evidence: .sisyphus/evidence/task-3-benchmark-db.txt
  ```

  **Commit**: YES | Message: `test(tripx): add sqlite benchmark ingestion` | Files: `task_tripletex/testing/benchmark.py`, `tests/task_tripletex_testing/test_benchmark.py`

- [ ] 4. Create the `tests/cloud/` Pytest Suite with Serialized Mutating Tests

  **What to do**: Add a new pytest suite in `tests/cloud/` that executes E2E tasks strictly against the deployed Cloud Run instance (via `--solve-url` or `SOLVE_URL` env var). The suite should reuse the packaged cases. It must use a pytest marker (e.g., `@pytest.mark.cloud`) and ensure mutating cases run sequentially to avoid Tripletex API sandbox corruption. The test should call the actual `POST /solve`, assert the `200` response contract, then retrieve the `/logs` to verify trace events and correctness.
  **Must NOT do**: Do not run the tests against local proxies by default in this folder. Do not allow `pytest-xdist` to parallelize mutating cases.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Orchestrating the Cloud Run HTTP assertions and logging trace correlations.
  - Skills: `[]`
  
  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 5 | Blocked By: 1
  
  **References**:
  - `task_tripletex/testing/cli.py` (Existing test invocation logic)
  - `tests/task_tripletex_testing/test_framework_integration.py`
  
  **Acceptance Criteria**:
  - [ ] Running `uv run pytest tests/cloud/` succeeds against a specified `--solve-url`.
  - [ ] Mutating tests verify exactly `{"status":"completed"}` and inspect the remote `/logs` endpoint for the specific `request_id`.
  
  **QA Scenarios**:
  ```
  Scenario: Cloud suite runs against deployed instance
    Tool: Bash
    Steps: SOLVE_URL="https://tripletex-124894558027.europe-north1.run.app/solve" uv run pytest tests/cloud/ -v
    Expected: Tests hit the endpoint and parse the subsequent trace logs.
    Evidence: .sisyphus/evidence/task-4-cloud-suite.txt
  ```

  **Commit**: YES | Message: `test(tripx): implement cloud pytest suite` | Files: `tests/cloud/conftest.py`, `tests/cloud/test_cloud_e2e.py`

- [ ] 5. Wire the Cloud Pytest Suite to the Benchmark Ingestion

  **What to do**: Integrate the new cloud pytest suite (Task 4) with the benchmark database (Task 3). Introduce a pytest hook or fixture that captures the `EvaluationResult` or `ScoreResult` of each cloud E2E run and immediately persists it into the SQLite `benchmark.db` and writes the JSON artifact to disk. This establishes the continuous benchmark logging mechanism.
  **Must NOT do**: Do not break local-only unit tests; this ingestion should only trigger during explicitly marked `cloud` integration runs.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Hooking up the previously built ingestion logic to the pytest test execution path.
  - Skills: `[]`
  
  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: Final Verification Wave | Blocked By: 3, 4
  
  **References**:
  - `tests/cloud/conftest.py`
  - `task_tripletex/testing/benchmark.py`
  
  **Acceptance Criteria**:
  - [ ] Running `uv run pytest tests/cloud/` adds a new row to the benchmark DB automatically.
  
  **QA Scenarios**:
  ```
  Scenario: Pytest run persists into benchmark DB
    Tool: Bash
    Steps: uv run pytest tests/cloud/ && sqlite3 benchmark.db "SELECT timestamp, total_score FROM benchmark_runs ORDER BY timestamp DESC LIMIT 1;"
    Expected: Returns a newly appended row representing the latest test execution.
    Evidence: .sisyphus/evidence/task-5-benchmark-integration.txt
  ```

  **Commit**: YES | Message: `test(tripx): auto-ingest cloud pytest results to benchmark db` | Files: `tests/cloud/conftest.py`

## Final Verification Wave (MANDATORY)
> 4 review agents run in PARALLEL. ALL must APPROVE.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Keep commits isolated: `chore(tripx): add pyproject.toml`, `docs(tripx): update agents.md`, `test(tripx): add benchmark ingestion`, `test(tripx): add cloud pytest suite`.