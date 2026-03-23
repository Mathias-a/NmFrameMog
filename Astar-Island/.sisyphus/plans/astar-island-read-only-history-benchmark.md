# Astar Island Read-Only History Benchmark

## TL;DR
> **Summary**: Add a read-only history-benchmark workflow that fetches one explicit completed round via approved GET endpoints, freezes a single-round dataset without cached-query contamination, and benchmarks the historical submitted prediction offline with the existing `evaluate-solution benchmark` path.
> **Deliverables**:
> - Read-only API helpers for `GET /astar-island/my-rounds` and `GET /astar-island/my-predictions/{round_id}`
> - Single-round dataset-refresh guardrails for history capture (`include_cached_queries=False`, `require_submitted_predictions=True`, extra source-endpoint provenance)
> - New `solver/history_benchmark.py` orchestration layer for preflight + capture + offline benchmark
> - New CLI command `benchmark-history`
> - TDD coverage for empty-artifact capture, completed-round validation, submission completeness, and benchmark output
> - README operator docs for the new command and its guardrails
> **Effort**: Medium
> **Parallel**: YES - 2 waves
> **Critical Path**: 1 → 2 → 4 → 5 → 6

## Context
### Original Request
Use the production API to evaluate previous runs without making real `simulate` or `submit` queries, and determine how the model would perform on the retrieved result set.

### Interview Summary
- User approved a read-only live policy: `GET /astar-island/my-rounds`, `GET /astar-island/my-predictions/{round_id}`, and `GET /astar-island/analysis/{round_id}/{seed_index}` are allowed.
- `POST /astar-island/simulate` and `POST /astar-island/submit` are explicitly forbidden.
- Scope is benchmark-only, not promote.
- Scope is one dataset version at a time and one round per benchmark run.
- The plan must benchmark the historical submitted prediction by default, not an auto-generated current solver candidate.
- Round selection must be explicit via `round_id`; do not auto-pick the latest completed round.
- Test strategy is TDD.

### Metis Review (gaps addressed)
- Fail closed if any seed lacks `submitted_prediction`; do not synthesize tensors from `my-predictions` argmax/confidence data.
- Do not reuse local cached query payloads in this history workflow; historical benchmarking must default to a clean empty query trace.
- Preserve the existing single-round offline benchmark contract; do not route this through curated or multi-round history datasets because `evaluate_solution` rejects them.
- Treat empty `.artifacts/astar-island` as a supported first-run state.
- Record `my-rounds` and `my-predictions` in dataset provenance so the frozen dataset shows the read-only retrieval path.

## Work Objectives
### Core Objective
Implement a deterministic, read-only workflow that benchmarks one historical team submission by fetching only approved GET endpoints, freezing a single-round dataset, and reusing the existing offline benchmark lane unchanged for scoring.

### Deliverables
- `idk_2/astar_island_dr_plan_1/solver/api.py`
- `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py`
- `idk_2/astar_island_dr_plan_1/solver/history_benchmark.py` (new)
- `idk_2/astar_island_dr_plan_1/cli.py`
- `idk_2/astar_island_dr_plan_1/README.md`
- `tests/astar/test_api.py` (new)
- `tests/astar/test_dataset_refresh.py`
- `tests/astar/test_history_benchmark.py` (new)

### Definition of Done (verifiable conditions with commands)
- `uv run --no-project --with pytest pytest tests/astar/test_api.py -q`
- `uv run --no-project --with pytest pytest tests/astar/test_dataset_refresh.py -q`
- `uv run --no-project --with pytest pytest tests/astar/test_history_benchmark.py -q`
- `uv run --no-project --with pytest pytest tests/astar/test_evaluate_skill.py -q`
- `PYTHONPATH=. uv run --no-project --with pytest pytest tests/astar -q`
- `uv run --no-project python -m nmframemog.astar_island benchmark-history --round-id <completed-round-id> --dataset-version <dataset-version> --cache-dir .artifacts/astar-island` writes a frozen dataset plus `benchmark-report.json`/`benchmark-summary.md` without calling `simulate` or `submit`.

### Must Have
- Only the approved GET endpoints are used in the new workflow.
- `round_id` is a required input; no implicit round selection.
- Historical benchmark candidate defaults to `submitted-<round_id>`.
- The history workflow freezes a normal single-round dataset under `datasets/<version>/`.
- The history workflow ignores local cached query payloads by default and emits an empty `query-trace.json` entry list.
- Dataset provenance records `/astar-island/my-rounds`, `/astar-island/my-predictions/{round_id}`, `/astar-island/rounds/{round_id}`, and `/astar-island/analysis/{round_id}/{seed_index}`.
- Missing submitted predictions fail the capture before benchmark output is written.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No calls to `POST /astar-island/simulate`.
- No calls to `POST /astar-island/submit`.
- No auto-selection of the latest completed round.
- No promote-mode workflow, blessed-reference handling, or multi-round history benchmark wrapper.
- No use of `GET /astar-island/my-predictions/{round_id}` as the scoring tensor source.
- No default reuse of local cached query payloads from prior experiments.
- No default `candidates.json` registry for the historical submission path.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: TDD using `pytest`.
- QA policy: every task includes a passing-path and a failure/edge-path scenario.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.txt`
- Benchmark policy: the new workflow must end in `evaluate_solution(..., mode="benchmark")` rather than a new scoring implementation.
- Network policy: tests must prove only GET endpoints are used; no POST fallback is allowed.

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: API + dataset refresh hardening (Tasks 1-3)

Wave 2: history preflight + orchestration + CLI + docs (Tasks 4-7)

### Dependency Matrix (full, all tasks)
| Task | Depends On | Unlocks |
|---|---|---|
| 1 | — | 4, 5, 6 |
| 2 | — | 5 |
| 3 | 2 | 5 |
| 4 | 1 | 5, 6 |
| 5 | 1, 2, 3, 4 | 6, 7 |
| 6 | 4, 5 | 7 |
| 7 | 5, 6 | Final verification |

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 3 tasks → `quick`, `unspecified-high`
- Wave 2 → 4 tasks → `deep`, `unspecified-high`, `writing`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Add read-only team-history API helpers

  **What to do**: In `idk_2/astar_island_dr_plan_1/solver/api.py`, add `get_my_rounds()` and `get_my_predictions(round_id: str)` methods on `AstarIslandClient` that call `GET /my-rounds` and `GET /my-predictions/{round_id}` through the existing `_request_json()` helper. Create `tests/astar/test_api.py` and cover the exact HTTP method/path pairs, auth header reuse, and error propagation. Keep the return type as raw JSON-compatible objects, matching the existing style of `get_rounds()` and `get_analysis()`.
  **Must NOT do**: Do not rename or remove `get_rounds()`, `simulate()`, or `submit_prediction()`. Do not add parsing logic for `my-predictions` tensors; that endpoint is metadata-only in this workflow.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: one production file plus one new test file with bounded behavior.
  - Skills: `[]` — No repository skill is needed for this isolated client change.
  - Omitted: `[quality-check]` — Targeted pytest coverage is the relevant gate here.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 5, 6 | Blocked By: —

  **References**:
  - Pattern: `docs/endpoint.md:22-31` — authoritative endpoint list including the allowed read-only team endpoints.
  - Pattern: `docs/endpoint.md:265-306` — `GET /astar-island/my-rounds` response fields used for completed-round validation.
  - Pattern: `docs/endpoint.md:314-339` — `GET /astar-island/my-predictions/{round_id}` response fields used for seed-coverage preflight.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/api.py:13-67` — existing client method style and `_request_json()` usage.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `tests/astar/test_api.py` exists and asserts `get_my_rounds()` issues `GET /my-rounds`.
  - [ ] `tests/astar/test_api.py` exists and asserts `get_my_predictions("round-001")` issues `GET /my-predictions/round-001`.
  - [ ] `uv run --no-project --with pytest pytest tests/astar/test_api.py -q` passes.
  - [ ] Existing client methods still use their original paths and verbs.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Read-only endpoint helpers use correct GET routes
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_api.py -q`
    Expected: Tests prove the new helpers call only GET routes and reuse the existing auth/request machinery.
    Evidence: .sisyphus/evidence/task-1-read-only-api.txt

  Scenario: Helper error propagation stays fail-closed
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_api.py -q -k "error or failure"`
    Expected: Tests prove HTTP failures surface as errors without retries, POST fallbacks, or silent coercion.
    Evidence: .sisyphus/evidence/task-1-read-only-api-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): add read-only history api helpers` | Files: `idk_2/astar_island_dr_plan_1/solver/api.py`, `tests/astar/test_api.py`

- [ ] 2. Isolate historical dataset capture from cached query artifacts

  **What to do**: Extend `refresh_dataset_snapshot()` in `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py` with `include_cached_queries: bool = True` and `additional_source_endpoints: Sequence[str] = ()`. Thread both values through `_capture_round_artifacts()`. When `include_cached_queries=False`, skip `_freeze_query_payloads()` entirely, emit `query_trace.entries == ()`, keep `query_budget` unchanged, do not copy `queries/<round_id>/...` into the dataset, and still write `query-trace.json` so `evaluate_solution` sees a valid single-round dataset. Merge `additional_source_endpoints` into a sorted unique `manifest.source_endpoints` collection alongside the existing round/detail and analysis endpoints. Add test coverage in `tests/astar/test_dataset_refresh.py` for the new flag and provenance merge.
  **Must NOT do**: Do not change the default `refresh_dataset_snapshot()` behavior used by the existing lane. Do not fetch or synthesize query payloads. Do not touch `refresh_history_snapshot()` or curated dataset code paths.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: contract-sensitive artifact changes in a core pipeline file.
  - Skills: `[]`
  - Omitted: `[astar-refresh-dataset]` — This task changes the implementation behind the skill, not the operator workflow itself.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5 | Blocked By: —

  **References**:
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py:288-365` — current single-round dataset creation flow.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py:731-890` — `_capture_round_artifacts()` source-endpoint collection and dataset artifact writing.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py:897-971` — `_freeze_query_payloads()` behavior to bypass in the history workflow.
  - API/Type: `idk_2/astar_island_dr_plan_1/solver/benchmark_suite.py:735-769` — empty query traces are legal as long as the trace remains single-version, contiguous, and within budget.
  - Test: `tests/astar/test_dataset_refresh.py:70-128` — existing snapshot test shape to extend without breaking defaults.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `refresh_dataset_snapshot(..., include_cached_queries=False, additional_source_endpoints=(...))` writes a valid dataset with `query_trace.entries == ()`.
  - [ ] The resulting dataset omits `queries/<round_id>/...` artifacts when cached-query capture is disabled.
  - [ ] `manifest.source_endpoints` includes the injected history-preflight endpoints plus `/astar-island/rounds/<round_id>` and `/astar-island/analysis/<round_id>/<seed_index>`.
  - [ ] `uv run --no-project --with pytest pytest tests/astar/test_dataset_refresh.py -q` passes.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Historical dataset capture ignores local query cache
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_dataset_refresh.py -q -k "cached_queries or source_endpoints"`
    Expected: Tests prove the history mode writes an empty query trace, no frozen query artifacts, and merged source endpoints.
    Evidence: .sisyphus/evidence/task-2-history-capture.txt

  Scenario: Default refresh behavior remains unchanged
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_dataset_refresh.py -q -k "freezes_new_versioned_dataset or records_manifest_and_hashes"`
    Expected: Existing single-round refresh behavior still captures queries when the new flag is not used.
    Evidence: .sisyphus/evidence/task-2-history-capture-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): add clean history dataset capture mode` | Files: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py`, `tests/astar/test_dataset_refresh.py`

- [ ] 3. Fail closed when a historical round lacks submitted predictions

  **What to do**: Extend `refresh_dataset_snapshot()` and `_capture_round_artifacts()` with `require_submitted_predictions: bool = False`. When that flag is true, raise `ValueError` as soon as any parsed analysis payload has `submitted_prediction is None`. Keep the temp directory cleanup semantics so no partial dataset version remains on disk. Add tests in `tests/astar/test_dataset_refresh.py` that cover both: (a) incomplete history rounds fail and leave no dataset directory behind, and (b) complete history rounds write `predictions/submitted-<round_id>/seed-XX.json` for every seed.
  **Must NOT do**: Do not make submitted-prediction completeness mandatory for all existing refresh flows. Do not backfill from `my-predictions` argmax/confidence. Do not write partial `predictions/submitted-<round_id>/...` trees when the flag is true and any seed is missing.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: this is a new hard failure in a core artifact writer.
  - Skills: `[]`
  - Omitted: `[astar-contract-check]` — This task is about dataset-capture preconditions, not saved-prediction schema validation.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 5 | Blocked By: 2

  **References**:
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py:1007-1045` — `analysis.prediction` is already parsed into `submitted_prediction`.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py:823-872` — current prediction artifact writing path for `submitted-<round_id>`.
  - Pattern: `docs/endpoint.md:348-372` — analysis payload contains the full `prediction` tensor and ground truth.
  - Test: `tests/astar/test_dataset_refresh.py:130-148` — existing fail-closed pattern for partial analysis capture.
  - Test: `tests/astar/test_evaluate_skill.py:414-499` — fake analysis client pattern with full per-seed prediction payloads.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `refresh_dataset_snapshot(..., require_submitted_predictions=True)` raises a clear error if any seed analysis omits `prediction`.
  - [ ] No partial dataset directory remains after the failure.
  - [ ] Complete history rounds still write `predictions/submitted-<round_id>/seed-XX.json` for every seed.
  - [ ] `uv run --no-project --with pytest pytest tests/astar/test_dataset_refresh.py -q` passes.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Complete historical round produces all submitted prediction artifacts
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_dataset_refresh.py -q -k "submitted_prediction or complete_history"`
    Expected: Tests prove a complete round writes `predictions/submitted-<round_id>/seed-XX.json` for every seed.
    Evidence: .sisyphus/evidence/task-3-submitted-predictions.txt

  Scenario: Missing submitted prediction fails with no partial dataset
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_dataset_refresh.py -q -k "missing_prediction or require_submitted"`
    Expected: Tests prove the capture aborts before finalizing the dataset version and leaves no partial dataset behind.
    Evidence: .sisyphus/evidence/task-3-submitted-predictions-error.txt
  ```

  **Commit**: YES | Message: `fix(astar): fail closed on incomplete historical submissions` | Files: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py`, `tests/astar/test_dataset_refresh.py`

- [ ] 4. Build explicit previous-run preflight for one completed round

  **What to do**: Create `idk_2/astar_island_dr_plan_1/solver/history_benchmark.py` with a read-only protocol that exposes only `get_my_rounds()`, `get_my_predictions(round_id)`, `get_round_detail(round_id)`, and `get_analysis(round_id, seed_index)`. In that file, add `@dataclass(frozen=True) class HistoryBenchmarkPreflight` with fields `round_id: str`, `seed_indices: tuple[int, ...]`, `candidate_id: str`, and `additional_source_endpoints: tuple[str, ...]`. Implement `prepare_history_benchmark(*, client, round_id: str) -> HistoryBenchmarkPreflight`: validate the round exists in `my-rounds`, require `status == "completed"`, parse round detail to get `seeds_count`, require `my-predictions` to return unique seed indices matching `range(seeds_count)`, and return `candidate_id = f"submitted-{round_id}"` plus `additional_source_endpoints = ("/astar-island/my-rounds", f"/astar-island/my-predictions/{round_id}")`. Add `tests/astar/test_history_benchmark.py` to cover completed-round success, active/missing round rejection, duplicate/missing seed rejection, and explicit-round requirement.
  **Must NOT do**: Do not auto-select the latest completed round. Do not call `refresh_history_snapshot()` or `build_curated_history_dataset()`. Do not treat `my-predictions` argmax/confidence values as substitute tensors for evaluation. Do not persist raw `my-rounds` or `my-predictions` payload files inside the dataset; provenance is recorded through `source_endpoints` only.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: this task locks the workflow’s control plane and fail-closed selection logic.
  - Skills: `[]`
  - Omitted: `[astar-benchmark-suite]` — The benchmark contract already exists; this task is only preflight and orchestration input validation.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 5, 6 | Blocked By: 1

  **References**:
  - Pattern: `docs/endpoint.md:265-306` — completed-round fields from `my-rounds`.
  - Pattern: `docs/endpoint.md:314-339` — seed coverage and score metadata from `my-predictions`.
  - Pattern: `docs/endpoint.md:65-92` — round detail carries `seeds_count` and initial states.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py:1153-1198` — existing completed-round selection logic to mirror, but without append-only snapshot semantics.
  - Test: `tests/astar/test_dataset_refresh.py:261-319` — fake history-refresh client patterns for `rounds_payload` and completed filtering.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `tests/astar/test_history_benchmark.py` proves non-completed or missing rounds are rejected before analysis fetch begins.
  - [ ] `tests/astar/test_history_benchmark.py` proves `my-predictions` coverage must equal `range(seeds_count)` from round detail.
  - [ ] The preflight output always uses `candidate_id == submitted-<round_id>`.
  - [ ] `uv run --no-project --with pytest pytest tests/astar/test_history_benchmark.py -q -k "preflight or round_selection or submission_coverage"` passes.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Completed round with full submission coverage passes preflight
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_history_benchmark.py -q -k "preflight and complete"`
    Expected: Tests prove the preflight accepts one explicit completed round and returns `submitted-<round_id>` as the candidate id.
    Evidence: .sisyphus/evidence/task-4-history-preflight.txt

  Scenario: Active round or missing seed coverage is rejected
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_history_benchmark.py -q -k "active or missing_seed or duplicate_seed"`
    Expected: Tests prove the workflow fails before capture when the round is not completed or submissions are incomplete.
    Evidence: .sisyphus/evidence/task-4-history-preflight-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): validate previous-run benchmark inputs` | Files: `idk_2/astar_island_dr_plan_1/solver/history_benchmark.py`, `tests/astar/test_history_benchmark.py`

- [ ] 5. Orchestrate read-only capture and offline benchmark for historical submissions

  **What to do**: In `solver/history_benchmark.py`, add `@dataclass(frozen=True) class HistoryBenchmarkOutputs` with fields `round_id`, `dataset_version`, `candidate_id`, `dataset_dir`, `report_path`, `summary_path`, `report_payload`, and `summary_text`. Implement `benchmark_previous_run(*, cache_dir: Path, client, round_id: str, dataset_version: str) -> HistoryBenchmarkOutputs` so it first calls `prepare_history_benchmark(*, client=client, round_id=round_id)`, then calls `refresh_dataset_snapshot(cache=LocalCache(cache_dir), client=client, round_id=round_id, dataset_version=dataset_version, solver_id="astar-history-submitted", include_cached_queries=False, require_submitted_predictions=True, additional_source_endpoints=preflight.additional_source_endpoints)`, then calls `evaluate_solution(cache_dir=cache.root, dataset_version=dataset_version, candidate_id=preflight.candidate_id, mode="benchmark")`. Return `HistoryBenchmarkOutputs` populated from those results. Cover the module with end-to-end tests in `tests/astar/test_history_benchmark.py` using an empty cache root and a fake read-only client; assert that no POST-only methods are needed or invoked.
  **Must NOT do**: Do not generate a current solver candidate. Do not create `candidates.json` for the default history path. Do not change `evaluate_solution()` or add a second benchmark engine. Do not freeze cached queries in this workflow.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: this task stitches together live-read preflight, dataset capture, and the offline benchmark lane.
  - Skills: `[]`
  - Omitted: `[astar-evaluate-solution]` — The implementation reuses the existing evaluator instead of re-documenting operator behavior.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 6, 7 | Blocked By: 1, 2, 3, 4

  **References**:
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py:288-365` — single-round dataset creation entrypoint to reuse.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/evaluate_skill.py:34-129` — canonical offline benchmark wrapper.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/evaluate_skill.py:132-169` — default candidate resolution for `submitted-<round_id>` without `candidates.json`.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/evaluate_skill.py:250-303` — benchmark output payload and summary shape to preserve.
  - Test: `tests/astar/test_evaluate_skill.py:15-60` — benchmark report expectations.
  - Test: `tests/astar/test_evaluate_skill.py:209-266` — curated/multi-round rejection that this workflow must avoid by staying single-round.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `benchmark_previous_run(...)` succeeds from an empty `.artifacts/astar-island` root when the fake client supplies a completed round, full predictions, and analyses.
  - [ ] The dataset manifest records `/astar-island/my-rounds`, `/astar-island/my-predictions/{round_id}`, `/astar-island/rounds/{round_id}`, and every `/astar-island/analysis/{round_id}/{seed_index}`.
  - [ ] The frozen dataset contains `predictions/submitted-<round_id>/seed-XX.json` for every seed and an empty `query-trace.json` entry list.
  - [ ] The benchmark output reports `mode == "benchmark"` and `candidate_id == submitted-<round_id>`.
  - [ ] `uv run --no-project --with pytest pytest tests/astar/test_history_benchmark.py -q` passes.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Empty-artifact history benchmark succeeds end-to-end
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_history_benchmark.py -q -k "end_to_end or benchmark_previous_run"`
    Expected: Tests prove the workflow captures a dataset from an empty cache root and writes benchmark outputs for `submitted-<round_id>`.
    Evidence: .sisyphus/evidence/task-5-history-benchmark.txt

  Scenario: Workflow never needs POST-only client methods
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_history_benchmark.py -q -k "no_post or read_only_client"`
    Expected: Tests prove the orchestration can run against a client that exposes only the approved GET methods.
    Evidence: .sisyphus/evidence/task-5-history-benchmark-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): benchmark historical submissions offline` | Files: `idk_2/astar_island_dr_plan_1/solver/history_benchmark.py`, `tests/astar/test_history_benchmark.py`

- [ ] 6. Add the `benchmark-history` CLI entrypoint

  **What to do**: In `idk_2/astar_island_dr_plan_1/cli.py`, add a top-level `benchmark-history` subcommand with required `--round-id` and `--dataset-version`, plus `--cache-dir`, `--base-url`, and `--token-env-var` matching existing CLI conventions. Build the client through `build_live_client_from_environment()`, fail if the token is absent, call `benchmark_previous_run(...)`, and print a single JSON object containing `round_id`, `dataset_version`, `candidate_id`, `dataset_dir`, `report_path`, `summary_path`, `report`, and `summary`. Cover it in `tests/astar/test_history_benchmark.py` by calling `cli.main([...])` with a fake read-only client injected via monkeypatch; assert the JSON contract and missing-token failure behavior.
  **Must NOT do**: Do not add `--candidate`, `--promote`, or auto-discovery flags. Do not reuse `solve-round` flags like `--execute-live-queries` or `--submit`. Do not let the command fall back to public `get_rounds()` for team history.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: the command is small, but it is the public operator surface for the new workflow.
  - Skills: `[]`
  - Omitted: `[git-master]` — No git operation is part of the implementation task itself.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 7 | Blocked By: 4, 5

  **References**:
  - Pattern: `idk_2/astar_island_dr_plan_1/cli.py:27-203` — existing subparser style and command dispatch.
  - Pattern: `idk_2/astar_island_dr_plan_1/cli.py:308-353` — output formatting patterns for `fetch-analysis` and `evaluate-solution`.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/pipeline.py:243-249` — token-gated client creation helper.
  - Test: `tests/astar/test_evaluate_skill.py:337-412` — existing CLI result-shape test pattern.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m nmframemog.astar_island benchmark-history --round-id <round-id> --dataset-version <version> --cache-dir <cache>` prints the expected JSON shape.
  - [ ] The command fails immediately when the configured token env var is missing.
  - [ ] The command delegates to `benchmark_previous_run(...)` and does not expose any candidate-selection or promote flags.
  - [ ] `uv run --no-project --with pytest pytest tests/astar/test_history_benchmark.py -q -k "cli"` passes.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: CLI prints benchmark-history JSON output
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_history_benchmark.py -q -k "cli and output"`
    Expected: Tests prove the command emits dataset/report paths plus the benchmark payload and summary.
    Evidence: .sisyphus/evidence/task-6-benchmark-history-cli.txt

  Scenario: CLI fails when token env var is missing
    Tool: Bash
    Steps: Run `uv run --no-project --with pytest pytest tests/astar/test_history_benchmark.py -q -k "cli and token"`
    Expected: Tests prove the command exits with a clear error before any history capture starts.
    Evidence: .sisyphus/evidence/task-6-benchmark-history-cli-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): add benchmark-history command` | Files: `idk_2/astar_island_dr_plan_1/cli.py`, `tests/astar/test_history_benchmark.py`

- [ ] 7. Document the read-only operator workflow

  **What to do**: Update `idk_2/astar_island_dr_plan_1/README.md` to add a dedicated `benchmark-history` command section after the existing `fetch-analysis` docs. Document the required token env var, required `--round-id`, required `--dataset-version`, default candidate id `submitted-<round_id>`, and the explicit read-only endpoint policy. Add a short artifact-layout note that this workflow writes to `datasets/<version>/...` and `evaluation/<version>/submitted-<round_id>/...` and intentionally freezes no query artifacts. Preserve the README’s current concise command style.
  **Must NOT do**: Do not document `simulate` or `submit` as part of the new workflow. Do not claim `my-predictions` supplies the benchmark tensor. Do not mention multi-round curated datasets in the operator instructions.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: this is operator-facing documentation work with concrete technical constraints.
  - Skills: `[]`
  - Omitted: `[deploy]` — No deployment work is involved.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: Final verification | Blocked By: 5, 6

  **References**:
  - Pattern: `idk_2/astar_island_dr_plan_1/README.md:51-121` — existing command documentation format.
  - Pattern: `docs/endpoint.md:22-31` — canonical allowed/disallowed endpoint list to cite accurately.
  - Pattern: `docs/endpoint.md:348-372` — analysis endpoint returns the full prediction tensor used for history benchmarking.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/evaluate_skill.py:298-303` — benchmark summary wording to keep operator expectations aligned.

  **Acceptance Criteria** (agent-executable only):
  - [ ] README includes a `benchmark-history` example command with `--round-id`, `--dataset-version`, and `--cache-dir`.
  - [ ] README explicitly states that the workflow uses only read-only GET endpoints and forbids `simulate`/`submit`.
  - [ ] README explains that the benchmark candidate is `submitted-<round_id>` and that cached query artifacts are intentionally excluded.
  - [ ] `grep -n "benchmark-history\|submitted-<round_id>\|read-only GET endpoints" idk_2/astar_island_dr_plan_1/README.md` returns the new documentation lines.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: README shows the exact operator command
    Tool: Bash
    Steps: Run `grep -n "benchmark-history" idk_2/astar_island_dr_plan_1/README.md`
    Expected: Output includes the new command section and at least one executable example.
    Evidence: .sisyphus/evidence/task-7-readme.txt

  Scenario: README states the workflow guardrails explicitly
    Tool: Bash
    Steps: Run `grep -n "simulate\|submit\|submitted-<round_id>\|cached query" idk_2/astar_island_dr_plan_1/README.md`
    Expected: Output shows the docs forbid simulate/submit in this path, explain the historical candidate id, and mention cached-query exclusion.
    Evidence: .sisyphus/evidence/task-7-readme-error.txt
  ```

  **Commit**: YES | Message: `docs(astar): document read-only history benchmark flow` | Files: `idk_2/astar_island_dr_plan_1/README.md`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit after each numbered task using the message specified in the task block.
- Do not squash Tasks 2 and 3 together until both their pytest scopes pass, because they change the same pipeline file but represent distinct guardrails.
- Keep `benchmark-history` CLI wiring (Task 6) separate from README updates (Task 7) so operator-surface regressions are isolated.
- Do not create any promote-mode commits in this workstream.

## Success Criteria
- One explicit command benchmarks one explicit completed round using only approved GET endpoints.
- The workflow succeeds from an empty `.artifacts/astar-island` tree.
- The frozen dataset is single-round, immutable, and accepted by the existing offline benchmark lane.
- The benchmark report candidate id is always `submitted-<round_id>` for the default historical-submission path.
- Any missing round completion, missing seed submission, missing `submitted_prediction`, or token absence fails with a clear error before benchmark output is trusted.
