# Astar Island Agentic Development Orchestration

## TL;DR
> **Summary**: Build an Astar-Island-specific, artifact-first development loop where live API data is refreshed into immutable dataset snapshots, all solver evaluation runs replay offline from those snapshots, and a single evaluation skill acts as both the promotion gate and the ad-hoc benchmark entrypoint.
> **Deliverables**:
> - Minimal `AGENTS.md` for Astar Island
> - Astar-specific skills for capture, contract checking, solving, benchmarking, and evaluation
> - Versioned dataset refresh/freeze pipeline using prior runs + analysis data
> - Replay-based benchmark suite with legality, regression, calibration, and adversarial coverage
> - Evaluation skill that emits machine-readable verdicts and human-readable reports
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: 1 → 2 → 3 → 6 → 7 → 10

## Context
### Original Request
Create a plan for optimal agentic development orchestration for the NM i AI Astar Island task, including which skills to create, how to structure a minimalistic `AGENTS.md`, how to design a strong evaluation suite using prior-run queries plus final state data, and how to design an AI skill that uses the suite to evaluate solutions.

### Interview Summary
- Optimize for benchmark rigor first, not raw iteration speed.
- Scope is Astar Island only.
- Primary benchmark target is agent workflows, not just isolated prediction models.
- The evaluation skill must support both mandatory promotion gating and ad-hoc benchmark runs.
- Prior-run data should be refreshed from APIs because the corpus can change as new runs are performed.

### Metis Review (gaps addressed)
- Freeze every live refresh into an immutable dataset version before benchmarking; never benchmark against a moving live corpus.
- Compare candidates against both a blessed baseline and the last blessed candidate.
- Report per-seed metrics, not just an overall mean.
- Explicitly protect against mixed dataset versions, partial analysis availability, illegal tensors, and regressions hidden by averages.
- Keep the system Astar-only and reuse the existing solver lane under `idk_2/astar_island_dr_plan_1/solver/` instead of inventing a generic framework.

## Work Objectives
### Core Objective
Create an implementation-ready orchestration design for Astar Island where agents operate through explicit artifacts, benchmark changes on frozen offline datasets, and use a single evaluation skill as the authoritative decision-maker for solver promotion.

### Deliverables
- `AGENTS.md` at the Astar Island root with role-to-lane mapping, exact commands, artifact handoffs, and stop conditions.
- Skills under `.claude/skills/` for:
  - `astar-contract-check`
  - `astar-refresh-dataset`
  - `astar-solve-round`
  - `astar-benchmark-suite`
  - `astar-evaluate-solution`
  - `astar-regression-review`
- Versioned dataset artifacts for rounds, seeds, query traces, analysis payloads, manifests, and hashes.
- Replay harness that loads frozen datasets and runs deterministic offline evaluation.
- Benchmark reports with legality, per-seed KL, aggregate score, calibration, stability, and adversarial summaries.

### Definition of Done (verifiable conditions with commands)
- `uv run pytest tests/astar/test_contract.py -q` passes.
- `uv run pytest tests/astar/test_dataset_refresh.py -q` passes.
- `uv run pytest tests/astar/test_replay_harness.py -q` passes.
- `uv run pytest tests/astar/test_benchmark_suite.py -q` passes.
- `uv run pytest tests/astar/test_evaluate_skill.py -q` passes.
- `uv run python -m nmframemog.astar_island refresh-dataset --round-id <completed-round-id> --cache-dir .artifacts/astar-island --dataset-version <version>` creates an immutable manifest with hashes.
- `uv run python -m nmframemog.astar_island benchmark --cache-dir .artifacts/astar-island --dataset-version <version> --candidate baseline` produces deterministic JSON and Markdown reports on repeated runs.
- `uv run python -m nmframemog.astar_island evaluate-solution promote --cache-dir .artifacts/astar-island --dataset-version <version> --candidate candidate` exits non-zero on regression and zero on pass.

### Must Have
- Artifact-driven lanes: capture, solve, evaluate, report.
- Frozen benchmark datasets created from live API refreshes.
- Canonical legality and scoring semantics aligned with challenge docs and existing validator/contract logic.
- Benchmark coverage for legality, regression, calibration, robustness, and per-seed behavior.
- Minimal `AGENTS.md` that is operational, not verbose.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No generic multi-challenge orchestration abstractions.
- No promotion gate that depends on live API calls.
- No score comparisons across mixed dataset versions or different rounds.
- No benchmark suite that reports only an average and hides per-seed regressions.
- No AGENTS file that duplicates broad project policy from `CLAUDE.md`.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: TDD with contract-first sequencing, using `pytest` plus existing quality gates.
- QA policy: Every task includes agent-executed happy-path and failure-path scenarios.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`
- Promotion policy: only frozen offline datasets may be used for `promote` mode.
- Determinism policy: repeated benchmark runs against the same dataset and candidate must produce identical verdict JSON.

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: contract + dataset + replay foundations (Tasks 1-4)

Wave 2: benchmark suite + skills + AGENTS orchestration docs (Tasks 5-9)

Wave 3: promotion gate + CI/gating integration + operator examples (Tasks 10-12)

### Dependency Matrix (full, all tasks)
| Task | Depends On | Unlocks |
|---|---|---|
| 1 | — | 2, 3, 4, 5 |
| 2 | 1 | 3, 5, 6, 10 |
| 3 | 1, 2 | 5, 6, 7, 10 |
| 4 | 1 | 8, 9 |
| 5 | 1, 2, 3 | 7, 10 |
| 6 | 2, 3 | 7, 10, 11 |
| 7 | 3, 5, 6 | 10, 11 |
| 8 | 1, 4 | 9, 12 |
| 9 | 4, 8 | 12 |
| 10 | 2, 3, 5, 6, 7 | 11, 12 |
| 11 | 6, 7, 10 | 12 |
| 12 | 8, 9, 10, 11 | Final verification |

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 4 tasks → `deep`, `unspecified-high`, `quick`
- Wave 2 → 5 tasks → `deep`, `writing`, `unspecified-high`, `quick`
- Wave 3 → 3 tasks → `unspecified-high`, `writing`, `quick`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Codify the canonical Astar evaluation contract

  **What to do**: Create a single canonical artifact contract for evaluation work, centered on `round -> seed -> query trace -> prediction tensor -> analysis payload -> benchmark report`. Reuse the existing solver contract and validator semantics rather than creating parallel definitions. Include dataset manifest fields for dataset version, capture timestamp, round id, seed ids, source endpoints, artifact hashes, solver id, and schema version. Define explicit invalid states: mixed dataset versions, missing seed analysis, missing hash, illegal tensor, stale class mapping.
  **Must NOT do**: Do not introduce a generalized experiment-tracking system. Do not create a second scoring definition separate from the existing validator/challenge rules.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: contract decisions affect every downstream lane.
  - Skills: `[]` — No existing special skill is required for the planning-target implementation task.
  - Omitted: `[quality-check]` — Not yet useful before code and tests exist.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 3, 4, 5 | Blocked By: —

  **References**:
  - Pattern: `docs/overview.md:21-26` — 5 seeds and 50 shared queries shape the dataset and benchmarking model.
  - Pattern: `docs/endpoint.md:222-263` — canonical tensor shape and validation rules.
  - Pattern: `docs/scoring.md:21-27` — entropy-weighted KL is the authoritative metric family.
  - Pattern: `docs/scoring.md:57-66` — mandatory 0.01 floor guidance.
  - API/Type: `idk_2/astar_island_dr_plan_1/solver/contract.py` — existing constants and mapping rules.
  - API/Type: `idk_2/astar_island_dr_plan_1/solver/validator.py` — existing tensor legality and scoring helper.
  - API/Type: `idk_2/astar_island_dr_plan_1/solver/models.py` — existing datamodel style for round/query payloads.

  **Acceptance Criteria**:
  - [ ] A schema module or equivalent typed contract exists for dataset manifests, frozen snapshots, benchmark inputs, and benchmark outputs.
  - [ ] A contract test proves the system rejects mixed-version datasets.
  - [ ] A contract test proves the system rejects tensors with zeros, wrong shapes, or bad normalization.
  - [ ] A contract test proves benchmark reports include per-seed metrics and aggregate metrics.

  **QA Scenarios**:
  ```
  Scenario: Canonical contract happy path
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_contract.py -q`
    Expected: All contract tests pass; report schema includes dataset_version, candidate_id, per_seed metrics, aggregate metrics, and verdict.
    Evidence: .sisyphus/evidence/task-1-contract.txt

  Scenario: Illegal tensor and mixed-dataset failure
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_contract.py -q -k "illegal or mixed"`
    Expected: Tests confirm zero-probability tensors and mixed dataset versions are rejected with explicit error messages.
    Evidence: .sisyphus/evidence/task-1-contract-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): codify evaluation artifact contract` | Files: `idk_2/astar_island_dr_plan_1/solver/evaluation_contract.py`, `Astar-Island/tests/astar/test_contract.py`

- [x] 2. Build the live-refresh to frozen-snapshot dataset pipeline

  **What to do**: Implement a refresh pipeline that fetches prior-run data from the live APIs, then freezes it into immutable dataset snapshots. Capture round metadata, seed metadata, query traces, analysis payloads, prior predictions if available, and a manifest with hashes. Snapshot creation must be append-only and timestamped/versioned so a later refresh cannot overwrite earlier benchmark corpora. Treat partial refresh as a hard failure for benchmark datasets.
  **Must NOT do**: Do not let benchmark execution fetch live data on the fly. Do not overwrite an existing dataset version. Do not silently skip missing seeds.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: it is mostly integration and artifact plumbing with strong guardrails.
  - Skills: `[]` — Reuse existing API/cache patterns directly.
  - Omitted: `[deploy]` — No deployment work is needed.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 3, 5, 6, 10 | Blocked By: 1

  **References**:
  - Pattern: `docs/endpoint.md:34-92` — round listing and round-detail retrieval.
  - Pattern: `docs/endpoint.md:129-179` — simulate payload structure for query traces.
  - Pattern: `docs/endpoint.md:348-363` — analysis endpoint for post-round ground truth.
  - Pattern: `docs/quickstart.md:27-36` — active round and auth flow examples.
  - API/Type: `idk_2/astar_island_dr_plan_1/solver/api.py` — existing endpoint client patterns.
  - API/Type: `idk_2/astar_island_dr_plan_1/solver/cache.py` — existing artifact persistence shape.
  - Pattern: `idk_2/astar_island_dr_plan_1/README.md` — local cache-first artifact workflow.

  **Acceptance Criteria**:
  - [ ] Refresh command creates a new dataset directory with manifest, hashes, and immutable raw payload copies.
  - [ ] Snapshot creation fails if any required seed analysis is missing.
  - [ ] Running refresh twice produces two versioned datasets; the earlier one remains untouched.
  - [ ] Dataset manifest records capture timestamp, round id, seed list, source endpoints, and file hashes.

  **QA Scenarios**:
  ```
  Scenario: Refresh and freeze dataset
    Tool: Bash
    Steps: Run `uv run python -m nmframemog.astar_island refresh-dataset --round-id <completed-round-id> --cache-dir .artifacts/astar-island --dataset-version <version>`
    Expected: A new immutable dataset directory is created under `.artifacts/astar-island/datasets/<version>/` with manifest.json, hashes.json, raw/ payloads, and analysis/ payloads for every seed.
    Evidence: .sisyphus/evidence/task-2-refresh.txt

  Scenario: Partial analysis failure
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_dataset_refresh.py -q -k partial`
    Expected: Refresh fails closed with a clear error when any required seed analysis is absent or incomplete.
    Evidence: .sisyphus/evidence/task-2-refresh-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): add frozen dataset refresh pipeline` | Files: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py`, `Astar-Island/tests/astar/test_dataset_refresh.py`

- [x] 3. Implement deterministic offline replay loading

  **What to do**: Build the replay harness that loads only frozen artifacts and reconstructs benchmark inputs without contacting live APIs. It must normalize ordering, seed iteration, file discovery, and report generation so repeated runs are byte-stable for the same candidate and dataset. Provide clear adapter seams for solver candidates and workflow candidates.
  **Must NOT do**: Do not allow hidden fallback to live endpoints. Do not rely on mutable in-memory handoff from the refresh pipeline.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: determinism and handoff design are central to trust in the benchmark.
  - Skills: `[]`
  - Omitted: `[solve-challenge]` — Too broad; this task needs lane-specific replay plumbing only.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 5, 6, 7, 10 | Blocked By: 1, 2

  **References**:
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/pipeline.py` — orchestration shape to preserve.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/cache.py` — artifact loading/storage expectations.
  - Pattern: `idk_2/astar_island_dr_plan_1/example_trace.json` — replay/debug fixture pattern.
  - API/Type: `idk_2/astar_island_dr_plan_1/solver/models.py` — seed/round/query object structure.

  **Acceptance Criteria**:
  - [ ] Offline replay command consumes only frozen artifacts.
  - [ ] Repeating the same replay twice on the same dataset and candidate produces identical JSON output.
  - [ ] Replay fails clearly if any manifest, hash, or artifact is missing.
  - [ ] Replay supports workflow-candidate adapters without rewriting the benchmark core.

  **QA Scenarios**:
  ```
  Scenario: Deterministic offline replay
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_replay_harness.py -q`
    Expected: Tests confirm repeated offline replay outputs are identical and no live API calls occur.
    Evidence: .sisyphus/evidence/task-3-replay.txt

  Scenario: Missing artifact failure
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_replay_harness.py -q -k missing`
    Expected: Replay aborts with explicit missing-artifact errors and no partial report is emitted.
    Evidence: .sisyphus/evidence/task-3-replay-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): add deterministic offline replay harness` | Files: `idk_2/astar_island_dr_plan_1/solver/replay_harness.py`, `Astar-Island/tests/astar/test_replay_harness.py`

- [x] 4. Write the minimal Astar-specific AGENTS.md

  **What to do**: Add a root-level `AGENTS.md` for the Astar Island subtree that maps the work into the four lanes (capture, solve, evaluate, report), assigns responsibilities to the canonical agents already described in `CLAUDE.md`, documents exact entry commands, defines artifact boundaries, and includes explicit stop conditions. Keep it short and operational.
  **Must NOT do**: Do not restate broad repo coding policy already covered in `CLAUDE.md`. Do not add motivational prose or generic agent theory.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: concise operational documentation is the deliverable.
  - Skills: `[]`
  - Omitted: `[quality-check]` — Documentation task first; code gates come later.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 8, 9 | Blocked By: 1

  **References**:
  - Pattern: `../CLAUDE.md` — canonical agent roster and quality gates.
  - Pattern: `docs/README.md` — challenge doc index to cite as source-of-truth.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/pipeline.py` — solver lane entrypoint reference.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/debug_visualization.py` — report/debug artifact lane.

  **Acceptance Criteria**:
  - [ ] `AGENTS.md` contains only: purpose, source-of-truth docs, artifact layout, lane ownership, exact commands, required gates, and out-of-scope rules.
  - [ ] Each lane has clear inputs, outputs, and stop conditions.
  - [ ] `AGENTS.md` points agents to the evaluation skill as the sole promotion authority.

  **QA Scenarios**:
  ```
  Scenario: AGENTS.md completeness
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_agents_doc.py -q`
    Expected: Tests confirm AGENTS.md includes all required sections and command references.
    Evidence: .sisyphus/evidence/task-4-agents.txt

  Scenario: AGENTS.md minimality guardrail
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_agents_doc.py -q -k minimal`
    Expected: Tests fail if AGENTS.md duplicates forbidden generic policy sections or omits lane stop conditions.
    Evidence: .sisyphus/evidence/task-4-agents-error.txt
  ```

  **Commit**: YES | Message: `docs(astar): add minimal agents guide` | Files: `AGENTS.md`, `Astar-Island/tests/astar/test_agents_doc.py`

- [x] 5. Add the benchmark metric suite beyond raw legality

  **What to do**: Implement the metric layer that evaluates candidates using per-seed entropy-weighted KL, aggregate score, legality, calibration, stability, and fallback detection. Calibration must consider full-distribution quality, not just top-class accuracy. Include comparisons against the blessed baseline and last blessed candidate.
  **Must NOT do**: Do not reduce evaluation to a single scalar. Do not report a pass based on aggregate score if any hard gate fails.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: scoring and calibration semantics are challenge-critical.
  - Skills: `[]`
  - Omitted: `[deploy]` — No deployment relevance.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7, 10 | Blocked By: 1, 2, 3

  **References**:
  - Pattern: `docs/scoring.md:29-49` — entropy-aware scoring guidance.
  - Pattern: `docs/scoring.md:72-79` — missing-seed penalty motivates hard failure on incomplete datasets.
  - API/Type: `idk_2/astar_island_dr_plan_1/solver/validator.py` — base legality and scoring behavior.
  - Pattern: `deep_research/plan_2.md` — forecasting-centric reasoning for evaluating belief quality.

  **Acceptance Criteria**:
  - [ ] Benchmark output includes per-seed scores, aggregate score, legality status, calibration summary, and baseline deltas.
  - [ ] Hard-gate failures include illegal tensors, mixed datasets, missing seeds, or missing blessed references.
  - [ ] Benchmark metric tests cover both improvement and regression cases.

  **QA Scenarios**:
  ```
  Scenario: Benchmark metrics happy path
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_benchmark_suite.py -q -k metrics`
    Expected: Per-seed KL, aggregate score, calibration metrics, and baseline deltas are emitted in the report schema.
    Evidence: .sisyphus/evidence/task-5-benchmark.txt

  Scenario: Illegal candidate hard failure
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_benchmark_suite.py -q -k illegal`
    Expected: Illegal tensors trigger a hard failure verdict rather than a degraded pass.
    Evidence: .sisyphus/evidence/task-5-benchmark-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): add benchmark metrics and baselines` | Files: `idk_2/astar_island_dr_plan_1/solver/benchmark_suite.py`, `Astar-Island/tests/astar/test_benchmark_suite.py`

- [x] 6. Build the prior-run dataset tasks for history refresh and backfill

  **What to do**: Implement history-refresh logic that iterates completed rounds, fetches prior-run artifacts from APIs, and composes benchmark datasets that combine queries, final predictions, and final analysis state. Build support for incremental backfill without mutating previously frozen benchmark versions. The output should distinguish raw refreshed captures from curated frozen benchmark manifests.
  **Must NOT do**: Do not assume a single round is representative. Do not merge newly refreshed data into an old frozen benchmark manifest.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: network/backfill logic plus artifact hygiene.
  - Skills: `[]`
  - Omitted: `[quality-check]` — run later across the full slice.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7, 10, 11 | Blocked By: 2, 3

  **References**:
  - Pattern: `docs/endpoint.md:34-54` — enumerate rounds for backfill.
  - Pattern: `docs/endpoint.md:348-363` — post-round analysis retrieval.
  - Pattern: `docs/overview.md:21-26` — shared query budget; dataset interpretation must preserve this context.
  - Pattern: `idk_2/astar_island_dr_plan_1/README.md` — cache-first workflow to emulate for historical corpora.

  **Acceptance Criteria**:
  - [ ] Backfill command creates raw history snapshots for completed rounds without touching existing frozen datasets.
  - [ ] Curated benchmark manifests can be generated from selected history snapshots.
  - [ ] Tests prove that new refreshes create new versions instead of mutating old versions.

  **QA Scenarios**:
  ```
  Scenario: History refresh and backfill
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_dataset_refresh.py -q -k history`
    Expected: Historical capture refresh creates separate raw snapshots and curated benchmark manifests with stable references.
    Evidence: .sisyphus/evidence/task-6-history.txt

  Scenario: Frozen dataset immutability
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_dataset_refresh.py -q -k immutable`
    Expected: Tests confirm refreshed data cannot mutate previously frozen benchmark manifests.
    Evidence: .sisyphus/evidence/task-6-history-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): add historical dataset backfill flow` | Files: `idk_2/astar_island_dr_plan_1/solver/dataset_refresh.py`, `Astar-Island/tests/astar/test_dataset_refresh.py`

- [x] 7. Implement adversarial and robustness evaluation cases

  **What to do**: Add robustness checks that specifically target Astar Island failure modes: edge-clamped viewports, per-seed regressions hidden by averages, incomplete seed sets, stale/mixed dataset inputs, illegal class ordering, and overconfident predictions that worsen KL despite looking decisive. Mark which checks are hard-gate versus report-only; default to hard-gating all data-integrity and legality failures, with adversarial performance regressions reported and optionally gated by threshold.
  **Must NOT do**: Do not collapse all robustness cases into a single generic “stress test.” Do not make adversarial checks depend on manual inspection.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: challenge-specific edge cases determine benchmark trustworthiness.
  - Skills: `[]`
  - Omitted: `[solve-challenge]` — Too broad for this targeted evaluation task.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 10, 11 | Blocked By: 3, 5, 6

  **References**:
  - Pattern: `docs/endpoint.md:153-179` — viewport bounds/clamping behavior.
  - Pattern: `docs/scoring.md:57-66` — overconfidence/zero-probability failure.
  - Pattern: `docs/scoring.md:72-79` — missing seed handling.
  - API/Type: `idk_2/astar_island_dr_plan_1/solver/contract.py` — class ordering and constants.
  - API/Type: `idk_2/astar_island_dr_plan_1/solver/validator.py` — legality enforcement foundation.

  **Acceptance Criteria**:
  - [ ] Benchmark suite includes dedicated tests for edge-clamped viewport traces, mixed dataset versions, per-seed regressions, incomplete seeds, and class-order mismatches.
  - [ ] Hard-gate cases fail with explicit verdict reasons.
  - [ ] Report-only robustness cases are clearly labeled and included in summary outputs.

  **QA Scenarios**:
  ```
  Scenario: Robustness suite happy path
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_benchmark_suite.py -q -k robustness`
    Expected: All challenge-specific robustness checks execute and report distinct pass/fail categories.
    Evidence: .sisyphus/evidence/task-7-robustness.txt

  Scenario: Per-seed hidden regression failure
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_benchmark_suite.py -q -k per_seed_regression`
    Expected: Candidate fails even if aggregate mean improves when a configured hard-gate per-seed regression threshold is violated.
    Evidence: .sisyphus/evidence/task-7-robustness-error.txt
  ```

  **Commit**: YES | Message: `test(astar): add robustness and adversarial benchmarks` | Files: `Astar-Island/tests/astar/test_benchmark_suite.py`, `idk_2/astar_island_dr_plan_1/solver/benchmark_suite.py`

- [x] 8. Create the minimal Astar skill set for orchestration lanes

  **What to do**: Implement the Astar-specific skills as thin, explicit wrappers around the existing solver and new evaluation modules. Required skills: `astar-contract-check`, `astar-refresh-dataset`, `astar-solve-round`, `astar-benchmark-suite`, `astar-evaluate-solution`, and `astar-regression-review`. Each skill must define purpose, allowed inputs, exact commands, artifacts produced, and refusal conditions.
  **Must NOT do**: Do not create redundant skills whose only difference is naming. Do not let skills contain hidden policy that contradicts `AGENTS.md` or the evaluation contract.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: skills are mostly structured operational wrappers.
  - Skills: `[]`
  - Omitted: `[deploy]` — Deployment is not part of this orchestration slice.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 9, 12 | Blocked By: 1, 4

  **References**:
  - Pattern: `.claude/skills/solve-challenge/SKILL.md` — repo skill format to mirror.
  - Pattern: `.claude/skills/quality-check/SKILL.md` — concise command-oriented skill pattern.
  - Pattern: `idk_2/astar_island_dr_plan_1/cli.py` — solver command entrypoint.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/pipeline.py` — actual execution lane to wrap.

  **Acceptance Criteria**:
  - [ ] Each required Astar skill exists with a single clear responsibility.
  - [ ] Skill docs specify commands, inputs, outputs, evidence paths, and refusal conditions.
  - [ ] Skill docs reference the frozen dataset rule for benchmarking and promotion.

  **QA Scenarios**:
  ```
  Scenario: Skill definitions complete
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_skill_docs.py -q`
    Expected: Tests confirm all six skill docs exist and include required command, input, output, and refusal sections.
    Evidence: .sisyphus/evidence/task-8-skills.txt

  Scenario: Missing refusal guardrail failure
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_skill_docs.py -q -k refusal`
    Expected: Tests fail if any skill omits the frozen-dataset or scope guardrail language.
    Evidence: .sisyphus/evidence/task-8-skills-error.txt
  ```

  **Commit**: YES | Message: `docs(astar): add orchestration skill set` | Files: `.claude/skills/astar-*/SKILL.md`, `Astar-Island/tests/astar/test_skill_docs.py`

- [x] 9. Align AGENTS.md with the skill set and solver lane entrypoints

  **What to do**: Update `AGENTS.md` so that each lane maps directly to one or more skills and to the relevant concrete code entrypoints. The file must explain exactly which skills an agent should invoke in each lane, where artifacts live, and when the agent must stop and hand off.
  **Must NOT do**: Do not let AGENTS.md and skill docs drift. Do not introduce undocumented alternative workflows.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: this is documentation alignment work.
  - Skills: `[]`
  - Omitted: `[quality-check]` — not the primary need.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 12 | Blocked By: 4, 8

  **References**:
  - Pattern: `AGENTS.md` — file created in Task 4.
  - Pattern: `.claude/skills/astar-*/SKILL.md` — skill docs created in Task 8.
  - Pattern: `../CLAUDE.md` — top-level role roster to stay consistent with.

  **Acceptance Criteria**:
  - [ ] Every lane in `AGENTS.md` references the concrete skill names and artifact handoffs.
  - [ ] No skill exists without being mentioned in `AGENTS.md`.
  - [ ] No undocumented workflow appears in examples or operator guidance.

  **QA Scenarios**:
  ```
  Scenario: AGENTS-to-skill alignment
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_agents_doc.py -q -k alignment`
    Expected: Tests confirm AGENTS.md references every required Astar skill and no extra lane workflows exist.
    Evidence: .sisyphus/evidence/task-9-alignment.txt

  Scenario: Drift detection failure
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_agents_doc.py -q -k drift`
    Expected: Tests fail if a documented lane references a missing skill or if a skill lacks AGENTS.md coverage.
    Evidence: .sisyphus/evidence/task-9-alignment-error.txt
  ```

  **Commit**: YES | Message: `docs(astar): align agents guide with skills` | Files: `AGENTS.md`, `.claude/skills/astar-*/SKILL.md`, `Astar-Island/tests/astar/test_agents_doc.py`

- [x] 10. Build the evaluation skill as the authoritative benchmark and promotion entrypoint

  **What to do**: Implement `astar-evaluate-solution` as the single authoritative interface for benchmarking candidate workflows. It must support exactly two modes: `benchmark` and `promote`. `benchmark` runs the full offline suite and emits reports without changing blessed references. `promote` runs the same suite, compares against the blessed baseline and last blessed candidate, and returns a machine-readable pass/fail verdict with explicit reasons.
  **Must NOT do**: Do not let `promote` mutate datasets. Do not let `benchmark` update blessed references. Do not add live API fetches here.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: it binds contract, replay, metrics, and policy.
  - Skills: `[]`
  - Omitted: `[deploy]` — irrelevant.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 11, 12 | Blocked By: 2, 3, 5, 6, 7

  **References**:
  - Pattern: `.claude/skills/quality-check/SKILL.md` — concise command-first skill style.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/validator.py` — underlying legality/scoring logic.
  - Pattern: `idk_2/astar_island_dr_plan_1/solver/pipeline.py` — candidate execution shape.
  - External: `docs/scoring.md` — authoritative scoring semantics.

  **Acceptance Criteria**:
  - [ ] `benchmark` mode produces report JSON, Markdown summary, per-seed breakdown, and evidence links.
  - [ ] `promote` mode exits non-zero on hard-gate failure or regression versus blessed references.
  - [ ] Both modes require a frozen dataset and reject live-only inputs.
  - [ ] Verdict payload includes candidate id, dataset version, baseline delta, last-blessed delta, hard-gate failures, and final status.

  **QA Scenarios**:
  ```
  Scenario: Benchmark mode happy path
    Tool: Bash
    Steps: Run `uv run python -m nmframemog.astar_island evaluate-solution benchmark --cache-dir .artifacts/astar-island --dataset-version <version> --candidate baseline`
    Expected: Report JSON and Markdown summary are produced without mutating blessed references.
    Evidence: .sisyphus/evidence/task-10-evaluate.txt

  Scenario: Promote mode regression failure
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_evaluate_skill.py -q -k promote_regression`
    Expected: Promote mode returns a failing verdict with explicit per-seed and baseline regression reasons.
    Evidence: .sisyphus/evidence/task-10-evaluate-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): add evaluation skill entrypoint` | Files: `idk_2/astar_island_dr_plan_1/solver/evaluate_skill.py`, `.claude/skills/astar-evaluate-solution/SKILL.md`, `Astar-Island/tests/astar/test_evaluate_skill.py`

- [x] 11. Add blessed-reference management and gate policy

  **What to do**: Implement the policy layer that records the blessed baseline and last blessed candidate separately, validates candidate comparisons against both, and prevents accidental promotion on incomplete evidence. Provide explicit storage for blessed references keyed by dataset version compatibility and candidate id.
  **Must NOT do**: Do not allow implicit “latest report wins” behavior. Do not compare candidates against references from incompatible dataset versions.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: promotion policy is stateful and easy to get wrong.
  - Skills: `[]`
  - Omitted: `[quality-check]` — final repo-wide checks happen after integration.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 12 | Blocked By: 6, 7, 10

  **References**:
  - Pattern: `docs/scoring.md:72-79` — incomplete seeds make score interpretation unsafe.
  - Pattern: `AGENTS.md` — promotion policy must align with operator guidance.
  - Pattern: `.claude/skills/astar-evaluate-solution/SKILL.md` — skill policy must stay synchronized.

  **Acceptance Criteria**:
  - [ ] Blessed baseline and last blessed candidate are stored separately.
  - [ ] Promotion fails if dataset version compatibility is invalid.
  - [ ] Promotion fails if either blessed reference is missing when policy requires it.
  - [ ] Tests cover pass, fail, and incompatible-reference cases.

  **QA Scenarios**:
  ```
  Scenario: Blessed reference policy happy path
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_evaluate_skill.py -q -k blessed`
    Expected: Candidate is compared against both blessed references and verdict contains both deltas.
    Evidence: .sisyphus/evidence/task-11-policy.txt

  Scenario: Incompatible reference failure
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_evaluate_skill.py -q -k incompatible`
    Expected: Promotion aborts with an explicit incompatible-dataset verdict instead of producing a misleading comparison.
    Evidence: .sisyphus/evidence/task-11-policy-error.txt
  ```

  **Commit**: YES | Message: `feat(astar): add blessed promotion policy` | Files: `idk_2/astar_island_dr_plan_1/solver/blessed_refs.py`, `Astar-Island/tests/astar/test_evaluate_skill.py`

- [x] 12. Add operator examples, CI gate wiring, and final command surface

  **What to do**: Finalize the operator-facing surface by documenting the exact commands for refreshing datasets, running ad-hoc benchmarks, and invoking promotion. Wire the benchmark suite into the local quality workflow and any existing CI/test command surface so the evaluation gate is runnable by agents without manual interpretation.
  **Must NOT do**: Do not hide important commands only inside prose docs. Do not make CI depend on live network access.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: mostly command wiring and documentation alignment after heavy logic is done.
  - Skills: `[quality-check]` — Reason: final command surface should align with repository gates.
  - Omitted: `[deploy]` — no deploy work.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Final verification | Blocked By: 8, 9, 10, 11

  **References**:
  - Pattern: `../CLAUDE.md` — quality gate commands to preserve.
  - Pattern: `.claude/skills/quality-check/SKILL.md` — quality-gate wrapping style.
  - Pattern: `AGENTS.md` — command surface must match the documented workflow.

  **Acceptance Criteria**:
  - [ ] Operator docs show exactly one command for dataset refresh, one for ad-hoc benchmark, and one for promotion.
  - [ ] Local test/quality workflow can run the benchmark suite offline.
  - [ ] CI/test commands do not require live API access.

  **QA Scenarios**:
  ```
  Scenario: Offline operator workflow happy path
    Tool: Bash
    Steps: Run `uv run pytest tests/astar -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
    Expected: All Astar evaluation tests and repo quality gates pass without live network access.
    Evidence: .sisyphus/evidence/task-12-ops.txt

  Scenario: Command surface drift failure
    Tool: Bash
    Steps: Run `uv run pytest tests/astar/test_agents_doc.py tests/astar/test_skill_docs.py -q`
    Expected: Tests fail if AGENTS.md, skill docs, and implemented command entrypoints diverge.
    Evidence: .sisyphus/evidence/task-12-ops-error.txt
  ```

  **Commit**: YES | Message: `docs(astar): finalize evaluation operator workflow` | Files: `AGENTS.md`, `.claude/skills/astar-*/SKILL.md`, `Astar-Island/tests/astar/*`, `pyproject.toml`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: `feat(astar): codify evaluation artifact contract`
- Commit 2: `feat(astar): add frozen dataset refresh pipeline`
- Commit 3: `feat(astar): add deterministic offline replay harness`
- Commit 4: `docs(astar): add minimal agents guide`
- Commit 5: `feat(astar): add benchmark metrics and baselines`
- Commit 6: `feat(astar): add historical dataset backfill flow`
- Commit 7: `test(astar): add robustness and adversarial benchmarks`
- Commit 8: `docs(astar): add orchestration skill set`
- Commit 9: `docs(astar): align agents guide with skills`
- Commit 10: `feat(astar): add evaluation skill entrypoint`
- Commit 11: `feat(astar): add blessed promotion policy`
- Commit 12: `docs(astar): finalize evaluation operator workflow`

## Success Criteria
- Agents can follow `AGENTS.md` without guessing responsibilities or commands.
- Frozen benchmark datasets can be refreshed from live APIs and replayed offline deterministically.
- The benchmark suite evaluates more than legality: per-seed score, calibration, regression, and robustness are all reported.
- `astar-evaluate-solution` is the only benchmark/promotion authority and supports both `benchmark` and `promote` modes.
- Promotion decisions are reproducible, dataset-version aware, and protected against hidden regressions.
