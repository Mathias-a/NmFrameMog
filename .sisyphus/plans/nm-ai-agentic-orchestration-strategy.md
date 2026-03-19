# NM i AI Agentic Orchestration Strategy

## Objective

Build a competition operating model that can win across all NM i AI challenge types by separating shared agent infrastructure from challenge-specific execution loops. The plan assumes this repository is currently early-stage, with empty `src/task_1`, `src/task_2`, and `src/task_3` directories and no committed challenge implementation yet.

## Success criteria

- Every challenge has a named owner workflow, evaluation loop, and deployment target.
- Shared agent infrastructure is reused where it creates leverage and avoided where it adds latency or confusion.
- Each loop has measurable gates: offline eval, sandbox eval, submission eval, and leaderboard learning.
- The team can run multiple improvement loops in parallel without stepping on each other.

## Challenge classification

| Challenge | Nature of task | Winning mode |
| --- | --- | --- |
| Grocery Bot | Real-time control over WebSocket | Fast state estimation, planning, and replay-driven policy tuning |
| NorgesGruppen Data | Packaged object detection model in sandbox | Dataset rigor, training quality, packaging stability, reproducible scoring |
| Tripletex | Tool-using accounting agent over HTTPS | Reliable task interpretation, API execution, and state verification |
| Astar Island | Probabilistic world-state prediction | Simulation insight, calibration, uncertainty modeling, iterative analysis |

## Strategic principle

Do not force a single agent architecture onto all four tasks. We should share infrastructure for evaluation, prompting, telemetry, and experiment tracking, while letting each challenge use the execution loop that matches its physics.

## Shared orchestration layer

### 1. Knowledge and experiment memory

- Store imported official docs in `docs/nm-ai/`.
- Maintain `docs/nm-ai/spec-confidence-register.md` so each lane knows which rules are validated, inferred, or still unknown.
- Add challenge notebooks or reports under `research/<challenge>/` as work begins.
- Persist leaderboard experiments in append-only logs so agents can compare approaches instead of re-learning the same lesson.

### 1b. Evaluator parity over convenient proxies

The highest-leverage rule for this repo is: optimize against the real judge or the closest judge-like environment available, not against docs excerpts or easy local proxies.

- If a rule is only excerpt-derived, do not build a brittle optimization around it without live confirmation.
- If offline metrics disagree with sandbox or live behavior, promote the judge-like environment and downgrade the proxy.
- Each challenge needs a frozen benchmark bundle that becomes the promotion target for challenger variants.

### 2. Role-based agent system

- **Scout agent**: reads docs, recent runs, and leaderboard deltas; produces next hypotheses.
- **Builder agent**: implements the smallest testable improvement.
- **Evaluator agent**: runs offline benchmarks, sandbox validations, and regression comparisons.
- **Reviewer agent**: checks whether the change actually improved score, latency, robustness, or calibration.
- **Submitter agent**: handles packaging or deployment and records the exact artifact submitted.

No agent should both change logic and declare victory. Evaluation must stay adversarial.

### 2b. Promotion rule per lane

Each challenge lane must define exactly:

- **One promotion metric** — the primary number that decides whether a challenger can replace the champion
- **One safety metric** — the hard guardrail that blocks reckless gains
- **One threshold** — the minimum improvement required to promote

If a proposed change cannot say how it clears those three fields, it does not get promoted.

### 3. Repo lane design

Recommended target layout:

- `src/shared/` — schemas, telemetry, experiment runners, model wrappers, deployment helpers
- `src/grocery_bot/` — WebSocket game client, policy engine, replay harness
- `src/norgesgruppen_data/` — training, inference, packaging, local scoring utilities
- `src/tripletex_agent/` — `/solve` endpoint, planner, tool executor, verification layer
- `src/astar_island/` — observation parsing, predictive models, calibration, analysis tooling

Until this layout exists, `src/task_1..3` can be used as incubation lanes, but the final codebase should be challenge-named rather than numerically named.

## How we win each challenge

### Grocery Bot

1. Build a protocol-correct client first: consume `game_state`, emit valid actions, handle `game_over`.
2. Capture every game as a replay artifact.
3. Train decision heuristics against replay data before introducing heavier planning.
4. Measure score by map and difficulty; optimize for worst-case maps, not just average-case easy runs.
5. Only add LLM-style reasoning if it proves net-positive under the round time limit.

**Promotion rule**

- Primary metric: median score across frozen replay bundle
- Safety metric: p95 response latency per round
- Threshold: promote only if median score improves and latency stays within budget

**Agentic orchestration:**

- Scout mines replay failures.
- Builder modifies routing or task-assignment policy.
- Evaluator replays the same seed set.
- Reviewer blocks merges unless the change improves median and p10 score.

**Judge-parity requirement**

- Maintain a frozen replay bundle sampled across maps and difficulty levels.
- Add environment-budget gates for per-round response time and protocol validity.

### NorgesGruppen Data

1. Build a local training/evaluation pipeline with the same input/output contract as `run.py`.
2. Create a packaging test that spins a local container and verifies the sandbox entrypoint.
3. Optimize data quality and augmentations before chasing model novelty.
4. Treat latency, memory, and artifact size as first-class constraints.
5. Track detection and classification components separately because the official score blends them.

**Promotion rule**

- Primary metric: hybrid score on frozen validation bundle
- Safety metric: packaging/runtime/memory success under sandbox-like conditions
- Threshold: promote only if score improves without breaking packaging determinism

**Agentic orchestration:**

- Scout proposes experiments from error clusters.
- Builder runs one controlled model or augmentation change at a time.
- Evaluator logs mAP@0.5 plus packaging-health metrics.
- Reviewer rejects experiments that improve local metrics but hurt sandbox determinism.

**Judge-parity requirement**

- Freeze a benchmark bundle that mirrors official category mix as closely as possible.
- Add hard environment-budget gates for artifact size, runtime, and memory in a sandbox-like container.

### Tripletex

1. Create a `/solve` service with strict request parsing and typed internal task state.
2. Build a tool layer around the Tripletex API with schema validation, retries, and dry-run logging.
3. Use a planner-executor-verifier pattern rather than a single free-form agent loop.
4. Add multilingual prompt normalization and attachment digestion as separate preprocessing steps.
5. Verify resulting API state after each action sequence because the official scorer does exactly that.

**Promotion rule**

- Primary metric: task correctness on frozen sandbox benchmark set
- Safety metric: idempotency + no invalid side effects on retry/failure cases
- Threshold: promote only if correctness improves and retry safety remains intact

**Agentic orchestration:**

- Scout clusters failure cases by task family.
- Builder improves planner prompts, tool routing, or state verification.
- Evaluator replays archived tasks against sandbox accounts.
- Reviewer scores correctness, token cost, API call count, and recovery behavior.

**Judge-parity requirement**

- Make idempotency and post-action state verification first-class gates, not optional design aspirations.
- Keep a frozen benchmark set of sandbox tasks, including partial failure and retry scenarios.

### Astar Island

1. Build data collectors for observations, predictions, and post-round analysis.
2. Start with strong probabilistic baselines before searching for exotic modeling.
3. Enforce probability floors and renormalization everywhere.
4. Use the analysis endpoint to calibrate error by terrain class and region type.
5. Optimize for calibrated distributions, not merely sharp predictions.

**Promotion rule**

- Primary metric: judge-like KL score on frozen benchmark rounds
- Safety metric: calibration stability plus zero-probability violations
- Threshold: promote only if KL improves and no zero-probability regressions appear

**Agentic orchestration:**

- Scout compares calibration gaps from the analysis endpoint.
- Builder adjusts model features, simulation assumptions, or post-processing.
- Evaluator computes offline KL-style proxies and live round deltas.
- Reviewer blocks any change that introduces zero-probability risk or unstable calibration.

**Judge-parity requirement**

- Use a frozen benchmark bundle of round observations plus post-round analysis artifacts.
- Treat the analysis endpoint as the source of truth when it disagrees with simplified offline proxies.

## Weekly execution cadence

### Daily loop

1. Ingest yesterday's experiments and leaderboard changes.
2. Select one hypothesis per challenge.
3. Run implementation and evaluation in parallel.
4. Submit only the top candidate that beats current baseline by predefined margins.
5. Record what changed, what score moved, and what failed.

### Weekly loop

1. Re-rank all challenges by score gap to podium and ease of improvement.
2. Shift compute and agent time toward the highest expected value lane.
3. Freeze one stable submission per challenge while another branch explores aggressively.

### Champion / challenger policy

- Keep one **champion** artifact per challenge that is always deployable or submittable.
- Test all risky improvements as **challengers** against the frozen benchmark bundle first.
- Replace the champion only when the challenger clears the lane-specific promotion rule.
- Never let daily experimentation overwrite the last known good submission path.

## Decision rules for orchestration

- Use agents for search, hypothesis generation, regression analysis, and controlled code edits.
- Use deterministic code for protocol handling, packaging, scoring, and verification.
- If an agent proposes a change without a measurable eval plan, do not implement it.
- If offline results disagree with sandbox results, privilege the environment that matches official scoring more closely.
- If one challenge starts yielding faster leaderboard gains, temporarily overweight it without abandoning the others.
- Promote code to `src/shared/` only after it clearly serves at least two challenge lanes.

## Verification gates

Every proposed improvement must clear all applicable gates:

1. **Spec gate** — matches official docs in `docs/nm-ai/`.
2. **Confidence gate** — any rule used for optimization is labeled validated, inferred, or unknown in the spec-confidence register.
3. **Local gate** — passes unit tests and local benchmarks.
4. **Sandbox gate** — works in the hosted or packaged environment.
5. **Regression gate** — beats or preserves the current champion on the frozen benchmark bundle.
6. **Submission gate** — exact artifact, config, and score are recorded.

## TDD-oriented execution model

For every challenge lane, agents should work in red-green-refactor order instead of directly coding toward a vague success state.

### Red

- Write or extend a failing test, replay assertion, sandbox contract check, or scorecard expectation that captures the next improvement target.
- Save the failure artifact so later agents can verify they fixed the same thing.

### Green

- Implement the smallest change that makes the new check pass.
- Prefer deterministic adapters, typed schemas, and traceable scoring utilities over opaque prompt-only fixes.

### Refactor

- Simplify code paths, remove duplicated prompt logic, and consolidate shared instrumentation only after the green check passes.
- Re-run the exact failing case plus the surrounding regression suite.

## Atomic commit strategy

When the team starts committing implementation work, use narrow slices so each change has a single proof target.

1. **Docs/import commits** — challenge docs, local inventories, strategy updates.
2. **Contract-test commits** — request/response schemas, replay fixtures, scorecard fixtures, failing tests.
3. **Infrastructure commits** — shared runners, telemetry, experiment log format, deployment plumbing.
4. **Challenge baseline commits** — minimal working bot/model/endpoint/predictor that clears one end-to-end test.
5. **Optimization commits** — one measurable improvement per commit with before/after evidence.

Every commit should answer one question: what new proof exists after this change that did not exist before?

## Immediate next implementation steps

### 1. Create challenge-named packages under `src/`

**Implementation target**

- Add `src/shared/`, `src/grocery_bot/`, `src/norgesgruppen_data/`, `src/tripletex_agent/`, and `src/astar_island/`.
- Keep `src/task_1..3` only as temporary migration stubs if needed.

**TDD / QA scenario**

- Add import-smoke tests that fail until every package exposes an `__init__.py` and one top-level public module.
- Verification commands:
  - `uv run python -c "import src.grocery_bot, src.norgesgruppen_data, src.tripletex_agent, src.astar_island"`
  - `uv run mypy`
- Expected result:
  - All imports succeed.
  - Type checking passes with no missing-module errors.

**Atomic commit slice**

- Commit only package creation plus import-smoke tests.

### 2. Build a shared experiment runner and result log format

**Implementation target**

- Add a shared experiment record schema, append-only JSONL or CSV log format, and one CLI entrypoint to register a run.

**TDD / QA scenario**

- Start with a failing test that asserts a run record contains challenge name, hypothesis id, metrics, artifact path, and timestamp.
- Verification commands:
  - `uv run pytest tests/shared/test_experiment_log.py`
  - `uv run python -m src.shared.experiments record --challenge grocery-bot --hypothesis baseline`
- Expected result:
  - The test passes.
  - A new log entry is appended with the required fields and stable serialization.

**Atomic commit slice**

- Commit schema, CLI, and tests together without mixing in challenge logic.

### 3. Prioritize one hosted lane and one offline lane

**Implementation target**

- Stand up a minimal Tripletex `/solve` endpoint and a minimal NorgesGruppen Data `run.py` packaging baseline.

**TDD / QA scenario**

- Tripletex red test: a fixture POST request with `tripletex_credentials` and a simple task must yield a structured response without crashing.
- NorgesGruppen red test: a local packaging check must fail until a valid zip containing `run.py` is produced.
- Verification commands:
  - `uv run pytest tests/tripletex_agent/test_solve_endpoint.py`
  - `uv run pytest tests/norgesgruppen_data/test_package_contract.py`
- Expected result:
  - Hosted baseline handles the request shape deterministically.
  - Offline baseline emits a sandbox-compatible artifact.

**Atomic commit slice**

- Use one commit for the hosted baseline and one separate commit for the offline packaging baseline.

### 4. Add replay tooling for Grocery Bot and analysis ingestion for Astar Island

**Implementation target**

- Implement a replay loader for saved Grocery Bot rounds and an Astar Island ingester for `analysis/{round_id}/{seed_index}` payloads.

**TDD / QA scenario**

- Grocery Bot red test: a recorded `game_state` fixture should fail until the replay harness can step through rounds and compute score summaries.
- Astar Island red test: a saved analysis payload should fail until parsed into calibrated error metrics per terrain class.
- Verification commands:
  - `uv run pytest tests/grocery_bot/test_replay_harness.py`
  - `uv run pytest tests/astar_island/test_analysis_ingestion.py`
- Expected result:
  - Replay summary includes rounds, score, and failure hotspots.
  - Analysis ingestion outputs class-level calibration/error artifacts.

**Atomic commit slice**

- Commit replay tooling independently from prediction-model changes.

### 5. Define baseline scorecards for each challenge before optimizing

**Implementation target**

- Create one machine-readable baseline scorecard per challenge under `scorecards/<challenge>/baseline.json`.

**TDD / QA scenario**

- Start with schema validation tests that fail until each scorecard includes metric name, current baseline, target threshold, environment, and artifact provenance.
- Verification commands:
  - `uv run pytest tests/scorecards/test_scorecard_schema.py`
  - `uv run python -m src.shared.scorecards validate scorecards`
- Expected result:
  - All scorecards validate.
  - Each challenge has a baseline artifact that later submissions can compare against.

**Atomic commit slice**

- Commit the schema and all four baseline scorecards together, with no unrelated model changes.

## Evidence we are actually winning

We are not winning because the architecture sounds good. We are winning when each challenge has a reproducible baseline, a faster improvement loop than our competitors, and a submission history that shows monotonic learning rather than random thrashing.
