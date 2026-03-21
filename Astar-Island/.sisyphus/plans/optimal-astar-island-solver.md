# Optimal Astar-Island Solver

## TL;DR
> **Summary**: Build a benchmark-first solver around a shared round-level particle posterior over hidden simulation parameters, with inner Monte Carlo over stochastic sim seeds, adaptive viewport selection, and a small calibration hedge only when posterior calibration fails.
> **Deliverables**:
> - benchmark-only solver core under `benchmark/src/astar_twin/solver/`
> - query-policy engine with explicit `10 + 30 + 10` budget policy
> - posterior-predictive tensor generator with safe flooring
> - repeated local benchmark suite (`10` full solver runs) and replay validation harness
> - thin benchmark/prod adapter seam so later prod switching is interface-only
> **Effort**: XL
> **Parallel**: YES - 3 waves
> **Critical Path**: Task 1 → Task 3 → Task 6 → Task 7 → Task 8 → Task 9 → Task 11 → Task 12

## Context
### Original Request
- Create a plan for an optimal Astar-Island solver.
- Use the benchmark to evaluate success.
- Account for viewport selection importance.
- Consider two-sided Monte Carlo or a better strategy.
- Ensemble model is viable.

### Interview Summary
- Optimize **locally against the benchmark**, not prod-first.
- Run **multiple local evaluations, default 10 full solver repeats**, to judge robustness.
- Use **particle + inner Monte Carlo** as the primary modeling approach.
- Use **benchmark + replay** for verification.
- Use **tests-after**, with the existing pytest stack.
- Keep architecture **simple to switch to prod later**, but do not build full prod integration in v1.

### Metis Review (gaps addressed)
- Locked the optimization target to **highest mean score across 10 repeated local runs**, tie-broken by best minimum score.
- Locked runtime ceilings to **≤120s per full local solve** and **≤20m per 10-repeat suite** on the implementation machine.
- Locked query allocation to **10 bootstrap + 30 adaptive + 10 reserve**.
- Locked hedge activation to a **gated fallback**, not a default ensemble.
- Locked the benchmark/prod boundary to a dedicated adapter interface so solver code never depends on benchmark stores or hidden fixture params directly.

## Work Objectives
### Core Objective
Create a benchmark-optimized solver that infers shared round-level hidden parameters from budgeted viewport observations, converts that posterior into full-map probability tensors for all 5 seeds, and proves its value through repeated local benchmark and replay evaluation.

### Deliverables
- `benchmark/src/astar_twin/solver/` package with:
  - adapter interfaces
  - observation feature extraction
  - particle prior / posterior / resampling
  - viewport candidate generation and allocator
  - posterior-predictive Monte Carlo tensor builder
  - optional conservative hedge calibrator
  - end-to-end pipeline entrypoint
- `benchmark/tests/solver/` test suite covering core solver behavior.
- Repeated benchmark runner and replay validator that emit machine-readable artifacts under `.sisyphus/evidence/`.
- Benchmark adapter implementation; prod-compatible adapter interface only.

### Definition of Done (verifiable conditions with commands)
- `cd benchmark && pytest -q tests/contracts tests/scoring tests/mc tests/replay tests/solver` exits `0` with no failed tests.
- `cd benchmark && python -m astar_twin.solver.eval.run_benchmark_suite --round-id test-round-001 --repeats 10 --output ../.sisyphus/evidence/task-11-benchmark-suite.json` exits `0`, writes the JSON artifact, and records `repeats=10`, candidate metrics, uniform baseline metrics, and fixed-coverage baseline metrics.
- The benchmark suite artifact shows `candidate.mean_score > fixed_coverage.mean_score` and `candidate.mean_score > uniform.mean_score`.
- `cd benchmark && python -m astar_twin.solver.eval.run_replay_validation --round-id test-round-001 --output ../.sisyphus/evidence/task-12-replay-validation.json` exits `0`, writes the JSON artifact, and records per-seed score, calibration, and disagreement diagnostics.
- `cd benchmark && python -m astar_twin.solver.eval.dump_prediction_stats --round-id test-round-001 --output ../.sisyphus/evidence/task-9-prediction-stats.json` exits `0`, writes the JSON artifact, and confirms every tensor has shape `40x40x6`, all cell sums are within `1e-6` of `1.0`, and no probability is `< 0.01`.

### Must Have
- Shared round-level posterior across all 5 seeds.
- Two-sided Monte Carlo: **outer particle posterior over hidden params + inner stochastic sim runs**.
- Explicit viewport utility based on entropy mass, posterior disagreement, and observation gain.
- Benchmark adapter that hides `simulation_params` from solver logic.
- Final output always passed through `safe_prediction`.
- Repeated evaluation with `10` full solver runs.
- Replay validation and baseline ablations.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No direct use of fixture `simulation_params` inside solver inference code.
- No dependence on `RoundStore`, `BudgetStore`, or HTTP route internals in solver core.
- No generic ensemble framework in v1.
- No full production auth/deployment work.
- No zero probabilities, NaN, Inf, or invalid tensor shapes.
- No hardcoded viewport coordinates tied to a single fixture unless they are generated from map analysis rules.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: **tests-after** with `pytest` from `benchmark/pyproject.toml:31-47`.
- QA policy: every implementation task includes an agent-executed happy-path and failure-path scenario.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`.
- Primary optimization metric: highest **mean score over 10 repeated full local solver runs**.
- Tie-breaker metric: highest **minimum score** across those 10 runs.
- Runtime guardrails:
  - one full local solve: `<= 120s`
  - full 10-repeat benchmark suite: `<= 20m`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: contracts, adapter seam, safety pipeline, observation features, baseline characterization  
Wave 2: particle prior, two-sided likelihood, posterior updates, viewport allocator, posterior-predictive tensor generation  
Wave 3: end-to-end solver pipeline, hedge fallback, repeated benchmark suite, replay validation

### Dependency Matrix (full, all tasks)
| Task | Depends On |
| --- | --- |
| 1 | - |
| 2 | 1 |
| 3 | 1 |
| 4 | 1 |
| 5 | 1 |
| 6 | 2, 3 |
| 7 | 5, 6 |
| 8 | 5, 7 |
| 9 | 4, 7 |
| 10 | 7, 8, 9 |
| 11 | 9, 10 |
| 12 | 10, 11 |

### Agent Dispatch Summary
| Wave | Task Count | Categories |
| --- | --- | --- |
| 1 | 5 | deep, quick |
| 2 | 4 | deep, ultrabrain |
| 3 | 3 | deep, unspecified-high |
| Final Verification | 4 | oracle, unspecified-high, deep |

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Characterize the benchmark contract and baseline floor

  **What to do**:
  - Create `benchmark/tests/solver/test_benchmark_characterization.py` and a minimal baseline helper module under `benchmark/src/astar_twin/solver/baselines.py`.
  - Lock the public benchmark assumptions that solver code must honor:
    - map tensor shape is `H x W x 6`
    - viewport bounds are `5..15`
    - query budget is `50`
    - `safe_prediction` flooring is mandatory
  - Implement two baseline strategies for all future comparisons:
    1. `uniform_baseline`
    2. `fixed_coverage_baseline` using a non-adaptive 50-query map sweep with 15x15 windows, then a default-parameter twin prediction.
  - Expose a machine-readable baseline summary used later by benchmark-suite tasks.

  **Must NOT do**:
  - Do not use fixture hidden params in the baseline.
  - Do not implement adaptive logic here.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: establishes invariants and baselines used by every downstream task.
  - Skills: `[]` — no extra skill required.
  - Omitted: `[quality-check]` — not needed until code exists.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2,3,4,5 | Blocked By: none

  **References**:
  - API limits: `docs/endpoint.md:129-194` — exact simulate request/response rules and 50-query budget.
  - Tensor/submit rules: `docs/endpoint.md:195-264` — exact tensor shape and normalization constraints.
  - Scoring rules: `docs/scoring.md:7-68` — entropy-weighted KL and mandatory probability floor guidance.
  - Public contracts: `benchmark/src/astar_twin/contracts/api_models.py:64-129` — request/response models.
  - Constants: `benchmark/src/astar_twin/contracts/types.py:60-79` — class mapping, viewport bounds, and `MAX_QUERIES`.
  - Test pattern: `benchmark/tests/scoring/test_kl_formula.py:10-34` — precise scoring assertions.
  - Test pattern: `benchmark/tests/mc/test_tensor_shape.py:11-44` — tensor shape and normalization assertions.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_benchmark_characterization.py` exits `0`.
  - [ ] The characterization test suite asserts both baselines return valid `40x40x6` safe tensors for all 5 seeds.
  - [ ] The baseline summary artifact includes `uniform` and `fixed_coverage` entries and records score fields for each.

  **QA Scenarios**:
  ```
  Scenario: Baselines produce valid tensors
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_benchmark_characterization.py`
    Expected: Exit code 0; no failed tests; baseline tensors are valid in assertions.
    Evidence: .sisyphus/evidence/task-1-characterization.txt

  Scenario: Unsafe tensor is rejected by characterization checks
    Tool: Bash
    Steps: Run the targeted negative test added in `test_benchmark_characterization.py` that feeds an all-zero tensor before flooring.
    Expected: Exit code 0; the negative test proves raw unsafe tensor fails pre-floor validation while safe output passes.
    Evidence: .sisyphus/evidence/task-1-characterization-error.txt
  ```

  **Commit**: YES | Message: `test(solver): lock benchmark contract and baseline floor` | Files: `benchmark/src/astar_twin/solver/baselines.py`, `benchmark/tests/solver/test_benchmark_characterization.py`

- [ ] 2. Create the solver package layout and adapter contracts

  **What to do**:
  - Create `benchmark/src/astar_twin/solver/` with `__init__.py`, `interfaces.py`, and `pipeline.py`.
  - Define solver-facing protocols / dataclasses for:
    - round detail access
    - viewport query execution
    - prediction submission sink
    - post-round analysis fetch
  - Freeze the solver core boundary so downstream code only depends on public contracts shaped like `InitialState`, `SimulateResponse`, `SubmitResponse`, and `AnalysisResponse`.
  - Ensure the pipeline accepts an adapter instance and never reaches into benchmark app state or stores.

  **Must NOT do**:
  - Do not implement HTTP auth or prod network calls.
  - Do not import `RoundStore`, `BudgetStore`, or `SubmissionStore` from solver core.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: mostly interface shaping and file scaffolding.
  - Skills: `[]`
  - Omitted: `[quality-check]` — save for larger code wave.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6 | Blocked By: 1

  **References**:
  - App wiring seam: `benchmark/src/astar_twin/api/app.py:18-44` — adapter should sit above this layer, not inside it.
  - Contracts: `benchmark/src/astar_twin/contracts/api_models.py:28-129` — canonical data shapes.
  - API semantics: `docs/quickstart.md:27-98` — future prod switch should preserve these steps.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_interfaces.py` exits `0`.
  - [ ] Interface tests prove solver core can run against a stub adapter without importing benchmark stores.
  - [ ] `pipeline.py` exposes one end-to-end entrypoint that accepts an adapter and returns 5 tensors.

  **QA Scenarios**:
  ```
  Scenario: Stub adapter drives the solver boundary
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_interfaces.py`
    Expected: Exit code 0; stub adapter satisfies interface contract; solver boundary tests pass.
    Evidence: .sisyphus/evidence/task-2-interfaces.txt

  Scenario: Direct store dependency is blocked
    Tool: Bash
    Steps: Run the negative test in `test_interfaces.py` that imports the solver pipeline with store-layer monkeypatch guards.
    Expected: Exit code 0; test confirms solver code does not require benchmark store imports.
    Evidence: .sisyphus/evidence/task-2-interfaces-error.txt
  ```

  **Commit**: YES | Message: `feat(solver): add adapter contracts and pipeline boundary` | Files: `benchmark/src/astar_twin/solver/__init__.py`, `benchmark/src/astar_twin/solver/interfaces.py`, `benchmark/src/astar_twin/solver/pipeline.py`, `benchmark/tests/solver/test_interfaces.py`

- [ ] 3. Implement the benchmark adapter without leaking hidden params

  **What to do**:
  - Create `benchmark/src/astar_twin/solver/adapters/benchmark.py` and `benchmark/tests/solver/test_benchmark_adapter.py`.
  - Implement an adapter that can run local rounds through benchmark fixtures / simulator plumbing while exposing only solver-facing contract objects.
  - The adapter may internally use benchmark fixtures and simulator components, but it must never expose `simulation_params` to the solver.
  - Support deterministic replay mode for tests via explicit RNG seeds.

  **Must NOT do**:
  - Do not let adapter methods return raw `RoundFixture` or `SimulationParams` objects.
  - Do not couple the adapter to FastAPI `TestClient` unless needed for a specific contract test.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: this is the trust boundary between benchmark truth and solver logic.
  - Skills: `[]`
  - Omitted: `[quality-check]` — defer to later wave.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6,10,11,12 | Blocked By: 1

  **References**:
  - App composition: `benchmark/src/astar_twin/api/app.py:18-44` — benchmark environment dependencies.
  - Fixture loading pattern: `benchmark/tests/conftest.py:20-64` — canonical fixture bootstrapping.
  - Round/query contract: `benchmark/tests/contracts/test_simulate.py:16-128` — response expectations and budget semantics.
  - Replay pattern: `benchmark/tests/replay/test_analysis_diff.py:12-41` — analysis path and score checks.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_benchmark_adapter.py` exits `0`.
  - [ ] Adapter tests confirm `simulation_params` are inaccessible from returned solver objects.
  - [ ] Deterministic replay mode produces identical observation transcripts for the same fixed RNG seed.

  **QA Scenarios**:
  ```
  Scenario: Benchmark adapter serves prod-like observations
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_benchmark_adapter.py`
    Expected: Exit code 0; returned objects match solver contracts and deterministic replay assertions pass.
    Evidence: .sisyphus/evidence/task-3-benchmark-adapter.txt

  Scenario: Hidden params cannot leak through adapter
    Tool: Bash
    Steps: Run the negative test in `test_benchmark_adapter.py` that attempts to access `simulation_params` from adapter outputs.
    Expected: Exit code 0; test proves the attribute is absent or inaccessible.
    Evidence: .sisyphus/evidence/task-3-benchmark-adapter-error.txt
  ```

  **Commit**: YES | Message: `feat(solver): add benchmark adapter seam` | Files: `benchmark/src/astar_twin/solver/adapters/benchmark.py`, `benchmark/tests/solver/test_benchmark_adapter.py`

- [ ] 4. Implement tensor finalization and safety enforcement

  **What to do**:
  - Create `benchmark/src/astar_twin/solver/predict/finalize.py` and `benchmark/tests/solver/test_finalize.py`.
  - Centralize final tensor validation and finalization steps:
    - shape check
    - finite-value check
    - per-cell normalization
    - `safe_prediction` application
    - static terrain override before flooring for ocean/mountain cells
  - Export one helper that every prediction path must call before returning tensors.

  **Must NOT do**:
  - Do not duplicate `safe_prediction` logic outside the finalizer.
  - Do not special-case fixture size; use adapter-provided `height` and `width`.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: bounded, safety-critical utility task.
  - Skills: `[]`
  - Omitted: `[quality-check]` — still premature.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 9,10,11,12 | Blocked By: 1

  **References**:
  - Safety implementation: `benchmark/src/astar_twin/scoring/safe_prediction.py:7-10` — canonical floor + renormalize behavior.
  - Scoring hazard: `docs/scoring.md:55-68` — why zeros are catastrophic.
  - Tensor tests: `benchmark/tests/mc/test_tensor_shape.py:19-44` — shape, sums, and >0 assertions.
  - Class mapping: `benchmark/src/astar_twin/contracts/types.py:61-79` — static terrain classes and shape constants.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_finalize.py` exits `0`.
  - [ ] Finalizer tests prove every cell sum is `1.0 ± 1e-6`, every value is finite, and every probability is `>= 0.01`.
  - [ ] Finalizer preserves ocean/mountain certainty directionally while still returning safe non-zero tensors.

  **QA Scenarios**:
  ```
  Scenario: Finalizer makes tensors safe and normalized
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_finalize.py`
    Expected: Exit code 0; tests confirm valid shape, finite values, normalized cells, and floor enforcement.
    Evidence: .sisyphus/evidence/task-4-finalize.txt

  Scenario: NaN or wrong-shape tensor is rejected
    Tool: Bash
    Steps: Run the negative tests in `test_finalize.py` for NaN input and malformed shape input.
    Expected: Exit code 0; tests confirm the finalizer raises or rejects invalid tensors before return.
    Evidence: .sisyphus/evidence/task-4-finalize-error.txt
  ```

  **Commit**: YES | Message: `feat(solver): centralize tensor finalization safety` | Files: `benchmark/src/astar_twin/solver/predict/finalize.py`, `benchmark/tests/solver/test_finalize.py`

- [ ] 5. Implement observation features and hotspot generation

  **What to do**:
  - Create `benchmark/src/astar_twin/solver/observe/features.py`, `benchmark/src/astar_twin/solver/policy/hotspots.py`, and `benchmark/tests/solver/test_hotspots.py`.
  - Extract map-driven candidate hotspots from each seed’s initial state:
    - coastal settlement clusters
    - forest-frontier growth zones
    - multi-settlement conflict corridors
    - reclaim-sensitive ruin/forest edges
  - Default viewport size to `15x15`.
  - Shrink to `10x10` only when hotspot bbox is `< 8x8` and the larger window would dilute mechanism signal.
  - Shrink to `5x5` only for contradiction-resolution probes on a single settlement/port toggle.
  - Produce observation summary features used later by the likelihood model:
    - cell-class counts in viewport
    - alive/dead settlement counts
    - port count
    - mean and variance of settlement `population`, `food`, `wealth`, `defense`

  **Must NOT do**:
  - Do not score candidates yet; only generate and featurize them.
  - Do not use random hotspot placement.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: this encodes the mechanism-aware viewport foundation.
  - Skills: `[]`
  - Omitted: `[quality-check]` — later.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 7,8 | Blocked By: 1

  **References**:
  - Mechanic phases: `benchmark/src/astar_twin/engine/simulator.py:112-129` — growth/conflict/trade/winter/environment execution order.
  - Parameter hooks: `benchmark/src/astar_twin/params/simulation_params.py:52-108` — port, expansion, raid, trade, winter, and reclaim parameters to target.
  - Simulate payload details: `docs/endpoint.md:157-184` — viewport returns grid plus settlement stats.
  - Viewport behavior pattern: `benchmark/tests/contracts/test_simulate.py:16-71` — shape and bounds expectations.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_hotspots.py` exits `0`.
  - [ ] Hotspot tests prove every seed yields at least two bootstrap candidates and every candidate respects viewport bounds.
  - [ ] Feature extraction tests prove all required observation summary fields are present and finite.

  **QA Scenarios**:
  ```
  Scenario: Mechanism-aware hotspots are generated from initial state
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_hotspots.py`
    Expected: Exit code 0; generated candidates respect 5..15 bounds and expected hotspot categories exist.
    Evidence: .sisyphus/evidence/task-5-hotspots.txt

  Scenario: Degenerate seed still gets valid fallback candidates
    Tool: Bash
    Steps: Run the negative/fallback test in `test_hotspots.py` using a sparse or coastless seed fixture.
    Expected: Exit code 0; fallback corridor/grid candidates are produced without crashing.
    Evidence: .sisyphus/evidence/task-5-hotspots-error.txt
  ```

  **Commit**: YES | Message: `feat(solver): add observation features and hotspot generation` | Files: `benchmark/src/astar_twin/solver/observe/features.py`, `benchmark/src/astar_twin/solver/policy/hotspots.py`, `benchmark/tests/solver/test_hotspots.py`

- [ ] 6. Define the particle state, priors, and parameter subset

  **What to do**:
  - Create `benchmark/src/astar_twin/solver/inference/particles.py` and `benchmark/tests/solver/test_particles.py`.
  - Model a particle as a subset of `SimulationParams` plus a log-weight.
  - Infer only this v1 parameter subset:
    - `adjacency_mode`
    - `distance_metric`
    - `update_order_mode`
    - `prosperity_threshold_port`
    - `prosperity_threshold_expand`
    - `expansion_rate`
    - `expansion_radius`
    - `raid_base_prob`
    - `raid_success_scale`
    - `trade_range`
    - `trade_value_scale`
    - `winter_severity_mean`
    - `winter_food_loss_per_population`
    - `collapse_threshold`
    - `collapse_softness`
    - `reclaim_threshold`
    - `ruin_forest_rate`
  - Freeze all other parameters at benchmark defaults from `SimulationParams()`.
  - Initialize exactly `24` particles around defaults.
  - Keep a deterministic sampling mode for test reproducibility.

  **Must NOT do**:
  - Do not infer all params in `SimulationParams`; that is v2 scope creep.
  - Do not tune priors from hidden fixture params.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: this fixes the hidden-state search space.
  - Skills: `[]`
  - Omitted: `[quality-check]` — wait until inference code stabilizes.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 7 | Blocked By: 2,3

  **References**:
  - Simulation params universe: `benchmark/src/astar_twin/params/simulation_params.py:24-108` — exact inferable knobs.
  - Simulator construction: `benchmark/src/astar_twin/engine/simulator.py:36-39` — particles must rehydrate into a `SimulationParams` object.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_particles.py` exits `0`.
  - [ ] Tests confirm `24` particles are initialized, all required fields are present, and deterministic mode is stable.
  - [ ] Tests confirm non-inferred params remain equal to `SimulationParams()` defaults.

  **QA Scenarios**:
  ```
  Scenario: Particle initialization is deterministic under fixed seed
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_particles.py`
    Expected: Exit code 0; deterministic particle initialization snapshot tests pass.
    Evidence: .sisyphus/evidence/task-6-particles.txt

  Scenario: Out-of-range particle values are clamped or rejected
    Tool: Bash
    Steps: Run the negative tests in `test_particles.py` for invalid numeric and enum values.
    Expected: Exit code 0; invalid particles fail validation without leaking into inference.
    Evidence: .sisyphus/evidence/task-6-particles-error.txt
  ```

  **Commit**: YES | Message: `feat(solver): add particle schema and priors` | Files: `benchmark/src/astar_twin/solver/inference/particles.py`, `benchmark/tests/solver/test_particles.py`

- [ ] 7. Implement two-sided likelihood, posterior updates, and resampling

  **What to do**:
  - Create `benchmark/src/astar_twin/solver/inference/likelihood.py`, `posterior.py`, and `benchmark/tests/solver/test_posterior.py`.
  - For each observed viewport, compute particle likelihood using inner MC:
    - run exactly `6` simulations per particle for bootstrap and adaptive updates
    - map simulated terrain to classes with `TERRAIN_TO_CLASS`
    - estimate per-cell predictive probabilities in the queried viewport
    - compute `loglik_grid = Σ log(max(p(observed_class), 1e-6))`
    - compute settlement-stat likelihood from summary features using diagonal Gaussian penalties across simulated summary vectors
    - total log-likelihood = `0.75 * loglik_grid + 0.25 * loglik_stats`
  - Update posterior weights in log-space.
  - After bootstrap, keep top `8` particles and resample to `12`.
  - During adaptive phase, resample only when ESS `< 6`.
  - If top-particle mass exceeds `0.85` while replay/benchmark disagreement remains high later, down-temper weights by exponent `0.5` before continuing.

  **Must NOT do**:
  - Do not compare particles only on exact full-map matches.
  - Do not ignore settlement stats returned by the viewport API.

  **Recommended Agent Profile**:
  - Category: `ultrabrain` — Reason: likelihood design and posterior stability are the highest-value reasoning problem in the plan.
  - Skills: `[]`
  - Omitted: `[quality-check]` — defer.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 8,9,10 | Blocked By: 5,6

  **References**:
  - Inner MC primitive: `benchmark/src/astar_twin/mc/runner.py:8-17` — base run loop.
  - Aggregation primitive: `benchmark/src/astar_twin/mc/aggregate.py:10-22` — convert simulated runs into probabilities.
  - Viewport settlements: `benchmark/src/astar_twin/contracts/api_models.py:73-100` — exact observation stats available.
  - Simulate semantics: `docs/endpoint.md:157-184` — one stochastic outcome per query.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_posterior.py` exits `0`.
  - [ ] Posterior tests prove weight updates are deterministic in replay mode and ESS-triggered resampling only fires below the threshold.
  - [ ] Tests prove the posterior ranking changes when observation evidence changes.

  **QA Scenarios**:
  ```
  Scenario: Posterior updates from viewport evidence
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_posterior.py`
    Expected: Exit code 0; posterior weights, ESS logic, and deterministic replay checks pass.
    Evidence: .sisyphus/evidence/task-7-posterior.txt

  Scenario: Posterior collapse is tempered instead of hard-failing
    Tool: Bash
    Steps: Run the negative test in `test_posterior.py` that injects overwhelming but contradictory evidence.
    Expected: Exit code 0; the test confirms tempering or resampling logic preserves multiple candidate particles.
    Evidence: .sisyphus/evidence/task-7-posterior-error.txt
  ```

  **Commit**: YES | Message: `feat(solver): add two-sided likelihood and posterior updates` | Files: `benchmark/src/astar_twin/solver/inference/likelihood.py`, `benchmark/src/astar_twin/solver/inference/posterior.py`, `benchmark/tests/solver/test_posterior.py`

- [ ] 8. Implement the explicit viewport-allocation policy

  **What to do**:
  - Create `benchmark/src/astar_twin/solver/policy/allocator.py` and `benchmark/tests/solver/test_allocator.py`.
  - Implement the exact query schedule:
    - **Bootstrap 10 queries**: `2` per seed
      - query A: top coastal settlement cluster with `15x15`
      - query B: top frontier/corridor growth-conflict cluster with `15x15`
      - if no coastal cluster exists, replace A with next-best corridor candidate
    - **Adaptive 30 queries**: `6` batches of `5` globally selected candidates
    - **Reserve 10 queries**: held back until a contradiction trigger fires, otherwise released as two final adaptive batches
  - Score each candidate window with:
    - `0.45 * entropy_mass`
    - `0.35 * posterior_disagreement`
    - `0.20 * expected_stat_gain`
    - minus `0.25 * overlap_penalty`
  - Normalize candidate components within each selection step.
  - Reject windows with overlap `> 60%` with a previously queried window in the same seed unless the window is flagged for contradiction resolution.
  - Contradiction triggers:
    - top 2 particles disagree on argmax in `>20%` of cells in a candidate window
    - ESS `< 6`
    - any seed has `< 8` total queries after the adaptive phase

  **Must NOT do**:
  - Do not distribute queries evenly after bootstrap; adaptive phase is global.
  - Do not choose windows by raw area alone.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: policy quality is the main leverage under the 50-query cap.
  - Skills: `[]`
  - Omitted: `[quality-check]` — later.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 10,11 | Blocked By: 5,7

  **References**:
  - Budget constant: `benchmark/src/astar_twin/contracts/types.py:75-79` — hard `50` query cap.
  - Viewport limits: `benchmark/src/astar_twin/contracts/api_models.py:64-71` — legal viewport sizes.
  - User-visible query semantics: `docs/overview.md:21-31` — why viewport choice matters.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_allocator.py` exits `0`.
  - [ ] Tests prove the allocator never exceeds `50` queries and follows the `10 + 30 + 10` policy exactly.
  - [ ] Tests prove contradiction triggers release reserve queries correctly.

  **QA Scenarios**:
  ```
  Scenario: Allocator respects the budget and phase schedule
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_allocator.py`
    Expected: Exit code 0; policy tests confirm bootstrap, adaptive, and reserve accounting exactly.
    Evidence: .sisyphus/evidence/task-8-allocator.txt

  Scenario: High-overlap candidate is rejected unless contradiction is flagged
    Tool: Bash
    Steps: Run the overlap negative test in `test_allocator.py`.
    Expected: Exit code 0; overlapping windows are skipped by default and only admitted under contradiction rules.
    Evidence: .sisyphus/evidence/task-8-allocator-error.txt
  ```

  **Commit**: YES | Message: `feat(solver): add adaptive viewport allocation policy` | Files: `benchmark/src/astar_twin/solver/policy/allocator.py`, `benchmark/tests/solver/test_allocator.py`

- [ ] 9. Implement posterior-predictive tensor generation for all 5 seeds

  **What to do**:
  - Create `benchmark/src/astar_twin/solver/predict/posterior_mc.py` and `benchmark/tests/solver/test_posterior_mc.py`.
  - For final prediction, use the top `6` particles.
  - Allocate exactly `64` simulations per seed across those particles, proportional to posterior weight with a minimum of `4` runs per selected particle.
  - For each particle/seed pair:
    - run full-map simulations with `MCRunner`
    - aggregate with `aggregate_runs`
  - Combine particle tensors by normalized posterior weight.
  - Pass the result through the shared tensor finalizer.
  - If runtime budget is already above `80%` of the ceiling before final prediction, drop to `32` simulations per seed instead of `64` and record the fallback in metrics.

  **Must NOT do**:
  - Do not build the final tensor from queried windows only.
  - Do not bypass the finalizer.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: connects inference to scoring-critical full-map output.
  - Skills: `[]`
  - Omitted: `[quality-check]` — wait for end-to-end wave.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 10,11,12 | Blocked By: 4,7

  **References**:
  - MC runner: `benchmark/src/astar_twin/mc/runner.py:8-17` — simulation batch primitive.
  - Aggregation: `benchmark/src/astar_twin/mc/aggregate.py:10-22` — tensor construction.
  - Tensor tests: `benchmark/tests/mc/test_tensor_shape.py:11-44` — shape and normalization assertions.
  - Score function: `benchmark/src/astar_twin/scoring/kl.py:9-18` — downstream score contract.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_posterior_mc.py` exits `0`.
  - [ ] `cd benchmark && python -m astar_twin.solver.eval.dump_prediction_stats --round-id test-round-001 --output ../.sisyphus/evidence/task-9-prediction-stats.json` exits `0`.
  - [ ] The JSON artifact confirms all 5 seed tensors are safe, normalized, and `40x40x6`.

  **QA Scenarios**:
  ```
  Scenario: Posterior predictive MC returns safe tensors for all seeds
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_posterior_mc.py && python -m astar_twin.solver.eval.dump_prediction_stats --round-id test-round-001 --output ../.sisyphus/evidence/task-9-prediction-stats.json`
    Expected: Exit code 0; tests pass; JSON artifact exists and records valid tensor stats for 5 seeds.
    Evidence: .sisyphus/evidence/task-9-posterior-mc.json

  Scenario: Runtime fallback switches from 64 to 32 runs per seed
    Tool: Bash
    Steps: Run the negative test in `test_posterior_mc.py` with a mocked elapsed-time budget crossing the 80% threshold.
    Expected: Exit code 0; the fallback path is exercised and recorded without producing invalid tensors.
    Evidence: .sisyphus/evidence/task-9-posterior-mc-error.txt
  ```

  **Commit**: YES | Message: `feat(solver): add posterior predictive tensor generation` | Files: `benchmark/src/astar_twin/solver/predict/posterior_mc.py`, `benchmark/tests/solver/test_posterior_mc.py`

- [ ] 10. Build the end-to-end benchmark solver pipeline

  **What to do**:
  - Wire `pipeline.py` to execute the full round flow:
    1. load round detail for 5 seeds
    2. generate bootstrap candidates
    3. issue 10 bootstrap queries
    4. update posterior
    5. iterate adaptive batches and reserve-release logic
    6. generate final tensors for all 5 seeds
    7. return structured metrics and query transcript
  - Create `benchmark/tests/solver/test_pipeline.py`.
  - Persist a query transcript artifact including seed, viewport, utility scores, posterior ESS, and contradiction flags.

  **Must NOT do**:
  - Do not submit tensors or call prod APIs.
  - Do not bypass the allocator or posterior modules with shortcut logic.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: orchestration task that integrates every prior component.
  - Skills: `[]`
  - Omitted: `[quality-check]` — save for near-final wave.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 11,12 | Blocked By: 3,7,8,9

  **References**:
  - Quickstart interaction order: `docs/quickstart.md:27-98` — round → detail → simulate → submit sequence.
  - Adapter seam: `benchmark/src/astar_twin/solver/interfaces.py` — task 2 output.
  - TestClient patterns: `benchmark/tests/contracts/test_simulate.py:16-128` — canonical query semantics.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_pipeline.py` exits `0`.
  - [ ] Pipeline tests confirm exactly `50` or fewer queries are used, 5 tensors are returned, and the transcript artifact is complete.
  - [ ] Replay-mode pipeline runs are deterministic under fixed adapter RNG.

  **QA Scenarios**:
  ```
  Scenario: Full pipeline runs end-to-end on the benchmark adapter
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_pipeline.py`
    Expected: Exit code 0; pipeline tests pass and transcript assertions succeed.
    Evidence: .sisyphus/evidence/task-10-pipeline.txt

  Scenario: Budget exhaustion mid-run still returns valid tensors
    Tool: Bash
    Steps: Run the negative pipeline test that injects a near-exhausted budget before reserve allocation.
    Expected: Exit code 0; pipeline degrades gracefully, stops querying, and still returns safe tensors for all 5 seeds.
    Evidence: .sisyphus/evidence/task-10-pipeline-error.txt
  ```

  **Commit**: YES | Message: `feat(solver): wire end-to-end benchmark pipeline` | Files: `benchmark/src/astar_twin/solver/pipeline.py`, `benchmark/tests/solver/test_pipeline.py`

- [ ] 11. Add the conservative hedge and repeated benchmark suite

  **What to do**:
  - Create `benchmark/src/astar_twin/solver/predict/hedge.py`, `benchmark/src/astar_twin/solver/eval/run_benchmark_suite.py`, and `benchmark/tests/solver/test_benchmark_suite.py`.
  - Implement a gated hedge only when either condition holds:
    - `candidate.mean_score <= fixed_coverage.mean_score + 5` after a benchmark suite
    - replay validation later shows calibration disagreement threshold exceeded on `>= 2` seeds
  - Hedge formula: `q_final = 0.85 * q_particle + 0.15 * q_fixed_coverage`, then pass through the finalizer.
  - Run exactly `10` full solver repeats and record:
    - mean, min, max, std
    - per-seed averages
    - runtime per run
    - hedge activation count
  - Emit `.sisyphus/evidence/task-11-benchmark-suite.json`.

  **Must NOT do**:
  - Do not enable the hedge by default.
  - Do not compare candidates on a single run only.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: evaluation harness plus calibration gating.
  - Skills: `[]`
  - Omitted: `[quality-check]` — optional until final verification.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 12 | Blocked By: 9,10

  **References**:
  - Score function: `benchmark/src/astar_twin/scoring/kl.py:9-18` — benchmark target metric.
  - Analysis score test: `benchmark/tests/replay/test_analysis_diff.py:12-41` — score range pattern.
  - User requirement: repeated local evaluation, default 10 runs — confirmed during interview.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_benchmark_suite.py` exits `0`.
  - [ ] `cd benchmark && python -m astar_twin.solver.eval.run_benchmark_suite --round-id test-round-001 --repeats 10 --output ../.sisyphus/evidence/task-11-benchmark-suite.json` exits `0`.
  - [ ] The JSON artifact records `repeats=10`, `candidate.mean_score > fixed_coverage.mean_score`, and `candidate.mean_score > uniform.mean_score`.

  **QA Scenarios**:
  ```
  Scenario: Benchmark suite evaluates the candidate against both baselines
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_benchmark_suite.py && python -m astar_twin.solver.eval.run_benchmark_suite --round-id test-round-001 --repeats 10 --output ../.sisyphus/evidence/task-11-benchmark-suite.json`
    Expected: Exit code 0; artifact exists; candidate and baseline metrics are all present.
    Evidence: .sisyphus/evidence/task-11-benchmark-suite.json

  Scenario: Hedge only activates when the gate is tripped
    Tool: Bash
    Steps: Run the negative test in `test_benchmark_suite.py` that simulates underperforming candidate metrics and then a healthy candidate run.
    Expected: Exit code 0; hedge activates only for failing calibration / score conditions and stays off otherwise.
    Evidence: .sisyphus/evidence/task-11-benchmark-suite-error.txt
  ```

  **Commit**: YES | Message: `feat(solver): add repeated benchmark suite and hedge gate` | Files: `benchmark/src/astar_twin/solver/predict/hedge.py`, `benchmark/src/astar_twin/solver/eval/run_benchmark_suite.py`, `benchmark/tests/solver/test_benchmark_suite.py`

- [ ] 12. Add replay validation, ablations, and prod-switch adapter seam validation

  **What to do**:
  - Create `benchmark/src/astar_twin/solver/eval/run_replay_validation.py`, `benchmark/tests/solver/test_replay_validation.py`, and `benchmark/tests/solver/test_prod_adapter_contract.py`.
  - Evaluate the final solver against:
    - uniform baseline
    - fixed-coverage baseline
    - particle solver without hedge
    - particle solver with hedge enabled when gated
  - Emit `.sisyphus/evidence/task-12-replay-validation.json` with:
    - per-seed score
    - calibration / disagreement metrics
    - chosen model variant
    - whether hedge gating fired
  - Add a prod-adapter contract test that validates the future adapter only needs:
    - round list/detail
    - simulate query
    - submit tensor
    - analysis fetch
  - Do **not** implement full prod auth/networking; only validate that the interface is sufficient.

  **Must NOT do**:
  - Do not ship real prod credentials or auth code.
  - Do not skip ablations; they are required to prove the particle policy earns its complexity.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: closes the loop on generalization and future portability.
  - Skills: `[]`
  - Omitted: `[quality-check]` — final verification wave will cover repo-wide quality.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Final Verification | Blocked By: 10,11

  **References**:
  - Replay test pattern: `benchmark/tests/replay/test_historical_rounds.py:10-14` — historical/replay validation style.
  - Quickstart/prod contract: `docs/quickstart.md:27-98` — only four capabilities need to be mirrored later.
  - Endpoint contract: `docs/endpoint.md:24-33` — prod-facing methods to preserve.

  **Acceptance Criteria**:
  - [ ] `cd benchmark && pytest -q tests/solver/test_replay_validation.py tests/solver/test_prod_adapter_contract.py` exits `0`.
  - [ ] `cd benchmark && python -m astar_twin.solver.eval.run_replay_validation --round-id test-round-001 --output ../.sisyphus/evidence/task-12-replay-validation.json` exits `0`.
  - [ ] The JSON artifact records candidate, ablation variants, and the selected winner; the winner must be the highest-mean configuration among tested variants.

  **QA Scenarios**:
  ```
  Scenario: Replay validation and ablations complete successfully
    Tool: Bash
    Steps: Run `cd benchmark && pytest -q tests/solver/test_replay_validation.py tests/solver/test_prod_adapter_contract.py && python -m astar_twin.solver.eval.run_replay_validation --round-id test-round-001 --output ../.sisyphus/evidence/task-12-replay-validation.json`
    Expected: Exit code 0; replay artifact exists; ablation metrics and selected winner are recorded.
    Evidence: .sisyphus/evidence/task-12-replay-validation.json

  Scenario: Prod-switch contract fails on missing adapter method
    Tool: Bash
    Steps: Run the negative contract test in `test_prod_adapter_contract.py` with one required method removed from the stub adapter.
    Expected: Exit code 0; the contract test fails the stub as expected, proving the seam is enforced.
    Evidence: .sisyphus/evidence/task-12-replay-validation-error.txt
  ```

  **Commit**: YES | Message: `test(solver): validate replay performance and prod adapter seam` | Files: `benchmark/src/astar_twin/solver/eval/run_replay_validation.py`, `benchmark/tests/solver/test_replay_validation.py`, `benchmark/tests/solver/test_prod_adapter_contract.py`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit after each numbered task.
- Keep the sequence strict because tasks 6-12 depend on behavior locked earlier.
- Do not squash until the benchmark suite and replay validation are green.
- If task 11 shows the hedge is unnecessary, keep the hedge code but leave it gated off by default.

## Success Criteria
- The solver’s **mean score across 10 repeated benchmark runs** beats both baselines.
- The solver’s **minimum score across 10 repeated benchmark runs** is not worse than the fixed-coverage baseline minimum.
- Query allocation logs prove the policy obeys the `10 + 30 + 10` schedule and never exceeds the 50-query cap.
- Output tensors for all 5 seeds are always safe, normalized, and shape-correct.
- Replay validation selects the particle solver as winner unless the hedge gate materially improves mean score.
- Switching to prod later requires only a new adapter implementation, not changes to posterior, policy, or prediction modules.
