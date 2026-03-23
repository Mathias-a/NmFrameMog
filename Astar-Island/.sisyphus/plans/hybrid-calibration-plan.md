# Hybrid Calibration + Confidence-Gated Solver Plan

## TL;DR
> **Summary**: Implement a production solver that learns shared round-level hidden parameters from live queries, but falls back toward a strong structural prior whenever posterior confidence is weak. The first milestone is not “better SMC”; it is a **hybrid solver** that combines the existing particle pipeline with a reusable heuristic anchor derived from the current `filter_baseline` logic.
>
> **Primary Deliverables**:
> 1. Reusable structural-anchor module shared by the benchmark `filter_baseline` strategy and live solver hedge
> 2. Confidence model for posterior trust / fallback decisions
> 3. Bootstrap query redesign for parameter-discriminative early probing
> 4. Likelihood + prediction-cost reductions (cache + adaptive MC allocation)
> 5. Replay/diagnostic outputs that measure transfer robustness, not just default-prior benchmark mean
>
> **Effort**: Large
>
> **Parallel**: YES — 4 waves after foundation extraction
>
> **Critical Path**: Task 1 (anchor extraction) → Task 2 (hybrid hedge) → Task 3 (confidence model) → Task 4 (pipeline integration) → Task 5 (bootstrap calibration queries) → Task 6 (diagnostics) → Task 7 (likelihood caching) → Task 8 (adaptive MC allocation) → Task 9 (evaluation)

---

## Context

### What the benchmark results actually mean
- All 15 stored “real” fixtures currently load as `params_source=DEFAULT_PRIOR` in practice, and `BenchmarkAdapter` explicitly warns those fixtures are for **ground-truth generation only, not calibration**.
- The current multi-round benchmark therefore mostly rewards strategies that are robust under **default simulator parameters** and strong structural priors.
- That is why `filter_baseline` wins the average benchmark (~32.3), while `mc_oracle` and `smc_particle_filter` are highly volatile (~24).
- This does **not** invalidate calibration for the live challenge. The live challenge still has the stronger exploitable signal: **hidden parameters are shared across all 5 seeds and we have 50 total queries**.

### Strategic conclusion
The next implementation target is a **hybrid live solver**:
1. Learn a shared posterior from real observations across seeds.
2. Score early queries by parameter discrimination, not just dynamic-cell coverage.
3. Blend or switch toward a strong heuristic anchor when posterior confidence is weak.

### Evidence already established
- `benchmark/src/astar_twin/solver/adapters/benchmark.py` warns against calibrating on `DEFAULT_PRIOR` fixtures.
- `benchmark/src/astar_twin/solver/pipeline.py` already has the production orchestration surface we should upgrade.
- `benchmark/src/astar_twin/solver/policy/allocator.py` already contains adaptive scoring and contradiction machinery.
- `benchmark/src/astar_twin/solver/predict/hedge.py` already supports blending, but blends against `fixed_coverage`, not the stronger `filter_baseline`-style prior.
- An oracle chooser between `filter_baseline` and `smc_particle_filter` would score ~40.5 mean on the 15-round benchmark, far above the current standalone best (~32.3). The biggest short-term gap is therefore **model selection / trust**, not one isolated model improvement.

---

## Work Objectives

### Core Objective
Build a confidence-gated hybrid solver that improves live challenge readiness while preserving current benchmark robustness.

### Secondary Objectives
1. Reuse the existing `filter_baseline` logic as a general-purpose structural anchor instead of leaving it isolated inside a benchmark-only strategy.
2. Make bootstrap queries explicitly identify shared hidden parameters earlier.
3. Reduce cost/noise in posterior updates so calibration becomes stable enough to trust.
4. Add diagnostics that reveal when the solver should trust the posterior versus the heuristic anchor.

### Out of Scope (for this plan)
- Full emulator / learned surrogate training pipeline
- Large offline training jobs
- New benchmark fixture generation with true server parameters
- Replacing the particle posterior with an entirely new inference family

---

## Deliverables

1. `benchmark/src/astar_twin/solver/predict/structural_anchor.py`
   - Shared structural-prior tensor builder for the live solver and benchmark strategy reuse.

2. `benchmark/src/astar_twin/solver/predict/hedge.py`
   - Upgraded hedge that can blend particle predictions with the structural anchor instead of only `fixed_coverage`.

3. `benchmark/src/astar_twin/strategies/filter_baseline/strategy.py`
   - Refactored to reuse the shared structural-anchor implementation instead of owning unique heuristic logic.

4. `benchmark/src/astar_twin/solver/inference/confidence.py` (new)
   - Confidence dataclasses + functions for round/seed-level trust scoring.

5. `benchmark/src/astar_twin/solver/pipeline.py`
   - Integration of anchor generation, confidence computation, and hybrid finalization.

6. `benchmark/src/astar_twin/solver/policy/allocator.py`
   - Bootstrap scoring redesigned for parameter-discriminative early probes.

7. `benchmark/src/astar_twin/solver/inference/likelihood.py`
   - In-memory cache for repeated likelihood work.

8. `benchmark/src/astar_twin/solver/predict/posterior_mc.py`
   - Adaptive run allocation for final prediction MC.

9. `benchmark/src/astar_twin/solver/eval/run_replay_validation.py`
   - Richer reporting around confidence, hedge activation, and disagreement.

10. `benchmark/src/astar_twin/solver/eval/run_diagnostic_suite.py`
    - Robustness checks under `prior_spread > 0` integrated into summary expectations.

11. Tests:
    - `benchmark/tests/solver/test_structural_anchor.py`
    - `benchmark/tests/solver/test_confidence.py`
    - `benchmark/tests/solver/test_allocator.py` updates
    - `benchmark/tests/solver/test_posterior.py` updates for likelihood-cache behavior
    - `benchmark/tests/solver/test_posterior_mc.py` updates
    - `benchmark/tests/solver/test_replay_validation.py` updates
    - New/updated integration test covering hybrid solver path

---

## Definition of Done

```bash
# Solver/unit/integration tests pass
cd benchmark && uv run pytest tests/solver tests/harness tests/strategies tests/test_integration.py -v

# Repository quality gates pass
cd benchmark && uv run ruff check . && uv run ruff format --check . && uv run mypy

# Replay validation runs end-to-end and reports confidence/hedge outputs
cd benchmark && uv run python -m astar_twin.solver.eval.run_replay_validation \
  --round-id cc5442dd-bc5d-418b-911b-7eb960cb0390 \
  --output results/replay-hybrid.json

# Diagnostic suite runs with prior spread sensitivity
cd benchmark && uv run python -m astar_twin.solver.eval.run_diagnostic_suite \
  --data-dir data \
  --output results/diagnostic-hybrid.json \
  --prior-spread 0.2
```

### Functional done criteria
- Hybrid solver path exists inside `solver/pipeline.py` and is exercised by tests.
- Structural anchor is reused by both the solver and `filter_baseline` strategy.
- Hedge decisions depend on explicit confidence metrics, not only mean-score comparison against `fixed_coverage`.
- Bootstrap query selection includes at least one parameter-discriminative scoring component.
- Likelihood caching and adaptive MC allocation are both implemented and covered by tests.
- Diagnostics expose confidence/hedge behavior and remain runnable under `prior_spread > 0`.

### Performance/behavior done criteria
- On the current 15-round default-prior benchmark, the hybrid path is **not worse than `filter_baseline` by more than 1.0 weighted score point**.
- On replay validation and/or prior-spread sensitivity runs, the hybrid path beats pure particle on disagreement-heavy cases.
- Hedge activation is explainable from emitted confidence metrics (ESS, top-mass, disagreement, entropy).

---

## Must Have

- Shared structural anchor extracted from the current `filter_baseline` logic
- Confidence scoring module with deterministic outputs
- Per-seed or round-level hybrid blending decisions
- Bootstrap query ranking that prefers parameter discrimination early
- Likelihood cache keyed deterministically
- Adaptive MC allocation in `posterior_mc.py`
- Replay/diagnostic outputs that show why hybrid mode activated

## Must NOT Have

- No benchmark-only shortcut that hardcodes round IDs or benchmark-specific branching
- No direct import of benchmark strategy classes from solver pipeline code
- No mutation of `SimulationParams` defaults
- No edits to `engine/`, `phases/`, or `mc/` source directories
- No surrogate/emulator training work in this milestone
- No removal of existing replay/diagnostic outputs

---

## Verification Strategy

> ZERO HUMAN INTERVENTION — all verification is agent-executed.

- Use unit tests for anchor/confidence/cache behavior.
- Use replay validation for hybrid decision correctness.
- Use diagnostic suite with `prior_spread > 0` as the robustness gate.
- Use the 15-round multi-fixture benchmark as a regression floor, **not** as the sole optimization target.

Evidence to collect during implementation:
- `.sisyphus/evidence/hybrid-replay.txt`
- `.sisyphus/evidence/hybrid-diagnostics.txt`
- `.sisyphus/evidence/hybrid-benchmark.txt`

---

## Execution Strategy

### Wave 1 — Extract the heuristic anchor

#### Task 1. Create reusable structural-anchor module

**Files**:
- `benchmark/src/astar_twin/solver/predict/structural_anchor.py` (new)
- `benchmark/src/astar_twin/strategies/filter_baseline/strategy.py`

**What to do**:
1. Read the current `filter_baseline` strategy and identify the pure tensor-building logic.
2. Move that logic into a shared function (or small set of functions) in `solver/predict/structural_anchor.py`.
3. Keep the API deterministic and independent of file I/O.
4. Refactor `filter_baseline` to call the shared helper.

**Key API to add**:
```python
def build_structural_anchor(
    initial_state: InitialState,
    height: int,
    width: int,
) -> NDArray[np.float64]: ...

def build_structural_anchors(
    initial_states: list[InitialState],
    height: int,
    width: int,
) -> list[NDArray[np.float64]]: ...
```

**Acceptance criteria**:
- `filter_baseline` output remains unchanged on current tests.
- New anchor helper has dedicated tests for shape, normalization, and static-cell behavior.

**QA scenario**:
```bash
cd benchmark && uv run pytest tests/strategies/test_filter_baseline.py tests/solver/test_structural_anchor.py -v
```
Expected:
- existing `filter_baseline` tests stay green
- new structural-anchor tests pass

**Why first**:
This avoids solver code importing strategy classes and creates the fallback primitive we need for the hybrid path.

---

### Wave 2 — Build the hybrid decision layer

#### Task 2. Upgrade hedge to use structural anchor

**Files**:
- `benchmark/src/astar_twin/solver/predict/hedge.py`

**What to do**:
1. Replace the current `fixed_coverage`-specific assumptions with a generic “anchor tensor” interface.
2. Allow per-seed blending weights instead of one all-round constant gate.
3. Keep finalization through `finalize_tensor()`.

**Key API changes**:
```python
def should_hedge_from_confidence(confidence: PosteriorConfidence) -> bool: ...

def compute_blend_weight(confidence: PosteriorConfidence) -> float: ...

def apply_anchor_hedge(
    particle_tensors: list[NDArray[np.float64]],
    anchor_tensors: list[NDArray[np.float64]],
    confidences: list[PosteriorConfidence],
    initial_states: list[InitialState],
    height: int,
    width: int,
) -> list[NDArray[np.float64]]: ...
```

**Acceptance criteria**:
- Blend weight decreases when confidence weakens.
- Existing hedge tests are updated or replaced with confidence-based tests.

**QA scenario**:
```bash
cd benchmark && uv run pytest tests/solver/test_replay_validation.py -v -k "hedge or blend"
```
Expected:
- hedge/blend tests pass
- replay serialization still passes

#### Task 3. Add posterior confidence model

**Files**:
- `benchmark/src/astar_twin/solver/inference/confidence.py` (new)
- `benchmark/src/astar_twin/solver/inference/posterior.py` (read-only integration surface only if needed)

**What to do**:
1. Create a dataclass that summarizes the solver’s trust in posterior-driven predictions.
2. Compute confidence from existing signals:
   - ESS
   - top particle mass
   - disagreement in candidate windows
   - entropy mass in current predictions
   - calibration disagreement from replay mode when available
3. Keep deterministic formulas and explicit thresholds.

**Suggested dataclass**:
```python
@dataclass(frozen=True)
class PosteriorConfidence:
    seed_index: int
    ess: float
    top_particle_mass: float
    disagreement: float
    entropy_mass: float
    confidence_score: float
    recommended_mode: str  # "particle" | "blend" | "anchor"
```

**Acceptance criteria**:
- Tests cover low-ESS / high-top-mass / high-disagreement edge cases.
- Same inputs produce identical `confidence_score` and `recommended_mode`.

**QA scenario**:
```bash
cd benchmark && uv run pytest tests/solver/test_confidence.py -v
```
Expected:
- confidence thresholds and deterministic-mode tests all pass

---

### Wave 3 — Integrate hybrid behavior into the live solver

#### Task 4. Integrate anchor + confidence into `solve()`

**Files**:
- `benchmark/src/astar_twin/solver/pipeline.py`

**What to do**:
1. Build anchor tensors immediately after loading `initial_states`.
2. Keep particle inference as-is initially, but compute confidence after bootstrap/adaptive/reserve updates.
3. At final prediction time, decide per seed whether to:
   - trust particle prediction,
   - blend particle with anchor,
   - or use anchor-only fallback.
4. Extend `SolveResult` with confidence/hedge telemetry.

**Acceptance criteria**:
- `SolveResult` includes per-seed confidence or hedge metadata.
- Hybrid finalization path is exercised in integration tests.

**QA scenario**:
```bash
cd benchmark && uv run pytest tests/solver/test_pipeline.py tests/test_integration.py -v -k "hybrid or confidence or hedge"
```
Expected:
- pipeline and integration tests confirm hybrid path is exercised
- `SolveResult` metadata assertions pass

#### Task 5. Redesign bootstrap queries for parameter identification

**Files**:
- `benchmark/src/astar_twin/solver/policy/allocator.py`
- `benchmark/src/astar_twin/solver/policy/hotspots.py` (only if candidate metadata must expand)
- `benchmark/src/astar_twin/solver/pipeline.py`

**What to do**:
1. Preserve the current broad phase structure (bootstrap/adaptive/reserve) to minimize blast radius.
2. Change bootstrap selection from category-priority only to discriminative scoring:
   - start with one guaranteed bootstrap query per seed,
   - use remaining bootstrap slots for globally highest parameter-disagreement windows.
3. Add a calibration-oriented bootstrap scorer that favors windows where top particles predict meaningfully different observations.

**Suggested function additions**:
```python
def score_bootstrap_calibration_candidate(...): ...
def plan_calibration_bootstrap_queries(...): ...
```

**Acceptance criteria**:
- Bootstrap plan remains deterministic.
- Query plan tests confirm minimum per-seed coverage plus globally discriminative extras.

**QA scenario**:
```bash
cd benchmark && uv run pytest tests/solver/test_allocator.py -v -k "bootstrap or calibration"
```
Expected:
- allocator tests show deterministic bootstrap planning
- per-seed minimum coverage and discriminative extras are asserted

---

### Wave 4 — Reduce inference noise and runtime cost

#### Task 6. Add likelihood cache

**Files**:
- `benchmark/src/astar_twin/solver/inference/likelihood.py`

**What to do**:
1. Add an internal cache for repeated simulation-derived likelihood components.
2. Key cache entries by deterministic inputs only:
   - hashed inferred params
   - viewport geometry
   - initial-state identity / seed index if needed
   - base seed and inner-run count if required to preserve determinism
3. Keep cache scope local to a solve/replay run.

**Acceptance criteria**:
- Cache hits are covered by unit tests.
- Cached and uncached outputs are bit-for-bit identical.
- No hidden global mutable state survives across tests.

**QA scenario**:
```bash
cd benchmark && uv run pytest tests/solver/test_posterior.py -v -k "cache or deterministic"
```
Expected:
- cache-hit and cache-equivalence tests pass
- deterministic replay tests remain green

#### Task 7. Add adaptive MC allocation for final prediction

**Files**:
- `benchmark/src/astar_twin/solver/predict/posterior_mc.py`

**What to do**:
1. Replace purely static/proportional run allocation with a two-stage process:
   - cheap probe runs per selected particle,
   - allocate remaining runs toward particles with highest posterior-weighted uncertainty.
2. Keep deterministic seed derivation.
3. Preserve current fallbacks if runtime is tight.

**Acceptance criteria**:
- `PredictionMetrics` reports probe-vs-final allocation.
- Tests confirm deterministic allocation and valid totals.

**QA scenario**:
```bash
cd benchmark && uv run pytest tests/solver/test_posterior_mc.py -v
```
Expected:
- posterior MC tests pass
- metrics assertions cover probe/final allocation totals

---

### Wave 5 — Improve evaluation so it measures the right thing

#### Task 8. Extend replay validation outputs

**Files**:
- `benchmark/src/astar_twin/solver/eval/run_replay_validation.py`

**What to do**:
1. Add confidence summaries, hedge activations, and hybrid-vs-anchor-vs-particle comparisons.
2. Emit enough detail to answer: “Why did the solver trust the posterior here?”
3. Report per-seed mode decisions.

**Acceptance criteria**:
- Replay JSON includes hedge mode and confidence fields.
- Tests verify serialization and winner selection still work.

**QA scenario**:
```bash
cd benchmark && uv run pytest tests/solver/test_replay_validation.py -v
```
Expected:
- replay validation tests pass
- JSON output contains confidence/hedge fields

#### Task 9. Tighten diagnostic suite for robustness checks

**Files**:
- `benchmark/src/astar_twin/solver/eval/run_diagnostic_suite.py`
- `benchmark/src/astar_twin/harness/diagnostics.py`

**What to do**:
1. Make `prior_spread > 0` runs part of the expected evaluation workflow.
2. Surface whether hybrid mode reduces worst-class loss or seed collapse under perturbed parameters.
3. Add summary fields for confidence/hedge statistics if needed.

**Acceptance criteria**:
- Diagnostic suite remains runnable with and without prior spread.
- Output summary makes hybrid robustness visible without manual inspection.

**QA scenario**:
```bash
cd benchmark && uv run pytest tests/solver/test_benchmark_characterization.py tests/solver/test_multi_fixture_suite.py -v && uv run python -m astar_twin.solver.eval.run_diagnostic_suite --data-dir data --output results/diagnostic-hybrid.json --prior-spread 0.2
```
Expected:
- suite tests pass
- diagnostic command exits 0 and writes `results/diagnostic-hybrid.json`
- output summary includes hybrid robustness fields

---

## Dependency Matrix

| Task | Blocks | Blocked By |
|------|--------|------------|
| 1: Structural anchor extraction | 2, 4 | — |
| 2: Hedge upgrade | 4, 8 | 1, 3 |
| 3: Confidence model | 2, 4, 8 | — |
| 4: Pipeline hybrid integration | 5, 8, 9 | 1, 2, 3 |
| 5: Bootstrap calibration queries | 9 | 4 |
| 6: Likelihood cache | 9 | 4 |
| 7: Adaptive MC allocation | 9 | 4 |
| 8: Replay validation expansion | 9 | 2, 3, 4 |
| 9: Evaluation + diagnostics | — | 4, 5, 6, 7, 8 |

---

## Recommended Implementation Order

1. Extract anchor from `filter_baseline`
2. Add confidence model
3. Upgrade hedge to use anchor + confidence
4. Integrate hybrid behavior in `pipeline.solve`
5. Redesign bootstrap query scoring for calibration
6. Add likelihood cache
7. Add adaptive MC allocation
8. Extend replay/diagnostics
9. Run full verification and compare against current baselines

---

## Evaluation Matrix

### Required comparisons
1. `filter_baseline`
2. `particle_no_hedge`
3. `particle + fixed_coverage hedge` (current)
4. `particle + structural anchor hybrid` (new)

### Required scenarios
1. Current 15-round default-prior benchmark (regression floor)
2. Replay validation on at least one representative strong-particle round and one strong-heuristic round
3. Diagnostic suite with `--prior-spread 0.2`

### Success thresholds
- Hybrid weighted mean on current rounds: **>= 31.4** (within 1.0 of current `filter_baseline` floor)
- Hybrid beats pure particle on disagreement-heavy rounds / replays
- Hybrid shows smaller degradation than pure particle under `prior_spread > 0`

---

## Risks and Mitigations

### Risk 1: Posterior becomes confidently wrong
**Mitigation**: confidence gating must treat low ESS, high top-mass, and high disagreement as reasons to blend away from the posterior.

### Risk 2: Query policy still chooses interesting but non-discriminative windows
**Mitigation**: bootstrap scoring must explicitly reward parameter disagreement, not just entropy or settlement density.

### Risk 3: Cache changes break determinism
**Mitigation**: cache keys include all deterministic inputs; tests compare cached vs uncached outputs exactly.

### Risk 4: Hybrid logic improves live realism but looks worse on default-prior benchmark
**Mitigation**: use replay + prior-spread diagnostics as the primary transfer guard, while holding benchmark regression within a capped tolerance.

---

## Explicit Non-Goals for This Milestone

- No neural surrogate training
- No full Bayesian experimental design library integration
- No replacement of the particle posterior with ABC / SBI / GP calibration stack
- No benchmark fixture backfill or remote API data refresh work

Those are follow-up avenues only if this hybrid-calibration milestone fails to produce reliable posterior sharpening.

---

## Final Verification Wave

- [ ] `uv run pytest tests/solver tests/harness tests/strategies tests/test_integration.py -v`
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy`
- [ ] `uv run python -m astar_twin.solver.eval.run_replay_validation --round-id cc5442dd-bc5d-418b-911b-7eb960cb0390 --output results/replay-hybrid.json`
- [ ] `uv run python -m astar_twin.solver.eval.run_diagnostic_suite --data-dir data --output results/diagnostic-hybrid.json --prior-spread 0.2`
- [ ] `uv run python -m astar_twin.solver.eval.run_multi_fixture_suite --data-dir data --output results/multi-hybrid.json --repeats 1`

---

## Bottom Line

Do **not** spend the next cycle trying to make the particle model win the current default-prior benchmark in isolation.

Instead, implement a solver that can:
1. learn shared round-level parameters from real queries,
2. recognize when that learning is unreliable,
3. and safely fall back toward a structural prior that already wins the robustness benchmark.

That is the shortest path to turning the current evidence into a production-ready improvement.
