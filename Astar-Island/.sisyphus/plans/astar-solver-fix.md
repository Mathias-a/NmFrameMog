# Astar-Island Solver — Verification Fix Plan

## TL;DR
> **Summary**: Fix all issues identified by the F1-F4 verification wave — critical correctness bugs, plan compliance gaps, and code quality violations.
> **Deliverables**: All 4 verification agents (F1-F4) approve on re-run.
> **Effort**: Medium (12 tasks across 3 waves)
> **Parallel**: YES — 3 waves
> **Critical Path**: Wave 1 (correctness) → Wave 2 (compliance) → Wave 3 (cleanup) → Re-run F1-F4

## Context

### Original Request
After building the Astar-Island solver (12 tasks, 121 tests passing), the Final Verification Wave (F1-F4) rejected the implementation. The user chose "Create fix plan" to address all issues.

### Verification Rejection Summary

**F1: Plan Compliance Audit (Oracle)** — REJECT
- 7 of 12 tasks rated PARTIAL
- `current_prediction` never updated (entropy-driven selection inert)
- Viewport sizing rules (15→10→5) not implemented
- `dump_prediction_stats` CLI missing
- Disagreement is weight-proxy, not cellwise top-2 argmax
- Reserve released in one batch, not two
- `fixed_coverage_baseline` isn't the planned 50-query sweep

**F2: Code Quality Review** — REJECT
- 202 ruff lint errors (74 auto-fixable)
- `Particle.params: dict[str, Any]` type leak
- Duplicated `_resilient_run_batch` in two eval scripts
- Misleading docstrings
- Runtime warnings in `scoring/kl.py` (divide by zero)
- Numerical safety: `predict_seed()` divides by zero when `k==0`

**F3: Real Manual QA** — REJECT (couldn't execute — read-only constraint)
**F4: Scope Fidelity Check** — REJECT (soft — eval coupling caveat only)

### Metis Review Findings (incorporated)
- `safe_prediction` tolerance: `>= 0.01 - 1e-9` (practical floating-point tolerance)
- Reserve batching: `5 + remainder` (degrade gracefully)
- `current_prediction` scope: per-seed — `dict[int, NDArray]`
- Empty-particle fallback: return uniform tensor via `uniform_baseline`
- `dump_prediction_stats`: CLI + library function, fixture-derived dimensions
- Disagreement: true cellwise top-2 argmax using lightweight inner MC (2 runs per particle for top-2 only)

## Work Objectives

### Core Objective
Make all 4 verification agents (F1-F4) approve on re-run.

### Deliverables
- All critical correctness bugs fixed (safe_prediction, empty-particle, KL warnings, live prediction)
- All plan compliance gaps closed (viewport sizing, disagreement, reserve batching, baseline, stats CLI)
- 202 ruff errors resolved, code cleaned up
- Full test suite passing with new edge-case coverage

### Definition of Done
- `uv run ruff check src/astar_twin/solver/ tests/solver/` → 0 errors
- `uv run ruff format --check src/astar_twin/solver/ tests/solver/` → no reformats needed
- `uv run pytest -q tests/solver/` → all pass, 0 failures
- F1 (Plan Compliance) → APPROVE
- F2 (Code Quality) → APPROVE
- F3 (Manual QA) → APPROVE (must actually execute tests + CLI)
- F4 (Scope Fidelity) → APPROVE

### Must Have
- Zero-probability safety: `safe_prediction` guarantees min ≥ 0.01 after renormalization
- Empty-particle guard in `predict_seed()` — returns uniform, never crashes
- KL scoring produces no runtime warnings
- `current_prediction` is live during adaptive phase (updated after bootstrap, after each adaptive batch)
- Viewport sizing follows 15→10→5 rules from original plan
- True cellwise top-2 argmax disagreement (not weight proxy)
- Reserve released in two batches of 5 (or 5+remainder)
- `dump_prediction_stats` CLI callable via `uv run python -m astar_twin.solver.eval.dump_prediction_stats`

### Must NOT Have
- Changes to files in `benchmark/src/astar_twin/engine/`, `benchmark/src/astar_twin/phases/`, or `benchmark/src/astar_twin/mc/`
- Changes to `SimulationParams` default field values
- New external dependencies
- Scope creep beyond fixing verification failures
- Any `# type: ignore` annotations to paper over real type issues

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: Tests-after — add targeted tests alongside each fix
- QA policy: Every task has agent-executed scenarios
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy

### Parallel Execution Waves

**Wave 1: Critical Correctness** (4 tasks — foundational safety fixes)
- Categories: `quick` × 3, `unspecified-low` × 1

**Wave 2: Plan Compliance** (5 tasks — behavior alignment with original plan)
- Categories: `unspecified-high` × 3, `quick` × 2

**Wave 3: Cleanup** (3 tasks — ruff, dedup, typing)
- Categories: `quick` × 2, `unspecified-low` × 1

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 (safe_prediction) | — | 3, 5, 8 |
| 2 (empty-particle) | — | 4, 8 |
| 3 (KL warnings) | 1 | 8 |
| 4 (live prediction) | 2 | 6, 7, 8 |
| 5 (viewport sizing) | — | 8 |
| 6 (disagreement) | 4 | 8 |
| 7 (reserve batching) | 4 | 8 |
| 8 (baseline sweep) | 1 | — |
| 9 (stats CLI) | — | — |
| 10 (ruff auto-fix) | 1–9 | 11, 12 |
| 11 (dedup + docstrings) | 10 | 12 |
| 12 (manual ruff + typing) | 11 | — |

### Agent Dispatch Summary
| Wave | Tasks | Categories |
|------|-------|-----------|
| 1 | 4 | quick × 3, unspecified-low × 1 |
| 2 | 5 | unspecified-high × 3, quick × 2 |
| 3 | 3 | quick × 2, unspecified-low × 1 |

## TODOs

- [x] 1. Fix `safe_prediction` to guarantee min ≥ 0.01 after renormalization

  **What to do**:
  In `benchmark/src/astar_twin/scoring/safe_prediction.py`, replace the single floor+renormalize pass with an iterative approach:
  ```python
  def safe_prediction(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
      result = tensor.astype(np.float64, copy=True)
      for _ in range(10):  # max 10 iterations (converges in 2-3)
          result = np.maximum(result, 0.01)
          sums = np.sum(result, axis=2, keepdims=True)
          result = result / sums
          if result.min() >= 0.01 - 1e-9:
              break
      return result
  ```
  Add tests in `benchmark/tests/solver/test_safe_prediction.py` (new file):
  - Test that output min ≥ 0.01 - 1e-9 for adversarial input (e.g., one class at 0.99, five at 0.002)
  - Test that output sums to 1.0 per cell
  - Test that already-safe input is unchanged (within tolerance)
  - Test with all-zero input (degenerate case — should produce uniform)

  **Must NOT do**: Don't change the function signature. Don't change the floor value from 0.01.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Single file edit + new test file, straightforward logic
  - Skills: [] — no special skills needed
  - Omitted: [`quality-check`] — Wave 3 handles lint

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [3, 5, 8] | Blocked By: []

  **References**:
  - Source: `benchmark/src/astar_twin/scoring/safe_prediction.py:1-10` — current implementation to replace
  - Contract: AGENTS.md — "NEVER output probability 0.0 — floor at 0.01, renormalize"
  - Scoring: `benchmark/src/astar_twin/scoring/kl.py:9-21` — downstream consumer

  **Acceptance Criteria**:
  - [ ] `safe_prediction(adversarial_tensor).min() >= 0.01 - 1e-9` where adversarial has one class at 0.99, five at 0.002
  - [ ] `np.allclose(safe_prediction(tensor).sum(axis=2), 1.0)` for any valid input
  - [ ] `uv run pytest tests/solver/test_safe_prediction.py -q` → all pass

  **QA Scenarios**:
  ```
  Scenario: Adversarial input preserves floor
    Tool: Bash
    Steps: uv run pytest tests/solver/test_safe_prediction.py::test_adversarial_floor -v
    Expected: PASSED — minimum probability ≥ 0.01 - 1e-9
    Evidence: .sisyphus/evidence/task-1-safe-prediction.txt

  Scenario: Degenerate all-zero input
    Tool: Bash
    Steps: uv run pytest tests/solver/test_safe_prediction.py::test_all_zero_input -v
    Expected: PASSED — produces uniform 1/6 per class
    Evidence: .sisyphus/evidence/task-1-safe-prediction-zero.txt
  ```

  **Commit**: NO — committed with Wave 1 batch

---

- [x] 2. Add empty-particle and zero-`top_k` guards in `predict_seed()`

  **What to do**:
  In `benchmark/src/astar_twin/solver/predict/posterior_mc.py`, add guard at the top of `predict_seed()` (after line 133):
  ```python
  # Guard: if no particles available, return uniform baseline
  k = min(top_k, len(posterior.particles))
  if k == 0:
      from astar_twin.solver.baselines import uniform_baseline
      uniform = uniform_baseline(map_height, map_width)
      metrics = PredictionMetrics(
          seed_index=seed_index, n_particles_used=0,
          total_sims=0, fallback_used=True, runs_per_particle=[],
      )
      return uniform, metrics
  ```
  Also fix line 145 — the `else` branch `selected_weights = [1.0 / k] * k` is already guarded by the above, but add explicit safety: if `w_sum == 0 and k == 0`, the early return handles it.

  In `benchmark/src/astar_twin/solver/inference/posterior.py`, add guards to `ess`, `top_particle_mass`, `normalized_weights`, and `top_k_indices` properties for empty particle lists:
  ```python
  @property
  def ess(self) -> float:
      if not self.particles:
          return 0.0
      ...
  ```
  Same pattern for `top_particle_mass` (return 0.0), `normalized_weights` (return []), `top_k_indices` (return []).

  Add tests in existing `benchmark/tests/solver/test_posterior_mc.py`:
  - Test `predict_seed()` with empty posterior → returns uniform tensor of correct shape
  - Test `predict_seed()` with `top_k=0` → returns uniform tensor
  
  Add tests in existing `benchmark/tests/solver/test_posterior.py`:
  - Test `PosteriorState` with empty particles → `ess` returns 0.0, `normalized_weights` returns []

  **Must NOT do**: Don't change the default `top_k` value. Don't import `uniform_baseline` at module level (circular import risk — use lazy import).

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Guard clauses in two files + test additions
  - Skills: [] — no special skills needed
  - Omitted: [`quality-check`] — Wave 3

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [4, 8] | Blocked By: []

  **References**:
  - Source: `benchmark/src/astar_twin/solver/predict/posterior_mc.py:107-179` — `predict_seed()` function
  - Source: `benchmark/src/astar_twin/solver/predict/posterior_mc.py:145` — the `1.0 / k` divide-by-zero line
  - Source: `benchmark/src/astar_twin/solver/inference/posterior.py:22-61` — `PosteriorState` class
  - Baseline: `benchmark/src/astar_twin/solver/baselines.py:16-20` — `uniform_baseline()` to import
  - Tests: `benchmark/tests/solver/test_posterior_mc.py` — existing test file to extend
  - Tests: `benchmark/tests/solver/test_posterior.py` — existing test file to extend

  **Acceptance Criteria**:
  - [ ] `predict_seed()` with empty `PosteriorState(particles=[])` returns `(tensor, metrics)` where `tensor.shape == (H, W, 6)` and `metrics.fallback_used == True`
  - [ ] `PosteriorState(particles=[]).ess == 0.0`
  - [ ] `PosteriorState(particles=[]).normalized_weights() == []`
  - [ ] `uv run pytest tests/solver/test_posterior_mc.py tests/solver/test_posterior.py -q` → all pass

  **QA Scenarios**:
  ```
  Scenario: Empty posterior returns uniform
    Tool: Bash
    Steps: uv run pytest tests/solver/test_posterior_mc.py::test_predict_seed_empty_posterior -v
    Expected: PASSED — shape correct, fallback_used=True
    Evidence: .sisyphus/evidence/task-2-empty-particle.txt

  Scenario: PosteriorState empty properties
    Tool: Bash
    Steps: uv run pytest tests/solver/test_posterior.py::test_empty_posterior_properties -v
    Expected: PASSED — ess=0.0, normalized_weights=[], top_k_indices=[]
    Evidence: .sisyphus/evidence/task-2-empty-posterior.txt
  ```

  **Commit**: NO — committed with Wave 1 batch

---

- [x] 3. Fix KL scoring to eliminate runtime warnings

  **What to do**:
  In `benchmark/src/astar_twin/scoring/kl.py`, the current implementation already uses `np.where` guards (lines 10-17), but runtime warnings still occur. Replace with explicit `np.clip` approach:
  ```python
  def compute_score(ground_truth: NDArray[np.float64], prediction: NDArray[np.float64]) -> float:
      eps = 1e-15
      gt_safe = np.clip(ground_truth, eps, None)
      pred_safe = np.clip(prediction, eps, None)
      
      # Entropy: -sum(gt * log(gt)) — only where gt > 0
      entropy = -np.sum(np.where(ground_truth > 0, ground_truth * np.log(gt_safe), 0.0), axis=2)
      mask = entropy >= 1e-10
      if not np.any(mask):
          return 100.0
  
      # KL: sum(gt * log(gt / pred)) — only where gt > 0
      kl = np.sum(
          np.where(ground_truth > 0, ground_truth * np.log(gt_safe / pred_safe), 0.0), axis=2
      )
      weighted_kl = float(np.sum(entropy[mask] * kl[mask]) / np.sum(entropy[mask]))
      score = 100.0 * math.exp(-3.0 * weighted_kl)
      return max(0.0, min(100.0, score))
  ```

  Add test in new file `benchmark/tests/solver/test_kl_safety.py`:
  - Test with ground truth containing zeros → no RuntimeWarning
  - Test with prediction containing near-zeros → no RuntimeWarning
  - Test perfect prediction → score 100.0
  - Test uniform prediction vs deterministic ground truth → score in (0, 100)

  **Must NOT do**: Don't change the scoring formula. Don't change the function signature.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Single file fix + new test file
  - Skills: [] — no special skills needed
  - Omitted: [`quality-check`] — Wave 3

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [8] | Blocked By: [1]

  **References**:
  - Source: `benchmark/src/astar_twin/scoring/kl.py:1-21` — full current implementation
  - Contract: AGENTS.md — scoring formula: `100 * exp(-3 * weighted_kl)`
  - Source: `benchmark/src/astar_twin/scoring/safe_prediction.py` — upstream safety (Task 1)

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/solver/test_kl_safety.py -q -W error::RuntimeWarning` → all pass (no warnings promoted to errors)
  - [ ] `compute_score(perfect_gt, perfect_pred) == 100.0` where gt == pred
  - [ ] `uv run pytest tests/solver/test_kl_safety.py -q` → all pass

  **QA Scenarios**:
  ```
  Scenario: No runtime warnings with zeros in ground truth
    Tool: Bash
    Steps: uv run pytest tests/solver/test_kl_safety.py::test_no_warnings_zero_gt -v -W error::RuntimeWarning
    Expected: PASSED — no RuntimeWarning raised
    Evidence: .sisyphus/evidence/task-3-kl-no-warnings.txt

  Scenario: Perfect prediction scores 100
    Tool: Bash
    Steps: uv run pytest tests/solver/test_kl_safety.py::test_perfect_prediction -v
    Expected: PASSED — score == 100.0
    Evidence: .sisyphus/evidence/task-3-kl-perfect.txt
  ```

  **Commit**: NO — committed with Wave 1 batch

---

- [x] 4. Make `current_prediction` live in pipeline adaptive phase

  **What to do**:
  In `benchmark/src/astar_twin/solver/pipeline.py`, update the adaptive phase to maintain a per-seed prediction dict and generate predictions after bootstrap and after each adaptive batch.

  After line 174 (post-bootstrap prune), add:
  ```python
  # Generate initial prediction from bootstrap posterior (lightweight: 16 sims)
  _bootstrap_tensors, _ = predict_all_seeds(
      posterior, initial_states[:n_seeds],
      map_height=height, map_width=width,
      top_k=min(4, len(posterior.particles)),
      sims_per_seed=16,
      base_seed=base_seed + 3000,
  )
  # Per-seed prediction dict for entropy-driven adaptive selection
  seed_predictions: dict[int, NDArray[np.float64]] = {
      i: t for i, t in enumerate(_bootstrap_tensors)
  }
  ```

  Change line 178 from:
  ```python
  current_prediction: NDArray[np.float64] | None = None
  ```
  to remove it entirely (replaced by `seed_predictions`).

  In `select_adaptive_batch()` call (line 184-188), pass the seed predictions. This requires updating `select_adaptive_batch` in `allocator.py` to accept `seed_predictions: dict[int, NDArray[np.float64]] | None = None` instead of a single `current_prediction`. Compute entropy_map per seed inside the scoring loop.

  After each adaptive batch loop body (after line 223, `temper_if_collapsed`), add:
  ```python
  # Update predictions for entropy scoring (lightweight: 16 sims)
  _batch_tensors, _ = predict_all_seeds(
      posterior, initial_states[:n_seeds],
      map_height=height, map_width=width,
      top_k=min(4, len(posterior.particles)),
      sims_per_seed=16,
      base_seed=base_seed + 4000 + batch_num * 100,
  )
  seed_predictions = {i: t for i, t in enumerate(_batch_tensors)}
  ```

  Update `select_adaptive_batch()` in `allocator.py` (line 301-341):
  - Change parameter `current_prediction: NDArray[np.float64] | None = None` to `seed_predictions: dict[int, NDArray[np.float64]] | None = None`
  - Inside the per-seed loop, compute entropy_map from `seed_predictions.get(seed_idx)` if available
  - Update `plan_reserve_queries()` (line 387-413) signature similarly

  Add/update tests in `benchmark/tests/solver/test_pipeline.py`:
  - Test that `solve()` completes without error (existing test should still pass)
  - Test that adaptive queries use non-None entropy by inspecting transcript (utility_score > 0 for adaptive queries)

  Update tests in `benchmark/tests/solver/test_allocator.py`:
  - Update existing `select_adaptive_batch` tests to use the new `seed_predictions` parameter

  **Must NOT do**: Don't increase `sims_per_seed` for mid-pipeline predictions beyond 16 (performance budget). Don't break the existing `solve()` contract.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` — Reason: Multi-file change (pipeline + allocator) but well-scoped
  - Skills: [] — no special skills needed
  - Omitted: [`quality-check`] — Wave 3

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [6, 7, 8] | Blocked By: [2]

  **References**:
  - Source: `benchmark/src/astar_twin/solver/pipeline.py:170-223` — adaptive phase to modify
  - Source: `benchmark/src/astar_twin/solver/policy/allocator.py:301-341` — `select_adaptive_batch()` signature to update
  - Source: `benchmark/src/astar_twin/solver/policy/allocator.py:387-413` — `plan_reserve_queries()` signature to update
  - Source: `benchmark/src/astar_twin/solver/predict/posterior_mc.py:182-228` — `predict_all_seeds()` used for mid-pipeline predictions
  - Tests: `benchmark/tests/solver/test_pipeline.py` — existing pipeline tests
  - Tests: `benchmark/tests/solver/test_allocator.py` — existing allocator tests

  **Acceptance Criteria**:
  - [ ] `solve()` completes without error on test fixture
  - [ ] Adaptive-phase transcript entries have `utility_score > 0` (entropy was used)
  - [ ] `uv run pytest tests/solver/test_pipeline.py tests/solver/test_allocator.py -q` → all pass

  **QA Scenarios**:
  ```
  Scenario: Pipeline completes with live predictions
    Tool: Bash
    Steps: uv run pytest tests/solver/test_pipeline.py -v -k "test_solve"
    Expected: PASSED — solve completes, adaptive queries have utility scores
    Evidence: .sisyphus/evidence/task-4-live-prediction.txt

  Scenario: Allocator uses per-seed entropy maps
    Tool: Bash
    Steps: uv run pytest tests/solver/test_allocator.py -v
    Expected: PASSED — all existing + new tests pass
    Evidence: .sisyphus/evidence/task-4-allocator.txt
  ```

  **Commit**: YES | Message: `fix(solver): critical correctness — safe_prediction, empty-particle, KL, live prediction` | Files: `benchmark/src/astar_twin/scoring/safe_prediction.py`, `benchmark/src/astar_twin/scoring/kl.py`, `benchmark/src/astar_twin/solver/predict/posterior_mc.py`, `benchmark/src/astar_twin/solver/inference/posterior.py`, `benchmark/src/astar_twin/solver/pipeline.py`, `benchmark/src/astar_twin/solver/policy/allocator.py`, `benchmark/tests/solver/test_safe_prediction.py`, `benchmark/tests/solver/test_kl_safety.py`, `benchmark/tests/solver/test_posterior_mc.py`, `benchmark/tests/solver/test_posterior.py`, `benchmark/tests/solver/test_pipeline.py`, `benchmark/tests/solver/test_allocator.py`

---

- [x] 5. Implement 15→10→5 viewport sizing rules in hotspots

  **What to do**:
  In `benchmark/src/astar_twin/solver/policy/hotspots.py`, the current implementation uses `MAX_VIEWPORT` (15) for all candidates. Modify `generate_hotspots()` to compute a bounding box of the feature points for each hotspot category, then apply sizing rules:

  Add a helper function:
  ```python
  def _select_viewport_size(bbox_w: int, bbox_h: int) -> int:
      """Select viewport size based on hotspot bounding box.
      
      Rules from plan:
        - Default: 15x15 (MAX_VIEWPORT)
        - Shrink to 10x10 when bbox < 8x8
        - Shrink to 5x5 only for contradiction-resolution probes
      """
      if bbox_w < 8 and bbox_h < 8:
          return 10
      return MAX_VIEWPORT
  ```

  Update each hotspot generation block to compute bbox of contributing points and call `_select_viewport_size()`:
  - **Coastal**: bbox of `coastal_settlements` points → `_select_viewport_size(bbox_w, bbox_h)` → pass to `_clamp_viewport`
  - **Corridor**: bbox of the two settlements → `_select_viewport_size(bbox_w, bbox_h)` → pass to `_clamp_viewport`
  - **Frontier**: bbox of `frontier` points → `_select_viewport_size(bbox_w, bbox_h)` → pass to `_clamp_viewport`
  - **Reclaim**: bbox of `ruin_positions` → `_select_viewport_size(bbox_w, bbox_h)` → pass to `_clamp_viewport`
  - **Fallback**: keep MAX_VIEWPORT (full coverage intent)

  Also add a `contradiction_probe` parameter to `generate_hotspots()` (default False). When True, force size=5 for all viewports. This is used by `_select_contradiction_queries` in `allocator.py`.

  Add/update tests in `benchmark/tests/solver/test_hotspots.py`:
  - Test that a hotspot with tight bbox (< 8x8) produces 10x10 viewport
  - Test that a hotspot with wide bbox (>= 8x8) produces 15x15 viewport
  - Test contradiction probe mode → all 5x5

  **Must NOT do**: Don't change the category names. Don't change the `ViewportCandidate` dataclass.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Logic changes in hotspot generation + new param threading
  - Skills: [] — no special skills needed
  - Omitted: [`quality-check`] — Wave 3

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8] | Blocked By: []

  **References**:
  - Source: `benchmark/src/astar_twin/solver/policy/hotspots.py:112-216` — `generate_hotspots()` function
  - Source: `benchmark/src/astar_twin/solver/policy/hotspots.py:45-56` — `_clamp_viewport()` helper
  - Constants: `benchmark/src/astar_twin/contracts/types.py` — `MAX_VIEWPORT=15`, `MIN_VIEWPORT=5`
  - Tests: `benchmark/tests/solver/test_hotspots.py` — existing test file to extend

  **Acceptance Criteria**:
  - [ ] Hotspot with 2 settlements 3 cells apart → viewport is 10x10
  - [ ] Hotspot with settlements 12 cells apart → viewport is 15x15
  - [ ] `generate_hotspots(..., contradiction_probe=True)` → all viewports 5x5
  - [ ] `uv run pytest tests/solver/test_hotspots.py -q` → all pass

  **QA Scenarios**:
  ```
  Scenario: Tight bbox produces 10x10
    Tool: Bash
    Steps: uv run pytest tests/solver/test_hotspots.py::test_tight_bbox_10x10 -v
    Expected: PASSED — viewport w==10, h==10
    Evidence: .sisyphus/evidence/task-5-viewport-sizing.txt

  Scenario: Contradiction probe produces 5x5
    Tool: Bash
    Steps: uv run pytest tests/solver/test_hotspots.py::test_contradiction_probe_5x5 -v
    Expected: PASSED — all viewports w==5, h==5
    Evidence: .sisyphus/evidence/task-5-viewport-probe.txt
  ```

  **Commit**: NO — committed with Wave 2 batch

---

- [x] 6. Implement true cellwise top-2 argmax disagreement

  **What to do**:
  In `benchmark/src/astar_twin/solver/policy/allocator.py`, replace `compute_posterior_disagreement()` (lines 234-261) with true cellwise disagreement using lightweight inner MC.

  New implementation:
  ```python
  def compute_posterior_disagreement(
      candidate: ViewportCandidate,
      posterior: PosteriorState,
      initial_state: InitialState,
  ) -> float:
      """Fraction of cells where top-2 particles disagree on argmax class.
      
      Uses lightweight inner MC (2 runs per particle) to get per-particle
      terrain predictions, then checks argmax disagreement in the viewport window.
      """
      if len(posterior.particles) < 2:
          return 0.0
      
      # Get top-2 particles by weight
      top_indices = posterior.top_k_indices(2)
      p1 = posterior.particles[top_indices[0]]
      p2 = posterior.particles[top_indices[1]]
      
      # Run 2 MC sims per particle over the viewport region
      from astar_twin.engine import Simulator
      from astar_twin.mc.runner import MCRunner
      from astar_twin.mc.aggregate import aggregate_runs
      
      h = candidate.h
      w = candidate.w
      
      def _particle_argmax(particle, seed_offset: int) -> NDArray:
          sim = Simulator(params=particle.to_simulation_params())
          runner = MCRunner(sim)
          runs = runner.run_batch(initial_state, n_runs=2, base_seed=seed_offset)
          tensor = aggregate_runs(runs, len(initial_state.grid), len(initial_state.grid[0]))
          # Extract viewport window
          window = tensor[candidate.y:candidate.y+h, candidate.x:candidate.x+w]
          return np.argmax(window, axis=2)
      
      argmax_1 = _particle_argmax(p1, 99000)
      argmax_2 = _particle_argmax(p2, 99100)
      
      total_cells = h * w
      if total_cells == 0:
          return 0.0
      disagreement = float(np.sum(argmax_1 != argmax_2)) / total_cells
      return float(np.clip(disagreement, 0.0, 1.0))
  ```

  Also update `check_argmax_disagreement()` (lines 368-384) to use the same real disagreement instead of the proxy.

  Update docstrings to accurately describe the cellwise approach (not "weight proxy").

  Add/update tests in `benchmark/tests/solver/test_allocator.py`:
  - Test that `compute_posterior_disagreement()` returns value in [0, 1]
  - Test with identical particles → disagreement ≈ 0 (may have stochastic variation from 2 MC runs, so test < 0.3)
  - Test with very different particles → disagreement > 0

  **Must NOT do**: Don't increase MC runs beyond 2 per particle (performance constraint). Don't change the scoring weights or threshold constants.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Algorithm replacement with MC integration
  - Skills: [] — no special skills needed
  - Omitted: [`quality-check`] — Wave 3

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8] | Blocked By: [4]

  **References**:
  - Source: `benchmark/src/astar_twin/solver/policy/allocator.py:234-261` — `compute_posterior_disagreement()` to replace
  - Source: `benchmark/src/astar_twin/solver/policy/allocator.py:368-384` — `check_argmax_disagreement()` to update
  - Engine: `benchmark/src/astar_twin/engine/` — `Simulator` class (import only, don't modify)
  - MC: `benchmark/src/astar_twin/mc/runner.py` — `MCRunner.run_batch()` (import only)
  - MC: `benchmark/src/astar_twin/mc/aggregate.py` — `aggregate_runs()` (import only)
  - Tests: `benchmark/tests/solver/test_allocator.py` — existing test file to extend

  **Acceptance Criteria**:
  - [ ] `compute_posterior_disagreement()` returns float in [0, 1]
  - [ ] With identical particles, disagreement < 0.3 (allowing stochastic MC noise)
  - [ ] Docstring says "cellwise top-2 argmax" not "weight proxy"
  - [ ] `uv run pytest tests/solver/test_allocator.py -q` → all pass

  **QA Scenarios**:
  ```
  Scenario: Disagreement with identical particles is low
    Tool: Bash
    Steps: uv run pytest tests/solver/test_allocator.py::test_disagreement_identical_particles -v
    Expected: PASSED — disagreement < 0.3
    Evidence: .sisyphus/evidence/task-6-disagreement.txt

  Scenario: Disagreement returns valid range
    Tool: Bash
    Steps: uv run pytest tests/solver/test_allocator.py::test_disagreement_range -v
    Expected: PASSED — 0.0 <= disagreement <= 1.0
    Evidence: .sisyphus/evidence/task-6-disagreement-range.txt
  ```

  **Commit**: NO — committed with Wave 2 batch

---

- [x] 7. Implement two-batch reserve release

  **What to do**:
  In `benchmark/src/astar_twin/solver/policy/allocator.py`, modify `plan_reserve_queries()` (lines 387-413) to release reserve in two batches of 5 (or 5+remainder if total reserve ≠ 10).

  In `benchmark/src/astar_twin/solver/pipeline.py`, modify the reserve phase (lines 226-262) to issue reserve queries in two separate batches with posterior update and resampling between them:
  ```python
  # 6. Reserve phase — release in two batches
  transition_phase(alloc)
  contradiction = check_contradiction_triggers(alloc, posterior)
  
  reserve_remaining = min(RESERVE_QUERIES, alloc.queries_remaining)
  if reserve_remaining > 0:
      batch_1_size = min(5, reserve_remaining)
      batch_2_size = reserve_remaining - batch_1_size
      
      # Batch 1
      reserve_batch_1 = plan_reserve_queries(
          alloc, posterior, initial_states[:n_seeds],
          seed_predictions=seed_predictions,
          n_queries=batch_1_size,
      )
      for seed_idx, vp in reserve_batch_1:
          # ... issue query, update posterior, record ...
      
      # Resample between batches
      posterior = resample_if_needed(posterior, ess_threshold=6.0, seed=base_seed + 6000)
      posterior = temper_if_collapsed(posterior)
      
      # Batch 2
      if batch_2_size > 0:
          reserve_batch_2 = plan_reserve_queries(
              alloc, posterior, initial_states[:n_seeds],
              seed_predictions=seed_predictions,
              n_queries=batch_2_size,
          )
          for seed_idx, vp in reserve_batch_2:
              # ... issue query, update posterior, record ...
  ```

  Update `plan_reserve_queries()` signature to accept `n_queries: int` parameter instead of using `RESERVE_QUERIES` constant directly.

  Add/update tests in `benchmark/tests/solver/test_pipeline.py`:
  - Test that reserve phase issues queries in two batches (check transcript for reserve-phase entries)

  **Must NOT do**: Don't change the total reserve budget (10). Don't skip the resample between batches.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Straightforward loop restructuring
  - Skills: [] — no special skills needed
  - Omitted: [`quality-check`] — Wave 3

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8] | Blocked By: [4]

  **References**:
  - Source: `benchmark/src/astar_twin/solver/pipeline.py:226-262` — reserve phase to restructure
  - Source: `benchmark/src/astar_twin/solver/policy/allocator.py:387-413` — `plan_reserve_queries()` to add `n_queries` param
  - Constants: `benchmark/src/astar_twin/solver/policy/allocator.py:44` — `RESERVE_QUERIES = 10`
  - Tests: `benchmark/tests/solver/test_pipeline.py` — existing pipeline tests

  **Acceptance Criteria**:
  - [ ] Reserve phase transcript shows two distinct batches (gap in timing or batch grouping)
  - [ ] `plan_reserve_queries()` accepts `n_queries` parameter
  - [ ] Total reserve queries ≤ 10
  - [ ] `uv run pytest tests/solver/test_pipeline.py -q` → all pass

  **QA Scenarios**:
  ```
  Scenario: Reserve queries issued in two batches
    Tool: Bash
    Steps: uv run pytest tests/solver/test_pipeline.py::test_reserve_two_batches -v
    Expected: PASSED — transcript has reserve queries split across two groups
    Evidence: .sisyphus/evidence/task-7-reserve-batching.txt
  ```

  **Commit**: NO — committed with Wave 2 batch

---

- [x] 8. Convert fixed_coverage_baseline to 50-query viewport sweep

  **What to do**:
  In `benchmark/src/astar_twin/solver/baselines.py`, modify `fixed_coverage_baseline()` (lines 23-42) to simulate the planned 50-query sweep strategy instead of full-map MC with default params.

  The planned approach: use 50 queries (10 per seed × 5 seeds), each a 15x15 viewport placed via a fixed grid pattern, then aggregate viewport observations into a full prediction tensor. This tests the "API query as primary data source" baseline.

  New implementation:
  ```python
  def fixed_coverage_baseline(
      initial_states: list[InitialState],
      height: int,
      width: int,
      n_mc_runs: int = 200,
      base_seed: int = 42,
      queries_per_seed: int = 10,
  ) -> list[NDArray[np.float64]]:
      """50-query fixed-viewport sweep baseline.
      
      Places queries_per_seed viewports per seed on a grid pattern,
      runs MC for each viewport, aggregates into full-map prediction.
      Non-observed cells get uniform 1/6 prior.
      """
      simulator = Simulator(SimulationParams())
      mc_runner = MCRunner(simulator)
      tensors: list[NDArray[np.float64]] = []
      
      for seed_idx, initial_state in enumerate(initial_states):
          # Generate fixed grid viewports
          viewports = _generate_grid_viewports(height, width, queries_per_seed)
          
          # Full-map MC for each viewport region
          combined = np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
          coverage_count = np.zeros((height, width), dtype=np.int32)
          
          runs = mc_runner.run_batch(
              initial_state, n_runs=n_mc_runs,
              base_seed=base_seed + seed_idx * 1000,
          )
          raw = aggregate_runs(runs, height, width)
          
          for vp_x, vp_y, vp_w, vp_h in viewports:
              combined[vp_y:vp_y+vp_h, vp_x:vp_x+vp_w] = raw[vp_y:vp_y+vp_h, vp_x:vp_x+vp_w]
              coverage_count[vp_y:vp_y+vp_h, vp_x:vp_x+vp_w] += 1
          
          tensors.append(safe_prediction(combined))
      
      return tensors
  
  def _generate_grid_viewports(
      height: int, width: int, n_viewports: int,
  ) -> list[tuple[int, int, int, int]]:
      """Generate a grid of viewport positions for coverage."""
      from astar_twin.contracts.types import MAX_VIEWPORT
      viewports = []
      # Grid step to achieve roughly n_viewports
      n_cols = max(1, int(np.ceil(np.sqrt(n_viewports * width / height))))
      n_rows = max(1, int(np.ceil(n_viewports / n_cols)))
      step_x = max(1, width // n_cols)
      step_y = max(1, height // n_rows)
      for row in range(n_rows):
          for col in range(n_cols):
              if len(viewports) >= n_viewports:
                  break
              x = min(col * step_x, max(0, width - MAX_VIEWPORT))
              y = min(row * step_y, max(0, height - MAX_VIEWPORT))
              w = min(MAX_VIEWPORT, width - x)
              h = min(MAX_VIEWPORT, height - y)
              viewports.append((x, y, w, h))
      return viewports
  ```

  Update `compute_baseline_summary()` to pass through correctly.

  Add tests in `benchmark/tests/solver/test_baselines.py` (update existing or new):
  - Test that `fixed_coverage_baseline()` returns list of tensors with correct shapes
  - Test that `_generate_grid_viewports()` returns correct number of viewports
  - Test that non-covered cells have uniform distribution (approximately 1/6)

  **Must NOT do**: Don't remove `uniform_baseline()`. Don't change the function return type. Keep backward compatibility with `compute_baseline_summary()`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: Algorithm redesign + helper function + test updates
  - Skills: [] — no special skills needed
  - Omitted: [`quality-check`] — Wave 3

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [] | Blocked By: [1]

  **References**:
  - Source: `benchmark/src/astar_twin/solver/baselines.py:23-42` — `fixed_coverage_baseline()` to rewrite
  - Source: `benchmark/src/astar_twin/solver/baselines.py:55-75` — `compute_baseline_summary()` consumer
  - Constants: `benchmark/src/astar_twin/contracts/types.py` — `MAX_VIEWPORT`, `NUM_CLASSES`
  - Original plan: `.sisyphus/plans/optimal-astar-island-solver.md` — Task 1 description of fixed-coverage baseline
  - Tests: `benchmark/tests/solver/` — look for existing baseline tests

  **Acceptance Criteria**:
  - [ ] `fixed_coverage_baseline()` returns `list[NDArray]` with shapes `(H, W, 6)`
  - [ ] `_generate_grid_viewports(40, 40, 10)` returns 10 viewports
  - [ ] `compute_baseline_summary()` still works with new implementation
  - [ ] `uv run pytest tests/solver/ -q -k baseline` → all pass

  **QA Scenarios**:
  ```
  Scenario: Fixed coverage returns correct shapes
    Tool: Bash
    Steps: uv run pytest tests/solver/ -v -k "test_fixed_coverage"
    Expected: PASSED — all tensors have shape (H, W, 6)
    Evidence: .sisyphus/evidence/task-8-baseline.txt

  Scenario: Grid viewports cover expected count
    Tool: Bash
    Steps: uv run pytest tests/solver/ -v -k "test_grid_viewports"
    Expected: PASSED — 10 viewports for 40x40 map
    Evidence: .sisyphus/evidence/task-8-grid.txt
  ```

  **Commit**: YES | Message: `fix(solver): plan compliance — viewport sizing, disagreement, reserve batching, baseline` | Files: `benchmark/src/astar_twin/solver/policy/hotspots.py`, `benchmark/src/astar_twin/solver/policy/allocator.py`, `benchmark/src/astar_twin/solver/pipeline.py`, `benchmark/src/astar_twin/solver/baselines.py`, `benchmark/tests/solver/test_hotspots.py`, `benchmark/tests/solver/test_allocator.py`, `benchmark/tests/solver/test_pipeline.py`

---

- [x] 9. Add `dump_prediction_stats` CLI + library function

  **What to do**:
  Create `benchmark/src/astar_twin/solver/eval/dump_prediction_stats.py`:
  ```python
  """Dump prediction statistics for a solver run.
  
  CLI: uv run python -m astar_twin.solver.eval.dump_prediction_stats
  Library: from astar_twin.solver.eval.dump_prediction_stats import dump_stats
  """
  from __future__ import annotations
  
  import json
  import sys
  from pathlib import Path
  
  import numpy as np
  from numpy.typing import NDArray
  
  from astar_twin.contracts.types import NUM_CLASSES
  
  
  def dump_stats(
      tensors: list[NDArray[np.float64]],
      height: int,
      width: int,
  ) -> dict:
      """Compute and return prediction statistics.
      
      Args:
          tensors: List of H×W×6 prediction tensors (one per seed).
          height: Expected map height.
          width: Expected map width.
      
      Returns:
          Dict with per-seed and aggregate statistics.
      """
      stats = {"seeds": [], "aggregate": {}}
      all_mins, all_maxs, all_entropies = [], [], []
      
      for i, t in enumerate(tensors):
          assert t.shape == (height, width, NUM_CLASSES), f"Seed {i}: shape {t.shape} != ({height}, {width}, {NUM_CLASSES})"
          seed_min = float(t.min())
          seed_max = float(t.max())
          sums = t.sum(axis=2)
          sum_ok = bool(np.allclose(sums, 1.0, atol=1e-6))
          # Per-cell entropy
          p = np.clip(t, 1e-15, 1.0)
          entropy = -np.sum(p * np.log(p), axis=2)
          mean_entropy = float(entropy.mean())
          
          stats["seeds"].append({
              "seed_index": i,
              "min_prob": seed_min,
              "max_prob": seed_max,
              "sum_check_passed": sum_ok,
              "mean_entropy": mean_entropy,
              "shape": list(t.shape),
          })
          all_mins.append(seed_min)
          all_maxs.append(seed_max)
          all_entropies.append(mean_entropy)
      
      stats["aggregate"] = {
          "n_seeds": len(tensors),
          "global_min_prob": min(all_mins) if all_mins else None,
          "global_max_prob": max(all_maxs) if all_maxs else None,
          "mean_entropy": float(np.mean(all_entropies)) if all_entropies else None,
      }
      return stats
  
  
  def main() -> None:
      """CLI entrypoint: load fixture, run solver, dump stats."""
      from astar_twin.data.loaders import load_fixture
      from astar_twin.solver.baselines import uniform_baseline
      
      fixture_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/rounds/test-round-001")
      fixture = load_fixture(fixture_dir)
      
      height = fixture.round_detail.map_height
      width = fixture.round_detail.map_width
      n_seeds = len(fixture.round_detail.initial_states)
      
      # Use uniform baseline as a quick demo
      tensors = [uniform_baseline(height, width) for _ in range(n_seeds)]
      stats = dump_stats(tensors, height, width)
      print(json.dumps(stats, indent=2))
  
  
  if __name__ == "__main__":
      main()
  ```

  Add tests in `benchmark/tests/solver/test_dump_prediction_stats.py`:
  - Test `dump_stats()` with valid tensors → returns expected keys
  - Test shape assertion fails with wrong shape
  - Test CLI entry point runs without error

  **Must NOT do**: Don't import solver pipeline in library function (keep it dependency-free). Don't use fixture `simulation_params`.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: New standalone file + tests
  - Skills: [] — no special skills needed
  - Omitted: [`quality-check`] — Wave 3

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [] | Blocked By: []

  **References**:
  - Source: `benchmark/src/astar_twin/solver/eval/__init__.py` — module to register in
  - Source: `benchmark/src/astar_twin/contracts/types.py` — `NUM_CLASSES`
  - Source: `benchmark/src/astar_twin/data/loaders.py` — `load_fixture()` for CLI
  - Source: `benchmark/src/astar_twin/solver/baselines.py:16-20` — `uniform_baseline()` for CLI demo

  **Acceptance Criteria**:
  - [ ] `uv run python -m astar_twin.solver.eval.dump_prediction_stats data/rounds/test-round-001` → prints JSON stats
  - [ ] `dump_stats()` returns dict with keys `seeds` and `aggregate`
  - [ ] `uv run pytest tests/solver/test_dump_prediction_stats.py -q` → all pass

  **QA Scenarios**:
  ```
  Scenario: CLI runs and outputs JSON
    Tool: Bash
    Steps: cd benchmark && uv run python -m astar_twin.solver.eval.dump_prediction_stats data/rounds/test-round-001
    Expected: JSON output with seeds and aggregate keys, exit code 0
    Evidence: .sisyphus/evidence/task-9-stats-cli.txt

  Scenario: Library function returns correct structure
    Tool: Bash
    Steps: uv run pytest tests/solver/test_dump_prediction_stats.py -v
    Expected: PASSED — all tests pass
    Evidence: .sisyphus/evidence/task-9-stats-lib.txt
  ```

  **Commit**: YES | Message: `feat(solver): add dump_prediction_stats CLI and library function` | Files: `benchmark/src/astar_twin/solver/eval/dump_prediction_stats.py`, `benchmark/tests/solver/test_dump_prediction_stats.py`

---

- [x] 10. Run ruff auto-fix on all solver + test files

  **What to do**:
  Run the following commands in sequence:
  ```bash
  cd benchmark
  uv run ruff check --fix src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/
  uv run ruff format src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/
  ```
  This resolves the 74 auto-fixable errors (unused imports, unsorted imports, quoted annotations).

  After auto-fix, run `uv run ruff check src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/` to see remaining errors.

  Run `uv run pytest -q tests/solver/` to verify no tests broke from auto-fixes.

  **Must NOT do**: Don't use `--unsafe-fixes`. Don't manually fix errors in this task (that's tasks 11-12).

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Two CLI commands + verification
  - Skills: [`quality-check`] — to verify ruff results
  - Omitted: [] — quality-check is appropriate here

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [11, 12] | Blocked By: [1-9]

  **References**:
  - Config: `benchmark/pyproject.toml` — ruff configuration
  - Source: all files in `benchmark/src/astar_twin/solver/` and `benchmark/tests/solver/`
  - Scoring: `benchmark/src/astar_twin/scoring/`

  **Acceptance Criteria**:
  - [ ] `uv run ruff format --check src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/` → 0 files would be reformatted
  - [ ] Auto-fixable errors reduced to 0
  - [ ] `uv run pytest -q tests/solver/` → all pass (no regressions)

  **QA Scenarios**:
  ```
  Scenario: Format check passes
    Tool: Bash
    Steps: cd benchmark && uv run ruff format --check src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/
    Expected: "0 files would be reformatted" or no output
    Evidence: .sisyphus/evidence/task-10-ruff-format.txt

  Scenario: Tests still pass after auto-fix
    Tool: Bash
    Steps: cd benchmark && uv run pytest -q tests/solver/
    Expected: All tests pass
    Evidence: .sisyphus/evidence/task-10-tests-after-fix.txt
  ```

  **Commit**: NO — committed with Wave 3 batch

---

- [ ] 11. Extract shared `_resilient_run_batch` + fix misleading docstrings

  **What to do**:
  
  **Part A: Extract `_resilient_run_batch`**
  Create `benchmark/src/astar_twin/solver/eval/_helpers.py`:
  ```python
  """Shared helpers for eval scripts."""
  from __future__ import annotations
  # ... extract _resilient_run_batch from run_benchmark_suite.py
  ```
  - Copy `_resilient_run_batch` from `run_benchmark_suite.py` (lines 104+) into `_helpers.py`
  - Replace both copies in `run_benchmark_suite.py` and `run_replay_validation.py` with: `from astar_twin.solver.eval._helpers import _resilient_run_batch`

  **Part B: Fix misleading docstrings**
  After Task 6, the disagreement function will have correct cellwise implementation. Verify the docstring matches. Also check:
  - `compute_posterior_disagreement()` docstring should say "cellwise top-2 argmax" (Task 6 handles the code; verify docstring here)
  - `check_argmax_disagreement()` docstring should match actual behavior
  - `plan_reserve_queries()` docstring should mention "two batches" (Task 7 handles code; verify docstring here)

  **Must NOT do**: Don't change any function behavior. Don't modify `_resilient_run_batch` logic.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: File extraction + docstring fixes
  - Skills: [] — no special skills needed
  - Omitted: [`quality-check`] — task 12 handles final lint

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [12] | Blocked By: [10]

  **References**:
  - Source: `benchmark/src/astar_twin/solver/eval/run_benchmark_suite.py:104+` — first copy of `_resilient_run_batch`
  - Source: `benchmark/src/astar_twin/solver/eval/run_replay_validation.py:69+` — second copy
  - Source: `benchmark/src/astar_twin/solver/policy/allocator.py:234-261` — disagreement docstring (after Task 6)
  - Source: `benchmark/src/astar_twin/solver/policy/allocator.py:368-384` — argmax docstring
  - Source: `benchmark/src/astar_twin/solver/policy/allocator.py:387-413` — reserve docstring (after Task 7)

  **Acceptance Criteria**:
  - [ ] `_resilient_run_batch` exists only in `_helpers.py` (not duplicated)
  - [ ] `grep -r "_resilient_run_batch" benchmark/src/astar_twin/solver/eval/` shows definition in `_helpers.py` and imports in the other two files
  - [ ] Docstrings for `compute_posterior_disagreement`, `check_argmax_disagreement`, `plan_reserve_queries` accurately describe behavior
  - [ ] `uv run pytest -q tests/solver/` → all pass

  **QA Scenarios**:
  ```
  Scenario: No duplicate _resilient_run_batch
    Tool: Bash
    Steps: grep -c "def _resilient_run_batch" benchmark/src/astar_twin/solver/eval/*.py
    Expected: _helpers.py:1, run_benchmark_suite.py:0, run_replay_validation.py:0
    Evidence: .sisyphus/evidence/task-11-dedup.txt

  Scenario: Tests still pass
    Tool: Bash
    Steps: cd benchmark && uv run pytest -q tests/solver/
    Expected: All pass
    Evidence: .sisyphus/evidence/task-11-tests.txt
  ```

  **Commit**: NO — committed with Wave 3 batch

---

- [ ] 12. Manual ruff fixes + tighten typing + final quality gate

  **What to do**:
  
  **Part A: Manual ruff fixes**
  After Task 10 auto-fixes, the remaining ~128 errors are:
  - `ANN201` (68): Add return type annotations to public functions (`: None` for tests, proper types for library code)
  - `ANN001` (31): Add type annotations to function arguments
  - `B905` (14): Add `strict=True` to `zip()` calls
  - `B007` (4): Rename unused loop variables to `_`
  - `F821` (2): Fix undefined names
  - `F841` (2): Remove unused variables
  - `ANN401` (1): Replace `Any` with specific type
  - `B011` (1): Replace `assert False` with `raise AssertionError`
  - `SIM108` (1): Simplify if-else to ternary
  - `E501` (2): Shorten long lines

  Fix all of these across `src/astar_twin/solver/`, `tests/solver/`, and `src/astar_twin/scoring/`.

  **Part B: Tighten `Particle.params` typing**
  In `benchmark/src/astar_twin/solver/inference/particles.py`, the `Particle` class uses `params: dict[str, Any]`. Create a `ParamValue` type alias:
  ```python
  from astar_twin.params.simulation_params import AdjacencyMode, DistanceMetric, UpdateOrderMode
  
  ParamValue = float | int | AdjacencyMode | DistanceMetric | UpdateOrderMode
  
  @dataclass
  class Particle:
      params: dict[str, ParamValue]
      log_weight: float = 0.0
  ```
  Update all downstream type signatures that accept `dict[str, Any]` from particle params to use `dict[str, ParamValue]`.

  **Part C: Final quality gate**
  Run all three quality checks:
  ```bash
  uv run ruff check src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/
  uv run ruff format --check src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/
  uv run pytest -q tests/solver/
  ```
  All must pass with 0 errors.

  **Must NOT do**: Don't change function behavior while fixing types. Don't add `# type: ignore` unless truly necessary (document why).

  **Recommended Agent Profile**:
  - Category: `unspecified-low` — Reason: Large volume of small fixes across many files
  - Skills: [`quality-check`] — to validate final results
  - Omitted: [] — quality-check appropriate for final gate

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [] | Blocked By: [11]

  **References**:
  - Source: `benchmark/src/astar_twin/solver/inference/particles.py:70-76` — `Particle` class to tighten
  - Config: `benchmark/pyproject.toml` — ruff rules configuration
  - Source: all files in `benchmark/src/astar_twin/solver/`, `benchmark/tests/solver/`, `benchmark/src/astar_twin/scoring/`

  **Acceptance Criteria**:
  - [ ] `uv run ruff check src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/` → 0 errors ("All checks passed!")
  - [ ] `uv run ruff format --check src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/` → 0 files would be reformatted
  - [ ] `uv run pytest -q tests/solver/` → all pass, 0 failures
  - [ ] `Particle.params` typed as `dict[str, ParamValue]` not `dict[str, Any]`

  **QA Scenarios**:
  ```
  Scenario: Zero ruff errors
    Tool: Bash
    Steps: cd benchmark && uv run ruff check src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/
    Expected: "All checks passed!" or empty output, exit code 0
    Evidence: .sisyphus/evidence/task-12-ruff-clean.txt

  Scenario: Full test suite passes
    Tool: Bash
    Steps: cd benchmark && uv run pytest -q tests/solver/
    Expected: All pass, 0 failures
    Evidence: .sisyphus/evidence/task-12-final-tests.txt

  Scenario: Particle typing tightened
    Tool: Bash
    Steps: grep -n "ParamValue" benchmark/src/astar_twin/solver/inference/particles.py
    Expected: ParamValue type alias defined, Particle.params uses it
    Evidence: .sisyphus/evidence/task-12-typing.txt
  ```

  **Commit**: YES | Message: `chore(solver): cleanup — ruff fixes, dedup helpers, tighten typing` | Files: all modified solver/test/scoring files

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback → fix → re-run → present again → wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ must execute tests and CLI, NOT read-only)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
| Commit | After Tasks | Message |
|--------|-------------|---------|
| 1 | 1-4 (Wave 1) | `fix(solver): critical correctness — safe_prediction, empty-particle, KL, live prediction` |
| 2 | 5-8 (Wave 2) | `fix(solver): plan compliance — viewport sizing, disagreement, reserve batching, baseline` |
| 3 | 9 | `feat(solver): add dump_prediction_stats CLI and library function` |
| 4 | 10-12 (Wave 3) | `chore(solver): cleanup — ruff fixes, dedup helpers, tighten typing` |

## Success Criteria
1. `uv run ruff check src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/` → 0 errors
2. `uv run ruff format --check src/astar_twin/solver/ tests/solver/ src/astar_twin/scoring/` → 0 reformats
3. `uv run pytest -q tests/solver/` → all pass, 0 failures
4. F1-F4 all APPROVE on re-run
5. No new files outside allowed paths
6. No changes to engine/phases/mc directories
