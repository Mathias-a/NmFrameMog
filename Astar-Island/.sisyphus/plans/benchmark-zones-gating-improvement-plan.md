## Goal

Improve the strongest current benchmark family by turning the zone-aware calibrated Monte Carlo strategy into a **conditionally trusted** strategy that keeps its large upside on zone-friendly rounds while avoiding the regression rounds where zone priors currently hurt.

This plan first maps the main improvement avenues, then commits to one concrete strategy.

---

## Current benchmark picture

### What is currently best

Using the real fixtures under `benchmark/data/rounds/` and the existing `BenchmarkRunner`, the strongest current non-cheating strategy family is the zone-aware calibrated MC line.

Multi-round mean over 15 real rounds:

- `calibrated_mc_zones`: **56.54**
- `calibrated_mc_zones_variance`: **56.54**
- `calibrated_mc_zones_adaptive`: **54.88**
- `calibrated_mc`: **54.86**
- `learned_calibrator`: **53.74**
- `filter_baseline`: **32.31**

Latest real round (round 15):

- `calibrated_mc_zones`: **72.23**
- `calibrated_mc_zones_variance`: **72.23**
- `calibrated_mc_zones_adaptive`: **71.33**
- `calibrated_mc`: **71.32**
- `learned_calibrator`: **69.27**

### What these results imply

`calibrated_mc_zones` is the best current baseline because its settlement-aware priors add real signal. But its gains are not universal.

It beats plain `calibrated_mc` strongly on rounds 1, 2, 3, 4, 5, 8, 9, 10, 13, and 15. It loses on rounds 6, 7, 11, 12, and 14. The losses are meaningful, not noise.

That means the main problem is not “zones are wrong.” The problem is:

> the strategy has no reliable way to decide when zone priors should dominate and when they should back off.

This is also consistent with the implementation in `benchmark/src/astar_twin/strategies/calibrated_mc/strategy.py`:

- zone templates are fixed,
- zone weights are fixed by hand,
- adaptive blending currently uses a coarse variance heuristic,
- `use_mc_variance` does not help the strongest zone strategy at all,
- `learned_calibrator` mostly collapses to a tiny global correction (`core=0.2`, everything else `0.0` in the saved CV output), so it is not learning a rich rescue policy.

---

## Improvement avenues mapped out

### Avenue A — brute-force tune the existing calibrated MC hyperparameters

Examples:

- `n_runs`
- `prior_weight`
- `temperature`
- `static_confidence`
- `n_subbatches`

### Why it could help

The current strategy clearly responds to blending behavior. A systematic sweep may find a better global operating point.

### Why I am not committing to it

This would mostly improve the average fixed setting. It does not address the observed regime-switching problem where zones help some rounds and hurt others. A better single global knob may raise the floor slightly, but it is unlikely to recover the big missed upside from conditional behavior.

### Verdict

Useful supporting experiment, but not the primary strategy.

---

### Avenue B — improve Monte Carlo uncertainty handling

Examples:

- fix subbatch remainder loss,
- add Dirichlet smoothing before temperature scaling,
- replace the current variance heuristic with a stronger uncertainty-to-blend mapping,
- make temperature or prior-weight depend on local uncertainty.

### Why it could help

The code currently reduces MC uncertainty to `rare_class_var * 2.0`, clipped into a small additive adjustment. That is a weak summary of uncertainty and likely throws away useful structure.

### Why I am not committing to it first

This is promising, but the current evidence says the biggest gains come specifically from **using settlement zones**, not from the variance feature. On both round 15 and the 15-round average, `calibrated_mc_zones` and `calibrated_mc_zones_variance` are identical. That strongly suggests variance handling is not currently the dominant lever.

### Verdict

Strong secondary workstream after the main strategy is in place.

---

### Avenue C — improve the learned calibrator

Examples:

- richer learned weights,
- round-level or feature-conditioned blending,
- better optimization than coarse coordinate search,
- use more than zone-wise interpolation between base and fallback.

### Why it could help

This path directly targets calibration and already has a training pipeline plus leave-one-round-out evaluation script.

### Why I am not committing to it first

The evidence says the current learned calibrator is not built on the right winning unit. Its saved CV only slightly beats its own base in weighted average, and on the direct 15-round benchmark it trails `calibrated_mc_zones` by almost 2.8 points. In practice it is learning only a mild global correction, not a robust “when zones help” policy.

### Verdict

Valuable if repurposed, but not as a standalone continuation of the current learned-calibrator path.

---

### Avenue D — introduce conditional zone trust on top of `calibrated_mc_zones`

Examples:

- infer whether a seed/round looks zone-friendly or zone-hostile,
- compute a seed-level selector that decides whether to trust the zone prior or fall back to plain calibrated MC,
- learn the selector from historical rounds using cheap structural features available at prediction time.

### Why it could help

This avenue matches the benchmark evidence exactly:

- zones are the strongest source of uplift,
- zones are also the source of some of the biggest regressions,
- the winning move is to keep the zone upside while suppressing the bad-regime failures.

### Verdict

**This is the strategy to commit to.**

---

## Committed strategy

## Build a simple seed-level selector on top of `calibrated_mc_zones`

### Core idea

Start from the two strongest relevant existing predictions:

1. plain `calibrated_mc`
2. `calibrated_mc_zones`

Then learn a **small deterministic seed-level selector** that decides whether to trust the zone-aware prediction or fall back to plain `calibrated_mc` for that seed.

Initial form:

- compute both parent predictions,
- compute seed-level features available at prediction time,
- choose either `zoned_prediction` or `plain_prediction`,
- default to `zoned_prediction` when the selector is uncertain.

This is different from the current `learned_calibrator` in an important way:

- current learned calibrator blends `calibrated_mc` with `filter_baseline`,
- the new strategy should select between **the two strongest modern predictions**,
- the selector should depend on features that explain the rounds where zones fail,
- the first iteration should **not** use cell-level gating.

The goal is not to invent a whole new model family. The goal is to make the current best family safer and more adaptive with the smallest possible extra model.

---

## Hypothesis

The zone prior is most useful when the seed’s settlement layout and coastline geometry line up with the handcrafted assumptions behind:

- `CORE`
- `EXPANSION_RING`
- `COASTAL_HUB`
- `REMOTE_COASTAL`
- `REMOTE_INLAND`

It hurts when those assumptions overstate the effect of settlement-centered expansion or coastal hub behavior.

So the best next strategy is:

> learn a lightweight seed-level selector that predicts when the zone prior is trustworthy, using only information available from the initial state and from the two candidate prediction tensors.

---

## Why this strategy is better than the alternatives

It directly targets the measured failure pattern.

It preserves the current best implementation instead of replacing it.

It is benchmark-safe because it only changes strategy code and training utilities, not the simulator.

It is testable with the existing historical fixtures and benchmark harness.

It can absorb later improvements in uncertainty handling: if variance features become more informative later, they can simply become extra gate features.

---

## Detailed execution plan

## TDD rule for this work

Every implementation phase starts with a failing or missing-test check, then code, then benchmark verification.

Sequence for each phase:

1. add or update the smallest test that should fail before the change,
2. implement only enough code to pass that test,
3. run the narrowest benchmark or evaluation command that proves the phase goal,
4. only then move to the next phase.

### Phase 1 — characterize the delta between zoned and plain predictions

#### Objective

Build a precise picture of when `calibrated_mc_zones` wins and where it loses.

#### Files

- `benchmark/src/astar_twin/strategies/calibrated_mc/strategy.py`
- `benchmark/src/astar_twin/strategies/learned_calibrator/training.py`
- new analysis helper under `benchmark/src/astar_twin/solver/eval/` or strategy training utilities

#### Work

Create an analysis utility that, for every historical round and seed:

- computes `plain_prediction = REGISTRY['calibrated_mc']()`,
- computes `zoned_prediction = REGISTRY['calibrated_mc_zones']()`,
- scores both against ground truth,
- computes per-zone delta metrics,
- records structural features from the initial state.

#### Features to log

Round/seed level:

- alive settlement count,
- port settlement count,
- settlement density,
- mean and max settlement-to-cell distance,
- fraction of dynamic cells in each zone,
- coastal dynamic fraction,
- map compactness / coastline exposure,
- average entropy difference between zoned and plain predictions,
- average absolute class-shift by zone,
- plain-vs-zoned disagreement statistics.

#### Success criterion

We can explain the losses of rounds 6, 7, 11, 12, and 14 in feature terms rather than only by score.

#### QA scenario

Run from `benchmark/`:

`uv run python -c "from pathlib import Path; import json; import numpy as np; from astar_twin.data.loaders import list_fixtures; from astar_twin.harness.runner import BenchmarkRunner; from astar_twin.strategies import REGISTRY; data_dir = Path('data'); fixtures = [f for f in list_fixtures(data_dir) if (not f.id.startswith('test-')) and f.ground_truths is not None]; fixtures.sort(key=lambda f: f.round_number); rows = [];\
for fixture in fixtures:\
    report = BenchmarkRunner(fixture=fixture, base_seed=42).run([REGISTRY['calibrated_mc'](), REGISTRY['calibrated_mc_zones']()]);\
    scores = {sr.strategy_name: sr.mean_score for sr in report.strategy_reports};\
    rows.append((fixture.round_number, scores['calibrated_mc'], scores['calibrated_mc_zones'], scores['calibrated_mc_zones'] - scores['calibrated_mc']));\
print(json.dumps(rows, indent=2))"`

Pass condition:

- the output clearly identifies the zone-winning and zone-losing rounds,
- the analysis artifact records enough seed-level features to support selector design,
- no simulator internals were modified.

---

### Phase 2 — design the selector target and representation

#### Objective

Define exactly what the model should predict.

#### Decision

Use a **seed-level selector first**, not a cell-level gate.

Selector output:

- `use_zones = 1` when seed features indicate zone priors are trustworthy,
- `use_zones = 0` otherwise,
- default to `1` when the rule is uncertain.

#### Proposed target

For each seed example, define an “oracle selector target” as:

- `1` if `calibrated_mc_zones` beats `calibrated_mc` on that seed,
- `0` otherwise.

The first implementation should prefer a simple thresholded linear score or hand-coded rule over probabilistic mixing.

#### Success criterion

We have a target definition that is learnable from historical fixtures and stable under leave-one-round-out validation.

#### QA scenario

Run from `benchmark/` after adding the selector-training utility:

`uv run python -m astar_twin.solver.eval.run_gated_calibrated_mc_cv --data-dir data --mode selector-target-dump`

Pass condition:

- every real round contributes only held-out seed labels for evaluation,
- no CV fold trains on the held-out round,
- the script prints class balance and target counts so selector feasibility is visible.

---

### Phase 3 — implement a lightweight gated strategy

#### Objective

Add a new strategy that chooses between plain and zoned calibrated MC using learned selector logic.

#### Proposed new package

- `benchmark/src/astar_twin/strategies/gated_calibrated_mc/strategy.py`
- possibly `model.py` and `training.py` alongside it

#### Strategy behavior

At prediction time:

1. compute plain calibrated MC tensor,
2. compute zoned calibrated MC tensor,
3. derive selector features from the initial state and the two tensors,
4. apply a deterministic learned selector,
5. return either the zoned or plain tensor.

#### Initial modeling choice

Keep the learned component intentionally simple and reproducible:

- logistic regression style formula implemented directly in NumPy, or
- small hand-coded linear model with fixed coefficients learned offline.

Avoid heavyweight dependencies or opaque training artifacts.

#### Why simple first

The historical dataset is small: 15 rounds × 5 seeds. The main risk is overfitting, not underfitting.

#### Success criterion

The new strategy is deterministic, easy to inspect, and produces valid `(H, W, 6)` tensors with no contract regressions.

#### TDD test list

Add tests in `benchmark/tests/strategies/test_gated_calibrated_mc.py` for:

- registry contains the new strategy,
- output shape matches fixture dimensions,
- probabilities sum to one,
- deterministic output for fixed seed,
- selector chooses zoned parent when forced,
- selector chooses plain parent when forced,
- uncertain selector path defaults to zoned parent.

#### QA scenario

Run from `benchmark/`:

- `uv run pytest tests/strategies/test_gated_calibrated_mc.py`
- `uv run pytest tests/strategies/test_calibrated_mc.py tests/strategies/test_learned_calibrator.py`

Pass condition:

- all new selector tests pass,
- no regressions in existing calibrated strategy tests,
- output contract remains deterministic and normalized.

---

### Phase 4 — training pipeline and model selection

#### Objective

Train the selector without leaking held-out round information.

#### Evaluation method

Use leave-one-round-out cross-validation, matching the existing learned-calibrator evaluation style.

#### Candidate model ladder

Try these in order, stopping when gains saturate:

1. hand-coded rule from one or two seed-level thresholds,
2. linear seed-level selector using a few normalized features,
3. calibrated score threshold with a default-to-zones fallback.

#### Model selection rule

Pick the simplest model that:

- improves weighted mean over `calibrated_mc_zones`,
- improves median held-out round score,
- reduces the worst-round regressions versus `calibrated_mc_zones`.

#### Explicit anti-overfitting checks

- no tuning on round 15 alone,
- no model chosen by single-round peak score,
- require gains on both weighted mean and downside control,
- inspect feature/threshold stability across folds,
- treat rounds, not cells, as the real generalization unit.

#### Success criterion

The chosen selector wins on cross-validation for the right reason: it keeps most of the zone uplift while shrinking the rounds where zones backfire.

#### QA scenario

Run from `benchmark/`:

`uv run python -m astar_twin.solver.eval.run_gated_calibrated_mc_cv --data-dir data --output results/gated_calibrated_mc_cv.json`

Pass condition:

- leave-one-round-out weighted mean is higher than `calibrated_mc_zones`,
- held-out median is not worse than `calibrated_mc_zones`,
- downside on the zone-losing rounds is reduced,
- the learned rule is simple enough to explain in a few lines.

---

### Phase 5 — benchmark verification against the current leaders

#### Objective

Prove the strategy is a real improvement, not a training artifact.

#### Required comparisons

- `calibrated_mc`
- `calibrated_mc_zones`
- `calibrated_mc_zones_variance`
- `calibrated_mc_zones_adaptive`
- `learned_calibrator`
- new gated strategy

#### Required metrics

- 15-round mean,
- weighted mean if round weighting matters for selection,
- median round score,
- number of rounds won,
- worst round,
- average gain on rounds where zones already help,
- average rescue on rounds where zones hurt.

#### Promotion rule

Promote the new strategy only if it beats `calibrated_mc_zones` on aggregate while also reducing the downside tail.

If it only ties mean but narrows worst-round loss substantially, keep it as a hedge candidate but do not replace the current leader yet.

#### QA scenario

Run from `benchmark/`:

`uv run python -c "from pathlib import Path; import json; import numpy as np; from astar_twin.data.loaders import list_fixtures; from astar_twin.harness.runner import BenchmarkRunner; from astar_twin.strategies import REGISTRY; names = ['calibrated_mc','calibrated_mc_zones','calibrated_mc_zones_variance','calibrated_mc_zones_adaptive','learned_calibrator','gated_calibrated_mc']; fixtures = [f for f in list_fixtures(Path('data')) if (not f.id.startswith('test-')) and f.ground_truths is not None]; fixtures.sort(key=lambda f: f.round_number); rows = []; summary = {name: [] for name in names};\
for fixture in fixtures:\
    report = BenchmarkRunner(fixture=fixture, base_seed=42).run([REGISTRY[name]() for name in names]);\
    scores = {sr.strategy_name: sr.mean_score for sr in report.strategy_reports};\
    rows.append({'round_number': fixture.round_number, **scores});\
    [summary[name].append(scores[name]) for name in names];\
print(json.dumps({'summary': {name: float(np.mean(vals)) for name, vals in summary.items()}, 'per_round': rows}, indent=2))"`

Pass condition:

- the new strategy beats `calibrated_mc_zones` on 15-round mean,
- median round score is no worse,
- worst losing-round delta versus zones is smaller,
- no regression below `calibrated_mc` on the majority of rounds.

---

### Phase 6 — only then revisit uncertainty improvements

If the seed-level selector succeeds, the next iteration should add stronger uncertainty features into the selector rather than directly rewriting the full calibrated MC blend logic.

Best follow-up candidates:

- subbatch remainder fix,
- Dirichlet smoothing of MC counts,
- better per-cell uncertainty features,
- uncertainty-aware temperature map.

That sequence is safer because it improves the same committed architecture rather than opening a second unrelated search space too early.

#### QA scenario

Run the Phase 5 benchmark again after each uncertainty-related change.

Pass condition:

- uncertainty features improve the committed selector architecture,
- they do not become a new independent strategy branch without evidence.

---

## Concrete file plan

### Files to add

- `benchmark/src/astar_twin/strategies/gated_calibrated_mc/strategy.py`
- `benchmark/src/astar_twin/strategies/gated_calibrated_mc/model.py`
- `benchmark/src/astar_twin/strategies/gated_calibrated_mc/training.py`
- `benchmark/tests/strategies/test_gated_calibrated_mc.py`
- optional eval script under `benchmark/src/astar_twin/solver/eval/run_gated_calibrated_mc_cv.py`

### Files to update

- `benchmark/src/astar_twin/strategies/__init__.py` to register the new strategy
- possibly shared training helpers if reuse is cleaner than duplication

### Files not to touch

- anything under `benchmark/src/astar_twin/engine/`
- anything under `benchmark/src/astar_twin/phases/`
- anything under `benchmark/src/astar_twin/mc/` for the first iteration

This keeps the work inside the strategy layer where it belongs.

---

## Validation plan

### Unit tests

Add tests for:

- deterministic output,
- valid probability tensor shape and normalization,
- selector chooses the expected parent under forced conditions,
- registry wiring.

### Cross-validation checks

Add a reproducible CV script that prints:

- new strategy weighted mean,
- zoned baseline weighted mean,
- plain baseline weighted mean,
- per-round deltas,
- learned thresholds / coefficients.

### Benchmark checks

Run the 15-round comparison script and store summary output for the final decision.

### Quality gates

After implementation:

- run targeted strategy tests,
- run the strategy benchmark comparison command,
- run project quality checks relevant to benchmark Python code.

Minimum command set from `benchmark/`:

- `uv run pytest tests/strategies/test_gated_calibrated_mc.py`
- `uv run pytest tests/strategies/test_calibrated_mc.py tests/strategies/test_learned_calibrator.py`
- `uv run python -m astar_twin.solver.eval.run_gated_calibrated_mc_cv --data-dir data --output results/gated_calibrated_mc_cv.json`

---

## Risks and mitigations

### Risk 1 — overfitting to 15 rounds

Mitigation:

- prefer simple models,
- use leave-one-round-out CV,
- choose models by robustness, not peak score.

### Risk 2 — too little signal for per-cell learning

Mitigation:

- do not do per-cell learning in the first iteration,
- stay with a seed-level selector until cross-validation proves there is headroom.

### Risk 3 — implementation cost drifts upward

Mitigation:

- start with a deterministic NumPy gate,
- do not add external ML frameworks,
- postpone uncertainty rewrites until after the first benchmark win.

### Risk 4 — new strategy matches the mean but not enough to justify complexity

Mitigation:

- require improvement over `calibrated_mc_zones` plus better downside control,
- if not met, keep the analysis artifacts and pivot to uncertainty-enhanced gating rather than discarding the work.

---

## Final commitment

The plan is to **improve the current best-performing benchmark solution by building a simple seed-level selector on top of `calibrated_mc_zones`**.

I am explicitly **not** committing first to generic hyperparameter tuning, variance rewrites, extending the current learned calibrator as-is, or building a cell-level gating system.

The chosen strategy is:

> learn when to trust `calibrated_mc_zones`, and fall back to plain `calibrated_mc` when seed-level evidence says the zone prior is likely to hurt.

That is the most direct path to increasing the current top-line score while reducing the rounds where the present best strategy fails.

---

## Atomic commit strategy

This is a planning artifact, not a request to commit now. But the intended implementation should be split into small reviewable commits:

1. **analysis scaffold**
   - add comparison / CV script skeleton,
   - no strategy behavior changes,
   - tests only for helpers if needed.

2. **strategy test scaffold**
   - add `test_gated_calibrated_mc.py`,
   - add registry test and forced-parent selector tests,
   - commit once tests fail for the right reason.

3. **minimal selector strategy**
   - implement the new strategy with a trivial rule or hardcoded threshold,
   - make the new tests pass,
   - no training pipeline yet.

4. **selector training and CV**
   - add training utility and `run_gated_calibrated_mc_cv` script,
   - commit once leave-one-round-out evaluation is reproducible.

5. **selector tuning pass**
   - adjust features / thresholds to beat `calibrated_mc_zones` under the promotion rule,
   - include updated benchmark result artifact if the repository convention allows it.

6. **optional uncertainty follow-up**
   - only after the selector proves itself,
   - separate commit so the architecture win and the uncertainty refinement are independently reviewable.
