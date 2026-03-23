# Worktree 8 Avenue — Posterior Predictive, Final Tensor Calibration, and Hedge Logic

## Mission Anchor

The final score is determined by the submitted probability tensor, not by the internal posterior alone. The docs also make score safety explicit: zero probabilities are catastrophic, and dynamic high-entropy cells matter most.

## Why This Avenue Is High-Value

The current final prediction path lives in:

- `benchmark/src/astar_twin/solver/predict/posterior_mc.py`
- `benchmark/src/astar_twin/solver/predict/hedge.py`
- `benchmark/src/astar_twin/solver/predict/finalize.py`
- `benchmark/src/astar_twin/scoring/*`

Today the solver uses a fixed top-k particle mix, proportional run allocation, a runtime fallback, and a fixed 85/15 hedge against fixed coverage. That likely leaves room for better calibration, especially on dynamic cells that dominate entropy-weighted KL.

## Core Hypothesis

Even if the posterior stays mostly the same, a stronger final aggregation and calibration layer could raise benchmark score. The plan should explore:

1. Better simulation-run allocation across particles and seeds.
2. More adaptive top-k selection.
3. Smarter hedge activation rules and blending weights.
4. Calibration methods that focus on dynamic terrain classes rather than static cells.

## Primary Files To Analyze

- `Astar-Island/docs/overview.md`
- `Astar-Island/docs/scoring.md`
- `Astar-Island/docs/RULESET.md`
- `Astar-Island/AGENTS.md`
- `Astar-Island/benchmark/src/astar_twin/solver/predict/posterior_mc.py`
- `Astar-Island/benchmark/src/astar_twin/solver/predict/hedge.py`
- `Astar-Island/benchmark/src/astar_twin/solver/predict/finalize.py`
- `Astar-Island/benchmark/src/astar_twin/scoring/`
- `Astar-Island/benchmark/src/astar_twin/solver/eval/run_benchmark_suite.py`

## Questions You Should Answer In The Plan

1. Are the current run-allocation rules over-spending on weak particles or under-spending on high-uncertainty seeds?
2. How often does hedge activation help versus merely covering for calibration problems upstream?
3. Can per-class or per-cell calibration improve entropy-weighted KL without overfitting?
4. Should runtime fallback depend on uncertainty structure, not just elapsed runtime fraction?

## Constraints

- Preserve valid `(H, W, 6)` tensors and deterministic output.
- Never introduce zero probabilities.
- Keep evaluation comparable to the current benchmark suite.
- Treat the simulator and parameter inference layers as upstream dependencies, not the main target.

## Scope Boundary

This avenue owns **final tensor construction and output calibration**, not upstream posterior math. That means:

- In scope: top-k particle selection, simulation-run allocation, runtime fallback logic, hedge activation, blend weights, calibration of final probabilities.
- Out of scope: likelihood weighting, posterior resampling rules, benchmark diagnostics/reporting, or new registry strategy packaging.

## Success Criteria

- A plan for raising benchmark mean through better final tensor construction.
- Clear experiments around hedge logic, calibration, and particle aggregation.
- Explicit safeguards against score regressions on high-entropy cells.

## Expected Output From The Planning Agent

Produce a plan that explains how the current final tensor is built, where calibration can improve, and how to validate gains without confusing upstream inference quality with downstream prediction quality.
