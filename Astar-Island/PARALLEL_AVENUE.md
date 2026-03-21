# Worktree 7 Avenue — Query Allocation and Hotspot Redesign

## Mission Anchor

The challenge is bottlenecked by 50 total queries shared across 5 seeds. The current solver already has a structured budget schedule, but the docs emphasize that query allocation and viewport placement are among the highest-value strategy levers.

## Why This Avenue Is High-Value

The current policy stack is concentrated in:

- `benchmark/src/astar_twin/solver/policy/hotspots.py`
- `benchmark/src/astar_twin/solver/policy/allocator.py`
- `benchmark/src/astar_twin/solver/pipeline.py`

Right now the solver uses a fixed 10 / 30 / 10 bootstrap-adaptive-reserve split, fixed hotspot categories, overlap rejection, and reserve release rules keyed mainly off ESS collapse and under-queried seeds. That is a strong baseline, but it likely leaves score on the table when rounds differ in settlement density, coastline structure, or posterior uncertainty patterns.

## Core Hypothesis

The biggest near-term gain may come from spending the same 50 queries more intelligently. The plan should explore:

1. Dynamic budget schedules instead of fixed per-phase counts.
2. Cross-seed value-of-information, since shared hidden parameters make some seeds more informative than others.
3. Better contradiction probes and reserve release logic.
4. Hotspot generation that better matches dynamic terrain transitions rather than static geometry alone.

## Primary Files To Analyze

- `Astar-Island/docs/overview.md`
- `Astar-Island/docs/mechanics.md`
- `Astar-Island/docs/RULESET.md`
- `Astar-Island/AGENTS.md`
- `Astar-Island/benchmark/src/astar_twin/solver/policy/hotspots.py`
- `Astar-Island/benchmark/src/astar_twin/solver/policy/allocator.py`
- `Astar-Island/benchmark/src/astar_twin/solver/pipeline.py`
- `Astar-Island/benchmark/src/astar_twin/solver/eval/run_benchmark_suite.py`

## Questions You Should Answer In The Plan

1. Which current hotspot categories are most aligned with the dynamic cells that dominate score?
2. Is the fixed 2-queries-per-seed bootstrap too rigid for rounds with uneven information density?
3. Can adaptive selection target parameter-identifying windows earlier, not just uncertain windows later?
4. How should reserve queries be spent when no contradiction trigger fires but uncertainty remains concentrated?

## Constraints

- Keep the total query budget at 50.
- Respect viewport size limits and determinism requirements.
- Treat the simulator and scoring layers as stable dependencies.
- Do not expand scope into engine or phase logic changes.

## Scope Boundary

This avenue owns **query selection and budget use**, not posterior math or benchmark tooling. That means:

- In scope: bootstrap scheduling, adaptive selection, reserve release policy, hotspot generation, transcript-level policy evaluation.
- Out of scope: likelihood redesign, final tensor calibration, hedge logic tuning, replay tooling, or benchmark report schema changes.

## Success Criteria

- A plan for improving mean score without increasing query count.
- Explicit comparison criteria against the current allocator and hotspot generator.
- A strategy for measuring transcript quality, not just final score.

## Expected Output From The Planning Agent

Produce a plan that turns query allocation into a measurable experimental program: what to change first, how to benchmark it, and how to tell whether the policy is learning the shared hidden parameters more efficiently.
