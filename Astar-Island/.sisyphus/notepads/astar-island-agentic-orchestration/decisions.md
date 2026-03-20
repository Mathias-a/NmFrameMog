2026-03-20 — Task 1
- Added `solver/evaluation_contract.py` as the canonical artifact identity layer for frozen manifests, shared round traces, candidate bundles, organizer analyses, benchmark inputs, and benchmark reports.
- Bound dataset and benchmark artifacts to the existing terrain mapping by storing and validating the canonical mapping hash, which fails closed on mapping drift.
- Kept candidate predictions separate from organizer analysis payloads so downstream replay and benchmark tasks can compare artifacts without conflating submitted tensors with organizer-provided ground truth.
2026-03-20 — Task 2
- Implemented `solver/dataset_refresh.py` as a refresh-only lane that fetches round detail and per-seed analysis from the live API, copies cached query payloads into `datasets/<version>/`, and emits `manifest.json`, `hashes.json`, and `query-trace.json` using the Task 1 contract types.
- Made dataset freezing append-only by writing to a temporary sibling directory and renaming into `datasets/<version>/` only after every required artifact, hash, and seed analysis is validated, so partial refreshes leave no benchmark corpus behind.
2026-03-20 — Task 3
- Added `solver/replay_harness.py` as the offline-only replay lane: it reconstructs `FrozenDatasetManifest`, `RoundQueryTrace`, organizer analyses, a candidate prediction bundle, and `BenchmarkInput` directly from frozen artifacts under `datasets/<version>/`.
- Kept replay extensible by separating candidate loading (`CandidateBundleAdapter`) from benchmark-input assembly (`BenchmarkInputAssembler`), so later solver-candidate and workflow-candidate integrations can plug in without changing the deterministic replay core.
2026-03-20 — Task 5
- Added `solver/benchmark_suite.py` as the Task 5 metric layer, with explicit hard failures for integrity or legality drift and a `BenchmarkSuiteReport` that surfaces per-seed KL-derived metrics, aggregate score, calibration, stability, fallback, and baseline-delta-ready structures.
- Kept baseline and last-blessed handling structural only: the suite can compare optional reference bundles and report exact-match fallback plus score deltas now, while leaving promotion-state persistence and policy decisions to later tasks.

2026-03-20 — Task 4
- Added a root `AGENTS.md` that maps Astar work into capture, solve, evaluate, and report lanes with exact `python -m nmframemog.astar_island ...` entry commands, artifact boundaries, and explicit stop conditions.
- Added `tests/astar/test_agents_doc.py` to enforce required sections, lane ownership fields, the evaluation-only promotion rule, and a minimality guardrail against duplicating top-level `CLAUDE.md` policy.

2026-03-20 — Task 6
- Extended `solver/dataset_refresh.py` with a shared round-capture helper, `refresh_history_snapshot()` for append-only completed-round backfill into `history/raw/<snapshot-version>/`, and `build_curated_history_dataset()` for immutable curated corpora built from selected history snapshots.
- Kept raw history captures distinct from curated benchmark manifests by storing per-round frozen manifests and query traces in the raw snapshot, then composing new versioned curated datasets instead of mutating older `datasets/<version>/` trees when new rounds arrive.
