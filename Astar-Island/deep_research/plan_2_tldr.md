## plan_2 TL;DR — implementation plan for solving Astar Island

This plan is based on the analysis of `plan_2.md`, cross-checked against NM AI MCP-derived docs captured in `worktree-5/docs/nm-ai/challenges/astar-island.md` and `worktree-5/docs/nm-ai/spec-confidence-register.md`.

### Document reconciliation

The local source plans conflict. This TL;DR follows the NM-validated contract first, treats `plan_2.md` as the strategic base because it matches the forecasting problem shape, and requires live doc/API verification for any mechanics that remain inferred rather than validated.

### Why plan_2 is the stronger starting point

`plan_2.md` matches the validated task shape much better than `plan_1.md`: it treats Astar Island as a partially observable forecasting problem with probabilistic outputs. That makes it the better blueprint for the actual implementation, although some details in `plan_2.md` should still be treated as hypotheses until re-confirmed against the live platform.

### Validated contract to treat as ground truth

- Output is a `W×H×6` probability tensor.
- Default world size appears to be `40×40`, but implementation should stay dimension-agnostic until the live platform confirms fixed size.
- There are 8 internal terrain types that map to 6 prediction classes.
- Scoring uses entropy-weighted KL divergence.
- Never emit `0.0` for any class; use a floor such as `0.01` and renormalize.
- There is an analysis endpoint that returns your prediction versus ground truth after a round completes.

### Explicit mapping requirement

Before modeling, write down the full 8-to-6 terrain mapping as a checked artifact and use that exact mapping in parsing, inference, validation, and submission generation.

### Details from `plan_2.md` that still need live confirmation

- Exact query budget per round.
- Exact viewport size.
- Number of seeds per round.
- Full list of hidden simulation variables.

### Step-by-step implementation plan

1. **Confirm the remaining mechanics against the live docs/API.** Before building the query strategy, verify the exact query budget, viewport shape, round structure, seed count, authentication flow, and request/response schema.
2. **Define the canonical terrain mapping.** Create one explicit mapping from internal terrain codes to the six output classes and use it everywhere: parser, simulator, predictor, validator, and submission serializer.
3. **Implement the challenge state container.** Track, per seed: initial map, all queried viewports, observed cell histories, unresolved hidden cells, and the current probability tensor.
4. **Build a cache-first data collection layer.** Every query response, round summary, and post-round analysis payload should be stored locally so model work can continue offline.
5. **Create a fully legal end-to-end baseline immediately.** Start with a simple prior that combines static geography and known terrain constraints, emit a valid `W×H×6` tensor with floor-and-renormalize logic, serialize it, and verify the entire solve path before adding model complexity.
6. **Implement submission validation as a hard gate.** Check dynamic width/height, class count, mapping coverage, per-cell normalization, floor application, and the guarantee that no class returns to zero after renormalization.
7. **Implement a query planner instead of manual querying.** Rank candidate viewports by expected information gain, frontier coverage, and uncertainty reduction so each limited query improves the future tensor as much as possible.
8. **Build feature extraction from observed neighborhoods.** Use local patterns such as coastline shape, mountain barriers, forest clusters, settlement neighborhoods, and ruin/port adjacency to estimate the likely future state of unseen cells.
9. **Implement a stochastic transition model.** Represent how dynamic classes evolve over time using interpretable probabilities or an ensemble of simple models. The first goal is stable calibration, not perfect simulation fidelity.
10. **Run Monte Carlo or ensemble rollouts offline.** Use repeated sampled futures to convert the transition model into per-cell class probabilities rather than hard labels.
11. **Aggregate rollouts into the submission tensor.** Average outcomes across rollouts, apply the `0.01` floor, renormalize, and validate every cell before submission.
12. **Use the analysis endpoint as a calibration oracle.** After the round completes, compare predictions with ground truth, identify systematic bias, and update both the query planner and the transition model. Do not imply live ground-truth feedback during the active round.
13. **Tune for entropy-weighted KL divergence explicitly.** Penalize overconfidence during calibration, cap confidence in uncertain regions, and prefer a slightly broader distribution when evidence is weak.
14. **Add experiment tracking.** Save model config, query choices, tensor snapshots, and score results for each run so you can identify what actually improved calibration.
15. **Automate the end-to-end solve path.** One command should: load cached/live state, select queries, update beliefs, generate rollouts, emit a validated tensor, save the artifact, and submit.
16. **Freeze on reliability, not novelty.** In the last phase, stop changing the model family. Focus on better calibration, safer query allocation, cache completeness, and serializer correctness.

### Recommended delivery order

1. contract check + terrain mapping
2. state container + cache
3. legal baseline tensor
4. query planner
5. stochastic transition model
6. rollout aggregation
7. post-round calibration loop
8. end-to-end submission command

### Main risk to watch

The biggest risk in `plan_2.md` is treating inferred simulator details as guaranteed facts. Keep the probabilistic forecasting architecture, but validate every mechanical assumption against the live NM AI docs/API before optimizing around it.
