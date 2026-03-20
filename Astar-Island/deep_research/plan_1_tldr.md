## plan_1 TL;DR — implementation plan for solving Astar Island

This plan is based on the analysis of `plan_1.md`, cross-checked against NM AI MCP-derived docs captured in `worktree-5/docs/nm-ai/challenges/astar-island.md` and `worktree-5/docs/nm-ai/spec-confidence-register.md`.

### Document reconciliation

The local source plans conflict. `plan_1.md` describes a live control/pathfinding bot, while the NM-validated Astar contract is a probabilistic forecasting task. This TL;DR follows the NM-validated contract first, uses live docs/API verification for any still-inferred mechanics, and reuses `plan_1.md` only for transferable execution discipline.

### What to keep from plan_1

`plan_1.md` is useful for execution discipline: instrument early, learn from telemetry, keep a fast baseline running, tune continuously, and freeze late. Its biggest weakness is that it treats Astar Island like a live pathfinding/control challenge, while the validated challenge contract is a probabilistic prediction task.

### Validated contract to treat as ground truth

- Output is a `W×H×6` probability tensor.
- Default world size appears to be `40×40`, but implementation should stay dimension-agnostic until the live platform confirms fixed size.
- There are 8 internal terrain types that map to 6 prediction classes.
- Scoring uses entropy-weighted KL divergence.
- Never emit `0.0` for any class; use a floor such as `0.01` and renormalize.
- There is an analysis endpoint that returns your prediction versus ground truth after a round completes.

### Explicit mapping requirement

Before modeling, write down the full 8-to-6 terrain mapping as a checked artifact and use that exact mapping in parsing, inference, validation, and submission generation.

### Step-by-step implementation plan

1. **Lock the real task contract before coding.** Re-check the live NM AI docs/API for any still-inferred details such as authentication flow, query budget, viewport size, seed count, and exact payload field names. Treat the MCP-derived docs above as the current source of truth unless the live platform contradicts them.
2. **Write down the canonical data model.** Define one internal representation for: round metadata, seed metadata, terrain-class mapping, observed cells, hidden-cell beliefs, and final `prediction[y][x][class]` output.
3. **Build a small API client with persistent caching.** Save every fetched response to disk so round metadata, post-round analysis, and any seed-specific artifacts can be replayed without hitting the API again.
4. **Implement strict tensor validation first.** Add a validator that checks dynamic width/height, per-cell normalization, class count, mapping coverage, floor application, and the guarantee that no class falls back to zero after renormalization.
5. **Produce a fully legal end-to-end baseline submission before modeling.** Start with a prior over the six classes using only the known map structure and static terrain assumptions, serialize a valid submission artifact, and confirm the full solve path works from input loading through output validation.
6. **Turn telemetry into features.** Reuse the best idea from `plan_1`: treat every observation and every post-round analysis result as telemetry. Extract features such as distance to coast, mountain adjacency, forest clusters, settlement/port neighborhoods, and observed transition frequency by cell type.
7. **Implement a belief-update layer.** As new observations arrive, update observed cells to near-certain distributions and propagate softer updates to neighboring hidden cells using rules derived from the terrain mapping and observed local structure.
8. **Build a lightweight dynamics model instead of a pathfinder.** Replace the pathfinding-specific algorithms from `plan_1` with a forecasting model that estimates how settlements, ports, ruins, forests, and empty land evolve over time. Start with interpretable transition rules; do not jump to heavyweight models before the replay loop works.
9. **Add an offline replay harness.** Use cached observations and analysis responses to replay completed rounds locally, compare predicted tensors with ground truth, and measure where the model is overconfident or underconfident.
10. **Optimize for KL-divergence safety, not point predictions.** Tune smoothing, confidence caps, and transition probabilities to avoid catastrophic KL penalties. The goal is calibrated uncertainty, not aggressive certainty.
11. **Introduce controlled experimentation.** Keep one safe baseline always runnable, and layer experiments on top of it: better priors, neighborhood features, transition tuning, and ensemble averaging across multiple forecast variants.
12. **Use post-round analysis as the main improvement loop.** After each completed round, diff prediction versus ground truth, record the biggest systematic misses, and update priors/transition rules accordingly. Do not assume any live ground-truth feedback during the active round.
13. **Prepare a submission pipeline early.** Add a single command that loads cached/live inputs, generates the full tensor, validates it, serializes it in the expected wire format, and stores the submitted artifact for later comparison.
14. **Freeze late, but only after replay proves stability.** In the final stage, stop adding model families and focus only on calibration, cache integrity, payload correctness, and small parameter tuning.

### Recommended delivery order

1. contract + validator
2. API client + cache
3. legal baseline tensor
4. replay harness
5. belief updates
6. dynamics model
7. calibration loop using analysis results
8. submission automation

### Main risk to watch

The main risk in `plan_1.md` is solving the wrong problem. Do not spend time on WebSocket control logic, action loops, latency masking, JPS, D* Lite, or CBS unless the live docs prove the task has changed. Reuse the plan's phased execution model, not its pathfinding-specific implementation advice.
