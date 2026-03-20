# Task 2 — Live MCP vs Local Docs Parity Report

## Summary

The repo already captured the broad challenge inventory correctly, but several local docs are now stale because the live MCP returns richer data than the repo currently describes.

## Highest-priority mismatches

1. `docs/nm-ai/README.md` still says content retrieval works as excerpt search, but the 19 listed challenge resources now return full readable bodies through `resources/read`.
2. `docs/nm-ai/inventory.md` still frames the `challenges/` pages as excerpt snapshots. That is stale for the 19 listed challenge resources.
3. `docs/nm-ai/spec-confidence-register.md` still marks several rules as inferred or unknown even though the live MCP now exposes the full answer.
4. The 4 Google Cloud docs are still real and useful, but they are **not** normal resources. They are search-discovered only.
5. Transport reliability is not documented anywhere locally. The server frequently requires retrying fresh sessions.

## File-by-file comparison matrix

| Local file | Local claim / state | Live evidence | Required correction |
| --- | --- | --- | --- |
| `docs/nm-ai/README.md` | Says retrieval works as excerpt search rather than full-page export | `resources/read` returns full bodies for all 19 listed challenge resources | Reword to distinguish **full listed resources** from **search-only Google Cloud excerpts** |
| `docs/nm-ai/inventory.md` | Says `challenges/` pages are excerpt-based snapshots | Live listed challenge resources are fully readable | Update export note and distinguish listed vs search-only surfaces |
| `docs/nm-ai/spec-confidence-register.md` | Grocery score formula = Unknown | Live `challenge://game/scoring` exposes full formula, leaderboard rules, end conditions | Mark as Validated |
| `docs/nm-ai/spec-confidence-register.md` | Tripletex rate limits = Inferred | Live `challenge://tripletex/scoring` exposes `10` concurrent / `Unlimited` daily | Mark as Validated |
| `docs/nm-ai/spec-confidence-register.md` | Tripletex common errors = Inferred | Live `challenge://tripletex/examples` exposes full Common Errors table | Mark as Validated |
| `docs/nm-ai/spec-confidence-register.md` | Astar auth/setup = Inferred | Live `challenge://astar-island/quickstart` and endpoint docs expose full auth and workflow | Mark as Validated |
| `docs/nm-ai/spec-confidence-register.md` | NG sandbox constraints = Inferred | Live `challenge://norgesgruppen-data/submission` exposes CPU/GPU/memory/network/package/security limits | Mark as Validated |
| `docs/nm-ai/spec-confidence-register.md` | NG common errors = Inferred | Live `challenge://norgesgruppen-data/examples` exposes full Common Errors table | Mark as Validated |
| `docs/nm-ai/shared/google-cloud.md` | Described as discovered by search | Still true | Keep, but label these as search-only and not readable via `resources/read` |

## Additional live findings not reflected clearly in local docs

### MCP transport and discovery behavior

- `notifications/initialized` is unreliable and should not be treated as a strict readiness signal.
- `resources.subscribe` is unavailable because the server advertises `subscribe = false`.
- `prompts/list` is empty despite the capability existing.
- `resources/templates/list` is empty.

### Google Cloud discovery gap

The live MCP has a split discovery model:

- **Listed + readable:** 19 challenge resources
- **Search-only:** 4 Google Cloud URIs

That means a full audit must use both:

- `list_docs` / `resources/list`
- `search_docs`

Relying on listing alone misses live content.

### Content drift inside challenge pages

The local challenge pages are good summaries, but they are lighter than the live docs in several important places:

- Grocery Bot live docs expose the exact scoring formula and rate limits
- Tripletex live docs expose tier multipliers, efficiency bonus details, and concrete rate limits
- Astar Island live docs expose the full REST surface, request/response schemas, validation errors, and leaderboard payloads
- NorgesGruppen live docs expose exact zip limits, GPU environment, package versions, blocked imports, and zip-packaging pitfalls

### Live-doc inconsistencies worth preserving

- Astar Island has an internal seed-count inconsistency in the live docs:
  - quickstart example uses `seeds_count = 5` and `seed_index` values `0–4`
  - endpoint prose also says “submit all 15 seeds” and labels `seed_index` as `0–4`
- Google Cloud pages are query-sensitive in `search_docs`:
  - broad queries like `google cloud` or `cloud run` return useful excerpts
  - narrower queries like `google cloud deploy` can return “No results found” even though the deploy page is still reachable through broader searches

These should be documented as live behavior, not normalized away.

## Correction priorities

### P0 — must be corrected to avoid wrong decisions

- Excerpt-only framing for the listed challenge resources
- Confidence-register rows that are now fully validated
- Missing note that Google Cloud pages are search-only

### P1 — should be corrected for completeness

- Expand local challenge summaries so they include the newly visible scoring, rate-limit, and endpoint details
- Add MCP transport reliability notes for future audits

### P2 — useful but not blocking

- Add a reusable audit recipe for future reruns
- Timestamp the observed surface as a snapshot because `listChanged` is advertised

## Final parity verdict

The local docs were directionally correct, but they understate the amount of live content now returned by the MCP. After correction, the repo should present the surface like this:

- 2 live tools
- 0 prompts
- 0 resource templates
- 19 listed and readable challenge resources
- 4 search-only Google Cloud docs
- flaky session lifecycle that requires retries
