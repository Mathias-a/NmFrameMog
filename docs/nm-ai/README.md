# NM i AI MCP Docs Snapshot

This directory stores a local snapshot of the documentation exposed through the `NM i AI` MCP in this workspace.

## Provenance

- **Catalog source:** live `list_docs` / `resources/list`
- **Content source:** live `resources/read` for listed challenge docs, plus `search_docs` for search-only docs

The live MCP currently has two content modes:

- **Listed challenge resources** — 19 challenge URIs are returned by `list_docs` and `resources/list`, and they are readable as full text through `resources/read`.
- **Search-only docs** — 4 Google Cloud pages are discoverable through `search_docs`, but they are not listed by `list_docs` / `resources/list` and currently fail under `resources/read`.

## Available challenge docs from the MCP catalog

### Grocery Bot Challenge

- `challenge://game/overview`
- `challenge://game/mechanics`
- `challenge://game/endpoint`
- `challenge://game/scoring`
- `challenge://game/examples`

### NorgesGruppen Data (Object Detection)

- `challenge://norgesgruppen-data/overview`
- `challenge://norgesgruppen-data/submission`
- `challenge://norgesgruppen-data/scoring`
- `challenge://norgesgruppen-data/examples`

### Tripletex (AI Accounting Agent)

- `challenge://tripletex/overview`
- `challenge://tripletex/endpoint`
- `challenge://tripletex/scoring`
- `challenge://tripletex/examples`
- `challenge://tripletex/sandbox`

### Astar Island (Norse World Prediction)

- `challenge://astar-island/overview`
- `challenge://astar-island/mechanics`
- `challenge://astar-island/endpoint`
- `challenge://astar-island/scoring`
- `challenge://astar-island/quickstart`

## Extra MCP docs discovered by search

These are live docs returned by `search_docs` even though they are not listed in the main challenge catalog and are not currently readable via `resources/read`:

- `challenge://google-cloud/overview`
- `challenge://google-cloud/setup`
- `challenge://google-cloud/services`
- `challenge://google-cloud/deploy`

## Layout

- `inventory.md` — full URI inventory and discovery-mode notes
- `spec-confidence-register.md` — what is validated vs inferred vs still unknown
- `challenges/` — local challenge summaries aligned to the live readable docs
- `shared/` — cross-cutting platform and infrastructure guidance, including search-only pages
