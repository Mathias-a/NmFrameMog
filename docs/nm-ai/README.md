# NM i AI MCP Docs Snapshot

This directory stores a local snapshot of the documentation exposed through the `NM i AI` MCP in this workspace.

## Provenance

- **Catalog source:** `NM-AI-docs_list_docs`
- **Content source:** `NM-AI-docs_search_docs`

The MCP reliably exposes a catalog of challenge URIs. Its content retrieval works as excerpt search rather than full-page export, so the files below preserve all retrievable snippets gathered during this session and clearly label them as MCP excerpts.

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

These were returned by the same docs server even though they were not listed in the initial catalog response:

- `challenge://google-cloud/overview`
- `challenge://google-cloud/setup`
- `challenge://google-cloud/services`
- `challenge://google-cloud/deploy`

## Layout

- `inventory.md` — full URI inventory and organization notes
- `spec-confidence-register.md` — what is validated vs inferred vs still unknown
- `challenges/` — excerpt-based challenge snapshots
- `shared/` — cross-cutting platform and infrastructure guidance
