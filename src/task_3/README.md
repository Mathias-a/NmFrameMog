# Task 3 — Final MCP Reference for This Project

## What this MCP is

The configured server is a remote documentation MCP for NM i AI competition materials.

- Server: `nmiai-challenge`
- Version: `3.1.1`
- Protocol: `2025-11-25`
- Transport: remote HTTP MCP endpoint at `https://mcp-docs.ainm.no/mcp`

## How to discover everything safely

Use this order:

1. `tools/list`
2. `prompts/list`
3. `resources/list`
4. `resources/templates/list`
5. `tools/call(list_docs)`
6. `tools/call(search_docs)` with targeted topic queries

Do **not** stop after `list_docs` or `resources/list`. They miss the Google Cloud docs.

## Tool reference

### `list_docs`

- Returns the main listed challenge catalog
- Good for initial inventory
- Not exhaustive

### `search_docs`

- Returns excerpt matches across docs
- Required for full coverage
- Only live path to the Google Cloud pages

## Empty surfaces right now

| Surface | Observed state |
| --- | --- |
| Prompts | empty |
| Resource templates | empty |
| Resource subscription | unsupported (`subscribe = false`) |

## Listed challenge resources

### Grocery Bot

- `challenge://game/overview`
- `challenge://game/mechanics`
- `challenge://game/endpoint`
- `challenge://game/scoring`
- `challenge://game/examples`

What they cover:

- real-time WebSocket gameplay
- `game_state` / `game_over` protocol
- exact scoring formula
- leaderboard rules and cooldown limits
- Python bot example

### NorgesGruppen Data

- `challenge://norgesgruppen-data/overview`
- `challenge://norgesgruppen-data/submission`
- `challenge://norgesgruppen-data/scoring`
- `challenge://norgesgruppen-data/examples`

What they cover:

- dataset and product-category setup
- exact `run.py` contract
- zip limits and allowed file types
- sandbox hardware/software restrictions
- common packaging and model-version failures

### Tripletex

- `challenge://tripletex/overview`
- `challenge://tripletex/endpoint`
- `challenge://tripletex/scoring`
- `challenge://tripletex/examples`
- `challenge://tripletex/sandbox`

What they cover:

- hosted `POST /solve` endpoint contract
- request and response schemas
- proxy/base URL and session-token auth model
- tiered scoring and efficiency bonus
- example API flows and error table

### Astar Island

- `challenge://astar-island/overview`
- `challenge://astar-island/mechanics`
- `challenge://astar-island/endpoint`
- `challenge://astar-island/scoring`
- `challenge://astar-island/quickstart`

What they cover:

- forecasting task definition
- terrain/class mapping
- full REST API surface
- submission tensor format and validation rules
- entropy-weighted KL scoring and zero-probability pitfall

## Search-only docs

These are reachable by `search_docs` but not by `resources/list` and currently fail under `resources/read`.

- `challenge://google-cloud/overview`
- `challenge://google-cloud/setup`
- `challenge://google-cloud/services`
- `challenge://google-cloud/deploy`

What they add:

- free GCP project details
- Cloud Shell setup
- Cloud Run vs Compute Engine guidance
- deployment guidance for Tripletex and Astar Island hosted endpoints

## Operational caveats

### Discovery is split

- `list_docs` and `resources/list` cover the main challenge resources
- `search_docs` is required to discover the Google Cloud pages

### Session lifecycle is noisy

- `notifications/initialized` can return `202` or `Session not found`
- the target MCP call can still succeed afterward
- best practice is fresh session per target call plus retries

### Inventories are snapshots, not guarantees

The server advertises `listChanged`, so this reference should be treated as a current observed snapshot rather than an immutable contract.

## Coverage checklist

- [x] `initialize` audited
- [x] `tools/list` audited
- [x] `tools/call(list_docs)` audited
- [x] `tools/call(search_docs)` audited
- [x] `prompts/list` audited
- [x] `resources/list` audited
- [x] `resources/templates/list` audited
- [x] all 19 listed challenge resources read
- [x] all 4 search-only Google Cloud URIs queried through search
- [x] session reliability issues documented
