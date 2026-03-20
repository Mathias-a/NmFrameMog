# Task 1 — Live MCP Audit Evidence

## Scope

This document records the live surface exposed by the configured MCP at `https://mcp-docs.ainm.no/mcp`.

## Audit method

- Transport: remote HTTP MCP endpoint from `.opencode/opencode.json`
- Initialization observed from live server:
  - `serverInfo.name`: `nmiai-challenge`
  - `serverInfo.version`: `3.1.1`
  - `protocolVersion`: `2025-11-25`
- Capability negotiation observed:
  - `tools.listChanged = true`
  - `prompts.listChanged = true`
  - `resources.listChanged = true`
  - `resources.subscribe = false`
  - `extensions.io.modelcontextprotocol/ui = {}`
- Reliability note:
  - `notifications/initialized` is not stable. Some calls return `202`, others return `Session not found`, while the follow-up MCP call still succeeds.
  - For this server, the safe audit pattern is **fresh initialize + one target query + retry**.

## Query coverage ledger

| Method / surface | Result | Notes |
| --- | --- | --- |
| `initialize` | succeeded | Stable enough to capture server metadata |
| `notifications/initialized` | flaky | Returned both `202` and `Session not found` |
| `tools/list` | succeeded | 2 tools exposed |
| `tools/call` → `list_docs` | succeeded | Returns 19 listed challenge resources |
| `tools/call` → `search_docs` | succeeded | Reveals additional search-only docs and excerpt snippets |
| `prompts/list` | succeeded | Empty list |
| `resources/list` | succeeded | 19 listed challenge resources |
| `resources/templates/list` | succeeded | Empty list |
| `resources/read` for 19 listed challenge resources | succeeded | Full readable bodies returned |
| `resources/read` for `challenge://google-cloud/*` | failed | `Unknown resource`; these docs are search-discovered only |
| `resources/subscribe` | not applicable | Capability explicitly says `subscribe = false` |

## Exposed tools

### `list_docs`

- Purpose: returns the listed challenge documentation catalog
- Input schema: empty object
- Output schema: wrapped object with `result: string`
- Important limitation: the returned catalog is **not exhaustive**. It omits the Google Cloud docs that are still reachable through `search_docs`.

### `search_docs`

- Purpose: search challenge documentation by free-text query
- Input schema:
  - `query: string` (required)
- Output schema: wrapped object with `result: string`
- Important behavior:
  - It returns excerpt blocks grouped by `challenge://...` URI
  - It can expose docs not returned by `list_docs` or `resources/list`
  - It is currently the only live way to reach the Google Cloud pages

## Empty surfaces

| Surface | Live result |
| --- | --- |
| Prompts | none exposed |
| Resource templates | none exposed |

## Listed readable resources

These 19 resources are returned by both `resources/list` and the challenge catalog from `list_docs`, and each one was read successfully with `resources/read`.

| URI | Category | Discovery | `resources/read` | Notes |
| --- | --- | --- | --- | --- |
| `challenge://game/overview` | Grocery Bot | listed | yes | Full challenge overview |
| `challenge://game/mechanics` | Grocery Bot | listed | yes | Includes map/difficulty mechanics |
| `challenge://game/endpoint` | Grocery Bot | listed | yes | Full WebSocket protocol |
| `challenge://game/scoring` | Grocery Bot | listed | yes | Full score formula and limits |
| `challenge://game/examples` | Grocery Bot | listed | yes | Submission guide, rate limits, example bot |
| `challenge://norgesgruppen-data/overview` | NorgesGruppen Data | listed | yes | Full task overview and dataset details |
| `challenge://norgesgruppen-data/submission` | NorgesGruppen Data | listed | yes | Full zip contract, sandbox limits, security rules |
| `challenge://norgesgruppen-data/scoring` | NorgesGruppen Data | listed | yes | Full scoring and submission limits |
| `challenge://norgesgruppen-data/examples` | NorgesGruppen Data | listed | yes | Baselines, ONNX, common errors |
| `challenge://tripletex/overview` | Tripletex | listed | yes | Full task overview |
| `challenge://tripletex/endpoint` | Tripletex | listed | yes | Full `/solve` contract |
| `challenge://tripletex/scoring` | Tripletex | listed | yes | Full scoring, tiers, efficiency, rate limits |
| `challenge://tripletex/examples` | Tripletex | listed | yes | Endpoint example, patterns, errors |
| `challenge://tripletex/sandbox` | Tripletex | listed | yes | Sandbox setup and API usage |
| `challenge://astar-island/overview` | Astar Island | listed | yes | Full challenge overview |
| `challenge://astar-island/mechanics` | Astar Island | listed | yes | Terrain and simulator mechanics |
| `challenge://astar-island/endpoint` | Astar Island | listed | yes | Full REST API reference |
| `challenge://astar-island/scoring` | Astar Island | listed | yes | Full entropy-weighted KL scoring |
| `challenge://astar-island/quickstart` | Astar Island | listed | yes | Full auth and submission quickstart |

## Search-only docs

These URIs are reachable through `search_docs` but are not present in `list_docs`, not present in `resources/list`, and currently fail under `resources/read`.

| URI | Discovery path | `resources/read` | What live search exposed |
| --- | --- | --- | --- |
| `challenge://google-cloud/overview` | `search_docs` | no | Free GCP project, no credit limits, collaboration tools |
| `challenge://google-cloud/setup` | `search_docs` | no | Cloud Shell includes Python, git, gcloud, Docker |
| `challenge://google-cloud/services` | `search_docs` | no | Cloud Run for Tripletex/Astar; Compute Engine for GPU/persistent workloads |
| `challenge://google-cloud/deploy` | `search_docs` | no | Cloud Run deployment guidance for hosted validator endpoints |

## Endpoint-level highlights captured from live resources

### Grocery Bot

- WebSocket URL format: `wss://game.ainm.no/ws?token=<jwt_token>`
- Message loop:
  - server sends `game_state`
  - client replies with `{ "actions": [...] }`
  - terminal event is `game_over`
- Scoring formula is fully visible:
  - `score = items_delivered × 1 + orders_completed × 5`
- Rate limits are fully visible:
  - 60 second cooldown
  - 40 games/hour
  - 300 games/day

### Tripletex

- Hosted endpoint: `POST /solve`
- Request includes:
  - `prompt`
  - optional `files[]`
  - `tripletex_credentials.base_url`
  - `tripletex_credentials.session_token`
- Response contract: `{"status":"completed"}`
- Auth contract: Basic Auth with username `0` and password = `session_token`
- Scoring now exposes concrete tier multipliers, efficiency scoring, and rate limits (`10` concurrent, `Unlimited` per day)

### Astar Island

- Base URL: `https://api.ainm.no/astar-island`
- Live docs expose these REST endpoints:
  - `GET /astar-island/rounds`
  - `GET /astar-island/rounds/{round_id}`
  - `GET /astar-island/budget`
  - `POST /astar-island/simulate`
  - `POST /astar-island/submit`
  - `GET /astar-island/my-rounds`
  - `GET /astar-island/my-predictions/{round_id}`
  - `GET /astar-island/analysis/{round_id}/{seed_index}`
  - `GET /astar-island/leaderboard`
- Submission tensor format is fully documented as `H × W × 6`
- Quickstart exposes the auth flow and the 50-query round budget

### NorgesGruppen Data

- `run.py` contract is explicit
- Submission limits and sandbox resource limits are explicit
- Security restrictions are explicit
- Scoring formula is explicit:
  - `0.7 × detection_mAP + 0.3 × classification_mAP`

## Reliability conclusion

The authoritative live surface is:

1. the 2 tools from `tools/list`
2. the empty prompt/template surfaces
3. the 19 readable challenge resources from `resources/list`
4. the 4 additional Google Cloud docs discoverable only via `search_docs`

Any consumer of this MCP should treat `list_docs` and `resources/list` as **incomplete discovery surfaces**, and should treat session retries as normal behavior.
