# NM i AI Spec Confidence Register

This register keeps the team honest about which challenge rules are fully validated versus only inferred from MCP excerpts.

Confidence levels:

- **Validated** — explicitly visible in MCP output or confirmed by runnable environment behavior
- **Inferred from excerpt** — partially visible in MCP output but not fully exported
- **Unknown** — mentioned by the docs catalog but not yet confirmed in detail

## Grocery Bot

| Rule / detail | Confidence | Evidence | Follow-up |
| --- | --- | --- | --- |
| Task uses WebSocket state/action loop | Validated | `challenge://game/overview`, `challenge://game/endpoint` excerpts | None |
| `game_state` and `game_over` message types exist | Validated | `challenge://game/endpoint`, `challenge://game/examples` excerpts | None |
| Maximum game length is 300 rounds | Validated | `challenge://game/mechanics`, `challenge://game/endpoint` excerpts | None |
| Difficulty levels vary grid size, bot count, aisles, item types, maps, and time limit | Inferred from excerpt | Heading/table shape exposed, not full table contents | Capture fuller table later |
| Full scoring formula | Validated | `challenge://game/scoring` full live resource now exposes `score = items_delivered × 1 + orders_completed × 5` plus leaderboard rules | None |
| Game cooldown of 60 seconds | Validated | `challenge://game/examples` excerpt | Verify against platform behavior |

## NorgesGruppen Data

| Rule / detail | Confidence | Evidence | Follow-up |
| --- | --- | --- | --- |
| Submission is a `.zip` executed in sandboxed Docker | Validated | `challenge://norgesgruppen-data/overview` excerpt | None |
| `run.py` must be at zip root | Validated | `challenge://norgesgruppen-data/submission` excerpt | None |
| Scoring includes mAP@0.5 | Validated | `challenge://norgesgruppen-data/scoring` excerpt | None |
| Final score combines detection and classification | Validated | `Hybrid Scoring` excerpt | None |
| Sandbox environment has explicit CPU / memory / GPU / network / timeout constraints | Validated | `challenge://norgesgruppen-data/submission` full live resource | None |
| Common errors table exists with concrete fixes | Validated | `challenge://norgesgruppen-data/examples` full live resource | None |

## Tripletex

| Rule / detail | Confidence | Evidence | Follow-up |
| --- | --- | --- | --- |
| Hosted HTTPS `/solve` endpoint is required | Validated | `challenge://tripletex/endpoint` excerpt | None |
| Requests contain `tripletex_credentials.base_url` and `session_token` | Validated | `challenge://tripletex/endpoint` excerpt | None |
| Auth uses username `0` and password=`session_token` | Validated | `challenge://tripletex/endpoint` excerpt | None |
| Judge verifies resulting API state field-by-field | Validated | `challenge://tripletex/scoring` excerpt | None |
| Rate limits exist and are explicit (`10` concurrent, unlimited per day) | Validated | `challenge://tripletex/scoring` full live resource | None |
| Common errors table exists with concrete causes and fixes | Validated | `challenge://tripletex/examples` full live resource | None |

## Astar Island

| Rule / detail | Confidence | Evidence | Follow-up |
| --- | --- | --- | --- |
| Output is a `W×H×6` probability tensor | Validated | `challenge://astar-island/overview` excerpt | None |
| Score uses entropy-weighted KL divergence | Validated | `challenge://astar-island/overview`, `challenge://astar-island/scoring` excerpts | None |
| Zero probabilities can destroy score | Validated | `challenge://astar-island/quickstart`, `challenge://astar-island/scoring` excerpts | None |
| Analysis endpoint returns prediction vs ground truth post-round | Validated | `challenge://astar-island/endpoint` excerpt | None |
| World defaults to 40×40 and 8 terrain types map to 6 classes | Validated | `challenge://astar-island/mechanics` excerpt | None |
| Full auth/setup details | Validated | `challenge://astar-island/quickstart` and `challenge://astar-island/endpoint` full live resources | None |

## Google Cloud guidance

| Rule / detail | Confidence | Evidence | Follow-up |
| --- | --- | --- | --- |
| Tripletex and Astar Island are good Cloud Run fits | Validated | `challenge://google-cloud/deploy`, `challenge://google-cloud/services` excerpts | None |
| Cloud Shell includes Python, git, gcloud, and Docker | Validated | `challenge://google-cloud/setup` excerpt | None |
| Selected teams get free GCP project access | Validated | `challenge://google-cloud/overview` excerpt | Confirm provisioning process when needed |
