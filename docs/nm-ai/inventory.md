# NM i AI Documentation Inventory

## URI pattern

The MCP follows the pattern `challenge://<challenge-id>/<resource>`.

Common resource types:

- `overview` — task introduction and getting started
- `mechanics` — simulation or game rules
- `endpoint` — HTTP or WebSocket protocol details
- `scoring` — evaluation formula and pitfalls
- `examples` — implementation and debugging guidance
- `submission`, `sandbox`, `quickstart` — challenge-specific operational details

## Catalog exported from the MCP

| Challenge | URI | Purpose |
| --- | --- | --- |
| Grocery Bot | `challenge://game/overview` | Challenge overview and getting started |
| Grocery Bot | `challenge://game/mechanics` | Game mechanics, store layout, difficulty levels |
| Grocery Bot | `challenge://game/endpoint` | HTTP/WebSocket request-response specification |
| Grocery Bot | `challenge://game/scoring` | Score formula, examples, strategy tips |
| Grocery Bot | `challenge://game/examples` | Submission, iteration, and debugging |
| NorgesGruppen Data | `challenge://norgesgruppen-data/overview` | Task overview, downloads, and getting started |
| NorgesGruppen Data | `challenge://norgesgruppen-data/submission` | Submission format, sandbox environment, security rules |
| NorgesGruppen Data | `challenge://norgesgruppen-data/scoring` | mAP@0.5 scoring, category groups, rate limits |
| NorgesGruppen Data | `challenge://norgesgruppen-data/examples` | Code examples, common errors, and tips |
| Tripletex | `challenge://tripletex/overview` | Task overview, how it works, and task categories |
| Tripletex | `challenge://tripletex/endpoint` | `/solve` endpoint specification and API reference |
| Tripletex | `challenge://tripletex/scoring` | Scoring formula, rolling averages, and rate limits |
| Tripletex | `challenge://tripletex/examples` | Code examples, API usage, and debugging tips |
| Tripletex | `challenge://tripletex/sandbox` | Free sandbox account for exploring the API and web UI |
| Astar Island | `challenge://astar-island/overview` | Task overview, concept, and getting started |
| Astar Island | `challenge://astar-island/mechanics` | Simulation mechanics, terrain types, and phases |
| Astar Island | `challenge://astar-island/endpoint` | API endpoint specification for observing and submitting |
| Astar Island | `challenge://astar-island/scoring` | Entropy-weighted KL divergence scoring formula |
| Astar Island | `challenge://astar-island/quickstart` | Authentication, MCP setup, and curl examples |

## Additional docs discovered via search

| URI | Purpose inferred from excerpts |
| --- | --- |
| `challenge://google-cloud/overview` | Competition-provided Google Cloud account overview |
| `challenge://google-cloud/setup` | Cloud Shell, editor, and Gemini setup |
| `challenge://google-cloud/services` | Which GCP services fit each challenge |
| `challenge://google-cloud/deploy` | Cloud Run deployment flow for hosted endpoints |

## Export note

The challenge pages in `challenges/` and `shared/` are excerpt-based snapshots from MCP search results, not perfect full-page mirrors.
