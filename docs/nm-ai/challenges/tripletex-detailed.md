# Tripletex AI Accounting Agent -- Detailed Research

Last updated: 2026-03-19

## Challenge Overview

Build an AI agent that completes accounting tasks in Tripletex, a Norwegian ERP system.
The agent must expose a single HTTPS endpoint and interact with the Tripletex v2 REST API
to fulfill natural-language accounting prompts.

| Fact | Status |
|------|--------|
| 30 distinct accounting task types | CONFIRMED |
| 56 variants per task (7 languages x 8 data sets) | CONFIRMED |
| Total possible submissions: 30 x 56 = 1,680 | INFERRED |
| Timeout: 5 minutes per submission | CONFIRMED |
| Each submission gets a brand-new Tripletex account | CONFIRMED |
| Sandbox account is persistent (for development only) | CONFIRMED |

## Endpoint Contract

### Request

The agent exposes `POST /solve` over HTTPS.

```json
{
  "prompt": "Create an employee named Ola Nordmann with email ola@example.com",
  "files": [
    {
      "content_base64": "...",
      "filename": "transactions.csv"
    }
  ],
  "tripletex_credentials": {
    "base_url": "https://tx-proxy.ainm.no/v2",
    "session_token": "<token>"
  }
}
```

| Field | Type | Required | Status |
|-------|------|----------|--------|
| `prompt` | `str` | Yes | CONFIRMED |
| `files` | `list[{content_base64, filename}]` | No (optional) | CONFIRMED |
| `tripletex_credentials.base_url` | `str` | Yes | CONFIRMED |
| `tripletex_credentials.session_token` | `str` | Yes | CONFIRMED |

### Response

```json
{"status": "completed"}
```

Status: CONFIRMED

### Authentication to Tripletex API

- Basic Auth with username `"0"` and password = `session_token` from the request.
- Proxy base URL: `https://tx-proxy.ainm.no/v2`

Status: CONFIRMED

## Languages

All 7 languages the prompt may arrive in:

| Code | Language |
|------|----------|
| `nb` | Norwegian Bokmal |
| `en` | English |
| `es` | Spanish |
| `pt` | Portuguese |
| `nn` | Norwegian Nynorsk |
| `de` | German |
| `fr` | French |

Status: CONFIRMED

Implication: The agent must parse accounting instructions in any of these languages and map
them to the correct Tripletex API calls. Norwegian characters (ae, oe, aa) work as UTF-8.
(CONFIRMED)

## Task Categories

| Category | Example Tasks (CONFIRMED from docs) |
|----------|-------------------------------------|
| Employees | Create employee |
| Customers & Products | Create customer |
| Invoicing | Create invoice, register payment, credit notes |
| Travel Expenses | (no specific examples given) |
| Projects | Project billing |
| Corrections | Error correction in ledger |
| Departments | (no specific examples given) |

Status: Categories CONFIRMED. Specific tasks within each category are partially known.

## Scoring System

### Field-by-Field Verification

After the agent responds with `{"status": "completed"}`, the system inspects the Tripletex
account to verify that the requested changes were made correctly.

Example -- "Create employee" scoring breakdown:

| Check | Points |
|-------|--------|
| Employee found | 2 |
| Correct first name | 1 |
| Correct last name | 1 |
| Correct email | 1 |
| Administrator role assigned | 5 |
| **Total** | **10** |

Correctness = `points_earned / max_points`, normalized to 0.0--1.0.

Status: CONFIRMED

### Tier System

| Tier | Multiplier | Opens | Example Tasks |
|------|-----------|-------|---------------|
| Tier 1 | x1 | From start (March 19) | Create employee, create customer |
| Tier 2 | x2 | Early Friday (March 21) | Create invoice, register payment, credit notes, project billing |
| Tier 3 | x3 | Early Saturday (March 22) | Bank reconciliation from CSV, error correction in ledger, year-end closing |

Status: CONFIRMED

### Efficiency Bonus

- Only applies to submissions with PERFECT correctness (1.0). CONFIRMED.
- Can up to double the tier score. CONFIRMED.
  - Tier 1 perfect: 1.0 x 1 = 1.0, with max efficiency bonus: 2.0
  - Tier 2 perfect: 1.0 x 2 = 2.0, with max efficiency bonus: 4.0
  - Tier 3 perfect: 1.0 x 3 = 3.0, with max efficiency bonus: 6.0
- Two factors:
  - **Call efficiency**: Fewer API calls = higher bonus. CONFIRMED.
  - **Error cleanliness**: Fewer 4xx errors = higher bonus. CONFIRMED.

### Score Range

- Minimum: 0.0 (failed / nothing correct)
- Maximum per task: 6.0 (perfect Tier 3 + best efficiency)
- Maximum theoretical total: 30 x 6.0 = 180.0 (INFERRED -- assumes all 30 tasks are Tier 3, which is unlikely)
- Best score per task is kept (all-time best across all submissions). CONFIRMED.
- Leaderboard = sum of best scores across all 30 task types. CONFIRMED.

### Rate Limits

| Team Status | Concurrent | Per Task Per Day |
|-------------|-----------|-----------------|
| Verified | 3 | 4 |
| Unverified | 1 | 2 |

Status: CONFIRMED

## API Patterns

### Response Formats

- Lists: `resp.json()["values"]` CONFIRMED
- Single entities: `resp.json()["value"]` CONFIRMED
- Use `?fields=*` to see all available fields on any endpoint. CONFIRMED.

### Common Task Patterns

| Pattern | Description | Status |
|---------|-------------|--------|
| Create single entity | e.g., POST /employee | CONFIRMED |
| Create with linking | e.g., create product then attach to invoice | CONFIRMED |
| Modify existing | e.g., PUT to update fields | CONFIRMED |
| Delete/reverse | e.g., credit note to reverse invoice | CONFIRMED |
| Multi-step setup | e.g., enable module, create entities, link them | CONFIRMED |

### Common Errors

| Code | Meaning | Status |
|------|---------|--------|
| 401 | Wrong auth (check username="0", password=token) | CONFIRMED |
| 404 | Wrong path | CONFIRMED |
| 422 | Missing required fields | CONFIRMED |
| Empty values | Check search parameters | CONFIRMED |
| Timeout | 5 minute limit exceeded | CONFIRMED |

### Key Gotchas

- Sandbox starts empty -- each competition submission is a fresh account. CONFIRMED.
- Some tasks require enabling modules before they can be used. CONFIRMED.
- Norwegian characters work as UTF-8. CONFIRMED.

## Sandbox (Development)

| Property | Value | Status |
|----------|-------|--------|
| Base URL | `https://kkpqfuj-amager.tripletex.dev/v2` | CONFIRMED |
| Token expiry | March 31, 2026 | CONFIRMED |
| Persistent | Yes (unlike competition accounts) | CONFIRMED |
| Example endpoints | `/employee`, `/customer` | CONFIRMED |

## Efficiency Optimization Tips

From the official docs (all CONFIRMED):

1. Plan before calling -- understand what endpoints are needed before making requests
2. Avoid trial-and-error -- each failed call hurts error cleanliness score
3. Minimize GET calls -- only fetch what you need
4. Batch where possible -- reduce total call count

## Gaps and Unknowns

### High Priority (affects architecture)

1. **Exact list of all 30 tasks** -- We know categories and some examples, but not the
   complete list. We need to discover them through the submission system or docs.

2. **Exact scoring rubrics per task** -- Only the "create employee" rubric is shown as an
   example. Other tasks likely have different point distributions and checks.

3. **Efficiency bonus formula** -- We know it can "up to double" the score and depends on
   call count and error count, but the exact formula is unknown.

4. **Which modules need enabling** -- Some tasks require enabling Tripletex modules first.
   We do not know which modules or how to enable them via API.

5. **File-based tasks** -- Files are optional and base64-encoded. We know Tier 3 includes
   "bank reconciliation from CSV" which likely uses files, but we do not know the exact
   file formats or which tasks include files.

### Medium Priority (affects implementation)

6. **Tier distribution of the 30 tasks** -- How many tasks are Tier 1 vs Tier 2 vs Tier 3?
   This affects prioritization strategy.

7. **Exact variant distribution** -- Are all 8 data sets meaningfully different, or minor
   variations? This affects how robust our parsing needs to be.

8. **Proxy behavior** -- Does `tx-proxy.ainm.no` pass through all Tripletex v2 endpoints
   or only a subset? Are there additional rate limits at the proxy level?

9. **Task prompt structure** -- Are prompts consistently structured or free-form? Do they
   always contain all necessary information or do we need to infer some values?

### Low Priority (nice to know)

10. **Competition account pre-configuration** -- Does the fresh account have any default
    data (chart of accounts, default settings) or is it completely blank?

11. **Partial credit details** -- For multi-step tasks, does partial completion still earn
    points for completed sub-steps?

12. **Concurrent submission behavior** -- If we hit the concurrent limit, does the request
    queue or fail immediately?

## Strategic Implications

### Priority Order

1. **Tier 1 tasks first** (available from start, March 19). Get perfect scores to build
   baseline. Focus on create employee, create customer, and similar simple CRUD tasks.

2. **Efficiency optimization** for perfected Tier 1 tasks -- since efficiency bonus only
   applies to perfect scores, optimize call count after achieving correctness.

3. **Tier 2 preparation** (opens Friday March 21). Invoicing and payment workflows are
   more complex. Pre-build invoice/payment logic against sandbox.

4. **Tier 3 preparation** (opens Saturday March 22). Bank reconciliation, error correction,
   year-end closing are the highest-value tasks (up to 6.0 each). These likely require
   file parsing (CSV) and deep Tripletex domain knowledge.

### Architecture Implications

- Agent needs robust multi-language prompt parsing (7 languages).
- Agent needs a task classifier to identify which of the 30 tasks is being requested.
- Agent should maintain a mapping of task type to API call sequence.
- Agent should minimize API calls from the start (efficiency matters for perfect scores).
- Agent must handle file attachments (base64 decode, parse CSV/other formats).
- 5-minute timeout means the agent cannot do extensive exploration of the API at runtime.
  Pre-computed knowledge of endpoints and required fields is essential.
