# Tripletex — AI Accounting Agent

Sources:

- `challenge://tripletex/overview`
- `challenge://tripletex/endpoint`
- `challenge://tripletex/scoring`
- `challenge://tripletex/examples`
- `challenge://tripletex/sandbox`

## Live MCP summary

### Overview

- "Build an AI agent that completes accounting tasks in Tripletex."
- "You receive a task prompt (in one of 7 languages), use the Tripletex API to execute it, and get scored on correctness and efficiency."
- How it works excerpt:
  1. Submit your HTTPS endpoint URL on the platform.
  2. A fresh Tripletex sandbox account is provisioned.
  3. A randomly selected accounting task is sent to your `/solve` endpoint.
  4. The agent reads the prompt and may process attached files such as PDFs and images.

### Endpoint

- The endpoint page is titled **Tripletex — Endpoint Specification**.
- "Your agent must expose a single HTTPS endpoint that accepts POST requests."
- Credential excerpt:

```json
"tripletex_credentials": {
  "base_url": "https://tx-proxy.ainm.no/v2",
  "session_token": "abc123..."
}
```

- Field descriptions returned by the MCP:
  - `tripletex_credentials.base_url` — proxy API URL to use instead of the standard Tripletex URL.
  - `tripletex_credentials.session_token` — session token for authentication.
- Auth note from response format section:
  - Username is `0`.
  - Password is the `session_token` from the request.
- The live endpoint page now also makes these operational requirements explicit:
  - endpoint must be **HTTPS**
  - must return `{"status": "completed"}` with HTTP 200
  - must respond within **300 seconds / 5 minutes**
  - all Tripletex API calls must go through the provided proxy `base_url`
- Optional protection mechanism now visible in the live docs:
  - if the team configured an API key at submission time, validators send `Authorization: Bearer <your-api-key>` to the hosted endpoint
- The live endpoint page also exposes a concrete Tripletex API reference table, including:
  - `/employee`
  - `/customer`
  - `/product`
  - `/invoice`
  - `/order`
  - `/travelExpense`
  - `/project`
  - `/department`
  - `/ledger/account`
  - `/ledger/posting`
  - `/ledger/voucher`

### Scoring

- The scoring page includes **Field-by-Field Verification (Correctness)**.
- "After your agent responds, we query the Tripletex API to verify what was created or modified. Each task has specific checks worth different point values."
- The docs mention example scoring for a "Create employee" task and also expose **Rate Limits**.
- The live scoring resource now exposes the full structure:
  - correctness is normalized to `0–1`
  - each task has a **tier multiplier** (`×1`, `×2`, `×3`)
  - perfect runs can receive an **efficiency bonus** based on API-call efficiency and 4xx-error cleanliness
  - best scores are preserved per task type
  - leaderboard score is the sum of all-time best scores across task types
- Explicit live rate limits:
  - `10` concurrent submissions
  - `Unlimited` submissions per day
- The page also states that efficiency benchmarks are periodically recalculated and best scores are recomputed every 12 hours.

### Examples

- The examples page includes a **Minimal `/solve` Endpoint**.
- Excerpts returned:

```python
files = body.get("files", [])
creds = body["tripletex_credentials"]
base_url = creds["base_url"]
token = creds["session_token"]
auth = ("0", token)
```

- The page also notes a TODO placeholder: use an LLM to interpret the prompt and execute the appropriate Tripletex API calls.
- A **Common Errors** table is available with `Error`, `Cause`, and `Fix` columns.
- The live examples page now exposes much more than the original excerpt:
  - full FastAPI `/solve` starter example
  - local launch command with `uvicorn`
  - HTTPS testing via `cloudflared`
  - concrete API examples for listing employees, creating customers, creating invoices, and searching for entities
  - a table of common task patterns such as create-with-linking, modify-existing, delete/reverse, and multi-step setup
- The Common Errors table is now fully visible in the live resource, including:
  - `401 Unauthorized`
  - `404 Not Found`
  - `422 Validation Error`
  - empty `values` arrays
  - 5-minute timeout failures

### Sandbox

- "Every team gets a free Tripletex sandbox account to explore the API and web interface before submitting to the competition."
- Sandbox acquisition flow:
  1. Go to the Tripletex submission page.
  2. Click **Get Sandbox Account**.
  3. The sandbox is provisioned instantly.
- Example sandbox constants and API usage are exposed, including `/employee` and `/customer` examples.
- The live sandbox page now also exposes:
  - a concrete Tripletex UI URL example
  - Visma Connect first-time login flow via **Forgot password**
  - the fact that the sandbox is a full Tripletex test environment
  - example `curl` usage with Basic Auth

## What matters for implementation

- This is the most classically agentic task: interpret ambiguous prompts, decide API actions, execute safely, and verify side effects.
- Reliability beats raw model creativity; the winning system likely needs tool selection, schema validation, idempotent retries, and post-action verification.
- Because the evaluator checks resulting state through the API and rewards efficient perfect runs, the architecture should plan around explicit transaction auditing, low-error execution, and minimal redundant API calls.
