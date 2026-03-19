# Tripletex — AI Accounting Agent

Sources:

- `challenge://tripletex/overview`
- `challenge://tripletex/endpoint`
- `challenge://tripletex/scoring`
- `challenge://tripletex/examples`
- `challenge://tripletex/sandbox`

## MCP excerpts

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

### Scoring

- The scoring page includes **Field-by-Field Verification (Correctness)**.
- "After your agent responds, we query the Tripletex API to verify what was created or modified. Each task has specific checks worth different point values."
- The docs mention example scoring for a "Create employee" task and also expose **Rate Limits**.

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

### Sandbox

- "Every team gets a free Tripletex sandbox account to explore the API and web interface before submitting to the competition."
- Sandbox acquisition flow:
  1. Go to the Tripletex submission page.
  2. Click **Get Sandbox Account**.
  3. The sandbox is provisioned instantly.
- Example sandbox constants and API usage are exposed, including `/employee` and `/customer` examples.

## What matters for implementation

- This is the most classically agentic task: interpret ambiguous prompts, decide API actions, execute safely, and verify side effects.
- Reliability beats raw model creativity; the winning system likely needs tool selection, schema validation, idempotent retries, and post-action verification.
- Because the evaluator checks resulting state through the API, the architecture should plan around explicit transaction auditing.
