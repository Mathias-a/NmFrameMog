# Agents & Infrastructure

We have **free and limitless access to use Google Cloud (gcloud)** in order to run, create, and evolve resources for this project. Agents are encouraged to freely provision Cloud Run services, Vertex AI endpoints, storage, and parallel compute nodes as needed to maximize performance and efficiency.

## Task Overview

The task is to build an AI accounting agent for Tripletex. The agent exposes a `/solve` endpoint that receives natural language prompts (in 7 languages) and optional file attachments. It must interpret these prompts and execute the correct sequence of Tripletex v2 REST API calls (like creating employees, invoices, or projects) within a 5-minute timeout. The solution is scored on correctness (field-by-field validation) and efficiency (fewest API calls and errors). Complete overview in docs.

## Current Architecture

- **FastAPI service** (`task_tripletex/service.py`) with `/solve` (POST) and `/logs` (GET) endpoints
- **Gemini 3.1 Pro Preview** agent (`task_tripletex/agent.py`) with ThinkingConfig HIGH, tool-calling loop (max 30 steps)
- **Comprehensive system prompt** with full API schema reference (~300 lines) — verified against live OpenAPI spec v2.74.00
- **In-memory task logger** (`task_tripletex/task_log.py`) — thread-safe singleton, traces every LLM call and API call
- **Tripletex HTTP client** (`task_tripletex/client.py`) — httpx-based, basic auth with session token
- **Testing framework** (`task_tripletex/testing/`) — 9 modules, proxy-based E2E testing with scoring

## Current Performance (verified 2026-03-21)

- **create_employee_admin test**: 100% correctness (5/5 points), 2 API calls, 0 errors, 18.5s
- **API call pattern**: GET /department → POST /employee (optimal for this task type)
- **Cloud Run revision**: `tripletex-00010-fwg`

## Deployment

- **Cloud Run**: `https://tripletex-124894558027.europe-north1.run.app`
- **Endpoints**: `POST /solve`, `GET /logs`
- **Project**: `ainm26osl-707`, region `europe-north1`
- **Deploy**: `gcloud builds submit --tag gcr.io/ainm26osl-707/tripletex --project=ainm26osl-707 && gcloud run deploy tripletex --image gcr.io/ainm26osl-707/tripletex --platform managed --region europe-north1 --project=ainm26osl-707 --allow-unauthenticated`

## Credentials

- **Gemini API key**: `AIzaSyDXWbMGEpkYrJUpf-qArxglOZrN56GTnr8` (hardcoded in agent.py)
- **Tripletex sandbox**: base URL `https://kkpqfuj-amager.tripletex.dev/v2`, session token in test commands below
- **Platform JWT**: in `/Users/mathias/ai-fun/NmFrameMog/worktree-3/.env`

## Testing

### E2E test on gcloud VM
```bash
# First sync code to VM:
gcloud compute scp --recurse /Users/mathias/ai-fun/NmFrameMog/worktree-3/TripX/task_tripletex devstar7073@yolov8-trainer:/home/devstar7073/NmFrameMog/TripX/ --project=ainm26osl-707 --zone=us-west4-a

# SSH to VM:
gcloud compute ssh devstar7073@yolov8-trainer --project=ainm26osl-707 --zone=us-west4-a

# On VM — kill old server, start fresh:
pkill -f uvicorn 2>/dev/null
cd /home/devstar7073/NmFrameMog/TripX
nohup .venv/bin/uvicorn task_tripletex.service:app --host 0.0.0.0 --port 8080 > /tmp/uvicorn.log 2>&1 &

# Run E2E test:
PYTHONPATH=/home/devstar7073/NmFrameMog/TripX .venv/bin/python -m task_tripletex.testing.cli \
  --packaged-case create_employee_admin \
  --solve-url http://127.0.0.1:8080/solve \
  --tripletex-base-url https://kkpqfuj-amager.tripletex.dev/v2 \
  --session-token "eyJ0b2tlbklkIjoyMTQ3NjU0MTEzLCJ0b2tlbiI6ImFhY2U4ODNmLWM0NTUtNGQxYy05MTkxLWJlOWUxZjQ0M2IwZSJ9" \
  --output json
```

### Quick test against Cloud Run (no proxy — tests real API directly)
```bash
curl -s -X POST https://tripletex-124894558027.europe-north1.run.app/solve \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.","files":[],"tripletex_credentials":{"base_url":"https://kkpqfuj-amager.tripletex.dev/v2","session_token":"eyJ0b2tlbklkIjoyMTQ3NjU0MTEzLCJ0b2tlbiI6ImFhY2U4ODNmLWM0NTUtNGQxYy05MTkxLWJlOWUxZjQ0M2IwZSJ9"}}'
```

### Check logs after a run
```bash
curl -s https://tripletex-124894558027.europe-north1.run.app/logs | python3 -m json.tool
```

### Submit for scoring
Go to `https://app.ainm.no/submit/tripletex` in browser. No public API for results.

## Key Findings & Known Issues

1. **Efficiency SOLVED for basic tasks**: Agent now makes optimal 2 API calls for create_employee_admin (was 19+). System prompt with full API schema eliminated trial-and-error.
2. **Gemini 3.x ThinkingConfig**: Use `types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH, include_thoughts=True)` — NOT `thinking_budget`.
3. **Tripletex API quirks**:
   - `EmployeeDTO` has no `isAdministrator` field; use `userType: 2` for admin, or entitlements endpoint
   - Employee creation: `dateOfBirth` required for Norwegian employees; `email` required for Tripletex users; `department` required if dept functionality activated
   - Voucher postings use `amountGross`/`amountGrossCurrency`, NOT `amount`
   - Order `isPrioritizeAmountsIncludingVat` must match price fields on order lines (VAT mismatch = 422)
   - GET on `/order`, `/invoice`, `/ledger/voucher` REQUIRE date range params
   - Customer `isCustomer` is readOnly — don't send it
   - Customer ledger accounts require `customer` ref in voucher postings (vendor→`supplier`, employee→`employee`)
4. **Test proxy limitation**: The E2E test proxy runs on localhost — Cloud Run can't reach it. Test locally on VM with `--solve-url http://127.0.0.1:8080/solve`, or test Cloud Run against real API and check `/logs`.
5. **SSH to VM is flaky**: Short commands (`echo OK`) work but longer commands often timeout. Break into small steps.
6. **Content-Type**: Cloud Run returns `application/json` correctly. VM local test may show null if running stale code.
7. **File attachments**: Not yet implemented (skipped in agent.py).

## Priority Next Steps

1. **Add file attachment support** — Upload files via Gemini File API before sending to chat (needed for PDF/image tasks)
2. **Add more test fixtures** — Currently only `create_employee_admin`; need invoice, customer, project, voucher fixtures
3. **Submit for official scoring** — Go to `https://app.ainm.no/submit/tripletex` and submit the Cloud Run URL
4. **Handle Tier 2/3 tasks** — Multi-step workflows (invoice+payment, credit notes, project billing) — the system prompt already has patterns for these
5. **Optimize further** — Inline orderLines in order creation, avoid unnecessary GETs
