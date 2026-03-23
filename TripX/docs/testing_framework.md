# Tripletex testing framework

This repository now includes a local evaluator under `task_tripletex/testing/` for exercising a live TripX `/solve` endpoint against the public Tripletex contract and the documented scoring rules.

## What it does

- calls a live `/solve` URL with `application/json`
- rewrites `tripletex_credentials.base_url` to a local recording proxy
- forwards proxy traffic to the original upstream `base_url`
- verifies Basic Auth usage (`0:<session_token>`)
- records write calls and 4xx responses for deterministic efficiency scoring
- runs field-by-field correctness checks from JSON case fixtures
- computes documented correctness and tier score exactly, plus a documented local efficiency bonus policy
- **disqualifies** (zeroes the score) when any contract or proxy violation is detected

## Modules

- `task_tripletex/testing/models.py` — evaluator dataclasses
- `task_tripletex/testing/endpoint_runner.py` — `/solve` caller and contract checks
- `task_tripletex/testing/reverse_proxy_recorder.py` — local recording proxy that forwards upstream
- `task_tripletex/testing/tripletex_read_helper.py` — authenticated read helper for verification
- `task_tripletex/testing/verifier.py` — declarative field-by-field verification
- `task_tripletex/testing/scoring.py` — correctness, tier, efficiency scoring, and disqualification logic
- `task_tripletex/testing/fixture_loader.py` — case fixture loading
- `task_tripletex/testing/cli.py` — runnable CLI entrypoint

## Usage

Run a packaged starter case:

```bash
python -m task_tripletex.testing.cli \
  --packaged-case create_employee_admin \
  --solve-url https://your-endpoint.example/solve \
  --tripletex-base-url https://provided-tripletex-proxy.example/v2 \
  --session-token your-session-token \
  --output text
```

Or load a custom fixture file:

```bash
python -m task_tripletex.testing.cli \
  --case-file path/to/case.json \
  --solve-url https://your-endpoint.example/solve \
  --tripletex-base-url https://provided-tripletex-proxy.example/v2 \
  --session-token your-session-token \
  --output json
```

## Fixture shape

Each case fixture declares:

- `prompt` and optional `files`
- `tier`
- `expected_min_proxy_calls`
- `verification.reads` — GET definitions with `path`, `query`, and `mode`
- `verification.checks` — point-bearing checks using `entity_exists` or `field_equals`
- `efficiency` — local benchmark policy for write calls and 4xx errors

The packaged starter fixture lives at `task_tripletex/testing/fixtures/create_employee_admin.json` and mirrors the scoring example from `docs/scoring.md`.

## Score-blocking violations

Contract and proxy violations are **score-blocking**: any violation zeroes the entire score regardless of correctness. Specifically:

**Contract violations** (from endpoint_runner):
- Solve URL is not HTTPS (skipped when `enforce_https=False` for local testing)
- Solve URL path is not exactly `/solve`
- Response is not HTTP 200 with `{"status": "completed"}`

**Proxy violations** (from proxy_metrics):
- Solution did not route traffic through the recording proxy
- Solution did not use the rewritten proxy base URL
- Any call used incorrect Basic Auth
- Any call was not forwarded to the upstream base URL

When disqualified, the `ScoreResult` contains `contract_valid=False` and/or `proxy_valid=False` with human-readable `disqualification_reasons`.

## Deterministic efficiency policy

The public docs describe the bonus qualitatively but do not publish the production formula. This framework therefore uses a documented fixture-driven policy instead of inventing hidden global constants.

For a perfect submission only:

- `base_score = correctness × tier`
- `write_efficiency` is normalized between `best_write_calls` and `max_write_calls`
- `error_efficiency` is normalized between `0` and `max_4xx_errors`
- `combined_efficiency` is the weighted average from `write_weight` and `error_weight`
- `efficiency_bonus = tier × combined_efficiency`
- `total_score = base_score + efficiency_bonus`

This preserves the documented ceiling of doubling the tier score while keeping the benchmark inputs explicit in each fixture.

## Proxy upstream timeout

The recording proxy (`ReverseProxyRecorder`) uses a default upstream timeout of **30 seconds** per forwarded request. The CLI caps the proxy timeout at `min(timeout_seconds, 30.0)`. If the upstream Tripletex API takes longer than 30s to respond to a single call, the proxy will error. Adjust `timeout_seconds` in the `ReverseProxyRecorder` constructor if needed.

## Verifier selector semantics

The verifier uses **first-match** semantics: when a check's `selector` matches multiple entities in the read snapshot, only the first match is used for `field_equals` evaluation. Ensure selectors are unique enough to target exactly one entity (e.g., include a unique field like `employeeId` or `name`).

## Notes

- Verification reads use the original upstream `base_url`, not the recording proxy, so post-run checks do not contaminate recorded efficiency metrics.
- The evaluator flags non-HTTPS solve URLs as contract violations. Local offline tests in this repo use `enforce_https=False` for convenience.
- The proxy accepts calls that keep `/v2` in the path and calls that rely on client base-URL joining, then forwards both forms to the original upstream base URL.
- The CLI output includes a full **recorded proxy call log** showing each method, path, status code, and tags ([write], [4xx], [bad-auth]) for debugging.
