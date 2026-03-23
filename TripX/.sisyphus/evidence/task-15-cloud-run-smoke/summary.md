# Task 15 Cloud Run deploy and smoke summary

- Deploy flow used: `gcloud builds submit --tag gcr.io/ainm26osl-707/tripletex --project=ainm26osl-707 && gcloud run deploy tripletex --image gcr.io/ainm26osl-707/tripletex --platform managed --region europe-north1 --project=ainm26osl-707 --allow-unauthenticated`
- Cloud Build result: `SUCCESS` for build `78aaeda5-b1c2-4462-954d-45efe0f96bd9` (`build_description.json`)
- Live Cloud Run revision: `tripletex-00011-sh9` serving 100% traffic (`service_description.json`)
- Live service URL: `https://tripletex-124894558027.europe-north1.run.app`

## Smoke outcomes

- Text-only `/solve` smoke: HTTP 200, `Content-Type: application/json`, exact body `{"status":"completed"}` (`text_only_solve_response.json`)
- File-backed `/solve` smoke: successful against live service using a genuinely valid one-page PDF attachment (`smoke_attachment.pdf`); HTTP 200, `Content-Type: application/json`, exact body `{"status":"completed"}` (`file_backed_solve_response.json`)

## `/logs` trace markers observed

- Successful text-only trace includes `request_context_initialized`, `request_context_decision`, `model_context_injected`, `model_runtime_configured`, `request_cache_miss`, `response_shaping_summary`, and `task_completed` (`logs_snapshot_after_text_only.json`)
- Successful file-backed trace includes `request_context_initialized`, `request_context_decision`, `file_attachment_prepared`, `model_context_injected`, `model_runtime_configured`, `llm_call`, `llm_response`, and `task_completed` (`logs_snapshot_after_file_backed_success.json`)
- Historical rejection trace from the earlier unsupported CSV probe is preserved and includes `request_context_initialized`, `request_context_decision`, and `file_attachment_rejected` (`logs_snapshot.json`)

## Notes

- Raw shell `tee` capture for the combined deploy command produced an empty `deploy.log`; authoritative deploy evidence is therefore captured via `gcloud builds describe` and `gcloud run services describe` JSON snapshots.
- A tiny synthetic PNG still triggered upstream Gemini `400 INVALID_ARGUMENT` (`Unable to process input image`), so the reviewer-ready file-backed smoke now uses a minimal valid paged PDF instead of an image placeholder.
