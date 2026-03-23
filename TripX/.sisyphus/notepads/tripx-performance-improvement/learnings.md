# Learnings

## Multimodal Input Patterns (google-genai Python SDK) — 2026-03-21

### Import style used in this repo (already correct)
```python
from google import genai
from google.genai import types
```

### SDK key facts (from official docs)
- Package: `google-genai` (NOT `google-generativeai` legacy lib)
- Client: `genai.Client(api_key=...)` or `genai.Client()` for env-var
- Models: `client.models.generate_content(...)` or async `client.aio.models.generate_content(...)`
- Chat: `client.aio.chats.create(model=..., config=...)` + `chat.send_message(contents)`

### Three ways to send binary/image/PDF content

#### 1. PIL Image (simplest — works directly in `contents` list)
Doc: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md
```python
from PIL import Image
image = Image.open('invoice.pdf_page.png')  # or .jpg, .png, .webp
response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents=[image, 'Describe this image in detail.'],
)
```
**Limitation**: PDF needs pre-conversion to image (e.g., pdf2image, Pillow for single pages).

#### 2. `Part.from_bytes` (most flexible — works for image, PDF, audio, video)
Doc: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md
```python
with open('document.pdf', 'rb') as f:
    data = f.read()
response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents=[
        types.Part.from_bytes(data=data, mime_type='application/pdf'),
        'Extract all text and numbers from this document.',
    ],
)
```
**Compatible with chat mode too** — pass `Part.from_bytes` objects in the contents list:
```python
chat = genai_client.aio.chats.create(model="gemini-3.1-pro-preview", config=config)
response = await chat.send_message([
    types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
    "Extract invoice data from this PDF.",
])
```

#### 3. File API (for large files >20MB, video, long audio)
Doc: https://ai.google.dev/gemini-api/docs/file-input-methods
```python
my_file = client.files.upload(file='video.mp4')
response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents=[my_file, 'What happens in this video?']
)
client.files.delete(name=my_file.name)
```

### API shape for chat with multimodal content
```python
contents: list[Any] = [request.prompt]
if request.files:
    for f in request.files:
        contents.append(types.Part.from_bytes(data=f.content, mime_type=f.mime_type or 'application/octet-stream'))
response = await chat.send_message(contents)
```

### Caveats for Task 7
1. **PDFs are binary bytes, not images** — use `Part.from_bytes` with `mime_type='application/pdf'`. The official file-input doc confirms reading local PDFs as bytes is the simplest approach.
2. **PIL.Image works** but only for image formats. If the task sends PDF invoices, you need PDF→image conversion first OR just pass raw PDF bytes.
3. **The repo's current async chat pattern** (`genai_client.aio.chats.create` + `chat.send_message`) supports `Part.from_bytes` objects in the contents list — no File API upload needed for small PDFs/images.
4. **No base64 encoding needed** — pass raw bytes directly to `Part.from_bytes`.
5. **MIME types to use**: `image/png`, `image/jpeg`, `image/webp`, `image/gif`, `application/pdf`, `audio/mp3`, `video/mp4`.
6. **Size limits**: Inline bytes (via Part.from_bytes) works for files up to ~20MB. Larger files need the File API upload.
7. **Model compatibility**: `gemini-3-flash-preview`, `gemini-2.5-flash`, `gemini-3.1-pro-preview` all support multimodal inputs.
8. **system_instruction** in GenerateContentConfig is passed separately (already done in agent.py line 324) — no need to prepend files to system prompt.

### Key URLs
- Official multimodal docs: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md
- File input methods: https://ai.google.dev/gemini-api/docs/file-input-methods
- SDK GitHub: https://github.com/googleapis/python-genai
- SDK on PyPI: `google-genai`
- Packaged Tripletex case fixtures must keep non-empty `verification.reads` and `verification.checks`; placeholder breadth fixtures can stay parseable by using stable unique prompt/selector anchors and minimal valid checks.
- Tier 1 customer fixtures are safest when they verify writable/readable fields like `name`, `email`, and `phoneNumber`; avoid read-only `isCustomer` in selectors or field checks.
- Tier 1 product fixtures are safer with `name`, `number`, and `priceExcludingVatCurrency` than with invented names like `productNumber`, because the agent prompt documents `number` and the verifier only checks what the read path actually returns.
- Invoice/order verification fixtures should use Tripletex list-query names `orderDateFrom`/`orderDateTo` for `/order` and `invoiceDateFrom`/`invoiceDateTo` for `/invoice`; generic `fromDate`/`toDate` placeholders are not aligned with the prompt's documented API quirks.
- If invoice fixtures use nested selectors like `orders.0.orderNumber` or `orderLines.0.description`, the corresponding verification `fields` must include those nested structures; otherwise the verifier reports missing-field failures even when the runtime flow is correct.
- Task 5 convention: keep unsupported-MIME and oversized-file expectations in focused tests only for now, using explicit assertions/`pytest.raises` against a temporary inline policy helper rather than changing runtime validation before Task 7.
- Task 5 convention: the temporary supplier-invoice file-backed integration gate should be one narrow `pytest.mark.xfail(reason="multimodal runtime not yet implemented", strict=True)` that asserts files reach Gemini request contents once multimodal support lands.

- Request-context injection can stay deterministic and non-destructive by adding the ISO date and budget policy to Gemini `system_instruction` while leaving `contents` as just the original user prompt; this preserves explicit user-provided dates and the existing `chat.send_message(contents)` flow.

- Task 7 runtime now assembles Gemini multimodal contents inline in :  remains the raw prompt, supported attachments are appended via , and unsupported/oversized files raise  after explicit  logging so  returns deterministic 422s instead of silently dropping files.
- Task 7 inline attachment policy is intentionally narrow for Cloud Run safety: allow only , , , and , with a decoded-size cap of 20 MiB per file; MIME falls back to filename-based guessing only when  is absent.

- Task 7 runtime now assembles Gemini multimodal contents inline in `task_tripletex/agent.py`: `contents[0]` remains the raw prompt, supported attachments are appended via `types.Part.from_bytes(data=decoded_bytes, mime_type=resolved_mime_type)`, and unsupported or oversized files raise `ValueError` after explicit `file_attachment_rejected` logging so `/solve` returns deterministic 422s instead of silently dropping files.
- Task 7 inline attachment policy is intentionally narrow for Cloud Run safety: allow only `application/pdf`, `image/jpeg`, `image/png`, and `image/webp`, with a decoded-size cap of 20 MiB per file; MIME falls back to filename-based guessing only when `mime_type` is absent.

- Task 8 prompt expansion stayed low-risk by extending task_tripletex/agent.py in the existing endpoint-reference, task-pattern, and efficiency-tip style: add concise sections for supplier invoices, purchase orders, timesheets/payroll, voucher import, bank reconciliation, attachments, and general batch/import routing, then protect them with direct SYSTEM_PROMPT assertions plus the existing employee and invoice regressions and the full tests/task_tripletex_testing suite.

- Task 9 keeps Tripletex response shaping at the task_tripletex/agent.py tool-response boundary instead of the HTTP client: successful value writes stay intact for downstream IDs/refs, 4xx responses preserve structured fields such as validationMessages, and large list GETs are deterministically capped to MAX_SHAPED_LIST_VALUES while retaining count/from/fullResultSize plus explicit truncation metadata.

- Task 10 keeps bounded 4xx recovery entirely inside `task_tripletex/agent.py`: only a structured `422` with non-empty `validationMessages` earns one repair turn, the repair hint is appended as a compact follow-up instruction after the function responses, and any later 4xx on the same `METHOD path` objective or any non-recoverable 4xx (`401/403/404` and other non-422 client errors) is logged with `client_error_decision` and stops the loop immediately.

- Task 11 keeps request-local GET deduplication inside  rather than : cache keys are exact  strings with sorted query serialization, cache scope lasts only for one  invocation, and any /// conservatively invalidates cached GETs for the same top-level resource family while emitting explicit , , and  trace events.

- Task 11 keeps request-local GET deduplication inside task_tripletex/agent.py rather than task_tripletex/client.py: cache keys are exact METHOD|path|serialized_query strings with sorted query serialization, cache scope lasts only for one run_agent invocation, and any POST/PUT/PATCH/DELETE conservatively invalidates cached GETs for the same top-level resource family while emitting explicit request_cache_miss, request_cache_hit, and request_cache_invalidate trace events.

- Task 12 keeps the existing Gemini model but makes runtime behavior more reproducible by centralizing `GenerateContentConfig` in `task_tripletex/agent.py` with explicit `temperature=0.0`, `top_p=1.0`, `candidate_count=1`, and `seed=0`; the config is now traceable via a `model_runtime_configured` log event alongside the existing request-budget markers.

- Task 13 follow-up: emit `request_context_decision` in `run_agent()` before `_build_initial_contents(request)` so early attachment-policy failures still leave the injected-date/request-context breadcrumb in `/logs`; protect it with a real `/solve` invalid-file regression that asserts the event order starts with request_context_initialized -> request_context_decision -> file_attachment_rejected.

- Task 14 VM proxy wave (2026-03-21): the localhost VM evaluator path still reproduces the optimal employee baseline (correctness 1.0, 2 proxy calls, 0 client errors) and `create_product` also ran green in one write call; the evaluator continues to flag localhost runs as non-HTTPS with null content-type, so those contract warnings are expected artifacts of the documented proxy workflow rather than solve regressions.

- Task 15 Cloud Run smoke gate (2026-03-21): the documented `gcloud builds submit ... && gcloud run deploy ...` flow successfully rolled Cloud Run service `tripletex` to revision `tripletex-00011-sh9`; direct live `/solve` returned exact JSON `{"status":"completed"}` for the text-only employee-admin prompt, and live `/logs` for that run showed the expected runtime breadcrumbs including `request_context_decision` and `model_runtime_configured`.

- Task 15 file-backed Cloud Run smoke refresh (2026-03-21): live  accepted a minimal one-page  attachment and returned exact  with  showing ; a tiny synthetic PNG remained unreliable because Gemini returned  (), so reviewer evidence should prefer a real paged PDF over microscopic image placeholders.

- Task 15 file-backed Cloud Run smoke refresh (2026-03-21): live POST /solve accepted a minimal one-page application/pdf attachment and returned exact {"status":"completed"} with /logs showing file_attachment_prepared; a tiny synthetic PNG remained unreliable because Gemini returned 400 INVALID_ARGUMENT (Unable to process input image), so reviewer evidence should prefer a real paged PDF over microscopic image placeholders.
