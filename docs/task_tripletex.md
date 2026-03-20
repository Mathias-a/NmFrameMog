# Tripletex

This package exposes a FastAPI service with the required delivery surface:

```bash
PYTHONPATH=src uv run uvicorn task_tripletex.service:app --reload
```

```http
POST /solve
```

## Request contract

The service accepts:

- `prompt: string`
- optional `files: [{filename, content_base64, mime_type?}]`
- `tripletex_credentials: {base_url, session_token}`

Tripletex API calls use Basic authentication with username `0` and password `session_token`, sent to the provided `base_url`.

## First-pass execution model

The upstream natural-language task matrix is under-specified, so this implementation does **not** guess undocumented Tripletex semantics. Instead, it executes an explicit JSON operation plan supplied either:

- as a JSON file in `files[]` (base64-encoded in `content_base64`), or
- as a JSON object / fenced `json` block in `prompt`

Supported plan shape:

```json
{
  "operations": [
    {
      "method": "GET",
      "path": "/company"
    },
    {
      "method": "POST",
      "path": "/some/path",
      "query": {"foo": "bar"},
      "body": {"example": true}
    }
  ]
}
```

On successful execution the service returns exactly:

```json
{"status":"completed"}
```

## Safety boundaries

- Only explicit structured operations are executed.
- The client does not invent undocumented request schemas.
- Failing Tripletex responses stop execution unless an operation explicitly sets `allow_failure: true`.
