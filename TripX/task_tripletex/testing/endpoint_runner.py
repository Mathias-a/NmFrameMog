from __future__ import annotations

import json
from collections.abc import Mapping
from time import perf_counter
from typing import Protocol, cast
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from task_tripletex.models import SolveFile, SolveRequest, TripletexCredentials
from task_tripletex.testing.models import (
    EndpointContractResult,
    EndpointRunResult,
)


class _UrlopenResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self) -> bytes: ...

    def close(self) -> None: ...


def _serialize_file(file: SolveFile) -> dict[str, str]:
    payload: dict[str, str] = {
        "filename": file.filename,
        "content_base64": file.content_base64,
    }
    if file.mime_type is not None:
        payload["mime_type"] = file.mime_type
    return payload


def _serialize_request(request: SolveRequest) -> dict[str, object]:
    return {
        "prompt": request.prompt,
        "files": [_serialize_file(file) for file in request.files],
        "tripletex_credentials": {
            "base_url": request.tripletex_credentials.base_url,
            "session_token": request.tripletex_credentials.session_token,
        },
    }


def _normalize_json_value(raw: object) -> object:
    if raw is None or isinstance(raw, (str, int, float, bool)):
        return raw
    if isinstance(raw, list):
        return [_normalize_json_value(item) for item in raw]
    if isinstance(raw, dict):
        normalized: dict[str, object] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings.")
            normalized[key] = _normalize_json_value(value)
        return normalized
    raise ValueError(f"Unsupported JSON value type: {type(raw).__name__}")


def _read_http_response(
    request: urllib_request.Request, timeout_seconds: float
) -> tuple[int, str | None, str]:
    try:
        response = cast(
            _UrlopenResponse, urllib_request.urlopen(request, timeout=timeout_seconds)
        )
        try:
            response_status_code = response.status
            response_headers = dict(response.headers.items())
            response_text = response.read().decode("utf-8", errors="replace")
        finally:
            response.close()
    except urllib_error.HTTPError as exc:
        response_status_code = exc.code
        response_headers = dict(exc.headers.items())
        response_text = exc.read().decode("utf-8", errors="replace")
        exc.close()
    lowercase_headers = {k.lower(): v for k, v in response_headers.items()}
    response_content_type = lowercase_headers.get("content-type")
    return response_status_code, response_content_type, response_text


async def run_solve_endpoint(
    solve_url: str,
    request: SolveRequest,
    *,
    proxy_base_url: str,
    api_key: str | None = None,
    timeout_seconds: float = 300.0,
) -> EndpointRunResult:
    rewritten_request = SolveRequest(
        prompt=request.prompt,
        files=request.files,
        tripletex_credentials=TripletexCredentials(
            base_url=proxy_base_url,
            session_token=request.tripletex_credentials.session_token,
        ),
    )
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    start = perf_counter()
    request_body = json.dumps(_serialize_request(rewritten_request)).encode("utf-8")
    http_request = urllib_request.Request(
        solve_url,
        data=request_body,
        headers=headers,
        method="POST",
    )
    response_status_code, response_content_type, response_text = _read_http_response(
        http_request, timeout_seconds
    )
    elapsed_seconds = perf_counter() - start

    response_json: object | None = None
    try:
        response_json = _normalize_json_value(cast(object, json.loads(response_text)))
    except json.JSONDecodeError:
        response_json = None

    parsed_url = urlparse(solve_url)
    url_is_https = parsed_url.scheme.lower() == "https"
    url_targets_solve = parsed_url.path == "/solve"
    exact_success_response = response_status_code == 200 and response_json == {
        "status": "completed"
    }

    errors: list[str] = []
    if not url_is_https:
        errors.append("Solve URL is not HTTPS.")
    if not url_targets_solve:
        errors.append("Solve URL does not target /solve.")
    if response_status_code != 200:
        errors.append(f"Solve endpoint returned HTTP {response_status_code}.")
    if response_content_type is None or "application/json" not in response_content_type:
        errors.append("Solve endpoint did not return application/json.")
    if response_json != {"status": "completed"}:
        errors.append(
            "Solve endpoint response body was not exactly {'status': 'completed'}."
        )

    return EndpointRunResult(
        original_request=request,
        rewritten_request=rewritten_request,
        sent_headers=headers,
        proxy_base_url=proxy_base_url,
        elapsed_seconds=elapsed_seconds,
        contract=EndpointContractResult(
            url_is_https=url_is_https,
            url_targets_solve=url_targets_solve,
            request_content_type=headers["Content-Type"],
            response_status_code=response_status_code,
            response_content_type=response_content_type,
            response_json=response_json,
            response_text=response_text,
            exact_success_response=exact_success_response,
            errors=errors,
        ),
    )
