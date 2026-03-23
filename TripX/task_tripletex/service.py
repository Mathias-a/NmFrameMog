from __future__ import annotations

import json
from base64 import b64decode
from binascii import Error as BinasciiError
from typing import cast

from fastapi import FastAPI, HTTPException, Request  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]

from task_tripletex.agent import build_request_context, run_agent
from task_tripletex.client import TripletexClient
from task_tripletex.models import (
    RequestContext,
    SolveFile,
    SolveExecutionOutcome,
    SolveRequest,
    SolveResponse,
    TripletexCredentials,
)
from task_tripletex.task_log import task_logger


async def _default_plan_executor(
    request: SolveRequest, request_context: RequestContext
) -> SolveExecutionOutcome:
    async with TripletexClient(request.tripletex_credentials) as client:
        return await run_agent(request, client, request_context=request_context)


def _coerce_execution_outcome(result: object) -> SolveExecutionOutcome:
    if isinstance(result, SolveExecutionOutcome):
        return result
    return SolveExecutionOutcome(status="completed", reason="legacy_executor_return")


def _require_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string.")
    return value


def _parse_solve_request(payload: object) -> SolveRequest:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    prompt = _require_string(payload, "prompt")
    credentials_raw = payload.get("tripletex_credentials")
    if not isinstance(credentials_raw, dict):
        raise ValueError("'tripletex_credentials' must be a JSON object.")

    credentials = TripletexCredentials(
        base_url=_require_string(credentials_raw, "base_url"),
        session_token=_require_string(credentials_raw, "session_token"),
    )

    files_raw_object: object = payload.get("files")
    if files_raw_object is None:
        files_list: list[object] = []
    elif isinstance(files_raw_object, list):
        files_list = cast(list[object], files_raw_object)
    else:
        raise ValueError("'files' must be a list when provided.")

    files: list[SolveFile] = []
    for index, file_raw in enumerate(files_list, start=1):
        if not isinstance(file_raw, dict):
            raise ValueError(f"File {index} must be a JSON object.")
        mime_type_raw = file_raw.get("mime_type")
        if mime_type_raw is not None and not isinstance(mime_type_raw, str):
            raise ValueError(f"File {index} mime_type must be a string when provided.")
        content_base64 = _require_string(file_raw, "content_base64")
        try:
            b64decode(content_base64, validate=True)
        except BinasciiError as exc:
            raise ValueError(
                f"File {index} content_base64 must contain valid base64 data."
            ) from exc
        files.append(
            SolveFile(
                filename=_require_string(file_raw, "filename"),
                content_base64=content_base64,
                mime_type=mime_type_raw,
            )
        )

    return SolveRequest(
        prompt=prompt,
        files=files,
        tripletex_credentials=credentials,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Tripletex task service")

    async def solve(request: Request) -> JSONResponse:
        try:
            request_body = await request.body()
            payload = cast(object, json.loads(request_body.decode("utf-8")))
            solve_request = _parse_solve_request(payload)
            request_context = build_request_context()
            request_context_detail = request_context.as_log_detail()
            request_id = task_logger.start_task(
                solve_request.prompt,
                request_context=request_context_detail,
            )
            request_context_detail["request_id"] = request_id
            task_logger.log("request_context_initialized", request_context_detail)
            try:
                outcome = _coerce_execution_outcome(
                    await _default_plan_executor(solve_request, request_context)
                )
                task_logger.log(
                    "solve_contract_response",
                    {
                        "request_id": request_id,
                        "external_status": SolveResponse(status="completed").status,
                        "internal_status": outcome.status,
                        "internal_reason": outcome.reason,
                    },
                )
                task_logger.finish_task(
                    request_id=request_id,
                    status=outcome.status,
                    final_reason=outcome.reason,
                )
            except Exception as exc:
                task_logger.finish_task(
                    request_id=request_id,
                    error=str(exc),
                    final_reason="exception",
                )
                raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail="Internal server error"
            ) from exc

        return JSONResponse(
            content={"status": SolveResponse(status="completed").status},
            media_type="application/json",
        )

    app.add_api_route("/", solve, methods=["POST"])
    app.add_api_route("/solve", solve, methods=["POST"])

    async def logs(request: Request) -> dict[str, object]:
        request_id = request.query_params.get("request_id")
        snapshot = task_logger.snapshot(request_id=request_id)
        if snapshot is None:
            return {"status": "no_tasks_yet", "trace": None}
        return {"status": "ok", "trace": snapshot}

    app.add_api_route("/logs", logs, methods=["GET"])

    return app


app = create_app()
