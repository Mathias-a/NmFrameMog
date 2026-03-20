from __future__ import annotations

import json
from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Awaitable, Callable
from typing import cast

from fastapi import FastAPI, HTTPException, Request

from task_tripletex.client import TripletexClient
from task_tripletex.models import (
    ExecutionPlan,
    SolveFile,
    SolveRequest,
    SolveResponse,
    TripletexCredentials,
)
from task_tripletex.planning import extract_execution_plan

PlanExecutor = Callable[[TripletexCredentials, ExecutionPlan], Awaitable[None]]


async def _default_plan_executor(
    credentials: TripletexCredentials, plan: ExecutionPlan
) -> None:
    async with TripletexClient(credentials) as client:
        await client.execute_plan(plan)


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


def create_app(executor: PlanExecutor | None = None) -> FastAPI:
    app = FastAPI(title="Tripletex task service")
    plan_executor = executor or _default_plan_executor

    async def solve(request: Request) -> dict[str, str]:
        try:
            request_body = await request.body()
            payload = cast(object, json.loads(request_body.decode("utf-8")))
            solve_request = _parse_solve_request(payload)
            plan = extract_execution_plan(solve_request)
            await plan_executor(solve_request.tripletex_credentials, plan)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {"status": SolveResponse(status="completed").status}

    app.add_api_route("/solve", solve, methods=["POST"])

    return app


app = create_app()
