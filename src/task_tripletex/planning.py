from __future__ import annotations

import json
import re
from typing import Literal, cast

from task_tripletex.models import (
    ExecutionPlan,
    SolveRequest,
    StructuredOperation,
)

JSON_BLOCK_PATTERN = re.compile(
    r"```json\s*(?P<document>[\s\S]+?)\s*```", re.IGNORECASE
)
HTTP_METHODS = frozenset({"DELETE", "GET", "PATCH", "POST", "PUT"})


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


def _load_json_document(text: str) -> object:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON document: {exc.msg}") from exc
    return _normalize_json_value(raw)


def _extract_json_document(text: str) -> object | None:
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return _load_json_document(stripped)

    match = JSON_BLOCK_PATTERN.search(text)
    if match is None:
        return None

    return _load_json_document(match.group("document"))


def _require_object(value: object, *, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return value


def _require_scalar_query_value(raw: object) -> str | int | float | bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (str, int, float)):
        return raw
    raise ValueError(
        "Query values must be strings, numbers, booleans, or lists of those values."
    )


def _parse_query(
    raw: object | None,
) -> dict[str, str | int | float | bool | list[str | int | float | bool]]:
    if raw is None:
        return {}

    query_object = _require_object(raw, description="The operation query")
    query: dict[str, str | int | float | bool | list[str | int | float | bool]] = {}
    for key, value in query_object.items():
        if isinstance(value, list):
            query[key] = [_require_scalar_query_value(item) for item in value]
        else:
            query[key] = _require_scalar_query_value(value)
    return query


def _parse_method(
    raw: object | None,
) -> Literal["DELETE", "GET", "PATCH", "POST", "PUT"]:
    if not isinstance(raw, str):
        raise ValueError("Each operation must include a string method.")

    method = raw.upper()
    if method not in HTTP_METHODS:
        raise ValueError(f"Unsupported HTTP method: {raw}")
    return cast(Literal["DELETE", "GET", "PATCH", "POST", "PUT"], method)


def _parse_path(raw: object | None) -> str:
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise ValueError("Each operation path must be a string starting with '/'.")
    return raw


def _parse_allow_failure(raw: object | None) -> bool:
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise ValueError("allow_failure must be a boolean when provided.")
    return raw


def _parse_operations(raw: object) -> list[StructuredOperation]:
    document = _require_object(raw, description="The plan document")
    operations_raw = document.get("operations")
    if not isinstance(operations_raw, list) or not operations_raw:
        raise ValueError(
            "The plan document must contain a non-empty 'operations' list."
        )
    operations_list = cast(list[object], operations_raw)

    operations: list[StructuredOperation] = []
    for index, operation_raw in enumerate(operations_list, start=1):
        operation_object = _require_object(
            operation_raw, description=f"Operation {index}"
        )
        operations.append(
            StructuredOperation(
                method=_parse_method(operation_object.get("method")),
                path=_parse_path(operation_object.get("path")),
                query=_parse_query(operation_object.get("query")),
                body=operation_object.get("body"),
                allow_failure=_parse_allow_failure(
                    operation_object.get("allow_failure")
                ),
            )
        )
    return operations


def extract_execution_plan(request: SolveRequest) -> ExecutionPlan:
    for file in request.files:
        is_json_file = (
            file.filename.lower().endswith(".json")
            or file.mime_type == "application/json"
        )
        if not is_json_file:
            continue

        document = _extract_json_document(file.decoded_content().decode("utf-8"))
        if document is None:
            continue

        try:
            operations = _parse_operations(document)
        except ValueError:
            continue

        return ExecutionPlan(source=f"file:{file.filename}", operations=operations)

    prompt_document = _extract_json_document(request.prompt)
    if prompt_document is None:
        raise ValueError(
            "This first-pass implementation executes explicit JSON operation "
            "plans only. "
            "Provide a JSON document with an 'operations' list in the prompt or files."
        )

    return ExecutionPlan(source="prompt", operations=_parse_operations(prompt_document))
