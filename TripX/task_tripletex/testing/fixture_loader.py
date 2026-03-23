from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

from task_tripletex.models import SolveFile, SolveRequest, TripletexCredentials
from task_tripletex.testing.models import (
    CheckDefinition,
    EfficiencyPolicy,
    EvaluationCase,
    ReadDefinition,
)


def _require_mapping(value: object, *, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return value


def _require_string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string.")
    return value


def _require_number(value: object, *, description: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{description} must be a number.")
    return float(value)


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


def _normalize_query_scalar(value: object) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    raise ValueError(
        "Query values must be strings, numbers, booleans, or lists of those values."
    )


def _parse_query(
    raw: object | None,
) -> dict[str, str | int | float | bool | list[str | int | float | bool]]:
    if raw is None:
        return {}
    mapping = _require_mapping(raw, description="Read query")
    query: dict[str, str | int | float | bool | list[str | int | float | bool]] = {}
    for key, value in mapping.items():
        if isinstance(value, list):
            query[key] = [_normalize_query_scalar(item) for item in value]
        else:
            query[key] = _normalize_query_scalar(value)
    return query


def _parse_files(raw: object | None) -> list[SolveFile]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Case files must be a list when provided.")

    files: list[SolveFile] = []
    for index, item in enumerate(cast(list[object], raw), start=1):
        mapping = _require_mapping(item, description=f"File {index}")
        mime_type_raw = mapping.get("mime_type")
        if mime_type_raw is not None and not isinstance(mime_type_raw, str):
            raise ValueError(f"File {index} mime_type must be a string when provided.")
        files.append(
            SolveFile(
                filename=_require_string(
                    mapping.get("filename"), description=f"File {index} filename"
                ),
                content_base64=_require_string(
                    mapping.get("content_base64"),
                    description=f"File {index} content_base64",
                ),
                mime_type=mime_type_raw,
            )
        )
    return files


def _parse_reads(raw: object) -> list[ReadDefinition]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("verification.reads must be a non-empty list.")

    reads: list[ReadDefinition] = []
    for index, item in enumerate(cast(list[object], raw), start=1):
        mapping = _require_mapping(item, description=f"Read {index}")
        mode_raw = mapping.get("mode", "list_values")
        if mode_raw not in {"list_values", "object"}:
            raise ValueError(f"Read {index} mode must be 'list_values' or 'object'.")
        mode = cast(Literal["list_values", "object"], mode_raw)
        path = _require_string(mapping.get("path"), description=f"Read {index} path")
        if not path.startswith("/"):
            raise ValueError(f"Read {index} path must start with '/'.")
        reads.append(
            ReadDefinition(
                name=_require_string(
                    mapping.get("name"), description=f"Read {index} name"
                ),
                path=path,
                query=_parse_query(mapping.get("query")),
                mode=mode,
            )
        )
    return reads


def _parse_checks(raw: object) -> list[CheckDefinition]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("verification.checks must be a non-empty list.")

    checks: list[CheckDefinition] = []
    for index, item in enumerate(cast(list[object], raw), start=1):
        mapping = _require_mapping(item, description=f"Check {index}")
        kind_raw = mapping.get("kind")
        if kind_raw not in {"entity_exists", "field_equals"}:
            raise ValueError(
                f"Check {index} kind must be 'entity_exists' or 'field_equals'."
            )
        kind = cast(Literal["entity_exists", "field_equals"], kind_raw)
        selector_mapping = _require_mapping(
            mapping.get("selector", {}), description=f"Check {index} selector"
        )
        selector: dict[str, object] = {
            key: _normalize_json_value(value) for key, value in selector_mapping.items()
        }
        field_path_raw = mapping.get("field_path")
        if field_path_raw is not None and not isinstance(field_path_raw, str):
            raise ValueError(
                f"Check {index} field_path must be a string when provided."
            )

        checks.append(
            CheckDefinition(
                name=_require_string(
                    mapping.get("name"), description=f"Check {index} name"
                ),
                points=_require_number(
                    mapping.get("points"), description=f"Check {index} points"
                ),
                read_name=_require_string(
                    mapping.get("read_name"), description=f"Check {index} read_name"
                ),
                kind=kind,
                selector=selector,
                field_path=field_path_raw,
                expected=_normalize_json_value(mapping.get("expected")),
            )
        )
    return checks


def _parse_efficiency_policy(raw: object) -> EfficiencyPolicy:
    mapping = _require_mapping(raw, description="efficiency")
    best_write_calls_raw = mapping.get("best_write_calls")
    max_write_calls_raw = mapping.get("max_write_calls")
    max_4xx_errors_raw = mapping.get("max_4xx_errors")
    if not isinstance(best_write_calls_raw, int) or best_write_calls_raw < 0:
        raise ValueError("efficiency.best_write_calls must be a non-negative integer.")
    if (
        not isinstance(max_write_calls_raw, int)
        or max_write_calls_raw < best_write_calls_raw
    ):
        raise ValueError(
            "efficiency.max_write_calls must be an integer greater than or "
            "equal to best_write_calls."
        )
    if not isinstance(max_4xx_errors_raw, int) or max_4xx_errors_raw < 0:
        raise ValueError("efficiency.max_4xx_errors must be a non-negative integer.")

    write_weight = _require_number(
        mapping.get("write_weight"), description="efficiency.write_weight"
    )
    error_weight = _require_number(
        mapping.get("error_weight"), description="efficiency.error_weight"
    )
    if write_weight < 0.0 or error_weight < 0.0 or write_weight + error_weight <= 0.0:
        raise ValueError(
            "efficiency weights must be non-negative and add up to a positive value."
        )

    return EfficiencyPolicy(
        best_write_calls=best_write_calls_raw,
        max_write_calls=max_write_calls_raw,
        max_4xx_errors=max_4xx_errors_raw,
        write_weight=write_weight,
        error_weight=error_weight,
    )


def load_case_fixture(path: str | Path) -> EvaluationCase:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    document = _require_mapping(payload, description="Fixture document")

    verification = _require_mapping(
        document.get("verification"), description="verification"
    )
    tier_raw = document.get("tier")
    if tier_raw not in {1, 2, 3}:
        raise ValueError("tier must be 1, 2, or 3.")
    tier = cast(Literal[1, 2, 3], tier_raw)

    expected_min_proxy_calls_raw = document.get("expected_min_proxy_calls", 0)
    if (
        not isinstance(expected_min_proxy_calls_raw, int)
        or expected_min_proxy_calls_raw < 0
    ):
        raise ValueError("expected_min_proxy_calls must be a non-negative integer.")

    return EvaluationCase(
        case_id=_require_string(document.get("case_id"), description="case_id"),
        description=_require_string(
            document.get("description"), description="description"
        ),
        tier=tier,
        prompt=_require_string(document.get("prompt"), description="prompt"),
        files=_parse_files(document.get("files")),
        reads=_parse_reads(verification.get("reads")),
        checks=_parse_checks(verification.get("checks")),
        efficiency_policy=_parse_efficiency_policy(document.get("efficiency")),
        expected_min_proxy_calls=expected_min_proxy_calls_raw,
    )


def load_packaged_case_fixture(case_name: str) -> EvaluationCase:
    fixture_path = Path(__file__).resolve().parent / "fixtures" / f"{case_name}.json"
    return load_case_fixture(fixture_path)


def build_solve_request(
    case: EvaluationCase, credentials: TripletexCredentials
) -> SolveRequest:
    return SolveRequest(
        prompt=case.prompt,
        files=case.files,
        tripletex_credentials=credentials,
    )
