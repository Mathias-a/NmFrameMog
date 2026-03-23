from __future__ import annotations

from typing import cast

from task_tripletex.models import TripletexCredentials
from task_tripletex.testing.models import (
    CheckDefinition,
    CheckResult,
    EvaluationCase,
    VerificationResult,
)
from task_tripletex.testing.tripletex_read_helper import TripletexReadHelper

_MISSING: object = object()


def _get_path_value(value: object, path: str) -> object:
    current: object = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part, _MISSING)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if 0 <= index < len(current) else _MISSING
        else:
            return _MISSING
        if current is _MISSING:
            return _MISSING
    return current


def _matches_selector(candidate: object, selector: dict[str, object]) -> bool:
    if not isinstance(candidate, dict):
        return False
    for path, expected in selector.items():
        observed = _get_path_value(candidate, path)
        if observed is _MISSING or observed != expected:
            return False
    return True


def _find_match(snapshot: object, selector: dict[str, object]) -> object | None:
    if isinstance(snapshot, list):
        for item in cast(list[object], snapshot):
            if _matches_selector(item, selector):
                return item
        return None
    if _matches_selector(snapshot, selector):
        return snapshot
    return None


def _evaluate_check(snapshot: object, check: CheckDefinition) -> CheckResult:
    matched = _find_match(snapshot, check.selector)
    if check.kind == "entity_exists":
        passed = matched is not None
        return CheckResult(
            name=check.name,
            points_awarded=check.points if passed else 0.0,
            max_points=check.points,
            passed=passed,
            details="Matched at least one entity."
            if passed
            else "No entity matched the selector.",
        )

    if matched is None:
        return CheckResult(
            name=check.name,
            points_awarded=0.0,
            max_points=check.points,
            passed=False,
            details="No entity matched the selector.",
        )
    if check.field_path is None:
        raise ValueError(
            f"Check '{check.name}' must define field_path for field_equals."
        )

    observed = _get_path_value(matched, check.field_path)
    passed = observed is not _MISSING and observed == check.expected
    if observed is _MISSING:
        details = f"Field '{check.field_path}' was missing."
    else:
        details = f"Observed {observed!r}; expected {check.expected!r}."
    return CheckResult(
        name=check.name,
        points_awarded=check.points if passed else 0.0,
        max_points=check.points,
        passed=passed,
        details=details,
    )


async def verify_case(
    case: EvaluationCase,
    credentials: TripletexCredentials,
    *,
    timeout_seconds: float = 30.0,
) -> VerificationResult:
    snapshots: dict[str, object] = {}
    async with TripletexReadHelper(
        credentials,
        timeout_seconds=timeout_seconds,
    ) as helper:
        for read in case.reads:
            if read.mode == "list_values":
                snapshots[read.name] = await helper.fetch_values(read.path, read.query)
            else:
                snapshots[read.name] = await helper.fetch_json(read.path, read.query)

    results: list[CheckResult] = []
    for check in case.checks:
        if check.read_name not in snapshots:
            raise ValueError(
                f"Unknown read_name '{check.read_name}' in check '{check.name}'."
            )
        results.append(_evaluate_check(snapshots[check.read_name], check))

    points_earned = sum(result.points_awarded for result in results)
    max_points = sum(result.max_points for result in results)
    correctness = 0.0 if max_points == 0.0 else points_earned / max_points
    return VerificationResult(
        points_earned=points_earned,
        max_points=max_points,
        correctness=correctness,
        checks=results,
        snapshots=snapshots,
    )
