"""Field-by-field verification logic for the Tripletex local evaluator."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime

from ai_accounting_agent.models import FieldResult
from ai_accounting_agent.task_library import TaskDefinition, TaskType
from ai_accounting_agent.tripletex_client import (
    EntityDict,
    FieldValue,
    TripletexAPIError,
    TripletexClient,
)

logger = logging.getLogger(__name__)

# Type alias for flat entity dicts used in verification results
FlatEntity = dict[str, str | int | float | bool | None]

# Task types where success means the entity is GONE (not found)
_DELETE_TASK_TYPES: frozenset[TaskType] = frozenset({TaskType.DELETE_TRAVEL_EXPENSE})

# Date format patterns for parsing
_DATE_PATTERNS: list[tuple[str, str]] = [
    (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
    (r"^\d{2}\.\d{2}\.\d{4}$", "%d.%m.%Y"),
    (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
]

_DATE_LIKE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{2}[./]\d{2}[./]\d{4}$")

_FLOAT_EPSILON = 1e-6


@dataclass
class VerificationResult:
    """Result of verifying a single task against the Tripletex API."""

    entity_found: bool
    field_results: list[FieldResult]
    score: float
    max_score: float
    raw_entity: FlatEntity | None


def fuzzy_match(expected: str, actual: str) -> bool:
    """Case-insensitive, whitespace-stripped string comparison."""
    return expected.strip().lower() == actual.strip().lower()


def _parse_date(value: str) -> date | None:
    """Attempt to parse a date string in several known formats."""

    for pattern, fmt in _DATE_PATTERNS:
        if re.match(pattern, value.strip()):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
    return None


def date_match(expected: str, actual: str) -> bool:
    """Compare two date strings that may be in different formats."""
    expected_date = _parse_date(expected)
    actual_date = _parse_date(actual)
    if expected_date is None or actual_date is None:
        # Fall back to fuzzy string match if parsing fails
        return fuzzy_match(expected, actual)
    return expected_date == actual_date


def _is_date_like(value: str) -> bool:
    """Check whether a string looks like a date."""
    return bool(_DATE_LIKE_PATTERN.match(value.strip()))


def _coerce_to_bool(value: str | int | float | bool | None) -> bool | None:
    """Try to interpret a value as a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    if isinstance(value, int | float):
        return bool(value)
    return None


def _coerce_to_float(value: str | int | float | bool | None) -> float | None:
    """Try to interpret a value as a float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def value_matches(
    expected: str | int | float | bool,
    actual: str | int | float | bool | None,
) -> bool:
    """Smart comparison handling type coercions and format differences."""
    if actual is None:
        return False

    # Bool comparison
    if isinstance(expected, bool):
        actual_bool = _coerce_to_bool(actual)
        return actual_bool is not None and expected == actual_bool

    # Numeric comparison (int/float) -- check before str since "123" might be expected
    if isinstance(expected, int | float):
        actual_num = _coerce_to_float(actual)
        if actual_num is not None:
            return abs(float(expected) - actual_num) < _FLOAT_EPSILON
        return False

    # String comparison — at this point expected is narrowed to str
    actual_str = str(actual) if not isinstance(actual, str) else actual

    # Date comparison if both look like dates
    if _is_date_like(expected) and _is_date_like(actual_str):
        return date_match(expected, actual_str)

    # Try numeric comparison if expected looks numeric
    expected_num = _coerce_to_float(expected)
    actual_num = _coerce_to_float(actual)
    if expected_num is not None and actual_num is not None:
        return abs(expected_num - actual_num) < _FLOAT_EPSILON

    # Try bool comparison if expected looks boolean
    expected_bool = _coerce_to_bool(expected)
    actual_bool = _coerce_to_bool(actual)
    if expected_bool is not None and actual_bool is not None:
        return expected_bool == actual_bool

    # Fall back to fuzzy string match
    return fuzzy_match(expected, actual_str)


def _get_nested_field(entity: EntityDict, dotted_key: str) -> FieldValue:
    """Retrieve a value from an entity dict, supporting dotted keys for nesting."""
    parts = dotted_key.split(".")
    current: object = entity
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
            if current is None:
                return None
        else:
            return None
    if isinstance(current, str | int | float | bool):
        return current
    return None


def _flatten_entity(entity: EntityDict) -> FlatEntity:
    """Flatten an EntityDict to a simple field-value dict for storage."""
    result: FlatEntity = {}
    for key, value in entity.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                result[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, list):
            # Skip list fields in flattening
            continue
        else:
            result[key] = value
    return result


def _search_entity_list(
    entities: list[EntityDict],
    expected_fields: dict[str, str | int | float | bool],
    match_keys: list[str],
) -> EntityDict | None:
    """Find the best matching entity from a list by comparing key fields."""
    # Only consider keys that actually exist in expected_fields
    effective_keys = [k for k in match_keys if k in expected_fields]
    if not effective_keys:
        # No criteria to match on — cannot reliably pick an entity
        return None

    for entity in entities:
        all_match = True
        for key in effective_keys:
            actual_value = _get_nested_field(entity, key)
            if actual_value is None:
                all_match = False
                break
            if not value_matches(expected_fields[key], actual_value):
                all_match = False
                break
        if all_match:
            return entity
    return None


def _filter_new_entities(
    entities: list[EntityDict],
    known_ids: set[int] | None,
) -> list[EntityDict]:
    """Remove entities that existed before the task was run."""
    if known_ids is None:
        return entities
    result: list[EntityDict] = []
    for entity in entities:
        entity_id = entity.get("id")
        if isinstance(entity_id, int) and entity_id not in known_ids:
            result.append(entity)
    return result


def find_entity(
    client: TripletexClient,
    task: TaskDefinition,
    known_ids: set[int] | None = None,
) -> EntityDict | None:
    """Search the Tripletex API for the entity a task should have created.

    Args:
        client: Authenticated Tripletex API client.
        task: The task definition to verify.
        known_ids: Entity IDs that existed before the task was sent.
            If provided, only entities with IDs NOT in this set are considered.
    """
    entity_type = task.expected_entity
    expected = task.expected_fields

    try:
        match entity_type:
            case "employee":
                first_name = str(expected.get("firstName", ""))
                last_name = str(expected.get("lastName", ""))
                if first_name or last_name:
                    entities = client.search_employees(
                        first_name=first_name or None,
                        last_name=last_name or None,
                    )
                else:
                    entities = client.list_employees()
                entities = _filter_new_entities(entities, known_ids)
                return _search_entity_list(
                    entities, expected, ["firstName", "lastName"]
                )

            case "customer":
                name = str(expected.get("name", ""))
                if name:
                    entities = client.search_customers(name=name)
                else:
                    entities = client.list_customers()
                entities = _filter_new_entities(entities, known_ids)
                return _search_entity_list(entities, expected, ["name"])

            case "product":
                name = str(expected.get("name", ""))
                entities = (
                    client.list_products(name=name) if name else client.list_products()
                )
                entities = _filter_new_entities(entities, known_ids)
                return _search_entity_list(entities, expected, ["name"])

            case "invoice":
                # Search by customer name or date
                params: dict[str, str | int] = {}
                if "invoiceDate" in expected:
                    params["invoiceDateFrom"] = str(expected["invoiceDate"])
                    params["invoiceDateTo"] = str(expected["invoiceDate"])
                entities = client.list_invoices(**params)
                entities = _filter_new_entities(entities, known_ids)
                match_keys = [
                    key
                    for key in ["invoiceDate", "invoiceDueDate", "customer.name"]
                    if key in expected
                ]
                return _search_entity_list(entities, expected, match_keys)

            case "project":
                name = str(expected.get("name", ""))
                entities = (
                    client.list_projects(name=name) if name else client.list_projects()
                )
                entities = _filter_new_entities(entities, known_ids)
                return _search_entity_list(entities, expected, ["name"])

            case "department":
                name = str(expected.get("name", ""))
                entities = (
                    client.list_departments(name=name)
                    if name
                    else client.list_departments()
                )
                entities = _filter_new_entities(entities, known_ids)
                return _search_entity_list(entities, expected, ["name"])

            case "contact":
                entities = client.list_contacts()
                entities = _filter_new_entities(entities, known_ids)
                return _search_entity_list(
                    entities, expected, ["firstName", "lastName"]
                )

            case "travelExpense":
                # Request expanded employee fields for nested matching
                entities = client.list_travel_expenses(
                    fields="id,title,date,amount,employee(firstName,lastName)",
                )
                entities = _filter_new_entities(entities, known_ids)
                return _search_entity_list(entities, expected, list(expected.keys()))

            case _:
                # Generic fallback: try listing via the entity type as endpoint
                logger.warning(
                    "Unknown entity type '%s', attempting generic list",
                    entity_type,
                )
                entities = client._list(entity_type)
                entities = _filter_new_entities(entities, known_ids)
                return _search_entity_list(entities, expected, list(expected.keys()))

    except TripletexAPIError as exc:
        logger.error("API error while searching for %s: %s", entity_type, exc)
        return None


def verify_task(
    client: TripletexClient,
    task: TaskDefinition,
    known_ids: set[int] | None = None,
) -> VerificationResult:
    """Verify that a task was completed correctly by checking the API state.

    Args:
        client: Authenticated Tripletex API client.
        task: The task definition to verify.
        known_ids: Entity IDs that existed before the task was sent.
            Passed through to find_entity for state isolation.
    """
    entity = find_entity(client, task, known_ids=known_ids)
    is_delete = task.task_type in _DELETE_TASK_TYPES

    if entity is None:
        if is_delete:
            # Entity gone = success for delete tasks
            return VerificationResult(
                entity_found=False,
                field_results=[],
                score=task.max_points,
                max_score=task.max_points,
                raw_entity=None,
            )
        return VerificationResult(
            entity_found=False,
            field_results=[],
            score=0.0,
            max_score=task.max_points,
            raw_entity=None,
        )

    if is_delete:
        # Entity still exists = failure for delete tasks
        return VerificationResult(
            entity_found=True,
            field_results=[],
            score=0.0,
            max_score=task.max_points,
            raw_entity=_flatten_entity(entity),
        )

    flat = _flatten_entity(entity)
    field_results: list[FieldResult] = []
    score = 0.0
    field_points = task.field_points

    # Award points for finding the entity
    found_points = field_points.get("_found", 0.0)
    score += found_points

    # Determine per-field point values
    expected_fields = task.expected_fields
    num_fields = len(expected_fields)

    # If field_points is empty or only has _found, distribute remaining points equally
    remaining_points = task.max_points - found_points
    has_explicit_field_points = any(key != "_found" for key in field_points)

    for field_name, expected_value in expected_fields.items():
        actual_value = _get_nested_field(entity, field_name)
        actual_str = str(actual_value) if actual_value is not None else None

        correct = value_matches(expected_value, actual_value)

        field_results.append(
            FieldResult(
                field_name=field_name,
                expected_value=str(expected_value),
                actual_value=actual_str,
                correct=correct,
            )
        )

        if correct:
            if has_explicit_field_points and field_name in field_points:
                score += field_points[field_name]
            elif not has_explicit_field_points and num_fields > 0:
                score += remaining_points / num_fields

    return VerificationResult(
        entity_found=True,
        field_results=field_results,
        score=score,
        max_score=task.max_points,
        raw_entity=flat,
    )
