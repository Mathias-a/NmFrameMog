"""Tests for field-by-field verification logic (no sandbox needed)."""

from __future__ import annotations

from ai_accounting_agent.models import FieldResult
from ai_accounting_agent.verification import (
    VerificationResult,
    date_match,
    fuzzy_match,
    value_matches,
)


class TestFuzzyMatchBasic:
    def test_fuzzy_match_basic(self) -> None:
        assert fuzzy_match("hello", "hello") is True
        assert fuzzy_match("hello", "world") is False

    def test_exact_match(self) -> None:
        assert fuzzy_match("Ola Nordmann", "Ola Nordmann") is True


class TestFuzzyMatchCaseInsensitive:
    def test_fuzzy_match_case_insensitive(self) -> None:
        assert fuzzy_match("Hello", "hello") is True
        assert fuzzy_match("JOHN", "john") is True
        assert fuzzy_match("Nordisk AS", "nordisk as") is True


class TestFuzzyMatchWhitespace:
    def test_fuzzy_match_whitespace(self) -> None:
        assert fuzzy_match("  hello  ", "hello") is True
        assert fuzzy_match("hello", "  hello  ") is True
        assert fuzzy_match("  Ola  ", "  Ola  ") is True


class TestDateMatchSameFormat:
    def test_date_match_same_format(self) -> None:
        assert date_match("2026-03-15", "2026-03-15") is True
        assert date_match("2026-03-15", "2026-04-15") is False

    def test_date_match_dd_mm_yyyy(self) -> None:
        assert date_match("15.03.2026", "15.03.2026") is True


class TestDateMatchDifferentFormats:
    def test_date_match_different_formats(self) -> None:
        assert date_match("2026-03-15", "15.03.2026") is True
        assert date_match("15.03.2026", "2026-03-15") is True
        assert date_match("2026-03-15", "15/03/2026") is True

    def test_date_match_mismatch(self) -> None:
        assert date_match("2026-03-15", "16.03.2026") is False


class TestValueMatchesStrings:
    def test_value_matches_strings(self) -> None:
        assert value_matches("Ola", "Ola") is True
        assert value_matches("Ola", "ola") is True
        assert value_matches("Ola", "Kari") is False

    def test_value_matches_strings_whitespace(self) -> None:
        assert value_matches("Ola", "  Ola  ") is True


class TestValueMatchesBooleans:
    def test_value_matches_booleans(self) -> None:
        assert value_matches(True, True) is True
        assert value_matches(False, False) is True
        assert value_matches(True, False) is False
        assert value_matches(False, True) is False

    def test_value_matches_bool_coercion(self) -> None:
        assert value_matches(True, "true") is True
        assert value_matches(True, "True") is True
        assert value_matches(False, "false") is True
        assert value_matches(True, 1) is True
        assert value_matches(False, 0) is True

    def test_value_matches_bool_none(self) -> None:
        assert value_matches(True, None) is False


class TestValueMatchesNumbers:
    def test_value_matches_integers(self) -> None:
        assert value_matches(42, 42) is True
        assert value_matches(42, 42.0) is True
        assert value_matches(42, "42") is True

    def test_value_matches_floats(self) -> None:
        assert value_matches(4999.00, 4999.0) is True
        assert value_matches(4999.00, "4999.00") is True
        assert value_matches(4999.00, 5000.0) is False

    def test_value_matches_number_none(self) -> None:
        assert value_matches(42, None) is False


class TestValueMatchesNone:
    def test_value_matches_none(self) -> None:
        assert value_matches("anything", None) is False
        assert value_matches(True, None) is False
        assert value_matches(42, None) is False


class TestVerificationResultZeroScore:
    def test_verification_result_zero_score(self) -> None:
        result = VerificationResult(
            entity_found=False,
            field_results=[],
            score=0.0,
            max_score=10.0,
            raw_entity=None,
        )
        assert result.score == 0.0
        assert result.max_score == 10.0
        assert result.entity_found is False


class TestVerificationResultFullScore:
    def test_verification_result_full_score(self) -> None:
        result = VerificationResult(
            entity_found=True,
            field_results=[
                FieldResult(
                    field_name="firstName",
                    expected_value="Ola",
                    actual_value="Ola",
                    correct=True,
                ),
                FieldResult(
                    field_name="lastName",
                    expected_value="Nordmann",
                    actual_value="Nordmann",
                    correct=True,
                ),
            ],
            score=10.0,
            max_score=10.0,
            raw_entity={"firstName": "Ola", "lastName": "Nordmann"},
        )
        assert result.score == result.max_score
        assert result.entity_found is True
        assert len(result.field_results) == 2
        assert all(fr.correct for fr in result.field_results)


class TestVerificationResultPartialScore:
    def test_verification_result_partial_score(self) -> None:
        result = VerificationResult(
            entity_found=True,
            field_results=[
                FieldResult(
                    field_name="firstName",
                    expected_value="Ola",
                    actual_value="Ola",
                    correct=True,
                ),
                FieldResult(
                    field_name="email",
                    expected_value="ola@example.com",
                    actual_value=None,
                    correct=False,
                ),
            ],
            score=5.0,
            max_score=10.0,
            raw_entity={"firstName": "Ola", "email": None},
        )
        assert result.score < result.max_score
        assert result.score > 0.0
        correct_count = sum(1 for fr in result.field_results if fr.correct)
        incorrect_count = sum(1 for fr in result.field_results if not fr.correct)
        assert correct_count == 1
        assert incorrect_count == 1


# -- Tests for _search_entity_list ------------------------------------------


class TestSearchEntityListEmptyMatchKeys:
    def test_empty_match_keys_returns_none(self) -> None:
        """Empty match_keys should return None, not the first entity."""
        from ai_accounting_agent.verification import _search_entity_list

        entities: list[
            dict[
                str,
                str
                | int
                | float
                | bool
                | None
                | dict[str, str | int | float | bool | None]
                | list[dict[str, str | int | float | bool | None]],
            ]
        ] = [
            {"id": 1, "name": "Test"},
            {"id": 2, "name": "Other"},
        ]
        result = _search_entity_list(entities, {}, [])
        assert result is None

    def test_match_keys_not_in_expected_returns_none(self) -> None:
        from ai_accounting_agent.verification import _search_entity_list

        entities: list[
            dict[
                str,
                str
                | int
                | float
                | bool
                | None
                | dict[str, str | int | float | bool | None]
                | list[dict[str, str | int | float | bool | None]],
            ]
        ] = [
            {"id": 1, "name": "Test"},
        ]
        result = _search_entity_list(entities, {}, ["name"])
        assert result is None


class TestSearchEntityListNormalMatch:
    def test_finds_matching_entity(self) -> None:
        from ai_accounting_agent.verification import _search_entity_list

        entities: list[
            dict[
                str,
                str
                | int
                | float
                | bool
                | None
                | dict[str, str | int | float | bool | None]
                | list[dict[str, str | int | float | bool | None]],
            ]
        ] = [
            {"id": 1, "firstName": "Ola", "lastName": "Nordmann"},
            {"id": 2, "firstName": "Kari", "lastName": "Berg"},
        ]
        expected = {"firstName": "Kari", "lastName": "Berg"}
        result = _search_entity_list(entities, expected, ["firstName", "lastName"])
        assert result is not None
        assert result["id"] == 2

    def test_no_match_returns_none(self) -> None:
        from ai_accounting_agent.verification import _search_entity_list

        entities: list[
            dict[
                str,
                str
                | int
                | float
                | bool
                | None
                | dict[str, str | int | float | bool | None]
                | list[dict[str, str | int | float | bool | None]],
            ]
        ] = [
            {"id": 1, "firstName": "Ola", "lastName": "Nordmann"},
        ]
        expected = {"firstName": "Hans", "lastName": "Müller"}
        result = _search_entity_list(entities, expected, ["firstName", "lastName"])
        assert result is None


# -- Tests for _get_nested_field --------------------------------------------


class TestGetNestedField:
    def test_simple_key(self) -> None:
        from ai_accounting_agent.verification import _get_nested_field

        entity: dict[
            str,
            str
            | int
            | float
            | bool
            | None
            | dict[str, str | int | float | bool | None]
            | list[dict[str, str | int | float | bool | None]],
        ] = {
            "firstName": "Ola",
            "email": "ola@example.com",
        }
        assert _get_nested_field(entity, "firstName") == "Ola"
        assert _get_nested_field(entity, "email") == "ola@example.com"

    def test_dotted_key(self) -> None:
        from ai_accounting_agent.verification import _get_nested_field

        entity: dict[
            str,
            str
            | int
            | float
            | bool
            | None
            | dict[str, str | int | float | bool | None]
            | list[dict[str, str | int | float | bool | None]],
        ] = {
            "customer": {"name": "Nordisk AS", "id": 42},
        }
        assert _get_nested_field(entity, "customer.name") == "Nordisk AS"
        assert _get_nested_field(entity, "customer.id") == 42

    def test_missing_key_returns_none(self) -> None:
        from ai_accounting_agent.verification import _get_nested_field

        entity: dict[
            str,
            str
            | int
            | float
            | bool
            | None
            | dict[str, str | int | float | bool | None]
            | list[dict[str, str | int | float | bool | None]],
        ] = {
            "firstName": "Ola",
        }
        assert _get_nested_field(entity, "lastName") is None
        assert _get_nested_field(entity, "customer.name") is None


# -- Tests for _filter_new_entities -----------------------------------------


class TestFilterNewEntities:
    def test_filters_known_ids(self) -> None:
        from ai_accounting_agent.verification import _filter_new_entities

        entities: list[
            dict[
                str,
                str
                | int
                | float
                | bool
                | None
                | dict[str, str | int | float | bool | None]
                | list[dict[str, str | int | float | bool | None]],
            ]
        ] = [
            {"id": 1, "name": "Old"},
            {"id": 2, "name": "Old2"},
            {"id": 3, "name": "New"},
        ]
        result = _filter_new_entities(entities, known_ids={1, 2})
        assert len(result) == 1
        assert result[0]["id"] == 3

    def test_none_known_ids_returns_all(self) -> None:
        from ai_accounting_agent.verification import _filter_new_entities

        entities: list[
            dict[
                str,
                str
                | int
                | float
                | bool
                | None
                | dict[str, str | int | float | bool | None]
                | list[dict[str, str | int | float | bool | None]],
            ]
        ] = [
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
        ]
        result = _filter_new_entities(entities, known_ids=None)
        assert len(result) == 2

    def test_all_known_returns_empty(self) -> None:
        from ai_accounting_agent.verification import _filter_new_entities

        entities: list[
            dict[
                str,
                str
                | int
                | float
                | bool
                | None
                | dict[str, str | int | float | bool | None]
                | list[dict[str, str | int | float | bool | None]],
            ]
        ] = [
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
        ]
        result = _filter_new_entities(entities, known_ids={1, 2})
        assert len(result) == 0


# -- Mock-based tests for verify_task and find_entity -----------------------


def _make_mock_client(
    entities: list[
        dict[
            str,
            str
            | int
            | float
            | bool
            | None
            | dict[str, str | int | float | bool | None]
            | list[dict[str, str | int | float | bool | None]],
        ]
    ],
) -> object:
    """Create a mock object that behaves like TripletexClient for search methods."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.search_employees.return_value = entities
    client.search_customers.return_value = entities
    client.list_employees.return_value = entities
    client.list_customers.return_value = entities
    client.list_products.return_value = entities
    client.list_invoices.return_value = entities
    client.list_projects.return_value = entities
    client.list_departments.return_value = entities
    client.list_contacts.return_value = entities
    client.list_travel_expenses.return_value = entities
    return client


class TestVerifyTaskCreateSuccess:
    def test_create_employee_full_score(self) -> None:
        from ai_accounting_agent.task_library import (
            Language,
            TaskDefinition,
            TaskType,
            Tier,
        )
        from ai_accounting_agent.verification import verify_task

        task = TaskDefinition(
            name="test_create",
            prompt="Create employee Ola Nordmann with email ola@example.com",
            language=Language.EN,
            task_type=TaskType.CREATE_EMPLOYEE,
            tier=Tier.TIER_1,
            expected_entity="employee",
            expected_fields={
                "firstName": "Ola",
                "lastName": "Nordmann",
                "email": "ola@example.com",
            },
            max_points=5.0,
            field_points={
                "_found": 2.0,
                "firstName": 1.0,
                "lastName": 1.0,
                "email": 1.0,
            },
        )
        mock_client = _make_mock_client(
            [
                {
                    "id": 10,
                    "firstName": "Ola",
                    "lastName": "Nordmann",
                    "email": "ola@example.com",
                }
            ]
        )
        result = verify_task(mock_client, task)  # type: ignore[arg-type]
        assert result.entity_found is True
        assert result.score == 5.0
        assert result.max_score == 5.0
        assert all(f.correct for f in result.field_results)


class TestVerifyTaskCreateNotFound:
    def test_create_employee_not_found(self) -> None:
        from ai_accounting_agent.task_library import (
            Language,
            TaskDefinition,
            TaskType,
            Tier,
        )
        from ai_accounting_agent.verification import verify_task

        task = TaskDefinition(
            name="test_not_found",
            prompt="Create employee",
            language=Language.EN,
            task_type=TaskType.CREATE_EMPLOYEE,
            tier=Tier.TIER_1,
            expected_entity="employee",
            expected_fields={"firstName": "Ola", "lastName": "Nordmann"},
            max_points=5.0,
            field_points={"_found": 2.0, "firstName": 1.5, "lastName": 1.5},
        )
        mock_client = _make_mock_client([])
        result = verify_task(mock_client, task)  # type: ignore[arg-type]
        assert result.entity_found is False
        assert result.score == 0.0


class TestVerifyTaskDeleteSuccess:
    def test_delete_entity_gone_gives_full_score(self) -> None:
        from ai_accounting_agent.task_library import (
            Language,
            TaskDefinition,
            TaskType,
            Tier,
        )
        from ai_accounting_agent.verification import verify_task

        task = TaskDefinition(
            name="test_delete_success",
            prompt="Delete travel expense",
            language=Language.EN,
            task_type=TaskType.DELETE_TRAVEL_EXPENSE,
            tier=Tier.TIER_2,
            expected_entity="travelExpense",
            expected_fields={"employee.firstName": "Ola"},
            max_points=5.0,
            field_points={"_found": 5.0},
        )
        # Empty list = entity is gone (deleted successfully)
        mock_client = _make_mock_client([])
        result = verify_task(mock_client, task)  # type: ignore[arg-type]
        assert result.entity_found is False
        assert result.score == 5.0
        assert result.max_score == 5.0


class TestVerifyTaskDeleteFailure:
    def test_delete_entity_still_exists_gives_zero(self) -> None:
        from ai_accounting_agent.task_library import (
            Language,
            TaskDefinition,
            TaskType,
            Tier,
        )
        from ai_accounting_agent.verification import verify_task

        task = TaskDefinition(
            name="test_delete_fail",
            prompt="Delete travel expense",
            language=Language.EN,
            task_type=TaskType.DELETE_TRAVEL_EXPENSE,
            tier=Tier.TIER_2,
            expected_entity="travelExpense",
            expected_fields={"employee.firstName": "Ola"},
            max_points=5.0,
            field_points={"_found": 5.0},
        )
        # Entity still exists = delete failed
        mock_client = _make_mock_client([{"id": 5, "employee": {"firstName": "Ola"}}])
        result = verify_task(mock_client, task)  # type: ignore[arg-type]
        assert result.entity_found is True
        assert result.score == 0.0


class TestVerifyTaskWithKnownIds:
    def test_known_ids_filters_old_entities(self) -> None:
        from ai_accounting_agent.task_library import (
            Language,
            TaskDefinition,
            TaskType,
            Tier,
        )
        from ai_accounting_agent.verification import verify_task

        task = TaskDefinition(
            name="test_isolation",
            prompt="Create customer",
            language=Language.EN,
            task_type=TaskType.CREATE_CUSTOMER,
            tier=Tier.TIER_1,
            expected_entity="customer",
            expected_fields={"name": "New Corp", "isCustomer": True},
            max_points=5.0,
            field_points={"_found": 2.0, "name": 2.0, "isCustomer": 1.0},
        )
        # Mock returns both old and new entities
        mock_client = _make_mock_client(
            [
                {"id": 1, "name": "Old Corp", "isCustomer": True},
                {"id": 99, "name": "New Corp", "isCustomer": True},
            ]
        )
        # With known_ids={1}, only entity 99 should be considered
        result = verify_task(mock_client, task, known_ids={1})  # type: ignore[arg-type]
        assert result.entity_found is True
        assert result.score == 5.0
        assert result.max_score == 5.0
