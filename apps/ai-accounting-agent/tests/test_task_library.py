"""Tests for task library coverage and consistency."""

from __future__ import annotations

from collections import Counter

import pytest
from ai_accounting_agent.task_library import (
    ALL_TASKS,
    Language,
    TaskType,
    Tier,
    get_all_languages,
    get_all_task_types,
    get_tasks_by_language,
    get_tasks_by_tier,
    get_tasks_by_type,
)


class TestAllTasksHavePrompts:
    def test_all_tasks_have_prompts(self) -> None:
        for task in ALL_TASKS:
            assert task.prompt, f"Task {task.name!r} has empty prompt"
            assert len(task.prompt) > 5, (
                f"Task {task.name!r} has suspiciously short prompt: {task.prompt!r}"
            )


class TestAllTasksHaveExpectedFields:
    def test_all_tasks_have_expected_entity(self) -> None:
        for task in ALL_TASKS:
            assert task.expected_entity, f"Task {task.name!r} has no expected_entity"

    def test_all_tasks_have_name(self) -> None:
        for task in ALL_TASKS:
            assert task.name, "Found a task with empty name"

    def test_all_tasks_have_valid_language(self) -> None:
        for task in ALL_TASKS:
            assert isinstance(task.language, Language), (
                f"Task {task.name!r} has invalid language: {task.language}"
            )

    def test_all_tasks_have_valid_type(self) -> None:
        for task in ALL_TASKS:
            assert isinstance(task.task_type, TaskType), (
                f"Task {task.name!r} has invalid task_type: {task.task_type}"
            )

    def test_all_tasks_have_valid_tier(self) -> None:
        for task in ALL_TASKS:
            assert isinstance(task.tier, Tier), (
                f"Task {task.name!r} has invalid tier: {task.tier}"
            )


class TestAllLanguagesRepresented:
    def test_all_languages_represented(self) -> None:
        """Every language in the task library should have at least one task."""
        present = get_all_languages()
        assert len(present) >= 2, "Expected at least 2 languages"
        # Verify nb and en are always present (core languages)
        assert Language.NB in present, "Norwegian Bokmal (nb) missing from tasks"
        assert Language.EN in present, "English (en) missing from tasks"


class TestNoDuplicateTaskNames:
    def test_no_duplicate_task_names(self) -> None:
        names = [task.name for task in ALL_TASKS]
        counts = Counter(names)
        duplicates = {name: count for name, count in counts.items() if count > 1}
        assert not duplicates, f"Duplicate task names found: {duplicates}"


class TestAllTaskTypesHaveVariations:
    def test_all_task_types_have_variations(self) -> None:
        """Each task type present in the library should have at least 2 tasks."""
        type_counts = Counter(task.task_type for task in ALL_TASKS)
        for task_type, count in type_counts.items():
            assert count >= 2, (
                f"Task type {task_type.value!r} has only {count} task(s), need >= 2"
            )


class TestTaskTypesEnumCoverage:
    def test_task_types_enum_coverage(self) -> None:
        """The task library should cover a reasonable subset of TaskType values."""
        present_types = get_all_task_types()
        total_enum_values = len(TaskType)
        # We expect at least some coverage (not all types need tasks yet)
        assert len(present_types) >= 3, (
            f"Only {len(present_types)} task types covered out of {total_enum_values}"
        )


class TestGetTasksByType:
    @pytest.mark.parametrize("task_type", list(get_all_task_types()))
    def test_get_tasks_by_type_returns_correct_type(self, task_type: TaskType) -> None:
        tasks = get_tasks_by_type(task_type)
        assert len(tasks) > 0, f"No tasks for type {task_type.value}"
        for task in tasks:
            assert task.task_type == task_type


class TestGetTasksByLanguage:
    @pytest.mark.parametrize("language", list(get_all_languages()))
    def test_get_tasks_by_language_returns_correct_language(
        self, language: Language
    ) -> None:
        tasks = get_tasks_by_language(language)
        assert len(tasks) > 0, f"No tasks for language {language.value}"
        for task in tasks:
            assert task.language == language


class TestGetTasksByTier:
    @pytest.mark.parametrize(
        "tier",
        [tier for tier in Tier if get_tasks_by_tier(tier)],
    )
    def test_get_tasks_by_tier_returns_correct_tier(self, tier: Tier) -> None:
        tasks = get_tasks_by_tier(tier)
        assert len(tasks) > 0, f"No tasks for tier {tier.value}"
        for task in tasks:
            assert task.tier == tier


class TestFieldPointsSumMatchesMaxPoints:
    def test_field_points_sum_equals_max_points(self) -> None:
        """Field_points sum must exactly equal max_points."""
        for task in ALL_TASKS:
            if task.field_points:
                total_field_points = sum(task.field_points.values())
                assert abs(total_field_points - task.max_points) < 0.01, (
                    f"Task {task.name!r}: field_points sum {total_field_points} "
                    f"!= max_points {task.max_points}"
                )
