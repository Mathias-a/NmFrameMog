"""Tests for response models: SolveResponse, TaskResult, EvaluationReport."""

from __future__ import annotations

import pytest
from ai_accounting_agent.models import (
    EvaluationReport,
    FieldResult,
    SolveResponse,
    TaskResult,
)


def _make_task_result(
    name: str = "task_1",
    task_type: str = "create_employee",
    language: str = "nb",
    score: float = 5.0,
    max_score: float = 10.0,
    format_ok: bool = True,
    fields: list[FieldResult] | None = None,
    duration: float = 1.0,
) -> TaskResult:
    """Helper to build a TaskResult with sensible defaults."""
    return TaskResult(
        task_name=name,
        task_type=task_type,
        language=language,
        format_ok=format_ok,
        fields=fields if fields is not None else [],
        score=score,
        max_score=max_score,
        duration_seconds=duration,
    )


def _make_report(
    results: list[TaskResult] | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        results=results if results is not None else [],
        agent_url="http://localhost:8080",
        timestamp="2026-03-19T12:00:00+00:00",
    )


class TestValidResponseSerializes:
    def test_valid_response_serializes(self) -> None:
        resp = SolveResponse(status="completed")
        data = resp.model_dump()
        assert data["status"] == "completed"
        assert isinstance(data, dict)


class TestMinimalResponse:
    def test_minimal_response(self) -> None:
        resp = SolveResponse(status="ok")
        assert resp.status == "ok"
        json_str = resp.model_dump_json()
        assert "ok" in json_str


class TestTaskResultProperties:
    def test_passed_when_full_score(self) -> None:
        result = _make_task_result(score=10.0, max_score=10.0)
        assert result.passed is True

    def test_not_passed_when_partial(self) -> None:
        result = _make_task_result(score=5.0, max_score=10.0)
        assert result.passed is False

    def test_score_ratio_normal(self) -> None:
        result = _make_task_result(score=7.5, max_score=10.0)
        assert result.score_ratio == pytest.approx(0.75)

    def test_score_ratio_zero_max(self) -> None:
        result = _make_task_result(score=0.0, max_score=0.0)
        assert result.score_ratio == 0.0

    def test_score_ratio_zero_score(self) -> None:
        result = _make_task_result(score=0.0, max_score=10.0)
        assert result.score_ratio == 0.0


class TestEvaluationReportAggregation:
    def test_total_tasks(self) -> None:
        report = _make_report([_make_task_result(), _make_task_result(name="task_2")])
        assert report.total_tasks == 2

    def test_passed_tasks(self) -> None:
        report = _make_report(
            [
                _make_task_result(score=10.0, max_score=10.0),
                _make_task_result(name="t2", score=0.0, max_score=10.0),
            ]
        )
        assert report.passed_tasks == 1

    def test_partial_tasks(self) -> None:
        report = _make_report([_make_task_result(score=5.0, max_score=10.0)])
        assert report.partial_tasks == 1

    def test_failed_tasks(self) -> None:
        report = _make_report([_make_task_result(score=0.0, max_score=10.0)])
        assert report.failed_tasks == 1

    def test_total_score(self) -> None:
        report = _make_report(
            [
                _make_task_result(score=5.0, max_score=10.0),
                _make_task_result(name="t2", score=3.0, max_score=10.0),
            ]
        )
        assert report.total_score == pytest.approx(8.0)
        assert report.total_max_score == pytest.approx(20.0)

    def test_score_ratio(self) -> None:
        report = _make_report([_make_task_result(score=8.0, max_score=10.0)])
        assert report.score_ratio == pytest.approx(0.8)

    def test_score_ratio_empty(self) -> None:
        report = _make_report([])
        assert report.score_ratio == 0.0


class TestReportPerLanguageScores:
    def test_per_language_scores(self) -> None:
        report = _make_report(
            [
                _make_task_result(language="nb", score=5.0, max_score=10.0),
                _make_task_result(name="t2", language="en", score=3.0, max_score=10.0),
                _make_task_result(name="t3", language="nb", score=2.0, max_score=10.0),
            ]
        )
        lang_scores = report.per_language_scores
        assert "nb" in lang_scores
        assert "en" in lang_scores
        nb_score, nb_max = lang_scores["nb"]
        assert nb_score == pytest.approx(7.0)
        assert nb_max == pytest.approx(20.0)
        en_score, en_max = lang_scores["en"]
        assert en_score == pytest.approx(3.0)
        assert en_max == pytest.approx(10.0)


class TestReportPerTypeScores:
    def test_per_type_scores(self) -> None:
        report = _make_report(
            [
                _make_task_result(
                    task_type="create_employee", score=5.0, max_score=10.0
                ),
                _make_task_result(
                    name="t2", task_type="create_customer", score=8.0, max_score=10.0
                ),
            ]
        )
        type_scores = report.per_type_scores
        assert "create_employee" in type_scores
        assert "create_customer" in type_scores
        emp_score, emp_max = type_scores["create_employee"]
        assert emp_score == pytest.approx(5.0)
        assert emp_max == pytest.approx(10.0)


class TestReportEmptyResults:
    def test_report_empty_results(self) -> None:
        report = _make_report([])
        assert report.total_tasks == 0
        assert report.passed_tasks == 0
        assert report.partial_tasks == 0
        assert report.failed_tasks == 0
        assert report.total_score == 0.0
        assert report.total_max_score == 0.0
        assert report.per_language_scores == {}
        assert report.per_type_scores == {}
