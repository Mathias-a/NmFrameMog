"""Tests for the LocalEvaluator."""

from __future__ import annotations

import httpx
from ai_accounting_agent.evaluator import LocalEvaluator, format_report
from ai_accounting_agent.models import (
    EvaluationReport,
    FieldResult,
    TaskResult,
)
from ai_accounting_agent.task_library import ALL_TASKS


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


class TestEvaluatorCreation:
    def test_evaluator_creation(self) -> None:
        """Evaluator can be instantiated without network calls."""
        evaluator = LocalEvaluator(
            agent_url="http://localhost:9999",
            base_url="https://tx-proxy.ainm.no/v2",
            session_token="fake-token",
        )
        assert evaluator.agent_url == "http://localhost:9999"
        assert evaluator.credentials.session_token == "fake-token"
        evaluator.close()

    def test_evaluator_context_manager(self) -> None:
        with LocalEvaluator(
            agent_url="http://localhost:9999",
            base_url="https://tx-proxy.ainm.no/v2",
            session_token="fake-token",
        ) as evaluator:
            assert evaluator is not None


class TestBuildRequestFormat:
    def test_build_request_format(self) -> None:
        """Build request should produce a dict with required keys."""
        with LocalEvaluator(
            agent_url="http://localhost:9999",
            base_url="https://tx-proxy.ainm.no/v2",
            session_token="test-token",
        ) as evaluator:
            task = ALL_TASKS[0]
            request_data = evaluator.build_task_request(task)

            assert "prompt" in request_data
            assert "tripletex_credentials" in request_data
            assert isinstance(request_data["prompt"], str)
            creds = request_data["tripletex_credentials"]
            assert isinstance(creds, dict)

    def test_build_request_includes_prompt_from_task(self) -> None:
        with LocalEvaluator(
            agent_url="http://localhost:9999",
            base_url="https://tx-proxy.ainm.no/v2",
            session_token="test-token",
        ) as evaluator:
            task = ALL_TASKS[0]
            request_data = evaluator.build_task_request(task)
            assert request_data["prompt"] == task.prompt


class TestValidateResponse:
    def test_validate_response_valid(self) -> None:
        """A 200 response with JSON containing 'status' is valid."""
        response = httpx.Response(
            status_code=200,
            json={"status": "completed"},
        )
        assert LocalEvaluator.validate_response(response) is True

    def test_validate_response_invalid_status(self) -> None:
        """Non-200 status code should fail validation."""
        response = httpx.Response(
            status_code=500,
            json={"status": "error"},
        )
        assert LocalEvaluator.validate_response(response) is False

    def test_validate_response_not_json(self) -> None:
        """Non-JSON response should fail validation."""
        response = httpx.Response(
            status_code=200,
            text="not json at all",
            headers={"content-type": "text/plain"},
        )
        assert LocalEvaluator.validate_response(response) is False

    def test_validate_response_json_missing_status(self) -> None:
        """JSON without 'status' key should fail validation."""
        response = httpx.Response(
            status_code=200,
            json={"result": "ok"},
        )
        assert LocalEvaluator.validate_response(response) is False


class TestFormatReport:
    def test_format_report_empty(self) -> None:
        report = EvaluationReport(
            results=[],
            agent_url="http://localhost:8080",
            timestamp="2026-03-19T12:00:00+00:00",
        )
        text = format_report(report)
        assert "Tripletex Agent Evaluation" in text
        assert "0 total" in text
        assert "http://localhost:8080" in text

    def test_format_report_with_results(self) -> None:
        results = [
            _make_task_result(
                name="create_employee_nb",
                score=10.0,
                max_score=10.0,
                fields=[
                    FieldResult(
                        field_name="firstName",
                        expected_value="Ola",
                        actual_value="Ola",
                        correct=True,
                    )
                ],
            ),
            _make_task_result(
                name="create_customer_nb",
                task_type="create_customer",
                score=0.0,
                max_score=10.0,
                format_ok=False,
            ),
        ]
        report = EvaluationReport(
            results=results,
            agent_url="http://localhost:8080",
            timestamp="2026-03-19T12:00:00+00:00",
        )
        text = format_report(report)
        assert "2 total" in text
        assert "1 passed" in text
        assert "1 failed" in text
        assert "create_employee" in text
        assert "Estimated score" in text


class TestRunTaskAgentUnreachable:
    def test_run_task_agent_unreachable(self) -> None:
        """When agent is not running, task should fail gracefully with zero score."""
        with LocalEvaluator(
            agent_url="http://localhost:1",  # unreachable port
            base_url="https://tx-proxy.ainm.no/v2",
            session_token="fake-token",
        ) as evaluator:
            task = ALL_TASKS[0]
            result = evaluator.run_task(task)
            assert result.format_ok is False
            assert result.score == 0.0
