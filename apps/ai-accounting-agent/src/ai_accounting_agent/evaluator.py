"""Local evaluator that simulates the NM i AI competition flow.

Sends tasks to a running agent, then verifies results against the Tripletex API
to produce a scored EvaluationReport.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import httpx

from ai_accounting_agent.models import (
    EvaluationReport,
    FileAttachment,
    TaskResult,
    TripletexCredentials,
    build_request,
)
from ai_accounting_agent.task_library import (
    ALL_TASKS,
    TaskDefinition,
    TaskType,
    Tier,
    get_tasks_by_tier,
    get_tasks_by_type,
)
from ai_accounting_agent.tripletex_client import TripletexAPIError, TripletexClient
from ai_accounting_agent.verification import VerificationResult, verify_task


class LocalEvaluator:
    """Drives the full evaluate-and-score pipeline against a local agent."""

    def __init__(
        self,
        agent_url: str,
        base_url: str,
        session_token: str,
    ) -> None:
        self.agent_url = agent_url
        self.credentials = TripletexCredentials(
            base_url=base_url,
            session_token=session_token,
        )
        self.client = TripletexClient(
            base_url=base_url,
            session_token=session_token,
        )
        self._http = httpx.Client(timeout=300.0)  # 5 min timeout like competition

    def close(self) -> None:
        """Release HTTP clients."""
        self.client.close()
        self._http.close()

    def __enter__(self) -> LocalEvaluator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def build_task_request(self, task: TaskDefinition) -> dict[str, object]:
        """Build a SolveRequest dict matching the competition format."""
        files: list[FileAttachment] | None = (
            [
                FileAttachment(
                    filename=f["filename"],
                    content_base64=f["content_base64"],
                )
                for f in task.files
            ]
            if task.files
            else None
        )
        return build_request(
            prompt=task.prompt,
            credentials=self.credentials,
            files=files,
        )

    # ------------------------------------------------------------------
    # Agent communication
    # ------------------------------------------------------------------

    def send_to_agent(self, request_data: dict[str, object]) -> httpx.Response:
        """POST the request to the agent's /solve endpoint."""
        return self._http.post(f"{self.agent_url}/solve", json=request_data)

    @staticmethod
    def validate_response(response: httpx.Response) -> bool:
        """Check response format: 200 status, valid JSON, has 'status' field."""
        if response.status_code != 200:
            return False
        try:
            data: object = response.json()
            return isinstance(data, dict) and "status" in data
        except (json.JSONDecodeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Single task execution
    # ------------------------------------------------------------------

    def run_task(self, task: TaskDefinition) -> TaskResult:
        """Run a single task through the full evaluation pipeline."""
        start = time.monotonic()

        # 0. Snapshot pre-existing entity IDs for state isolation
        try:
            pre_ids: set[int] | None = self.client.list_entity_ids(task.expected_entity)
        except (httpx.HTTPError, TripletexAPIError):
            pre_ids = None  # Cannot snapshot — verification will check all entities

        # 1. Build request
        request_data = self.build_task_request(task)

        # 2. Send to agent
        try:
            response = self.send_to_agent(request_data)
            format_ok = self.validate_response(response)
        except httpx.HTTPError:
            format_ok = False

        # 3. Verify result via Tripletex API (only new entities)
        if format_ok:
            verification = verify_task(self.client, task, known_ids=pre_ids)
        else:
            verification = VerificationResult(
                entity_found=False,
                field_results=[],
                score=0.0,
                max_score=task.max_points,
                raw_entity=None,
            )

        duration = time.monotonic() - start
        tier_multiplier = task.tier.value

        return TaskResult(
            task_name=task.name,
            task_type=task.task_type.value,
            language=task.language.value,
            format_ok=format_ok,
            fields=verification.field_results,
            score=verification.score * tier_multiplier,
            max_score=verification.max_score * tier_multiplier,
            duration_seconds=round(duration, 2),
        )

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def run_all(
        self,
        tasks: list[TaskDefinition] | None = None,
    ) -> EvaluationReport:
        """Run all tasks and return aggregate report."""
        task_list = tasks if tasks is not None else ALL_TASKS
        results: list[TaskResult] = [self.run_task(task) for task in task_list]
        return EvaluationReport(
            results=results,
            agent_url=self.agent_url,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def run_by_type(self, task_type: TaskType) -> EvaluationReport:
        """Run all tasks of a specific type."""
        return self.run_all(get_tasks_by_type(task_type))

    def run_by_tier(self, tier: Tier) -> EvaluationReport:
        """Run all tasks of a specific tier."""
        return self.run_all(get_tasks_by_tier(tier))


# ----------------------------------------------------------------------
# Report formatting
# ----------------------------------------------------------------------


def format_report(report: EvaluationReport) -> str:
    """Format an EvaluationReport as a human-readable summary string."""
    lines: list[str] = []
    lines.append("=== Tripletex Agent Evaluation ===")
    lines.append(f"Agent URL: {report.agent_url}")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(
        f"Tasks: {report.total_tasks} total, "
        f"{report.passed_tasks} passed, "
        f"{report.partial_tasks} partial, "
        f"{report.failed_tasks} failed"
    )
    lines.append("")

    # Per task type
    type_scores = report.per_type_scores
    if type_scores:
        lines.append("Per task type:")
        for task_type, (score, max_score) in sorted(type_scores.items()):
            pct = (score / max_score * 100) if max_score > 0 else 0.0
            lines.append(
                f"  {task_type:30s}  {score:.1f}/{max_score:.1f}  ({pct:.0f}%)"
            )
        lines.append("")

    # Per language
    lang_scores = report.per_language_scores
    if lang_scores:
        lines.append("Per language:")
        for lang, (score, max_score) in sorted(lang_scores.items()):
            pct = (score / max_score * 100) if max_score > 0 else 0.0
            lines.append(f"  {lang:5s}  {score:.1f}/{max_score:.1f}  ({pct:.0f}%)")
        lines.append("")

    # Field accuracy
    total_fields = 0
    correct_fields = 0
    for result in report.results:
        for field_result in result.fields:
            total_fields += 1
            if field_result.correct:
                correct_fields += 1

    if total_fields > 0:
        field_pct = correct_fields / total_fields * 100
        lines.append(
            f"Field accuracy: {correct_fields}/{total_fields} fields correct "
            f"({field_pct:.1f}%)"
        )
    else:
        lines.append("Field accuracy: no fields evaluated")

    # Estimated score
    lines.append(
        f"Estimated score: {report.total_score:.1f}/{report.total_max_score:.1f}"
    )
    lines.append("")

    # Failures
    failures: list[str] = []
    for result in report.results:
        if result.score == 0:
            if not result.format_ok:
                failures.append(
                    f"  {result.task_name} ({result.task_type}, {result.language}): "
                    f"agent returned invalid response"
                )
            else:
                missing = [f.field_name for f in result.fields if not f.correct]
                detail = ", ".join(missing) if missing else "entity not found"
                failures.append(
                    f"  {result.task_name} "
                    f"({result.task_type}, {result.language}): "
                    f"missing fields: {detail}"
                )

    if failures:
        lines.append("Failures:")
        lines.extend(failures)
    else:
        lines.append("No failures.")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Report persistence
# ----------------------------------------------------------------------


def save_report(report: EvaluationReport, path: str) -> None:
    """Save raw evaluation results to a JSON file."""
    dumped: dict[str, object] = report.model_dump()
    Path(path).write_text(
        json.dumps(dumped, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
