"""Pydantic models for the Tripletex AI Accounting Agent."""

from __future__ import annotations

import base64
from collections import defaultdict

from pydantic import BaseModel, Field


class TripletexCredentials(BaseModel):  # type: ignore[explicit-any,decorated-any]
    """Credentials for authenticating with the Tripletex API proxy."""

    base_url: str
    session_token: str

    @property
    def auth_tuple(self) -> tuple[str, str]:
        """Return HTTP basic auth tuple (username='0', password=session_token)."""
        return ("0", self.session_token)


class FileAttachment(BaseModel):  # type: ignore[explicit-any,decorated-any]
    """A base64-encoded file attachment included in a solve request."""

    filename: str
    content_base64: str

    def decode(self) -> bytes:
        """Decode the base64 content into raw bytes."""
        return base64.b64decode(self.content_base64)


class SolveRequest(BaseModel):  # type: ignore[explicit-any,decorated-any]
    """Incoming request to the /solve endpoint."""

    prompt: str
    files: list[FileAttachment] = Field(default_factory=list)
    tripletex_credentials: TripletexCredentials


class SolveResponse(BaseModel):  # type: ignore[explicit-any,decorated-any]
    """Response returned from the /solve endpoint."""

    status: str


class FieldResult(BaseModel):  # type: ignore[explicit-any,decorated-any]
    """Result of evaluating a single field within a task."""

    field_name: str
    expected_value: str
    actual_value: str | None
    correct: bool


class TaskResult(BaseModel):  # type: ignore[explicit-any,decorated-any]
    """Result of evaluating a single task."""

    task_name: str
    task_type: str
    language: str
    format_ok: bool
    fields: list[FieldResult]
    score: float
    max_score: float
    duration_seconds: float

    @property
    def passed(self) -> bool:
        """Whether the task achieved a perfect score."""
        return self.score == self.max_score

    @property
    def score_ratio(self) -> float:
        """Ratio of score to max_score, or 0.0 if max_score is zero."""
        if self.max_score > 0:
            return self.score / self.max_score
        return 0.0


class EvaluationReport(BaseModel):  # type: ignore[explicit-any,decorated-any]
    """Full evaluation report across all tasks."""

    results: list[TaskResult]
    agent_url: str
    timestamp: str

    @property
    def total_tasks(self) -> int:
        """Total number of tasks evaluated."""
        return len(self.results)

    @property
    def passed_tasks(self) -> int:
        """Number of tasks with a perfect score."""
        return sum(1 for task in self.results if task.passed)

    @property
    def partial_tasks(self) -> int:
        """Number of tasks with partial credit (0 < score < max_score)."""
        return sum(1 for task in self.results if 0 < task.score < task.max_score)

    @property
    def failed_tasks(self) -> int:
        """Number of tasks with zero score."""
        return sum(1 for task in self.results if task.score == 0)

    @property
    def total_score(self) -> float:
        """Sum of all task scores."""
        return sum(task.score for task in self.results)

    @property
    def total_max_score(self) -> float:
        """Sum of all task max scores."""
        return sum(task.max_score for task in self.results)

    @property
    def score_ratio(self) -> float:
        """Overall score ratio across all tasks."""
        if self.total_max_score > 0:
            return self.total_score / self.total_max_score
        return 0.0

    @property
    def per_language_scores(self) -> dict[str, tuple[float, float]]:
        """Aggregate (score, max_score) grouped by language."""
        scores: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for task in self.results:
            scores[task.language][0] += task.score
            scores[task.language][1] += task.max_score
        return {lang: (vals[0], vals[1]) for lang, vals in scores.items()}

    @property
    def per_type_scores(self) -> dict[str, tuple[float, float]]:
        """Aggregate (score, max_score) grouped by task type."""
        scores: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for task in self.results:
            scores[task.task_type][0] += task.score
            scores[task.task_type][1] += task.max_score
        return {ttype: (vals[0], vals[1]) for ttype, vals in scores.items()}


def parse_request(data: dict[str, object]) -> SolveRequest:
    """Parse a raw dictionary into a SolveRequest model."""
    return SolveRequest.model_validate(data)


def build_request(
    prompt: str,
    credentials: TripletexCredentials,
    files: list[FileAttachment] | None = None,
) -> dict[str, object]:
    """Build a request dictionary suitable for JSON serialization."""
    request = SolveRequest(
        prompt=prompt,
        tripletex_credentials=credentials,
        files=files if files is not None else [],
    )
    return request.model_dump()  # type: ignore[any]
