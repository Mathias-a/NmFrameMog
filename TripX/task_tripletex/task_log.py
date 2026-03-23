"""In-memory task logger for the solve endpoint.

Stores a rolling trace of the most recent task execution so that a GET /logs
endpoint can return it without touching disk or blocking the hot path.

Thread-safety: all mutations go through a single `threading.Lock`.
The lock is only held for fast list-append / list-copy operations so it will
never contend with the agent's async I/O.
"""

from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Literal, cast
from uuid import uuid4


_REDACTED = "[redacted]"
_SENSITIVE_KEY_NAMES = frozenset(
    {
        "session_token",
        "token",
        "authorization",
        "content_base64",
        "decoded_content",
        "file_bytes",
        "file_body",
        "raw_file_contents",
        "raw_file_content",
        "file_content",
    }
)


def _sanitize_for_log(value: object) -> object:
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            if key.lower() in _SENSITIVE_KEY_NAMES:
                sanitized[key] = _REDACTED
            else:
                sanitized[key] = _sanitize_for_log(item)
        return sanitized

    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]

    if isinstance(value, tuple):
        return [_sanitize_for_log(item) for item in value]

    return value


@dataclass
class LogEntry:
    """One event in a task trace."""

    ts: float
    event: str
    detail: dict[str, object] = field(default_factory=dict)


@dataclass
class TaskTrace:
    """Full trace of a single /solve invocation."""

    request_id: str = ""
    started_at: float = 0.0
    finished_at: float | None = None
    prompt: str = ""
    request_context: dict[str, object] = field(default_factory=dict)
    status: Literal["running", "completed", "error", "incomplete"] = "running"
    final_reason: str | None = None
    error: str | None = None
    entries: list[LogEntry] = field(default_factory=list)


class TaskLogger:
    _MAX_TRACE_HISTORY = 50

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._traces: dict[str, TaskTrace] = {}
        self._latest_trace_id: str | None = None
        self._current_trace_id: contextvars.ContextVar[str | None] = (
            contextvars.ContextVar("task_logger_current_trace_id", default=None)
        )

    def _resolve_trace_id(self, request_id: str | None = None) -> str | None:
        if request_id is not None:
            return request_id

        bound_request_id = self._current_trace_id.get()
        if bound_request_id is not None:
            return bound_request_id

        return self._latest_trace_id

    def _prune_locked(self) -> None:
        while len(self._traces) > self._MAX_TRACE_HISTORY:
            oldest_request_id = next(iter(self._traces))
            if oldest_request_id == self._latest_trace_id and len(self._traces) > 1:
                oldest_request_id = next(
                    iter(tuple(self._traces.keys())[1:]), oldest_request_id
                )
            self._traces.pop(oldest_request_id, None)

    # -- writing (called from agent / service) --------------------------------

    def start_task(
        self,
        prompt: str,
        *,
        request_context: dict[str, object] | None = None,
    ) -> str:
        request_id = uuid4().hex
        sanitized_request_context = cast(
            dict[str, object], _sanitize_for_log(request_context or {})
        )
        sanitized_request_context["request_id"] = request_id
        with self._lock:
            self._traces[request_id] = TaskTrace(
                request_id=request_id,
                started_at=time.time(),
                prompt=prompt,
                request_context=sanitized_request_context,
            )
            self._latest_trace_id = request_id
            self._prune_locked()
        self._current_trace_id.set(request_id)
        return request_id

    def log(
        self,
        event: str,
        detail: dict[str, object] | None = None,
        *,
        request_id: str | None = None,
    ) -> None:
        with self._lock:
            resolved_request_id = self._resolve_trace_id(request_id)
            if resolved_request_id is None:
                return
            trace = self._traces.get(resolved_request_id)
            if trace is None:
                return
            trace.entries.append(
                LogEntry(
                    ts=time.time(),
                    event=event,
                    detail=cast(dict[str, object], _sanitize_for_log(detail or {})),
                )
            )

    def finish_task(
        self,
        *,
        error: str | None = None,
        status: Literal["completed", "error", "incomplete"] | None = None,
        final_reason: str | None = None,
        request_id: str | None = None,
    ) -> None:
        with self._lock:
            resolved_request_id = self._resolve_trace_id(request_id)
            if resolved_request_id is None:
                return
            trace = self._traces.get(resolved_request_id)
            if trace is None:
                return
            trace.finished_at = time.time()
            trace.final_reason = final_reason
            if error is not None:
                trace.status = "error"
                trace.error = error
            elif status is not None:
                trace.status = status
            else:
                trace.status = "completed"
            self._latest_trace_id = resolved_request_id

    # -- reading (called from GET /logs) ---------------------------------------

    def snapshot(self, request_id: str | None = None) -> dict[str, object] | None:
        """Return a JSON-safe copy of the current trace, or None."""
        with self._lock:
            resolved_request_id = self._resolve_trace_id(request_id)
            if resolved_request_id is None:
                return None
            trace = self._traces.get(resolved_request_id)
            if trace is None:
                return None
            elapsed = (
                (trace.finished_at - trace.started_at)
                if trace.finished_at
                else (time.time() - trace.started_at)
            )
            return {
                "request_id": trace.request_id,
                "started_at": trace.started_at,
                "finished_at": trace.finished_at,
                "elapsed_seconds": round(elapsed, 2),
                "prompt": trace.prompt,
                "request_context": trace.request_context,
                "status": trace.status,
                "final_reason": trace.final_reason,
                "error": trace.error,
                "entries": [asdict(e) for e in trace.entries],
                "entry_count": len(trace.entries),
            }


# Module-level singleton so both service.py and agent.py share the same instance.
task_logger = TaskLogger()
