"""CLI script for running the full Tripletex local evaluation."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from ai_accounting_agent.evaluator import LocalEvaluator, format_report, save_report
from ai_accounting_agent.task_library import (
    ALL_TASKS,
    TaskType,
    Tier,
)


@dataclass(frozen=True)
class _CLIArgs:
    """Typed container for parsed CLI arguments."""

    agent_url: str
    base_url: str
    token: str | None
    task_type: str | None
    tier: str | None
    output: str | None
    verbose: bool


def _parse_args() -> _CLIArgs:
    """Build argparse parser and return typed arguments."""
    parser = argparse.ArgumentParser(
        description="Tripletex AI Accounting Agent Local Evaluator",
    )
    parser.add_argument(
        "--agent-url",
        default="http://localhost:8080",
        help="Agent endpoint URL",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Tripletex session token (or set TRIPLETEX_SESSION_TOKEN)",
    )
    parser.add_argument(
        "--base-url",
        default="https://tx-proxy.ainm.no/v2",
        help="Tripletex API base URL",
    )
    parser.add_argument(
        "--task-type",
        default=None,
        help="Filter by task type (e.g. create_employee)",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Filter by tier (1, 2, or 3)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Save JSON results to file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed per-field results",
    )
    ns = parser.parse_args()
    return _CLIArgs(
        agent_url=str(ns.agent_url),  # type: ignore[any]
        base_url=str(ns.base_url),  # type: ignore[any]
        token=str(ns.token) if ns.token is not None else None,  # type: ignore[any]
        task_type=str(ns.task_type) if ns.task_type is not None else None,  # type: ignore[any]
        tier=str(ns.tier) if ns.tier is not None else None,  # type: ignore[any]
        output=str(ns.output) if ns.output is not None else None,  # type: ignore[any]
        verbose=bool(ns.verbose),  # type: ignore[any]
    )


def main() -> None:
    """Parse arguments and run the local evaluation pipeline."""
    cli = _parse_args()

    token = cli.token or os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    if not token:
        print(
            "Error: No session token provided. "
            "Use --token or set TRIPLETEX_SESSION_TOKEN",
        )
        sys.exit(1)

    # Filter tasks based on CLI arguments (both filters intersect)
    tasks = list(ALL_TASKS)
    if cli.task_type is not None:
        task_type = TaskType(cli.task_type)
        tasks = [t for t in tasks if t.task_type == task_type]
    if cli.tier is not None:
        tier = Tier(int(cli.tier))
        tasks = [t for t in tasks if t.tier == tier]

    # Run evaluation
    with LocalEvaluator(
        agent_url=cli.agent_url,
        base_url=cli.base_url,
        session_token=token,
    ) as evaluator:
        report = evaluator.run_all(tasks)

    # Print formatted report
    print(format_report(report))

    # Verbose mode: print per-field details
    if cli.verbose:
        for result in report.results:
            if result.fields:
                print(
                    f"\n  {result.task_name} ({result.task_type}, {result.language}):",
                )
                for field in result.fields:
                    status = "PASS" if field.correct else "FAIL"
                    print(
                        f"    [{status}] {field.field_name}: "
                        f"expected={field.expected_value}, "
                        f"actual={field.actual_value}",
                    )

    # Save results to file if requested
    if cli.output is not None:
        save_report(report, cli.output)
        print(f"\nResults saved to {cli.output}")

    # Exit with appropriate code
    sys.exit(0 if report.passed_tasks == report.total_tasks else 1)


if __name__ == "__main__":
    main()
