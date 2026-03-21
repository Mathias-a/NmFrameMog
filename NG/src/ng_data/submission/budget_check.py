from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from . import (
    MAX_FILES,
    MAX_PYTHON_FILES,
    MAX_WEIGHT_BYTES,
    MAX_WEIGHT_FILES,
    WEIGHT_FILE_TYPES,
    run_smoke_submission,
)

MAX_ZIP_UNCOMPRESSED_BYTES = 420 * 1024 * 1024
SMOKE_RUNTIME_LIMIT_SECONDS = 240.0
MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024
SANDBOX_TIMEOUT_SECONDS = 300.0
JsonDict = dict[str, Any]
SmokeRunner = Callable[[Path, Path, Path], list[dict[str, Any]]]
Timer = Callable[[], float]


class BudgetCheckError(ValueError):
    pass


class BudgetCheckArgs(argparse.Namespace):
    zip: str
    input: str
    out: str


def _budget_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a built submission zip against local smoke-run budgets."
    )
    parser.add_argument(
        "--zip", required=True, help="Path to the built submission zip."
    )
    parser.add_argument(
        "--input", required=True, help="Directory containing local smoke input images."
    )
    parser.add_argument(
        "--out", required=True, help="Path where budget evidence JSON will be written."
    )
    return parser


def _archive_metrics(zip_path: Path) -> JsonDict:
    if not zip_path.exists():
        raise BudgetCheckError(f"Submission zip does not exist: {zip_path}")
    if not zip_path.is_file():
        raise BudgetCheckError(f"Submission zip path is not a file: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]

    weight_members = [
        info
        for info in members
        if Path(info.filename).suffix.lower() in WEIGHT_FILE_TYPES
    ]
    python_file_count = sum(
        1 for info in members if Path(info.filename).suffix.lower() == ".py"
    )
    return {
        "zip_bytes": zip_path.stat().st_size,
        "zip_uncompressed_bytes": sum(info.file_size for info in members),
        "file_count": len(members),
        "python_file_count": python_file_count,
        "weight_file_count": len(weight_members),
        "weight_bytes": sum(info.file_size for info in weight_members),
        "smoke_runtime_seconds": None,
        "prediction_count": 0,
        "memory_peak_bytes": None,
    }


def _check_entry(
    *,
    name: str,
    observed: int | float | None,
    threshold: int | float,
    enforced: bool,
    detail: str | None = None,
) -> JsonDict:
    if observed is None:
        status = "not_run" if enforced else "not_measured"
    else:
        status = "pass" if observed <= threshold else "fail"
    return {
        "comparison": "<=",
        "detail": detail,
        "enforced": enforced,
        "name": name,
        "observed": observed,
        "status": status,
        "threshold": threshold,
    }


def _build_checks(observed: JsonDict, runtime_limit_seconds: float) -> list[JsonDict]:
    return [
        _check_entry(
            name="zip_uncompressed_bytes",
            observed=cast(int, observed["zip_uncompressed_bytes"]),
            threshold=MAX_ZIP_UNCOMPRESSED_BYTES,
            enforced=True,
        ),
        _check_entry(
            name="file_count",
            observed=cast(int, observed["file_count"]),
            threshold=MAX_FILES,
            enforced=True,
        ),
        _check_entry(
            name="python_file_count",
            observed=cast(int, observed["python_file_count"]),
            threshold=MAX_PYTHON_FILES,
            enforced=True,
        ),
        _check_entry(
            name="weight_file_count",
            observed=cast(int, observed["weight_file_count"]),
            threshold=MAX_WEIGHT_FILES,
            enforced=True,
        ),
        _check_entry(
            name="weight_bytes",
            observed=cast(int, observed["weight_bytes"]),
            threshold=MAX_WEIGHT_BYTES,
            enforced=True,
        ),
        _check_entry(
            name="smoke_runtime_seconds",
            observed=cast(float | None, observed["smoke_runtime_seconds"]),
            threshold=runtime_limit_seconds,
            enforced=True,
        ),
        _check_entry(
            name="memory_peak_bytes",
            observed=cast(None, observed["memory_peak_bytes"]),
            threshold=MEMORY_LIMIT_BYTES,
            enforced=False,
            detail=(
                "Placeholder only; local budget check does not measure peak memory yet."
            ),
        ),
    ]


def _failed_checks(checks: list[JsonDict]) -> list[JsonDict]:
    return [
        check for check in checks if check["enforced"] and check["status"] == "fail"
    ]


def _summary(failed_checks: list[JsonDict], smoke_error: str | None) -> str:
    if smoke_error is not None:
        return f"Budget check failed: smoke run error: {smoke_error}"
    if not failed_checks:
        return "Budget check passed."
    reasons = ", ".join(
        (
            f"{check['name']} exceeded {check['threshold']} with observed "
            f"{check['observed']}"
        )
        for check in failed_checks
    )
    return f"Budget check failed: {reasons}."


def collect_budget_evidence(
    zip_path: Path,
    input_dir: Path,
    *,
    runtime_limit_seconds: float = SMOKE_RUNTIME_LIMIT_SECONDS,
    smoke_runner: SmokeRunner | None = None,
    timer: Timer | None = None,
) -> JsonDict:
    resolved_smoke_runner = (
        run_smoke_submission if smoke_runner is None else smoke_runner
    )
    resolved_timer = perf_counter if timer is None else timer
    observed = _archive_metrics(zip_path)
    smoke_error: str | None = None

    checks = _build_checks(observed, runtime_limit_seconds)
    if not _failed_checks(checks):
        with tempfile.TemporaryDirectory(prefix="ng-budget-check-") as temp_dir:
            output_path = Path(temp_dir) / "predictions.json"
            start = resolved_timer()
            try:
                predictions = resolved_smoke_runner(zip_path, input_dir, output_path)
            except Exception as exc:
                observed["smoke_runtime_seconds"] = max(resolved_timer() - start, 0.0)
                smoke_error = str(exc)
            else:
                observed["smoke_runtime_seconds"] = max(resolved_timer() - start, 0.0)
                observed["prediction_count"] = len(predictions)
        checks = _build_checks(observed, runtime_limit_seconds)

    failed_checks = _failed_checks(checks)
    status = "fail" if smoke_error is not None or failed_checks else "pass"
    return {
        "checks": checks,
        "failed_checks": [check["name"] for check in failed_checks],
        "input_dir": str(input_dir),
        "observed": observed,
        "smoke_error": smoke_error,
        "status": status,
        "summary": _summary(failed_checks, smoke_error),
        "thresholds": {
            "max_files": MAX_FILES,
            "max_memory_bytes": MEMORY_LIMIT_BYTES,
            "max_python_files": MAX_PYTHON_FILES,
            "max_smoke_runtime_seconds": runtime_limit_seconds,
            "max_weight_bytes": MAX_WEIGHT_BYTES,
            "max_weight_files": MAX_WEIGHT_FILES,
            "max_zip_uncompressed_bytes": MAX_ZIP_UNCOMPRESSED_BYTES,
            "memory_check": "placeholder_not_measured",
            "sandbox_timeout_seconds": SANDBOX_TIMEOUT_SECONDS,
        },
        "zip_path": str(zip_path),
    }


def write_budget_evidence(
    zip_path: Path,
    input_dir: Path,
    output_path: Path,
    *,
    runtime_limit_seconds: float = SMOKE_RUNTIME_LIMIT_SECONDS,
    smoke_runner: SmokeRunner | None = None,
    timer: Timer | None = None,
) -> JsonDict:
    evidence = collect_budget_evidence(
        zip_path,
        input_dir,
        runtime_limit_seconds=runtime_limit_seconds,
        smoke_runner=smoke_runner,
        timer=timer,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if evidence["status"] != "pass":
        raise BudgetCheckError(cast(str, evidence["summary"]))
    return evidence


def budget_check_main(argv: list[str] | None = None) -> int:
    parser = _budget_parser()
    args = cast(BudgetCheckArgs, parser.parse_args(argv))
    try:
        evidence = write_budget_evidence(
            zip_path=Path(args.zip),
            input_dir=Path(args.input),
            output_path=Path(args.out),
        )
    except BudgetCheckError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(budget_check_main())
