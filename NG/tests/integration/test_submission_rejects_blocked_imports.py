from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from src.ng_data.submission import (
    SubmissionBuildError,
    SubmissionSecurityError,
    scan_python_source,
    validate_bundle_sources,
)

ASSERTIONS = TestCase()


def test_security_scan_rejects_blocked_import(tmp_path: Path) -> None:
    source_path = tmp_path / "run.py"
    source_path.write_text(
        "from pathlib import Path\nimport subprocess\n\nPath('x')\n",
        encoding="utf-8",
    )

    with ASSERTIONS.assertRaisesRegex(
        SubmissionSecurityError, "Blocked import 'subprocess'"
    ):
        scan_python_source(source_path)


def test_bundle_validation_fails_before_publication_for_blocked_import(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run.py"
    run_path.write_text("import yaml\n", encoding="utf-8")

    with ASSERTIONS.assertRaisesRegex(SubmissionSecurityError, "Blocked import 'yaml'"):
        validate_bundle_sources([run_path], tmp_path)


def test_bundle_validation_requires_zip_root_run_py(tmp_path: Path) -> None:
    helper_path = tmp_path / "helper.py"
    helper_path.write_text("from pathlib import Path\n", encoding="utf-8")

    with ASSERTIONS.assertRaisesRegex(SubmissionBuildError, "run.py at zip root"):
        validate_bundle_sources([helper_path], tmp_path)
