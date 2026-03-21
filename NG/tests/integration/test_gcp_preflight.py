from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_stub_bin(
    tmp_path: Path,
    *,
    active_account: str,
    enabled_services: str = "compute.googleapis.com\nstorage.googleapis.com\n",
    accelerator_types: str = "nvidia-tesla-t4\n",
) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "command.log"

    uv_script = f"""#!/usr/bin/env bash
set -euo pipefail
printf 'uv %s\n' "$*" >> {log_path!s}
case "$*" in
  'run python -m src.ng_data.cli.doctor --root .')
    printf 'doctor ok\n'
    ;;
  'run python -m src.ng_data.cloud.validate_config --config configs/cloud/main.json --project ainm26osl-707 --region europe-west1 --dry-run')
    printf '{{"status":"ok","tool":"validate_config"}}\n'
    ;;
  'run python -m src.ng_data.cloud.print_paths --config configs/cloud/main.json')
    printf '{{"status":"ok","tool":"print_paths"}}\n'
    ;;
  *)
    printf 'unexpected uv invocation: %s\n' "$*" >&2
    exit 1
    ;;
esac
"""

    gcloud_script = f"""#!/usr/bin/env bash
set -euo pipefail
printf 'gcloud %s\n' "$*" >> {log_path!s}
case "$*" in
  'config get-value project')
    printf 'ainm26osl-707\n'
    ;;
  'auth list --filter=status:ACTIVE --format=value(account)')
    printf '%s' {active_account!r}
    ;;
  'services list --enabled --filter=name:(compute.googleapis.com storage.googleapis.com) --format=value(name)')
    printf '%s' {enabled_services!r}
    ;;
  'compute accelerator-types list --filter=zone:( europe-west1-b ) AND name:nvidia-tesla-t4 --format=value(name)')
    printf '%s' {accelerator_types!r}
    ;;
  'storage ls gs://ainm26osl-707-ng-artifacts/')
    printf 'gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/\n'
    ;;
  *)
    printf 'unexpected gcloud invocation: %s\n' "$*" >&2
    exit 1
    ;;
esac
"""

    _write_executable(bin_dir / "uv", uv_script)
    _write_executable(bin_dir / "gcloud", gcloud_script)
    return bin_dir, log_path


def _run_preflight(
    tmp_path: Path,
    *,
    active_account: str,
    enabled_services: str = "compute.googleapis.com\nstorage.googleapis.com\n",
    accelerator_types: str = "nvidia-tesla-t4\n",
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts/gcp/preflight.sh"
    bin_dir, _ = _make_stub_bin(
        tmp_path,
        active_account=active_account,
        enabled_services=enabled_services,
        accelerator_types=accelerator_types,
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    return subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_preflight_runs_expected_order_with_stage_labels(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, active_account="ng@example.com")
    log_path = tmp_path / "command.log"

    assert result.returncode == 0
    assert "==> local.doctor" in result.stdout
    assert '{"stage":"gcloud.bucket_access","status":"ok"}' in result.stdout
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "uv run python -m src.ng_data.cli.doctor --root .",
        (
            "uv run python -m src.ng_data.cloud.validate_config --config "
            "configs/cloud/main.json --project ainm26osl-707 --region europe-west1 "
            "--dry-run"
        ),
        "uv run python -m src.ng_data.cloud.print_paths --config configs/cloud/main.json",
        "gcloud config get-value project",
        "gcloud auth list --filter=status:ACTIVE --format=value(account)",
        (
            "gcloud services list --enabled --filter=name:(compute.googleapis.com "
            "storage.googleapis.com) --format=value(name)"
        ),
        (
            "gcloud compute accelerator-types list --filter=zone:( europe-west1-b ) "
            "AND name:nvidia-tesla-t4 --format=value(name)"
        ),
        "gcloud storage ls gs://ainm26osl-707-ng-artifacts/",
    ]


def test_preflight_fails_early_when_no_active_gcloud_account(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, active_account="")
    log_path = tmp_path / "command.log"

    assert result.returncode != 0
    assert (
        "Preflight failed at gcloud.active_account: expected active output but received none"
        in result.stderr
    )
    assert "gcloud services list" not in log_path.read_text(encoding="utf-8")


def test_preflight_fails_when_enabled_services_output_is_empty(tmp_path: Path) -> None:
    result = _run_preflight(
        tmp_path,
        active_account="ng@example.com",
        enabled_services="",
    )
    log_path = tmp_path / "command.log"

    assert result.returncode != 0
    assert (
        "Preflight failed at gcloud.enabled_services: expected enabled service output "
        "but received none" in result.stderr
    )
    assert "gcloud compute accelerator-types list" not in log_path.read_text(
        encoding="utf-8"
    )


def test_preflight_fails_when_accelerator_output_is_empty(tmp_path: Path) -> None:
    result = _run_preflight(
        tmp_path,
        active_account="ng@example.com",
        accelerator_types="",
    )
    log_path = tmp_path / "command.log"

    assert result.returncode != 0
    assert (
        "Preflight failed at gcloud.accelerator_type: expected accelerator output "
        "but received none" in result.stderr
    )
    assert (
        "gcloud storage ls gs://ainm26osl-707-ng-artifacts/"
        not in log_path.read_text(encoding="utf-8")
    )
