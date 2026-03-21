from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Protocol, cast

from src.ng_data.cloud.config import load_cloud_config
from src.ng_data.cloud.print_paths import main as print_paths_main
from src.ng_data.cloud.validate_config import main as validate_config_main
from src.ng_data.cloud.validation import validate_cloud_config


class CaptureResult(Protocol):
    out: str


class CapsysLike(Protocol):
    def readouterr(self) -> CaptureResult: ...


def test_main_gcp_layout_is_deterministic(capsys: CapsysLike) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs/cloud/main.json"
    bootstrap_script = repo_root / "scripts/gcp/bootstrap_vm.sh"
    sync_script = repo_root / "scripts/gcp/sync_artifacts.sh"
    helper_readme = repo_root / "scripts/gcp/README.md"

    config = load_cloud_config(config_path)
    validate_cloud_config(
        config,
        expected_project="ainm26osl-707",
        expected_region="europe-west1",
    )

    assert bootstrap_script.is_file()
    assert sync_script.is_file()
    assert helper_readme.is_file()

    validate_exit_code = validate_config_main(
        [
            "--config",
            str(config_path),
            "--project",
            "ainm26osl-707",
            "--region",
            "europe-west1",
            "--dry-run",
        ]
    )
    validate_output = capsys.readouterr().out
    validate_payload = cast(dict[str, Any], json.loads(validate_output))

    assert validate_exit_code == 0
    assert validate_payload == {
        "bucket_root": "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/",
        "config": str(config_path),
        "dry_run": True,
        "project_id": "ainm26osl-707",
        "region": "europe-west1",
        "status": "ok",
        "vertex_ai_mode": "optional_escalation_only",
        "vm_name": "ng-trainer-gpu-01",
    }

    print_exit_code = print_paths_main(["--config", str(config_path)])
    print_output = capsys.readouterr().out
    rendered_paths = cast(dict[str, Any], json.loads(print_output))

    assert print_exit_code == 0
    assert rendered_paths["workflow"] == {
        "canonical_storage": "gcs",
        "primary_execution": "compute_engine",
        "vertex_ai_mode": "optional_escalation_only",
    }
    assert rendered_paths["compute_engine"] == {
        "bootstrap_script": "scripts/gcp/bootstrap_vm.sh",
        "instance_resource": (
            "projects/ainm26osl-707/zones/europe-west1-b/instances/ng-trainer-gpu-01"
        ),
        "local_artifact_dir": "/home/ng/workspace/NG/artifacts",
        "workspace_dir": "/home/ng/workspace/NG",
        "zone": "europe-west1-b",
    }
    assert rendered_paths["artifacts"] == {
        "checkpoints": (
            "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/"
            "artifacts/checkpoints/"
        ),
        "data": "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/",
        "eval_outputs": (
            "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/artifacts/eval/"
        ),
        "release_bundles": (
            "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/"
            "artifacts/releases/"
        ),
        "run_manifests": (
            "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/"
            "artifacts/manifests/"
        ),
        "runs": "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/runs/",
    }


def test_bootstrap_vm_dry_run_renders_gpu_ready_commands() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts/gcp/bootstrap_vm.sh"
    config_path = repo_root / "configs/cloud/main.json"

    result = subprocess.run(
        ["bash", str(script_path), str(config_path)],
        cwd=repo_root,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--image-project=deeplearning-platform-release" in result.stdout
    assert "--image-family=common-cu128-ubuntu-2204-nvidia-570" in result.stdout
    assert "--metadata=install-nvidia-driver=True" in result.stdout
    assert (
        "sudo apt-get install -y git curl python3-pip build-essential pkg-config"
        in result.stdout
    )
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in result.stdout
    assert 'export PATH="$HOME/.local/bin:$PATH"' in result.stdout
    assert "uv --version" in result.stdout
    assert "python3 --version" in result.stdout
    assert "uv python install 3.11" in result.stdout
    assert 'REMOTE_HOME="$HOME"' in result.stdout
    assert 'REMOTE_WORKSPACE_DIR="${REMOTE_HOME}/workspace/NG"' in result.stdout
    assert (
        'REMOTE_LOCAL_ARTIFACT_DIR="${REMOTE_HOME}/workspace/NG/artifacts"'
        in result.stdout
    )
    assert (
        'export UV_PROJECT_ENVIRONMENT="${REMOTE_WORKSPACE_DIR}/.venv"' in result.stdout
    )
    assert "for attempt in 1 2 3 4 5 6 7 8 9 10; do" in result.stdout
    assert "if nvidia-smi; then" in result.stdout
    assert "sleep 30" in result.stdout
    assert "nvidia-smi did not become ready after %s attempts" in result.stdout
    assert (
        'mkdir -p "$REMOTE_WORKSPACE_DIR" "$REMOTE_LOCAL_ARTIFACT_DIR"' in result.stdout
    )
    assert "Remote workspace dir: %s" in result.stdout
    assert "Remote artifact dir: %s" in result.stdout
    assert "if [ -f pyproject.toml ]; then" in result.stdout
    assert "uv sync --python 3.11" in result.stdout
    assert "uv run --python 3.11 python -c" in result.stdout
    assert (
        "Skipping repo environment sync because %s/pyproject.toml is not present yet"
        in result.stdout
    )
    assert "import torch; assert torch.cuda.is_available();" in result.stdout
    assert "print(torch.cuda.get_device_name(0))" in result.stdout
