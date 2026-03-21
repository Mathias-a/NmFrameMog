from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_sync(mode: str) -> subprocess.CompletedProcess[str]:
    repo_root = _repo_root()
    script_path = repo_root / "scripts/gcp/sync_artifacts.sh"
    return subprocess.run(
        ["bash", str(script_path), mode, "configs/cloud/main.json"],
        cwd=repo_root,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )


def test_sync_artifacts_push_dry_run_includes_detector_eval_release_and_manifest() -> (
    None
):
    result = _run_sync("push")

    assert result.returncode == 0
    assert (
        "Dry run: gcloud storage rsync --recursive artifacts/models/detector "
        "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/runs/"
    ) in result.stdout
    assert (
        "Dry run: gcloud storage rsync --recursive artifacts/eval "
        "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/artifacts/eval/"
    ) in result.stdout
    assert (
        "Dry run: gcloud storage rsync --recursive dist "
        "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/artifacts/releases/"
    ) in result.stdout
    assert (
        "Dry run: gcloud storage cp artifacts/run_manifest.json "
        "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/artifacts/manifests/"
    ) in result.stdout


def test_sync_artifacts_pull_dry_run_mirrors_detector_eval_release_and_manifest() -> (
    None
):
    result = _run_sync("pull")

    assert result.returncode == 0
    assert (
        "Dry run: gcloud storage rsync --recursive "
        "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/runs/ "
        "artifacts/models/detector"
    ) in result.stdout
    assert (
        "Dry run: gcloud storage rsync --recursive "
        "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/artifacts/eval/ "
        "artifacts/eval"
    ) in result.stdout
    assert (
        "Dry run: gcloud storage rsync --recursive "
        "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/artifacts/releases/ "
        "dist"
    ) in result.stdout
    assert (
        "Dry run: gcloud storage cp "
        "gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/"
        "artifacts/manifests/run_manifest.json "
        "artifacts/run_manifest.json"
    ) in result.stdout
