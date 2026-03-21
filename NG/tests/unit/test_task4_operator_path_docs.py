from __future__ import annotations

from pathlib import Path


def _read_repo_file(relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / relative_path).read_text(encoding="utf-8")


AUDIT_COMMAND = (
    'uv run python -c "from src.ng_data.data.manifest import '
    "audit_dataset_manifest; import json; print(json.dumps("
    "audit_dataset_manifest('data/processed/manifests/dataset_manifest.json'), "
    'indent=2, sort_keys=True))"'
)


def test_gcp_readme_documents_real_archive_operator_path() -> None:
    readme = _read_repo_file("scripts/gcp/README.md")

    assert (
        "The checked-in `data/processed/` tree is a tiny deterministic fixture"
        in readme
    )
    assert (
        "uv run python -m src.ng_data.data.ingest --config configs/data/main.json --raw data/raw --processed data/processed"
        in readme
    )
    assert (
        "uv run python -m src.ng_data.eval.make_splits --config configs/data/splits.json --manifest data/processed/manifests/dataset_manifest.json"
        in readme
    )
    assert AUDIT_COMMAND in readme
    assert (
        "gcloud storage cp data/raw/NM_NGD_coco_dataset.zip gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/raw/"
        in readme
    )
    assert (
        "gcloud storage cp data/raw/NM_NGD_product_images.zip gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/raw/"
        in readme
    )
    assert (
        "gcloud storage rsync --recursive data/processed gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/processed"
        in readme
    )
    assert (
        "gcloud storage rsync --recursive gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data /home/ng/workspace/NG/data"
        in readme
    )


def test_submission_checklist_repeats_real_data_staging_commands() -> None:
    checklist = _read_repo_file("docs/submission-checklist.md")

    assert "fixture-sized local test outputs" in checklist
    assert (
        "uv run python -m src.ng_data.data.ingest --config configs/data/main.json --raw data/raw --processed data/processed"
        in checklist
    )
    assert (
        "uv run python -m src.ng_data.eval.make_splits --config configs/data/splits.json --manifest data/processed/manifests/dataset_manifest.json"
        in checklist
    )
    assert AUDIT_COMMAND in checklist
    assert (
        "gcloud storage cp data/raw/NM_NGD_coco_dataset.zip gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/raw/"
        in checklist
    )
    assert (
        "gcloud storage cp data/raw/NM_NGD_product_images.zip gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/raw/"
        in checklist
    )
    assert (
        "gcloud storage rsync --recursive data/processed gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/processed"
        in checklist
    )
    assert (
        "gcloud storage rsync --recursive gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data /home/ng/workspace/NG/data"
        in checklist
    )
