from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, cast

from src.ng_data.data.ingest import main as ingest_main

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures/ingest_official"


def _write_zip_from_tree(source_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for source_path in sorted(
            path for path in source_root.rglob("*") if path.is_file()
        ):
            archive.write(
                source_path, arcname=source_path.relative_to(source_root).as_posix()
            )


def _create_invalid_coco_archive(raw_root: Path) -> None:
    payload = cast(
        dict[str, Any],
        json.loads(
            (FIXTURE_ROOT / "coco_source/annotations.json").read_text(encoding="utf-8")
        ),
    )
    broken_annotation = cast(dict[str, Any], payload["annotations"][0])
    broken_annotation.pop("corrected")

    archive_path = raw_root / "NM_NGD_coco_dataset.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("annotations.json", json.dumps(payload, indent=2))
        for source_path in sorted(
            path
            for path in (FIXTURE_ROOT / "coco_source/shelves").rglob("*")
            if path.is_file()
        ):
            archive.write(
                source_path,
                arcname=(
                    Path("shelves")
                    / source_path.relative_to(FIXTURE_ROOT / "coco_source/shelves")
                ).as_posix(),
            )


def _create_valid_reference_archive(raw_root: Path) -> None:
    _write_zip_from_tree(
        FIXTURE_ROOT / "reference_source",
        raw_root / "NM_NGD_product_images.zip",
    )


def test_ingest_rejects_corrupt_archives_without_manifest(tmp_path: Path) -> None:
    raw_root = tmp_path / "data/raw"
    processed_root = tmp_path / "data/processed"
    _create_invalid_coco_archive(raw_root)
    _create_valid_reference_archive(raw_root)

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs/data/main.json"

    try:
        ingest_main(
            [
                "--config",
                str(config_path),
                "--raw",
                str(raw_root),
                "--processed",
                str(processed_root),
            ]
        )
    except SystemExit as error:
        assert str(error) == "Expected 'corrected' to be a boolean."
    else:
        raise AssertionError("Expected corrupt COCO archive to raise SystemExit")

    assert not (processed_root / "manifests/dataset_manifest.json").exists()
