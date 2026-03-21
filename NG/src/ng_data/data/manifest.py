from __future__ import annotations

import hashlib
import json
import posixpath
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


class DataManifestValidationError(ValueError):
    pass


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def directory_snapshot(path: Path) -> dict[str, object]:
    if not path.exists():
        raise DataManifestValidationError(f"Expected directory does not exist: {path}")
    if not path.is_dir():
        raise DataManifestValidationError(f"Expected a directory path: {path}")

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = file_path.relative_to(path).as_posix()
        file_hash = sha256_file(file_path)
        size_bytes = file_path.stat().st_size
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size_bytes).encode("utf-8"))
        digest.update(b"\n")
        file_count += 1
        total_bytes += size_bytes

    return {
        "file_count": file_count,
        "sha256": digest.hexdigest(),
        "total_bytes": total_bytes,
    }


def file_snapshot(path: Path) -> dict[str, object]:
    if not path.exists():
        raise DataManifestValidationError(f"Expected file does not exist: {path}")
    if not path.is_file():
        raise DataManifestValidationError(f"Expected a file path: {path}")
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_output_entry(path: Path, processed_root: Path) -> dict[str, object]:
    relative_path = path.relative_to(processed_root).as_posix()
    if path.is_dir():
        snapshot = directory_snapshot(path)
        return {
            "kind": "directory",
            "path": relative_path,
            **snapshot,
        }

    snapshot = file_snapshot(path)
    return {
        "kind": "file",
        "path": relative_path,
        **snapshot,
    }


def build_dataset_manifest(
    *,
    processed_root: Path,
    raw_root: Path,
    raw_archives: dict[str, Path],
    extracted_roots: dict[str, Path],
    output_paths: dict[str, Path],
    counts: dict[str, int],
) -> JsonDict:
    raw_root_relative = posixpath.relpath(str(raw_root), str(processed_root))
    raw_archive_entries: dict[str, object] = {}
    for name, archive_path in sorted(raw_archives.items()):
        raw_archive_entries[name] = {
            "extract_path": extracted_roots[name].relative_to(raw_root).as_posix(),
            "path": archive_path.relative_to(raw_root).as_posix(),
            **file_snapshot(archive_path),
        }

    processed_entries: dict[str, object] = {}
    for name, output_path in sorted(output_paths.items()):
        processed_entries[name] = build_output_entry(output_path, processed_root)

    return {
        "counts": counts,
        "processed_outputs": processed_entries,
        "raw_archives": raw_archive_entries,
        "raw_root_relative": raw_root_relative,
        "schema_version": 1,
    }


def load_manifest(path: str | Path) -> JsonDict:
    manifest_path = Path(path)
    try:
        data = cast(JsonDict, json.loads(manifest_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise DataManifestValidationError(
            f"Manifest is not valid JSON: {manifest_path}"
        ) from error
    if not isinstance(data, dict):
        raise DataManifestValidationError("Manifest root must be a JSON object.")
    return data


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise DataManifestValidationError(
            f"Expected manifest field '{key}' to be an object."
        )
    return cast(JsonDict, value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise DataManifestValidationError(
            f"Expected manifest field '{key}' to be a non-empty string."
        )
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataManifestValidationError(
            f"Expected manifest field '{key}' to be an integer."
        )
    return value


def _audit_entry(root: Path, name: str, entry: JsonDict) -> None:
    kind = _require_string(entry, "kind")
    relative_path = _require_string(entry, "path")
    target_path = root / relative_path
    if kind == "file":
        expected_sha = _require_string(entry, "sha256")
        expected_size = _require_int(entry, "size_bytes")
        snapshot = file_snapshot(target_path)
        if (
            snapshot["sha256"] != expected_sha
            or snapshot["size_bytes"] != expected_size
        ):
            raise DataManifestValidationError(
                f"Processed file snapshot mismatch for '{name}': {relative_path}"
            )
        return

    if kind == "directory":
        expected_sha = _require_string(entry, "sha256")
        expected_count = _require_int(entry, "file_count")
        expected_bytes = _require_int(entry, "total_bytes")
        snapshot = directory_snapshot(target_path)
        if (
            snapshot["sha256"] != expected_sha
            or snapshot["file_count"] != expected_count
            or snapshot["total_bytes"] != expected_bytes
        ):
            raise DataManifestValidationError(
                f"Processed directory snapshot mismatch for '{name}': {relative_path}"
            )
        return

    raise DataManifestValidationError(
        f"Processed output '{name}' has unsupported kind '{kind}'."
    )


def audit_dataset_manifest(manifest_path: str | Path) -> JsonDict:
    path = Path(manifest_path)
    manifest = load_manifest(path)
    if _require_int(manifest, "schema_version") != 1:
        raise DataManifestValidationError("Unsupported manifest schema version.")

    counts = _require_mapping(manifest, "counts")
    for name in (
        "annotation_count",
        "category_count",
        "image_count",
        "reference_product_count",
    ):
        _require_int(counts, name)

    processed_root = path.parents[1]
    raw_root_relative = _require_string(manifest, "raw_root_relative")
    raw_root = (processed_root / raw_root_relative).resolve()

    processed_outputs = _require_mapping(manifest, "processed_outputs")
    for name, value in sorted(processed_outputs.items()):
        if not isinstance(value, dict):
            raise DataManifestValidationError(
                f"Processed output '{name}' must be a JSON object."
            )
        _audit_entry(processed_root, name, cast(JsonDict, value))

    raw_archives = _require_mapping(manifest, "raw_archives")
    for name, value in sorted(raw_archives.items()):
        if not isinstance(value, dict):
            raise DataManifestValidationError(
                f"Raw archive '{name}' must be a JSON object."
            )
        entry = cast(JsonDict, value)
        archive_path = raw_root / _require_string(entry, "path")
        expected_sha = _require_string(entry, "sha256")
        expected_size = _require_int(entry, "size_bytes")
        snapshot = file_snapshot(archive_path)
        if (
            snapshot["sha256"] != expected_sha
            or snapshot["size_bytes"] != expected_size
        ):
            raise DataManifestValidationError(
                f"Raw archive snapshot mismatch for '{name}': {archive_path}"
            )
        extract_path = raw_root / _require_string(entry, "extract_path")
        if not extract_path.exists() or not extract_path.is_dir():
            raise DataManifestValidationError(
                f"Missing extracted raw payload for '{name}': {extract_path}"
            )

    return {
        "counts": counts,
        "manifest": str(path),
        "status": "ok",
    }
