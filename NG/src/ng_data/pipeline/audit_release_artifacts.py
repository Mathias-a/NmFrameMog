from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from src.ng_data.data.manifest import (
    DataManifestValidationError,
    file_snapshot,
    load_manifest,
)
from src.ng_data.pipeline.final_train import (
    SCHEMA_VERSION,
    FinalTrainError,
    _resolve_repo_root,
    _select_repo_root,
    build_release_manifest,
)

JsonDict = dict[str, Any]

_CANNOT_VALIDATE = [
    (
        "release-grade final weights cannot be validated because final retraining "
        "was blocked and never ran"
    ),
    (
        "release-grade final-training provenance cannot be validated because "
        "final retraining was blocked and never ran"
    ),
]


class AuditReleaseArtifactsArgs(argparse.Namespace):
    manifest: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the blocked final hold-manifest for truthfulness and snapshot "
            "consistency against the current repo state."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the release manifest JSON file to audit.",
    )
    return parser


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise DataManifestValidationError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise DataManifestValidationError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataManifestValidationError(f"Expected '{key}' to be an integer.")
    return value


def _require_bool(data: JsonDict, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise DataManifestValidationError(f"Expected '{key}' to be a boolean.")
    return value


def _resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _validate_snapshot(*, label: str, path: Path, expected: JsonDict) -> JsonDict:
    expected_sha256 = _require_string(expected, "sha256")
    expected_size = _require_int(expected, "size_bytes")
    observed = file_snapshot(path)
    if observed["sha256"] != expected_sha256 or observed["size_bytes"] != expected_size:
        raise DataManifestValidationError(
            f"Artifact snapshot mismatch for '{label}': {path.as_posix()}"
        )
    return cast(JsonDict, observed)


def audit_release_artifacts(manifest_path: str | Path) -> JsonDict:
    path = Path(manifest_path)
    manifest = load_manifest(path)
    if _require_int(manifest, "schema_version") != SCHEMA_VERSION:
        raise DataManifestValidationError(
            "Unsupported release manifest schema version."
        )

    status = _require_string(manifest, "status")
    if status != "blocked_hold_manifest":
        raise DataManifestValidationError(
            "Release manifest status must be 'blocked_hold_manifest' "
            "for the current repo state."
        )

    final_training = _require_mapping(manifest, "final_training")
    if _require_bool(final_training, "attempted"):
        raise DataManifestValidationError(
            "Blocked hold-manifest must record final_training.attempted=false."
        )
    if not _require_bool(final_training, "blocked"):
        raise DataManifestValidationError(
            "Blocked hold-manifest must record final_training.blocked=true."
        )
    if not _require_bool(final_training, "hold_manifest_only"):
        raise DataManifestValidationError(
            "Blocked hold-manifest must record final_training.hold_manifest_only=true."
        )

    repo_root = _resolve_repo_root(path)
    frozen_config = _require_mapping(manifest, "frozen_config")
    artifact_references = _require_mapping(manifest, "artifact_references")
    provenance = _require_mapping(manifest, "provenance")
    referenced_files = _require_mapping(provenance, "referenced_files")

    repo_root = _select_repo_root(
        path,
        [
            cast(JsonDict, artifact_references[name])
            for name in sorted(artifact_references)
            if isinstance(artifact_references[name], dict)
        ],
    )
    release_manifest = _require_mapping(manifest, "release_manifest")
    release_manifest_path = _resolve_repo_path(
        repo_root, _require_string(release_manifest, "path")
    )
    frozen_config_path = _resolve_repo_path(
        repo_root, _require_string(frozen_config, "path")
    )
    expected_frozen_config = _validate_snapshot(
        label="frozen_config",
        path=frozen_config_path,
        expected=_require_mapping(frozen_config, "snapshot"),
    )

    if set(referenced_files) != set(artifact_references):
        raise DataManifestValidationError(
            "provenance.referenced_files keys must match artifact_references keys."
        )

    checked_files: list[JsonDict] = [
        {
            "label": "frozen_config",
            "path": frozen_config_path.as_posix(),
            "sha256": cast(str, expected_frozen_config["sha256"]),
            "size_bytes": cast(int, expected_frozen_config["size_bytes"]),
        }
    ]
    for name in sorted(artifact_references):
        entry = cast(JsonDict, artifact_references[name])
        if not isinstance(entry, dict):
            raise DataManifestValidationError(
                f"artifact_references['{name}'] must be a JSON object."
            )
        entry_path = _resolve_repo_path(repo_root, _require_string(entry, "path"))
        observed_snapshot = _validate_snapshot(
            label=name,
            path=entry_path,
            expected=_require_mapping(entry, "snapshot"),
        )
        if referenced_files[name] != entry["snapshot"]:
            raise DataManifestValidationError(
                f"provenance.referenced_files['{name}'] does not match "
                f"artifact_references['{name}'].snapshot."
            )
        checked_files.append(
            {
                "label": name,
                "path": entry_path.as_posix(),
                "sha256": cast(str, observed_snapshot["sha256"]),
                "size_bytes": cast(int, observed_snapshot["size_bytes"]),
            }
        )

    try:
        expected_manifest = build_release_manifest(
            config_path=frozen_config_path,
            out_path=release_manifest_path,
        )
    except FinalTrainError as error:
        raise DataManifestValidationError(str(error)) from error

    if manifest["status"] != expected_manifest["status"]:
        raise DataManifestValidationError(
            "Release manifest status does not match the current frozen "
            "release contract."
        )
    expected_release_manifest = _require_mapping(expected_manifest, "release_manifest")
    if _require_string(release_manifest, "kind") != _require_string(
        expected_release_manifest, "kind"
    ):
        raise DataManifestValidationError(
            "Release manifest release_manifest block does not match the current "
            "frozen release contract."
        )
    expected_release_manifest_path = _resolve_repo_path(
        repo_root, _require_string(expected_release_manifest, "path")
    )
    if release_manifest_path.resolve() != expected_release_manifest_path.resolve():
        raise DataManifestValidationError(
            "Release manifest release_manifest path does not resolve to the current "
            "frozen release contract."
        )
    if final_training != expected_manifest["final_training"]:
        raise DataManifestValidationError(
            "Release manifest final_training block does not match the "
            "current frozen release contract."
        )
    expected_frozen_config_block = _require_mapping(expected_manifest, "frozen_config")
    expected_frozen_config_path = _resolve_repo_path(
        repo_root, _require_string(expected_frozen_config_block, "path")
    )
    if frozen_config_path.resolve() != expected_frozen_config_path.resolve():
        raise DataManifestValidationError(
            "Release manifest frozen_config path does not resolve to the "
            "current frozen config file."
        )
    if _require_mapping(frozen_config, "snapshot") != _require_mapping(
        expected_frozen_config_block, "snapshot"
    ):
        raise DataManifestValidationError(
            "Release manifest frozen_config block does not match the current "
            "frozen release contract."
        )

    expected_artifact_references = _require_mapping(
        expected_manifest, "artifact_references"
    )

    if set(artifact_references) != set(expected_artifact_references):
        raise DataManifestValidationError(
            "Release manifest artifact_references keys do not match the "
            "frozen release contract."
        )
    for name in sorted(artifact_references):
        entry = cast(JsonDict, artifact_references[name])
        if entry != expected_artifact_references[name]:
            raise DataManifestValidationError(
                f"Release manifest artifact reference '{name}' does not "
                "match the current frozen release contract."
            )

    expected_provenance = _require_mapping(expected_manifest, "provenance")
    if _require_string(provenance, "cwd") != _require_string(
        expected_provenance, "cwd"
    ):
        raise DataManifestValidationError(
            "Release manifest provenance.cwd does not match the current frozen "
            "release contract."
        )

    return {
        "status": "ok",
        "release_state": status,
        "checked_files": checked_files,
        "cannot_validate": list(_CANNOT_VALIDATE),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(AuditReleaseArtifactsArgs, parser.parse_args(argv))
    try:
        audit_result = audit_release_artifacts(args.manifest)
    except (DataManifestValidationError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
