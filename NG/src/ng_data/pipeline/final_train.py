from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from src.ng_data.data.manifest import file_snapshot, write_json

JsonDict = dict[str, Any]
SCHEMA_VERSION = 1


class FinalTrainError(ValueError):
    pass


class FinalTrainArgs(argparse.Namespace):
    config: str
    out: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write a truthful final-train release manifest or abort when frozen "
            "promotion evidence is not eligible."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the frozen final pipeline config JSON file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the release manifest JSON output.",
    )
    return parser


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FinalTrainError(f"Expected file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise FinalTrainError(f"Invalid JSON file: {path}") from error


def _load_json_object(path: Path) -> JsonDict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise FinalTrainError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise FinalTrainError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_bool(data: JsonDict, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise FinalTrainError(f"Expected '{key}' to be a boolean.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FinalTrainError(f"Expected '{key}' to be an integer.")
    return value


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise FinalTrainError(f"Expected '{key}' to be a non-empty string.")
    return value


def _validate_snapshot(
    *, artifact_name: str, path: Path, expected: JsonDict
) -> dict[str, object]:
    expected_sha256 = _require_string(expected, "sha256")
    expected_size = _require_int(expected, "size_bytes")
    observed = file_snapshot(path)
    if observed["sha256"] != expected_sha256 or observed["size_bytes"] != expected_size:
        raise FinalTrainError(
            "Frozen config drift detected for "
            f"{artifact_name}: expected sha256={expected_sha256} "
            f"size_bytes={expected_size}, "
            f"got sha256={observed['sha256']} size_bytes={observed['size_bytes']}"
        )
    return observed


def _locked_artifact_entry(
    *, name: str, config_entry: JsonDict, repo_root: Path
) -> JsonDict:
    relative_path = _require_string(config_entry, "path")
    expected_snapshot = _require_mapping(config_entry, "snapshot")
    artifact_path = repo_root / relative_path
    observed_snapshot = _validate_snapshot(
        artifact_name=name,
        path=artifact_path,
        expected=expected_snapshot,
    )
    return {
        "path": relative_path,
        "snapshot": {
            "sha256": cast(str, observed_snapshot["sha256"]),
            "size_bytes": cast(int, observed_snapshot["size_bytes"]),
        },
    }


def _resolve_repo_root(config_path: Path) -> Path:
    resolved_config = config_path.resolve()
    if len(resolved_config.parents) >= 3:
        return resolved_config.parents[2]
    return Path.cwd().resolve()


def _select_repo_root(config_path: Path, artifact_entries: list[JsonDict]) -> Path:
    candidates = [_resolve_repo_root(config_path), Path.cwd().resolve()]
    seen: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.append(candidate)
        if all(
            (candidate / _require_string(entry, "path")).exists()
            for entry in artifact_entries
        ):
            return candidate
    return seen[0]


def build_release_manifest(*, config_path: Path, out_path: Path) -> JsonDict:
    config_payload = _load_json_object(config_path)
    if _require_int(config_payload, "schema_version") != 1:
        raise FinalTrainError("Unsupported frozen config schema version.")

    artifacts = _require_mapping(config_payload, "artifacts")
    final_train = _require_mapping(config_payload, "final_train")
    hold_manifest_only = _require_bool(final_train, "hold_manifest_only")
    selected_variant_must_be_eligible = _require_bool(
        final_train, "selected_variant_must_be_eligible"
    )

    decision_entry = _require_mapping(artifacts, "decision")
    detector_entry = _require_mapping(artifacts, "detector_train_summary")
    classifier_entry = _require_mapping(artifacts, "classifier_train_summary")
    repo_root = _select_repo_root(
        config_path, [decision_entry, detector_entry, classifier_entry]
    )

    locked_decision = _locked_artifact_entry(
        name="decision",
        config_entry=decision_entry,
        repo_root=repo_root,
    )
    locked_detector = _locked_artifact_entry(
        name="detector_train_summary",
        config_entry=detector_entry,
        repo_root=repo_root,
    )
    locked_classifier = _locked_artifact_entry(
        name="classifier_train_summary",
        config_entry=classifier_entry,
        repo_root=repo_root,
    )

    decision_payload = _load_json_object(
        repo_root / _require_string(decision_entry, "path")
    )
    detector_payload = _load_json_object(
        repo_root / _require_string(detector_entry, "path")
    )
    classifier_payload = _load_json_object(
        repo_root / _require_string(classifier_entry, "path")
    )

    decision = _require_mapping(decision_payload, "decision")
    selected_variant = _require_string(decision, "selected_variant")
    selected_variant_eligible = _require_bool(decision, "selected_variant_eligible")
    promotion_applied = _require_bool(decision, "promotion_applied")

    detector_mode = _require_string(detector_payload, "mode")
    detector_training = _require_mapping(detector_payload, "training")
    detector_placeholder = _require_bool(detector_training, "placeholder")

    classifier_training = _require_mapping(classifier_payload, "training")
    classifier_mode = _require_string(classifier_training, "mode")

    if (
        selected_variant_must_be_eligible
        and not hold_manifest_only
        and not selected_variant_eligible
    ):
        raise FinalTrainError(
            "Final retraining blocked: decision.selected_variant_eligible=false and "
            "frozen config requires an eligible selected variant."
        )

    if selected_variant_eligible and hold_manifest_only:
        raise FinalTrainError(
            "Frozen config is stale: hold_manifest_only=true but promotion "
            "evidence is eligible."
        )

    status = (
        "blocked_hold_manifest" if hold_manifest_only else "ready_for_final_retrain"
    )
    blocked = not selected_variant_eligible
    blocking_reasons = cast(list[str], decision.get("rationale", [])) if blocked else []
    if blocked and detector_mode == "smoke":
        blocking_reasons = [
            *blocking_reasons,
            "detector train summary mode=smoke blocks real final retraining",
        ]
    if blocked and detector_placeholder:
        blocking_reasons = [
            *blocking_reasons,
            "detector train summary training.placeholder=true blocks real "
            "final retraining",
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "final_training": {
            "attempted": False,
            "blocked": blocked,
            "hold_manifest_only": hold_manifest_only,
            "promotion_applied": promotion_applied,
            "selected_variant": selected_variant,
            "selected_variant_eligible": selected_variant_eligible,
            "selected_variant_must_be_eligible": selected_variant_must_be_eligible,
            "blocking_reasons": blocking_reasons,
        },
        "frozen_config": {
            "path": config_path.as_posix(),
            "snapshot": file_snapshot(config_path),
        },
        "release_manifest": {"path": out_path.as_posix(), "kind": "final_manifest"},
        "artifact_references": {
            "decision": {
                **locked_decision,
                "decision": {
                    "promotion_applied": promotion_applied,
                    "selected_variant": selected_variant,
                    "selected_variant_eligible": selected_variant_eligible,
                    "strategy": _require_string(decision, "strategy"),
                },
            },
            "detector_train_summary": {
                **locked_detector,
                "summary": {
                    "mode": detector_mode,
                    "output_dir": _require_string(detector_payload, "output_dir"),
                    "run_name": _require_string(
                        _require_mapping(detector_payload, "search"), "run_name"
                    ),
                    "runtime_version": _require_string(
                        detector_payload, "runtime_version"
                    ),
                    "training_placeholder": detector_placeholder,
                },
            },
            "classifier_train_summary": {
                **locked_classifier,
                "summary": {
                    "mode": classifier_mode,
                    "output_dir": _require_string(classifier_payload, "output_dir"),
                    "weights_artifact_name": _require_string(
                        classifier_training, "weights_artifact_name"
                    ),
                },
            },
        },
        "provenance": {
            "cwd": repo_root.as_posix(),
            "referenced_files": {
                "classifier_train_summary": file_snapshot(
                    repo_root / _require_string(classifier_entry, "path")
                ),
                "decision": file_snapshot(
                    repo_root / _require_string(decision_entry, "path")
                ),
                "detector_train_summary": file_snapshot(
                    repo_root / _require_string(detector_entry, "path")
                ),
            },
        },
    }


def run_final_train(*, config_path: str | Path, out_path: str | Path) -> JsonDict:
    config = Path(config_path)
    out = Path(out_path)
    manifest = build_release_manifest(config_path=config, out_path=out)
    write_json(out, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(FinalTrainArgs, parser.parse_args(argv))
    try:
        manifest = run_final_train(config_path=args.config, out_path=args.out)
    except (FinalTrainError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
