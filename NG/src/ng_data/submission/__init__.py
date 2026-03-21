from __future__ import annotations

import argparse
import ast
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any, cast

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
ALLOWED_FILE_TYPES = frozenset(
    {
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".cfg",
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".npy",
    }
)
WEIGHT_FILE_TYPES = frozenset({".pt", ".pth", ".onnx", ".safetensors", ".npy"})
BLOCKED_IMPORTS = frozenset(
    {
        "builtins",
        "code",
        "codeop",
        "ctypes",
        "gc",
        "http.client",
        "importlib",
        "marshal",
        "multiprocessing",
        "os",
        "pickle",
        "pty",
        "requests",
        "shutil",
        "signal",
        "socket",
        "subprocess",
        "sys",
        "shelve",
        "threading",
        "urllib",
        "yaml",
    }
)
BLOCKED_CALLS = frozenset({"eval", "exec", "compile", "__import__", "getattr"})
IMAGE_ID_PATTERN = re.compile(r"(\d+)$")
MAX_FILES = 1000
MAX_PYTHON_FILES = 10
MAX_WEIGHT_FILES = 3
MAX_WEIGHT_BYTES = 420 * 1024 * 1024
JsonDict = dict[str, Any]


class SubmissionConfigValidationError(ValueError):
    pass


class SubmissionContractError(ValueError):
    pass


class SubmissionSecurityError(ValueError):
    pass


class SubmissionBuildError(ValueError):
    pass


class SmokeRunError(ValueError):
    pass


@dataclass(frozen=True)
class SubmissionBundleConfig:
    output_zip: str
    source_files: tuple[str, ...]


class BuildZipArgs(argparse.Namespace):
    config: str


class SmokeRunArgs(argparse.Namespace):
    zip: str
    input: str
    output: str


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SubmissionConfigValidationError(f"Expected '{key}' to be an integer.")
    return value


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise SubmissionConfigValidationError(
            f"Expected '{key}' to be a non-empty string."
        )
    return value


def _require_string_list(data: JsonDict, key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise SubmissionConfigValidationError(
            f"Expected '{key}' to be a non-empty list of strings."
        )
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or entry == "":
            raise SubmissionConfigValidationError(
                f"Expected every '{key}' entry to be a non-empty string."
            )
        items.append(entry)
    return tuple(items)


def parse_submission_bundle_config(data: JsonDict) -> SubmissionBundleConfig:
    schema_version = _require_int(data, "schema_version")
    if schema_version != 1:
        raise SubmissionConfigValidationError(
            f"Expected 'schema_version' to be 1, got {schema_version}."
        )
    return SubmissionBundleConfig(
        output_zip=_require_string(data, "output_zip"),
        source_files=_require_string_list(data, "source_files"),
    )


def load_submission_bundle_config(config_path: str | Path) -> SubmissionBundleConfig:
    path = Path(config_path)
    data = cast(JsonDict, json.loads(path.read_text(encoding="utf-8")))
    return parse_submission_bundle_config(data)


def list_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise SubmissionContractError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise SubmissionContractError(f"Input path is not a directory: {input_dir}")
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def infer_image_id(image_path: Path, fallback_id: int) -> int:
    match = IMAGE_ID_PATTERN.search(image_path.stem)
    if match is None:
        return fallback_id
    return int(match.group(1))


def validate_prediction_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise SubmissionContractError("Predictions payload must be a JSON array.")

    validated: list[dict[str, Any]] = []
    for index, prediction in enumerate(payload):
        if not isinstance(prediction, dict):
            raise SubmissionContractError(
                f"Prediction at index {index} must be an object."
            )
        image_id = prediction.get("image_id")
        category_id = prediction.get("category_id")
        bbox = prediction.get("bbox")
        score = prediction.get("score")

        if not isinstance(image_id, int) or isinstance(image_id, bool):
            raise SubmissionContractError(
                f"Prediction at index {index} has invalid image_id."
            )
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            raise SubmissionContractError(
                f"Prediction at index {index} has invalid category_id."
            )
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise SubmissionContractError(
                f"Prediction at index {index} has invalid score."
            )
        if float(score) < 0 or float(score) > 1:
            raise SubmissionContractError(
                f"Prediction at index {index} has score outside [0, 1]."
            )
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise SubmissionContractError(
                f"Prediction at index {index} must have bbox with four numbers."
            )
        normalized_bbox: list[float] = []
        for value in bbox:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SubmissionContractError(
                    f"Prediction at index {index} has non-numeric bbox values."
                )
            normalized_bbox.append(float(value))
        validated.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": normalized_bbox,
                "score": float(score),
            }
        )
    return validated


def load_predictions(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        raise SubmissionContractError(
            f"Predictions output file does not exist: {output_path}"
        )
    payload = cast(Any, json.loads(output_path.read_text(encoding="utf-8")))
    return validate_prediction_payload(payload)


def _is_blocked_import(module_name: str) -> bool:
    return any(
        module_name == blocked or module_name.startswith(f"{blocked}.")
        for blocked in BLOCKED_IMPORTS
    )


def _extract_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def scan_python_source(source_path: Path) -> None:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_blocked_import(alias.name):
                    raise SubmissionSecurityError(
                        "Blocked import "
                        f"'{alias.name}' found in {source_path.as_posix()}."
                    )
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if _is_blocked_import(module_name):
                raise SubmissionSecurityError(
                    f"Blocked import '{module_name}' found in {source_path.as_posix()}."
                )
        if isinstance(node, ast.Call):
            call_name = _extract_call_name(node.func)
            if call_name in BLOCKED_CALLS:
                raise SubmissionSecurityError(
                    f"Blocked call '{call_name}' found in {source_path.as_posix()}."
                )


def scan_python_files(source_paths: list[Path]) -> None:
    for source_path in source_paths:
        scan_python_source(source_path)


def repo_root_from_config(config_path: Path) -> Path:
    return config_path.resolve().parents[2]


def resolve_bundle_sources(
    repo_root: Path, relative_paths: tuple[str, ...]
) -> list[Path]:
    source_paths: list[Path] = []
    for relative_path in relative_paths:
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise SubmissionBuildError(
                f"Configured bundle source does not exist: {relative_path}"
            )
        source_paths.append(source_path)
    return source_paths


def validate_bundle_sources(source_paths: list[Path], repo_root: Path) -> None:
    arc_names = [path.relative_to(repo_root).name for path in source_paths]
    if "run.py" not in arc_names:
        raise SubmissionBuildError("Submission bundle must include run.py at zip root.")

    python_count = 0
    weight_count = 0
    weight_bytes = 0
    for source_path in source_paths:
        suffix = source_path.suffix.lower()
        if suffix not in ALLOWED_FILE_TYPES:
            raise SubmissionBuildError(
                f"Disallowed file type for submission bundle: {source_path.name}"
            )
        if suffix == ".py":
            python_count += 1
        if suffix in WEIGHT_FILE_TYPES:
            weight_count += 1
            weight_bytes += source_path.stat().st_size

    if len(source_paths) > MAX_FILES:
        raise SubmissionBuildError(
            f"Submission bundle exceeds max file count of {MAX_FILES}."
        )
    if python_count > MAX_PYTHON_FILES:
        raise SubmissionBuildError(
            f"Submission bundle exceeds max Python file count of {MAX_PYTHON_FILES}."
        )
    if weight_count > MAX_WEIGHT_FILES:
        raise SubmissionBuildError(
            f"Submission bundle exceeds max weight file count of {MAX_WEIGHT_FILES}."
        )
    if weight_bytes > MAX_WEIGHT_BYTES:
        raise SubmissionBuildError(
            "Submission bundle exceeds max total weight size of 420 MB."
        )

    python_files = [path for path in source_paths if path.suffix.lower() == ".py"]
    scan_python_files(python_files)


def build_submission_zip(config_path: Path) -> Path:
    config = load_submission_bundle_config(config_path)
    repo_root = repo_root_from_config(config_path)
    source_paths = resolve_bundle_sources(repo_root, config.source_files)
    validate_bundle_sources(source_paths, repo_root)

    output_zip_path = repo_root / config.output_zip
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for source_path in source_paths:
            archive.write(source_path, arcname=source_path.name)

    return output_zip_path


def build_zip_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a contract-safe submission zip for offline smoke checks."
    )
    parser.add_argument(
        "--config",
        default="configs/submission/smoke.json",
        help="Path to the submission bundle config JSON file.",
    )
    return parser


def build_zip_main(argv: list[str] | None = None) -> int:
    parser = build_zip_parser()
    args = cast(BuildZipArgs, parser.parse_args(argv))
    config_path = Path(args.config)
    output_zip_path = build_submission_zip(config_path)
    print(
        json.dumps(
            {
                "config": str(config_path),
                "output_zip": str(output_zip_path),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _assert_zip_contains_root_run_py(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    if "run.py" not in names:
        raise SmokeRunError("Submission zip does not contain run.py at the root.")


def run_smoke_submission(
    zip_path: Path, input_dir: Path, output_path: Path
) -> list[dict[str, Any]]:
    _assert_zip_contains_root_run_py(zip_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ng-submission-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_root)

        run_path = temp_root / "run.py"
        completed: CompletedProcess[str] = run(
            [
                "python",
                str(run_path),
                "--input",
                str(input_dir),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise SmokeRunError(
                "Smoke run failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )

    return load_predictions(output_path)


def smoke_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a packaged submission zip against local fixtures offline."
    )
    parser.add_argument("--zip", required=True, help="Path to the submission zip.")
    parser.add_argument(
        "--input", required=True, help="Directory containing local fixture images."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output predictions JSON path for the smoke run.",
    )
    return parser


def smoke_run_main(argv: list[str] | None = None) -> int:
    parser = smoke_run_parser()
    args = cast(SmokeRunArgs, parser.parse_args(argv))
    predictions = run_smoke_submission(
        zip_path=Path(args.zip),
        input_dir=Path(args.input),
        output_path=Path(args.output),
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "prediction_count": len(validate_prediction_payload(predictions)),
                "status": "ok",
                "zip": args.zip,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "SubmissionBuildError",
    "SubmissionContractError",
    "SubmissionSecurityError",
    "SmokeRunError",
    "build_submission_zip",
    "build_zip_main",
    "infer_image_id",
    "list_image_files",
    "run_smoke_submission",
    "scan_python_source",
    "smoke_run_main",
    "validate_bundle_sources",
    "validate_prediction_payload",
]
