from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
SRC_ROOT_TEXT = str(SRC_ROOT)

if SRC_ROOT_TEXT not in sys.path:
    sys.path.insert(0, SRC_ROOT_TEXT)

DETECTOR_CONFIG_MODULE = import_module("ng_data.detector.config")
DetectorConfigValidationError = DETECTOR_CONFIG_MODULE.DetectorConfigValidationError
load_and_validate_detector_config = (
    DETECTOR_CONFIG_MODULE.load_and_validate_detector_config
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_yolov8m_search_config_loads_for_supported_baseline() -> None:
    config_path = _repo_root() / "configs/detector/yolov8m-search.json"

    config = load_and_validate_detector_config(config_path)

    assert config.runtime.framework == "ultralytics"
    assert config.runtime.version == "8.1.0"
    assert config.runtime.export_format == "pt"
    assert config.model.name == "yolov8m"


def _write_detector_config(tmp_path: Path, *, version: str, export_format: str) -> Path:
    payload = {
        "schema_version": 1,
        "baseline": "detector_only_search",
        "runtime": {
            "framework": "ultralytics",
            "version": version,
            "export_format": export_format,
        },
        "model": {"name": "yolov8m", "weights": "yolov8m.pt"},
        "search": {
            "device": "cuda",
            "epochs": 30,
            "image_size": 960,
            "batch_size": 16,
            "patience": 10,
            "run_name": "guard-check",
        },
    }
    config_path = tmp_path / f"invalid-detector-{version}-{export_format}.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _assert_invalid_version_export_combination(
    tmp_path: Path, *, version: str, export_format: str
) -> None:
    config_path = _write_detector_config(
        tmp_path, version=version, export_format=export_format
    )

    try:
        load_and_validate_detector_config(config_path)
    except DetectorConfigValidationError as error:
        message = str(error)
    else:
        raise AssertionError("Expected detector config validation to fail.")

    assert (
        "Unsupported detector runtime version/export combination: "
        f"ultralytics=={version} with export_format='{export_format}'"
    ) in message


def test_invalid_detector_version_export_combinations_fail_fast(
    tmp_path: Path,
) -> None:
    _assert_invalid_version_export_combination(
        tmp_path, version="8.2.0", export_format="pt"
    )
    _assert_invalid_version_export_combination(
        tmp_path, version="8.1.0", export_format="onnx"
    )
