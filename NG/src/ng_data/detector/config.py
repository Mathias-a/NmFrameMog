from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]

SUPPORTED_DETECTOR_FRAMEWORK = "ultralytics"
SUPPORTED_DETECTOR_VERSION = "8.1.0"
SUPPORTED_EXPORT_FORMAT = "pt"
SUPPORTED_BASELINE = "detector_only_search"
SUPPORTED_DEVICE_VALUES = {"cpu", "cuda"}


class DetectorConfigValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DetectorRuntimeConfig:
    framework: str
    version: str
    export_format: str


@dataclass(frozen=True)
class DetectorModelConfig:
    name: str
    weights: str


@dataclass(frozen=True)
class DetectorSearchConfig:
    device: str
    epochs: int
    image_size: int
    batch_size: int
    patience: int
    run_name: str


@dataclass(frozen=True)
class DetectorConfig:
    schema_version: int
    baseline: str
    runtime: DetectorRuntimeConfig
    model: DetectorModelConfig
    search: DetectorSearchConfig


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise DetectorConfigValidationError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise DetectorConfigValidationError(
            f"Expected '{key}' to be a non-empty string."
        )
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DetectorConfigValidationError(f"Expected '{key}' to be an integer.")
    return value


def _validate_relative_path(label: str, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        raise DetectorConfigValidationError(
            f"Expected '{label}' to be a relative path, got '{value}'."
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise DetectorConfigValidationError(
            f"Expected '{label}' to avoid empty, '.' or '..' path segments."
        )
    return value


def _parse_runtime(data: JsonDict) -> DetectorRuntimeConfig:
    return DetectorRuntimeConfig(
        framework=_require_string(data, "framework"),
        version=_require_string(data, "version"),
        export_format=_require_string(data, "export_format"),
    )


def _parse_model(data: JsonDict) -> DetectorModelConfig:
    return DetectorModelConfig(
        name=_require_string(data, "name"),
        weights=_validate_relative_path("weights", _require_string(data, "weights")),
    )


def _parse_search(data: JsonDict) -> DetectorSearchConfig:
    return DetectorSearchConfig(
        device=_require_string(data, "device"),
        epochs=_require_int(data, "epochs"),
        image_size=_require_int(data, "image_size"),
        batch_size=_require_int(data, "batch_size"),
        patience=_require_int(data, "patience"),
        run_name=_require_string(data, "run_name"),
    )


def parse_detector_config(data: JsonDict) -> DetectorConfig:
    return DetectorConfig(
        schema_version=_require_int(data, "schema_version"),
        baseline=_require_string(data, "baseline"),
        runtime=_parse_runtime(_require_mapping(data, "runtime")),
        model=_parse_model(_require_mapping(data, "model")),
        search=_parse_search(_require_mapping(data, "search")),
    )


def load_detector_config(config_path: str | Path) -> DetectorConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        data = cast(JsonDict, json.load(config_file))
    return parse_detector_config(data)


def validate_detector_config(config: DetectorConfig) -> None:
    if config.schema_version != 1:
        raise DetectorConfigValidationError(
            f"Unsupported schema version: {config.schema_version}"
        )
    if config.baseline != SUPPORTED_BASELINE:
        raise DetectorConfigValidationError(
            "Unsupported detector baseline: "
            f"{config.baseline}. Expected '{SUPPORTED_BASELINE}'."
        )
    if config.runtime.framework != SUPPORTED_DETECTOR_FRAMEWORK:
        raise DetectorConfigValidationError(
            "Unsupported detector framework: "
            f"{config.runtime.framework}. Expected '{SUPPORTED_DETECTOR_FRAMEWORK}'."
        )
    if config.runtime.version != SUPPORTED_DETECTOR_VERSION:
        raise DetectorConfigValidationError(
            "Unsupported detector runtime version/export combination: "
            f"{config.runtime.framework}=={config.runtime.version} with "
            f"export_format='{config.runtime.export_format}'. "
            f"The detector baseline requires {SUPPORTED_DETECTOR_FRAMEWORK}=="
            f"{SUPPORTED_DETECTOR_VERSION} with export_format='"
            f"{SUPPORTED_EXPORT_FORMAT}'."
        )
    if config.runtime.export_format != SUPPORTED_EXPORT_FORMAT:
        raise DetectorConfigValidationError(
            "Unsupported detector runtime version/export combination: "
            f"{config.runtime.framework}=={config.runtime.version} with "
            f"export_format='{config.runtime.export_format}'. "
            f"The detector baseline requires {SUPPORTED_DETECTOR_FRAMEWORK}=="
            f"{SUPPORTED_DETECTOR_VERSION} with export_format='"
            f"{SUPPORTED_EXPORT_FORMAT}'."
        )
    if config.search.device not in SUPPORTED_DEVICE_VALUES:
        allowed_devices = ", ".join(sorted(SUPPORTED_DEVICE_VALUES))
        raise DetectorConfigValidationError(
            f"Unsupported search device '{config.search.device}'. Expected one of: "
            f"{allowed_devices}."
        )
    if not config.model.name.startswith("yolov8"):
        raise DetectorConfigValidationError(
            "The detector-only baseline currently supports YOLOv8 model names only."
        )
    if Path(config.model.weights).name != config.model.weights:
        raise DetectorConfigValidationError(
            "Expected 'weights' to be a sandbox-safe filename without directories."
        )
    if not config.model.weights.endswith(".pt"):
        raise DetectorConfigValidationError(
            "The detector-only baseline expects '.pt' weights for ultralytics 8.1.0."
        )
    if config.search.epochs < 1:
        raise DetectorConfigValidationError("epochs must be at least 1")
    if config.search.image_size < 32:
        raise DetectorConfigValidationError("image_size must be at least 32")
    if config.search.batch_size < 1:
        raise DetectorConfigValidationError("batch_size must be at least 1")
    if config.search.patience < 0:
        raise DetectorConfigValidationError("patience must be at least 0")


def load_and_validate_detector_config(config_path: str | Path) -> DetectorConfig:
    config = load_detector_config(config_path)
    validate_detector_config(config)
    return config
