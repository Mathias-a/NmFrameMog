from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]

RELATIVE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*[A-Za-z0-9]$")
ARCHIVE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+\.zip$")


class DataConfigValidationError(ValueError):
    pass


@dataclass(frozen=True)
class RawArchiveConfig:
    coco_dataset: str
    product_images: str


@dataclass(frozen=True)
class RawLayoutConfig:
    extract_root: str


@dataclass(frozen=True)
class ProcessedLayoutConfig:
    images_dir: str
    annotations_path: str
    categories_path: str
    reference_images_dir: str
    reference_metadata_path: str
    product_index_path: str
    manifest_path: str


@dataclass(frozen=True)
class DataConfig:
    schema_version: int
    raw_archives: RawArchiveConfig
    raw_layout: RawLayoutConfig
    processed_layout: ProcessedLayoutConfig


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise DataConfigValidationError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise DataConfigValidationError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataConfigValidationError(f"Expected '{key}' to be an integer.")
    return value


def _validate_archive_name(label: str, value: str) -> str:
    if not ARCHIVE_NAME_PATTERN.fullmatch(value):
        raise DataConfigValidationError(
            f"Expected '{label}' to be a .zip filename without directory components."
        )
    if "/" in value or "\\" in value:
        raise DataConfigValidationError(
            f"Expected '{label}' to be a bare archive filename, got '{value}'."
        )
    return value


def _validate_relative_path(label: str, value: str) -> str:
    if not RELATIVE_PATH_PATTERN.fullmatch(value):
        raise DataConfigValidationError(
            f"Expected '{label}' to be a relative POSIX-style path, got '{value}'."
        )
    parts = Path(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise DataConfigValidationError(
            f"Expected '{label}' to avoid empty, '.' or '..' path segments."
        )
    return value


def _parse_raw_archives(data: JsonDict) -> RawArchiveConfig:
    return RawArchiveConfig(
        coco_dataset=_validate_archive_name(
            "coco_dataset", _require_string(data, "coco_dataset")
        ),
        product_images=_validate_archive_name(
            "product_images", _require_string(data, "product_images")
        ),
    )


def _parse_raw_layout(data: JsonDict) -> RawLayoutConfig:
    return RawLayoutConfig(
        extract_root=_validate_relative_path(
            "extract_root", _require_string(data, "extract_root")
        )
    )


def _parse_processed_layout(data: JsonDict) -> ProcessedLayoutConfig:
    return ProcessedLayoutConfig(
        images_dir=_validate_relative_path(
            "images_dir", _require_string(data, "images_dir")
        ),
        annotations_path=_validate_relative_path(
            "annotations_path", _require_string(data, "annotations_path")
        ),
        categories_path=_validate_relative_path(
            "categories_path", _require_string(data, "categories_path")
        ),
        reference_images_dir=_validate_relative_path(
            "reference_images_dir", _require_string(data, "reference_images_dir")
        ),
        reference_metadata_path=_validate_relative_path(
            "reference_metadata_path",
            _require_string(data, "reference_metadata_path"),
        ),
        product_index_path=_validate_relative_path(
            "product_index_path", _require_string(data, "product_index_path")
        ),
        manifest_path=_validate_relative_path(
            "manifest_path", _require_string(data, "manifest_path")
        ),
    )


def parse_data_config(data: JsonDict) -> DataConfig:
    return DataConfig(
        schema_version=_require_int(data, "schema_version"),
        raw_archives=_parse_raw_archives(_require_mapping(data, "raw_archives")),
        raw_layout=_parse_raw_layout(_require_mapping(data, "raw_layout")),
        processed_layout=_parse_processed_layout(
            _require_mapping(data, "processed_layout")
        ),
    )


def load_data_config(config_path: str | Path) -> DataConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        data = cast(JsonDict, json.load(config_file))
    return parse_data_config(data)
