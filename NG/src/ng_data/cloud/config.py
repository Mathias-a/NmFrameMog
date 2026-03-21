from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


class ConfigValidationError(ValueError):
    """Raised when the cloud configuration is malformed or inconsistent."""


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class StorageConfig:
    bucket: str
    namespace_prefix: str
    prefixes: dict[str, str]


@dataclass(frozen=True)
class ComputeEngineConfig:
    vm_name: str
    zone: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    boot_disk_gb: int
    image_project: str
    image_family: str
    workspace_dir: str
    local_artifact_dir: str
    bootstrap_script: str


@dataclass(frozen=True)
class VertexAIConfig:
    enabled_by_default: bool
    mode: str
    notes: list[str]


@dataclass(frozen=True)
class WorkflowConfig:
    primary_execution: str
    canonical_storage: str
    mirror_local_artifacts_to_gcs: bool
    vertex_ai: VertexAIConfig


@dataclass(frozen=True)
class CloudConfig:
    schema_version: int
    provider: str
    project_id: str
    region: str
    storage: StorageConfig
    compute_engine: ComputeEngineConfig
    workflow: WorkflowConfig


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigValidationError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise ConfigValidationError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigValidationError(f"Expected '{key}' to be an integer.")
    return value


def _require_bool(data: JsonDict, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigValidationError(f"Expected '{key}' to be a boolean.")
    return value


def _require_string_list(data: JsonDict, key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item != "" for item in value
    ):
        raise ConfigValidationError(
            f"Expected '{key}' to be a list of non-empty strings."
        )
    return cast(list[str], value)


def _parse_storage(data: JsonDict) -> StorageConfig:
    prefixes = _require_mapping(data, "prefixes")
    parsed_prefixes: dict[str, str] = {}
    for name, value in prefixes.items():
        if not isinstance(value, str) or value == "":
            raise ConfigValidationError(
                f"Expected storage prefix '{name}' to be a non-empty string."
            )
        parsed_prefixes[name] = value

    return StorageConfig(
        bucket=_require_string(data, "bucket"),
        namespace_prefix=_require_string(data, "namespace_prefix"),
        prefixes=parsed_prefixes,
    )


def _parse_compute_engine(data: JsonDict) -> ComputeEngineConfig:
    return ComputeEngineConfig(
        vm_name=_require_string(data, "vm_name"),
        zone=_require_string(data, "zone"),
        machine_type=_require_string(data, "machine_type"),
        accelerator_type=_require_string(data, "accelerator_type"),
        accelerator_count=_require_int(data, "accelerator_count"),
        boot_disk_gb=_require_int(data, "boot_disk_gb"),
        image_project=_require_string(data, "image_project"),
        image_family=_require_string(data, "image_family"),
        workspace_dir=_require_string(data, "workspace_dir"),
        local_artifact_dir=_require_string(data, "local_artifact_dir"),
        bootstrap_script=_require_string(data, "bootstrap_script"),
    )


def _parse_vertex_ai(data: JsonDict) -> VertexAIConfig:
    return VertexAIConfig(
        enabled_by_default=_require_bool(data, "enabled_by_default"),
        mode=_require_string(data, "mode"),
        notes=_require_string_list(data, "notes"),
    )


def _parse_workflow(data: JsonDict) -> WorkflowConfig:
    return WorkflowConfig(
        primary_execution=_require_string(data, "primary_execution"),
        canonical_storage=_require_string(data, "canonical_storage"),
        mirror_local_artifacts_to_gcs=_require_bool(
            data, "mirror_local_artifacts_to_gcs"
        ),
        vertex_ai=_parse_vertex_ai(_require_mapping(data, "vertex_ai")),
    )


def parse_cloud_config(data: JsonDict) -> CloudConfig:
    schema_version = _require_int(data, "schema_version")
    return CloudConfig(
        schema_version=schema_version,
        provider=_require_string(data, "provider"),
        project_id=_require_string(data, "project_id"),
        region=_require_string(data, "region"),
        storage=_parse_storage(_require_mapping(data, "storage")),
        compute_engine=_parse_compute_engine(_require_mapping(data, "compute_engine")),
        workflow=_parse_workflow(_require_mapping(data, "workflow")),
    )


def load_cloud_config(config_path: str | Path) -> CloudConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        data = cast(JsonDict, json.load(config_file))
    return parse_cloud_config(data)
