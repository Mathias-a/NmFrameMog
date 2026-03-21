from __future__ import annotations

import re

from src.ng_data.cloud.config import CloudConfig, ConfigValidationError

REQUIRED_PREFIX_NAMES = (
    "data",
    "runs",
    "checkpoints",
    "eval_outputs",
    "release_bundles",
    "run_manifests",
)

PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
REGION_PATTERN = re.compile(r"^[a-z]+-[a-z]+[0-9]+$")
ZONE_PATTERN = re.compile(r"^[a-z]+-[a-z]+[0-9]+-[a-z]$")
RESOURCE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,62}$")
BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
PREFIX_PATTERN = re.compile(r"^[a-z0-9][a-z0-9/_-]*[a-z0-9]$")
ABSOLUTE_PATH_PATTERN = re.compile(r"^/[A-Za-z0-9._/-]+$")


def _validate_project_id(project_id: str) -> None:
    if not PROJECT_ID_PATTERN.fullmatch(project_id) or "--" in project_id:
        raise ConfigValidationError(f"Invalid project id: {project_id}")


def _validate_region(region: str) -> None:
    if not REGION_PATTERN.fullmatch(region):
        raise ConfigValidationError(f"Invalid region: {region}")


def _validate_zone(zone: str, region: str) -> None:
    if not ZONE_PATTERN.fullmatch(zone):
        raise ConfigValidationError(f"Invalid zone: {zone}")
    if not zone.startswith(f"{region}-"):
        raise ConfigValidationError(
            f"Zone '{zone}' must live inside region '{region}'."
        )


def _validate_resource_name(label: str, value: str) -> None:
    if not RESOURCE_NAME_PATTERN.fullmatch(value) or "--" in value:
        raise ConfigValidationError(f"Invalid {label}: {value}")


def _validate_bucket(bucket: str) -> None:
    if not BUCKET_PATTERN.fullmatch(bucket) or ".." in bucket or "-." in bucket:
        raise ConfigValidationError(f"Invalid bucket name: {bucket}")


def _validate_prefix(label: str, prefix: str) -> None:
    if (
        not PREFIX_PATTERN.fullmatch(prefix)
        or "//" in prefix
        or prefix.startswith("/")
        or prefix.endswith("/")
    ):
        raise ConfigValidationError(f"Invalid storage prefix '{label}': {prefix}")


def _validate_absolute_path(label: str, path_value: str) -> None:
    if not ABSOLUTE_PATH_PATTERN.fullmatch(path_value) or "//" in path_value:
        raise ConfigValidationError(f"Invalid {label}: {path_value}")


def validate_cloud_config(
    config: CloudConfig,
    *,
    expected_project: str | None = None,
    expected_region: str | None = None,
) -> None:
    if config.schema_version != 1:
        raise ConfigValidationError(
            f"Unsupported schema version: {config.schema_version}"
        )

    if config.provider != "gcp":
        raise ConfigValidationError(
            f"Unsupported cloud provider '{config.provider}'; expected 'gcp'."
        )

    _validate_project_id(config.project_id)
    _validate_region(config.region)
    _validate_bucket(config.storage.bucket)
    _validate_prefix("namespace_prefix", config.storage.namespace_prefix)
    _validate_zone(config.compute_engine.zone, config.region)
    _validate_resource_name("vm name", config.compute_engine.vm_name)
    _validate_absolute_path("workspace_dir", config.compute_engine.workspace_dir)
    _validate_absolute_path(
        "local_artifact_dir", config.compute_engine.local_artifact_dir
    )

    if config.compute_engine.accelerator_count < 1:
        raise ConfigValidationError("accelerator_count must be at least 1")
    if config.compute_engine.boot_disk_gb < 50:
        raise ConfigValidationError("boot_disk_gb must be at least 50")

    missing_prefixes = [
        name for name in REQUIRED_PREFIX_NAMES if name not in config.storage.prefixes
    ]
    if missing_prefixes:
        missing_display = ", ".join(missing_prefixes)
        raise ConfigValidationError(
            f"Missing required storage prefixes: {missing_display}"
        )

    for name, prefix in config.storage.prefixes.items():
        _validate_prefix(name, prefix)

    workflow = config.workflow
    if workflow.primary_execution != "compute_engine":
        raise ConfigValidationError(
            "primary_execution must be 'compute_engine' for the mainline path"
        )
    if workflow.canonical_storage != "gcs":
        raise ConfigValidationError("canonical_storage must be 'gcs'")
    if not workflow.mirror_local_artifacts_to_gcs:
        raise ConfigValidationError("mirror_local_artifacts_to_gcs must be true")

    vertex_ai = workflow.vertex_ai
    if vertex_ai.enabled_by_default:
        raise ConfigValidationError("Vertex AI must remain disabled by default")
    if vertex_ai.mode != "optional_escalation_only":
        raise ConfigValidationError("vertex_ai.mode must be 'optional_escalation_only'")
    if len(vertex_ai.notes) == 0:
        raise ConfigValidationError("vertex_ai.notes must contain escalation guidance")

    if expected_project is not None and expected_project != config.project_id:
        message = (
            f"Config project '{config.project_id}' did not match expected "
            f"'{expected_project}'."
        )
        raise ConfigValidationError(message)
    if expected_region is not None and expected_region != config.region:
        message = (
            f"Config region '{config.region}' did not match expected "
            f"'{expected_region}'."
        )
        raise ConfigValidationError(message)
