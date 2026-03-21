from __future__ import annotations

from src.ng_data.cloud.config import CloudConfig


def _with_trailing_slash(value: str) -> str:
    return value if value.endswith("/") else f"{value}/"


def gcs_namespace_root(config: CloudConfig) -> str:
    return _with_trailing_slash(
        f"gs://{config.storage.bucket}/{config.storage.namespace_prefix}"
    )


def render_storage_paths(config: CloudConfig) -> dict[str, str]:
    namespace_root = gcs_namespace_root(config)
    rendered = {
        "bucket_root": _with_trailing_slash(f"gs://{config.storage.bucket}"),
        "namespace_root": namespace_root,
    }
    for name, prefix in sorted(config.storage.prefixes.items()):
        rendered[name] = _with_trailing_slash(f"{namespace_root}{prefix}")
    return rendered


def render_compute_engine_paths(config: CloudConfig) -> dict[str, str]:
    return {
        "bootstrap_script": config.compute_engine.bootstrap_script,
        "instance_resource": (
            f"projects/{config.project_id}/zones/{config.compute_engine.zone}"
            f"/instances/{config.compute_engine.vm_name}"
        ),
        "local_artifact_dir": config.compute_engine.local_artifact_dir,
        "workspace_dir": config.compute_engine.workspace_dir,
        "zone": config.compute_engine.zone,
    }


def render_artifact_paths(config: CloudConfig) -> dict[str, str]:
    storage_paths = render_storage_paths(config)
    return {
        "checkpoints": storage_paths["checkpoints"],
        "data": storage_paths["data"],
        "eval_outputs": storage_paths["eval_outputs"],
        "release_bundles": storage_paths["release_bundles"],
        "run_manifests": storage_paths["run_manifests"],
        "runs": storage_paths["runs"],
    }


def render_paths(config: CloudConfig) -> dict[str, object]:
    return {
        "artifacts": render_artifact_paths(config),
        "compute_engine": render_compute_engine_paths(config),
        "storage": render_storage_paths(config),
        "workflow": {
            "canonical_storage": config.workflow.canonical_storage,
            "primary_execution": config.workflow.primary_execution,
            "vertex_ai_mode": config.workflow.vertex_ai.mode,
        },
    }


def render_shell_environment(config: CloudConfig) -> dict[str, str]:
    artifact_paths = render_artifact_paths(config)
    storage_paths = render_storage_paths(config)
    return {
        "GCE_ACCELERATOR_COUNT": str(config.compute_engine.accelerator_count),
        "GCE_ACCELERATOR_TYPE": config.compute_engine.accelerator_type,
        "GCE_BOOT_DISK_GB": str(config.compute_engine.boot_disk_gb),
        "GCE_BOOTSTRAP_SCRIPT": config.compute_engine.bootstrap_script,
        "GCE_IMAGE_FAMILY": config.compute_engine.image_family,
        "GCE_IMAGE_PROJECT": config.compute_engine.image_project,
        "GCE_LOCAL_ARTIFACT_DIR": config.compute_engine.local_artifact_dir,
        "GCE_MACHINE_TYPE": config.compute_engine.machine_type,
        "GCE_VM_NAME": config.compute_engine.vm_name,
        "GCE_WORKSPACE_DIR": config.compute_engine.workspace_dir,
        "GCE_ZONE": config.compute_engine.zone,
        "GCP_BUCKET": config.storage.bucket,
        "GCP_CANONICAL_STORAGE": config.workflow.canonical_storage,
        "GCP_PRIMARY_EXECUTION": config.workflow.primary_execution,
        "GCP_PROJECT_ID": config.project_id,
        "GCP_REGION": config.region,
        "GCS_BUCKET_ROOT": storage_paths["bucket_root"],
        "GCS_CHECKPOINTS": artifact_paths["checkpoints"],
        "GCS_DATA": artifact_paths["data"],
        "GCS_EVAL_OUTPUTS": artifact_paths["eval_outputs"],
        "GCS_NAMESPACE_ROOT": storage_paths["namespace_root"],
        "GCS_RELEASE_BUNDLES": artifact_paths["release_bundles"],
        "GCS_RUN_MANIFESTS": artifact_paths["run_manifests"],
        "GCS_RUNS": artifact_paths["runs"],
        "VERTEX_AI_MODE": config.workflow.vertex_ai.mode,
    }
