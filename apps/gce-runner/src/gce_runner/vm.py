"""GCE VM lifecycle management for ephemeral code execution."""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

REPO_URL = "https://github.com/Mathias-a/NmFrameMog.git"
DEFAULT_ZONE = "europe-north1-a"
DEFAULT_MACHINE_TYPE = "e2-medium"
LABEL_KEY = "gce-runner"
LABEL_VALUE = "ephemeral"

# Packages available in the workspace
VALID_PACKAGES = frozenset(
    {
        "object-detection",
        "ai-accounting-agent",
        "astar-island",
    }
)


class VMStatus(StrEnum):
    PROVISIONING = "PROVISIONING"
    STAGING = "STAGING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    TERMINATED = "TERMINATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VMInfo:
    name: str
    zone: str
    status: VMStatus
    machine_type: str
    created: str
    package: str
    command: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "zone": self.zone,
            "status": self.status.value,
            "machine_type": self.machine_type,
            "created": self.created,
            "package": self.package,
            "command": self.command,
        }


@dataclass
class VMResult:
    vm_name: str
    success: bool
    message: str
    output: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, str | bool | list[str]]:
        return {
            "vm_name": self.vm_name,
            "success": self.success,
            "message": self.message,
            "output": self.output,
            "errors": self.errors,
        }


def _run_gcloud(
    args: list[str], *, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    """Run a gcloud command and return the result."""
    return subprocess.run(
        ["gcloud", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _get_project() -> str:
    """Get the current gcloud project."""
    result = _run_gcloud(["config", "get-value", "project"])
    project = result.stdout.strip()
    if not project:
        msg = (
            "No gcloud project configured. Run: gcloud config set project <PROJECT_ID>"
        )
        raise RuntimeError(msg)
    return project


def _json_str(data: Mapping[str, object], key: str, default: str = "") -> str:
    """Safely extract a string value from a JSON dict."""
    val = data.get(key, default)
    return str(val) if val is not None else default


def _build_startup_script(
    package: str,
    command: str,
    branch: str,
    env_vars: dict[str, str],
) -> str:
    """Build a startup script that clones the repo, installs deps, runs the command.

    Output goes to /tmp/gce-runner-output and /tmp/gce-runner-error.
    Completion is signaled by writing SUCCESS or FAILED to /tmp/gce-runner-done.
    Progress is logged to /tmp/gce-runner-status.
    """
    env_lines: list[str] = []
    for k, v in env_vars.items():
        # Escape single quotes in values for safe shell embedding
        escaped = v.replace("'", "'\\''")
        env_lines.append(f"export {k}='{escaped}'")
    env_exports = "\n".join(env_lines)

    return f"""#!/bin/bash
set -euo pipefail

MARKER="/tmp/gce-runner-done"
OUTPUT_FILE="/tmp/gce-runner-output"
ERROR_FILE="/tmp/gce-runner-error"
STATUS_FILE="/tmp/gce-runner-status"

log() {{ echo "$(date -u +%H:%M:%S) $1" | tee -a "$STATUS_FILE"; }}

log "=== GCE Runner: Starting ==="

# Install system dependencies
log "Installing system deps..."
apt-get update -qq
apt-get install -y -qq git curl python3 python3-venv > /dev/null 2>&1

# Install uv
log "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="/root/.local/bin:$PATH"

# Clone repo
log "Cloning repo (branch: {branch})..."
git clone --depth 1 --branch {branch} {REPO_URL} /opt/nmframemog
cd /opt/nmframemog

# Set environment variables
{env_exports}

# Install workspace deps
log "Installing dependencies..."
uv sync 2>&1 | tail -5

# Run the command
log "Running: uv run --package {package} {command}"
if uv run --package {package} {command} > "$OUTPUT_FILE" 2>"$ERROR_FILE"; then
    echo "SUCCESS" > "$MARKER"
    log "=== GCE Runner: Command succeeded ==="
else
    echo "FAILED" > "$MARKER"
    log "=== GCE Runner: Command failed ==="
fi
"""


def create_vm(
    package: str,
    command: str,
    *,
    branch: str = "main",
    machine_type: str = DEFAULT_MACHINE_TYPE,
    zone: str = DEFAULT_ZONE,
    env_vars: dict[str, str] | None = None,
) -> VMResult:
    """Create an ephemeral GCE VM that clones the repo and runs a command.

    Args:
        package: Workspace package name (e.g. 'astar-island').
        command: Command to run via ``uv run --package <package> <command>``.
        branch: Git branch to clone (default: main).
        machine_type: GCE machine type (default: e2-medium).
        zone: GCE zone (default: europe-north1-a).
        env_vars: Environment variables to set on the VM.

    Returns:
        VMResult with the VM name and status.
    """
    if package not in VALID_PACKAGES:
        valid = ", ".join(sorted(VALID_PACKAGES))
        return VMResult(
            vm_name="",
            success=False,
            message=f"Invalid package '{package}'. Valid: {valid}",
        )

    vm_name = f"gce-runner-{uuid.uuid4().hex[:8]}"
    startup_script = _build_startup_script(package, command, branch, env_vars or {})
    project = _get_project()

    # Write startup script to a temp file (avoids shell escaping issues)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="gce-runner-"
    ) as f:
        f.write(startup_script)
        script_path = f.name

    result = _run_gcloud(
        [
            "compute",
            "instances",
            "create",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            f"--machine-type={machine_type}",
            "--image-family=debian-13",
            "--image-project=debian-cloud",
            f"--metadata-from-file=startup-script={script_path}",
            f"--metadata=gce-runner-package={package},gce-runner-command={command}",
            f"--labels={LABEL_KEY}={LABEL_VALUE}",
            "--scopes=cloud-platform",
            "--no-restart-on-failure",
            "--format=json",
        ],
        timeout=180,
    )

    if result.returncode != 0:
        return VMResult(
            vm_name=vm_name,
            success=False,
            message=f"Failed to create VM: {result.stderr.strip()}",
            errors=[result.stderr.strip()],
        )

    return VMResult(
        vm_name=vm_name,
        success=True,
        message=(
            f"VM '{vm_name}' created in {zone}. "
            f"Running: uv run --package {package} {command}. "
            f"Boot + setup takes ~2-3 min. "
            f"Use vm_output('{vm_name}') to check results."
        ),
    )


def _parse_vm_metadata(data: Mapping[str, object]) -> tuple[str, str]:
    """Extract gce-runner-package and gce-runner-command from instance metadata."""
    package = ""
    command = ""
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return package, command
    items = metadata.get("items")
    if not isinstance(items, list):
        return package, command
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        value = item.get("value")
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if key == "gce-runner-package":
            package = value
        elif key == "gce-runner-command":
            command = value
    return package, command


def _parse_vm_status(status_str: str) -> VMStatus:
    """Parse a VM status string into a VMStatus enum."""
    try:
        return VMStatus(status_str)
    except ValueError:
        return VMStatus.UNKNOWN


def _parse_instance_data(data: Mapping[str, object], zone: str) -> VMInfo:
    """Parse a single GCE instance JSON dict into VMInfo."""
    package, command = _parse_vm_metadata(data)
    machine_type_full = _json_str(data, "machineType")

    return VMInfo(
        name=_json_str(data, "name"),
        zone=zone,
        status=_parse_vm_status(_json_str(data, "status", "UNKNOWN")),
        machine_type=(
            machine_type_full.rsplit("/", 1)[-1] if machine_type_full else ""
        ),
        created=_json_str(data, "creationTimestamp"),
        package=package,
        command=command,
    )


def get_vm_status(vm_name: str, *, zone: str = DEFAULT_ZONE) -> VMInfo | None:
    """Get the status of a VM."""
    project = _get_project()
    result = _run_gcloud(
        [
            "compute",
            "instances",
            "describe",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--format=json",
        ]
    )

    if result.returncode != 0:
        return None

    raw = json.loads(result.stdout)
    if not isinstance(raw, dict):
        return None
    return _parse_instance_data(raw, zone)


def get_vm_output(vm_name: str, *, zone: str = DEFAULT_ZONE) -> VMResult:
    """Get execution output from a VM via SSH, falling back to serial port.

    Reads /tmp/gce-runner-{done,output,error,status} files on the VM.
    """
    project = _get_project()

    # Try SSH first — gives us structured output from our marker files
    ssh_result = _run_gcloud(
        [
            "compute",
            "ssh",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--command="
            "cat /tmp/gce-runner-done 2>/dev/null; "
            "echo '---SEP---'; "
            "cat /tmp/gce-runner-output 2>/dev/null; "
            "echo '---SEP---'; "
            "cat /tmp/gce-runner-error 2>/dev/null; "
            "echo '---SEP---'; "
            "cat /tmp/gce-runner-status 2>/dev/null",
            "--quiet",
            "--strict-host-key-checking=no",
        ],
        timeout=30,
    )

    if ssh_result.returncode == 0:
        parts = ssh_result.stdout.split("---SEP---")
        done = parts[0].strip() if len(parts) > 0 else ""
        stdout_out = parts[1].strip() if len(parts) > 1 else ""
        stderr_out = parts[2].strip() if len(parts) > 2 else ""
        status_log = parts[3].strip() if len(parts) > 3 else ""

        if done in ("SUCCESS", "FAILED"):
            succeeded = done == "SUCCESS"
            msg = "succeeded" if succeeded else "failed"
            return VMResult(
                vm_name=vm_name,
                success=succeeded,
                message=f"Execution {msg}.",
                output=stdout_out or status_log,
                errors=[stderr_out] if stderr_out else [],
            )
        return VMResult(
            vm_name=vm_name,
            success=True,
            message="Still running...",
            output=status_log or "Startup script in progress.",
        )

    # Fall back to serial port output
    serial = _run_gcloud(
        [
            "compute",
            "instances",
            "get-serial-port-output",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
        ],
        timeout=30,
    )

    if serial.returncode == 0:
        lines = [
            line
            for line in serial.stdout.splitlines()
            if "GCE Runner" in line or "gce-runner" in line
        ]
        return VMResult(
            vm_name=vm_name,
            success=True,
            message="SSH not ready yet. Serial port output:",
            output=("\n".join(lines[-30:]) if lines else "VM is still booting..."),
        )

    return VMResult(
        vm_name=vm_name,
        success=True,
        message="VM is still booting. Try again in a minute.",
        output="",
    )


def list_vms(*, zone: str = DEFAULT_ZONE) -> list[VMInfo]:
    """List all VMs created by gce-runner."""
    project = _get_project()
    result = _run_gcloud(
        [
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--zones={zone}",
            f"--filter=labels.{LABEL_KEY}={LABEL_VALUE}",
            "--format=json",
        ]
    )

    if result.returncode != 0:
        return []

    raw = json.loads(result.stdout)
    if not isinstance(raw, list):
        return []

    vms: list[VMInfo] = []
    for item in raw:
        if isinstance(item, dict):
            vms.append(_parse_instance_data(item, zone))
    return vms


def delete_vm(vm_name: str, *, zone: str = DEFAULT_ZONE) -> VMResult:
    """Delete a VM."""
    project = _get_project()
    result = _run_gcloud(
        [
            "compute",
            "instances",
            "delete",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--quiet",
        ],
        timeout=120,
    )

    if result.returncode != 0:
        return VMResult(
            vm_name=vm_name,
            success=False,
            message=f"Failed to delete: {result.stderr.strip()}",
            errors=[result.stderr.strip()],
        )

    return VMResult(
        vm_name=vm_name,
        success=True,
        message=f"VM '{vm_name}' deleted.",
    )


def delete_all_vms(*, zone: str = DEFAULT_ZONE) -> VMResult:
    """Delete all gce-runner VMs."""
    vms = list_vms(zone=zone)
    if not vms:
        return VMResult(
            vm_name="",
            success=True,
            message="No gce-runner VMs found.",
        )

    names = [vm.name for vm in vms]
    project = _get_project()
    result = _run_gcloud(
        [
            "compute",
            "instances",
            "delete",
            *names,
            f"--project={project}",
            f"--zone={zone}",
            "--quiet",
        ],
        timeout=180,
    )

    if result.returncode != 0:
        return VMResult(
            vm_name="",
            success=False,
            message=f"Failed to delete: {result.stderr.strip()}",
            errors=[result.stderr.strip()],
        )

    return VMResult(
        vm_name="",
        success=True,
        message=f"Deleted {len(names)} VM(s): {', '.join(names)}",
    )
