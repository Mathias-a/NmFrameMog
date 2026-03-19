"""MCP server exposing GCE VM management tools for LLM consumption."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from gce_runner.vm import (
    VALID_PACKAGES,
    create_vm,
    delete_all_vms,
    delete_vm,
    get_vm_output,
    get_vm_status,
    list_vms,
)

mcp: FastMCP[None] = FastMCP(
    name="gce-runner",
    instructions=(
        "Manage ephemeral Google Compute Engine VMs that clone the NmFrameMog repo "
        "and run workspace packages. VMs auto-provision with Python 3.13 + uv. "
        "Typical lifecycle: run_on_vm → (wait ~2-3 min) → vm_output → vm_delete."
    ),
)


@mcp.tool()
def run_on_vm(
    package: str,
    command: str,
    branch: str = "main",
    machine_type: str = "e2-medium",
    zone: str = "europe-north1-a",
    env_vars: str = "{}",
) -> str:
    """Spawn an ephemeral GCE VM to run a command from this repo.

    The VM clones the repo, installs deps with uv, and runs:
      uv run --package <package> <command>

    Args:
        package: Workspace package. One of:
            object-detection, ai-accounting-agent, astar-island
        command: Command for uv run
            (e.g. 'python -m astar_island.solver')
        branch: Git branch to clone (default: main)
        machine_type: GCE machine type (default: e2-medium).
            Use e2-standard-4 for heavier workloads.
        zone: GCE zone (default: europe-north1-a)
        env_vars: JSON object of environment variables to set (default: "{}")

    Returns:
        JSON with vm_name, success status, and instructions for next steps.
    """
    parsed_env: dict[str, str] = {}
    if env_vars and env_vars != "{}":
        raw = json.loads(env_vars)
        if isinstance(raw, dict):
            parsed_env = {str(k): str(v) for k, v in raw.items()}
    result = create_vm(
        package,
        command,
        branch=branch,
        machine_type=machine_type,
        zone=zone,
        env_vars=parsed_env,
    )
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def vm_status(vm_name: str, zone: str = "europe-north1-a") -> str:
    """Check the status of a GCE runner VM.

    Args:
        vm_name: Name of the VM (returned by run_on_vm)
        zone: GCE zone (default: europe-north1-a)

    Returns:
        JSON with VM name, status, and metadata.
    """
    info = get_vm_status(vm_name, zone=zone)
    if info is None:
        return json.dumps({"error": f"VM '{vm_name}' not found in zone {zone}."})
    return json.dumps(info.to_dict(), indent=2)


@mcp.tool()
def vm_output(vm_name: str, zone: str = "europe-north1-a") -> str:
    """Get the execution output from a GCE runner VM.

    Reads stdout/stderr from the command that was run on the VM.
    If the VM is still running, returns progress so far.
    If SSH is not ready yet, falls back to serial port output.

    Args:
        vm_name: Name of the VM (returned by run_on_vm)
        zone: GCE zone (default: europe-north1-a)

    Returns:
        JSON with success, output text, and any errors.
    """
    result = get_vm_output(vm_name, zone=zone)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def vm_list(zone: str = "europe-north1-a") -> str:
    """List all active GCE runner VMs.

    Args:
        zone: GCE zone to search (default: europe-north1-a)

    Returns:
        JSON array of VMs with name, status, package, and command.
    """
    vms = list_vms(zone=zone)
    return json.dumps([vm.to_dict() for vm in vms], indent=2)


@mcp.tool()
def vm_delete(vm_name: str, zone: str = "europe-north1-a") -> str:
    """Delete a GCE runner VM.

    Call this after retrieving output to clean up resources.

    Args:
        vm_name: Name of the VM to delete
        zone: GCE zone (default: europe-north1-a)

    Returns:
        JSON with success status and message.
    """
    result = delete_vm(vm_name, zone=zone)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def vm_delete_all(zone: str = "europe-north1-a") -> str:
    """Delete ALL GCE runner VMs. Use for cleanup.

    Args:
        zone: GCE zone (default: europe-north1-a)

    Returns:
        JSON with count of deleted VMs.
    """
    result = delete_all_vms(zone=zone)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def vm_packages() -> str:
    """List available workspace packages that can be run on VMs.

    Returns:
        JSON array of valid package names.
    """
    return json.dumps(sorted(VALID_PACKAGES))
