from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

ALLOWED_SANDBOX_RUNTIME_PINS = {
    "timm": "0.9.12",
    "torchvision": "0.21.0",
    "ultralytics": "8.1.0",
}


def load_pyproject() -> dict[str, Any]:
    toml_library = importlib.import_module("tomllib")
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        return cast(dict[str, Any], toml_library.load(pyproject_file))


def runtime_dependencies(pyproject: dict[str, Any]) -> list[str]:
    project = cast(dict[str, Any], pyproject["project"])
    dependencies = list(cast(list[str], project.get("dependencies", [])))
    optional_dependencies = cast(
        dict[str, list[str]], project.get("optional-dependencies", {})
    )
    for group_dependencies in optional_dependencies.values():
        dependencies.extend(group_dependencies)
    return dependencies


def dependency_for(requirements: list[str], package_name: str) -> str | None:
    prefix = f"{package_name}=="
    for requirement in requirements:
        if requirement.startswith(prefix):
            return requirement
        if requirement.split("==", maxsplit=1)[0] == package_name:
            return requirement
    return None


def test_python_requirement_matches_sandbox_version() -> None:
    pyproject = load_pyproject()

    project = cast(dict[str, Any], pyproject["project"])
    assert project["requires-python"] == ">=3.11,<3.12"


def test_sandbox_sensitive_runtime_dependencies_are_absent_or_exactly_pinned() -> None:
    pyproject = load_pyproject()

    requirements = runtime_dependencies(pyproject)
    for package_name, expected_version in ALLOWED_SANDBOX_RUNTIME_PINS.items():
        requirement = dependency_for(requirements, package_name)
        if requirement is None:
            continue

        assert requirement == f"{package_name}=={expected_version}"
