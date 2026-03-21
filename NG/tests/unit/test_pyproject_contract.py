from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

EXPECTED_DEV_DEPENDENCIES = {
    "mypy": "1.18.2",
    "pytest": "8.4.2",
    "ruff": "0.15.6",
}

EXPECTED_EXCLUDED_PATHS = {
    ".pytest_cache",
    ".venv",
    "build",
    "dist",
    "tests/unit/fixtures",
}


def load_pyproject() -> dict[str, Any]:
    toml_library = importlib.import_module("tomllib")
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        return cast(dict[str, Any], toml_library.load(pyproject_file))


def parse_exact_pins(requirements: list[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for requirement in requirements:
        name, version = requirement.split("==", maxsplit=1)
        pins[name] = version
    return pins


def test_task_2_pyproject_contract_is_deterministic() -> None:
    pyproject = load_pyproject()

    project = cast(dict[str, Any], pyproject["project"])
    assert project["requires-python"] == ">=3.11,<3.12"
    assert project["dependencies"] == []

    dependency_groups = cast(dict[str, Any], pyproject["dependency-groups"])
    dev_dependencies = cast(list[str], dependency_groups["dev"])
    assert parse_exact_pins(dev_dependencies) == EXPECTED_DEV_DEPENDENCIES


def test_pytest_and_ruff_configs_protect_local_workflow_paths() -> None:
    pyproject = load_pyproject()

    tool_config = cast(dict[str, Any], pyproject["tool"])
    pytest_config = cast(dict[str, Any], tool_config["pytest"])["ini_options"]
    assert pytest_config["minversion"] == "8.4"
    assert pytest_config["testpaths"] == ["tests/unit"]
    assert EXPECTED_EXCLUDED_PATHS.issubset(set(pytest_config["norecursedirs"]))

    ruff_config = cast(dict[str, Any], tool_config["ruff"])
    assert ruff_config["target-version"] == "py311"
    assert EXPECTED_EXCLUDED_PATHS.issubset(set(ruff_config["exclude"]))
    assert ruff_config["lint"]["select"] == ["B", "E", "F", "I", "UP"]


def test_mypy_config_targets_ng_sources_under_python_3_11() -> None:
    pyproject = load_pyproject()

    tool_config = cast(dict[str, Any], pyproject["tool"])
    mypy_config = cast(dict[str, Any], tool_config["mypy"])
    assert mypy_config["python_version"] == "3.11"
    assert mypy_config["files"] == ["src", "tests"]
    assert mypy_config["namespace_packages"] is True
    assert mypy_config["explicit_package_bases"] is True
    exclude_pattern = mypy_config["exclude"]
    assert "tests/unit/fixtures" in exclude_pattern
    assert "\\.venv" in exclude_pattern
    assert "build" in exclude_pattern
    assert "dist" in exclude_pattern
