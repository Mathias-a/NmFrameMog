from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from src.ng_data.cloud.validate_config import main as validate_config_main


def load_main_config() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs/cloud/main.json"
    return cast(dict[str, Any], json.loads(config_path.read_text(encoding="utf-8")))


def test_invalid_project_id_is_rejected(tmp_path: Path) -> None:
    config = load_main_config()
    config["project_id"] = "BadProject"

    invalid_path = tmp_path / "invalid_project.json"
    invalid_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        validate_config_main(["--config", str(invalid_path), "--dry-run"])
    except SystemExit as error:
        assert str(error) == "Invalid project id: BadProject"
    else:
        raise AssertionError("Expected invalid project id to raise SystemExit")


def test_invalid_bucket_prefix_is_rejected(tmp_path: Path) -> None:
    config = load_main_config()
    storage = cast(dict[str, Any], dict(cast(dict[str, Any], config["storage"])))
    prefixes = cast(dict[str, Any], dict(cast(dict[str, Any], storage["prefixes"])))
    prefixes["checkpoints"] = "/bad-prefix/"
    storage["prefixes"] = prefixes
    config["storage"] = storage

    invalid_path = tmp_path / "invalid_prefix.json"
    invalid_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        validate_config_main(["--config", str(invalid_path), "--dry-run"])
    except SystemExit as error:
        assert str(error) == "Invalid storage prefix 'checkpoints': /bad-prefix/"
    else:
        raise AssertionError("Expected invalid storage prefix to raise SystemExit")
