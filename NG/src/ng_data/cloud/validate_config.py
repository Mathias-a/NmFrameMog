from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from src.ng_data.cloud.config import (
    CloudConfig,
    ConfigValidationError,
    load_cloud_config,
)
from src.ng_data.cloud.layout import gcs_namespace_root
from src.ng_data.cloud.validation import validate_cloud_config


class ValidateConfigArgs(argparse.Namespace):
    config: str
    project: str | None
    region: str | None
    dry_run: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the deterministic GCP training layout config."
    )
    parser.add_argument(
        "--config",
        default="configs/cloud/main.json",
        help="Path to the cloud config JSON file.",
    )
    parser.add_argument(
        "--project",
        help="Expected GCP project id for an extra consistency check.",
    )
    parser.add_argument(
        "--region",
        help="Expected GCP region for an extra consistency check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate locally without attempting any cloud calls.",
    )
    return parser


def validation_summary(config: CloudConfig, config_path: Path) -> dict[str, object]:
    return {
        "bucket_root": gcs_namespace_root(config),
        "config": str(config_path),
        "dry_run": True,
        "project_id": config.project_id,
        "region": config.region,
        "status": "ok",
        "vm_name": config.compute_engine.vm_name,
        "vertex_ai_mode": config.workflow.vertex_ai.mode,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(ValidateConfigArgs, parser.parse_args(argv))

    config_path = Path(args.config)
    try:
        config = load_cloud_config(config_path)
        validate_cloud_config(
            config,
            expected_project=args.project,
            expected_region=args.region,
        )
    except ConfigValidationError as error:
        raise SystemExit(str(error)) from error

    summary = validation_summary(config, config_path)
    summary["dry_run"] = args.dry_run
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
