from __future__ import annotations

import argparse
import json
import shlex
from typing import cast

from src.ng_data.cloud.config import ConfigValidationError, load_cloud_config
from src.ng_data.cloud.layout import render_paths, render_shell_environment
from src.ng_data.cloud.validation import validate_cloud_config


class PrintPathsArgs(argparse.Namespace):
    config: str
    format: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print deterministic storage, VM, and artifact paths."
    )
    parser.add_argument(
        "--config",
        default="configs/cloud/main.json",
        help="Path to the cloud config JSON file.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "env"),
        default="json",
        help="Output JSON or shell export lines for helper scripts.",
    )
    return parser


def _format_env_exports(values: dict[str, str]) -> str:
    lines = []
    for key, value in sorted(values.items()):
        lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(PrintPathsArgs, parser.parse_args(argv))

    try:
        config = load_cloud_config(args.config)
        validate_cloud_config(config)
    except ConfigValidationError as error:
        raise SystemExit(str(error)) from error

    if args.format == "env":
        print(_format_env_exports(render_shell_environment(config)))
    else:
        print(json.dumps(render_paths(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
