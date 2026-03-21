from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from src.ng_data.eval.splits import (
    SplitConfigValidationError,
    load_split_config,
    write_split_manifest,
)


class MakeSplitsArgs(argparse.Namespace):
    config: str
    manifest: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic holdout and CV split manifests."
    )
    parser.add_argument(
        "--config",
        default="configs/data/splits.json",
        help="Path to the split-generation config JSON file.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifests/dataset_manifest.json",
        help="Path to the processed dataset manifest JSON file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(MakeSplitsArgs, parser.parse_args(argv))
    try:
        config = load_split_config(args.config)
        _, payload = write_split_manifest(config, Path(args.manifest))
    except (SplitConfigValidationError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
