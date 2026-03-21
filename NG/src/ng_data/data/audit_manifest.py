from __future__ import annotations

import argparse
import json
from typing import cast

from src.ng_data.data.manifest import (
    DataManifestValidationError,
    audit_dataset_manifest,
)


class AuditManifestArgs(argparse.Namespace):
    manifest: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a generated dataset manifest against local files."
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifests/dataset_manifest.json",
        help="Path to the generated dataset manifest JSON file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(AuditManifestArgs, parser.parse_args(argv))
    try:
        summary = audit_dataset_manifest(args.manifest)
    except DataManifestValidationError as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
