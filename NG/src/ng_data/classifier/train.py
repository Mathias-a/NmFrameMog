from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from src.ng_data.classifier.baseline import ClassifierBaselineError, run_training
from src.ng_data.classifier.data import ClassifierDataValidationError


class TrainArgs(argparse.Namespace):
    config: str
    output_dir: str
    processed_root: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write deterministic classifier artifacts from GT crop manifests."
    )
    parser.add_argument(
        "--config",
        default="configs/classifier/search.json",
        help="Path to the classifier config JSON file.",
    )
    parser.add_argument(
        "--processed-root",
        default="data/processed",
        help="Path to the processed dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/classifier",
        help="Directory where classifier artifacts should be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(TrainArgs, parser.parse_args(argv))
    try:
        summary = run_training(
            config_path=Path(args.config),
            processed_root=Path(args.processed_root),
            output_dir=Path(args.output_dir),
        )
    except (
        ClassifierBaselineError,
        ClassifierDataValidationError,
        ValueError,
    ) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
