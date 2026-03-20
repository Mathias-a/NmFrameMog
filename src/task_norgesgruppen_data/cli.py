from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from task_norgesgruppen_data.predictor import (
    generate_predictions,
    write_predictions_json,
)


class RunnerArgs(argparse.Namespace):
    input: str
    output: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate COCO-style predictions for an image folder."
    )
    parser.add_argument(
        "--input", required=True, help="Directory containing input images."
    )
    parser.add_argument(
        "--output", required=True, help="Output JSON path for predictions."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(RunnerArgs, parser.parse_args(argv))

    input_dir = Path(args.input)
    output_path = Path(args.output)

    predictions = generate_predictions(input_dir)
    write_predictions_json(predictions, output_path)
    return 0
