from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from src.ng_data.classifier.baseline import (
    DETECTOR_BOXES_MODE,
    GT_BOXES_MODE,
    ClassifierBaselineError,
    run_evaluation,
)


class EvaluateArgs(argparse.Namespace):
    detector_predictions: str | None
    mode: str
    out: str
    weights: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic classifier artifacts on GT or detector boxes."
        )
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the classifier weights artifact.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[GT_BOXES_MODE, DETECTOR_BOXES_MODE],
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--detector-predictions",
        help="Path to detector predictions JSON when using detector_boxes mode.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the evaluation metrics JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(EvaluateArgs, parser.parse_args(argv))
    try:
        payload = run_evaluation(
            weights_path=Path(args.weights),
            mode=args.mode,
            out_path=Path(args.out),
            detector_predictions_path=(
                None
                if args.detector_predictions is None
                else Path(args.detector_predictions)
            ),
        )
    except (ClassifierBaselineError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
