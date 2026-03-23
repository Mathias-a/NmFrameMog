from __future__ import annotations

import json
import sys
from pathlib import Path

from astar_twin.strategies.learned_calibrator.training import cross_validate_zone_calibrator


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run leave-one-round-out benchmark for the learned zone calibrator"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory containing rounds/ subdirectory (default: data).",
    )
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=25,
        help="MC runs for the calibrated MC base strategy while building features.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    report = cross_validate_zone_calibrator(data_dir, n_runs=args.n_runs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print("Learned calibrator leave-one-round-out benchmark")
    print(f"  Learned weighted mean:  {report.learned_weighted_mean:.2f}")
    print(f"  Base weighted mean:     {report.base_weighted_mean:.2f}")
    print(f"  Fallback weighted mean: {report.fallback_weighted_mean:.2f}")
    print(f"  Mean learned weights:   {report.mean_learned_weights()}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
