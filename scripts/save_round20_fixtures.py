#!/usr/bin/env python3
"""Convert Round #20 analysis JSON files into fixture format for training.

Reads the raw analysis JSONs saved by analyze_round20.py and writes
proper fixture files to data/fixtures/.
"""

from __future__ import annotations

import json
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis" / "round20"
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
ROUND_20_ID = "fd82f643-15e2-40e7-9866-8d8f5157081c"


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for seed in range(5):
        raw_path = ANALYSIS_DIR / f"seed{seed}_raw.json"
        if not raw_path.exists():
            print(f"Seed {seed}: raw file not found at {raw_path}, skipping")
            continue

        raw = json.loads(raw_path.read_text())

        fixture = {
            "round_id": ROUND_20_ID,
            "seed_index": seed,
            "initial_grid": raw["initial_grid"],
            "ground_truth": raw["ground_truth"],
            "official_score": raw.get("score", 0.0),
        }

        out_path = FIXTURE_DIR / f"{ROUND_20_ID}_seed{seed}.json"
        out_path.write_text(json.dumps(fixture))
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(
            f"Seed {seed}: saved {out_path.name} ({size_mb:.1f} MB, score={fixture['official_score']:.2f})"
        )

    # Verify total count
    total = len(list(FIXTURE_DIR.glob("*_seed*.json")))
    print(f"\nTotal fixtures: {total}")


if __name__ == "__main__":
    main()
