from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from build_submission import STAGING_ROOT, copy_required_files

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local sanity check for the submission package.")
    parser.add_argument("--train-image-dir", required=True)
    parser.add_argument("--subset-size", type=int, default=130)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_image_dir = Path(args.train_image_dir).resolve()
    subset_dir = STAGING_ROOT.parent / "sanity_input_130"
    output_path = STAGING_ROOT.parent / "sanity_predictions.json"

    copy_required_files()
    if subset_dir.exists():
        for path in subset_dir.iterdir():
            path.unlink()
    else:
        subset_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in train_image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    selected = images[: args.subset_size]
    for image_path in selected:
        target = subset_dir / image_path.name
        if target.exists():
            target.unlink()
        target.symlink_to(image_path)

    start = time.perf_counter()
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    subprocess.run(
        [
            "python",
            str(STAGING_ROOT / "run.py"),
            "--input",
            str(subset_dir),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=STAGING_ROOT,
        env=env,
    )
    elapsed = time.perf_counter() - start

    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise RuntimeError("Predictions output is not a JSON list")
    for prediction in predictions[:20]:
        required_keys = {"image_id", "category_id", "bbox", "score"}
        if set(prediction.keys()) != required_keys:
            raise RuntimeError(f"Unexpected prediction keys: {prediction.keys()}")
        if len(prediction["bbox"]) != 4:
            raise RuntimeError("Prediction bbox does not have length 4")

    print(f"subset_images={len(selected)}")
    print(f"predictions={len(predictions)}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
