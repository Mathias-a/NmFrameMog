from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from task_norgesgruppen_data.predictor import infer_image_id


def test_infer_image_id_uses_digits_when_present() -> None:
    assert infer_image_id(Path("shelf_0042.png"), 7) == 42
    assert infer_image_id(Path("plain_name.jpg"), 7) == 7


def test_run_py_writes_valid_coco_predictions(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    image_path = image_dir / "sample_12.png"
    Image.new("RGB", (20, 10), color="white").save(image_path)

    output_path = tmp_path / "predictions.json"
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run.py"),
            "--input",
            str(image_dir),
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == [
        {
            "image_id": 12,
            "category_id": 1,
            "bbox": [5, 2, 10, 5],
            "score": 0.5,
        }
    ]
