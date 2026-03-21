from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
WEIGHTS_ROOT = REPO_ROOT / "NG" / "yolov8x_product_pipeline" / "weights" / "yv8x_3fold_050ep_001"
DIST_ROOT = ROOT / "dist"
STAGING_ROOT = DIST_ROOT / "yv8x_3fold_050ep_001_submission"
ZIP_PATH = DIST_ROOT / "yv8x_3fold_050ep_001_submission.zip"


def copy_required_files() -> None:
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    DIST_ROOT.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ROOT / "run.py", STAGING_ROOT / "run.py")
    shutil.copy2(ROOT / "recognizer_model.py", STAGING_ROOT / "recognizer_model.py")
    shutil.copy2(WEIGHTS_ROOT / "detector_fold1_best.pt", STAGING_ROOT / "detector.pt")
    shutil.copy2(WEIGHTS_ROOT / "recognizer_fold2_best.pt", STAGING_ROOT / "recognizer.pt")


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STAGING_ROOT.iterdir()):
            archive.write(path, arcname=path.name)


def main() -> int:
    copy_required_files()
    build_zip()
    print(STAGING_ROOT)
    print(ZIP_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
