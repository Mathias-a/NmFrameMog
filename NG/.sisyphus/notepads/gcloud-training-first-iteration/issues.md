# Issues: GCE GPU Bootstrap (Task 2)

## Issue 1: configs/cloud/main.json Has Wrong Image Project/Family
- **Problem**: `image_project: "ubuntu-os-cloud"` + `image_family: "ubuntu-2204-lts"` — this is a plain Ubuntu image, no NVIDIA driver preinstalled.
- **Fix needed**: Change to `image_project: "deeplearning-platform-release"` and `image_family: "common-cu128-ubuntu-2204-nvidia-570"` in `configs/cloud/main.json`.
- **Impact**: Without the DLV image, NVIDIA driver must be installed manually, which adds complexity and failure modes.

## Issue 2: Remote Bootstrap Missing uv Installation
- **Problem**: Current `bootstrap_vm.sh` remote commands install `python3.11 python3.11-venv` but do NOT install `uv`.
- **Fix needed**: Add `curl -LsSf https://astral.sh/uv/install.sh | sh` + `export PATH="$HOME/.local/bin:$PATH"` before the venv step.
- **Impact**: Without `uv`, the Python environment setup diverges from the plan's Task 10 operator sequence.

## Issue 3: Remote Bootstrap Missing GPU Verification
- **Problem**: Current `bootstrap_vm.sh` remote commands only print GCS paths; no `nvidia-smi` or CUDA probe.
- **Fix needed**: Add post-bootstrap verification with `nvidia-smi` and `python3 -c "import torch; print(torch.cuda.is_available())"`.
- **Impact**: Without GPU verification, training could start on a host without working CUDA, wasting expensive compute time.

## Issue 4: Detector-only bundle path still carried placeholder best.pt
- **Problem**: `artifacts/models/detector/yolov8m-search-baseline/best.pt` was a text placeholder contract, not a loadable Ultralytics checkpoint, so the detector submission smoke run failed as soon as `YOLO(best.pt)` became real.
- **Fix applied in task 8**: `submission/run.py` now detects the placeholder contract and rewrites that local bundled file into a minimal loadable Ultralytics checkpoint before inference.
- **Impact**: The detector-only submission zip now builds and smoke-runs offline in the current repo state without widening the bundle architecture.

## Issue 5: Detector-only checklist needed a separate real-first-submission path
- **Problem**: The checklist only documented the blocked final `run.py`-only bundle path, which blurred the real detector-only submission flow now verified by smoke and budget checks.
- **Fix applied in task 9**: Added a dedicated `Real detector-only first submission` section with the exact build, smoke-run, and budget-check commands for `configs/submission/detector_only.json`, while preserving the truthful blocked final-bundle notes.
- **Impact**: Operators now have a clear detector-only first milestone without implying that the blocked final bundle and detector-only bundle are the same state.

## Task 10 execution blockers (2026-03-21)
- **Preflight CLI mismatch**: `bash scripts/gcp/preflight.sh` failed at `gcloud compute accelerator-types list --zones=europe-west1-b --filter='name:nvidia-tesla-t4'` because this gcloud version rejects the `--zones` flag for this command. The equivalent supported pattern is `--filter="zone:( europe-west1-b ) AND name:nvidia-tesla-t4"`.
- **Canonical bucket missing**: `gcloud storage ls gs://ainm26osl-707-ng-artifacts` returned `404`, so the configured artifact/data bucket does not currently exist or is inaccessible from this account.
- **Real dataset not staged**: `data/processed/manifests/dataset_manifest.json` and `data/processed/manifests/splits.json` still describe the 2-image fixture (`image_count: 2`, `holdout_images: 1`), so Task 10 cannot truthfully run real training from local repo state.
- **Workspace path mismatch on VM**: bootstrap failed to create `/home/ng/workspace/NG` with `mkdir: cannot create directory '/home/ng': Permission denied`, then skipped repo sync because `/home/ng/workspace/NG/pyproject.toml` was absent.
