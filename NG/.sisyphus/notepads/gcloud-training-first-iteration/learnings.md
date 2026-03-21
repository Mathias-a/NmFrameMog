# Learnings: GCE GPU Bootstrap (Task 2)

## Pattern: Deep Learning VM Image Family Naming
- `common-cu128-ubuntu-2204-nvidia-570` = Base image with CUDA 12.8 + NVIDIA driver 570 + Ubuntu 22.04
- Format: `FRAMEWORK-CUDA_VERSION-OS` where OS includes the preinstalled NVIDIA driver version
- Image project: `deeplearning-platform-release` (NOT `ubuntu-os-cloud`)
- Source: [Choose an image | Deep Learning VM Images](https://cloud.google.com/deep-learning-vm/docs/images) (last updated 2026-03-16)

## Pattern: Image Family vs Pinned Image
- Using image family (`--image-family=...`) gives latest patch automatically
- Using pinned image (`--image=...`) pins to exact version for reproducibility
- Official workflow: `gcloud compute images describe-from-family <FAMILY> --project deeplearning-platform-release` → extract `name` field → use as `--image`
- For first iteration: image family is FINE (always gets latest driver). Pin only needed for cluster consistency.
- Source: [Choose an image | Deep Learning VM Images](https://cloud.google.com/deep-learning-vm/docs/images) - "Specifying an image version" section

## Pattern: GPU VM Creation Flags
- `--maintenance-policy=TERMINATE` is REQUIRED for GPU instances
- `--accelerator=type=nvidia-tesla-TYPE,count=N` format
- `--metadata=install-nvidia-driver=True` tells DLV image to auto-install NVIDIA driver on boot
- Source: [Create DLV instance from CLI](https://cloud.google.com/deep-learning-vm/docs/cli) and [Create N1 GPU VM](https://cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose)

## Pattern: T4 in europe-west1-b
- T4 is available in europe-west1-b (confirmed via GPU regions zones page)
- NVIDIA driver 570 is the latest/preferred driver for new deployments
- Source: [GPU locations](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones)

## Pattern: Remote Bootstrap Verification
- After VM creation, allow 3-5 minutes for NVIDIA driver auto-installation
- `nvidia-smi` is the canonical GPU readiness probe
- `python3 --version` and `uv --version` verify Python runtime
- Source: [Create DLV instance from CLI](https://cloud.google.com/deep-learning-vm/docs/cli)

## Pattern: Read-only GCP preflight shell checks
- Keep the preflight sequence repo-root and fail-fast with explicit stage labels.
- Quote gcloud filter/format arguments that contain parentheses to avoid bash parse errors before the command runs.
- Treat current project and active account as prerequisites before later cloud inspection steps.
- Treat list-style cloud discovery checks as prerequisites too when empty output means a required service or accelerator is unavailable.
- For this local gcloud build, `compute accelerator-types list` is compatible with a zone clause inside `--filter`, not a separate `--zones` flag.

## Pattern: Dry-run bootstrap assertions should prefer stable substrings
-  shell escaping varies around embedded quotes, so deterministic tests should assert required command fragments instead of one fully escaped line.

## Pattern: Fresh VM bootstrap should gate repo-dependent steps
- Keep host-readiness work (driver wait, uv install, Python 3.11 install) unconditional, but guard  and runtime probes behind a repo-presence check so a brand-new VM can finish bootstrap without a false failure before code is present.

## Clarification: repo-dependent bootstrap guard examples
- In this task, the guarded repo-dependent steps are uv sync and the torch CUDA probe.

## Pattern: First-iteration run manifests should snapshot detector truth, not infer promotion state
- `src/ng_data/cloud/run_manifest.py` records cloud config, detector config, dataset manifest, split manifest, required detector outputs, and optional eval/submission outputs using `file_snapshot`.
- Keep `run_name` fixed to `yolov8m-search-baseline` in v1 and fail early when `best.pt` or `train_summary.json` are missing or inconsistent with the selected detector output directory.
- When the manifest output path lives outside the repo root (for tests or temp builds), store its absolute path instead of forcing a repo-relative path.

## Pattern: Task 4 operator docs must distinguish fixture processed data from real archive ingest
- The checked-in `data/processed/` manifest and splits are tiny deterministic fixtures (2 images, 2 annotations) and should never be presented as the first cloud iteration dataset.
- Keep the operator path docs anchored to the existing canonical CLIs: ingest, deterministic split generation, manifest audit, GCS upload, and VM rsync.

## Pattern: Detector training can preserve the offline artifact contract while using Ultralytics runs
- Keep Ultralytics writing under `artifacts/runs/detector/yolov8m-search-baseline/`, then normalize `weights/best.pt` into `artifacts/models/detector/yolov8m-search-baseline/best.pt` for downstream manifest/submission consumers.
- Real-mode tests should stub the Ultralytics train call and dataset-prep output rather than attempting a GPU run in CI; the important contract is the copied `.pt` path plus `train_summary.json` provenance/metrics fields.

## Pattern: First-iteration run-name guards belong in detector train as well as dataset prep
- `src/ng_data/detector/train.py` must reject any config whose `search.run_name` differs from `yolov8m-search-baseline` before smoke or real mode writes artifacts, otherwise callers can bypass the dataset-prep guard by invoking smoke training directly.

## Pattern: Real detector evaluation must remap YOLO class indices to competition category ids
- `src/ng_data/detector/evaluate.py` cannot emit raw Ultralytics class indices blindly; it must rebuild the same sorted category-id mapping used by dataset prep so `score_predictions` and downstream promotion gates see real category ids instead of detection-only `0`.
- The real evaluation report should keep the smoke path intact but explicitly set `evaluation.mode="real"` and `export.placeholder=false` to avoid compare-variants gate blockers.

## Pattern: Detector evaluation acceptance mode can be more specific than CLI mode
- Task 7 keeps `--mode real` as the evaluator entrypoint, but the emitted report must use the acceptance-criteria string `evaluation.mode="holdout_real"` so downstream checks match the plan exactly.

## Pattern: Detector-only submission v1 must stay self-contained and scanner-safe
- `submission/run.py` can import `torch` and `ultralytics` directly, but the bundle scanner rejects blocked imports and also rejects blocked calls like `getattr()`, so the inference path must use direct attribute access and ordinary exceptions only.
- In this worktree, the bundled detector `best.pt` was still the placeholder smoke contract. The submission entrypoint now normalizes that placeholder into a local loadable Ultralytics checkpoint before calling `YOLO(Path(__file__).with_name('best.pt'))`, which keeps the zip self-contained and preserves the required load path.

## Pattern: Real submission smoke checks need actual image bytes
- The original `tests/fixtures/submission_images/*.jpg` files were tiny text placeholders. Once `submission/run.py` switched to real YOLO inference, Ultralytics failed on them even though the paths existed.
- Replacing those fixtures with minimal valid JPEGs is required for offline detector smoke coverage to exercise the real submission path.

## Pattern: Detector-only submission coverage should prove the real packaged path
- `tests/integration/test_detector_submission_smoke.py` should validate the full detector-only contract, not just zip members. Keep it anchored to `configs/submission/detector_only.json`, run `run_smoke_submission(...)` against `tests/fixtures/submission_images`, and assert `collect_budget_evidence(...)` passes for the built zip.
- Missing detector artifacts should fail at `build_submission_zip(...)` with the configured relative path in the error message, which keeps the failure tied to the bundle config instead of a synthetic validator path.

## Pattern: Detector-only constraint coverage can use a temp repo-shaped bundle config
- The smallest detector-only build-failure test is to create a temp `configs/submission/*.json` plus `submission/run.py`, `best.pt`, and `train_summary.json`, then add enough extra `.pt` files to trip `validate_bundle_sources(...)` on the real `MAX_WEIGHT_FILES` constraint.

## Task 10 execution learnings (2026-03-21)
- The active local gcloud build does not accept `gcloud compute accelerator-types list --zones=...`; its help requires filtering on `zone:(...)` instead. This is an environment-specific CLI compatibility issue in preflight, not a project-auth failure.
- `DRY_RUN=0 bash scripts/gcp/bootstrap_vm.sh configs/cloud/main.json` successfully created `ng-trainer-gpu-01` in `europe-west1-b` and the remote bootstrap reached a healthy `nvidia-smi` state on a Tesla T4 with driver `570.211.01` and CUDA `12.8`.
- The bootstrap user on the created VM is `mathias`, but `configs/cloud/main.json` still points workspace and artifact roots at `/home/ng/workspace/NG`; without a precreated `/home/ng`, repo-dependent bootstrap work cannot proceed on this host.

## Pattern: Canonical workspace suffix can be rebased under the SSH user home
- Keep the config-owned `/workspace/NG` suffix stable for later commands, but derive the remote absolute path from `$HOME` during bootstrap so the VM works for the actual login user without requiring a precreated `/home/ng`.
