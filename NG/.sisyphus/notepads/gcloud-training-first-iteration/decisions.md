# Decisions: GCE GPU Bootstrap (Task 2)

## Decision 1: Use Image Family for First Iteration
- **Chosen**: `--image-family=common-cu128-ubuntu-2204-nvidia-570` + `--image-project=deeplearning-platform-release`
- **Rationale**: This is the base GPU image with CUDA 12.8 + driver 570. PyTorch is installed separately via pip/uv. Image family gives latest patch for first iteration. Pinning only needed when reproducibility across cluster creation matters.
- **Alternative rejected**: Pinned `--image=common-cu128-ubuntu-2204-nvidia-570-vYYYYMMDD` — adds fragility without benefit for single-VM first iteration.

## Decision 2: Remote Bootstrap Installs uv + Python 3.11
- **Chosen**: Install `uv` via official installer, then `uv python install 3.11`, create venv, sync deps
- **Rationale**: DLV images ship Python 3.10 as base (see Base-cu128 note: Python 3.10). Plan task 2 requires Python 3.11. `uv` is the repo's standard package manager. Installing pip wheels for build tools (`git`, `curl`) covers wheel compilation needs.

## Decision 3: GPU Verification Uses nvidia-smi + torch CUDA Probe
- **Chosen**: SSH bootstrap checks `nvidia-smi` then `python3 -c "import torch; print(torch.cuda.is_available())"`
- **Rationale**: Official DLV docs recommend `nvidia-smi` as the post-boot driver probe. The torch CUDA probe confirms cuDNN + drivers + CUDA runtime are all coherent.

## Decision 4: Sync detector runs through the canonical `runs/` GCS prefix
- **Chosen**: Mirror `artifacts/models/detector/` with `gcloud storage rsync` to and from `$GCS_RUNS`, and keep `artifacts/run_manifest.json` as a required standalone copy target under `artifacts/manifests/run_manifest.json`.
- **Rationale**: The first-iteration run manifest points at detector run artifacts under `artifacts/models/detector/yolov8m-search-baseline`, so sync coverage must include that tree explicitly and handle manifest pull as well as push.

## Decision 5: Verify Task 4 via docs command-contract tests, not live archive regeneration
- **Chosen**: Add a narrow unit test that asserts the operator docs contain the exact ingest, split, audit, GCS upload, and VM rsync commands plus the fixture-vs-real-data warning.
- **Rationale**: Task 4 is about codifying the first cloud operator path while keeping CI deterministic and independent from the real competition archives.

## Decision 6: Keep detector train mode defaulting to smoke while layering real Ultralytics training beside it
- **Chosen**: Add `--mode {smoke,real}` with default `smoke`, keep the existing placeholder payload unchanged, and add a separate real-mode summary with `training.placeholder=false`, runtime timing, dataset-prep provenance, and normalized weights paths.
- **Rationale**: Existing smoke tests and downstream version guards still need the deterministic placeholder contract, while Task 6 requires a real training path that remains compatible with the fixed `yolov8m-search-baseline` artifact layout.

## Decision 7: Stub detector inference in real-eval tests, not scorer integration
- **Chosen**: Patch `_run_ultralytics_inference` in detector evaluation tests and still run the emitted predictions through `score_predictions`.
- **Rationale**: This keeps CI deterministic and GPU-free while still proving the real-mode JSON contract, category remapping, and scoring compatibility that downstream gates depend on.

## Task 10 execution decisions (2026-03-21)
- Followed the plan command order up through doctor, preflight, and real bootstrap.
- Stopped before repo copy, dataset prep, and training because the bucket was missing and the repo-local manifests were still fixture-only, which would violate the task guardrail against claiming a real run from fixture data.
- Treated the accelerator-listing flag issue as an unavoidable local CLI adjustment and verified via `gcloud help compute accelerator-types list` that this environment expects a zone filter expression instead of `--zones`.
