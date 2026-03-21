# Google Cloud Detector-Only First Iteration

## TL;DR
> **Summary**: Move the repo from smoke-only detector scaffolding to a real single-model Ultralytics detector workflow on GCE + GCS, then produce a contract-valid `dist/ng_detector_only_submission.zip` that is locally smoke/budget validated and ready for manual upload at the competition submit page.
> **Deliverables**:
> - GPU-ready GCE bootstrap and cloud preflight flow for `configs/cloud/main.json`
> - Real detector dataset-prep, train, and evaluate path that preserves the existing artifact contract under `artifacts/models/detector/yolov8m-search-baseline/`
> - Bundled `submission/run.py` that loads `best.pt` and emits COCO-format predictions
> - Upload-ready detector bundle in `dist/ng_detector_only_submission.zip` plus synced GCS artifacts and provenance manifest
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: 1 → 2 → 5 → 6 → 7 → 8 → 11

## Context
### Original Request
Create a concrete implementation and command plan that takes the current project iteration to Google Cloud training and a first working submission path as quickly as possible, with specific commands and design decisions.

### Interview Summary
- Assume the GCP project is ready enough that auth, APIs, billing, and CLI setup can be completed quickly.
- Optimize for the fastest path to a first submission rather than lowest cost or maximum polish.
- Lock the first iteration to **detector-only architecture**.
- Include the path up to an upload-ready artifact; actual website upload remains manual because the repo/docs only say “upload at the submit page” and do not expose an automatable upload contract.

### Metis Review (gaps addressed)
- Do **not** treat `configs/submission/final.json` or the current `submission/run.py` baseline as the first real release path; they are scaffold/truthful-blocked-state tooling only.
- Keep the current smoke path intact and add a parallel real-train/real-eval path instead of replacing the existing smoke contract blindly.
- Stop the MVP at a validated detector bundle (`configs/submission/detector_only.json`) staged for manual upload; do not invent undocumented endpoint APIs.
- Add blocking cloud preflight, GPU preflight, and submission-contract preflight before claiming the bundle is ready.

## Work Objectives
### Core Objective
Produce the smallest set of repo and operator changes required to train a real YOLOv8 detector on GCE, evaluate it locally/on-VM, bundle it into a detector-only submission zip, validate that zip offline, and stage it in GCS and `dist/` for manual submission.

### Deliverables
- `scripts/gcp/preflight.sh` that validates local repo state plus GCP readiness
- Updated `configs/cloud/main.json` + `scripts/gcp/bootstrap_vm.sh` for a GPU-usable VM bootstrap path using `deeplearning-platform-release` / `common-cu128-ubuntu-2204-nvidia-570`
- `src/ng_data/detector/prepare_dataset.py` using the canonical processed dataset + split manifests
- Extended detector training/evaluation commands that write non-placeholder artifacts
- Updated `submission/run.py` that performs real model inference using bundled `best.pt`
- Strengthened detector bundle tests and updated command checklist/docs for the detector-only path
- `src/ng_data/cloud/run_manifest.py`
- `artifacts/run_manifest.json`
- `dist/ng_detector_only_submission.zip`

### Definition of Done (verifiable conditions with commands)
- `uv run python -m src.ng_data.cli.doctor --root .` exits `0`.
- `uv run python -m src.ng_data.cloud.validate_config --config configs/cloud/main.json --project ainm26osl-707 --region europe-west1 --dry-run` prints `"status": "ok"`.
- `uv run python -m pytest -q tests/integration/test_gcp_layout_config.py tests/integration/test_detector_submission_smoke.py` exits `0`.
- `uv run python -m src.ng_data.detector.train --mode real --config configs/detector/yolov8m-search.json --manifest data/processed/manifests/dataset_manifest.json --splits data/processed/manifests/splits.json --output-dir artifacts/models/detector` writes `artifacts/models/detector/yolov8m-search-baseline/best.pt` and `train_summary.json` with `training.placeholder=false`.
- `uv run python -m src.ng_data.detector.evaluate --mode real --weights artifacts/models/detector/yolov8m-search-baseline/best.pt --split data/processed/annotations/instances.coco.json --out artifacts/eval/detector_holdout_metrics.json` writes a non-placeholder detector report and predictions file.
- `uv run python -m src.ng_data.submission.build_zip --config configs/submission/detector_only.json` writes `dist/ng_detector_only_submission.zip`.
- `uv run python -m src.ng_data.submission.smoke_run --zip dist/ng_detector_only_submission.zip --input tests/fixtures/submission_images --output artifacts/smoke/detector_only_predictions.json` exits `0`.
- `uv run python -m src.ng_data.submission.budget_check --zip dist/ng_detector_only_submission.zip --input tests/fixtures/submission_images --out artifacts/smoke/detector_only_budget.json` exits `0` with `"status": "pass"`.
- `gcloud storage ls gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/artifacts/releases/` shows the detector bundle after sync.

### Must Have
- Preserve the existing repo-first cloud pattern: one persistent GCE GPU VM + one canonical GCS namespace from `configs/cloud/main.json`.
- Keep detector runtime pinned to sandbox-compatible Ultralytics `8.1.0` `.pt` flow.
- Keep the current smoke contracts working while adding the real training/evaluation path.
- Keep detector artifacts under `artifacts/models/detector/yolov8m-search-baseline/` for v1.
- Use `configs/submission/detector_only.json` as the first upload-ready bundle target.
- Generate and sync a provenance manifest for every real run.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Do **not** add retrieval, classifier-in-bundle logic, ONNX export, Vertex AI, hyperparameter search, or multi-VM orchestration.
- Do **not** repurpose `configs/submission/final.json` as the first real detector artifact.
- Do **not** package placeholder weights or claim smoke-only artifacts are release-grade.
- Do **not** introduce blocked imports/calls into bundled Python files; `submission/run.py` must remain scanner-safe against `src/ng_data/submission/__init__.py` rules.
- Do **not** require website clicking/logging in as part of acceptance criteria.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: **TDD** using existing `pytest` infrastructure plus repo-native smoke/budget CLIs.
- QA policy: Every task includes at least one happy path and one failure/edge-case scenario.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: cloud/runtime foundation (1-4)
- cloud preflight
- GPU-ready bootstrap/config
- artifact sync + run manifest
- full-data ingest/split regeneration + staging

Wave 2: detector implementation core (5-8)
- Ultralytics dataset prep
- real detector training
- real detector evaluation
- real bundled inference in `submission/run.py`

Wave 3: bundle validation and operator handoff (9-11)
- detector-only bundle tests + checklist updates
- first GCE training run + artifact sync
- build upload-ready detector zip + local validation + release staging

### Dependency Matrix (full, all tasks)
| Task | Depends On | Blocks |
|---|---|---|
| 1 | - | 2, 3, 4, 10 |
| 2 | 1 | 10 |
| 3 | 1 | 10, 11 |
| 4 | 1 | 5, 6, 7, 10 |
| 5 | 4 | 6 |
| 6 | 5 | 7, 8, 10, 11 |
| 7 | 6 | 10, 11 |
| 8 | 6 | 9, 11 |
| 9 | 8 | 11 |
| 10 | 1, 2, 3, 4, 6, 7 | 11 |
| 11 | 3, 7, 8, 9, 10 | F1-F4 |

### Agent Dispatch Summary (wave → task count → categories)
| Wave | Task Count | Categories |
|---|---:|---|
| 1 | 4 | unspecified-high, quick |
| 2 | 4 | deep, unspecified-high |
| 3 | 3 | unspecified-high, writing |

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Add a blocking local-and-GCP preflight command path

  **What to do**: Create `scripts/gcp/preflight.sh` as the repo-native preflight entrypoint for the first cloud iteration. It must run in order: `uv run python -m src.ng_data.cli.doctor --root .`, `uv run python -m src.ng_data.cloud.validate_config --config configs/cloud/main.json --project ainm26osl-707 --region europe-west1 --dry-run`, `uv run python -m src.ng_data.cloud.print_paths --config configs/cloud/main.json`, `gcloud config get-value project`, `gcloud auth list --filter=status:ACTIVE`, `gcloud services list --enabled --filter='NAME:(compute.googleapis.com OR storage.googleapis.com)'`, `gcloud compute accelerator-types list --zones europe-west1-b --filter='name:nvidia-tesla-t4'`, and `gcloud storage ls gs://ainm26osl-707-ng-artifacts`. Make the script fail fast on any missing prerequisite and print a JSON-like summary or clearly labeled section output for each preflight stage. Keep it read-only.
  **Must NOT do**: Do not create buckets, enable APIs, or mutate quotas inside the preflight script. Do not hardcode any path outside `configs/cloud/main.json` except the explicit expected project/region values already tested in `tests/integration/test_gcp_layout_config.py`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: combines shell scripting, repo alignment, and test updates.
  - Skills: `[]` — no special skill is needed.
  - Omitted: `quality-check` — linting belongs after code changes, not as plan content.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2, 3, 4, 10 | Blocked By: none

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `AGENTS.md:17-21` — preferred workflow starts with `python -m src.ng_data.cli.doctor`.
  - Pattern: `src/ng_data/cli/doctor.py:12-56` — required project paths and exit behavior.
  - Pattern: `src/ng_data/cloud/validate_config.py:24-80` — canonical config validation CLI and JSON summary.
  - Pattern: `src/ng_data/cloud/print_paths.py:18-57` — canonical path rendering CLI.
  - Test: `tests/integration/test_gcp_layout_config.py:21-102` — expected validated project/region and rendered cloud paths.
  - Config: `configs/cloud/main.json:1-45` — project, region, zone, VM, bucket, and namespace values.
  - Doc: `scripts/gcp/README.md:1-7` — one persistent GCE GPU VM plus one canonical GCS namespace.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `bash scripts/gcp/preflight.sh` exits `0` when the local env and GCP project are ready.
  - [ ] `bash scripts/gcp/preflight.sh` exits non-zero and prints a clear failure when no active `gcloud` account is present.
  - [ ] `uv run python -m pytest -q tests/integration/test_gcp_layout_config.py tests/unit/test_project_structure.py` exits `0` after the preflight addition.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Happy path preflight
    Tool: Bash
    Steps: Run `bash scripts/gcp/preflight.sh` from repo root with valid gcloud auth and config.
    Expected: Every stage prints success, exits 0, and includes the configured project `ainm26osl-707`, region `europe-west1`, zone `europe-west1-b`, and bucket `gs://ainm26osl-707-ng-artifacts`.
    Evidence: .sisyphus/evidence/task-1-cloud-preflight.txt

  Scenario: Missing gcloud auth fails early
    Tool: Bash
    Steps: Run the script in an environment where `gcloud auth list --filter=status:ACTIVE` returns empty or stub the command in a test shell.
    Expected: Script exits non-zero before any VM command is suggested and prints a specific auth failure message.
    Evidence: .sisyphus/evidence/task-1-cloud-preflight-error.txt
  ```

  **Commit**: YES | Message: `feat(cloud): add first-iteration gcp preflight` | Files: `scripts/gcp/preflight.sh`, `tests/integration/test_gcp_layout_config.py`

- [x] 2. Make VM bootstrap produce a GPU-usable training host

  **What to do**: Update `configs/cloud/main.json` and `scripts/gcp/bootstrap_vm.sh` so the generated/real bootstrap flow uses `image_project=deeplearning-platform-release` and `image_family=common-cu128-ubuntu-2204-nvidia-570`, giving the single T4 host a preinstalled NVIDIA driver path. Keep the existing `uv run python -m src.ng_data.cloud.print_paths --config ... --format env` source of truth, but extend the remote bootstrap to install `git`, `curl`, `python3-pip`, and any minimal OS build tools needed by pip wheels, then verify `nvidia-smi`, `python3 --version`, and a torch CUDA probe after the repo environment is created. The remote bootstrap must also install `uv` so later tasks can create a Python 3.11 environment with `uv python install 3.11`.
  **Must NOT do**: Do not switch away from `europe-west1-b` or the single persistent VM shape for v1. If quota blocks creation, stop and resolve quota in `europe-west1-b` before continuing. Do not introduce Vertex AI logic.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: cloud bootstrap plus reproducible runtime checks.
  - Skills: `[]`
  - Omitted: `deploy` — this is training-host bootstrap, not Cloud Run deployment.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 10 | Blocked By: 1

  **References**:
  - Pattern: `scripts/gcp/bootstrap_vm.sh:4-52` — existing create/ssh bootstrap structure.
  - Config: `configs/cloud/main.json:18-30` — vm name, zone, machine type, accelerator, disk, image family/project, workspace paths.
  - API/Type: `src/ng_data/cloud/layout.py:65-94` — shell env names exported for scripts.
  - Test: `tests/integration/test_gcp_layout_config.py:75-83` — expected compute resource rendering.
  - Doc: `docs/submission.md:87-158` — sandbox Python/CUDA/runtime packages; pin against these.
  - External: `/ultralytics/ultralytics` docs queried via Context7 — YOLO `.pt` training/loading pattern to preserve runtime compatibility.

  **Acceptance Criteria**:
  - [ ] `DRY_RUN=1 bash scripts/gcp/bootstrap_vm.sh configs/cloud/main.json` prints a create command and an SSH bootstrap command that include GPU instance settings and runtime verification commands.
  - [ ] `DRY_RUN=0 bash scripts/gcp/bootstrap_vm.sh configs/cloud/main.json` succeeds on a quota-ready project and leaves a VM where `nvidia-smi`, `uv --version`, and a torch CUDA probe succeed over SSH.
  - [ ] `uv run python -m pytest -q tests/integration/test_gcp_layout_config.py` still passes if script expectations are asserted there or in a new integration test.

  **QA Scenarios**:
  ```
  Scenario: Dry-run bootstrap renders correct commands
    Tool: Bash
    Steps: Run `DRY_RUN=1 bash scripts/gcp/bootstrap_vm.sh configs/cloud/main.json`.
    Expected: Output contains `gcloud compute instances create ng-trainer-gpu-01`, `--zone=europe-west1-b`, `--accelerator=type=nvidia-tesla-t4,count=1`, `--image-project=deeplearning-platform-release`, `--image-family=common-cu128-ubuntu-2204-nvidia-570`, and a remote bootstrap sequence with `nvidia-smi` and `uv` installation/verification.
    Evidence: .sisyphus/evidence/task-2-bootstrap-dry-run.txt

  Scenario: GPU missing on VM fails clearly
    Tool: Bash
    Steps: Execute the remote bootstrap verification on a VM without functional GPU drivers or with broken CUDA.
    Expected: SSH bootstrap exits non-zero on `nvidia-smi` or CUDA probe and prints a clear message that the host is not training-ready.
    Evidence: .sisyphus/evidence/task-2-bootstrap-error.txt
  ```

  **Commit**: YES | Message: `fix(cloud): bootstrap a gpu-ready training vm` | Files: `scripts/gcp/bootstrap_vm.sh`, relevant integration test/docs

- [x] 3. Add canonical run-manifest generation and syncable artifact provenance

  **What to do**: Introduce `src/ng_data/cloud/run_manifest.py` as a lightweight run-manifest producer that writes `artifacts/run_manifest.json` for each real detector run. The manifest must include: project/region/zone from `configs/cloud/main.json`, run name `yolov8m-search-baseline`, detector config path, dataset manifest snapshot, split manifest snapshot, output artifact paths (`best.pt`, `train_summary.json`, evaluation report, submission zip when available), and timestamps in UTC. Then update `scripts/gcp/sync_artifacts.sh` so `push` and `pull` cover the detector model directory and release bundle path consistently with the manifest. Keep `artifacts/run_manifest.json` as a required sync target because the script already assumes it exists.
  **Must NOT do**: Do not overwrite or fake final release-manifest behavior from `src/ng_data/pipeline/final_train.py`. This is a first-iteration run manifest, not a promotion-grade final manifest.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: small new provenance module plus sync integration.
  - Skills: `[]`
  - Omitted: `writing` — implementation and tests dominate.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 10, 11 | Blocked By: 1

  **References**:
  - Pattern: `scripts/gcp/sync_artifacts.sh:1-46` — existing sync flow and manifest copy assumption.
  - Pattern: `src/ng_data/data/manifest.py:16-89` — `write_json`, `file_snapshot`, and directory snapshot helpers.
  - Pattern: `src/ng_data/pipeline/audit_release_artifacts.py:98-191` — artifact snapshot/provenance structure worth mirroring lightly.
  - Pattern: `src/ng_data/pipeline/publish_promotion_decision.py:103-170` — concise JSON decision/report writing style.
  - Config: `src/ng_data/cloud/layout.py:40-94` — canonical GCS prefixes including `run_manifests`, `checkpoints`, `release_bundles`.
  - Test: `tests/integration/test_gcp_layout_config.py:84-101` — expected canonical GCS locations.

  **Acceptance Criteria**:
  - [ ] `uv run python -m src.ng_data.cloud.run_manifest --config configs/cloud/main.json --detector-output artifacts/models/detector/yolov8m-search-baseline --out artifacts/run_manifest.json` writes a schema-valid manifest with artifact snapshots.
  - [ ] `DRY_RUN=1 bash scripts/gcp/sync_artifacts.sh push configs/cloud/main.json` includes commands for checkpoints/model artifacts, eval outputs, dist bundles, and `artifacts/run_manifest.json`.
  - [ ] A corresponding pull dry run mirrors the same canonical locations back into the repo.

  **QA Scenarios**:
  ```
  Scenario: Real-run provenance manifest is generated
    Tool: Bash
    Steps: Run the new manifest command after a detector train/eval cycle with existing artifact files present.
    Expected: `artifacts/run_manifest.json` exists, includes run name, cloud identifiers, file snapshots, and no placeholder-only claims.
    Evidence: .sisyphus/evidence/task-3-run-manifest.json

  Scenario: Missing detector artifact blocks manifest generation
    Tool: Bash
    Steps: Remove or rename `artifacts/models/detector/yolov8m-search-baseline/best.pt` and rerun the manifest command.
    Expected: Command exits non-zero and names the missing artifact path explicitly.
    Evidence: .sisyphus/evidence/task-3-run-manifest-error.txt
  ```

  **Commit**: YES | Message: `feat(cloud): add run manifest provenance and sync coverage` | Files: new manifest module/test, `scripts/gcp/sync_artifacts.sh`

- [x] 4. Regenerate full processed data and deterministic splits, then stage them for cloud use

  **What to do**: Make the operator path use the real competition archives, not the tiny fixture dataset currently present in `data/raw` and `data/processed`. Use `src/ng_data.data.ingest` and `src/ng_data.eval.make_splits` to regenerate `data/processed/manifests/dataset_manifest.json` and `data/processed/manifests/splits.json` from the full downloaded archives. Stage the resulting `data/raw` and `data/processed` tree through GCS, then mirror them into `/home/ng/workspace/NG/data/` on the VM. Use these exact commands as the canonical staging path: `gcloud storage cp data/raw/NM_NGD_coco_dataset.zip gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/raw/`, `gcloud storage cp data/raw/NM_NGD_product_images.zip gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/raw/`, `gcloud storage rsync --recursive data/processed gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/processed`, and on the VM `gcloud storage rsync --recursive gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data /home/ng/workspace/NG/data`.
  **Must NOT do**: Do not train against the 2-image fixture manifest and call it “first iteration done.” Do not alter the manifest schema.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: data pipeline plumbing and deterministic command path.
  - Skills: `[]`
  - Omitted: `solve-challenge` — full challenge orchestration is broader than this targeted task.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5, 6, 7, 10 | Blocked By: 1

  **References**:
  - Pattern: `docs/overview.md:11-39` — official data download expectations and archive names.
  - Pattern: `src/ng_data/data/ingest.py:653-697` — canonical ingest CLI and defaults.
  - Pattern: `configs/data/main.json:1-19` — processed layout and raw archive names.
  - Pattern: `src/ng_data/eval/make_splits.py:20-47` — deterministic split manifest CLI.
  - Pattern: `src/ng_data/eval/splits.py:258-338` — how split manifests are generated and written.
  - Pattern: `src/ng_data/data/manifest.py:200-255` — manifest auditing logic to prove processed data integrity.
  - Current State: `data/processed/manifests/dataset_manifest.json:1-64` and `splits.json:1-60` — currently tiny/fixture-like and unsuitable for real training.

  **Acceptance Criteria**:
  - [ ] `uv run python -m src.ng_data.data.ingest --config configs/data/main.json --raw data/raw --processed data/processed` exits `0` against the full downloaded archives.
  - [ ] `uv run python -m src.ng_data.eval.make_splits --config configs/data/splits.json --manifest data/processed/manifests/dataset_manifest.json` exits `0` and writes a non-fixture `splits.json`.
  - [ ] `uv run python -c "from src.ng_data.data.manifest import audit_dataset_manifest; import json; print(json.dumps(audit_dataset_manifest('data/processed/manifests/dataset_manifest.json'), indent=2, sort_keys=True))"` exits `0`.
  - [ ] A documented sync command moves the processed dataset to the VM or GCS data prefix before training.

  **QA Scenarios**:
  ```
  Scenario: Full archive ingest and split generation succeed
    Tool: Bash
    Steps: Run the ingest command, then the split-generation command, then audit the manifest.
    Expected: Manifest and splits reference the full processed dataset paths, counts are larger than the fixture-only values, and audit passes.
    Evidence: .sisyphus/evidence/task-4-data-ingest.txt

  Scenario: Missing archive fails deterministically
    Tool: Bash
    Steps: Remove `data/raw/NM_NGD_coco_dataset.zip` and rerun the ingest command.
    Expected: Command exits non-zero with `Missing required archive` and does not emit a partial manifest.
    Evidence: .sisyphus/evidence/task-4-data-ingest-error.txt
  ```

  **Commit**: YES | Message: `docs(data): codify full ingest and split staging path` | Files: `scripts/gcp/README.md`, `docs/submission-checklist.md`

- [x] 5. Add a real YOLO dataset-preparation path from canonical processed manifests

  **What to do**: Implement `src/ng_data/detector/prepare_dataset.py` to convert the canonical processed COCO annotations plus split manifest into a YOLO training layout consumable by Ultralytics 8.1.0. The module must: read `data/processed/manifests/dataset_manifest.json`, `data/processed/manifests/splits.json`, and `data/processed/annotations/instances.coco.json`; create a deterministic workspace under `artifacts/runs/detector/yolov8m-search-baseline/dataset/`; generate YOLO label files; and write a plain-text dataset YAML with keys `path`, `train`, `val`, `test`, `names`, and `nc`. Lock the split policy for v1 as: `train = folds[0].train_image_ids`, `val = folds[0].val_image_ids`, and `test = holdout.image_ids`. Build `names` from `data/processed/categories.json`, sorted by category `id`, so YOLO class indices remain aligned with competition `category_id` values. Keep image references stable and absolute for training-time robustness on the VM.
  **Must NOT do**: Do not change the submission bundle format. Do not add retrieval or classifier prep here. Do not rely on fixture-only paths or manual label editing.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: this is the main translation layer from repo data contracts to Ultralytics training layout.
  - Skills: `[]`
  - Omitted: `quality-check` — defer to implementation review.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 6 | Blocked By: 4

  **References**:
  - Pattern: `src/ng_data/data/ingest.py:198-244` — normalized COCO image/category structures already used in repo data handling.
  - Pattern: `src/ng_data/eval/splits.py:258-338` — deterministic split manifest structure and `holdout`/`folds` semantics.
  - Pattern: `src/ng_data/detector/config.py:132-196` — detector runtime must stay Ultralytics 8.1.0 + `.pt`.
  - Current State: `configs/detector/yolov8m-search.json:1-21` — run name, image size, batch size, epochs, patience, and weight filename.
  - External: `/ultralytics/ultralytics` docs — `YOLO("...pt").train(data="dataset.yaml", epochs=..., imgsz=...)` and COCO-to-YOLO conversion guidance.
  - Guardrail: `docs/submission.md:121-177` — stay on direct `.pt` path for first iteration.

  **Acceptance Criteria**:
  - [ ] `uv run python -m src.ng_data.detector.prepare_dataset --config configs/detector/yolov8m-search.json --manifest data/processed/manifests/dataset_manifest.json --splits data/processed/manifests/splits.json --out artifacts/runs/detector/yolov8m-search-baseline/dataset` exits `0`.
  - [ ] The command writes a deterministic dataset config file plus YOLO labels/image references for `train`, `val`, and `test` using `folds[0]` + `holdout` exactly.
  - [ ] Re-running the command without changing inputs leaves file snapshots unchanged.
  - [ ] New unit/integration tests assert that missing source manifests or mismatched split manifests fail fast.

  **QA Scenarios**:
  ```
  Scenario: Deterministic YOLO dataset prep
    Tool: Bash
    Steps: Run the dataset-prep command twice against the same processed dataset and compare snapshots/hashes of the output directory.
    Expected: Outputs are byte-for-byte stable, and the generated dataset config points to `folds[0].train_image_ids` for train, `folds[0].val_image_ids` for val, and `holdout.image_ids` for test.
    Evidence: .sisyphus/evidence/task-5-detector-dataset-prep.txt

  Scenario: Split/manifest mismatch is rejected
    Tool: Bash
    Steps: Tamper with `source_manifest` or `source_annotations` in a copy of `splits.json` and rerun the prep command.
    Expected: Command exits non-zero with a specific mismatch message before writing partial YOLO data.
    Evidence: .sisyphus/evidence/task-5-detector-dataset-prep-error.txt
  ```

  **Commit**: YES | Message: `feat(detector): add yolo dataset preparation path` | Files: `src/ng_data/detector/prepare_dataset.py`, detector prep tests, `scripts/gcp/README.md`

- [x] 6. Replace placeholder detector training with a real Ultralytics train path while preserving artifact contracts

  **What to do**: Extend `src/ng_data/detector/train.py` so it supports both the existing smoke contract and a new real training mode. Add a `--mode {smoke,real}` argument defaulting to `smoke` to preserve old tests, and implement `real` using the exact Ultralytics call shape `YOLO(config.model.weights).train(data=prepared_dataset_yaml, epochs=config.search.epochs, imgsz=config.search.image_size, batch=config.search.batch_size, device=config.search.device, patience=config.search.patience, project='artifacts/runs/detector', name=config.search.run_name, exist_ok=True, verbose=False)`. Before that, update `pyproject.toml` and `uv.lock` so the repo pins `ultralytics==8.1.0`, `torch==2.6.0`, and `torchvision==0.21.0` for local implementation/test environments. After training, normalize the Ultralytics output by copying `artifacts/runs/detector/yolov8m-search-baseline/weights/best.pt` to `artifacts/models/detector/yolov8m-search-baseline/best.pt`, then emit `train_summary.json` with the same high-level contract keys already used by downstream code (`config_path`, `runtime`, `output_artifacts`, `output_dir`, `training`, etc.), but with `training.placeholder=false`, a real runtime summary, dataset-prep provenance, and training metrics/paths. Keep `run_name` fixed to `yolov8m-search-baseline` in this first iteration to match `configs/submission/detector_only.json` and existing tests.
  **Must NOT do**: Do not break `tests/integration/test_detector_train_smoke.py`. Do not save full-model pickles incompatible with the sandbox; save the Ultralytics-compatible `.pt` artifact that `YOLO(path/to/best.pt)` can load.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: this is the core model-training implementation with compatibility constraints.
  - Skills: `[]`
  - Omitted: `solve-challenge` — this task is detector-specific, not the whole competition stack.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 7, 8, 10, 11 | Blocked By: 5

  **References**:
  - Pattern: `src/ng_data/detector/train.py:32-303` — current CLI, validation, output directory, and summary contract.
  - Test: `tests/integration/test_detector_train_smoke.py:14-88` — smoke behavior that must remain intact.
  - Pattern: `src/ng_data/classifier/baseline.py:772-825` — repo precedent for `best.pt` + `train_summary.json` output and artifact provenance.
  - Pattern: `tests/integration/test_detector_export_version_guard.py:25-88` — version/export pinning that must remain true.
  - Config: `configs/detector/yolov8m-search.json:1-21` — source of epochs, imgsz, batch, patience, device, run name.
  - External: `/ultralytics/ultralytics` docs — official `YOLO("weights.pt").train(data=..., epochs=..., imgsz=...)` pattern.

  **Acceptance Criteria**:
  - [ ] `uv sync --dev` succeeds locally after the dependency pin update.
  - [ ] `uv run python -m src.ng_data.detector.train --mode smoke --config configs/detector/yolov8m-search.json --manifest data/processed/manifests/dataset_manifest.json --splits data/processed/manifests/splits.json --output-dir artifacts/models/detector` still passes the existing smoke tests.
  - [ ] `uv run python -m src.ng_data.detector.train --mode real --config configs/detector/yolov8m-search.json --manifest data/processed/manifests/dataset_manifest.json --splits data/processed/manifests/splits.json --output-dir artifacts/models/detector` writes `best.pt` and `train_summary.json` under `artifacts/models/detector/yolov8m-search-baseline/`.
  - [ ] The real-mode `train_summary.json` contains `training.placeholder: false` and references the actual output weight path.
  - [ ] New tests cover both smoke compatibility and real-mode contract generation.

  **QA Scenarios**:
  ```
  Scenario: Real training writes upload-intent artifacts
    Tool: Bash
    Steps: Run `uv sync --dev`, then run the real training command against the prepared full dataset on a GPU-enabled environment.
    Expected: `artifacts/models/detector/yolov8m-search-baseline/best.pt` exists, `train_summary.json` exists, and the summary marks `training.placeholder=false`.
    Evidence: .sisyphus/evidence/task-6-detector-train-real.txt

  Scenario: Invalid runtime/export config fails before training
    Tool: Bash
    Steps: Run the real training command with a copied detector config modified to `version=8.2.0` or `export_format=onnx`.
    Expected: Command exits non-zero with the same unsupported runtime/version message asserted in existing guard tests.
    Evidence: .sisyphus/evidence/task-6-detector-train-real-error.txt
  ```

  **Commit**: YES | Message: `feat(detector): add real ultralytics training mode` | Files: `src/ng_data/detector/train.py`, detector train tests, `pyproject.toml`, `uv.lock`

- [x] 7. Add a non-placeholder detector evaluation path that downstream gates can trust

  **What to do**: Extend `src/ng_data/detector/evaluate.py` so it can evaluate real detector predictions in addition to the current smoke placeholder contract. Keep smoke mode intact, but add a real evaluation branch that loads `best.pt`, runs inference against the holdout annotation file in `data/processed/annotations/instances.coco.json`, converts predictions into the repo’s COCO prediction payload, scores them with `src/ng_data.eval.score.score_predictions`, and writes `artifacts/eval/detector_holdout_metrics.json` plus `artifacts/eval/detector_holdout_metrics.predictions.json`. Ensure the real report does **not** set `export.placeholder=true` or `evaluation.mode=smoke`; this is necessary because `src/ng_data/pipeline/compare_variants.py` explicitly blocks those values from promotion readiness.
  **Must NOT do**: Do not remove smoke evaluation support. Do not emit detection-only placeholder boxes for the real path.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: model loading/inference + scoring integration.
  - Skills: `[]`
  - Omitted: `quality-check` — keep focus on detector evaluation functionality.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 10, 11 | Blocked By: 6

  **References**:
  - Pattern: `src/ng_data/detector/evaluate.py:35-419` — current smoke CLI shape and output structure.
  - Test: `tests/integration/test_detector_evaluate_smoke.py:23-124` — smoke-path expectations that must remain valid.
  - Pattern: `src/ng_data/eval/score.py:81-277` — ground-truth loading, prediction validation, and scoring routine.
  - Pattern: `src/ng_data/pipeline/compare_variants.py:137-172` — real evaluation must avoid `mode=smoke`, `placeholder=true`, and `category_id=0`-only outputs.
  - External: `/ultralytics/ultralytics` docs — iterate `result.boxes.xyxy`, `result.boxes.cls`, `result.boxes.conf`, convert to COCO `[x, y, w, h]`.

  **Acceptance Criteria**:
  - [ ] Existing smoke evaluation test continues to pass unchanged.
  - [ ] `uv run python -m src.ng_data.detector.evaluate --mode real --weights artifacts/models/detector/yolov8m-search-baseline/best.pt --split data/processed/annotations/instances.coco.json --out artifacts/eval/detector_holdout_metrics.json` exits `0` after a real training run.
  - [ ] The real evaluation report contains `schema_version: 1`, `evaluation.mode: holdout_real`, `export.placeholder: false`, and a valid predictions export path.
  - [ ] `score_predictions` accepts the emitted predictions without validation errors.

  **QA Scenarios**:
  ```
  Scenario: Real detector evaluation produces promotion-usable evidence
    Tool: Bash
    Steps: Run the real evaluation command after completing a real training run.
    Expected: Metrics JSON and predictions JSON are written, both validate, and the metrics report does not contain smoke-only markers.
    Evidence: .sisyphus/evidence/task-7-detector-eval-real.json

  Scenario: Missing weights file fails deterministically
    Tool: Bash
    Steps: Point `--weights` at a nonexistent path and rerun the command.
    Expected: Command exits non-zero with a message naming the missing weights file.
    Evidence: .sisyphus/evidence/task-7-detector-eval-real-error.txt
  ```

  **Commit**: YES | Message: `feat(detector): add real evaluation path` | Files: `src/ng_data/detector/evaluate.py`, detector evaluation tests, `scripts/gcp/README.md`

- [x] 8. Turn `submission/run.py` into real detector inference while staying sandbox-safe

  **What to do**: Replace the current deterministic fake-box logic in `submission/run.py` with minimal real inference over bundled `best.pt`. Keep imports restricted to sandbox-safe packages and `pathlib`/`json`/`argparse`; do not import from `src/ng_data/submission` because that module uses blocked imports such as `subprocess`. Implement the runtime using `YOLO(Path(__file__).with_name('best.pt'))`, then for each input image call prediction with fixed v1 parameters `imgsz=960`, `conf=0.05`, `iou=0.6`, `device=0 if torch.cuda.is_available() else 'cpu'`, and `verbose=False`. Emit a JSON array of predictions with `image_id`, YOLO-predicted `category_id`, COCO `bbox` (`x,y,w,h` converted from `xyxy`), and `score`. Preserve `build_parser`, `list_image_files`, and `infer_image_id` shape where useful, but make prediction generation real.
  **Must NOT do**: Do not introduce blocked imports listed in `docs/submission.md` and enforced by `src/ng_data/submission/__init__.py`. Do not rely on network or pip installs. Do not import any helper module from outside `run.py` for v1 unless that helper file is explicitly added to `configs/submission/detector_only.json` and covered by bundle tests.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: inference implementation under strict sandbox constraints.
  - Skills: `[]`
  - Omitted: `deploy` — this is submission runtime logic, not service deployment.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 9, 11 | Blocked By: 6

  **References**:
  - Current State: `submission/run.py:1-81` — current fake prediction implementation to replace.
  - Guardrail: `docs/submission.md:29-61` — required `run.py --input --output` contract and output schema.
  - Guardrail: `docs/submission.md:180-225` — blocked imports/calls and root-zip requirement.
  - Enforcement: `src/ng_data/submission/__init__.py:245-347` — AST scanner and bundle-source validation.
  - Test: `tests/integration/test_detector_submission_smoke.py:13-24` — detector bundle structure.
  - External: `/ultralytics/ultralytics` docs — official custom `.pt` loading and result box iteration.

  **Acceptance Criteria**:
  - [ ] `uv run python -m src.ng_data.submission.build_zip --config configs/submission/detector_only.json` succeeds with the updated `submission/run.py`.
  - [ ] `uv run python -m src.ng_data.submission.smoke_run --zip dist/ng_detector_only_submission.zip --input tests/fixtures/submission_images --output artifacts/smoke/detector_only_predictions.json` exits `0`.
  - [ ] `uv run python -m pytest -q tests/integration/test_submission_rejects_blocked_imports.py tests/integration/test_detector_submission_smoke.py` exits `0`.
  - [ ] The generated predictions JSON passes `validate_prediction_payload` shape checks.

  **QA Scenarios**:
  ```
  Scenario: Bundled detector zip performs real inference offline
    Tool: Bash
    Steps: Build `dist/ng_detector_only_submission.zip` and run `uv run python -m src.ng_data.submission.smoke_run --zip dist/ng_detector_only_submission.zip --input tests/fixtures/submission_images --output artifacts/smoke/detector_only_predictions.json`.
    Expected: Smoke run exits 0 and writes schema-valid predictions produced from the bundled model file.
    Evidence: .sisyphus/evidence/task-8-submission-run-real.json

  Scenario: Blocked import is rejected at bundle-build time
    Tool: Bash
    Steps: Temporarily add a blocked import such as `import os` or `import yaml` to `submission/run.py` in a failing test fixture and run the detector bundle build.
    Expected: Build exits non-zero with `Blocked import` from the submission scanner.
    Evidence: .sisyphus/evidence/task-8-submission-run-real-error.txt
  ```

  **Commit**: YES | Message: `feat(submission): load bundled detector weights for inference` | Files: `submission/run.py`, detector submission tests, `docs/submission-checklist.md`

- [x] 9. Strengthen detector-only bundle tests and update the operator checklist around the real path

  **What to do**: Expand the detector submission integration coverage beyond zip-member presence. Keep `tests/integration/test_detector_submission_smoke.py`, but add or update tests so the detector-only bundle path proves: the zip contains `run.py`, `best.pt`, and `train_summary.json`; `run_smoke_submission` succeeds against fixture images; `budget_check` passes; and the build fails when the detector artifact path is missing or the bundle exceeds scanner constraints. Then update `docs/submission-checklist.md` to add a separate “real detector-only first submission” section with the exact detector-only commands, while preserving the existing truthful-blocked-state final bundle notes.
  **Must NOT do**: Do not delete the current blocked-state final checklist. Do not claim `configs/submission/final.json` is the same milestone as `configs/submission/detector_only.json`.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: checklist update plus test adjustments.
  - Skills: `[]`
  - Omitted: `quality-check` — not central to the task itself.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 11 | Blocked By: 8

  **References**:
  - Test: `tests/integration/test_detector_submission_smoke.py:13-24` — existing detector-only zip membership assertion.
  - Test: `tests/integration/test_final_submission_bundle.py:16-45` — pattern for build + smoke + budget assertions.
  - Enforcement: `src/ng_data/submission/__init__.py:333-458` — bundle build and smoke-run entrypoints.
  - Config: `configs/submission/detector_only.json:1-9` — detector-only zip contents and canonical artifact path.
  - Doc: `docs/submission-checklist.md:1-35` — current blocked final bundle checklist that must remain truthful.

  **Acceptance Criteria**:
  - [ ] `uv run python -m pytest -q tests/integration/test_detector_submission_smoke.py tests/integration/test_final_submission_bundle.py` exits `0`.
  - [ ] New/updated detector submission tests assert smoke-run success and budget-check pass for the detector-only bundle.
  - [ ] `docs/submission-checklist.md` contains two clearly separated paths: blocked final bundle and real detector-only first submission.

  **QA Scenarios**:
  ```
  Scenario: Detector bundle tests pass on real artifact path
    Tool: Bash
    Steps: Run the detector submission integration tests after generating a real `best.pt` and `train_summary.json`.
    Expected: Tests pass and assert both bundle contents and smoke/budget success.
    Evidence: .sisyphus/evidence/task-9-detector-bundle-tests.txt

  Scenario: Missing best.pt causes detector bundle test/build failure
    Tool: Bash
    Steps: Remove `artifacts/models/detector/yolov8m-search-baseline/best.pt` and rerun the detector bundle test or build command.
    Expected: The test/build fails with `Configured bundle source does not exist`.
    Evidence: .sisyphus/evidence/task-9-detector-bundle-tests-error.txt
  ```

  **Commit**: YES | Message: `test(submission): validate detector-only bundle end to end` | Files: detector submission tests, `docs/submission-checklist.md`

- [ ] 10. Execute the first real GCE training run and sync the resulting artifacts to GCS

  **What to do**: Run the exact first-iteration operator sequence on the persistent VM. From the local machine: authenticate/configure gcloud, run the new preflight, bootstrap the VM, copy the repo to the VM, and stage the full dataset. On the VM, install `uv`, install Python 3.11 with `uv`, create a project venv, sync the repo dependencies, reinstall CUDA-enabled torch/torchvision from the official cu124 index, run dataset prep, run real detector training, run real evaluation, write `artifacts/run_manifest.json`, and then sync checkpoints/eval outputs/dist/manifest back to GCS with `DRY_RUN=0 bash scripts/gcp/sync_artifacts.sh push configs/cloud/main.json`. Use these concrete commands:
  - Local: `uv run python -m src.ng_data.cli.doctor --root .`
  - Local: `bash scripts/gcp/preflight.sh`
  - Local: `DRY_RUN=0 bash scripts/gcp/bootstrap_vm.sh configs/cloud/main.json`
  - Local: `gcloud compute ssh ng-trainer-gpu-01 --project=ainm26osl-707 --zone=europe-west1-b`
  - VM: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - VM: `export PATH="$HOME/.local/bin:$PATH"`
  - VM: `uv python install 3.11`
  - VM: `uv venv --python 3.11`
  - VM: `. .venv/bin/activate`
  - VM: `uv sync --dev`
  - VM: `python -m pip install --upgrade pip`
  - VM: `python -m pip install --index-url https://download.pytorch.org/whl/cu124 --force-reinstall torch==2.6.0 torchvision==0.21.0`
  - VM: `uv run python -m src.ng_data.detector.prepare_dataset --config configs/detector/yolov8m-search.json --manifest data/processed/manifests/dataset_manifest.json --splits data/processed/manifests/splits.json --out artifacts/runs/detector/yolov8m-search-baseline/dataset`
  - VM: `uv run python -m src.ng_data.detector.train --mode real --config configs/detector/yolov8m-search.json --manifest data/processed/manifests/dataset_manifest.json --splits data/processed/manifests/splits.json --output-dir artifacts/models/detector`
  - VM: `uv run python -m src.ng_data.detector.evaluate --mode real --weights artifacts/models/detector/yolov8m-search-baseline/best.pt --split data/processed/annotations/instances.coco.json --out artifacts/eval/detector_holdout_metrics.json`
  - VM: `uv run python -m src.ng_data.cloud.run_manifest --config configs/cloud/main.json --detector-output artifacts/models/detector/yolov8m-search-baseline --evaluation artifacts/eval/detector_holdout_metrics.json --out artifacts/run_manifest.json`
  - VM or Local: `DRY_RUN=0 bash scripts/gcp/sync_artifacts.sh push configs/cloud/main.json`
  **Must NOT do**: Do not start with the tiny fixture dataset. Do not run real training until `nvidia-smi` and the CUDA probe succeed. Do not upload a bundle before local smoke/budget validation in Task 11.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: this is the concrete operator execution sequence across local and VM contexts.
  - Skills: `[]`
  - Omitted: `deploy` — still not a deployment workflow.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 11 | Blocked By: 1, 2, 3, 4, 6, 7

  **References**:
  - Config: `configs/cloud/main.json:3-45` — project, region, bucket, vm name, zone, machine type, accelerator.
  - Pattern: `scripts/gcp/bootstrap_vm.sh:7-52` — local-to-VM bootstrap flow.
  - Pattern: `scripts/gcp/sync_artifacts.sh:12-46` — push/pull sync targets and manifest copy.
  - Guardrail: `docs/submission.md:140-169` — exact sandbox-aligned training package pins and version compatibility caveats.
  - Test/Config guard: `tests/integration/test_detector_export_version_guard.py:25-88` and `tests/unit/test_sandbox_version_guards.py:7-58` — keep runtime pins aligned.

  **Acceptance Criteria**:
  - [ ] A first real VM run completes and leaves `artifacts/models/detector/yolov8m-search-baseline/best.pt` plus `artifacts/eval/detector_holdout_metrics.json`.
  - [ ] `artifacts/run_manifest.json` exists and references the detector and evaluation artifacts from that run.
  - [ ] `gcloud storage ls gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/artifacts/checkpoints/` and `.../artifacts/eval/` show synced outputs after the push.
  - [ ] The synced artifacts can be pulled back with `DRY_RUN=0 bash scripts/gcp/sync_artifacts.sh pull configs/cloud/main.json` without path drift.

  **QA Scenarios**:
  ```
  Scenario: First real GCE detector run completes end to end
    Tool: Bash
    Steps: Execute the exact local + VM command sequence above on the configured project and VM.
    Expected: Real detector artifacts, evaluation report, and run manifest exist locally/on-VM and sync successfully to the configured GCS prefixes.
    Evidence: .sisyphus/evidence/task-10-first-gce-run.txt

  Scenario: GPU unavailable blocks training before expensive run
    Tool: Bash
    Steps: On the VM, run the runtime probe after bootstrap with CUDA disabled or on a misconfigured host.
    Expected: The probe exits non-zero, the operator stops before training, and no misleading `best.pt` is produced.
    Evidence: .sisyphus/evidence/task-10-first-gce-run-error.txt
  ```

  **Commit**: NO | Message: `n/a` | Files: none; this is an execution milestone using prior implementation work

- [ ] 11. Build the upload-ready detector bundle and complete local release validation

  **What to do**: After the first real detector run is available, execute the release packaging sequence that ends in an upload-ready artifact. Use `configs/submission/detector_only.json` unchanged. Run, in order:
  1. `uv run python -m src.ng_data.submission.build_zip --config configs/submission/detector_only.json`
  2. `uv run python -m src.ng_data.submission.smoke_run --zip dist/ng_detector_only_submission.zip --input tests/fixtures/submission_images --output artifacts/smoke/detector_only_predictions.json`
  3. `uv run python -m src.ng_data.submission.budget_check --zip dist/ng_detector_only_submission.zip --input tests/fixtures/submission_images --out artifacts/smoke/detector_only_budget.json`
  4. `DRY_RUN=0 bash scripts/gcp/sync_artifacts.sh push configs/cloud/main.json`
  5. Copy `dist/ng_detector_only_submission.zip` to the release handoff path recorded in `artifacts/run_manifest.json` and update the manifest in the same task if that path field is not present yet.
  Then stop at the manual handoff boundary: the operator uploads `dist/ng_detector_only_submission.zip` at the competition submit page described in `docs/overview.md`. Do not invent a web API step.
  **Must NOT do**: Do not upload the `final.json` run.py-only bundle as if it were the first real detector submission. Do not mark the iteration complete unless smoke-run and budget-check both pass.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: packaging, validation, and release staging are tightly coupled.
  - Skills: `[]`
  - Omitted: `writing` — the focus here is executable validation.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: F1-F4 | Blocked By: 3, 7, 8, 9, 10

  **References**:
  - Config: `configs/submission/detector_only.json:1-9` — canonical upload-ready detector bundle source list.
  - Pattern: `src/ng_data/submission/__init__.py:333-458` — build and smoke-run entrypoints.
  - Pattern: `src/ng_data/submission/__init__.py:292-330` — weight/python-file limits and scanner checks.
  - Doc: `docs/submission.md:7-27` — zip size, file count, python file count, weight limits.
  - Doc: `docs/overview.md:13-18` — upload occurs at the submit page after zip creation.
  - Checklist: `docs/submission-checklist.md:10-35` — preserve truthful final-bundle notes while adding detector-only release notes.

  **Acceptance Criteria**:
  - [ ] `uv run python -m src.ng_data.submission.build_zip --config configs/submission/detector_only.json` exits `0` and writes `dist/ng_detector_only_submission.zip`.
  - [ ] `uv run python -m src.ng_data.submission.smoke_run --zip dist/ng_detector_only_submission.zip --input tests/fixtures/submission_images --output artifacts/smoke/detector_only_predictions.json` exits `0`.
  - [ ] `uv run python -m src.ng_data.submission.budget_check --zip dist/ng_detector_only_submission.zip --input tests/fixtures/submission_images --out artifacts/smoke/detector_only_budget.json` exits `0` with `"status": "pass"` and no failed checks.
  - [ ] `dist/ng_detector_only_submission.zip`, `artifacts/smoke/detector_only_predictions.json`, `artifacts/smoke/detector_only_budget.json`, and `artifacts/run_manifest.json` are synced to the canonical GCS release/manifests prefixes.
  - [ ] The final handoff explicitly says: “upload `dist/ng_detector_only_submission.zip` at the submit page manually.”

  **QA Scenarios**:
  ```
  Scenario: Detector-only release bundle is locally validated and staged
    Tool: Bash
    Steps: Run the build, smoke_run, budget_check, and sync commands in order.
    Expected: Zip build succeeds, predictions JSON is written, budget JSON reports pass, and release artifacts appear in the configured GCS release/manifest prefixes.
    Evidence: .sisyphus/evidence/task-11-detector-release.txt

  Scenario: Oversized or malformed bundle is rejected
    Tool: Bash
    Steps: Add extra large weight files or remove `run.py` from a copied bundle config and rerun the build/budget checks.
    Expected: Build or budget-check fails with the correct contract violation message instead of producing an upload-ready claim.
    Evidence: .sisyphus/evidence/task-11-detector-release-error.txt
  ```

  **Commit**: NO | Message: `n/a` | Files: none; this is the validated release execution milestone

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: Cloud preflight + GPU bootstrap alignment
- Commit 2: Artifact sync + run manifest contract
- Commit 3: Detector dataset prep + real train path
- Commit 4: Real detector evaluation + bundled `run.py` inference
- Commit 5: Detector submission tests/checklist + first release staging commands

## Success Criteria
- The repo still passes existing smoke-oriented structure and submission checks.
- A real detector training run produces non-placeholder weights and a non-placeholder evaluation report.
- `dist/ng_detector_only_submission.zip` is contract-valid, smoke-valid, budget-valid, and synced to the canonical GCS release prefix.
- The operator has an exact command sequence from local preflight through VM run and final manual upload handoff.
