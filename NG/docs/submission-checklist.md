# Submission checklist

## Current status

- Final retraining is blocked.
- `artifacts/release/final_manifest.json` is a blocked hold-manifest.
- The current final bundle is contract-valid and smoke-valid.
- The current final bundle is not evidence of release-grade trained model quality.

## Honest final bundle for the current repo state

1. Build the final zip with the run.py-only config:

   ```bash
   uv run python -m src.ng_data.submission.build_zip --config configs/submission/final.json
   ```

2. Smoke the built zip against the local fixture images:

   ```bash
   uv run python -m src.ng_data.submission.smoke_run --zip dist/ng_submission.zip --input tests/fixtures/submission_images --output artifacts/smoke/final_predictions.json
   ```

3. Record local budget evidence for the same bundle:

   ```bash
   uv run python -m src.ng_data.submission.budget_check --zip dist/ng_submission.zip --input tests/fixtures/submission_images --out artifacts/smoke/final_budget.json
   ```

## Real detector-only first submission

Use this path first when you need the real detector-only bundle that includes the packaged detector artifacts from `configs/submission/detector_only.json`.

1. Build the detector-only zip:

   ```bash
   uv run python -m src.ng_data.submission.build_zip --config configs/submission/detector_only.json
   ```

2. Smoke the built detector-only zip against the local fixture images:

   ```bash
   uv run python -m src.ng_data.submission.smoke_run --zip dist/ng_detector_only_submission.zip --input tests/fixtures/submission_images --output artifacts/smoke/detector_only_predictions.json
   ```

3. Record local budget evidence for the same detector-only bundle:

   ```bash
   uv run python -m src.ng_data.submission.budget_check --zip dist/ng_detector_only_submission.zip --input tests/fixtures/submission_images --out artifacts/smoke/detector_only_budget.json
   ```

## Operator handoff notes

- Do not package placeholder detector weights as a final model.
- Do not claim release-grade accuracy from this zip.
- Treat this bundle as a truthful contract check for the current blocked state.
- Replace this run.py-only bundle only after promotion-grade retraining evidence exists and the blocked hold-manifest state is cleared.

## First cloud iteration dataset staging

- The checked-in `data/processed/` files are fixture-sized local test outputs, not the real competition dataset for cloud training.
- Before the first cloud run, regenerate `data/processed/` from the real challenge archives placed in `data/raw/`:

  ```bash
  uv run python -m src.ng_data.data.ingest --config configs/data/main.json --raw data/raw --processed data/processed
  uv run python -m src.ng_data.eval.make_splits --config configs/data/splits.json --manifest data/processed/manifests/dataset_manifest.json
  uv run python -c "from src.ng_data.data.manifest import audit_dataset_manifest; import json; print(json.dumps(audit_dataset_manifest('data/processed/manifests/dataset_manifest.json'), indent=2, sort_keys=True))"
  ```

- Stage the same real archives and regenerated processed tree to the canonical bucket paths:

  ```bash
  gcloud storage cp data/raw/NM_NGD_coco_dataset.zip gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/raw/
  gcloud storage cp data/raw/NM_NGD_product_images.zip gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/raw/
  gcloud storage rsync --recursive data/processed gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/processed
  ```

- On the training VM, pull the canonical data prefix into the workspace before running detector jobs:

  ```bash
  gcloud storage rsync --recursive gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data /home/ng/workspace/NG/data
  ```
