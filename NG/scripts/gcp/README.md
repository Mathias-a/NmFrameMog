# GCP helper workflow

The mainline cloud path is **one persistent Compute Engine GPU VM plus one canonical GCS bucket namespace** in `europe-west1`.

- `bootstrap_vm.sh` renders the `gcloud compute` commands for the baseline GPU host and its first-boot workspace prep.
- `sync_artifacts.sh` mirrors checkpoints, eval outputs, release bundles, and the current run manifest into the canonical GCS prefixes.
- Vertex AI is **not** the default path here. Treat it only as an optional escalation for later final runs after the GCE + GCS workflow is stable and using the same bucket layout.

## First cloud iteration: canonical real-data operator path

The checked-in `data/processed/` tree is a tiny deterministic fixture for local tests. Do **not** treat it as the first cloud iteration dataset. The operator path for the first real cloud run starts from the competition downloads in `data/raw/` and regenerates processed outputs locally before syncing them to GCS and the VM.

1. Place the real competition archives from the challenge website in `data/raw/`:

   - `NM_NGD_coco_dataset.zip`
   - `NM_NGD_product_images.zip`

2. Regenerate the canonical processed dataset from those real archives:

   ```bash
   uv run python -m src.ng_data.data.ingest --config configs/data/main.json --raw data/raw --processed data/processed
   ```

3. Generate deterministic holdout and CV splits from the regenerated dataset manifest:

   ```bash
   uv run python -m src.ng_data.eval.make_splits --config configs/data/splits.json --manifest data/processed/manifests/dataset_manifest.json
   ```

4. Audit the dataset manifest before any cloud sync:

   ```bash
   uv run python -c "from src.ng_data.data.manifest import audit_dataset_manifest; import json; print(json.dumps(audit_dataset_manifest('data/processed/manifests/dataset_manifest.json'), indent=2, sort_keys=True))"
   ```

5. Stage the real archives and processed outputs into the canonical GCS prefixes:

   ```bash
   gcloud storage cp data/raw/NM_NGD_coco_dataset.zip gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/raw/
   gcloud storage cp data/raw/NM_NGD_product_images.zip gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/raw/
   gcloud storage rsync --recursive data/processed gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data/processed
   ```

6. On the VM, mirror the canonical data prefix into the workspace:

   ```bash
   gcloud storage rsync --recursive gs://ainm26osl-707-ng-artifacts/norgesgruppen-agentic-cv/data /home/ng/workspace/NG/data
   ```

This keeps the first iteration simple: local ingest + deterministic split generation + manifest audit + GCS sync + VM pull, all against the real competition archives instead of the fixture-sized repo sample.
