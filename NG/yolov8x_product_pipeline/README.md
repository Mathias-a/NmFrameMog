# YOLOv8x Product Pipeline

This folder contains the primary training pipeline for the NorgesGruppen object-detection task.

The pipeline is built around:

- a **YOLOv8x one-class detector** optimized for product recall
- a **fine-grained recognizer** trained on shelf crops plus product reference images
- fold-aware dataset preparation, logging, validation visualization, and per-run artifacts

## Why this setup

The local NG data inspection showed:

- `248` shelf images
- `22,731` boxes
- `356` categories
- severe long-tail class imbalance
- highly variable image sizes
- high object density per image
- strong but incomplete linkage to the close-up product-image set

That favors a recall-first detector and a separate recognizer instead of a single large multiclass detector.

## Suggested GCP setup

- GPU: `L4` minimum, `A100 40GB` preferred for faster iteration
- CPU: `8+` vCPU
- RAM: `32+ GB`
- Disk: `200+ GB`
- Python: `3.11` or `3.10`

## Install on the VM

```bash
cd NmFrameMog
uv python install 3.11
uv sync --python 3.11 --group ng-train
```

Then run the pipeline from the repo root with `uv run`.

## End-to-end run

```bash
uv run --python 3.11 python NG/yolov8x_product_pipeline/run_pipeline.py \
  --data-root /path/to/NM \
  --workspace-root /path/to/NmFrameMog/NG/yolov8x_product_pipeline/workspace \
  --run-name yv8x_main_001
```

## Main stages

1. `prepare_data.py`
2. `train_detector.py`
3. `generate_oof_crops.py`
4. `train_recognizer.py`

## Artifacts

Every run is stored under:

```text
workspace/
  prepared/
  runs/
    <run_name>/
      config.json
      events.jsonl
      human_summary.md
      detector/
      recognizer/
```

Detailed logs live in machine-readable JSON and CSV files. Human-readable summaries are written as Markdown.

## Example high-quality detector run

```bash
uv run --python 3.11 python NG/yolov8x_product_pipeline/train_detector.py \
  --workspace-root /path/to/workspace \
  --run-name yv8x_main_001 \
  --model yolov8x.pt \
  --imgsz 1536 \
  --epochs 140 \
  --batch 6 \
  --patience 35 \
  --lr0 0.0025 \
  --lrf 0.12 \
  --weight-decay 0.0005 \
  --warmup-epochs 5 \
  --mosaic 0.35 \
  --mixup 0.0 \
  --close-mosaic 12 \
  --degrees 0.0 \
  --translate 0.04 \
  --scale 0.25 \
  --shear 0.0 \
  --perspective 0.0 \
  --fliplr 0.0 \
  --amp true \
  --cache ram
```

## Example recognizer run

```bash
uv run --python 3.11 python NG/yolov8x_product_pipeline/train_recognizer.py \
  --workspace-root /path/to/workspace \
  --run-name yv8x_main_001 \
  --backbone convnextv2_base.fcmae_ft_in22k_in1k \
  --epochs 30 \
  --image-size 288 \
  --batch-size 64 \
  --lr 0.00025 \
  --weight-decay 0.00005 \
  --embedding-dim 512 \
  --ce-weight 1.0 \
  --supcon-weight 0.15 \
  --label-smoothing 0.03 \
  --use-detector-oof true
```

## Recommended knobs to tune between runs

- detector:
  - `--imgsz`
  - `--epochs`
  - `--batch`
  - `--lr0`
  - `--lrf`
  - `--weight-decay`
  - `--mosaic`
  - `--close-mosaic`
  - `--translate`
  - `--scale`
  - `--conf-threshold`
  - `--iou-threshold`
- recognizer:
  - `--backbone`
  - `--image-size`
  - `--batch-size`
  - `--lr`
  - `--weight-decay`
  - `--embedding-dim`
  - `--ce-weight`
  - `--supcon-weight`
  - `--label-smoothing`
  - `--prototype-temperature`
  - `--use-detector-oof`

## Validation visuals

Detector training stores validation image artifacts from every validation loop by copying Ultralytics validation images into epoch-specific folders. This gives you per-epoch GT and predicted box snapshots for visual inspection.

## Notes

- `requirements-gcp.txt` is kept as a fallback reference, but the intended path is the root `uv` environment with the `ng-train` group.
- Use Python `3.11` on the VM for the training environment even though the monorepo itself targets newer Python for general development.
