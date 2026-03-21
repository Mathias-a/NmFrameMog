from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    EventLogger,
    copy_files_if_present,
    ensure_dir,
    patch_torch_load_for_trusted_checkpoints,
    parse_bool,
    parse_fold_indices,
    read_csv_rows,
    read_json,
    seed_everything,
    write_json,
    write_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the YOLOv8x one-class detector.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="yolov8x.pt")
    parser.add_argument("--selected-folds", default="all")
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0025)
    parser.add_argument("--lrf", type=float, default=0.12)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.2)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--hsv-h", type=float, default=0.010)
    parser.add_argument("--hsv-s", type=float, default=0.35)
    parser.add_argument("--hsv-v", type=float, default=0.18)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.04)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=0.35)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=12)
    parser.add_argument("--amp", default="true")
    parser.add_argument("--cos-lr", default="true")
    parser.add_argument("--cache", default="ram")
    parser.add_argument("--device", default="")
    parser.add_argument("--conf-threshold", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=350)
    parser.add_argument("--save-period", type=int, default=10)
    return parser


def _archive_epoch_visuals(trainer: Any, archive_root: Path) -> None:
    epoch_dir = ensure_dir(archive_root / f"epoch_{int(trainer.epoch) + 1:03d}")
    source_dir = Path(trainer.save_dir)
    source_files = sorted(path.name for path in source_dir.glob("val_batch*.jpg"))
    copied = copy_files_if_present(
        src_dir=source_dir,
        dst_dir=epoch_dir,
        patterns=["val_batch*_labels.jpg", "val_batch*_pred.jpg"],
    )
    write_json(
        epoch_dir / "debug.json",
        {
            "epoch": int(trainer.epoch) + 1,
            "source_dir": str(source_dir),
            "source_files": source_files,
            "copied_files": copied,
        },
    )


def _best_metrics_from_results(results_csv_path: Path) -> dict[str, Any]:
    rows = read_csv_rows(results_csv_path)
    if not rows:
        return {}
    best_row = None
    best_map50 = float("-inf")
    best_epoch = None
    for row in rows:
        try:
            score = float(row.get("metrics/mAP50(B)", "nan"))
        except ValueError:
            continue
        if score > best_map50:
            best_map50 = score
            best_row = row
            best_epoch = int(float(row.get("epoch", "0")))
    return {
        "best_epoch": best_epoch,
        "best_row": best_row,
    }


def _write_human_summary(path: Path, run_name: str, fold_summaries: list[dict[str, Any]]) -> None:
    lines = [f"# Detector Summary for `{run_name}`", ""]
    for summary in fold_summaries:
        best_row = summary.get("best_row") or {}
        lines.extend(
            [
                f"## Fold {summary['fold']}",
                "",
                f"- Best epoch: {summary.get('best_epoch')}",
                f"- mAP50: {best_row.get('metrics/mAP50(B)', 'n/a')}",
                f"- mAP50-95: {best_row.get('metrics/mAP50-95(B)', 'n/a')}",
                f"- Precision: {best_row.get('metrics/precision(B)', 'n/a')}",
                f"- Recall: {best_row.get('metrics/recall(B)', 'n/a')}",
                "",
            ]
        )
    write_text(path, "\n".join(lines))


def train_detector_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed_everything(args.seed)
    patch_torch_load_for_trusted_checkpoints()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("ultralytics must be installed to run train_detector.py") from exc

    workspace_root = Path(args.workspace_root).resolve()
    prepared_root = workspace_root / "prepared"
    dataset_summary = read_json(prepared_root / "dataset_summary.json")
    fold_specs = {spec["fold"]: spec for spec in dataset_summary["detector_folds"]}
    selected_folds = parse_fold_indices(dataset_summary["num_folds"], args.selected_folds)

    run_root = ensure_dir(workspace_root / "runs" / args.run_name)
    detector_root = ensure_dir(run_root / "detector")
    event_logger = EventLogger(run_root / "events.jsonl")
    run_config = {
        key: value for key, value in vars(args).items()
    }
    write_json(run_root / "config_detector.json", run_config)
    fold_summaries: list[dict[str, Any]] = []

    for fold in selected_folds:
        fold_spec = fold_specs[fold]
        fold_root = ensure_dir(detector_root / f"fold_{fold}")
        visuals_root = ensure_dir(fold_root / "epoch_visuals")
        project_root = ensure_dir(fold_root / "ultralytics")
        event_logger.log("detector_fold_start", fold=fold, model=args.model)

        model = YOLO(args.model)
        archive_callback = lambda trainer, archive_root=visuals_root: _archive_epoch_visuals(trainer, archive_root)
        model.add_callback("on_val_end", archive_callback)
        model.add_callback("on_fit_epoch_end", archive_callback)

        model.train(
            data=fold_spec["one_class_yaml"],
            project=str(project_root),
            name="train",
            exist_ok=True,
            seed=args.seed + fold,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            box=args.box,
            cls=args.cls,
            dfl=args.dfl,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            close_mosaic=args.close_mosaic,
            amp=parse_bool(args.amp),
            cos_lr=parse_bool(args.cos_lr),
            cache=args.cache,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            max_det=args.max_det,
            save=True,
            save_period=args.save_period,
            plots=True,
            device=args.device or None,
            pretrained=True,
            val=True,
            single_cls=True,
            rect=True,
            verbose=True,
        )

        results_dir = project_root / "train"
        results_csv = results_dir / "results.csv"
        if not results_csv.exists():
            raise FileNotFoundError(f"Expected Ultralytics results at {results_csv}")

        rows = read_csv_rows(results_csv)
        with (fold_root / "metrics_detailed.csv").open("w", encoding="utf-8", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        best_summary = _best_metrics_from_results(results_csv)
        weights_dir = results_dir / "weights"
        payload = {
            "fold": fold,
            "results_csv": str(results_csv),
            "best_weights": str(weights_dir / "best.pt"),
            "last_weights": str(weights_dir / "last.pt"),
            **best_summary,
        }
        write_json(fold_root / "summary.json", payload)
        fold_summaries.append(payload)
        event_logger.log("detector_fold_complete", **payload)

    write_json(detector_root / "summary_all_folds.json", fold_summaries)
    _write_human_summary(detector_root / "human_summary.md", args.run_name, fold_summaries)
    return fold_summaries


def main() -> int:
    args = build_parser().parse_args()
    train_detector_runs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
