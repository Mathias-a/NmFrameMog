from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import EventLogger, ensure_dir, write_json, write_text
from generate_oof_crops import generate_oof_crops
from prepare_data import prepare_workspace
from train_detector import train_detector_runs
from train_recognizer import train_recognizer_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full YOLOv8x product pipeline.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--prepare-only", action="store_true")

    parser.add_argument("--detector-model", default="yolov8x.pt")
    parser.add_argument("--detector-epochs", type=int, default=140)
    parser.add_argument("--detector-imgsz", type=int, default=1536)
    parser.add_argument("--detector-batch", type=int, default=6)
    parser.add_argument("--detector-patience", type=int, default=35)
    parser.add_argument("--detector-workers", type=int, default=8)
    parser.add_argument("--detector-lr0", type=float, default=0.0025)
    parser.add_argument("--detector-lrf", type=float, default=0.12)
    parser.add_argument("--detector-weight-decay", type=float, default=0.0005)
    parser.add_argument("--detector-warmup-epochs", type=float, default=5.0)
    parser.add_argument("--detector-mosaic", type=float, default=0.35)
    parser.add_argument("--detector-mixup", type=float, default=0.0)
    parser.add_argument("--detector-copy-paste", type=float, default=0.0)
    parser.add_argument("--detector-close-mosaic", type=int, default=12)
    parser.add_argument("--detector-device", default="")
    parser.add_argument("--detector-cache", default="ram")

    parser.add_argument("--recognizer-backbone", default="convnextv2_base.fcmae_ft_in22k_in1k")
    parser.add_argument("--recognizer-image-size", type=int, default=288)
    parser.add_argument("--recognizer-epochs", type=int, default=30)
    parser.add_argument("--recognizer-batch-size", type=int, default=64)
    parser.add_argument("--recognizer-num-workers", type=int, default=8)
    parser.add_argument("--recognizer-lr", type=float, default=0.00025)
    parser.add_argument("--recognizer-weight-decay", type=float, default=0.00005)
    parser.add_argument("--recognizer-embedding-dim", type=int, default=512)
    parser.add_argument("--recognizer-dropout", type=float, default=0.15)
    parser.add_argument("--recognizer-ce-weight", type=float, default=1.0)
    parser.add_argument("--recognizer-supcon-weight", type=float, default=0.15)
    parser.add_argument("--recognizer-label-smoothing", type=float, default=0.03)
    parser.add_argument("--recognizer-use-detector-oof", default="true")
    parser.add_argument("--recognizer-device", default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    run_root = ensure_dir(workspace_root / "runs" / args.run_name)
    event_logger = EventLogger(run_root / "events.jsonl")
    write_json(run_root / "config.json", vars(args))

    event_logger.log("pipeline_start", run_name=args.run_name)
    dataset_summary = prepare_workspace(
        data_root=Path(args.data_root).resolve(),
        workspace_root=workspace_root,
        num_folds=args.num_folds,
        seed=args.seed,
        crop_padding=0.08,
        category_overrides_path=Path(__file__).resolve().parent / "category_name_overrides.json",
    )
    event_logger.log("pipeline_prepare_complete", dataset_summary=dataset_summary)

    if args.prepare_only:
        write_text(run_root / "human_summary.md", "# Pipeline Summary\n\nPreparation completed.\n")
        return 0

    detector_summaries = train_detector_runs(
        argparse.Namespace(
            workspace_root=str(workspace_root),
            run_name=args.run_name,
            model=args.detector_model,
            selected_folds="all",
            seed=args.seed,
            epochs=args.detector_epochs,
            imgsz=args.detector_imgsz,
            batch=args.detector_batch,
            patience=args.detector_patience,
            workers=args.detector_workers,
            optimizer="AdamW",
            lr0=args.detector_lr0,
            lrf=args.detector_lrf,
            weight_decay=args.detector_weight_decay,
            warmup_epochs=args.detector_warmup_epochs,
            box=7.5,
            cls=0.2,
            dfl=1.5,
            hsv_h=0.010,
            hsv_s=0.35,
            hsv_v=0.18,
            degrees=0.0,
            translate=0.04,
            scale=0.25,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=args.detector_mosaic,
            mixup=args.detector_mixup,
            copy_paste=args.detector_copy_paste,
            close_mosaic=args.detector_close_mosaic,
            amp="true",
            cos_lr="true",
            cache=args.detector_cache,
            device=args.detector_device,
            conf_threshold=0.001,
            iou_threshold=0.60,
            max_det=350,
            save_period=10,
        )
    )
    event_logger.log("pipeline_detector_complete", folds=len(detector_summaries))

    oof_summary = generate_oof_crops(
        argparse.Namespace(
            workspace_root=str(workspace_root),
            run_name=args.run_name,
            selected_folds="all",
            imgsz=args.detector_imgsz,
            device=args.detector_device,
            conf_threshold=0.01,
            iou_threshold=0.60,
            match_iou_threshold=0.40,
            crop_padding=0.06,
        )
    )
    event_logger.log("pipeline_oof_complete", **oof_summary)

    recognizer_summaries = train_recognizer_runs(
        argparse.Namespace(
            workspace_root=str(workspace_root),
            run_name=args.run_name,
            selected_folds="all",
            seed=args.seed,
            backbone=args.recognizer_backbone,
            image_size=args.recognizer_image_size,
            epochs=args.recognizer_epochs,
            batch_size=args.recognizer_batch_size,
            num_workers=args.recognizer_num_workers,
            lr=args.recognizer_lr,
            weight_decay=args.recognizer_weight_decay,
            embedding_dim=args.recognizer_embedding_dim,
            dropout=args.recognizer_dropout,
            ce_weight=args.recognizer_ce_weight,
            supcon_weight=args.recognizer_supcon_weight,
            label_smoothing=args.recognizer_label_smoothing,
            warmup_epochs=3,
            prototype_temperature=0.10,
            amp="true",
            device=args.recognizer_device,
            use_detector_oof=args.recognizer_use_detector_oof,
        )
    )
    event_logger.log("pipeline_recognizer_complete", folds=len(recognizer_summaries))

    human_summary = [
        f"# Full Pipeline Summary for `{args.run_name}`",
        "",
        "## Detector",
        "",
        f"- Folds trained: {len(detector_summaries)}",
        f"- OOF crops generated: {oof_summary['crop_count']}",
        "",
        "## Recognizer",
        "",
        f"- Folds trained: {len(recognizer_summaries)}",
        "",
        "See the per-stage human summaries for details.",
    ]
    write_text(run_root / "human_summary.md", "\n".join(human_summary) + "\n")
    write_json(
        run_root / "summary.json",
        {
            "detector": detector_summaries,
            "oof": oof_summary,
            "recognizer": recognizer_summaries,
        },
    )
    event_logger.log("pipeline_complete", run_name=args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
