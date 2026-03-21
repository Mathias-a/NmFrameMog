from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    EventLogger,
    compute_iou,
    ensure_dir,
    padded_box,
    parse_fold_indices,
    patch_torch_load_for_trusted_checkpoints,
    read_json,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate out-of-fold detector crops for recognizer refinement.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--selected-folds", default="all")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="")
    parser.add_argument("--conf-threshold", type=float, default=0.01)
    parser.add_argument("--iou-threshold", type=float, default=0.60)
    parser.add_argument("--match-iou-threshold", type=float, default=0.40)
    parser.add_argument("--crop-padding", type=float, default=0.06)
    return parser


def _greedy_match(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    match_iou_threshold: float,
) -> list[dict[str, Any]]:
    candidate_pairs: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, ground_truth in enumerate(ground_truths):
            iou = compute_iou(prediction["bbox"], ground_truth["bbox"])
            if iou >= match_iou_threshold:
                candidate_pairs.append((iou, pred_index, gt_index))
    candidate_pairs.sort(reverse=True)

    matched_pred_indices: set[int] = set()
    matched_gt_indices: set[int] = set()
    matches = []
    for iou, pred_index, gt_index in candidate_pairs:
        if pred_index in matched_pred_indices or gt_index in matched_gt_indices:
            continue
        matched_pred_indices.add(pred_index)
        matched_gt_indices.add(gt_index)
        matches.append(
            {
                "prediction_index": pred_index,
                "ground_truth_index": gt_index,
                "iou": iou,
            }
        )
    return matches


def generate_oof_crops(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for generate_oof_crops.py") from exc

    patch_torch_load_for_trusted_checkpoints()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("ultralytics must be installed to run generate_oof_crops.py") from exc

    workspace_root = Path(args.workspace_root).resolve()
    prepared_root = workspace_root / "prepared"
    run_root = workspace_root / "runs" / args.run_name
    dataset_summary = read_json(prepared_root / "dataset_summary.json")
    image_manifest = {entry["image_id"]: entry for entry in read_json(prepared_root / "image_manifest.json")}
    gt_crops_manifest = read_json(prepared_root / "recognizer" / "gt_crops_manifest.json")
    fold_assignment = {int(key): value for key, value in read_json(prepared_root / "fold_assignment.json").items()}
    selected_folds = parse_fold_indices(dataset_summary["num_folds"], args.selected_folds)
    detector_summary = read_json(run_root / "detector" / "summary_all_folds.json")
    detector_summary_by_fold = {entry["fold"]: entry for entry in detector_summary}
    oof_root = ensure_dir(prepared_root / "recognizer" / "oof_detector_crops" / args.run_name)
    event_logger = EventLogger(run_root / "events.jsonl")

    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in gt_crops_manifest:
        gt_by_image[entry["image_id"]].append(entry)

    generated_manifest: list[dict[str, Any]] = []
    for fold in selected_folds:
        best_weights = Path(detector_summary_by_fold[fold]["best_weights"])
        if not best_weights.exists():
            raise FileNotFoundError(f"Missing best weights for fold {fold}: {best_weights}")
        model = YOLO(str(best_weights))
        fold_dir = ensure_dir(oof_root / f"fold_{fold}")
        images_in_fold = [
            image_manifest[image_id]
            for image_id, assigned_fold in fold_assignment.items()
            if assigned_fold == fold
        ]
        event_logger.log("oof_fold_start", fold=fold, best_weights=str(best_weights), image_count=len(images_in_fold))

        for image_entry in images_in_fold:
            image_path = Path(image_entry["image_path"])
            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf_threshold,
                iou=args.iou_threshold,
                device=args.device or None,
                max_det=350,
                verbose=False,
            )
            boxes: list[dict[str, Any]] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box_index in range(len(result.boxes)):
                    x1, y1, x2, y2 = result.boxes.xyxy[box_index].tolist()
                    boxes.append(
                        {
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(result.boxes.conf[box_index].item()),
                        }
                    )

            ground_truths = gt_by_image[image_entry["image_id"]]
            matches = _greedy_match(boxes, ground_truths, args.match_iou_threshold)
            matched_predictions = {match["prediction_index"]: match for match in matches}

            with Image.open(image_path) as pil_image:
                rgb_image = pil_image.convert("RGB")
                for prediction_index, prediction in enumerate(boxes):
                    if prediction_index not in matched_predictions:
                        continue
                    match = matched_predictions[prediction_index]
                    ground_truth = ground_truths[match["ground_truth_index"]]
                    crop_path = (
                        fold_dir
                        / f"img{image_entry['image_id']:05d}_pred{prediction_index:03d}_ann{ground_truth['annotation_id']:05d}.jpg"
                    )
                    x1, y1, x2, y2 = padded_box(
                        prediction["bbox"],
                        width=image_entry["width"],
                        height=image_entry["height"],
                        padding_ratio=args.crop_padding,
                    )
                    if not crop_path.exists():
                        crop = rgb_image.crop((x1, y1, x2, y2))
                        crop.save(crop_path, quality=95)
                    generated_manifest.append(
                        {
                            "fold": fold,
                            "image_id": image_entry["image_id"],
                            "image_path": str(image_path),
                            "crop_path": str(crop_path),
                            "category_id": ground_truth["category_id"],
                            "category_name": ground_truth["category_name"],
                            "score": prediction["score"],
                            "iou_with_gt": match["iou"],
                            "source": "detector_oof",
                        }
                    )
        event_logger.log("oof_fold_complete", fold=fold)

    manifest_path = oof_root / "manifest.json"
    write_json(manifest_path, generated_manifest)
    event_logger.log("oof_generation_complete", manifest_path=str(manifest_path), crop_count=len(generated_manifest))
    return {"manifest_path": str(manifest_path), "crop_count": len(generated_manifest)}


def main() -> int:
    args = build_parser().parse_args()
    generate_oof_crops(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
