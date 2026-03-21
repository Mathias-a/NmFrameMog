#!/usr/bin/env python3
"""
Competition-grade Deformable DETR fine-tuning script for dense retail shelf detection.

Key features
------------
- Stable, reproducible multilabel K-fold split
- Proper validation metrics: mAP, mAP@0.30, mAP@0.50, mAP@0.75
- Structured logging to run directory
- Best and last checkpoint saving
- Qualitative validation visualizations
- Safe first-pass augmentation pipeline (easy to disable)
- Aspect-ratio-preserving resize (no forced 800x800 distortion)
- Mixed precision training
- Warmup + cosine LR schedule
- Gradient clipping
- Easy-to-navigate run folder structure

Example
-------
uv run python train_deformable_detr_retail.py \
    --data-root /home/matiasfernandezjr/Data/NM/train \
    --run-name detr_retail_fold0 \
    --fold-index 0 \
    --num-folds 5 \
    --epochs 40 \
    --batch-size 2 \
    --grad-accum-steps 4 \
    --use-aug

Suggested uv additions
----------------------
uv add torch torchvision transformers pillow tqdm numpy albumentations torchmetrics pycocotools matplotlib
"""

import argparse
import copy
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from transformers.image_transforms import center_to_corners_format


# ----------------------------
# General utilities
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dump(data, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, run_dir: Path, run_name: str):
        self.run_dir = run_dir
        self.run_name = run_name
        self.log_txt_path = run_dir / f"log_{run_name}.txt"
        self.log_jsonl_path = run_dir / f"metrics_{run_name}.jsonl"

    def log(self, message: str) -> None:
        line = f"[{now_str()}] {message}"
        print(line, flush=True)
        with self.log_txt_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_epoch_metrics(self, metrics: dict) -> None:
        with self.log_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


# ----------------------------
# COCO loading and validation
# ----------------------------

def load_coco(annotation_file: Path) -> dict:
    with annotation_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def clip_bbox_xywh(bbox, width: int, height: int):
    x, y, w, h = bbox
    x1 = min(max(float(x), 0.0), float(width))
    y1 = min(max(float(y), 0.0), float(height))
    x2 = min(max(float(x + w), 0.0), float(width))
    y2 = min(max(float(y + h), 0.0), float(height))
    new_w = max(0.0, x2 - x1)
    new_h = max(0.0, y2 - y1)
    return [x1, y1, new_w, new_h]


def sanitize_annotations(coco: dict):
    images_by_id = {img["id"]: img for img in coco["images"]}
    annotations_by_image = defaultdict(list)

    invalid_count = 0
    clipped_count = 0

    for ann in coco["annotations"]:
        image_info = images_by_id[ann["image_id"]]
        width = int(image_info["width"])
        height = int(image_info["height"])

        bbox = clip_bbox_xywh(ann["bbox"], width, height)
        if bbox != ann["bbox"]:
            clipped_count += 1

        if bbox[2] <= 1.0 or bbox[3] <= 1.0:
            invalid_count += 1
            continue

        annotations_by_image[ann["image_id"]].append(
            {
                "id": int(ann["id"]),
                "image_id": int(ann["image_id"]),
                "category_id": int(ann["category_id"]),
                "bbox": bbox,
                "area": float(ann.get("area", bbox[2] * bbox[3])),
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
        )

    return annotations_by_image, {
        "num_invalid_boxes_removed": invalid_count,
        "num_boxes_clipped_to_image": clipped_count,
    }


# ----------------------------
# Stable multilabel K-fold split
# ----------------------------

def build_image_label_sets(images, annotations_by_image):
    image_label_sets = {}
    for img in images:
        image_id = img["id"]
        labels = [ann["category_id"] for ann in annotations_by_image.get(image_id, [])]
        image_label_sets[image_id] = sorted(set(labels))
    return image_label_sets


def stable_multilabel_kfold(images, annotations_by_image, num_folds: int, seed: int):
    """
    Greedy multilabel-aware fold assignment.

    Idea:
    - sort images by rarity of contained labels and number of unique labels
    - place each image into the fold that best balances:
      (1) counts of its labels across folds
      (2) fold sizes

    This is not exact iterative stratification, but for small dense datasets it is
    deterministic, reproducible, and much better than naive random splitting.
    """
    rng = random.Random(seed)
    images = copy.deepcopy(images)
    image_label_sets = build_image_label_sets(images, annotations_by_image)

    label_freq = Counter()
    for labels in image_label_sets.values():
        label_freq.update(labels)

    def rarity_score(image_info):
        labels = image_label_sets[image_info["id"]]
        if not labels:
            return (10**9, 0, image_info["id"])
        rarity = min(label_freq[l] for l in labels)
        return (rarity, -len(labels), image_info["id"])

    images_sorted = sorted(images, key=rarity_score)

    fold_images = [[] for _ in range(num_folds)]
    fold_label_counts = [Counter() for _ in range(num_folds)]
    fold_sizes = [0 for _ in range(num_folds)]

    for image_info in images_sorted:
        image_id = image_info["id"]
        labels = image_label_sets[image_id]

        fold_order = list(range(num_folds))
        rng.shuffle(fold_order)

        best_fold = None
        best_score = None

        for fold_idx in fold_order:
            label_balance_penalty = sum(fold_label_counts[fold_idx][lab] for lab in labels)
            size_penalty = fold_sizes[fold_idx]
            score = (label_balance_penalty, size_penalty, fold_idx)
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold_idx

        fold_images[best_fold].append(image_info)
        fold_sizes[best_fold] += 1
        fold_label_counts[best_fold].update(labels)

    return fold_images


# ----------------------------
# Augmentations
# ----------------------------

def build_transforms(use_aug: bool):
    if not use_aug:
        return None

    # Conservative first-pass augmentation for retail shelves:
    # - mild photometric changes
    # - slight blur/noise/compression
    # - tiny affine scale/translate
    # Avoid risky heavy crops or large perspective changes.
    return A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.12,
                        contrast_limit=0.12,
                        p=1.0,
                    ),
                    A.ColorJitter(
                        brightness=0.08,
                        contrast=0.08,
                        saturation=0.08,
                        hue=0.03,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
                    A.ImageCompression(quality_range=(85, 100), p=1.0),
                ],
                p=0.25,
            ),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-1.5, 1.5),
                shear=(-1.0, 1.0),
                fit_output=False,
                p=0.25,
            ),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            clip=True,
            min_area=4.0,
            min_visibility=0.2,
        ),
    )


# ----------------------------
# Dataset
# ----------------------------

class CocoRetailDetectionDataset(Dataset):
    def __init__(
        self,
        images,
        annotations_by_image,
        images_dir: Path,
        image_processor,
        transforms=None,
        return_visualization_data: bool = False,
    ):
        self.images = images
        self.annotations_by_image = annotations_by_image
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.transforms = transforms
        self.return_visualization_data = return_visualization_data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = int(image_info["id"])
        image_path = self.images_dir / image_info["file_name"]

        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        anns = copy.deepcopy(self.annotations_by_image.get(image_id, []))

        if self.transforms is not None and len(anns) > 0:
            bboxes = [ann["bbox"] for ann in anns]
            category_ids = [ann["category_id"] for ann in anns]

            transformed = self.transforms(
                image=image_np,
                bboxes=bboxes,
                category_ids=category_ids,
            )
            image_np = transformed["image"]

            new_anns = []
            for i, bbox in enumerate(transformed["bboxes"]):
                ann = {
                    "id": i,
                    "image_id": image_id,
                    "category_id": int(transformed["category_ids"][i]),
                    "bbox": [float(v) for v in bbox],
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
                if ann["bbox"][2] > 1.0 and ann["bbox"][3] > 1.0:
                    new_anns.append(ann)
            anns = new_anns

        image_pil = Image.fromarray(image_np)

        target = {
            "image_id": image_id,
            "annotations": anns,
        }

        encoding = self.image_processor(
            images=image_pil,
            annotations=target,
            return_tensors="pt",
        )

        item = {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
            "image_id": image_id,
            "orig_size_hw": tuple(image_pil.size[::-1]),  # (H, W)
        }

        if "pixel_mask" in encoding:
            item["pixel_mask"] = encoding["pixel_mask"].squeeze(0)

        if self.return_visualization_data:
            item["image_pil"] = image_pil.copy()
            item["gt_annotations"] = anns

        return item


def collate_fn(batch, image_processor):
    pixel_values = [item["pixel_values"] for item in batch]

    # Hugging Face DETR-style processors vary by version:
    # some use pad_and_create_pixel_mask, some expose pad differently.
    if hasattr(image_processor, "pad_and_create_pixel_mask"):
        encoding = image_processor.pad_and_create_pixel_mask(
            pixel_values,
            return_tensors="pt",
        )
        pixel_values_batched = encoding["pixel_values"]
        pixel_mask_batched = encoding["pixel_mask"]
    else:
        # Fallback that works across more versions
        max_h = max(x.shape[1] for x in pixel_values)
        max_w = max(x.shape[2] for x in pixel_values)
        batch_size = len(pixel_values)
        channels = pixel_values[0].shape[0]

        pixel_values_batched = torch.zeros(
            (batch_size, channels, max_h, max_w),
            dtype=pixel_values[0].dtype,
        )
        pixel_mask_batched = torch.zeros(
            (batch_size, max_h, max_w),
            dtype=torch.long,
        )

        for i, x in enumerate(pixel_values):
            _, h, w = x.shape
            pixel_values_batched[i, :, :h, :w] = x
            pixel_mask_batched[i, :h, :w] = 1

    data = {
        "pixel_values": pixel_values_batched,
        "pixel_mask": pixel_mask_batched,
        "labels": [item["labels"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "orig_size_hw": [item["orig_size_hw"] for item in batch],
    }

    if "image_pil" in batch[0]:
        data["image_pil"] = [item["image_pil"] for item in batch]
        data["gt_annotations"] = [item["gt_annotations"] for item in batch]

    return data


# ----------------------------
# Metrics and box conversions
# ----------------------------

def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size_hw):
    """
    YOLO center format normalized [cx, cy, w, h] -> Pascal VOC xyxy absolute.
    image_size_hw: (H, W)
    """
    boxes = center_to_corners_format(boxes)
    height, width = image_size_hw
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
    return boxes * scale


@dataclass
class ModelOutputShim:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def build_metric(iou_thresholds=None, class_metrics=False, max_dets=(300, 300, 300)):
    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
        class_metrics=class_metrics,
        max_detection_thresholds=list(max_dets),
        backend="pycocotools",
    )


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    image_processor,
    score_threshold: float,
    logger: RunLogger,
):
    model.eval()

    metric_main = build_metric(class_metrics=False)
    metric_30 = build_metric(iou_thresholds=[0.30], class_metrics=False)
    metric_50 = build_metric(iou_thresholds=[0.50], class_metrics=True)
    metric_75 = build_metric(iou_thresholds=[0.75], class_metrics=False)

    val_loss_sum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        labels = [{k: v.to(device) for k, v in label.items()} for label in batch["labels"]]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )

        loss = outputs.loss
        val_loss_sum += float(loss.item())
        num_batches += 1

        target_sizes = torch.tensor(batch["orig_size_hw"], device=device)
        predictions = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )

        targets_for_metric = []
        preds_for_metric = []

        for pred, label_cpu in zip(predictions, batch["labels"]):
            orig_size = tuple(int(x) for x in label_cpu["orig_size"].tolist())
            boxes = convert_bbox_yolo_to_pascal(
                label_cpu["boxes"].cpu(),
                orig_size,
            )
            labels_cpu = label_cpu["class_labels"].cpu()

            targets_for_metric.append(
                {
                    "boxes": boxes,
                    "labels": labels_cpu,
                }
            )

            preds_for_metric.append(
                {
                    "boxes": pred["boxes"].detach().cpu(),
                    "scores": pred["scores"].detach().cpu(),
                    "labels": pred["labels"].detach().cpu(),
                }
            )
        if num_batches == 1 and len(preds_for_metric) > 0 and len(targets_for_metric) > 0:
        logger.log(
            f"eval sample: "
            f"num_pred={len(preds_for_metric[0]['boxes'])}, "
            f"num_tgt={len(targets_for_metric[0]['boxes'])}, "
            f"pred_labels_dtype={preds_for_metric[0]['labels'].dtype}, "
            f"tgt_labels_dtype={targets_for_metric[0]['labels'].dtype}"
        )

        if len(targets_for_metric[0]["boxes"]) > 0:
            logger.log(f"first tgt box: {targets_for_metric[0]['boxes'][0].tolist()}")
        if len(preds_for_metric[0]["boxes"]) > 0:
            logger.log(f"first pred box: {preds_for_metric[0]['boxes'][0].tolist()}")

        metric_main.update(preds_for_metric, targets_for_metric)
        metric_30.update(preds_for_metric, targets_for_metric)
        metric_50.update(preds_for_metric, targets_for_metric)
        metric_75.update(preds_for_metric, targets_for_metric)

    results_main = metric_main.compute()
    results_30 = metric_30.compute()
    results_50 = metric_50.compute()
    results_75 = metric_75.compute()

    avg_val_loss = val_loss_sum / max(1, num_batches)

    metrics = {
        "val_loss": round(avg_val_loss, 6),
        "val_map": round(float(results_main["map"].item()), 6),
        "val_map30": round(float(results_30["map"].item()), 6),
        "val_map50": round(float(results_50["map"].item()), 6),
        "val_map75": round(float(results_75["map"].item()), 6),
    }

    # Add whichever mAR key actually exists
    for k in ["mar_100", "mar_300", "mar_10", "mar_1"]:
        if k in results_main:
            metrics[f"val_{k}"] = round(float(results_main[k].item()), 6)

    if "classes" in results_50 and "map_per_class" in results_50:
        classes = results_50["classes"].tolist()
        map_per_class = results_50["map_per_class"].tolist()
        metrics["val_map50_per_class"] = {
            str(int(cls_id)): round(float(cls_map), 6)
            for cls_id, cls_map in zip(classes, map_per_class)
        }

    logger.log(
        "Validation metrics: "
        + ", ".join(
            f"{k}={v}" for k, v in metrics.items() if k != "val_map50_per_class"
        )
    )
    return metrics


# ----------------------------
# Visualization
# ----------------------------

def draw_gt_boxes(draw: ImageDraw.ImageDraw, anns, color="lime"):
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x2 = x + w
        y2 = y + h
        draw.rectangle((x, y, x2, y2), outline=color, width=3)


def draw_pred_boxes(draw: ImageDraw.ImageDraw, preds, id2label, color="red", max_boxes=40):
    scores = preds["scores"].detach().cpu().tolist()
    labels = preds["labels"].detach().cpu().tolist()
    boxes = preds["boxes"].detach().cpu().tolist()

    order = list(range(len(scores)))
    order.sort(key=lambda i: scores[i], reverse=True)

    for i in order[:max_boxes]:
        x1, y1, x2, y2 = boxes[i]
        label_id = int(labels[i])
        score = float(scores[i])
        name = id2label.get(label_id, str(label_id))
        text = f"{name[:24]} {score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), text, fill=color)


@torch.no_grad()
def save_validation_visualizations(
    model,
    dataloader,
    device,
    image_processor,
    id2label,
    output_dir: Path,
    epoch: int,
    score_threshold: float,
    num_images: int = 4,
):
    ensure_dir(output_dir)
    model.eval()

    saved = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )

        target_sizes = torch.tensor(batch["orig_size_hw"], device=device)
        predictions = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )

        for image_pil, gt_anns, pred, image_id in zip(
            batch["image_pil"],
            batch["gt_annotations"],
            predictions,
            batch["image_ids"],
        ):
            vis = image_pil.copy()
            draw = ImageDraw.Draw(vis)
            draw_gt_boxes(draw, gt_anns, color="lime")
            draw_pred_boxes(draw, pred, id2label=id2label, color="red", max_boxes=60)

            out_path = output_dir / f"epoch_{epoch:03d}_image_{int(image_id):05d}.jpg"
            vis.save(out_path, quality=95)

            saved += 1
            if saved >= num_images:
                return


# ----------------------------
# Scheduler
# ----------------------------

def build_warmup_cosine_lambda(total_optimizer_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int):
        if total_optimizer_steps <= 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ----------------------------
# Main
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)

    parser.add_argument("--shortest-edge", type=int, default=960)
    parser.add_argument("--longest-edge", type=int, default=1600)

    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--visualize-every", type=int, default=1)
    parser.add_argument("--num-vis-images", type=int, default=4)

    parser.add_argument("--use-aug", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    runs_root = script_dir / args.run_name
    ensure_dir(runs_root)
    ensure_dir(runs_root / "checkpoints")
    ensure_dir(runs_root / "visualizations")
    ensure_dir(runs_root / "split")

    logger = RunLogger(runs_root, args.run_name)
    logger.log(f"Run started: {args.run_name}")
    logger.log(f"Arguments: {vars(args)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    logger.log(f"Using device={device}, mixed_precision={use_amp}")

    annotations_path = args.data_root / "annotations.json"
    images_dir = args.data_root / "images"

    coco = load_coco(annotations_path)

    categories = sorted(coco["categories"], key=lambda x: x["id"])
    id2label = {int(cat["id"]): cat["name"] for cat in categories}
    label2id = {cat["name"]: int(cat["id"]) for cat in categories}

    annotations_by_image, sanitize_stats = sanitize_annotations(coco)
    images = list(coco["images"])

    logger.log(f"Num images: {len(images)}")
    logger.log(f"Num categories: {len(categories)}")
    logger.log(f"Num raw annotations: {len(coco['annotations'])}")
    logger.log(f"Sanitization: {sanitize_stats}")

    if not (0 <= args.fold_index < args.num_folds):
        raise ValueError("--fold-index must be in [0, num_folds-1]")

    folds = stable_multilabel_kfold(
        images=images,
        annotations_by_image=annotations_by_image,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    val_images = folds[args.fold_index]
    train_images = [img for i, fold in enumerate(folds) if i != args.fold_index for img in fold]

    logger.log(f"Fold {args.fold_index}/{args.num_folds - 1}")
    logger.log(f"Train images: {len(train_images)}")
    logger.log(f"Val images: {len(val_images)}")

    json_dump(
        {
            "train_image_ids": [int(x["id"]) for x in train_images],
            "val_image_ids": [int(x["id"]) for x in val_images],
        },
        runs_root / "split" / "split_ids.json",
    )

    # Aspect-ratio-preserving resize.
    image_processor = AutoImageProcessor.from_pretrained(
        args.checkpoint,
        do_resize=True,
        size={
            "shortest_edge": args.shortest_edge,
            "longest_edge": args.longest_edge,
        },
        do_pad=True,
    )

    # Increase object query count for dense shelves.
    model = DeformableDetrForObjectDetection.from_pretrained(
        args.checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        num_queries=100,
    )
    model.to(device)

    train_transforms = build_transforms(args.use_aug)

    train_dataset = CocoRetailDetectionDataset(
        images=train_images,
        annotations_by_image=annotations_by_image,
        images_dir=images_dir,
        image_processor=image_processor,
        transforms=train_transforms,
        return_visualization_data=False,
    )
    val_dataset = CocoRetailDetectionDataset(
        images=val_images,
        annotations_by_image=annotations_by_image,
        images_dir=images_dir,
        image_processor=image_processor,
        transforms=None,
        return_visualization_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=lambda batch: collate_fn(batch, image_processor),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=lambda batch: collate_fn(batch, image_processor),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    total_optimizer_steps = args.epochs * optimizer_steps_per_epoch
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_warmup_cosine_lambda(total_optimizer_steps, warmup_steps),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_map50 = -1.0
    history = []
    global_step = 0
    optimizer_step_count = 0

    config_snapshot = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sanitize_stats": sanitize_stats,
        "num_images": len(images),
        "num_categories": len(categories),
        "num_annotations_raw": len(coco["annotations"]),
    }
    json_dump(config_snapshot, runs_root / "config.json")

    logger.log("Training loop started")

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_batches = 0
        loss_dict_sums = defaultdict(float)

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for batch_idx, batch in enumerate(pbar, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
            labels = [{k: v.to(device) for k, v in label.items()} for label in batch["labels"]]

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if batch_idx % args.grad_accum_steps == 0 or batch_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step_count += 1

            train_loss_sum += float(outputs.loss.item())
            train_batches += 1

            for k, v in outputs.loss_dict.items():
                loss_dict_sums[k] += float(v.detach().item())

            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{outputs.loss.item():.4f}",
                lr=f"{current_lr:.2e}",
            )
            global_step += 1

        avg_train_loss = train_loss_sum / max(1, train_batches)
        avg_loss_dict = {f"train_{k}": round(v / max(1, train_batches), 6) for k, v in loss_dict_sums.items()}

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 10),
            "optimizer_steps": optimizer_step_count,
        }
        epoch_metrics.update(avg_loss_dict)

        if epoch % args.eval_every == 0:
            val_metrics = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                image_processor=image_processor,
                score_threshold=args.score_threshold,
                logger=logger,
            )
            epoch_metrics.update({k: v for k, v in val_metrics.items() if k != "val_map50_per_class"})

            if "val_map50_per_class" in val_metrics:
                json_dump(
                    val_metrics["val_map50_per_class"],
                    runs_root / f"per_class_map50_epoch_{epoch:03d}.json",
                )

            if epoch % args.visualize_every == 0:
                save_validation_visualizations(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    image_processor=image_processor,
                    id2label=id2label,
                    output_dir=runs_root / "visualizations",
                    epoch=epoch,
                    score_threshold=args.score_threshold,
                    num_images=args.num_vis_images,
                )

            current_map50 = float(epoch_metrics["val_map50"])
            if current_map50 > best_map50:
                best_map50 = current_map50
                best_dir = runs_root / "checkpoints" / "best_model"
                ensure_dir(best_dir)
                model.save_pretrained(best_dir)
                image_processor.save_pretrained(best_dir)

                json_dump(
                    {
                        "best_epoch": epoch,
                        "best_val_map50": best_map50,
                        "score_threshold": args.score_threshold,
                    },
                    best_dir / "best_info.json",
                )
                logger.log(f"New best model saved at epoch {epoch} with val_map50={best_map50:.6f}")

        last_dir = runs_root / "checkpoints" / "last_model"
        ensure_dir(last_dir)
        model.save_pretrained(last_dir)
        image_processor.save_pretrained(last_dir)

        history.append(epoch_metrics)
        logger.log_epoch_metrics(epoch_metrics)

        summary_keys = [
            "train_loss",
            "val_loss",
            "val_map",
            "val_map30",
            "val_map50",
            "val_map75",
            "lr",
        ]
        summary = ", ".join(
            f"{k}={epoch_metrics[k]}"
            for k in summary_keys
            if k in epoch_metrics
        )
        logger.log(f"Epoch {epoch} complete | {summary}")

        json_dump(history, runs_root / "history.json")

    logger.log("Training finished")
    logger.log(f"Best val_map50: {best_map50:.6f}")


if __name__ == "__main__":
    main()
