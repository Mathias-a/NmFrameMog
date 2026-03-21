from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

from recognizer_model import crop_to_tensor_batches, load_recognizer

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
DETECTOR_WEIGHTS = Path("detector.pt")
RECOGNIZER_WEIGHTS = Path("recognizer.pt")
DETECTOR_IMGSZ = 1280
DETECTOR_CONF = 0.02
DETECTOR_IOU = 0.60
DETECTOR_MAX_DET = 350
RECOGNIZER_BATCH_SIZE = 64
CROP_PADDING_RATIO = 0.06


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NorgesGruppen final submission runner.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def patch_torch_load_for_trusted_checkpoints() -> None:
    if getattr(torch.load, "_ng_submission_patch", False):
        return
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    patched_torch_load._ng_submission_patch = True  # type: ignore[attr-defined]
    torch.load = patched_torch_load  # type: ignore[assignment]


def infer_image_id(image_path: Path) -> int:
    matches = re.findall(r"\d+", image_path.stem)
    if not matches:
        raise ValueError(f"Could not infer image id from filename: {image_path.name}")
    return int(matches[-1])


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def padded_xyxy(box_xywh: list[float], width: int, height: int, padding_ratio: float) -> tuple[int, int, int, int]:
    x, y, w, h = box_xywh
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio
    x1 = max(0, int(math.floor(x - pad_w)))
    y1 = max(0, int(math.floor(y - pad_h)))
    x2 = min(width, int(math.ceil(x + w + pad_w)))
    y2 = min(height, int(math.ceil(y + h + pad_h)))
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    return x1, y1, x2, y2


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_torch_load_for_trusted_checkpoints()

    detector = YOLO(str(DETECTOR_WEIGHTS))
    recognizer, category_to_index, transform = load_recognizer(RECOGNIZER_WEIGHTS, device=device)
    index_to_category_id = {index: category_id for category_id, index in category_to_index.items()}

    predictions: list[dict[str, object]] = []
    for image_path in list_images(input_dir):
        image_id = infer_image_id(image_path)
        with Image.open(image_path) as pil_image:
            width, height = pil_image.size
            detection_results = detector.predict(
                source=str(image_path),
                imgsz=DETECTOR_IMGSZ,
                conf=DETECTOR_CONF,
                iou=DETECTOR_IOU,
                device=0 if device.type == "cuda" else "cpu",
                max_det=DETECTOR_MAX_DET,
                verbose=False,
            )

            boxes_xywh: list[list[float]] = []
            detector_scores: list[float] = []
            crop_boxes_xyxy: list[tuple[int, int, int, int]] = []

            for result in detection_results:
                if result.boxes is None:
                    continue
                for box_index in range(len(result.boxes)):
                    x1, y1, x2, y2 = result.boxes.xyxy[box_index].tolist()
                    xywh = [x1, y1, x2 - x1, y2 - y1]
                    boxes_xywh.append(xywh)
                    detector_scores.append(float(result.boxes.conf[box_index].item()))
                    crop_boxes_xyxy.append(padded_xyxy(xywh, width, height, CROP_PADDING_RATIO))

            if not boxes_xywh:
                continue

            all_logits: list[torch.Tensor] = []
            with torch.inference_mode():
                for batch in crop_to_tensor_batches(
                    image=pil_image,
                    boxes_xyxy=crop_boxes_xyxy,
                    transform=transform,
                    batch_size=RECOGNIZER_BATCH_SIZE,
                    device=device,
                ):
                    _embeddings, logits = recognizer(batch)
                    all_logits.append(logits.detach().cpu())

            logits_tensor = torch.cat(all_logits, dim=0)
            probabilities = torch.softmax(logits_tensor, dim=1)
            top_probabilities, top_indices = probabilities.max(dim=1)

            for box_xywh, detector_score, cls_probability, cls_index in zip(
                boxes_xywh,
                detector_scores,
                top_probabilities.tolist(),
                top_indices.tolist(),
            ):
                category_id = int(index_to_category_id[int(cls_index)])
                joint_score = float(detector_score * math.sqrt(max(cls_probability, 1e-8)))
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [round(box_xywh[0], 1), round(box_xywh[1], 1), round(box_xywh[2], 1), round(box_xywh[3], 1)],
                        "score": round(joint_score, 4),
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
