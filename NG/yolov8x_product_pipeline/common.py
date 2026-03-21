from __future__ import annotations

import csv
import json
import math
import random
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    serializable = _to_serializable(payload)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return normalized or "item"


def normalize_product_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def parse_fold_indices(fold_count: int, selected: str | None) -> list[int]:
    if selected is None or selected.strip() == "" or selected.strip().lower() == "all":
        return list(range(fold_count))
    indices = []
    for chunk in selected.split(","):
        value = int(chunk.strip())
        if value < 0 or value >= fold_count:
            raise ValueError(f"Fold index {value} is outside 0..{fold_count - 1}")
        indices.append(value)
    return sorted(set(indices))


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def patch_torch_load_for_trusted_checkpoints() -> None:
    try:
        import torch
    except Exception:
        return

    if getattr(torch.load, "_ng_trusted_patch", False):
        return

    original_torch_load = torch.load

    def patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    patched_torch_load._ng_trusted_patch = True  # type: ignore[attr-defined]
    torch.load = patched_torch_load  # type: ignore[assignment]
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def greedy_multilabel_folds(
    image_ids: Sequence[int],
    image_labels: dict[int, set[int]],
    label_counts: dict[int, int],
    num_folds: int,
) -> dict[int, int]:
    fold_image_counts = [0 for _ in range(num_folds)]
    fold_label_counts = [dict.fromkeys(label_counts.keys(), 0) for _ in range(num_folds)]

    def rarity_score(img_id: int) -> tuple[float, int, int]:
        labels = image_labels[img_id]
        rarity = sum(1.0 / max(label_counts[label], 1) for label in labels)
        return (-rarity, -len(labels), img_id)

    assignment: dict[int, int] = {}
    for image_id in sorted(image_ids, key=rarity_score):
        labels = image_labels[image_id]
        best_fold = 0
        best_score: tuple[float, float, int] | None = None
        for fold_index in range(num_folds):
            fold_label_score = 0.0
            for label in labels:
                total = max(label_counts[label], 1)
                fold_label_score += (fold_label_counts[fold_index][label] + 1) / total
            if labels:
                fold_label_score /= len(labels)
            fold_size_score = fold_image_counts[fold_index] / max(1, len(image_ids))
            candidate = (fold_label_score, fold_size_score, fold_index)
            if best_score is None or candidate < best_score:
                best_score = candidate
                best_fold = fold_index
        assignment[image_id] = best_fold
        fold_image_counts[best_fold] += 1
        for label in labels:
            fold_label_counts[best_fold][label] += 1
    return assignment


def xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def clip_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = max(0, min(width - 1, int(math.floor(x))))
    y1 = max(0, min(height - 1, int(math.floor(y))))
    x2 = max(x1 + 1, min(width, int(math.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(math.ceil(y + h))))
    return x1, y1, x2, y2


def padded_box(
    box: Sequence[float],
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio
    expanded = [x - pad_w, y - pad_h, w + 2 * pad_w, h + 2 * pad_h]
    return clip_box(expanded, width, height)


def copy_files_if_present(src_dir: Path, dst_dir: Path, patterns: Iterable[str]) -> list[str]:
    copied: list[str] = []
    ensure_dir(dst_dir)
    for pattern in patterns:
        for path in src_dir.glob(pattern):
            destination = dst_dir / path.name
            shutil.copy2(path, destination)
            copied.append(destination.name)
    return sorted(copied)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_line = "| " + " | ".join(str(header) for header in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


@dataclass
class EventLogger:
    events_path: Path

    def log(self, event_type: str, **payload: Any) -> None:
        append_jsonl(
            self.events_path,
            {
                "timestamp": now_utc_iso(),
                "event_type": event_type,
                **payload,
            },
        )


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value
