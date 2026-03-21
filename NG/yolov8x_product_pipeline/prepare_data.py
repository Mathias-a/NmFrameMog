from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    EventLogger,
    clip_box,
    ensure_dir,
    greedy_multilabel_folds,
    normalize_product_name,
    read_json,
    seed_everything,
    slugify,
    write_json,
    write_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare NG data for the YOLOv8x pipeline.")
    parser.add_argument("--data-root", required=True, help="Root directory that contains train/ and NM_NGD_product_images/.")
    parser.add_argument("--workspace-root", required=True, help="Workspace root for prepared artifacts and runs.")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--crop-padding", type=float, default=0.08)
    parser.add_argument(
        "--category-overrides",
        default=str(Path(__file__).resolve().parent / "category_name_overrides.json"),
        help="JSON map from annotation category name to metadata product_name.",
    )
    return parser


def prepare_workspace(
    data_root: Path,
    workspace_root: Path,
    num_folds: int,
    seed: int,
    crop_padding: float,
    category_overrides_path: Path,
) -> dict[str, Any]:
    seed_everything(seed)

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for prepare_data.py") from exc

    prepared_root = ensure_dir(workspace_root / "prepared")
    yolo_root = ensure_dir(prepared_root / "yolo")
    recognizer_root = ensure_dir(prepared_root / "recognizer")
    detector_dataset_root = ensure_dir(yolo_root / "dataset")
    one_class_root = ensure_dir(detector_dataset_root / "one_class")
    multi_class_root = ensure_dir(detector_dataset_root / "multi_class")
    gt_crop_root = ensure_dir(recognizer_root / "gt_crops")
    event_logger = EventLogger(prepared_root / "prepare_events.jsonl")

    train_root = data_root / "train"
    image_root = train_root / "images"
    annotation_path = train_root / "annotations.json"
    reference_root = data_root / "NM_NGD_product_images"
    metadata_path = reference_root / "metadata.json"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {annotation_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    annotations = read_json(annotation_path)
    metadata = read_json(metadata_path)
    overrides = read_json(category_overrides_path) if category_overrides_path.exists() else {}

    images = annotations["images"]
    categories = annotations["categories"]
    anns = annotations["annotations"]
    image_by_id = {image["id"]: image for image in images}
    category_by_id = {category["id"]: category for category in categories}
    image_annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    image_labels: dict[int, set[int]] = defaultdict(set)
    annotation_counts = Counter()

    for ann in anns:
        image_annotations[ann["image_id"]].append(ann)
        image_labels[ann["image_id"]].add(ann["category_id"])
        annotation_counts[ann["category_id"]] += 1

    fold_assignment = greedy_multilabel_folds(
        image_ids=[image["id"] for image in images],
        image_labels=image_labels,
        label_counts=dict(annotation_counts),
        num_folds=num_folds,
    )

    metadata_by_name = {product["product_name"]: product for product in metadata["products"]}
    metadata_by_normalized_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for product in metadata["products"]:
        metadata_by_normalized_name[normalize_product_name(product["product_name"])].append(product)

    mapped_categories = []
    unresolved_categories = []
    for category in categories:
        category_name = category["name"]
        matched_product: dict[str, Any] | None = None
        matched_by = "none"
        if category_name in overrides:
            override_name = overrides[category_name]
            matched_product = metadata_by_name.get(override_name)
            matched_by = "override"
        if matched_product is None and category_name in metadata_by_name:
            matched_product = metadata_by_name[category_name]
            matched_by = "exact"
        if matched_product is None:
            candidates = metadata_by_normalized_name.get(normalize_product_name(category_name), [])
            if len(candidates) == 1:
                matched_product = candidates[0]
                matched_by = "normalized"

        record = {
            "category_id": category["id"],
            "category_name": category_name,
            "annotation_count": annotation_counts[category["id"]],
            "matched": matched_product is not None,
            "matched_by": matched_by,
        }
        if matched_product is not None:
            record["product_code"] = matched_product["product_code"]
            record["product_name"] = matched_product["product_name"]
        mapped_categories.append(record)
        if matched_product is None:
            unresolved_categories.append(record)

    write_json(recognizer_root / "category_reference_mapping.json", mapped_categories)
    write_json(recognizer_root / "unresolved_categories.json", unresolved_categories)
    event_logger.log(
        "category_mapping_complete",
        matched_count=sum(1 for record in mapped_categories if record["matched"]),
        unresolved_count=len(unresolved_categories),
    )

    one_class_images_dir = ensure_dir(one_class_root / "images")
    one_class_labels_dir = ensure_dir(one_class_root / "labels")
    multi_class_images_dir = ensure_dir(multi_class_root / "images")
    multi_class_labels_dir = ensure_dir(multi_class_root / "labels")

    gt_crop_manifest: list[dict[str, Any]] = []
    reference_manifest: list[dict[str, Any]] = []
    image_manifest: list[dict[str, Any]] = []
    image_sizes = Counter((image["width"], image["height"]) for image in images)
    annotations_per_image = [len(image_annotations[image["id"]]) for image in images]

    one_class_train_lists = {fold: [] for fold in range(num_folds)}
    one_class_val_lists = {fold: [] for fold in range(num_folds)}
    multi_class_train_lists = {fold: [] for fold in range(num_folds)}
    multi_class_val_lists = {fold: [] for fold in range(num_folds)}

    for image in images:
        image_id = image["id"]
        file_name = image["file_name"]
        source_image_path = image_root / file_name
        if not source_image_path.exists():
            raise FileNotFoundError(f"Missing training image: {source_image_path}")

        fold_index = fold_assignment[image_id]
        image_stem = Path(file_name).stem
        one_class_image_link = one_class_images_dir / file_name
        multi_class_image_link = multi_class_images_dir / file_name
        if not one_class_image_link.exists():
            one_class_image_link.symlink_to(source_image_path)
        if not multi_class_image_link.exists():
            multi_class_image_link.symlink_to(source_image_path)

        one_class_label_path = one_class_labels_dir / f"{image_stem}.txt"
        multi_class_label_path = multi_class_labels_dir / f"{image_stem}.txt"
        one_class_lines: list[str] = []
        multi_class_lines: list[str] = []

        for ann in image_annotations[image_id]:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2.0) / image["width"]
            cy = (y + h / 2.0) / image["height"]
            nw = w / image["width"]
            nh = h / image["height"]
            one_class_lines.append(f"0 {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}")
            multi_class_lines.append(f"{ann['category_id']} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}")

        write_text(one_class_label_path, "\n".join(one_class_lines) + ("\n" if one_class_lines else ""))
        write_text(multi_class_label_path, "\n".join(multi_class_lines) + ("\n" if multi_class_lines else ""))

        one_class_path_str = str(one_class_image_link.resolve())
        multi_class_path_str = str(multi_class_image_link.resolve())
        for fold in range(num_folds):
            if fold == fold_index:
                one_class_val_lists[fold].append(one_class_path_str)
                multi_class_val_lists[fold].append(multi_class_path_str)
            else:
                one_class_train_lists[fold].append(one_class_path_str)
                multi_class_train_lists[fold].append(multi_class_path_str)

        image_manifest.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "image_path": str(source_image_path),
                "width": image["width"],
                "height": image["height"],
                "fold": fold_index,
                "annotation_count": len(image_annotations[image_id]),
                "labels": sorted(image_labels[image_id]),
            }
        )

        with Image.open(source_image_path) as source_image:
            rgb_image = source_image.convert("RGB")
            for ann in image_annotations[image_id]:
                category = category_by_id[ann["category_id"]]
                crop_dir = ensure_dir(gt_crop_root / f"fold_{fold_index}" / f"class_{ann['category_id']:03d}")
                crop_name = (
                    f"{image_stem}_ann{ann['id']:05d}_"
                    f"{slugify(category['name'])}.jpg"
                )
                crop_path = crop_dir / crop_name
                x1, y1, x2, y2 = clip_box(
                    [
                        ann["bbox"][0] - ann["bbox"][2] * crop_padding,
                        ann["bbox"][1] - ann["bbox"][3] * crop_padding,
                        ann["bbox"][2] * (1.0 + crop_padding * 2.0),
                        ann["bbox"][3] * (1.0 + crop_padding * 2.0),
                    ],
                    image["width"],
                    image["height"],
                )
                if not crop_path.exists():
                    crop = rgb_image.crop((x1, y1, x2, y2))
                    crop.save(crop_path, quality=95)
                gt_crop_manifest.append(
                    {
                        "annotation_id": ann["id"],
                        "image_id": image_id,
                        "image_path": str(source_image_path),
                        "fold": fold_index,
                        "category_id": ann["category_id"],
                        "category_name": category["name"],
                        "crop_path": str(crop_path),
                        "bbox": ann["bbox"],
                        "crop_box_xyxy": [x1, y1, x2, y2],
                        "source": "shelf_gt",
                    }
                )

    mapped_by_category_name = {record["category_name"]: record for record in mapped_categories if record["matched"]}
    for product_dir in sorted(reference_root.iterdir()):
        if not product_dir.is_dir():
            continue
        category_records = [
            record for record in mapped_categories if record.get("product_code") == product_dir.name
        ]
        category_id = category_records[0]["category_id"] if len(category_records) == 1 else None
        category_name = category_records[0]["category_name"] if len(category_records) == 1 else None
        for image_path in sorted(product_dir.glob("*.jpg")):
            reference_manifest.append(
                {
                    "product_code": product_dir.name,
                    "product_name": metadata_by_name.get(category_name, {}).get("product_name", category_name),
                    "category_id": category_id,
                    "category_name": category_name,
                    "reference_image_type": image_path.stem,
                    "image_path": str(image_path),
                    "source": "reference",
                }
            )

    write_json(prepared_root / "image_manifest.json", image_manifest)
    write_json(recognizer_root / "gt_crops_manifest.json", gt_crop_manifest)
    write_json(recognizer_root / "reference_manifest.json", reference_manifest)
    write_json(prepared_root / "fold_assignment.json", fold_assignment)

    detector_fold_specs = []
    for fold in range(num_folds):
        fold_dir = ensure_dir(yolo_root / f"fold_{fold}")
        train_one_class_txt = fold_dir / "train_one_class.txt"
        val_one_class_txt = fold_dir / "val_one_class.txt"
        train_multi_class_txt = fold_dir / "train_multi_class.txt"
        val_multi_class_txt = fold_dir / "val_multi_class.txt"

        write_text(train_one_class_txt, "\n".join(sorted(one_class_train_lists[fold])) + "\n")
        write_text(val_one_class_txt, "\n".join(sorted(one_class_val_lists[fold])) + "\n")
        write_text(train_multi_class_txt, "\n".join(sorted(multi_class_train_lists[fold])) + "\n")
        write_text(val_multi_class_txt, "\n".join(sorted(multi_class_val_lists[fold])) + "\n")

        one_class_yaml = fold_dir / "one_class.yaml"
        multi_class_yaml = fold_dir / "multi_class.yaml"
        one_class_yaml_text = "\n".join(
            [
                f"path: {fold_dir}",
                f"train: {train_one_class_txt}",
                f"val: {val_one_class_txt}",
                "names:",
                "  0: product",
                "nc: 1",
                "",
            ]
        )
        multi_class_yaml_text = "\n".join(
            [
                f"path: {fold_dir}",
                f"train: {train_multi_class_txt}",
                f"val: {val_multi_class_txt}",
                "names:",
                *[
                    f"  {category['id']}: \"{category['name'].replace(chr(34), '')}\""
                    for category in categories
                ],
                f"nc: {len(categories)}",
                "",
            ]
        )
        write_text(one_class_yaml, one_class_yaml_text)
        write_text(multi_class_yaml, multi_class_yaml_text)
        detector_fold_specs.append(
            {
                "fold": fold,
                "one_class_yaml": str(one_class_yaml),
                "multi_class_yaml": str(multi_class_yaml),
                "train_count": len(one_class_train_lists[fold]),
                "val_count": len(one_class_val_lists[fold]),
            }
        )

    dataset_summary = {
        "images": len(images),
        "annotations": len(anns),
        "categories": len(categories),
        "num_folds": num_folds,
        "num_unique_image_sizes": len(image_sizes),
        "image_size_counts_top10": [
            {"size": [width, height], "count": count}
            for (width, height), count in image_sizes.most_common(10)
        ],
        "annotations_per_image_mean": statistics.mean(annotations_per_image),
        "annotations_per_image_median": statistics.median(annotations_per_image),
        "annotations_per_image_min": min(annotations_per_image),
        "annotations_per_image_max": max(annotations_per_image),
        "annotations_per_class_mean": statistics.mean(annotation_counts.values()),
        "annotations_per_class_median": statistics.median(annotation_counts.values()),
        "annotations_per_class_min": min(annotation_counts.values()),
        "annotations_per_class_max": max(annotation_counts.values()),
        "classes_lt_10": sum(1 for count in annotation_counts.values() if count < 10),
        "classes_lt_20": sum(1 for count in annotation_counts.values() if count < 20),
        "classes_ge_100": sum(1 for count in annotation_counts.values() if count >= 100),
        "reference_products_total": metadata["total_products"],
        "reference_products_with_images": metadata["products_with_images"],
        "reference_images_total": metadata["total_images"],
        "matched_reference_categories": sum(1 for record in mapped_categories if record["matched"]),
        "unresolved_reference_categories": len(unresolved_categories),
        "detector_folds": detector_fold_specs,
    }
    write_json(prepared_root / "dataset_summary.json", dataset_summary)

    human_summary = [
        "# Prepared Dataset Summary",
        "",
        f"- Images: {dataset_summary['images']}",
        f"- Annotations: {dataset_summary['annotations']}",
        f"- Categories: {dataset_summary['categories']}",
        f"- Matched reference categories: {dataset_summary['matched_reference_categories']}",
        f"- Unresolved reference categories: {dataset_summary['unresolved_reference_categories']}",
        f"- Mean annotations/image: {dataset_summary['annotations_per_image_mean']:.2f}",
        f"- Median annotations/class: {dataset_summary['annotations_per_class_median']}",
        f"- Classes with < 10 annotations: {dataset_summary['classes_lt_10']}",
        f"- Classes with < 20 annotations: {dataset_summary['classes_lt_20']}",
        "",
        "## Fold counts",
        "",
    ]
    for fold_spec in detector_fold_specs:
        human_summary.append(
            f"- Fold {fold_spec['fold']}: train={fold_spec['train_count']} val={fold_spec['val_count']}"
        )
    write_text(prepared_root / "prepare_summary.md", "\n".join(human_summary) + "\n")

    event_logger.log(
        "prepare_complete",
        dataset_summary=dataset_summary,
        workspace_root=str(workspace_root),
    )
    return dataset_summary


def main() -> int:
    args = build_parser().parse_args()
    prepare_workspace(
        data_root=Path(args.data_root).resolve(),
        workspace_root=Path(args.workspace_root).resolve(),
        num_folds=args.num_folds,
        seed=args.seed,
        crop_padding=args.crop_padding,
        category_overrides_path=Path(args.category_overrides).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
