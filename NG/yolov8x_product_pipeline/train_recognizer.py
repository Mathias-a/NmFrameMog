from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    EventLogger,
    ensure_dir,
    parse_bool,
    parse_fold_indices,
    read_json,
    seed_everything,
    write_json,
    write_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the SKU recognizer for the NG pipeline.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--selected-folds", default="all")
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--backbone", default="convnextv2_base.fcmae_ft_in22k_in1k")
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--weight-decay", type=float, default=0.00005)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--supcon-weight", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--prototype-temperature", type=float, default=0.10)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--amp", default="true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-detector-oof", default="true")
    return parser


@dataclass
class SampleRecord:
    image_path: str
    category_id: int
    category_name: str
    fold: int
    source: str


def _build_transforms(image_size: int) -> tuple[Any, Any]:
    try:
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError("torchvision is required for train_recognizer.py") from exc

    train_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.12)),
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)],
                p=0.8,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.20),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.15), ratio=(0.3, 3.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.10)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


class ProductCropDataset:
    def __init__(self, samples: list[SampleRecord], category_to_index: dict[int, int], transform: Any) -> None:
        self.samples = samples
        self.category_to_index = category_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        from PIL import Image

        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.category_to_index[sample.category_id], sample.image_path


class ProductRecognizerModel:
    def __init__(self, backbone_name: str, embedding_dim: int, num_classes: int, dropout: float) -> None:
        try:
            import timm
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch and timm are required for train_recognizer.py") from exc

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="avg")
                feature_dim = self.backbone.num_features
                self.dropout = nn.Dropout(dropout)
                self.embedding = nn.Linear(feature_dim, embedding_dim)
                self.classifier = nn.Linear(embedding_dim, num_classes)

            def forward(self, x: Any) -> tuple[Any, Any]:
                features = self.backbone(x)
                embeddings = self.embedding(self.dropout(features))
                embeddings = F.normalize(embeddings, dim=1)
                logits = self.classifier(embeddings)
                return embeddings, logits

        self.module = Model()


class SupConLoss:
    def __init__(self, temperature: float = 0.07) -> None:
        self.temperature = temperature

    def __call__(self, features: Any, labels: Any) -> Any:
        import torch
        import torch.nn.functional as F

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        similarity = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=features.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp_min(1.0)
        loss = -mean_log_prob_pos.mean()
        return loss


def _load_samples(workspace_root: Path, run_name: str, use_detector_oof: bool) -> tuple[list[SampleRecord], list[SampleRecord]]:
    recognizer_root = workspace_root / "prepared" / "recognizer"
    gt_samples = [
        SampleRecord(
            image_path=entry["crop_path"],
            category_id=entry["category_id"],
            category_name=entry["category_name"],
            fold=entry["fold"],
            source=entry["source"],
        )
        for entry in read_json(recognizer_root / "gt_crops_manifest.json")
    ]
    reference_samples = []
    for entry in read_json(recognizer_root / "reference_manifest.json"):
        if entry["category_id"] is None:
            continue
        reference_samples.append(
            SampleRecord(
                image_path=entry["image_path"],
                category_id=entry["category_id"],
                category_name=entry["category_name"],
                fold=-1,
                source="reference",
            )
        )

    detector_samples: list[SampleRecord] = []
    if use_detector_oof:
        detector_manifest = (
            workspace_root / "prepared" / "recognizer" / "oof_detector_crops" / run_name / "manifest.json"
        )
        if detector_manifest.exists():
            detector_samples = [
                SampleRecord(
                    image_path=entry["crop_path"],
                    category_id=entry["category_id"],
                    category_name=entry["category_name"],
                    fold=entry["fold"],
                    source="detector_oof",
                )
                for entry in read_json(detector_manifest)
            ]
    return gt_samples, reference_samples + detector_samples


def _make_loaders(
    args: argparse.Namespace,
    selected_fold: int,
    category_ids: list[int],
) -> tuple[Any, Any, dict[int, int], list[SampleRecord], list[SampleRecord]]:
    try:
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for train_recognizer.py") from exc

    train_transform, val_transform = _build_transforms(args.image_size)
    gt_samples, auxiliary_samples = _load_samples(
        workspace_root=Path(args.workspace_root).resolve(),
        run_name=args.run_name,
        use_detector_oof=parse_bool(args.use_detector_oof),
    )

    train_samples = [sample for sample in gt_samples if sample.fold != selected_fold]
    val_samples = [sample for sample in gt_samples if sample.fold == selected_fold]
    train_samples.extend(auxiliary_samples)

    category_to_index = {category_id: index for index, category_id in enumerate(sorted(category_ids))}
    train_dataset = ProductCropDataset(train_samples, category_to_index, train_transform)
    val_dataset = ProductCropDataset(val_samples, category_to_index, val_transform)

    label_counter = Counter(sample.category_id for sample in train_samples)
    weights = [1.0 / math.sqrt(label_counter[sample.category_id]) for sample in train_samples]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=max(2, args.num_workers // 2),
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader, category_to_index, train_samples, val_samples


def train_recognizer_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.cuda.amp import GradScaler, autocast
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch and numpy are required for train_recognizer.py") from exc

    seed_everything(args.seed)
    workspace_root = Path(args.workspace_root).resolve()
    prepared_root = workspace_root / "prepared"
    dataset_summary = read_json(prepared_root / "dataset_summary.json")
    selected_folds = parse_fold_indices(dataset_summary["num_folds"], args.selected_folds)
    all_category_ids = sorted({entry["category_id"] for entry in read_json(prepared_root / "recognizer" / "gt_crops_manifest.json")})

    run_root = ensure_dir(workspace_root / "runs" / args.run_name)
    recognizer_root = ensure_dir(run_root / "recognizer")
    write_json(recognizer_root / "config.json", vars(args))
    event_logger = EventLogger(run_root / "events.jsonl")
    fold_summaries: list[dict[str, Any]] = []

    for fold in selected_folds:
        fold_root = ensure_dir(recognizer_root / f"fold_{fold}")
        train_loader, val_loader, category_to_index, train_samples, val_samples = _make_loaders(
            args=args,
            selected_fold=fold,
            category_ids=all_category_ids,
        )
        inverse_category_index = {index: category_id for category_id, index in category_to_index.items()}

        model = ProductRecognizerModel(
            backbone_name=args.backbone,
            embedding_dim=args.embedding_dim,
            num_classes=len(category_to_index),
            dropout=args.dropout,
        ).module
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.lr * 0.05)
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        criterion_supcon = SupConLoss(temperature=args.prototype_temperature)
        scaler = GradScaler(enabled=parse_bool(args.amp))

        metrics_rows: list[dict[str, Any]] = []
        best_metric = float("-inf")
        best_summary: dict[str, Any] | None = None

        print(
            f"[recognizer] fold {fold} starting | "
            f"train_samples={len(train_samples)} val_samples={len(val_samples)} "
            f"backbone={args.backbone} image_size={args.image_size} batch_size={args.batch_size} "
            f"val_every={args.val_every}"
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_ce = 0.0
            total_supcon = 0.0
            total_samples = 0
            train_progress = tqdm(
                train_loader,
                desc=f"Fold {fold} Train {epoch:02d}/{args.epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            for images, labels, _paths in train_progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=parse_bool(args.amp)):
                    embeddings, logits = model(images)
                    ce_loss = criterion_ce(logits, labels)
                    supcon_loss = criterion_supcon(embeddings, labels)
                    loss = args.ce_weight * ce_loss + args.supcon_weight * supcon_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item()) * images.size(0)
                total_ce += float(ce_loss.item()) * images.size(0)
                total_supcon += float(supcon_loss.item()) * images.size(0)
                total_samples += images.size(0)
                train_progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ce=f"{ce_loss.item():.4f}",
                    supcon=f"{supcon_loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            if epoch > args.warmup_epochs:
                scheduler.step()

            should_validate = epoch == 1 or epoch == args.epochs or (epoch % max(1, args.val_every) == 0)
            val_loss = None
            val_top1 = None
            val_top5 = None
            confusion_samples: list[dict[str, Any]] = []

            if should_validate:
                model.eval()
                raw_val_loss = 0.0
                val_count = 0
                top1_correct = 0
                top5_correct = 0
                with torch.no_grad():
                    val_progress = tqdm(
                        val_loader,
                        desc=f"Fold {fold} Val   {epoch:02d}/{args.epochs}",
                        leave=False,
                        dynamic_ncols=True,
                    )
                    for images, labels, paths in val_progress:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        with autocast(enabled=parse_bool(args.amp)):
                            embeddings, logits = model(images)
                            ce_loss = criterion_ce(logits, labels)
                            supcon_loss = criterion_supcon(embeddings, labels)
                            loss = args.ce_weight * ce_loss + args.supcon_weight * supcon_loss
                        raw_val_loss += float(loss.item()) * images.size(0)
                        val_count += images.size(0)
                        top1 = logits.argmax(dim=1)
                        top1_correct += int((top1 == labels).sum().item())
                        top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
                        top5_correct += int((top5 == labels.unsqueeze(1)).any(dim=1).sum().item())
                        val_progress.set_postfix(
                            val_loss=f"{(raw_val_loss / max(val_count, 1)):.4f}",
                            top1=f"{(top1_correct / max(val_count, 1)):.4f}",
                            top5=f"{(top5_correct / max(val_count, 1)):.4f}",
                        )
                        if len(confusion_samples) < 32:
                            for path, predicted_index, label_index in zip(paths, top1.tolist(), labels.tolist()):
                                if predicted_index != label_index:
                                    confusion_samples.append(
                                        {
                                            "image_path": path,
                                            "predicted_category_id": inverse_category_index[predicted_index],
                                            "true_category_id": inverse_category_index[label_index],
                                        }
                                    )
                                if len(confusion_samples) >= 32:
                                    break
                val_loss = raw_val_loss / max(val_count, 1)
                val_top1 = top1_correct / max(val_count, 1)
                val_top5 = top5_correct / max(val_count, 1)

            row = {
                "epoch": epoch,
                "train_loss": total_loss / max(total_samples, 1),
                "train_ce_loss": total_ce / max(total_samples, 1),
                "train_supcon_loss": total_supcon / max(total_samples, 1),
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top5": val_top5,
                "lr": optimizer.param_groups[0]["lr"],
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "validated": should_validate,
            }
            metrics_rows.append(row)
            if should_validate:
                write_json(fold_root / f"epoch_{epoch:03d}_mistakes.json", confusion_samples)
            event_logger.log("recognizer_epoch", fold=fold, **row)
            if should_validate:
                print(
                    f"[recognizer] fold {fold} epoch {epoch:02d}/{args.epochs} | "
                    f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
                    f"val_top1={row['val_top1']:.4f} val_top5={row['val_top5']:.4f} "
                    f"lr={row['lr']:.2e}"
                )
            else:
                print(
                    f"[recognizer] fold {fold} epoch {epoch:02d}/{args.epochs} | "
                    f"train_loss={row['train_loss']:.4f} val=skipped lr={row['lr']:.2e}"
                )

            if should_validate and row["val_top1"] is not None and row["val_top1"] > best_metric:
                best_metric = row["val_top1"]
                best_summary = row
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "category_to_index": category_to_index,
                        "args": vars(args),
                    },
                    fold_root / "best.pt",
                )

        with (fold_root / "metrics_detailed.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

        if best_summary is None:
            best_summary = {}
        summary_payload = {
            "fold": fold,
            "best": best_summary,
            "best_checkpoint": str(fold_root / "best.pt"),
        }
        write_json(fold_root / "summary.json", summary_payload)
        fold_summaries.append(summary_payload)

    write_json(recognizer_root / "summary_all_folds.json", fold_summaries)
    lines = [f"# Recognizer Summary for `{args.run_name}`", ""]
    for summary in fold_summaries:
        best = summary["best"]
        lines.extend(
            [
                f"## Fold {summary['fold']}",
                "",
                f"- Best epoch: {best.get('epoch')}",
                f"- Validation top-1: {best.get('val_top1')}",
                f"- Validation top-5: {best.get('val_top5')}",
                f"- Validation loss: {best.get('val_loss')}",
                "",
            ]
        )
    write_text(recognizer_root / "human_summary.md", "\n".join(lines))
    return fold_summaries


def main() -> int:
    args = build_parser().parse_args()
    train_recognizer_runs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
