from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class ProductRecognizer(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        feature_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(feature_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        embeddings = self.embedding(self.dropout(features))
        embeddings = F.normalize(embeddings, dim=1)
        logits = self.classifier(embeddings)
        return embeddings, logits


def build_val_transform(image_size: int) -> Any:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.10)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_recognizer(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ProductRecognizer, dict[int, int], Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    category_to_index_raw = checkpoint["category_to_index"]
    category_to_index = {int(category_id): int(index) for category_id, index in category_to_index_raw.items()}

    model = ProductRecognizer(
        backbone_name=args["backbone"],
        embedding_dim=int(args["embedding_dim"]),
        num_classes=len(category_to_index),
        dropout=float(args["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    transform = build_val_transform(int(args["image_size"]))
    return model, category_to_index, transform


def crop_to_tensor_batches(
    image: Image.Image,
    boxes_xyxy: list[tuple[int, int, int, int]],
    transform: Any,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    current: list[torch.Tensor] = []
    rgb_image = image.convert("RGB")
    for x1, y1, x2, y2 in boxes_xyxy:
        crop = rgb_image.crop((x1, y1, x2, y2))
        current.append(transform(crop))
        if len(current) >= batch_size:
            batches.append(torch.stack(current).to(device))
            current = []
    if current:
        batches.append(torch.stack(current).to(device))
    return batches
