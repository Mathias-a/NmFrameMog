import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from tqdm import tqdm


DATA_ROOT = Path("/Users/matiasfernandezjr/Data/NM/train")
ANNOTATIONS_PATH = DATA_ROOT / "annotations.json"
IMAGES_DIR = DATA_ROOT / "images"

CHECKPOINT = "SenseTime/deformable-detr"
BATCH_SIZE = 2
NUM_EPOCHS = 1
LR = 1e-5
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.1
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_coco(annotation_file):
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco


class CocoDetectionForHF(Dataset):
    def __init__(self, images, annotations_by_image, images_dir, image_processor):
        self.images = images
        self.annotations_by_image = annotations_by_image
        self.images_dir = images_dir
        self.image_processor = image_processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]
        image_path = self.images_dir / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")

        anns = self.annotations_by_image.get(image_id, [])

        # Hugging Face object detection processors expect:
        # {"image_id": int, "annotations": [ ... ]}
        target = {
            "image_id": image_id,
            "annotations": anns,
        }

        encoding = self.image_processor(
        images=image,
        annotations=target,
        return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        item = {
            "pixel_values": pixel_values,
            "labels": labels,
        }

        if "pixel_mask" in encoding:
            item["pixel_mask"] = encoding["pixel_mask"].squeeze(0)

        return item


def collate_fn(batch, image_processor):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]

    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])

    return data


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    coco = load_coco(ANNOTATIONS_PATH)

    categories = sorted(coco["categories"], key=lambda x: x["id"])
    id2label = {cat["id"]: cat["name"] for cat in categories}
    label2id = {cat["name"]: cat["id"] for cat in categories}

    print(f"Num categories: {len(categories)}")
    print(f"Num images: {len(coco['images'])}")
    print(f"Num annotations: {len(coco['annotations'])}")

    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(
            {
                "id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],          # COCO format: [x, y, w, h]
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
            }
        )

    images = list(coco["images"])
    random.shuffle(images)

    n_val = max(1, int(len(images) * VAL_RATIO))
    val_images = images[:n_val]
    train_images = images[n_val:]

    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

    image_processor = AutoImageProcessor.from_pretrained(
    CHECKPOINT,
    do_resize=True,
    size={"height": 800, "width": 800},
)

    model = DeformableDetrForObjectDetection.from_pretrained(
        CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_dataset = CocoDetectionForHF(
        images=train_images,
        annotations_by_image=annotations_by_image,
        images_dir=IMAGES_DIR,
        image_processor=image_processor,
    )
    val_dataset = CocoDetectionForHF(
        images=val_images,
        annotations_by_image=annotations_by_image,
        images_dir=IMAGES_DIR,
        image_processor=image_processor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, image_processor),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, image_processor),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_sum = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - train")
        for batch in train_bar:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [
                {k: v.to(device) for k, v in label.items()}
                for label in batch["labels"]
            ]

            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - val")
            for batch in val_bar:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = [
                    {k: v.to(device) for k, v in label.items()}
                    for label in batch["labels"]
                ]

                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels,
                )

                loss = outputs.loss
                val_loss_sum += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss_sum / max(1, len(val_loader))

        print(f"\nEpoch {epoch+1} done")
        print(f"Average train loss: {avg_train_loss:.4f}")
        print(f"Average val loss:   {avg_val_loss:.4f}")

    out_dir = Path("./deformable_detr_nm_level1")
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    image_processor.save_pretrained(out_dir)

    print(f"Saved model and processor to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()