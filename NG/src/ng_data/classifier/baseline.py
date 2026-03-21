from __future__ import annotations

# pyright: reportMissingImports=false
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import timm  # type: ignore[import-untyped]
import torch
from PIL import Image, UnidentifiedImageError
from timm.data import (  # type: ignore[import-untyped]
    create_transform,
    resolve_data_config,
)

from src.ng_data.classifier.data import (
    build_classifier_crop_dataset,
    load_and_validate_classifier_data_config,
)
from src.ng_data.data.manifest import file_snapshot, write_json
from src.ng_data.eval.score import (
    ScoreValidationError,
    score_predictions,
    validate_predictions,
)

JsonDict = dict[str, Any]
CHECKPOINT_FORMAT_VERSION = "classifier-timm-state-dict-v1"
GT_BOXES_MODE = "gt_boxes"
DETECTOR_BOXES_MODE = "detector_boxes"
DETECTION_ONLY_CATEGORY_ID = 0


class ClassifierBaselineError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingOptions:
    artifact_name: str
    device: str
    mode: str
    weight_decay: float


@dataclass(frozen=True)
class ClassMapEntry:
    category_ids: tuple[int, ...]
    class_id: int
    product_code: str
    product_name: str
    reference_image_paths: tuple[str, ...]


@dataclass(frozen=True)
class CropRecord:
    annotation_id: int
    bbox_xywh: tuple[float, float, float, float]
    category_id: int
    class_id: int
    crop_path: str
    image_id: int
    source_image_path: str


@dataclass(frozen=True)
class LabeledImage:
    class_id: int
    image_path: Path
    source: str


@dataclass(frozen=True)
class ModelBundle:
    class_entries: list[ClassMapEntry]
    crop_manifest: JsonDict
    crop_records: list[CropRecord]
    metadata: JsonDict
    model: torch.nn.Module
    preprocessing: JsonDict
    processed_root: Path


def _fallback_rgb_image(image_path: Path) -> Image.Image:
    raw = image_path.read_bytes()
    if not raw:
        raw = image_path.as_posix().encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    size = 512
    pixel_count = size * size * 3
    repeated = (raw + digest) * ((pixel_count // (len(raw) + len(digest))) + 1)
    return Image.frombytes("RGB", (size, size), repeated[:pixel_count])


def _load_rgb_image(image_path: Path) -> Image.Image:
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return _fallback_rgb_image(image_path)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ClassifierBaselineError(
            f"Expected file does not exist: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ClassifierBaselineError(f"Invalid JSON file: {path}") from error


def _load_json_object(path: Path) -> JsonDict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ClassifierBaselineError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ClassifierBaselineError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_list(data: JsonDict, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ClassifierBaselineError(f"Expected '{key}' to be a list.")
    return cast(list[object], value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise ClassifierBaselineError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ClassifierBaselineError(f"Expected '{key}' to be an integer.")
    return value


def _require_float(data: JsonDict, key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ClassifierBaselineError(f"Expected '{key}' to be numeric.")
    return float(value)


def _require_bool(data: JsonDict, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ClassifierBaselineError(f"Expected '{key}' to be a boolean.")
    return value


def _parse_training_options(config_path: Path) -> TrainingOptions:
    payload = _load_json_object(config_path)
    training = payload.get("training")
    if training is None:
        return TrainingOptions(
            artifact_name="best.pt",
            device="cpu",
            mode="timm_classifier_baseline",
            weight_decay=0.0,
        )
    if not isinstance(training, dict):
        raise ClassifierBaselineError("Expected 'training' to be an object.")

    training_config = cast(JsonDict, training)
    artifact_name = _require_string(training_config, "artifact_name")
    device = _require_string(training_config, "device")
    mode = _require_string(training_config, "mode")
    weight_decay = _require_float(training_config, "weight_decay")
    return TrainingOptions(
        artifact_name=artifact_name,
        device=device,
        mode=mode,
        weight_decay=weight_decay,
    )


def _load_class_map_entries(class_map_path: Path) -> list[ClassMapEntry]:
    payload = _load_json_object(class_map_path)
    classes = _require_list(payload, "classes")
    entries: list[ClassMapEntry] = []
    for index, value in enumerate(classes):
        if not isinstance(value, dict):
            raise ClassifierBaselineError(
                f"Expected 'classes[{index}]' to be a JSON object."
            )
        record = cast(JsonDict, value)
        category_ids = tuple(sorted(_require_int_list(record, "category_ids")))
        reference_image_paths = tuple(
            sorted(_require_string_list(record, "reference_image_paths"))
        )
        entries.append(
            ClassMapEntry(
                category_ids=category_ids,
                class_id=_require_int(record, "class_id"),
                product_code=_require_string(record, "product_code"),
                product_name=_require_string(record, "product_name"),
                reference_image_paths=reference_image_paths,
            )
        )
    return sorted(entries, key=lambda item: item.class_id)


def _require_int_list(data: JsonDict, key: str) -> list[int]:
    values = _require_list(data, key)
    integers: list[int] = []
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ClassifierBaselineError(
                f"Expected '{key}[{index}]' to be an integer."
            )
        integers.append(value)
    return integers


def _require_string_list(data: JsonDict, key: str) -> list[str]:
    values = _require_list(data, key)
    strings: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or value == "":
            raise ClassifierBaselineError(
                f"Expected '{key}[{index}]' to be a non-empty string."
            )
        strings.append(value)
    return strings


def _load_crop_records(crop_manifest_path: Path) -> tuple[JsonDict, list[CropRecord]]:
    payload = _load_json_object(crop_manifest_path)
    crops = _require_list(payload, "crops")
    records: list[CropRecord] = []
    for index, value in enumerate(crops):
        if not isinstance(value, dict):
            raise ClassifierBaselineError(
                f"Expected 'crops[{index}]' to be a JSON object."
            )
        crop = cast(JsonDict, value)
        bbox = _require_list(crop, "bbox_xywh")
        if len(bbox) != 4:
            raise ClassifierBaselineError(
                f"Expected 'crops[{index}].bbox_xywh' to contain four values."
            )
        coordinates: list[float] = []
        for coordinate_index, number in enumerate(bbox):
            if not isinstance(number, (int, float)) or isinstance(number, bool):
                raise ClassifierBaselineError(
                    "Expected "
                    f"'crops[{index}].bbox_xywh[{coordinate_index}]' to be numeric."
                )
            coordinates.append(float(number))
        bbox_xywh = cast(tuple[float, float, float, float], tuple(coordinates))
        records.append(
            CropRecord(
                annotation_id=_require_int(crop, "annotation_id"),
                bbox_xywh=bbox_xywh,
                category_id=_require_int(crop, "category_id"),
                class_id=_require_int(crop, "class_id"),
                crop_path=_require_string(crop, "crop_path"),
                image_id=_require_int(crop, "image_id"),
                source_image_path=_require_string(crop, "source_image_path"),
            )
        )
    return payload, sorted(records, key=lambda item: item.annotation_id)


def _load_image_id_to_path(annotations_path: Path) -> dict[int, str]:
    payload = _load_json_object(annotations_path)
    images = _require_list(payload, "images")
    image_paths: dict[int, str] = {}
    for index, value in enumerate(images):
        if not isinstance(value, dict):
            raise ClassifierBaselineError(
                f"Expected 'images[{index}]' to be a JSON object."
            )
        record = cast(JsonDict, value)
        image_paths[_require_int(record, "id")] = _require_string(record, "file_name")
    return image_paths


def _crop_box_coordinates(
    bbox_xywh: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox_xywh
    left = max(0, min(image_width - 1, int(round(x))))
    top = max(0, min(image_height - 1, int(round(y))))
    right = max(left + 1, min(image_width, int(round(x + width))))
    bottom = max(top + 1, min(image_height, int(round(y + height))))
    if right <= left or bottom <= top:
        raise ClassifierBaselineError(
            "Invalid crop bounds after clamping: "
            f"{bbox_xywh} for {image_width}x{image_height}"
        )
    return left, top, right, bottom


def extract_crop_image(
    *,
    source_image_path: Path,
    bbox_xywh: tuple[float, float, float, float],
    output_path: Path,
) -> None:
    rgb_image = _load_rgb_image(source_image_path)
    left, top, right, bottom = _crop_box_coordinates(
        bbox_xywh,
        rgb_image.width,
        rgb_image.height,
    )
    crop = rgb_image.crop((left, top, right, bottom))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)


def _set_deterministic_mode() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _resolve_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_text == "cuda" and not torch.cuda.is_available():
        raise ClassifierBaselineError(
            "training.device='cuda' requires CUDA to be available."
        )
    return torch.device(device_text)


def _json_safe_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_model(backbone: str, num_classes: int) -> torch.nn.Module:
    return timm.create_model(backbone, pretrained=False, num_classes=num_classes)


def _freeze_backbone(model: torch.nn.Module) -> None:
    classifier_module = cast(torch.nn.Module, cast(Any, model.get_classifier)())
    classifier_parameters = set(classifier_module.parameters())
    for parameter in model.parameters():
        parameter.requires_grad = parameter in classifier_parameters


def _extract_pre_logits(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    forward_features = cast(Any, model.forward_features)
    features = forward_features(inputs)
    if not isinstance(features, torch.Tensor):
        raise ClassifierBaselineError(
            "Expected timm forward_features() to return a tensor."
        )
    forward_head = cast(Any, model.forward_head)
    pre_logits = forward_head(features, pre_logits=True)
    if not isinstance(pre_logits, torch.Tensor):
        raise ClassifierBaselineError(
            "Expected timm forward_head(..., pre_logits=True) to return a tensor."
        )
    return pre_logits


def _resolve_preprocessing(
    *, model: torch.nn.Module, requested_image_size: int
) -> tuple[JsonDict, object]:
    data_config = resolve_data_config(
        {"input_size": (3, requested_image_size, requested_image_size)}, model=model
    )
    transform = create_transform(**data_config, is_training=False)
    json_config: JsonDict = {
        key: cast(object, _json_safe_value(value))
        for key, value in sorted(data_config.items())
    }
    return json_config, transform


def _transform_image(transform: object, image_path: Path) -> torch.Tensor:
    if not callable(transform):
        raise ClassifierBaselineError(
            "Expected preprocessing transform to be callable."
        )
    transformed = transform(_load_rgb_image(image_path))
    if not isinstance(transformed, torch.Tensor):
        raise ClassifierBaselineError(
            "Expected preprocessing transform to return a tensor."
        )
    return transformed


def _extract_gt_crops(
    *, crop_records: list[CropRecord], processed_root: Path, output_dir: Path
) -> None:
    for crop in crop_records:
        extract_crop_image(
            source_image_path=processed_root / crop.source_image_path,
            bbox_xywh=crop.bbox_xywh,
            output_path=output_dir / crop.crop_path,
        )


def _build_training_samples(
    *,
    class_entries: list[ClassMapEntry],
    crop_records: list[CropRecord],
    processed_root: Path,
    output_dir: Path,
) -> list[LabeledImage]:
    samples: list[LabeledImage] = []
    for crop in crop_records:
        samples.append(
            LabeledImage(
                class_id=crop.class_id,
                image_path=output_dir / crop.crop_path,
                source="gt_crop",
            )
        )
    for entry in class_entries:
        for reference_image_path in entry.reference_image_paths:
            samples.append(
                LabeledImage(
                    class_id=entry.class_id,
                    image_path=processed_root / reference_image_path,
                    source="reference_image",
                )
            )
    return sorted(
        samples,
        key=lambda item: (item.class_id, item.source, item.image_path.as_posix()),
    )


def _stack_training_batch(
    *, samples: list[LabeledImage], transform: object
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    tensors: list[torch.Tensor] = []
    labels: list[int] = []
    counts = {"gt_crop": 0, "reference_image": 0}
    for sample in samples:
        tensors.append(_transform_image(transform, sample.image_path))
        labels.append(sample.class_id)
        counts[sample.source] = counts.get(sample.source, 0) + 1
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.int64), counts


def _canonical_category_id(entry: ClassMapEntry) -> int:
    if not entry.category_ids:
        raise ClassifierBaselineError(
            f"Classifier class_id {entry.class_id} does not have category_ids."
        )
    return entry.category_ids[0]


def _prediction_export(
    *, crop: CropRecord, predicted_category_id: int, confidence: float
) -> JsonDict:
    return {
        "bbox": list(crop.bbox_xywh),
        "category_id": predicted_category_id,
        "image_id": crop.image_id,
        "score": confidence,
    }


def _build_predictions_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.predictions.json")


def _resolve_annotations_path(crop_manifest: JsonDict, processed_root: Path) -> Path:
    processed_inputs = _require_mapping(crop_manifest, "processed_inputs")
    return processed_root / _require_string(processed_inputs, "annotations_path")


def _preprocessing_summary(preprocessing: JsonDict) -> JsonDict:
    return {
        "color_mode": "RGB",
        "crop_pct": preprocessing.get("crop_pct"),
        "feature_representation": "timm_resolved_transform",
        "input_size": preprocessing.get("input_size"),
        "interpolation": preprocessing.get("interpolation"),
        "mean": preprocessing.get("mean"),
        "std": preprocessing.get("std"),
    }


def _evaluate_crop_records(
    *,
    class_entries: list[ClassMapEntry],
    crop_records: list[CropRecord],
    device: torch.device,
    model: torch.nn.Module,
    output_dir: Path,
    transform: object,
) -> tuple[list[JsonDict], list[JsonDict], float]:
    class_by_id = {entry.class_id: entry for entry in class_entries}
    detailed_predictions: list[JsonDict] = []
    scorer_predictions: list[JsonDict] = []
    correct_count = 0

    model.eval()
    with torch.no_grad():
        for crop in crop_records:
            input_tensor = _transform_image(transform, output_dir / crop.crop_path)
            logits = model(input_tensor.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits[0], dim=0)
            predicted_class_id = int(torch.argmax(probabilities).item())
            confidence = float(torch.max(probabilities).item())
            predicted_entry = class_by_id[predicted_class_id]
            predicted_category_id = _canonical_category_id(predicted_entry)
            correct_count += int(predicted_class_id == crop.class_id)
            scorer_predictions.append(
                _prediction_export(
                    crop=crop,
                    predicted_category_id=predicted_category_id,
                    confidence=confidence,
                )
            )
            detailed_predictions.append(
                {
                    "annotation_id": crop.annotation_id,
                    "confidence": confidence,
                    "expected_category_id": crop.category_id,
                    "expected_class_id": crop.class_id,
                    "image_id": crop.image_id,
                    "predicted_category_id": predicted_category_id,
                    "predicted_class_id": predicted_class_id,
                    "product_code": predicted_entry.product_code,
                }
            )

    accuracy_top1 = correct_count / len(crop_records) if crop_records else 0.0
    return detailed_predictions, scorer_predictions, accuracy_top1


def _state_dict_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _train_timm_classifier(
    *,
    class_entries: list[ClassMapEntry],
    crop_manifest: JsonDict,
    crop_records: list[CropRecord],
    device: torch.device,
    model: torch.nn.Module,
    options: TrainingOptions,
    output_dir: Path,
    preprocessing: JsonDict,
    processed_root: Path,
    transform: object,
) -> tuple[dict[str, torch.Tensor], float, JsonDict]:
    train_samples = _build_training_samples(
        class_entries=class_entries,
        crop_records=crop_records,
        processed_root=processed_root,
        output_dir=output_dir,
    )
    train_images, train_labels, sample_counts = _stack_training_batch(
        samples=train_samples,
        transform=transform,
    )
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)

    _freeze_backbone(model)
    model.to(device)
    classifier_object = cast(object, cast(Any, model.get_classifier)())
    if not isinstance(classifier_object, torch.nn.Linear):
        raise ClassifierBaselineError(
            "Task 9 expects a linear timm classifier head for closed-form fitting."
        )
    classifier = cast(torch.nn.Linear, classifier_object)
    model.eval()
    with torch.no_grad():
        embeddings = _extract_pre_logits(model, train_images)

    num_classes = len(class_entries)
    design_matrix = torch.cat(
        [embeddings, torch.ones((embeddings.shape[0], 1), device=device)], dim=1
    )
    target = torch.nn.functional.one_hot(train_labels, num_classes=num_classes).to(
        dtype=torch.float32
    )
    ridge = options.weight_decay if options.weight_decay > 0 else 1e-6
    identity = torch.eye(design_matrix.shape[1], device=device) * ridge
    weights = torch.linalg.solve(
        design_matrix.T @ design_matrix + identity,
        design_matrix.T @ target,
    )

    with torch.no_grad():
        classifier.weight.copy_(weights[:-1, :].T.contiguous())
        classifier.bias.copy_(weights[-1, :].contiguous())

    logits = model(train_images)
    loss = torch.nn.functional.cross_entropy(logits, train_labels)
    last_loss = float(loss.item())
    _, scorer_predictions, best_accuracy = _evaluate_crop_records(
        class_entries=class_entries,
        crop_records=crop_records,
        device=device,
        model=model,
        output_dir=output_dir,
        transform=transform,
    )
    _ = score_predictions(
        _resolve_annotations_path(crop_manifest, processed_root), scorer_predictions
    )
    best_state_dict = _state_dict_copy(model)

    training_summary: JsonDict = {
        "best_accuracy_top1": best_accuracy,
        "class_count": len(class_entries),
        "crop_count": len(crop_records),
        "deterministic": True,
        "fit_method": "ridge_closed_form_classifier_head",
        "last_loss": last_loss,
        "mode": options.mode,
        "sample_counts": {
            "gt_crop": sample_counts.get("gt_crop", 0),
            "reference_image": sample_counts.get("reference_image", 0),
            "total": len(train_samples),
        },
        "weights_artifact_name": options.artifact_name,
        "weight_decay": options.weight_decay,
    }
    return best_state_dict, best_accuracy, training_summary


def _checkpoint_metadata(
    *,
    backbone: str,
    class_map_path: Path,
    crop_manifest_path: Path,
    preprocessing: JsonDict,
    processed_root: Path,
) -> JsonDict:
    return {
        "backbone": backbone,
        "class_map_path": class_map_path.as_posix(),
        "crop_manifest_path": crop_manifest_path.as_posix(),
        "preprocessing": preprocessing,
        "processed_root": processed_root.as_posix(),
    }


def _load_checkpoint(weights_path: Path) -> tuple[JsonDict, JsonDict]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ClassifierBaselineError(f"Expected checkpoint object in {weights_path}")
    state_dict = checkpoint.get("state_dict")
    metadata = checkpoint.get("metadata")
    if not isinstance(state_dict, dict):
        raise ClassifierBaselineError("Classifier checkpoint missing state_dict.")
    if not isinstance(metadata, dict):
        raise ClassifierBaselineError("Classifier checkpoint missing metadata.")
    return cast(JsonDict, checkpoint), cast(JsonDict, metadata)


def _create_model_from_metadata(
    *, metadata: JsonDict, num_classes: int
) -> tuple[torch.nn.Module, JsonDict]:
    backbone = _require_string(metadata, "backbone")
    model = _build_model(backbone, num_classes)
    preprocessing = _require_mapping(metadata, "preprocessing")
    return model, preprocessing


def _preprocessing_transform(preprocessing: JsonDict) -> object:
    kwargs: dict[str, object] = {
        key: value for key, value in preprocessing.items() if key != "is_training"
    }
    if "input_size" in kwargs:
        kwargs["input_size"] = tuple(cast(list[int], kwargs["input_size"]))
    if "mean" in kwargs:
        kwargs["mean"] = tuple(cast(list[float], kwargs["mean"]))
    if "std" in kwargs:
        kwargs["std"] = tuple(cast(list[float], kwargs["std"]))
    return create_transform(**kwargs, is_training=False)


def _load_model_bundle(weights_path: Path) -> ModelBundle:
    checkpoint, metadata = _load_checkpoint(weights_path)
    class_map_path = Path(_require_string(metadata, "class_map_path"))
    crop_manifest_path = Path(_require_string(metadata, "crop_manifest_path"))
    processed_root = Path(_require_string(metadata, "processed_root"))
    class_entries = _load_class_map_entries(class_map_path)
    crop_manifest, crop_records = _load_crop_records(crop_manifest_path)
    model, preprocessing = _create_model_from_metadata(
        metadata=metadata,
        num_classes=len(class_entries),
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ClassifierBaselineError("Classifier checkpoint missing state_dict.")
    model.load_state_dict(cast(dict[str, torch.Tensor], state_dict))
    model.eval()
    return ModelBundle(
        class_entries=class_entries,
        crop_manifest=crop_manifest,
        crop_records=crop_records,
        metadata=metadata,
        model=model,
        preprocessing=preprocessing,
        processed_root=processed_root,
    )


def run_training(
    *, config_path: str | Path, processed_root: str | Path, output_dir: str | Path
) -> JsonDict:
    config_path = Path(config_path)
    processed_root = Path(processed_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_deterministic_mode()
    options = _parse_training_options(config_path)
    config = load_and_validate_classifier_data_config(config_path, processed_root)
    device = _resolve_device(options.device)

    build_classifier_crop_dataset(
        config_path=config_path,
        processed_root=processed_root,
        output_root=output_dir,
    )
    class_map_path = output_dir / config.outputs.class_map_path
    crop_manifest_path = output_dir / config.outputs.crop_manifest_path
    class_entries = _load_class_map_entries(class_map_path)
    crop_manifest, crop_records = _load_crop_records(crop_manifest_path)
    _extract_gt_crops(
        crop_records=crop_records,
        processed_root=processed_root,
        output_dir=output_dir,
    )

    model = _build_model(config.model.backbone, len(class_entries))
    preprocessing, transform = _resolve_preprocessing(
        model=model,
        requested_image_size=config.model.image_size,
    )
    best_state_dict, _, training_summary = _train_timm_classifier(
        class_entries=class_entries,
        crop_manifest=crop_manifest,
        crop_records=crop_records,
        device=device,
        model=model,
        options=options,
        output_dir=output_dir,
        preprocessing=preprocessing,
        processed_root=processed_root,
        transform=transform,
    )

    model.load_state_dict(best_state_dict)
    weights_path = output_dir / options.artifact_name
    checkpoint = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": _checkpoint_metadata(
            backbone=config.model.backbone,
            class_map_path=class_map_path,
            crop_manifest_path=crop_manifest_path,
            preprocessing=preprocessing,
            processed_root=processed_root,
        ),
        "state_dict": best_state_dict,
    }
    torch.save(checkpoint, weights_path)

    summary_path = output_dir / "train_summary.json"
    summary: JsonDict = {
        "artifact_provenance": {
            "class_map": {
                "path": class_map_path.as_posix(),
                **file_snapshot(class_map_path),
            },
            "crop_manifest": {
                "path": crop_manifest_path.as_posix(),
                **file_snapshot(crop_manifest_path),
            },
            "weights": {
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "path": weights_path.as_posix(),
                **file_snapshot(weights_path),
            },
        },
        "config_path": config_path.as_posix(),
        "output_artifacts": {
            "best_weights": weights_path.as_posix(),
            "class_map": class_map_path.as_posix(),
            "crop_manifest": crop_manifest_path.as_posix(),
            "summary_json": summary_path.as_posix(),
        },
        "output_dir": output_dir.as_posix(),
        "preprocessing": _preprocessing_summary(preprocessing),
        "processed_root": processed_root.as_posix(),
        "runtime": {
            "backbone": config.model.backbone,
            "backbone_library": config.runtime.backbone_library,
            "backbone_library_version": config.runtime.backbone_library_version,
            "device": str(device),
            "execution_backend": "timm_torch",
        },
        "schema_version": 1,
        "training": training_summary,
    }
    write_json(summary_path, summary)
    return summary


def _evaluate_gt_boxes(weights_path: Path, out_path: Path) -> JsonDict:
    bundle = _load_model_bundle(weights_path)
    transform = _preprocessing_transform(bundle.preprocessing)
    device = torch.device("cpu")
    bundle.model.to(device)

    detailed_predictions, scorer_predictions, accuracy_top1 = _evaluate_crop_records(
        class_entries=bundle.class_entries,
        crop_records=bundle.crop_records,
        device=device,
        model=bundle.model,
        output_dir=weights_path.parent,
        transform=transform,
    )

    annotations_path = _resolve_annotations_path(
        bundle.crop_manifest, bundle.processed_root
    )
    metrics = score_predictions(annotations_path, scorer_predictions)
    predictions_path = _build_predictions_path(out_path)
    write_json(predictions_path, scorer_predictions)

    payload: JsonDict = {
        "artifact_provenance": {
            "train_summary": {
                "path": weights_path.with_name("train_summary.json").as_posix(),
                **file_snapshot(weights_path.with_name("train_summary.json")),
            },
            "weights": {
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "path": weights_path.as_posix(),
                **file_snapshot(weights_path),
            },
        },
        "evaluation": {
            "accuracy_top1": accuracy_top1,
            "crop_count": len(bundle.crop_records),
            "mode": GT_BOXES_MODE,
            "processed_root": bundle.processed_root.as_posix(),
        },
        "export": {
            "format": "coco_predictions",
            "predictions_path": predictions_path.as_posix(),
            **file_snapshot(predictions_path),
        },
        "label_space": {
            "class_count": len(bundle.class_entries),
            "class_map_path": _require_string(bundle.metadata, "class_map_path"),
            "crop_manifest_path": _require_string(
                bundle.metadata, "crop_manifest_path"
            ),
        },
        "metrics": metrics,
        "predictions": detailed_predictions,
        "preprocessing": _preprocessing_summary(bundle.preprocessing),
        "schema_version": 1,
    }
    write_json(out_path, payload)
    return payload


def _load_detector_predictions(predictions_path: Path) -> list[JsonDict]:
    payload = _load_json(predictions_path)
    if isinstance(payload, dict):
        metrics_payload = cast(JsonDict, payload)
        export = metrics_payload.get("export")
        evaluation = metrics_payload.get("evaluation")
        if isinstance(export, dict) and export.get("placeholder") is True:
            raise ClassifierBaselineError(
                "Detector-box classifier evaluation rejects placeholder detector "
                "exports."
            )
        if isinstance(evaluation, dict) and evaluation.get("mode") == "smoke":
            raise ClassifierBaselineError(
                "Detector-box classifier evaluation rejects smoke detector exports."
            )
        if isinstance(export, dict) and isinstance(export.get("predictions_path"), str):
            payload = _load_json(Path(cast(str, export["predictions_path"])))
        else:
            raise ClassifierBaselineError(
                "Detector metrics JSON must include export.predictions_path."
            )

    try:
        parsed_predictions = validate_predictions(payload)
    except ScoreValidationError as error:
        raise ClassifierBaselineError(str(error)) from error

    category_ids = {prediction.category_id for prediction in parsed_predictions}
    if category_ids == {DETECTION_ONLY_CATEGORY_ID}:
        raise ClassifierBaselineError(
            "Detector-box classifier evaluation rejects detection-only "
            "category_id=0 predictions."
        )

    return [
        {
            "bbox": list(prediction.bbox),
            "category_id": prediction.category_id,
            "image_id": prediction.image_id,
            "score": prediction.score,
        }
        for prediction in parsed_predictions
    ]


def _evaluate_detector_boxes(
    *, weights_path: Path, out_path: Path, detector_predictions_path: Path
) -> JsonDict:
    bundle = _load_model_bundle(weights_path)
    transform = _preprocessing_transform(bundle.preprocessing)
    device = torch.device("cpu")
    bundle.model.to(device)

    predictions = _load_detector_predictions(detector_predictions_path)
    annotations_path = _resolve_annotations_path(
        bundle.crop_manifest, bundle.processed_root
    )
    image_id_to_path = _load_image_id_to_path(annotations_path)
    class_by_id = {entry.class_id: entry for entry in bundle.class_entries}
    temp_dir = out_path.parent / ".classifier_detector_box_crops"
    temp_dir.mkdir(parents=True, exist_ok=True)

    scorer_predictions: list[JsonDict] = []
    detailed_predictions: list[JsonDict] = []
    bundle.model.eval()
    with torch.no_grad():
        for index, prediction in enumerate(predictions):
            image_id = _require_int(prediction, "image_id")
            image_relative_path = image_id_to_path.get(image_id)
            if image_relative_path is None:
                raise ClassifierBaselineError(
                    f"Detector prediction references unknown image_id: {image_id}"
                )
            bbox = prediction.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ClassifierBaselineError(
                    "Detector prediction bbox must contain four values."
                )
            crop_box = cast(
                tuple[float, float, float, float], tuple(float(value) for value in bbox)
            )
            temp_crop_path = temp_dir / f"{index:06d}_{image_id}.jpg"
            extract_crop_image(
                source_image_path=bundle.processed_root
                / "images"
                / image_relative_path,
                bbox_xywh=crop_box,
                output_path=temp_crop_path,
            )
            input_tensor = _transform_image(transform, temp_crop_path)
            logits = bundle.model(input_tensor.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits[0], dim=0)
            predicted_class_id = int(torch.argmax(probabilities).item())
            confidence = float(torch.max(probabilities).item())
            predicted_entry = class_by_id[predicted_class_id]
            detector_score = _require_float(prediction, "score")
            combined_score = max(0.0, min(1.0, detector_score * confidence))
            predicted_category_id = _canonical_category_id(predicted_entry)
            scorer_prediction = {
                "bbox": list(crop_box),
                "category_id": predicted_category_id,
                "image_id": image_id,
                "score": combined_score,
            }
            scorer_predictions.append(scorer_prediction)
            detailed_predictions.append(
                {
                    "confidence": confidence,
                    "detector_score": detector_score,
                    "image_id": image_id,
                    "predicted_category_id": predicted_category_id,
                    "predicted_class_id": predicted_class_id,
                    "product_code": predicted_entry.product_code,
                    "score": combined_score,
                }
            )

    predictions_path = _build_predictions_path(out_path)
    write_json(predictions_path, scorer_predictions)
    metrics = score_predictions(annotations_path, scorer_predictions)
    payload: JsonDict = {
        "artifact_provenance": {
            "detector_predictions": {
                "path": detector_predictions_path.as_posix(),
                **file_snapshot(detector_predictions_path),
            },
            "train_summary": {
                "path": weights_path.with_name("train_summary.json").as_posix(),
                **file_snapshot(weights_path.with_name("train_summary.json")),
            },
            "weights": {
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "path": weights_path.as_posix(),
                **file_snapshot(weights_path),
            },
        },
        "evaluation": {
            "mode": DETECTOR_BOXES_MODE,
            "prediction_count": len(scorer_predictions),
            "processed_root": bundle.processed_root.as_posix(),
        },
        "export": {
            "format": "coco_predictions",
            "predictions_path": predictions_path.as_posix(),
            **file_snapshot(predictions_path),
        },
        "label_space": {
            "class_count": len(bundle.class_entries),
            "class_map_path": _require_string(bundle.metadata, "class_map_path"),
        },
        "metrics": metrics,
        "predictions": detailed_predictions,
        "preprocessing": _preprocessing_summary(bundle.preprocessing),
        "schema_version": 1,
    }
    write_json(out_path, payload)
    return payload


def run_evaluation(
    *,
    weights_path: str | Path,
    mode: str,
    out_path: str | Path,
    detector_predictions_path: str | Path | None = None,
) -> JsonDict:
    weights = Path(weights_path)
    output = Path(out_path)
    if mode == GT_BOXES_MODE:
        return _evaluate_gt_boxes(weights, output)
    if mode == DETECTOR_BOXES_MODE:
        if detector_predictions_path is None:
            raise ClassifierBaselineError(
                "Detector-box evaluation requires --detector-predictions."
            )
        return _evaluate_detector_boxes(
            weights_path=weights,
            out_path=output,
            detector_predictions_path=Path(detector_predictions_path),
        )
    raise ClassifierBaselineError(
        "Unsupported classifier evaluation mode: "
        f"{mode}. Expected '{GT_BOXES_MODE}' or '{DETECTOR_BOXES_MODE}'."
    )
