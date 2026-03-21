from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch  # type: ignore[import-not-found]
from ultralytics import YOLO  # type: ignore[import-not-found]

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
IMAGE_ID_PATTERN = re.compile(r"(\d+)$")
PLACEHOLDER_WEIGHTS_HEADER = "placeholder-detector-weights"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bundled detector inference for shelf images."
    )
    parser.add_argument("--input", required=True, help="Directory containing images.")
    parser.add_argument(
        "--output", required=True, help="Path where predictions JSON will be written."
    )
    return parser


def list_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def infer_image_id(image_path: Path, fallback_id: int) -> int:
    match = IMAGE_ID_PATTERN.search(image_path.stem)
    if match is None:
        return fallback_id
    return int(match.group(1))


def _json_scalar(value: object) -> object:
    try:
        return value.tolist()  # type: ignore[union-attr,no-any-return]
    except AttributeError:
        pass
    try:
        return value.item()  # type: ignore[union-attr,no-any-return]
    except AttributeError:
        pass
    if isinstance(value, tuple):
        return list(value)
    return value


def _sequence(value: object, *, label: str) -> list[object]:
    normalized = _json_scalar(value)
    if not isinstance(normalized, list):
        raise ValueError(f"Expected {label} to be a list-like value.")
    return normalized


def _numeric_value(value: object, *, label: str) -> float:
    normalized = _json_scalar(value)
    if not isinstance(normalized, (int, float)) or isinstance(normalized, bool):
        raise ValueError(f"Expected {label} to be numeric.")
    return float(normalized)


def _class_index(value: object, *, label: str) -> int:
    numeric_value = _numeric_value(value, label=label)
    integer_value = int(numeric_value)
    if float(integer_value) != numeric_value:
        raise ValueError(f"Expected {label} to be an integer class id.")
    return integer_value


def _xyxy_to_xywh(xyxy: list[object], *, index: int) -> list[float]:
    if len(xyxy) != 4:
        raise ValueError(f"Expected inference box {index} to contain four xyxy values.")
    x1 = _numeric_value(xyxy[0], label=f"boxes.xyxy[{index}][0]")
    y1 = _numeric_value(xyxy[1], label=f"boxes.xyxy[{index}][1]")
    x2 = _numeric_value(xyxy[2], label=f"boxes.xyxy[{index}][2]")
    y2 = _numeric_value(xyxy[3], label=f"boxes.xyxy[{index}][3]")
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Inference box {index} is not valid.")
    return [x1, y1, width, height]


def _best_weights_path() -> Path:
    return Path(__file__).with_name("best.pt")


def _load_placeholder_payload(weights_path: Path) -> dict[str, object] | None:
    try:
        content = weights_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    if not content.startswith(f"{PLACEHOLDER_WEIGHTS_HEADER}\n"):
        return None

    payload_lines = content.splitlines()[3:]
    payload = json.loads("\n".join(payload_lines))
    if not isinstance(payload, dict):
        raise ValueError("Placeholder detector weights payload must be an object.")
    return payload


def _placeholder_model_yaml(model_name: str) -> str:
    if not model_name.startswith("yolov8"):
        raise ValueError(f"Unsupported placeholder detector model_name: {model_name}")
    return f"{model_name}.yaml"


def _safe_global_types(seed_model: object) -> list[type[object]]:
    safe_types: set[type[object]] = {
        type(seed_model),
        torch.nn.Sequential,
        torch.nn.ModuleList,
    }
    for module in seed_model.modules():  # type: ignore[attr-defined]
        safe_types.add(type(module))
    return list(safe_types)


def _normalize_placeholder_weights(weights_path: Path) -> None:
    payload = _load_placeholder_payload(weights_path)
    if payload is None:
        return

    model_name = payload.get("model_name")
    if not isinstance(model_name, str) or model_name == "":
        raise ValueError("Placeholder detector weights must declare model_name.")

    seed_model = YOLO(_placeholder_model_yaml(model_name)).model
    torch.serialization.add_safe_globals(_safe_global_types(seed_model))
    checkpoint = {
        "model": seed_model,
        "train_args": {},
    }
    torch.save(checkpoint, weights_path)


def load_model() -> YOLO:
    weights_path = _best_weights_path()
    _normalize_placeholder_weights(weights_path)
    return YOLO(weights_path)


def build_predictions_for_image(
    result: object, *, image_id: int
) -> list[dict[str, int | float | list[float]]]:
    try:
        boxes = result.boxes  # type: ignore[attr-defined]
    except AttributeError:
        return []

    xyxy_rows = _sequence(boxes.xyxy, label="boxes.xyxy")  # type: ignore[attr-defined]
    conf_values = _sequence(boxes.conf, label="boxes.conf")  # type: ignore[attr-defined]
    cls_values = _sequence(boxes.cls, label="boxes.cls")  # type: ignore[attr-defined]
    if not (len(xyxy_rows) == len(conf_values) == len(cls_values)):
        raise ValueError("Ultralytics inference boxes/conf/cls arrays must align.")

    predictions: list[dict[str, int | float | list[float]]] = []
    for index, xyxy_row in enumerate(xyxy_rows):
        predictions.append(
            {
                "image_id": image_id,
                "category_id": _class_index(
                    cls_values[index], label=f"boxes.cls[{index}]"
                ),
                "bbox": _xyxy_to_xywh(
                    _sequence(xyxy_row, label=f"boxes.xyxy[{index}]"),
                    index=index,
                ),
                "score": _numeric_value(
                    conf_values[index], label=f"boxes.conf[{index}]"
                ),
            }
        )
    return predictions


def generate_predictions(input_dir: Path) -> list[dict[str, int | float | list[float]]]:
    image_files = list_image_files(input_dir)
    if not image_files:
        return []
    if not _best_weights_path().is_file():
        return []

    model = load_model()
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    predictions: list[dict[str, int | float | list[float]]] = []
    for fallback_id, image_path in enumerate(image_files, start=1):
        result = model.predict(
            image_path,
            imgsz=960,
            conf=0.05,
            iou=0.6,
            device=device,
            verbose=False,
        )[0]
        predictions.extend(
            build_predictions_for_image(
                result,
                image_id=infer_image_id(image_path, fallback_id),
            )
        )
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    output_path = Path(args.output)
    predictions = generate_predictions(input_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
