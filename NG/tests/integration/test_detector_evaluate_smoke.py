from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

from src.ng_data.eval.score import score_predictions


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_evaluate_module() -> Any:
    module_path = _repo_root() / "src/ng_data/detector/evaluate.py"
    spec = importlib.util.spec_from_file_location("ng_detector_evaluate", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load detector evaluate module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_real_train_summary(weights_path: Path) -> None:
    summary_path = weights_path.with_name("train_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "artifact_provenance": {
                    "normalized_best_weights": {
                        "path": str(weights_path),
                        "sha256": "fake-weights-sha",
                        "size_bytes": weights_path.stat().st_size,
                    },
                    "ultralytics_best_weights": {
                        "path": (
                            "artifacts/runs/detector/"
                            "yolov8m-search-baseline/weights/best.pt"
                        ),
                        "sha256": "fake-ultralytics-sha",
                        "size_bytes": weights_path.stat().st_size,
                    },
                },
                "config_path": "configs/detector/yolov8m-search.json",
                "dataset_preparation": {
                    "dataset_yaml": (
                        "artifacts/runs/detector/"
                        "yolov8m-search-baseline/dataset/dataset.yaml"
                    ),
                    "names": ["TEST PRODUCT ONE", "unknown_product"],
                    "nc": 2,
                    "output_dir": (
                        "artifacts/runs/detector/yolov8m-search-baseline/dataset"
                    ),
                    "source_annotations": "annotations/instances.coco.json",
                    "source_manifest": "manifests/dataset_manifest.json",
                    "split_counts": {"test": 1, "train": 1, "val": 0},
                    "splits_path": "data/processed/manifests/splits.json",
                },
                "mode": "real",
                "model_name": "yolov8m",
                "output_artifacts": {
                    "best_weights": str(weights_path),
                    "summary_json": str(summary_path),
                },
                "output_dir": str(weights_path.parent),
                "processed_manifest_path": (
                    "data/processed/manifests/dataset_manifest.json"
                ),
                "runtime": {
                    "export_format": "pt",
                    "framework": "ultralytics",
                    "version": "8.1.0",
                },
                "runtime_version": "8.1.0",
                "search": {
                    "batch_size": 16,
                    "device": "cuda",
                    "epochs": 30,
                    "image_size": 960,
                    "patience": 10,
                    "run_name": "yolov8m-search-baseline",
                },
                "split_counts": {
                    "cv_folds": 4,
                    "cv_pool_images": 1,
                    "holdout_images": 1,
                    "total_images": 2,
                },
                "splits_path": "data/processed/manifests/splits.json",
                "source_annotations": "annotations/instances.coco.json",
                "source_manifest": "manifests/dataset_manifest.json",
                "summary_format": "json",
                "training": {
                    "counts": {
                        "annotation_count": 2,
                        "category_count": 2,
                        "image_count": 2,
                        "reference_product_count": 2,
                    },
                    "metrics": {"fitness": 0.9, "metrics/mAP50(B)": 1.0},
                    "paths": {
                        "best_weights": str(weights_path),
                        "prepared_dataset_yaml": (
                            "artifacts/runs/detector/"
                            "yolov8m-search-baseline/dataset/dataset.yaml"
                        ),
                        "ultralytics_best_weights": (
                            "artifacts/runs/detector/"
                            "yolov8m-search-baseline/weights/best.pt"
                        ),
                        "ultralytics_project_dir": "artifacts/runs/detector",
                        "ultralytics_run_dir": (
                            "artifacts/runs/detector/yolov8m-search-baseline"
                        ),
                    },
                    "placeholder": False,
                    "runtime_summary": {
                        "completed_at_utc": "2026-03-21T12:00:01+00:00",
                        "duration_seconds": 1.0,
                        "started_at_utc": "2026-03-21T12:00:00+00:00",
                    },
                    "weights": "yolov8m.pt",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class _FakeBoxes:
    def __init__(self, *, xyxy: list[list[float]], conf: list[float], cls: list[float]):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes | None):
        self.boxes = boxes


def _build_fake_inference_results() -> list[_FakeResult]:
    return [
        _FakeResult(
            _FakeBoxes(
                xyxy=[[10.0, 20.0, 40.0, 60.0]],
                conf=[0.95],
                cls=[0.0],
            )
        ),
        _FakeResult(
            _FakeBoxes(
                xyxy=[[5.0, 6.0, 12.0, 14.0]],
                conf=[0.85],
                cls=[1.0],
            )
        ),
    ]


def test_detector_evaluate_smoke_writes_deterministic_metrics_and_predictions(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    evaluate_main = _load_evaluate_module().main
    weights_path = (
        repo_root / "artifacts/models/detector/yolov8m-search-baseline/best.pt"
    )
    split_path = repo_root / "data/processed/annotations/instances.coco.json"
    out_path = tmp_path / "artifacts/eval/detector_holdout_metrics.json"

    first_exit = evaluate_main(
        [
            "--weights",
            str(weights_path),
            "--split",
            str(split_path),
            "--out",
            str(out_path),
        ]
    )

    predictions_path = out_path.with_name("detector_holdout_metrics.predictions.json")
    first_metrics_text = out_path.read_text(encoding="utf-8")
    first_predictions_text = predictions_path.read_text(encoding="utf-8")

    second_exit = evaluate_main(
        [
            "--weights",
            str(weights_path),
            "--split",
            str(split_path),
            "--out",
            str(out_path),
        ]
    )

    assert first_exit == 0
    assert second_exit == 0
    assert out_path.read_text(encoding="utf-8") == first_metrics_text
    assert predictions_path.read_text(encoding="utf-8") == first_predictions_text

    metrics_payload = cast(dict[str, Any], json.loads(first_metrics_text))
    predictions_payload = cast(list[dict[str, Any]], json.loads(first_predictions_text))

    assert metrics_payload["schema_version"] == 1
    assert metrics_payload["evaluation"] == {
        "mode": "smoke",
        "prediction_count": 2,
        "scoring_rule": "0.7*detection_map + 0.3*classification_map",
        "split_path": str(split_path),
    }
    assert metrics_payload["metrics"] == {
        "classification_map": 0.5,
        "detection_map": 1.0,
        "hybrid_score": 0.85,
    }
    assert metrics_payload["smoke_contract"] == {
        "annotation_count": 2,
        "annotations_path": "data/processed/annotations/instances.coco.json",
        "holdout_images": 1,
        "image_count": 2,
        "model_name": "yolov8m",
        "run_name": "yolov8m-search-baseline",
        "runtime": {
            "export_format": "pt",
            "framework": "ultralytics",
            "version": "8.1.0",
        },
        "seed": 20260320,
        "train_summary_path": str(
            repo_root
            / "artifacts/models/detector/yolov8m-search-baseline/train_summary.json"
        ),
        "training_output_dir": "artifacts/models/detector/yolov8m-search-baseline",
    }
    assert metrics_payload["split_summary"] == {
        "annotation_count": 2,
        "category_count": 2,
        "image_count": 2,
    }
    assert metrics_payload["export"]["format"] == "coco_predictions"
    assert metrics_payload["export"]["placeholder"] is True
    assert metrics_payload["export"]["predictions_path"] == str(predictions_path)
    assert metrics_payload["artifact_provenance"]["weights"]["contract_version"] == (
        "detector-train-smoke-v1"
    )

    assert predictions_payload == [
        {
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "category_id": 0,
            "image_id": 1,
            "score": 1.0,
        },
        {
            "bbox": [5.0, 6.0, 7.0, 8.0],
            "category_id": 0,
            "image_id": 2,
            "score": 1.0,
        },
    ]


def test_detector_evaluate_real_mode_requires_existing_weights(tmp_path: Path) -> None:
    evaluate_main = _load_evaluate_module().main
    out_path = tmp_path / "artifacts/eval/detector_holdout_metrics.json"
    missing_weights = (
        tmp_path / "artifacts/models/detector/yolov8m-search-baseline/best.pt"
    )
    split_path = _repo_root() / "data/processed/annotations/instances.coco.json"

    try:
        evaluate_main(
            [
                "--mode",
                "real",
                "--weights",
                str(missing_weights),
                "--split",
                str(split_path),
                "--out",
                str(out_path),
            ]
        )
    except SystemExit as error:
        assert (
            str(error)
            == f"Real detector weights file does not exist: {missing_weights}"
        )
    else:
        raise AssertionError(
            "Expected real detector evaluation to fail on missing weights."
        )


def test_detector_evaluate_real_mode_writes_contract_valid_report(
    tmp_path: Path, monkeypatch: Any
) -> None:
    repo_root = _repo_root()
    evaluate_module = _load_evaluate_module()
    weights_path = (
        tmp_path / "artifacts/models/detector/yolov8m-search-baseline/best.pt"
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"real-detector-weights")
    _write_real_train_summary(weights_path)

    split_path = repo_root / "data/processed/annotations/instances.coco.json"
    out_path = tmp_path / "artifacts/eval/detector_holdout_metrics.json"
    captured_call: dict[str, object] = {}

    def fake_run_ultralytics_inference(
        weights_path: Path, images: list[object]
    ) -> list[_FakeResult]:
        captured_call["weights_path"] = weights_path
        captured_call["image_count"] = len(images)
        return _build_fake_inference_results()

    monkeypatch.setattr(
        evaluate_module,
        "_run_ultralytics_inference",
        fake_run_ultralytics_inference,
    )

    exit_code = evaluate_module.main(
        [
            "--mode",
            "real",
            "--weights",
            str(weights_path),
            "--split",
            str(split_path),
            "--out",
            str(out_path),
        ]
    )

    predictions_path = out_path.with_name("detector_holdout_metrics.predictions.json")
    metrics_payload = cast(
        dict[str, Any], json.loads(out_path.read_text(encoding="utf-8"))
    )
    predictions_payload = cast(
        list[dict[str, Any]], json.loads(predictions_path.read_text(encoding="utf-8"))
    )

    assert exit_code == 0
    assert captured_call == {
        "image_count": 2,
        "weights_path": weights_path,
    }
    assert metrics_payload["schema_version"] == 1
    assert metrics_payload["evaluation"] == {
        "mode": "holdout_real",
        "prediction_count": 2,
        "scoring_rule": "0.7*detection_map + 0.3*classification_map",
        "split_path": str(split_path),
    }
    assert metrics_payload["export"]["format"] == "coco_predictions"
    assert metrics_payload["export"]["placeholder"] is False
    assert metrics_payload["export"]["predictions_path"] == str(predictions_path)
    assert metrics_payload["trained_model"] == {
        "mode": "real",
        "model_name": "yolov8m",
        "output_dir": str(weights_path.parent),
        "run_name": "yolov8m-search-baseline",
        "runtime": {
            "export_format": "pt",
            "framework": "ultralytics",
            "version": "8.1.0",
        },
        "training_placeholder": False,
    }
    assert metrics_payload["split_summary"] == {
        "annotation_count": 2,
        "category_count": 2,
        "image_count": 2,
    }
    assert "smoke_contract" not in metrics_payload
    assert metrics_payload["metrics"] == {
        "classification_map": 1.0,
        "detection_map": 1.0,
        "hybrid_score": 1.0,
    }
    assert predictions_payload == [
        {
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "category_id": 0,
            "image_id": 1,
            "score": 0.95,
        },
        {
            "bbox": [5.0, 6.0, 7.0, 8.0],
            "category_id": 356,
            "image_id": 2,
            "score": 0.85,
        },
    ]
    assert (
        score_predictions(split_path, predictions_payload) == metrics_payload["metrics"]
    )
