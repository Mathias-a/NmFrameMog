from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from src.ng_data.classifier.evaluate import main as evaluate_main
from src.ng_data.classifier.train import main as train_main


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_classifier_train_and_gt_evaluate_write_deterministic_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/classifier/search.json"
    processed_root = repo_root / "data/processed"
    output_dir = tmp_path / "artifacts/models/classifier"
    metrics_path = tmp_path / "artifacts/eval/classifier_gt_metrics.json"

    first_train_exit = train_main(
        [
            "--config",
            str(config_path),
            "--processed-root",
            str(processed_root),
            "--output-dir",
            str(output_dir),
        ]
    )
    first_eval_exit = evaluate_main(
        [
            "--weights",
            str(output_dir / "best.pt"),
            "--mode",
            "gt_boxes",
            "--out",
            str(metrics_path),
        ]
    )

    first_summary_text = (output_dir / "train_summary.json").read_text(encoding="utf-8")
    first_metrics_text = metrics_path.read_text(encoding="utf-8")
    first_predictions_text = metrics_path.with_name(
        "classifier_gt_metrics.predictions.json"
    ).read_text(encoding="utf-8")

    second_train_exit = train_main(
        [
            "--config",
            str(config_path),
            "--processed-root",
            str(processed_root),
            "--output-dir",
            str(output_dir),
        ]
    )
    second_eval_exit = evaluate_main(
        [
            "--weights",
            str(output_dir / "best.pt"),
            "--mode",
            "gt_boxes",
            "--out",
            str(metrics_path),
        ]
    )

    assert first_train_exit == 0
    assert second_train_exit == 0
    assert first_eval_exit == 0
    assert second_eval_exit == 0
    assert (output_dir / "train_summary.json").read_text(
        encoding="utf-8"
    ) == first_summary_text
    assert metrics_path.read_text(encoding="utf-8") == first_metrics_text
    assert (
        metrics_path.with_name("classifier_gt_metrics.predictions.json").read_text(
            encoding="utf-8"
        )
        == first_predictions_text
    )

    summary = cast(dict[str, Any], json.loads(first_summary_text))
    metrics = cast(dict[str, Any], json.loads(first_metrics_text))
    predictions = cast(list[dict[str, Any]], json.loads(first_predictions_text))

    assert summary["runtime"] == {
        "backbone": "efficientnet_b0",
        "backbone_library": "timm",
        "backbone_library_version": "0.9.12",
        "device": "cpu",
        "execution_backend": "timm_torch",
    }
    assert summary["preprocessing"]["color_mode"] == "RGB"
    assert (
        summary["preprocessing"]["feature_representation"] == "timm_resolved_transform"
    )
    assert summary["preprocessing"]["input_size"] == [3, 224, 224]
    assert summary["preprocessing"]["mean"] == [0.485, 0.456, 0.406]
    assert summary["preprocessing"]["std"] == [0.229, 0.224, 0.225]
    assert summary["training"]["class_count"] == 2
    assert summary["training"]["crop_count"] == 2
    assert summary["training"]["deterministic"] is True
    assert summary["training"]["mode"] == "timm_classifier_baseline"
    assert summary["training"]["sample_counts"] == {
        "gt_crop": 2,
        "reference_image": 3,
        "total": 5,
    }
    assert summary["training"]["weights_artifact_name"] == "best.pt"
    assert 0.0 <= summary["training"]["best_accuracy_top1"] <= 1.0
    assert math.isfinite(summary["training"]["last_loss"])
    assert summary["training"]["fit_method"] == "ridge_closed_form_classifier_head"
    assert summary["runtime"]["execution_backend"] != "prototype_stub"
    assert summary["preprocessing"]["feature_representation"] != "flattened_resized_rgb"
    assert summary["training"]["mode"] != "prototype_stub"
    assert summary["output_artifacts"]["best_weights"] == str(output_dir / "best.pt")
    assert summary["output_artifacts"]["class_map"] == str(
        output_dir / "manifests/classifier_class_map.json"
    )
    assert summary["output_artifacts"]["crop_manifest"] == str(
        output_dir / "manifests/classifier_crop_manifest.json"
    )
    assert (output_dir / "crops/gt/000000/000001_1111111111111.jpg").is_file()
    assert (output_dir / "crops/gt/000001/000002_2222222222222.jpg").is_file()

    assert metrics["schema_version"] == 1
    assert metrics["evaluation"]["crop_count"] == 2
    assert metrics["evaluation"]["mode"] == "gt_boxes"
    assert metrics["evaluation"]["processed_root"] == str(processed_root)
    assert 0.0 <= metrics["evaluation"]["accuracy_top1"] <= 1.0
    assert metrics["label_space"] == {
        "class_count": 2,
        "class_map_path": str(output_dir / "manifests/classifier_class_map.json"),
        "crop_manifest_path": str(
            output_dir / "manifests/classifier_crop_manifest.json"
        ),
    }
    assert set(metrics["metrics"]) == {
        "classification_map",
        "detection_map",
        "hybrid_score",
    }
    assert 0.0 <= metrics["metrics"]["classification_map"] <= 1.0
    assert 0.0 <= metrics["metrics"]["detection_map"] <= 1.0
    assert 0.0 <= metrics["metrics"]["hybrid_score"] <= 1.0
    assert (
        metrics["preprocessing"]["feature_representation"] == "timm_resolved_transform"
    )
    assert metrics["preprocessing"]["input_size"] == [3, 224, 224]
    assert len(predictions) == 2
    assert predictions[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert predictions[0]["image_id"] == 1
    assert predictions[1]["bbox"] == [5.0, 6.0, 7.0, 8.0]
    assert predictions[1]["image_id"] == 2
    assert isinstance(predictions[0]["category_id"], int)
    assert isinstance(predictions[1]["category_id"], int)
    assert math.isfinite(predictions[0]["score"])
    assert math.isfinite(predictions[1]["score"])
    assert 0.0 <= predictions[0]["score"] <= 1.0
    assert 0.0 <= predictions[1]["score"] <= 1.0


def test_classifier_detector_box_evaluate_rejects_smoke_detector_predictions(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    output_dir = tmp_path / "artifacts/models/classifier"
    train_exit = train_main(
        [
            "--config",
            str(repo_root / "configs/classifier/search.json"),
            "--processed-root",
            str(repo_root / "data/processed"),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert train_exit == 0

    detector_metrics_path = repo_root / "artifacts/eval/detector_holdout_metrics.json"
    try:
        evaluate_main(
            [
                "--weights",
                str(output_dir / "best.pt"),
                "--mode",
                "detector_boxes",
                "--detector-predictions",
                str(detector_metrics_path),
                "--out",
                str(tmp_path / "artifacts/eval/classifier_detector_metrics.json"),
            ]
        )
    except SystemExit as error:
        message = str(error)
    else:
        raise AssertionError("Expected detector-box classifier evaluation to fail.")

    assert (
        message
        == "Detector-box classifier evaluation rejects placeholder detector exports."
    )
