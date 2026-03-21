from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import src.ng_data.detector.train as detector_train
from src.ng_data.detector.train import main as train_main


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_detector_train_smoke_writes_deterministic_placeholder_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/detector/yolov8m-search.json"
    manifest_path = repo_root / "data/processed/manifests/dataset_manifest.json"
    splits_path = repo_root / "data/processed/manifests/splits.json"
    output_dir = tmp_path / "artifacts/models/detector"

    first_exit = train_main(
        [
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--splits",
            str(splits_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    run_dir = output_dir / "yolov8m-search-baseline"
    best_weights_path = run_dir / "best.pt"
    summary_path = run_dir / "train_summary.json"
    first_best_weights = best_weights_path.read_bytes()
    first_summary_text = summary_path.read_text(encoding="utf-8")

    second_exit = train_main(
        [
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--splits",
            str(splits_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert first_exit == 0
    assert second_exit == 0
    assert first_best_weights == best_weights_path.read_bytes()
    assert first_summary_text == summary_path.read_text(encoding="utf-8")

    summary = cast(dict[str, Any], json.loads(first_summary_text))
    assert summary["mode"] == "smoke"
    assert summary["model_name"] == "yolov8m"
    assert summary["runtime_version"] == "detector-train-smoke-v1"
    assert summary["config_path"] == str(config_path)
    assert summary["processed_manifest_path"] == str(manifest_path)
    assert summary["splits_path"] == str(splits_path)
    assert summary["source_manifest"] == "manifests/dataset_manifest.json"
    assert summary["source_annotations"] == "annotations/instances.coco.json"
    assert summary["training"]["placeholder"] is True
    assert summary["training"]["counts"] == {
        "annotation_count": 2,
        "category_count": 2,
        "image_count": 2,
        "reference_product_count": 2,
    }
    assert summary["split_counts"] == {
        "cv_folds": 4,
        "cv_pool_images": 1,
        "holdout_images": 1,
        "total_images": 2,
    }
    assert summary["output_artifacts"] == {
        "best_weights": str(best_weights_path),
        "summary_json": str(summary_path),
    }
    assert best_weights_path.read_text(encoding="utf-8").startswith(
        "placeholder-detector-weights\n"
    )


@dataclass
class _FakeTrainResult:
    results_dict: dict[str, float]


def test_detector_train_real_mode_normalizes_ultralytics_outputs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    repo_root = _repo_root()
    config_path = repo_root / "configs/detector/yolov8m-search.json"
    manifest_path = repo_root / "data/processed/manifests/dataset_manifest.json"
    splits_path = repo_root / "data/processed/manifests/splits.json"
    output_dir = tmp_path / "artifacts/models/detector"
    dataset_dir = tmp_path / "prepared-dataset"
    prepared_dataset_yaml = dataset_dir / "dataset.yaml"
    prepared_dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    prepared_dataset_yaml.write_text("path: fake\n", encoding="utf-8")
    ultralytics_best = (
        repo_root / "artifacts/runs/detector/yolov8m-search-baseline/weights/best.pt"
    )
    ultralytics_best.parent.mkdir(parents=True, exist_ok=True)

    def fake_prepare_detector_dataset(**_: object) -> dict[str, object]:
        return {
            "config_path": str(config_path),
            "dataset_yaml": str(prepared_dataset_yaml),
            "names": ["apple", "banana"],
            "nc": 2,
            "output_dir": str(dataset_dir),
            "processed_manifest_path": str(manifest_path),
            "source_annotations": "annotations/instances.coco.json",
            "source_manifest": "manifests/dataset_manifest.json",
            "split_counts": {"train": 1, "val": 0, "test": 1},
            "splits_path": str(splits_path),
        }

    captured_call: dict[str, object] = {}

    def fake_run_ultralytics_training(
        *, config: object, prepared_dataset_yaml: str
    ) -> _FakeTrainResult:
        captured_call["config"] = config
        captured_call["prepared_dataset_yaml"] = prepared_dataset_yaml
        ultralytics_best.write_bytes(b"real-detector-weights")
        return _FakeTrainResult(
            results_dict={"metrics/mAP50(B)": 0.42, "fitness": 0.43}
        )

    monkeypatch.setattr(
        detector_train,
        "prepare_detector_dataset",
        fake_prepare_detector_dataset,
    )
    monkeypatch.setattr(
        detector_train,
        "_run_ultralytics_training",
        fake_run_ultralytics_training,
    )

    exit_code = train_main(
        [
            "--mode",
            "real",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--splits",
            str(splits_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    run_dir = output_dir / "yolov8m-search-baseline"
    best_weights_path = run_dir / "best.pt"
    summary_path = run_dir / "train_summary.json"
    summary = cast(dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8")))

    assert exit_code == 0
    assert captured_call["prepared_dataset_yaml"] == str(prepared_dataset_yaml)
    assert best_weights_path.read_bytes() == b"real-detector-weights"
    assert summary["mode"] == "real"
    assert summary["training"]["placeholder"] is False
    assert summary["output_artifacts"]["best_weights"] == str(best_weights_path)
    assert summary["training"]["paths"]["best_weights"] == str(best_weights_path)
    assert summary["training"]["paths"]["prepared_dataset_yaml"] == str(
        prepared_dataset_yaml
    )
    assert summary["training"]["paths"]["ultralytics_best_weights"] == (
        "artifacts/runs/detector/yolov8m-search-baseline/weights/best.pt"
    )
    assert summary["dataset_preparation"] == {
        "dataset_yaml": str(prepared_dataset_yaml),
        "names": ["apple", "banana"],
        "nc": 2,
        "output_dir": str(dataset_dir),
        "source_annotations": "annotations/instances.coco.json",
        "source_manifest": "manifests/dataset_manifest.json",
        "split_counts": {"train": 1, "val": 0, "test": 1},
        "splits_path": str(splits_path),
    }
    assert summary["training"]["metrics"] == {
        "fitness": 0.43,
        "metrics/mAP50(B)": 0.42,
    }
    assert summary["training"]["runtime_summary"]["duration_seconds"] >= 0.0
    assert summary["artifact_provenance"]["normalized_best_weights"]["path"] == str(
        best_weights_path
    )
    assert summary["artifact_provenance"]["ultralytics_best_weights"]["path"] == (
        "artifacts/runs/detector/yolov8m-search-baseline/weights/best.pt"
    )


def test_detector_train_rejects_noncanonical_run_name(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    base_config_path = repo_root / "configs/detector/yolov8m-search.json"
    manifest_path = repo_root / "data/processed/manifests/dataset_manifest.json"
    splits_path = repo_root / "data/processed/manifests/splits.json"
    output_dir = tmp_path / "artifacts/models/detector"
    config_payload = cast(
        dict[str, Any], json.loads(base_config_path.read_text(encoding="utf-8"))
    )
    config_payload["search"]["run_name"] = "wrong-run-name"
    tampered_config_path = tmp_path / "tampered-detector-config.json"
    tampered_config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        train_main(
            [
                "--mode",
                "smoke",
                "--config",
                str(tampered_config_path),
                "--manifest",
                str(manifest_path),
                "--splits",
                str(splits_path),
                "--output-dir",
                str(output_dir),
            ]
        )
    except SystemExit as error:
        assert (
            "Detector training requires search.run_name='yolov8m-search-baseline'"
            in str(error)
        )
    else:
        raise AssertionError("Expected detector training to reject wrong run_name.")

    assert not (output_dir / "wrong-run-name").exists()
