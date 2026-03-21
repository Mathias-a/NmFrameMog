from __future__ import annotations

import json
from pathlib import Path

from src.ng_data.data.manifest import write_json
from src.ng_data.pipeline.compare_variants import main as compare_variants_main
from src.ng_data.pipeline.publish_promotion_decision import (
    main as publish_promotion_decision_main,
)


def _write_detector_artifacts(tmp_path: Path) -> Path:
    predictions_path = (
        tmp_path / "artifacts/eval/detector_holdout_metrics.predictions.json"
    )
    write_json(
        predictions_path,
        [
            {
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "category_id": 0,
                "image_id": 1,
                "score": 1.0,
            }
        ],
    )
    detector_report_path = tmp_path / "artifacts/eval/detector_holdout_metrics.json"
    write_json(
        detector_report_path,
        {
            "evaluation": {
                "mode": "smoke",
                "prediction_count": 1,
                "scoring_rule": "0.7*detection_map + 0.3*classification_map",
                "split_path": "data/processed/annotations/instances.coco.json",
            },
            "export": {
                "format": "coco_predictions",
                "placeholder": True,
                "predictions_path": str(predictions_path),
            },
            "metrics": {
                "classification_map": 0.5,
                "detection_map": 1.0,
                "hybrid_score": 0.85,
            },
            "schema_version": 1,
        },
    )
    return detector_report_path


def _write_classifier_report(tmp_path: Path) -> Path:
    classifier_report_path = tmp_path / "artifacts/eval/classifier_gt_metrics.json"
    write_json(
        classifier_report_path,
        {
            "evaluation": {
                "accuracy_top1": 0.5,
                "crop_count": 1,
                "mode": "gt_boxes",
                "processed_root": "data/processed",
            },
            "metrics": {
                "classification_map": 0.5,
                "detection_map": 1.0,
                "hybrid_score": 0.85,
            },
            "schema_version": 1,
        },
    )
    return classifier_report_path


def _write_retrieval_manifest(tmp_path: Path) -> Path:
    retrieval_manifest_path = tmp_path / "artifacts/retrieval/gallery_manifest.json"
    write_json(
        retrieval_manifest_path,
        {
            "config_path": "configs/retrieval/gallery.json",
            "counts": {
                "product_count": 1,
                "products_with_missing_views": 1,
                "products_without_prototypes": 0,
                "prototype_count": 1,
            },
            "index": {
                "embedding_dim": 32,
                "format": "npz",
                "normalize": True,
                "path": "artifacts/retrieval/gallery_index.npz",
                "prototype_strategy": "mean_reference_hash",
            },
            "products": [
                {
                    "available_views": ["main.jpg"],
                    "category_ids": [0],
                    "missing_views": ["front.jpg"],
                    "product_code": "1111111111111",
                    "product_name": "TEST PRODUCT ONE",
                    "prototype_count": 1,
                    "prototype_index_range": [0, 1],
                    "reference_image_paths": [
                        "reference_images/1111111111111/main.jpg"
                    ],
                }
            ],
            "schema_version": 1,
        },
    )
    return retrieval_manifest_path


def test_compare_variants_rejects_placeholder_detector_and_gt_classifier(
    tmp_path: Path,
) -> None:
    detector_report_path = _write_detector_artifacts(tmp_path)
    classifier_report_path = _write_classifier_report(tmp_path)
    retrieval_manifest_path = _write_retrieval_manifest(tmp_path)
    out_path = tmp_path / "artifacts/eval/retrieval_gate_report.json"

    exit_code = compare_variants_main(
        [
            "--detector-report",
            str(detector_report_path),
            "--classifier-report",
            str(classifier_report_path),
            "--retrieval-manifest",
            str(retrieval_manifest_path),
            "--retrieval-eval-report",
            str(tmp_path / "artifacts/eval/retrieval_end_to_end_metrics.json"),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    detector_variant = payload["variants"]["detector_only"]
    classifier_variant = payload["variants"]["detector_classifier"]
    retrieval_variant = payload["variants"]["detector_classifier_retrieval"]

    assert payload["schema_version"] == 1
    assert payload["recommended_baseline_variant"] == "detector_only"
    assert detector_variant["eligible"] is False
    assert detector_variant["status"] == "ineligible"
    assert (
        "detector evaluation.mode=smoke is not promotion-grade evidence"
        in detector_variant["blocking_reasons"]
    )
    assert (
        "detector export.placeholder=true is not promotion-grade evidence"
        in detector_variant["blocking_reasons"]
    )
    assert (
        "detector predictions are detection-only category_id=0 outputs"
        in detector_variant["blocking_reasons"]
    )
    assert classifier_variant["eligible"] is False
    assert (
        "classifier evaluation.mode=gt_boxes is upper-bound reference evidence only"
        in classifier_variant["blocking_reasons"]
    )
    assert (
        "classifier does not have accepted end-to-end detector-box evaluation evidence"
        in classifier_variant["blocking_reasons"]
    )
    assert retrieval_variant["eligible"] is False
    assert (
        "retrieval has no end-to-end evaluation artifact; gallery manifest "
        "alone is insufficient" in retrieval_variant["blocking_reasons"]
    )


def test_publish_promotion_decision_keeps_detector_only_without_promotion(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "artifacts/eval/retrieval_gate_report.json"
    write_json(
        report_path,
        {
            "recommended_baseline_variant": "detector_only",
            "schema_version": 1,
            "variants": {
                "detector_only": {
                    "blocking_reasons": [
                        "detector evaluation.mode=smoke is not promotion-grade evidence"
                    ],
                    "eligible": False,
                    "status": "ineligible",
                },
                "detector_classifier": {
                    "blocking_reasons": [
                        "classifier evaluation.mode=gt_boxes is upper-bound "
                        "reference evidence only"
                    ],
                    "eligible": False,
                    "status": "ineligible",
                },
                "detector_classifier_retrieval": {
                    "blocking_reasons": [
                        "retrieval has no end-to-end evaluation artifact; "
                        "gallery manifest alone is insufficient"
                    ],
                    "eligible": False,
                    "status": "ineligible",
                },
            },
        },
    )
    out_path = tmp_path / "artifacts/eval/final_variant_decision.json"

    exit_code = publish_promotion_decision_main(
        [
            "--report",
            str(report_path),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["eligible_variants"] == []
    assert payload["decision"] == {
        "baseline_variant": "detector_only",
        "promotion_applied": False,
        "rationale": [
            "No variant has promotion-grade end-to-end evidence, so the "
            "baseline stays detector-only.",
            "detector evaluation.mode=smoke is not promotion-grade evidence",
        ],
        "selected_variant": "detector_only",
        "selected_variant_eligible": False,
        "strategy": "keep_detector_only",
    }
    assert (
        payload["variant_summaries"]["detector_classifier_retrieval"]["eligible"]
        is False
    )


def test_publish_promotion_decision_keeps_eligible_baseline_without_false_rationale(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "artifacts/eval/retrieval_gate_report.json"
    write_json(
        report_path,
        {
            "recommended_baseline_variant": "detector_only",
            "schema_version": 1,
            "variants": {
                "detector_only": {
                    "blocking_reasons": [],
                    "eligible": True,
                    "status": "eligible",
                },
                "detector_classifier": {
                    "blocking_reasons": ["classifier end-to-end evidence missing"],
                    "eligible": False,
                    "status": "ineligible",
                },
                "detector_classifier_retrieval": {
                    "blocking_reasons": ["retrieval end-to-end evidence missing"],
                    "eligible": False,
                    "status": "ineligible",
                },
            },
        },
    )
    out_path = tmp_path / "artifacts/eval/final_variant_decision.json"

    exit_code = publish_promotion_decision_main(
        [
            "--report",
            str(report_path),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["eligible_variants"] == ["detector_only"]
    assert payload["decision"]["promotion_applied"] is False
    assert payload["decision"]["selected_variant"] == "detector_only"
    assert payload["decision"]["rationale"] == [
        "Kept baseline detector_only because it remains the highest-priority "
        "eligible variant."
    ]
