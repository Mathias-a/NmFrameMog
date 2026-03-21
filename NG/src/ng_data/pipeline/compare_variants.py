from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from src.ng_data.data.manifest import write_json
from src.ng_data.eval.score import ScoreValidationError, load_predictions

JsonDict = dict[str, Any]

DEFAULT_DETECTOR_REPORT = Path("artifacts/eval/detector_holdout_metrics.json")
DEFAULT_CLASSIFIER_REPORT = Path("artifacts/eval/classifier_gt_metrics.json")
DEFAULT_RETRIEVAL_MANIFEST = Path("artifacts/retrieval/gallery_manifest.json")
DEFAULT_RETRIEVAL_EVAL_REPORT = Path("artifacts/eval/retrieval_end_to_end_metrics.json")
SCHEMA_VERSION = 1

DETECTOR_ONLY = "detector_only"
DETECTOR_CLASSIFIER = "detector_classifier"
DETECTOR_CLASSIFIER_RETRIEVAL = "detector_classifier_retrieval"


class CompareVariantsError(ValueError):
    pass


class CompareArgs(argparse.Namespace):
    classifier_report: str
    detector_report: str
    out: str
    retrieval_eval_report: str
    retrieval_manifest: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare detector-first pipeline variants conservatively."
    )
    parser.add_argument(
        "--detector-report",
        default=str(DEFAULT_DETECTOR_REPORT),
        help="Path to detector evaluation metrics JSON.",
    )
    parser.add_argument(
        "--classifier-report",
        default=str(DEFAULT_CLASSIFIER_REPORT),
        help="Path to classifier evaluation metrics JSON.",
    )
    parser.add_argument(
        "--retrieval-manifest",
        default=str(DEFAULT_RETRIEVAL_MANIFEST),
        help="Path to retrieval gallery manifest JSON.",
    )
    parser.add_argument(
        "--retrieval-eval-report",
        default=str(DEFAULT_RETRIEVAL_EVAL_REPORT),
        help="Path to retrieval end-to-end evaluation metrics JSON.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the variant comparison report JSON output.",
    )
    return parser


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CompareVariantsError(f"Expected file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise CompareVariantsError(f"Invalid JSON file: {path}") from error


def _load_optional_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise CompareVariantsError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _load_json_object(path: Path) -> JsonDict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise CompareVariantsError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise CompareVariantsError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise CompareVariantsError(f"Expected '{key}' to be a non-empty string.")
    return value


def _json_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _metrics_summary(payload: JsonDict) -> JsonDict | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    summary: JsonDict = {}
    for key in ("classification_map", "detection_map", "hybrid_score"):
        metric_value = _json_float(metrics.get(key))
        if metric_value is not None:
            summary[key] = metric_value
    return summary or None


def _reference_entry(path: Path, payload: JsonDict | None) -> JsonDict:
    entry: JsonDict = {
        "exists": payload is not None,
        "path": path.as_posix(),
    }
    if payload is not None:
        metrics = _metrics_summary(payload)
        if metrics is not None:
            entry["metrics"] = metrics
    return entry


def _detector_blocking_reasons(detector_payload: JsonDict) -> list[str]:
    reasons: list[str] = []
    evaluation = _require_mapping(detector_payload, "evaluation")
    export = _require_mapping(detector_payload, "export")
    if evaluation.get("mode") == "smoke":
        reasons.append("detector evaluation.mode=smoke is not promotion-grade evidence")
    if export.get("placeholder") is True:
        reasons.append(
            "detector export.placeholder=true is not promotion-grade evidence"
        )
    predictions_path_value = export.get("predictions_path")
    if isinstance(predictions_path_value, str) and predictions_path_value != "":
        predictions_path = Path(predictions_path_value)
        if not predictions_path.is_absolute():
            predictions_path = Path.cwd() / predictions_path
        try:
            category_ids = {
                prediction.category_id
                for prediction in load_predictions(predictions_path)
            }
        except (
            CompareVariantsError,
            ScoreValidationError,
            FileNotFoundError,
            OSError,
            ValueError,
        ):
            reasons.append(
                "detector predictions could not be validated for promotion gating"
            )
        else:
            if category_ids == {0}:
                reasons.append(
                    "detector predictions are detection-only category_id=0 outputs"
                )
    return reasons


def _classifier_blocking_reasons(classifier_payload: JsonDict) -> list[str]:
    reasons: list[str] = []
    evaluation = _require_mapping(classifier_payload, "evaluation")
    if evaluation.get("mode") == "gt_boxes":
        reasons.append(
            "classifier evaluation.mode=gt_boxes is upper-bound reference evidence only"
        )
    if evaluation.get("mode") != "detector_boxes":
        reasons.append(
            "classifier does not have accepted end-to-end detector-box "
            "evaluation evidence"
        )
    return reasons


def _retrieval_blocking_reasons(retrieval_eval_payload: JsonDict | None) -> list[str]:
    if retrieval_eval_payload is None:
        return [
            "retrieval has no end-to-end evaluation artifact; gallery manifest "
            "alone is insufficient"
        ]
    evaluation = retrieval_eval_payload.get("evaluation")
    if not isinstance(evaluation, dict):
        return ["retrieval evaluation artifact is missing an evaluation object"]
    mode = evaluation.get("mode")
    if mode != "detector_classifier_retrieval":
        return [
            "retrieval evaluation artifact does not declare "
            "mode=detector_classifier_retrieval"
        ]
    return []


def _variant_entry(
    *,
    name: str,
    summary: str,
    metrics_payload: JsonDict | None,
    evidence_paths: list[str],
    blocking_reasons: list[str],
) -> JsonDict:
    eligible = not blocking_reasons
    status = "eligible" if eligible else "ineligible"
    entry: JsonDict = {
        "blocking_reasons": blocking_reasons,
        "eligible": eligible,
        "evidence_paths": evidence_paths,
        "name": name,
        "status": status,
        "summary": summary,
    }
    metrics = None if metrics_payload is None else _metrics_summary(metrics_payload)
    if metrics is not None:
        entry["metrics"] = metrics
    return entry


def build_report(
    *,
    detector_report_path: Path,
    classifier_report_path: Path,
    retrieval_manifest_path: Path,
    retrieval_eval_report_path: Path,
) -> JsonDict:
    detector_payload = _load_json_object(detector_report_path)
    classifier_payload = _load_json_object(classifier_report_path)
    retrieval_manifest_payload = _load_json_object(retrieval_manifest_path)
    retrieval_eval_payload = _load_optional_json(retrieval_eval_report_path)
    detector_blocking_reasons = _detector_blocking_reasons(detector_payload)
    classifier_blocking_reasons = _classifier_blocking_reasons(classifier_payload)
    retrieval_blocking_reasons = _retrieval_blocking_reasons(retrieval_eval_payload)

    detector_variant = _variant_entry(
        name=DETECTOR_ONLY,
        summary="Baseline detector evidence is present but currently smoke-only.",
        metrics_payload=detector_payload,
        evidence_paths=[detector_report_path.as_posix()],
        blocking_reasons=detector_blocking_reasons,
    )
    classifier_variant = _variant_entry(
        name=DETECTOR_CLASSIFIER,
        summary=(
            "Classifier evidence exists, but current metrics are GT-box "
            "reference only and "
            "do not prove end-to-end promotion readiness."
        ),
        metrics_payload=classifier_payload,
        evidence_paths=[
            detector_report_path.as_posix(),
            classifier_report_path.as_posix(),
        ],
        blocking_reasons=[*detector_blocking_reasons, *classifier_blocking_reasons],
    )
    retrieval_variant = _variant_entry(
        name=DETECTOR_CLASSIFIER_RETRIEVAL,
        summary=(
            "Retrieval has gallery foundation artifacts, but no accepted "
            "end-to-end evidence "
            "for promotion."
        ),
        metrics_payload=retrieval_eval_payload,
        evidence_paths=[
            detector_report_path.as_posix(),
            classifier_report_path.as_posix(),
            retrieval_manifest_path.as_posix(),
            retrieval_eval_report_path.as_posix(),
        ],
        blocking_reasons=[
            *detector_blocking_reasons,
            *classifier_blocking_reasons,
            *retrieval_blocking_reasons,
        ],
    )

    return {
        "artifact_references": {
            "classifier_report": _reference_entry(
                classifier_report_path, classifier_payload
            ),
            "detector_report": _reference_entry(detector_report_path, detector_payload),
            "retrieval_eval_report": _reference_entry(
                retrieval_eval_report_path, retrieval_eval_payload
            ),
            "retrieval_manifest": _reference_entry(
                retrieval_manifest_path, retrieval_manifest_payload
            ),
        },
        "comparison_policy": {
            "classifier_gt_boxes_are_reference_only": True,
            "detector_placeholder_exports_are_not_promotion_grade": True,
            "detector_smoke_evaluations_are_not_promotion_grade": True,
            "retrieval_gallery_manifest_requires_end_to_end_eval": True,
        },
        "recommended_baseline_variant": DETECTOR_ONLY,
        "schema_version": SCHEMA_VERSION,
        "variants": {
            DETECTOR_ONLY: detector_variant,
            DETECTOR_CLASSIFIER: classifier_variant,
            DETECTOR_CLASSIFIER_RETRIEVAL: retrieval_variant,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(CompareArgs, parser.parse_args(argv))
    try:
        report = build_report(
            detector_report_path=Path(args.detector_report),
            classifier_report_path=Path(args.classifier_report),
            retrieval_manifest_path=Path(args.retrieval_manifest),
            retrieval_eval_report_path=Path(args.retrieval_eval_report),
        )
    except (CompareVariantsError, ScoreValidationError, ValueError) as error:
        raise SystemExit(str(error)) from error

    out_path = Path(args.out)
    write_json(out_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
