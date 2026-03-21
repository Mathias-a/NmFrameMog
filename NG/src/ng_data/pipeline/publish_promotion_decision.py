from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from src.ng_data.data.manifest import write_json
from src.ng_data.pipeline.compare_variants import (
    DETECTOR_CLASSIFIER,
    DETECTOR_CLASSIFIER_RETRIEVAL,
    DETECTOR_ONLY,
)

JsonDict = dict[str, Any]
SCHEMA_VERSION = 1
VARIANT_PRIORITY = (
    DETECTOR_CLASSIFIER_RETRIEVAL,
    DETECTOR_CLASSIFIER,
    DETECTOR_ONLY,
)


class PublishPromotionDecisionError(ValueError):
    pass


class PublishArgs(argparse.Namespace):
    out: str
    report: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a conservative final variant decision from a comparison report."
        )
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to the retrieval gate comparison report JSON.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the final variant decision JSON output.",
    )
    return parser


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PublishPromotionDecisionError(
            f"Expected file does not exist: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise PublishPromotionDecisionError(f"Invalid JSON file: {path}") from error


def _load_json_object(path: Path) -> JsonDict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise PublishPromotionDecisionError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _require_mapping(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise PublishPromotionDecisionError(f"Expected '{key}' to be an object.")
    return cast(JsonDict, value)


def _require_bool(data: JsonDict, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise PublishPromotionDecisionError(f"Expected '{key}' to be a boolean.")
    return value


def _require_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise PublishPromotionDecisionError(
            f"Expected '{key}' to be a non-empty string."
        )
    return value


def _variant(payload: JsonDict, name: str) -> JsonDict:
    variants = _require_mapping(payload, "variants")
    value = variants.get(name)
    if not isinstance(value, dict):
        raise PublishPromotionDecisionError(
            f"Expected report.variants['{name}'] to be an object."
        )
    return cast(JsonDict, value)


def build_decision(report_payload: JsonDict, *, report_path: Path) -> JsonDict:
    eligible_variants: list[str] = []
    for name in VARIANT_PRIORITY:
        if _require_bool(_variant(report_payload, name), "eligible"):
            eligible_variants.append(name)

    recommended_baseline = _require_string(
        report_payload, "recommended_baseline_variant"
    )
    selected_variant = (
        eligible_variants[0] if eligible_variants else recommended_baseline
    )
    promotion_applied = (
        bool(eligible_variants) and selected_variant != recommended_baseline
    )
    selected_entry = _variant(report_payload, selected_variant)

    rationale: list[str] = []
    if promotion_applied:
        rationale.append(
            f"Selected {selected_variant} because it is the highest-priority "
            "eligible variant."
        )
    elif eligible_variants:
        rationale.append(
            f"Kept baseline {recommended_baseline} because it remains the highest-"
            "priority eligible variant."
        )
    else:
        rationale.append(
            "No variant has promotion-grade end-to-end evidence, so the "
            "baseline stays detector-only."
        )
    if not eligible_variants:
        rationale.extend(cast(list[str], selected_entry.get("blocking_reasons", [])))

    return {
        "decision": {
            "baseline_variant": recommended_baseline,
            "promotion_applied": promotion_applied,
            "rationale": rationale,
            "selected_variant": selected_variant,
            "selected_variant_eligible": _require_bool(selected_entry, "eligible"),
            "strategy": (
                "promote_best_eligible_variant"
                if promotion_applied
                else "keep_detector_only"
            ),
        },
        "eligible_variants": eligible_variants,
        "report_path": report_path.as_posix(),
        "schema_version": SCHEMA_VERSION,
        "variant_summaries": {
            name: {
                "blocking_reasons": cast(
                    list[str],
                    _variant(report_payload, name).get("blocking_reasons", []),
                ),
                "eligible": _require_bool(_variant(report_payload, name), "eligible"),
                "status": _require_string(_variant(report_payload, name), "status"),
            }
            for name in (
                DETECTOR_ONLY,
                DETECTOR_CLASSIFIER,
                DETECTOR_CLASSIFIER_RETRIEVAL,
            )
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(PublishArgs, parser.parse_args(argv))
    report_path = Path(args.report)
    try:
        decision = build_decision(
            _load_json_object(report_path), report_path=report_path
        )
    except (PublishPromotionDecisionError, ValueError) as error:
        raise SystemExit(str(error)) from error

    out_path = Path(args.out)
    write_json(out_path, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
