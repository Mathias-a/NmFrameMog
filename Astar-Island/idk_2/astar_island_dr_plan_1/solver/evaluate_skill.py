from __future__ import annotations

# pyright: reportMissingImports=false
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .benchmark_suite import BenchmarkSuiteReport, evaluate_benchmark_suite
from .blessed_refs import (
    BlessedReferences,
    ReferenceBundleLocator,
    load_blessed_references,
)
from .cache import LocalCache
from .evaluation_contract import CandidatePredictionBundle
from .replay_harness import FrozenPredictionCandidateAdapter, load_offline_replay

EvaluateMode = Literal["benchmark", "promote"]

_CANDIDATE_REGISTRY_FILENAME = "candidates.json"
_DEFAULT_CANDIDATE_REGISTRY_SCHEMA_VERSION = "astar-evaluation-candidates-v1"
_DEFAULT_PER_SEED_REGRESSION_GATE = 0.0


@dataclass(frozen=True)
class EvaluationOutputs:
    report_path: Path
    summary_path: Path
    report_payload: dict[str, object]
    summary_text: str


def evaluate_solution(
    *,
    cache_dir: Path,
    dataset_version: str,
    candidate_id: str,
    mode: EvaluateMode,
) -> EvaluationOutputs:
    if mode not in {"benchmark", "promote"}:
        raise ValueError(
            "Unsupported evaluate-solution mode "
            f"{mode!r}; expected 'benchmark' or 'promote'."
        )

    cache = LocalCache(cache_dir)
    dataset_dir = cache.root / "datasets" / dataset_version
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Frozen dataset version {dataset_version!r} was not found at "
            f"{dataset_dir}."
        )

    candidate_locator = _resolve_candidate_locator(
        dataset_dir=dataset_dir,
        dataset_version=dataset_version,
        candidate_id=candidate_id,
    )
    replay = load_offline_replay(
        dataset_dir=dataset_dir,
        candidate_bundle_adapter=FrozenPredictionCandidateAdapter(
            prediction_run_id=candidate_locator.prediction_run_id,
            candidate_id=candidate_locator.candidate_id,
            solver_id=candidate_locator.solver_id,
        ),
    )
    if replay.dataset_manifest.dataset_version != dataset_version:
        raise ValueError(
            "Frozen dataset input mismatch: manifest dataset_version does not "
            "match the requested dataset version."
        )

    references = load_blessed_references(
        cache.root,
        dataset_version=dataset_version,
        require_complete_for_promote=mode == "promote",
    )
    baseline_bundle = _load_reference_bundle(
        dataset_dir=dataset_dir,
        dataset_version=dataset_version,
        locator=(
            None
            if references.blessed_baseline is None
            else references.blessed_baseline.locator
        ),
    )
    last_blessed_bundle = _load_reference_bundle(
        dataset_dir=dataset_dir,
        dataset_version=dataset_version,
        locator=(
            None
            if references.last_blessed_candidate is None
            else references.last_blessed_candidate.locator
        ),
    )
    report = evaluate_benchmark_suite(
        replay.benchmark_input,
        blessed_baseline_bundle=baseline_bundle,
        last_blessed_bundle=last_blessed_bundle,
        per_seed_regression_gate=(
            None if mode == "benchmark" else _DEFAULT_PER_SEED_REGRESSION_GATE
        ),
    )
    output_dir = cache.root / "evaluation" / dataset_version / candidate_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "benchmark":
        payload = _build_benchmark_payload(report)
        report_path = output_dir / "benchmark-report.json"
        summary_path = output_dir / "benchmark-summary.md"
        summary_text = _render_benchmark_summary(report)
    else:
        payload = _build_promote_payload(report)
        report_path = output_dir / "promote-verdict.json"
        summary_path = output_dir / "promote-summary.md"
        summary_text = _render_promote_summary(report)

    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    return EvaluationOutputs(
        report_path=report_path,
        summary_path=summary_path,
        report_payload=payload,
        summary_text=summary_text,
    )


def _resolve_candidate_locator(
    *, dataset_dir: Path, dataset_version: str, candidate_id: str
) -> ReferenceBundleLocator:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frozen dataset manifest is missing: {manifest_path}"
        )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_mapping = _require_mapping(manifest_payload, context="Dataset manifest")
    if "round_id" in manifest_mapping:
        registry_locator = _load_candidate_locator_from_registry(
            dataset_dir=dataset_dir,
            dataset_version=dataset_version,
            candidate_id=candidate_id,
        )
        if registry_locator is not None:
            return registry_locator
        round_id = _require_str(manifest_mapping["round_id"], field_name="round_id")
        prediction_run_id = f"submitted-{round_id}"
        solver_id = _require_str(
            manifest_mapping.get("solver_id"), field_name="solver_id"
        )
        if candidate_id != prediction_run_id:
            raise ValueError(
                "Candidate resolution requires an explicit candidates.json "
                "mapping when candidate_id does not equal the frozen "
                "prediction_run_id."
            )
        return ReferenceBundleLocator(
            candidate_id=candidate_id,
            solver_id=solver_id,
            prediction_run_id=prediction_run_id,
        )
    raise ValueError(
        "evaluate-solution requires one frozen single-round dataset snapshot "
        "and rejects mixed or curated dataset manifests."
    )


def _load_candidate_locator_from_registry(
    *, dataset_dir: Path, dataset_version: str, candidate_id: str
) -> ReferenceBundleLocator | None:
    candidates_path = dataset_dir / _CANDIDATE_REGISTRY_FILENAME
    if not candidates_path.exists():
        return None
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    mapping = _require_mapping(payload, context="Evaluation candidates registry")
    schema_version = _require_str(
        mapping.get("schema_version"), field_name="schema_version"
    )
    if schema_version != _DEFAULT_CANDIDATE_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported evaluation candidate registry schema_version "
            f"{schema_version!r}."
        )
    registry_dataset_version = _require_str(
        mapping.get("dataset_version"), field_name="dataset_version"
    )
    if registry_dataset_version != dataset_version:
        raise ValueError(
            "Evaluation candidate registry mixes dataset versions; frozen "
            "inputs must use a single dataset version."
        )
    candidates_payload = mapping.get("candidates")
    if not isinstance(candidates_payload, list):
        raise ValueError("Evaluation candidates registry candidates must be a list.")
    for entry in candidates_payload:
        locator = _parse_reference_locator(entry, field_name="candidate")
        if locator is not None and locator.candidate_id == candidate_id:
            return locator
    raise ValueError(
        f"Candidate {candidate_id!r} is not registered in the frozen "
        "evaluation candidates registry."
    )


def _parse_reference_locator(
    payload: object, *, field_name: str
) -> ReferenceBundleLocator | None:
    if payload is None:
        return None
    mapping = _require_mapping(payload, context=field_name)
    return ReferenceBundleLocator(
        candidate_id=_require_str(
            mapping.get("candidate_id"), field_name="candidate_id"
        ),
        solver_id=_require_str(mapping.get("solver_id"), field_name="solver_id"),
        prediction_run_id=_require_str(
            mapping.get("prediction_run_id"), field_name="prediction_run_id"
        ),
    )


def _load_reference_bundle(
    *,
    dataset_dir: Path,
    dataset_version: str,
    locator: ReferenceBundleLocator | None,
) -> CandidatePredictionBundle | None:
    if locator is None:
        return None
    replay = load_offline_replay(
        dataset_dir=dataset_dir,
        candidate_bundle_adapter=FrozenPredictionCandidateAdapter(
            prediction_run_id=locator.prediction_run_id,
            candidate_id=locator.candidate_id,
            solver_id=locator.solver_id,
        ),
    )
    if replay.dataset_manifest.dataset_version != dataset_version:
        raise ValueError(
            "Reference bundle mixes dataset versions; evaluate-solution "
            "accepts frozen single-version inputs only."
        )
    return replay.candidate_bundle


def _build_benchmark_payload(report: BenchmarkSuiteReport) -> dict[str, object]:
    payload = report.to_payload()
    return {
        "mode": "benchmark",
        "status": "completed",
        "dataset_version": report.dataset_version,
        "round_id": report.round_id,
        "candidate_id": report.candidate_id,
        "solver_id": report.solver_id,
        "final_status": report.verdict,
        "reasons": list(report.verdict_reasons),
        "hard_gate_failures": [
            check.to_payload() for check in report.robustness.hard_gate_failures
        ],
        "baseline_deltas": cast(dict[str, object], payload["blessed_baseline"]),
        "last_blessed_deltas": cast(
            dict[str, object], payload["last_blessed_candidate"]
        ),
        "benchmark_report": payload,
    }


def _build_promote_payload(report: BenchmarkSuiteReport) -> dict[str, object]:
    payload = report.to_payload()
    reasons = list(report.verdict_reasons)
    if report.verdict == "pass":
        reasons.append("Candidate passed the offline evaluation suite.")
    return {
        "mode": "promote",
        "status": "completed",
        "dataset_version": report.dataset_version,
        "round_id": report.round_id,
        "candidate_id": report.candidate_id,
        "solver_id": report.solver_id,
        "final_status": report.verdict,
        "promotion_verdict": report.verdict,
        "reasons": reasons,
        "hard_gate_failures": [
            check.to_payload() for check in report.robustness.hard_gate_failures
        ],
        "baseline_deltas": cast(dict[str, object], payload["blessed_baseline"]),
        "last_blessed_deltas": cast(
            dict[str, object], payload["last_blessed_candidate"]
        ),
        "benchmark_report": payload,
    }


def _render_benchmark_summary(report: BenchmarkSuiteReport) -> str:
    return (
        f"benchmark {report.candidate_id} on {report.dataset_version}: "
        f"{report.verdict} (aggregate_score={report.aggregate.aggregate_score:.6f}, "
        f"hard_gate_failures={len(report.robustness.hard_gate_failures)})"
    )


def _render_promote_summary(report: BenchmarkSuiteReport) -> str:
    return (
        f"promote {report.candidate_id} on {report.dataset_version}: "
        f"{report.verdict} (reasons={len(report.verdict_reasons)})"
    )


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    return cast(dict[str, object], value)


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


__all__ = [
    "EvaluationOutputs",
    "BlessedReferences",
    "EvaluateMode",
    "ReferenceBundleLocator",
    "evaluate_solution",
]
