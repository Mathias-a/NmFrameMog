from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import cast

from task_tripletex.models import TripletexCredentials
from task_tripletex.testing.endpoint_runner import run_solve_endpoint
from task_tripletex.testing.fixture_loader import (
    build_solve_request,
    load_case_fixture,
    load_packaged_case_fixture,
)
from task_tripletex.testing.models import EvaluationCase, EvaluationResult
from task_tripletex.testing.reverse_proxy_recorder import ReverseProxyRecorder
from task_tripletex.testing.scoring import compute_score
from task_tripletex.testing.verifier import verify_case


@dataclass(frozen=True)
class CliArgs:
    case_file: str | None
    packaged_case: str | None
    solve_url: str
    tripletex_base_url: str
    session_token: str
    api_key: str | None
    timeout_seconds: float
    proxy_host: str
    proxy_port: int
    output: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a live TripX /solve endpoint against a declarative case fixture."
        )
    )
    fixture_group = parser.add_mutually_exclusive_group(required=True)
    fixture_group.add_argument("--case-file", help="Path to a JSON case fixture.")
    fixture_group.add_argument(
        "--packaged-case",
        help="Name of a packaged fixture under task_tripletex/testing/fixtures/.",
    )
    parser.add_argument(
        "--solve-url", required=True, help="Full URL to the /solve endpoint."
    )
    parser.add_argument(
        "--tripletex-base-url",
        required=True,
        help="Original upstream Tripletex base URL that the proxy should forward to.",
    )
    parser.add_argument(
        "--session-token",
        required=True,
        help=(
            "Tripletex session token; the evaluator verifies Basic Auth "
            "username 0 with this token."
        ),
    )
    parser.add_argument(
        "--api-key",
        help="Optional bearer token to send to the /solve endpoint.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for the solve request in seconds. Default: 300.",
    )
    parser.add_argument(
        "--proxy-host",
        default="127.0.0.1",
        help="Local host for the recording proxy. Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=0,
        help="Local port for the recording proxy. Default: 0 (ephemeral).",
    )
    parser.add_argument(
        "--output",
        choices=("json", "text"),
        default="text",
        help="Output format. Default: text.",
    )
    return parser


def _load_case(args: argparse.Namespace) -> EvaluationCase:
    typed_args = _typed_args(args)
    if typed_args.case_file is not None:
        return load_case_fixture(typed_args.case_file)
    if typed_args.packaged_case is None:
        raise ValueError("Either --case-file or --packaged-case is required.")
    return load_packaged_case_fixture(typed_args.packaged_case)


def _typed_args(args: argparse.Namespace) -> CliArgs:
    return CliArgs(
        case_file=cast(str | None, args.case_file),
        packaged_case=cast(str | None, args.packaged_case),
        solve_url=cast(str, args.solve_url),
        tripletex_base_url=cast(str, args.tripletex_base_url),
        session_token=cast(str, args.session_token),
        api_key=cast(str | None, args.api_key),
        timeout_seconds=cast(float, args.timeout_seconds),
        proxy_host=cast(str, args.proxy_host),
        proxy_port=cast(int, args.proxy_port),
        output=cast(str, args.output),
    )


def _result_to_jsonable(result: EvaluationResult) -> dict[str, object]:
    return {
        "case": {
            "case_id": result.case.case_id,
            "description": result.case.description,
            "tier": result.case.tier,
        },
        "endpoint_run": {
            "proxy_base_url": result.endpoint_run.proxy_base_url,
            "elapsed_seconds": result.endpoint_run.elapsed_seconds,
            "contract": {
                "response_status_code": (
                    result.endpoint_run.contract.response_status_code
                ),
                "response_content_type": (
                    result.endpoint_run.contract.response_content_type
                ),
                "response_json": result.endpoint_run.contract.response_json,
                "response_text": result.endpoint_run.contract.response_text,
                "exact_success_response": (
                    result.endpoint_run.contract.exact_success_response
                ),
                "errors": result.endpoint_run.contract.errors,
            },
        },
        "proxy_metrics": {
            "total_calls": result.proxy_metrics.total_calls,
            "write_calls": result.proxy_metrics.write_calls,
            "client_error_calls": result.proxy_metrics.client_error_calls,
            "invalid_auth_paths": result.proxy_metrics.invalid_auth_paths,
            "invalid_forward_paths": result.proxy_metrics.invalid_forward_paths,
            "calls": [
                {
                    "method": call.method,
                    "path": call.path,
                    "forwarded_url": call.forwarded_url,
                    "response_status_code": call.response_status_code,
                    "write_call": call.write_call,
                    "client_error": call.client_error,
                    "used_expected_basic_auth": call.used_expected_basic_auth,
                }
                for call in result.proxy_metrics.calls
            ],
        },
        "verification": {
            "points_earned": result.verification.points_earned,
            "max_points": result.verification.max_points,
            "correctness": result.verification.correctness,
            "snapshots": result.verification.snapshots,
        },
        "score": {
            "base_score": result.score.base_score,
            "efficiency_bonus": result.score.efficiency_bonus,
            "total_score": result.score.total_score,
            "write_efficiency": result.score.write_efficiency,
            "error_efficiency": result.score.error_efficiency,
            "combined_efficiency": result.score.combined_efficiency,
            "contract_valid": result.score.contract_valid,
            "proxy_valid": result.score.proxy_valid,
            "disqualification_reasons": result.score.disqualification_reasons,
        },
    }


async def _evaluate(args: argparse.Namespace) -> EvaluationResult:
    typed_args = _typed_args(args)
    case = _load_case(args)
    credentials = TripletexCredentials(
        base_url=typed_args.tripletex_base_url,
        session_token=typed_args.session_token,
    )
    request = build_solve_request(case, credentials)
    with ReverseProxyRecorder(
        credentials.base_url,
        credentials.session_token,
        host=typed_args.proxy_host,
        port=typed_args.proxy_port,
        timeout_seconds=min(typed_args.timeout_seconds, 30.0),
    ) as recorder:
        endpoint_run = await run_solve_endpoint(
            typed_args.solve_url,
            request,
            proxy_base_url=recorder.advertised_base_url,
            api_key=typed_args.api_key,
            timeout_seconds=typed_args.timeout_seconds,
        )
        proxy_metrics = recorder.summarize(
            rewritten_base_url=endpoint_run.rewritten_request.tripletex_credentials.base_url,
            expected_min_proxy_calls=case.expected_min_proxy_calls,
        )
    verification = await verify_case(case, credentials)
    score = compute_score(case, verification, proxy_metrics, endpoint_run.contract)
    return EvaluationResult(
        case=case,
        endpoint_run=endpoint_run,
        proxy_metrics=proxy_metrics,
        verification=verification,
        score=score,
    )


def _format_text(result: EvaluationResult) -> str:
    contract = result.endpoint_run.contract
    proxy = result.proxy_metrics
    verification = result.verification
    score = result.score
    lines = [
        f"Case: {result.case.case_id} (tier {result.case.tier})",
        (
            "Solve response: "
            f"HTTP {contract.response_status_code}, "
            f"exact success body={contract.exact_success_response}"
        ),
        (
            f"Proxy: calls={proxy.total_calls}, writes={proxy.write_calls}, "
            f"4xx={proxy.client_error_calls}, "
            "basic_auth_ok="
            f"{proxy.all_calls_used_expected_basic_auth}"
        ),
        (
            "Correctness: "
            f"{verification.points_earned:.2f}/{verification.max_points:.2f} "
            f"= {verification.correctness:.3f}"
        ),
        (
            f"Score: base={score.base_score:.3f}, "
            f"efficiency_bonus={score.efficiency_bonus:.3f}, "
            f"total={score.total_score:.3f}"
        ),
    ]
    if contract.errors:
        lines.append("Contract issues:")
        lines.extend(f"- {error}" for error in contract.errors)
    if proxy.invalid_auth_paths:
        lines.append("Auth issues:")
        lines.extend(f"- {path}" for path in proxy.invalid_auth_paths)
    if proxy.invalid_forward_paths:
        lines.append("Forwarding issues:")
        lines.extend(f"- {path}" for path in proxy.invalid_forward_paths)
    if score.disqualification_reasons:
        lines.append("DISQUALIFIED — score zeroed due to:")
        lines.extend(f"- {reason}" for reason in score.disqualification_reasons)
    if proxy.calls:
        lines.append(f"Recorded proxy calls ({len(proxy.calls)}):")
        for call in proxy.calls:
            status_tag = ""
            if call.write_call:
                status_tag += " [write]"
            if call.client_error:
                status_tag += " [4xx]"
            if not call.used_expected_basic_auth:
                status_tag += " [bad-auth]"
            lines.append(
                f"  {call.method} {call.path} -> "
                f"{call.response_status_code}{status_tag}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    typed_args = _typed_args(args)
    result = asyncio.run(_evaluate(args))
    if typed_args.output == "json":
        print(json.dumps(_result_to_jsonable(result), indent=2, ensure_ascii=False))
    else:
        print(_format_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
