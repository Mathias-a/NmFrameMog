from __future__ import annotations

from typing import Protocol

from task_tripletex.testing.cli import main
from tests.task_tripletex_testing.helpers import (
    LocalHTTPServer,
    SolveHandler,
    SolveState,
    UpstreamState,
    UpstreamTripletexHandler,
)


class _CapturedOutput(Protocol):
    out: str


class _Capsys(Protocol):
    def readouterr(self) -> _CapturedOutput: ...


def test_cli_runs_against_local_servers(capsys: _Capsys) -> None:
    upstream_state = UpstreamState()
    solve_state = SolveState()

    with LocalHTTPServer(UpstreamTripletexHandler, upstream_state) as upstream:
        with LocalHTTPServer(SolveHandler, solve_state) as solve_server:
            exit_code = main(
                [
                    "--packaged-case",
                    "create_employee_admin",
                    "--solve-url",
                    f"{solve_server.base_url}/solve",
                    "--tripletex-base-url",
                    f"{upstream.base_url}/v2",
                    "--session-token",
                    "test-session-token",
                    "--output",
                    "text",
                ]
            )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Case: create_employee_admin (tier 1)" in captured.out
    assert "writes=2, 4xx=1" in captured.out
    assert "Recorded proxy calls" in captured.out
    assert "DISQUALIFIED" in captured.out
