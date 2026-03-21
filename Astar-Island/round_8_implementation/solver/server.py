from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.parse import urlsplit

from .emulator import (
    AstarIslandEmulator,
    BudgetExceededError,
    EmulatorError,
    RoundInactiveError,
    RoundNotFoundError,
)

BASE_PATH = "/astar-island"


class AstarIslandHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        emulator: AstarIslandEmulator,
    ) -> None:
        super().__init__(server_address, AstarIslandRequestHandler)
        self.emulator = emulator


class AstarIslandRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if self._matches_path(path, "/rounds"):
            server = cast(AstarIslandHTTPServer, self.server)
            self._send_json(HTTPStatus.OK, server.emulator.list_rounds())
            return
        if self._matches_path(path, "/budget"):
            server = cast(AstarIslandHTTPServer, self.server)
            self._send_json(HTTPStatus.OK, server.emulator.get_budget())
            return
        analysis_request = self._extract_analysis_request(path)
        if analysis_request is not None:
            round_id, seed_index = analysis_request
            server = cast(AstarIslandHTTPServer, self.server)
            self._handle_emulator_call(
                lambda: server.emulator.get_analysis(round_id, seed_index)
            )
            return
        round_id = self._extract_round_id(path)
        if round_id is not None:
            server = cast(AstarIslandHTTPServer, self.server)
            self._handle_emulator_call(
                lambda: server.emulator.get_round_detail(round_id)
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            payload = self._read_json_body()
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        if self._matches_path(path, "/simulate"):
            server = cast(AstarIslandHTTPServer, self.server)
            self._handle_emulator_call(lambda: server.emulator.simulate(payload))
            return
        if self._matches_path(path, "/submit"):
            server = cast(AstarIslandHTTPServer, self.server)
            self._handle_emulator_call(lambda: server.emulator.submit(payload))
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle_emulator_call(self, action: Callable[[], object]) -> None:
        try:
            self._send_json(HTTPStatus.OK, action())
        except RoundNotFoundError as error:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
        except RoundInactiveError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except BudgetExceededError as error:
            self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(error)})
        except (ValueError, EmulatorError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _read_json_body(self) -> object:
        content_length_header = self.headers.get("Content-Length")
        if content_length_header is None:
            return {}
        try:
            content_length = int(content_length_header)
        except ValueError as error:
            raise ValueError("Invalid Content-Length header.") from error
        if content_length <= 0:
            return {}
        body = self.rfile.read(content_length)
        if not body:
            return {}
        try:
            decoded = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Request body must contain valid JSON.") from error
        return cast(object, decoded)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _matches_path(self, path: str, suffix: str) -> bool:
        return path == suffix or path == f"{BASE_PATH}{suffix}"

    def _extract_round_id(self, path: str) -> str | None:
        prefixes = ("/rounds/", f"{BASE_PATH}/rounds/")
        for prefix in prefixes:
            if path.startswith(prefix):
                round_id = path.removeprefix(prefix)
                if round_id:
                    return round_id
        return None

    def _extract_analysis_request(self, path: str) -> tuple[str, int] | None:
        prefixes = ("/analysis/", f"{BASE_PATH}/analysis/")
        for prefix in prefixes:
            if not path.startswith(prefix):
                continue
            remainder = path.removeprefix(prefix)
            round_id, separator, seed_index_raw = remainder.rpartition("/")
            if not separator or not round_id or not seed_index_raw:
                return None
            try:
                seed_index = int(seed_index_raw)
            except ValueError as error:
                raise ValueError(
                    "Seed index in analysis path must be an integer."
                ) from error
            return round_id, seed_index
        return None


def run_emulator_server(
    *,
    emulator: AstarIslandEmulator,
    host: str,
    port: int,
) -> int:
    with AstarIslandHTTPServer((host, port), emulator) as server:
        print(f"http://{host}:{port}{BASE_PATH}")
        server.serve_forever()
    return 0
