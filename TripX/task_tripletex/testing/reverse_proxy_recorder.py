from __future__ import annotations

# ruff: noqa: UP035
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from typing import Protocol, cast
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from task_tripletex.testing.models import ProxyMetrics, RecordedTripletexCall

_WRITE_METHODS = frozenset({"DELETE", "PATCH", "POST", "PUT"})
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _build_basic_auth_header(session_token: str) -> str:
    import base64

    raw = f"0:{session_token}".encode()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"Basic {encoded}"


def _strip_trailing_slash(value: str) -> str:
    return value[:-1] if value.endswith("/") else value


@dataclass(frozen=True)
class _ForwardedResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes


class _UrlopenResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self) -> bytes: ...

    def close(self) -> None: ...


class _HeadersWithItems(Protocol):
    def items(self) -> list[tuple[str, str]]: ...


class _RecorderHTTPServer(ThreadingHTTPServer):
    def __init__(
        self, server_address: tuple[str, int], recorder: ReverseProxyRecorder
    ) -> None:
        super().__init__(server_address, _ProxyRequestHandler)
        self.recorder = recorder


class _ProxyRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_DELETE(self) -> None:
        self._handle_request()

    def do_GET(self) -> None:
        self._handle_request()

    def do_PATCH(self) -> None:
        self._handle_request()

    def do_POST(self) -> None:
        self._handle_request()

    def do_PUT(self) -> None:
        self._handle_request()

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _handle_request(self) -> None:
        server = self.server
        if not isinstance(server, _RecorderHTTPServer):
            raise RuntimeError("Unexpected server type.")
        recorder = server.recorder
        response = recorder.forward_request(
            method=self.command,
            raw_path=self.path,
            incoming_headers={key: value for key, value in self.headers.items()},
            body=self.rfile.read(int(self.headers.get("Content-Length", "0"))),
        )

        self.send_response(response.status_code)
        for key, value in response.headers.items():
            if key.lower() in _HOP_BY_HOP_HEADERS:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)


class ReverseProxyRecorder:
    def __init__(
        self,
        upstream_base_url: str,
        session_token: str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout_seconds: float = 30.0,
    ) -> None:
        parsed = urlparse(upstream_base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("upstream_base_url must be an absolute http or https URL.")
        self._upstream_base_url = _strip_trailing_slash(upstream_base_url)
        self._upstream_origin = f"{parsed.scheme}://{parsed.netloc}"
        self._upstream_base_path = parsed.path.rstrip("/") or ""
        self._expected_auth_header = _build_basic_auth_header(session_token)
        self._timeout_seconds = timeout_seconds
        self._server = _RecorderHTTPServer((host, port), self)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._lock = Lock()
        self._calls: list[RecordedTripletexCall] = []
        self._started = False

    def start(self) -> ReverseProxyRecorder:
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._started:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> ReverseProxyRecorder:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    @property
    def advertised_base_url(self) -> str:
        address = self._server.server_address
        host = str(address[0])
        port = int(address[1])
        base_path = self._upstream_base_path
        return f"http://{host}:{port}{base_path}"

    @property
    def upstream_base_url(self) -> str:
        return self._upstream_base_url

    def recorded_calls(self) -> list[RecordedTripletexCall]:
        with self._lock:
            return list(self._calls)

    def forward_request(
        self,
        *,
        method: str,
        raw_path: str,
        incoming_headers: dict[str, str],
        body: bytes,
    ) -> _ForwardedResponse:
        forwarded_url = self._build_forwarded_url(raw_path)
        forwarded_headers = {
            key: value
            for key, value in incoming_headers.items()
            if key.lower() not in {"host", "content-length"}
        }
        request = urllib_request.Request(
            forwarded_url,
            data=body if body else None,
            headers=forwarded_headers,
            method=method,
        )
        try:
            response = cast(
                _UrlopenResponse,
                urllib_request.urlopen(request, timeout=self._timeout_seconds),
            )
            try:
                response_status_code = response.status
                response_headers = dict(response.headers.items())
                response_body = response.read()
            finally:
                response.close()
        except urllib_error.HTTPError as exc:
            response_status_code = exc.code
            response_headers = dict(cast(_HeadersWithItems, exc.headers).items())
            response_body = exc.read()
        filtered_headers = {
            key: value
            for key, value in response_headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS
            and key.lower() != "content-length"
        }
        request_body_text = body.decode("utf-8", errors="replace") if body else None
        call = RecordedTripletexCall(
            method=method,
            path=raw_path.split("?", maxsplit=1)[0],
            query_string=raw_path.split("?", maxsplit=1)[1] if "?" in raw_path else "",
            forwarded_url=forwarded_url,
            request_headers=dict(incoming_headers),
            request_body_text=request_body_text,
            response_status_code=response_status_code,
            response_body_text=response_body.decode("utf-8", errors="replace"),
            used_expected_basic_auth=incoming_headers.get("Authorization")
            == self._expected_auth_header,
            write_call=method.upper() in _WRITE_METHODS,
            client_error=400 <= response_status_code < 500,
        )
        with self._lock:
            self._calls.append(call)

        return _ForwardedResponse(
            status_code=response_status_code,
            headers=filtered_headers,
            body=response_body,
        )

    def summarize(
        self, *, rewritten_base_url: str, expected_min_proxy_calls: int
    ) -> ProxyMetrics:
        calls = self.recorded_calls()
        invalid_auth_paths = [
            f"{call.method} {call.path}"
            for call in calls
            if not call.used_expected_basic_auth
        ]
        invalid_forward_paths = [
            f"{call.method} {call.path}"
            for call in calls
            if not call.forwarded_url.startswith(self._upstream_base_url)
        ]
        return ProxyMetrics(
            total_calls=len(calls),
            write_calls=sum(1 for call in calls if call.write_call),
            client_error_calls=sum(1 for call in calls if call.client_error),
            used_proxy=len(calls) >= expected_min_proxy_calls,
            base_url_rewritten=rewritten_base_url == self.advertised_base_url,
            all_calls_used_expected_basic_auth=not invalid_auth_paths,
            all_calls_forwarded_to_upstream_base_url=not invalid_forward_paths,
            invalid_auth_paths=invalid_auth_paths,
            invalid_forward_paths=invalid_forward_paths,
            calls=calls,
        )

    def _build_forwarded_url(self, raw_path: str) -> str:
        path = raw_path if raw_path.startswith("/") else f"/{raw_path}"
        if self._upstream_base_path and not path.startswith(
            f"{self._upstream_base_path}/"
        ):
            if path == self._upstream_base_path:
                forwarded_path = path
            else:
                forwarded_path = f"{self._upstream_base_path}{path}"
        else:
            forwarded_path = path
        return f"{self._upstream_origin}{forwarded_path}"
