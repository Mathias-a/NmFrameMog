from __future__ import annotations

# ruff: noqa: UP035
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from typing import Protocol, cast
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse


class _UrlopenResponse(Protocol):
    def close(self) -> None: ...


class _LocalHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        state: object,
    ) -> None:
        super().__init__(server_address, handler)
        self.state = state


class LocalHTTPServer:
    def __init__(
        self,
        handler: type[BaseHTTPRequestHandler],
        state: object,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._server = _LocalHTTPServer((host, port), handler, state)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._started = False

    def __enter__(self) -> LocalHTTPServer:
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self._server.shutdown()
        self._server.server_close()
        if self._started:
            self._thread.join(timeout=1.0)

    @property
    def base_url(self) -> str:
        address = self._server.server_address
        host = str(address[0])
        port = int(address[1])
        return f"http://{host}:{port}"

    @property
    def state(self) -> object:
        return self._server.state


class UpstreamState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.employees: list[dict[str, object]] = []
        self.customers: list[dict[str, object]] = []
        self.products: list[dict[str, object]] = []
        self.projects: list[dict[str, object]] = []
        self.vouchers: list[dict[str, object]] = []
        self.travel_expenses: list[dict[str, object]] = []

    def create_employee(self, payload: dict[str, object]) -> dict[str, object]:
        with self.lock:
            employee = {
                "id": len(self.employees) + 1,
                "firstName": payload["firstName"],
                "lastName": payload["lastName"],
                "email": payload["email"],
                "isAdministrator": payload.get("isAdministrator", False),
            }
            self.employees.append(employee)
        return employee

    def create_customer(self, payload: dict[str, object]) -> dict[str, object]:
        with self.lock:
            customer = {
                "id": len(self.customers) + 1,
                "name": payload["name"],
                "email": payload["email"],
                "phoneNumber": payload["phoneNumber"],
            }
            self.customers.append(customer)
        return customer

    def create_product(self, payload: dict[str, object]) -> dict[str, object]:
        with self.lock:
            product = {
                "id": len(self.products) + 1,
                "name": payload["name"],
                "number": payload["number"],
                "priceExcludingVatCurrency": payload["priceExcludingVatCurrency"],
            }
            self.products.append(product)
        return product

    def create_project(self, payload: dict[str, object]) -> dict[str, object]:
        with self.lock:
            project = {
                "id": len(self.projects) + 1,
                "name": payload["name"],
                "number": payload["number"],
                "startDate": payload["startDate"],
                "projectManager": payload["projectManager"],
            }
            self.projects.append(project)
        return project

    def create_voucher(self, payload: dict[str, object]) -> dict[str, object]:
        with self.lock:
            voucher = {
                "id": len(self.vouchers) + 1,
                "description": payload["description"],
                "date": payload["date"],
                "postings": payload["postings"],
            }
            self.vouchers.append(voucher)
        return voucher

    def create_travel_expense(self, payload: dict[str, object]) -> dict[str, object]:
        with self.lock:
            travel_expense = {
                "id": len(self.travel_expenses) + 1,
                "title": payload["title"],
                "employee": payload["employee"],
            }
            self.travel_expenses.append(travel_expense)
        return travel_expense


class UpstreamTripletexHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/v2/employee":
            server = self.server
            if not isinstance(server, _LocalHTTPServer):
                raise RuntimeError("Unexpected server type.")
            state = server.state
            assert isinstance(state, UpstreamState)
            with state.lock:
                employees = list(state.employees)
            self._send_json(
                200, {"fullResultSize": len(employees), "values": employees}
            )
            return
        if parsed.path == "/v2/customer":
            server = self.server
            if not isinstance(server, _LocalHTTPServer):
                raise RuntimeError("Unexpected server type.")
            state = server.state
            assert isinstance(state, UpstreamState)
            with state.lock:
                customers = list(state.customers)
            self._send_json(
                200, {"fullResultSize": len(customers), "values": customers}
            )
            return
        if parsed.path == "/v2/product":
            server = self.server
            if not isinstance(server, _LocalHTTPServer):
                raise RuntimeError("Unexpected server type.")
            state = server.state
            assert isinstance(state, UpstreamState)
            with state.lock:
                products = list(state.products)
            self._send_json(200, {"fullResultSize": len(products), "values": products})
            return
        if parsed.path == "/v2/project":
            server = self.server
            if not isinstance(server, _LocalHTTPServer):
                raise RuntimeError("Unexpected server type.")
            state = server.state
            assert isinstance(state, UpstreamState)
            with state.lock:
                projects = list(state.projects)
            self._send_json(200, {"fullResultSize": len(projects), "values": projects})
            return
        if parsed.path == "/v2/ledger/voucher":
            server = self.server
            if not isinstance(server, _LocalHTTPServer):
                raise RuntimeError("Unexpected server type.")
            state = server.state
            assert isinstance(state, UpstreamState)
            with state.lock:
                vouchers = list(state.vouchers)
            self._send_json(200, {"fullResultSize": len(vouchers), "values": vouchers})
            return
        if parsed.path == "/v2/travelExpense":
            server = self.server
            if not isinstance(server, _LocalHTTPServer):
                raise RuntimeError("Unexpected server type.")
            state = server.state
            assert isinstance(state, UpstreamState)
            with state.lock:
                travel_expenses = list(state.travel_expenses)
            self._send_json(
                200,
                {"fullResultSize": len(travel_expenses), "values": travel_expenses},
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/v2/employee":
            body = self._read_json()
            server = self.server
            if not isinstance(server, _LocalHTTPServer):
                raise RuntimeError("Unexpected server type.")
            state = server.state
            assert isinstance(state, UpstreamState)
            employee = state.create_employee(body)
            self._send_json(201, {"value": employee})
            return
        self._send_json(404, {"error": "not found"})

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/v2/employee/999":
            self._send_json(422, {"error": "validation failed"})
            return
        self._send_json(404, {"error": "not found"})

    def _read_json(self) -> dict[str, object]:
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = cast(object, json.loads(raw.decode("utf-8")))
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object.")
        return cast(dict[str, object], payload)

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SolveState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.last_payload: dict[str, object] | None = None


class SolveHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_POST(self) -> None:
        if self.path != "/solve":
            self._send_json(404, {"error": "not found"})
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = cast(object, json.loads(raw.decode("utf-8")))
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object.")

        server = self.server
        if not isinstance(server, _LocalHTTPServer):
            raise RuntimeError("Unexpected server type.")
        state = server.state
        assert isinstance(state, SolveState)
        with state.lock:
            state.last_payload = cast(dict[str, object], payload)

        credentials = payload["tripletex_credentials"]
        if not isinstance(credentials, dict):
            raise ValueError("Expected tripletex_credentials object.")
        base_url = credentials["base_url"]
        token = credentials["session_token"]
        if not isinstance(base_url, str) or not isinstance(token, str):
            raise ValueError("Invalid credentials.")

        auth_header = self._basic_auth_header(token)
        self._send_request(
            f"{base_url}/employee?fields=id,firstName,lastName,email,isAdministrator",
            method="GET",
            headers={"Authorization": auth_header},
        )
        self._send_request(
            f"{base_url}/employee",
            method="POST",
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/json",
            },
            payload={
                "firstName": "Ola",
                "lastName": "Nordmann",
                "email": "ola.admin.test@example.org",
                "isAdministrator": True,
            },
        )
        self._send_request(
            f"{base_url}/employee/999",
            method="PUT",
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/json",
            },
            payload={"noop": True},
        )

        self._send_json(200, {"status": "completed"})

    @staticmethod
    def _basic_auth_header(session_token: str) -> str:
        import base64

        raw = f"0:{session_token}".encode()
        encoded = base64.b64encode(raw).decode("ascii")
        return f"Basic {encoded}"

    def _send_request(
        self,
        url: str,
        *,
        method: str,
        headers: dict[str, str],
        payload: dict[str, object] | None = None,
    ) -> None:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            response = cast(
                _UrlopenResponse, urllib_request.urlopen(request, timeout=10.0)
            )
            response.close()
        except urllib_error.HTTPError:
            pass

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
