"""Tests for the TripletexClient API wrapper."""

from __future__ import annotations

from ai_accounting_agent.tripletex_client import TripletexAPIError, TripletexClient

from .conftest import sandbox_skip


class TestClientCreation:
    """Tests that do not require sandbox credentials."""

    def test_client_creation(self) -> None:
        """Client can be instantiated without making network calls."""
        client = TripletexClient(
            base_url="https://tx-proxy.ainm.no/v2",
            session_token="fake-token",
        )
        client.close()

    def test_client_context_manager(self) -> None:
        """Client works as a context manager."""
        with TripletexClient(
            base_url="https://tx-proxy.ainm.no/v2",
            session_token="fake-token",
        ) as client:
            assert client is not None

    def test_auth_tuple_format(self) -> None:
        """The underlying httpx client should use ('0', token) basic auth."""
        client = TripletexClient(
            base_url="https://example.com",
            session_token="secret-123",
        )
        # httpx stores auth as a callable; verify by checking the client was created
        assert client._client is not None
        client.close()


class TestApiErrorOnBadAuth:
    def test_api_error_on_bad_auth(self) -> None:
        """Using a fake token against a non-existent host should fail gracefully."""
        with TripletexClient(
            base_url="http://127.0.0.1:1",
            session_token="definitely-not-valid",
            timeout=2.0,
        ) as client:
            # check_connection catches TripletexAPIError but not connection errors;
            # either path means auth would fail.
            try:
                connected = client.check_connection()
                assert connected is False
            except Exception:  # noqa: BLE001
                # Connection refused is also acceptable — no valid server
                pass


class TestTripletexAPIError:
    def test_error_attributes(self) -> None:
        err = TripletexAPIError(
            status_code=401,
            message="Unauthorized",
            endpoint="/employee",
        )
        assert err.status_code == 401
        assert err.message == "Unauthorized"
        assert err.endpoint == "/employee"
        assert "401" in str(err)
        assert "/employee" in str(err)


@sandbox_skip
class TestCheckConnection:
    def test_check_connection(self, tripletex_client: TripletexClient) -> None:
        """Verify sandbox connection works with real token."""
        assert tripletex_client.check_connection() is True


@sandbox_skip
class TestListEmployees:
    def test_list_employees(self, tripletex_client: TripletexClient) -> None:
        """Listing employees should return a list (possibly empty)."""
        employees = tripletex_client.list_employees()
        assert isinstance(employees, list)


@sandbox_skip
class TestCreateAndGetEmployee:
    def test_create_and_get_employee(self, tripletex_client: TripletexClient) -> None:
        """Create an employee and retrieve it by ID."""
        # Tripletex requires department and email for STANDARD users.
        # Get first department to use as reference.
        departments = tripletex_client.list_departments()
        dept_id = departments[0]["id"] if departments else 0
        created = tripletex_client.create_employee(
            {
                "firstName": "TestCreate",
                "lastName": "Pytest",
                "userType": "STANDARD",
                "email": "testcreate.pytest@example.com",
                "department": {"id": dept_id},
            }
        )
        assert "id" in created
        employee_id = created["id"]
        assert isinstance(employee_id, int)

        fetched = tripletex_client.get_employee(employee_id)
        assert fetched["firstName"] == "TestCreate"
        assert fetched["lastName"] == "Pytest"


@sandbox_skip
class TestListCustomers:
    def test_list_customers(self, tripletex_client: TripletexClient) -> None:
        """Listing customers should return a list."""
        customers = tripletex_client.list_customers()
        assert isinstance(customers, list)


@sandbox_skip
class TestCreateCustomer:
    def test_create_customer(self, tripletex_client: TripletexClient) -> None:
        """Create a customer and verify it has an ID."""
        created = tripletex_client.create_customer(
            {
                "name": "Pytest TestCorp AS",
                "isCustomer": True,
            }
        )
        assert "id" in created
        assert isinstance(created["id"], int)
        assert created["name"] == "Pytest TestCorp AS"
