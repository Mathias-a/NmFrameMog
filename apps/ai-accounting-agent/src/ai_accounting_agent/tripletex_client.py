"""Typed Tripletex API client for the AI Accounting Agent."""

from __future__ import annotations

from types import TracebackType

import httpx

# Type aliases for API field values
FieldValue = str | int | float | bool | None
EntityDict = dict[str, FieldValue | dict[str, FieldValue] | list[dict[str, FieldValue]]]


class TripletexAPIError(Exception):
    """Raised when the Tripletex API returns a non-2xx response."""

    def __init__(self, status_code: int, message: str, endpoint: str) -> None:
        self.status_code = status_code
        self.message = message
        self.endpoint = endpoint
        super().__init__(f"Tripletex API error {status_code} on {endpoint}: {message}")


class TripletexClient:
    """Typed wrapper around the Tripletex REST API.

    Uses HTTP Basic Auth with username '0' and a session token as password.
    Designed for use in the NM i AI accounting agent competition.
    """

    def __init__(
        self,
        base_url: str,
        session_token: str,
        timeout: float = 30.0,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            auth=("0", session_token),
            timeout=timeout,
        )
        self.call_count: int = 0
        self.error_count: int = 0

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> TripletexClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Generic CRUD helpers
    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        """Reset API call and error counters."""
        self.call_count = 0
        self.error_count = 0

    def _raise_for_status(self, response: httpx.Response, endpoint: str) -> None:
        """Raise TripletexAPIError if the response is not 2xx."""
        if response.is_success:
            return
        self.error_count += 1
        try:
            body = response.text
        except Exception:  # noqa: BLE001
            body = "<unreadable body>"
        raise TripletexAPIError(
            status_code=response.status_code,
            message=body,
            endpoint=endpoint,
        )

    def _list(
        self,
        endpoint: str,
        params: dict[str, str | int] | None = None,
    ) -> list[EntityDict]:
        """GET /{endpoint} and return the 'values' list."""
        self.call_count += 1
        response = self._client.get(f"/{endpoint}", params=params)
        self._raise_for_status(response, endpoint)
        data: dict[str, list[EntityDict]] = response.json()
        return data.get("values", [])

    def _get(self, endpoint: str, entity_id: int) -> EntityDict:
        """GET /{endpoint}/{id} and return the 'value' dict."""
        self.call_count += 1
        url = f"/{endpoint}/{entity_id}"
        response = self._client.get(url)
        self._raise_for_status(response, url)
        data: dict[str, EntityDict] = response.json()
        return data["value"]

    def _create(self, endpoint: str, data: EntityDict) -> EntityDict:
        """POST /{endpoint} with JSON body, return the created 'value'."""
        self.call_count += 1
        response = self._client.post(f"/{endpoint}", json=data)
        self._raise_for_status(response, endpoint)
        result: dict[str, EntityDict] = response.json()
        return result["value"]

    def _update(
        self,
        endpoint: str,
        entity_id: int,
        data: EntityDict,
    ) -> EntityDict:
        """PUT /{endpoint}/{id} with JSON body, return the updated 'value'."""
        self.call_count += 1
        url = f"/{endpoint}/{entity_id}"
        response = self._client.put(url, json=data)
        self._raise_for_status(response, url)
        result: dict[str, EntityDict] = response.json()
        return result["value"]

    def _delete(self, endpoint: str, entity_id: int) -> bool:
        """DELETE /{endpoint}/{id}. Returns True on success."""
        self.call_count += 1
        url = f"/{endpoint}/{entity_id}"
        response = self._client.delete(url)
        self._raise_for_status(response, url)
        return True

    # ------------------------------------------------------------------
    # Connection check
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Verify authentication by fetching a single employee record.

        Returns True if the API responds with 2xx, False otherwise.
        """
        try:
            self._list("employee", params={"count": 1})
        except TripletexAPIError:
            return False
        return True

    def list_entity_ids(self, entity_type: str) -> set[int]:
        """Return the set of all entity IDs for a given type.

        Used for pre-task state snapshotting so the evaluator can
        distinguish newly-created entities from pre-existing ones.
        """
        # Map entity types to their endpoints and any required params
        endpoint_map: dict[str, str] = {
            "employee": "employee",
            "customer": "customer",
            "product": "product",
            "invoice": "invoice",
            "project": "project",
            "department": "department",
            "contact": "contact",
            "order": "order",
            "travelExpense": "travelExpense",
            "voucher": "ledger/voucher",
            "supplier": "supplier",
        }
        # Some endpoints require date range params
        date_params: dict[str, dict[str, str]] = {
            "invoice": {
                "invoiceDateFrom": "2020-01-01",
                "invoiceDateTo": "2030-12-31",
            },
            "order": {
                "orderDateFrom": "2020-01-01",
                "orderDateTo": "2030-12-31",
            },
            "ledger/voucher": {
                "dateFrom": "2020-01-01",
                "dateTo": "2030-12-31",
            },
        }
        endpoint = endpoint_map.get(entity_type, entity_type)
        params: dict[str, str | int] = {"fields": "id"}
        if endpoint in date_params:
            params.update(date_params[endpoint])
        entities = self._list(endpoint, params=params)
        result: set[int] = set()
        for entity in entities:
            entity_id = entity.get("id")
            if isinstance(entity_id, int):
                result.add(entity_id)
        return result

    # ------------------------------------------------------------------
    # Employee
    # ------------------------------------------------------------------

    def list_employees(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List employees with optional filter parameters."""
        return self._list("employee", params=params or None)

    def get_employee(self, employee_id: int) -> EntityDict:
        """Get a single employee by ID."""
        return self._get("employee", employee_id)

    def create_employee(self, data: EntityDict) -> EntityDict:
        """Create a new employee."""
        return self._create("employee", data)

    def search_employees(
        self,
        first_name: str | None = None,
        last_name: str | None = None,
        **params: str | int,
    ) -> list[EntityDict]:
        """Search employees by first/last name and optional extra filters."""
        search_params: dict[str, str | int] = dict(params)
        if first_name is not None:
            search_params["firstName"] = first_name
        if last_name is not None:
            search_params["lastName"] = last_name
        return self._list("employee", params=search_params)

    # ------------------------------------------------------------------
    # Customer
    # ------------------------------------------------------------------

    def list_customers(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List customers with optional filter parameters."""
        return self._list("customer", params=params or None)

    def get_customer(self, customer_id: int) -> EntityDict:
        """Get a single customer by ID."""
        return self._get("customer", customer_id)

    def create_customer(self, data: EntityDict) -> EntityDict:
        """Create a new customer."""
        return self._create("customer", data)

    def search_customers(
        self,
        name: str | None = None,
        **params: str | int,
    ) -> list[EntityDict]:
        """Search customers by name and optional extra filters."""
        search_params: dict[str, str | int] = dict(params)
        if name is not None:
            search_params["name"] = name
        return self._list("customer", params=search_params)

    # ------------------------------------------------------------------
    # Product
    # ------------------------------------------------------------------

    def list_products(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List products with optional filter parameters."""
        return self._list("product", params=params or None)

    def get_product(self, product_id: int) -> EntityDict:
        """Get a single product by ID."""
        return self._get("product", product_id)

    def create_product(self, data: EntityDict) -> EntityDict:
        """Create a new product."""
        return self._create("product", data)

    # ------------------------------------------------------------------
    # Invoice
    # ------------------------------------------------------------------

    def list_invoices(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List invoices with filter parameters.

        Note: Tripletex requires invoiceDateFrom and invoiceDateTo params.
        If not provided, defaults to a wide date range.
        """
        if "invoiceDateFrom" not in params:
            params["invoiceDateFrom"] = "2020-01-01"
        if "invoiceDateTo" not in params:
            params["invoiceDateTo"] = "2030-12-31"
        return self._list("invoice", params=params)

    def get_invoice(self, invoice_id: int) -> EntityDict:
        """Get a single invoice by ID."""
        return self._get("invoice", invoice_id)

    def create_invoice(self, data: EntityDict) -> EntityDict:
        """Create a new invoice."""
        return self._create("invoice", data)

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    def list_projects(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List projects with optional filter parameters."""
        return self._list("project", params=params or None)

    def get_project(self, project_id: int) -> EntityDict:
        """Get a single project by ID."""
        return self._get("project", project_id)

    def create_project(self, data: EntityDict) -> EntityDict:
        """Create a new project."""
        return self._create("project", data)

    # ------------------------------------------------------------------
    # Department
    # ------------------------------------------------------------------

    def list_departments(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List departments with optional filter parameters."""
        return self._list("department", params=params or None)

    def get_department(self, department_id: int) -> EntityDict:
        """Get a single department by ID."""
        return self._get("department", department_id)

    def create_department(self, data: EntityDict) -> EntityDict:
        """Create a new department."""
        return self._create("department", data)

    # ------------------------------------------------------------------
    # Contact
    # ------------------------------------------------------------------

    def list_contacts(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List contacts with optional filter parameters."""
        return self._list("contact", params=params or None)

    def create_contact(self, data: EntityDict) -> EntityDict:
        """Create a new contact."""
        return self._create("contact", data)

    # ------------------------------------------------------------------
    # Order
    # ------------------------------------------------------------------

    def list_orders(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List orders with filter parameters.

        Note: Tripletex requires orderDateFrom and orderDateTo params.
        If not provided, defaults to a wide date range.
        """
        if "orderDateFrom" not in params:
            params["orderDateFrom"] = "2020-01-01"
        if "orderDateTo" not in params:
            params["orderDateTo"] = "2030-12-31"
        return self._list("order", params=params)

    def get_order(self, order_id: int) -> EntityDict:
        """Get a single order by ID."""
        return self._get("order", order_id)

    def create_order(self, data: EntityDict) -> EntityDict:
        """Create a new order."""
        return self._create("order", data)

    # ------------------------------------------------------------------
    # Travel Expense
    # ------------------------------------------------------------------

    def list_travel_expenses(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List travel expenses with optional filter parameters."""
        return self._list("travelExpense", params=params or None)

    def get_travel_expense(self, expense_id: int) -> EntityDict:
        """Get a single travel expense by ID."""
        return self._get("travelExpense", expense_id)

    def create_travel_expense(self, data: EntityDict) -> EntityDict:
        """Create a new travel expense."""
        return self._create("travelExpense", data)

    def delete_travel_expense(self, expense_id: int) -> bool:
        """Delete a travel expense by ID."""
        return self._delete("travelExpense", expense_id)

    # ------------------------------------------------------------------
    # Voucher
    # ------------------------------------------------------------------

    def list_vouchers(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List vouchers with filter parameters.

        Note: Tripletex uses /ledger/voucher (not /voucher) and requires
        dateFrom and dateTo params.
        """
        if "dateFrom" not in params:
            params["dateFrom"] = "2020-01-01"
        if "dateTo" not in params:
            params["dateTo"] = "2030-12-31"
        return self._list("ledger/voucher", params=params)

    def get_voucher(self, voucher_id: int) -> EntityDict:
        """Get a single voucher by ID."""
        return self._get("ledger/voucher", voucher_id)

    # ------------------------------------------------------------------
    # Supplier
    # ------------------------------------------------------------------

    def list_suppliers(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List suppliers with optional filter parameters."""
        return self._list("supplier", params=params or None)

    def get_supplier(self, supplier_id: int) -> EntityDict:
        """Get a single supplier by ID."""
        return self._get("supplier", supplier_id)

    def create_supplier(self, data: EntityDict) -> EntityDict:
        """Create a new supplier."""
        return self._create("supplier", data)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def list_accounts(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List accounts with optional filter parameters."""
        return self._list("ledger/account", params=params or None)

    def get_account(self, account_id: int) -> EntityDict:
        """Get a single account by ID."""
        return self._get("ledger/account", account_id)

    # ------------------------------------------------------------------
    # Payment Type
    # ------------------------------------------------------------------

    def list_payment_types(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List payment types with optional filter parameters."""
        return self._list("ledger/paymentType", params=params or None)

    def get_payment_type(self, payment_type_id: int) -> EntityDict:
        """Get a single payment type by ID."""
        return self._get("ledger/paymentType", payment_type_id)

    # ------------------------------------------------------------------
    # Currency
    # ------------------------------------------------------------------

    def list_currencies(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List currencies with optional filter parameters."""
        return self._list("currency", params=params or None)

    def get_currency(self, currency_id: int) -> EntityDict:
        """Get a single currency by ID."""
        return self._get("currency", currency_id)

    # ------------------------------------------------------------------
    # Country
    # ------------------------------------------------------------------

    def list_countries(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List countries with optional filter parameters."""
        return self._list("country", params=params or None)

    def get_country(self, country_id: int) -> EntityDict:
        """Get a single country by ID."""
        return self._get("country", country_id)

    # ------------------------------------------------------------------
    # Activity
    # ------------------------------------------------------------------

    def list_activities(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List activities with optional filter parameters."""
        return self._list("activity", params=params or None)

    def get_activity(self, activity_id: int) -> EntityDict:
        """Get a single activity by ID."""
        return self._get("activity", activity_id)

    # ------------------------------------------------------------------
    # Salary Type
    # ------------------------------------------------------------------

    def list_salary_types(
        self,
        **params: str | int,
    ) -> list[EntityDict]:
        """List salary types with optional filter parameters."""
        return self._list("salary/type", params=params or None)

    def get_salary_type(self, salary_type_id: int) -> EntityDict:
        """Get a single salary type by ID."""
        return self._get("salary/type", salary_type_id)
