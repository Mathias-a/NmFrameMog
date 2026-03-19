"""Task definitions for local evaluation of the Tripletex AI Accounting Agent.

Each TaskDefinition describes a prompt, expected API entity, and field-level
scoring so the local evaluator can measure agent accuracy per language and tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    """Categories of accounting tasks the agent must handle."""

    CREATE_EMPLOYEE = "create_employee"
    CREATE_CUSTOMER = "create_customer"
    CREATE_PRODUCT = "create_product"
    CREATE_INVOICE = "create_invoice"
    CREATE_PROJECT = "create_project"
    CREATE_DEPARTMENT = "create_department"
    CREATE_CONTACT = "create_contact"
    REGISTER_PAYMENT = "register_payment"
    CREATE_TRAVEL_EXPENSE = "create_travel_expense"
    DELETE_TRAVEL_EXPENSE = "delete_travel_expense"
    CREATE_CREDIT_NOTE = "create_credit_note"
    UPDATE_EMPLOYEE = "update_employee"
    UPDATE_CUSTOMER = "update_customer"


class Language(Enum):
    """Languages that prompts may be written in."""

    NB = "nb"  # Norwegian Bokmal
    EN = "en"
    ES = "es"
    PT = "pt"
    NN = "nn"  # Nynorsk
    DE = "de"
    FR = "fr"


class Tier(Enum):
    """Difficulty tiers with corresponding point multipliers."""

    TIER_1 = 1  # x1
    TIER_2 = 2  # x2
    TIER_3 = 3  # x3


@dataclass(frozen=True)
class TaskDefinition:
    """A single test task for the local evaluator.

    Attributes:
        name: Human-readable task identifier.
        prompt: The natural-language instruction sent to the agent.
        language: Language the prompt is written in.
        task_type: Category of the task.
        tier: Difficulty tier (affects scoring multiplier).
        expected_entity: Tripletex API entity to verify, e.g. "employee".
        expected_fields: Mapping of field names to their expected values.
        files: Optional file attachments (FileAttachment-compatible dicts).
        max_points: Maximum raw points for this task.
        field_points: Per-field point allocation.
    """

    name: str
    prompt: str
    language: Language
    task_type: TaskType
    tier: Tier
    expected_entity: str
    expected_fields: dict[str, str | int | float | bool]
    files: list[dict[str, str]] = field(default_factory=list)
    max_points: float = 5.0
    field_points: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

_EMPLOYEE_FIELD_POINTS: dict[str, float] = {
    "_found": 2.0,
    "firstName": 1.0,
    "lastName": 1.0,
    "email": 1.0,
}

_CREATE_EMPLOYEE_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="create_employee_nb",
        prompt=(
            "Opprett en ny ansatt med fornavn Ola og etternavn"
            " Nordmann med e-post ola@example.com"
        ),
        language=Language.NB,
        task_type=TaskType.CREATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={
            "firstName": "Ola",
            "lastName": "Nordmann",
            "email": "ola@example.com",
        },
        field_points=_EMPLOYEE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_employee_en",
        prompt=(
            "Create a new employee with first name John"
            " and last name Smith with email john@example.com"
        ),
        language=Language.EN,
        task_type=TaskType.CREATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={
            "firstName": "John",
            "lastName": "Smith",
            "email": "john@example.com",
        },
        field_points=_EMPLOYEE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_employee_de",
        prompt=(
            "Erstellen Sie einen neuen Mitarbeiter mit Vorname"
            " Hans und Nachname Müller mit E-Mail hans@example.com"
        ),
        language=Language.DE,
        task_type=TaskType.CREATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={
            "firstName": "Hans",
            "lastName": "Müller",
            "email": "hans@example.com",
        },
        field_points=_EMPLOYEE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_employee_fr",
        prompt=(
            "Créez un nouvel employé avec le prénom Pierre"
            " et le nom Dupont avec l'email pierre@example.com"
        ),
        language=Language.FR,
        task_type=TaskType.CREATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={
            "firstName": "Pierre",
            "lastName": "Dupont",
            "email": "pierre@example.com",
        },
        field_points=_EMPLOYEE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_employee_es",
        prompt=(
            "Cree un nuevo empleado con nombre Juan y apellido"
            " García con correo juan@example.com"
        ),
        language=Language.ES,
        task_type=TaskType.CREATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={
            "firstName": "Juan",
            "lastName": "García",
            "email": "juan@example.com",
        },
        field_points=_EMPLOYEE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_employee_pt",
        prompt=(
            "Crie um novo funcionário com nome João"
            " e sobrenome Silva com email joao@example.com"
        ),
        language=Language.PT,
        task_type=TaskType.CREATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={
            "firstName": "João",
            "lastName": "Silva",
            "email": "joao@example.com",
        },
        field_points=_EMPLOYEE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_employee_nn",
        prompt=(
            "Opprett ein ny tilsett med fornamn Kari og"
            " etternamn Berg med e-post kari@example.com"
        ),
        language=Language.NN,
        task_type=TaskType.CREATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={
            "firstName": "Kari",
            "lastName": "Berg",
            "email": "kari@example.com",
        },
        field_points=_EMPLOYEE_FIELD_POINTS,
    ),
]

# -- CREATE_CUSTOMER --------------------------------------------------------

_CUSTOMER_FIELD_POINTS: dict[str, float] = {
    "_found": 2.0,
    "name": 2.0,
    "email": 1.0,
    "isCustomer": 1.0,
}

_CREATE_CUSTOMER_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="create_customer_nb",
        prompt="Opprett en ny kunde med navn Nordisk AS og e-post post@nordisk.no",
        language=Language.NB,
        task_type=TaskType.CREATE_CUSTOMER,
        tier=Tier.TIER_1,
        expected_entity="customer",
        expected_fields={
            "name": "Nordisk AS",
            "email": "post@nordisk.no",
            "isCustomer": True,
        },
        max_points=6.0,
        field_points=_CUSTOMER_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_customer_en",
        prompt="Create a new customer named Nordic Corp with email info@nordic.com",
        language=Language.EN,
        task_type=TaskType.CREATE_CUSTOMER,
        tier=Tier.TIER_1,
        expected_entity="customer",
        expected_fields={
            "name": "Nordic Corp",
            "email": "info@nordic.com",
            "isCustomer": True,
        },
        max_points=6.0,
        field_points=_CUSTOMER_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_customer_de",
        prompt=(
            "Erstellen Sie einen neuen Kunden mit dem Namen"
            " Nordisch GmbH und E-Mail info@nordisch.de"
        ),
        language=Language.DE,
        task_type=TaskType.CREATE_CUSTOMER,
        tier=Tier.TIER_1,
        expected_entity="customer",
        expected_fields={
            "name": "Nordisch GmbH",
            "email": "info@nordisch.de",
            "isCustomer": True,
        },
        max_points=6.0,
        field_points=_CUSTOMER_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_customer_es",
        prompt="Cree un nuevo cliente llamado Nórdico SL con correo info@nordico.es",
        language=Language.ES,
        task_type=TaskType.CREATE_CUSTOMER,
        tier=Tier.TIER_1,
        expected_entity="customer",
        expected_fields={
            "name": "Nórdico SL",
            "email": "info@nordico.es",
            "isCustomer": True,
        },
        max_points=6.0,
        field_points=_CUSTOMER_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_customer_fr",
        prompt=(
            "Créez un nouveau client nommé Nordique SARL"
            " avec l'email contact@nordique.fr"
        ),
        language=Language.FR,
        task_type=TaskType.CREATE_CUSTOMER,
        tier=Tier.TIER_1,
        expected_entity="customer",
        expected_fields={
            "name": "Nordique SARL",
            "email": "contact@nordique.fr",
            "isCustomer": True,
        },
        max_points=6.0,
        field_points=_CUSTOMER_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_customer_pt",
        prompt=(
            "Crie um novo cliente com nome Nórdico Ltda e email contato@nordico.com.br"
        ),
        language=Language.PT,
        task_type=TaskType.CREATE_CUSTOMER,
        tier=Tier.TIER_1,
        expected_entity="customer",
        expected_fields={
            "name": "Nórdico Ltda",
            "email": "contato@nordico.com.br",
            "isCustomer": True,
        },
        max_points=6.0,
        field_points=_CUSTOMER_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_customer_nn",
        prompt=(
            "Opprett ein ny kunde med namn Vestlandsk AS og e-post post@vestlandsk.no"
        ),
        language=Language.NN,
        task_type=TaskType.CREATE_CUSTOMER,
        tier=Tier.TIER_1,
        expected_entity="customer",
        expected_fields={
            "name": "Vestlandsk AS",
            "email": "post@vestlandsk.no",
            "isCustomer": True,
        },
        max_points=6.0,
        field_points=_CUSTOMER_FIELD_POINTS,
    ),
]

# -- CREATE_PRODUCT ---------------------------------------------------------

# NOTE: Verify actual API field name against sandbox; may differ
_PRODUCT_FIELD_POINTS: dict[str, float] = {
    "_found": 2.0,
    "name": 2.0,
    "priceExcludingVatCurrency": 2.0,
}

_CREATE_PRODUCT_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="create_product_nb",
        prompt="Opprett et nytt produkt med navn Kontorpult og pris 4999.00",
        language=Language.NB,
        task_type=TaskType.CREATE_PRODUCT,
        tier=Tier.TIER_1,
        expected_entity="product",
        expected_fields={
            "name": "Kontorpult",
            "priceExcludingVatCurrency": 4999.00,
        },
        max_points=6.0,
        field_points=_PRODUCT_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_product_en",
        prompt="Create a new product called Office Desk with price 4999.00",
        language=Language.EN,
        task_type=TaskType.CREATE_PRODUCT,
        tier=Tier.TIER_1,
        expected_entity="product",
        expected_fields={
            "name": "Office Desk",
            "priceExcludingVatCurrency": 4999.00,
        },
        max_points=6.0,
        field_points=_PRODUCT_FIELD_POINTS,
    ),
]

# -- CREATE_INVOICE ---------------------------------------------------------

_INVOICE_FIELD_POINTS: dict[str, float] = {
    "_found": 3.0,
    "invoiceDate": 2.0,
    "invoiceDueDate": 2.0,
}

_CREATE_INVOICE_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="create_invoice_nb",
        prompt=(
            "Opprett en faktura for kunden Nordisk AS"
            " datert 2026-03-15 med forfallsdato 2026-04-15"
        ),
        language=Language.NB,
        task_type=TaskType.CREATE_INVOICE,
        tier=Tier.TIER_2,
        expected_entity="invoice",
        expected_fields={
            "invoiceDate": "2026-03-15",
            "invoiceDueDate": "2026-04-15",
        },
        max_points=7.0,
        field_points=_INVOICE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_invoice_en",
        prompt=(
            "Create an invoice for customer Nordic Corp dated 2026-03-15 due 2026-04-15"
        ),
        language=Language.EN,
        task_type=TaskType.CREATE_INVOICE,
        tier=Tier.TIER_2,
        expected_entity="invoice",
        expected_fields={
            "invoiceDate": "2026-03-15",
            "invoiceDueDate": "2026-04-15",
        },
        max_points=7.0,
        field_points=_INVOICE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_invoice_nn",
        prompt=(
            "Opprett ein faktura for kunden Vestlandsk AS"
            " datert 2026-03-15 med forfallsdato 2026-04-15"
        ),
        language=Language.NN,
        task_type=TaskType.CREATE_INVOICE,
        tier=Tier.TIER_2,
        expected_entity="invoice",
        expected_fields={
            "invoiceDate": "2026-03-15",
            "invoiceDueDate": "2026-04-15",
        },
        max_points=7.0,
        field_points=_INVOICE_FIELD_POINTS,
    ),
]

# -- CREATE_PROJECT ---------------------------------------------------------

_PROJECT_FIELD_POINTS: dict[str, float] = {
    "_found": 3.0,
    "name": 3.0,
}

_CREATE_PROJECT_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="create_project_nb",
        prompt="Opprett et nytt prosjekt med navn Digitaliseringsprosjektet",
        language=Language.NB,
        task_type=TaskType.CREATE_PROJECT,
        tier=Tier.TIER_2,
        expected_entity="project",
        expected_fields={"name": "Digitaliseringsprosjektet"},
        max_points=6.0,
        field_points=_PROJECT_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_project_en",
        prompt="Create a new project named Digitalization Project",
        language=Language.EN,
        task_type=TaskType.CREATE_PROJECT,
        tier=Tier.TIER_2,
        expected_entity="project",
        expected_fields={"name": "Digitalization Project"},
        max_points=6.0,
        field_points=_PROJECT_FIELD_POINTS,
    ),
]

# -- CREATE_DEPARTMENT ------------------------------------------------------

_DEPARTMENT_FIELD_POINTS: dict[str, float] = {
    "_found": 3.0,
    "name": 3.0,
}

_CREATE_DEPARTMENT_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="create_department_nb",
        prompt="Opprett en ny avdeling med navn Markedsføring",
        language=Language.NB,
        task_type=TaskType.CREATE_DEPARTMENT,
        tier=Tier.TIER_1,
        expected_entity="department",
        expected_fields={"name": "Markedsføring"},
        max_points=6.0,
        field_points=_DEPARTMENT_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_department_en",
        prompt="Create a new department named Marketing",
        language=Language.EN,
        task_type=TaskType.CREATE_DEPARTMENT,
        tier=Tier.TIER_1,
        expected_entity="department",
        expected_fields={"name": "Marketing"},
        max_points=6.0,
        field_points=_DEPARTMENT_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_department_de",
        prompt="Erstellen Sie eine neue Abteilung mit dem Namen Vertrieb",
        language=Language.DE,
        task_type=TaskType.CREATE_DEPARTMENT,
        tier=Tier.TIER_1,
        expected_entity="department",
        expected_fields={"name": "Vertrieb"},
        max_points=6.0,
        field_points=_DEPARTMENT_FIELD_POINTS,
    ),
]

# -- CREATE_TRAVEL_EXPENSE --------------------------------------------------

# Sandbox-validated: travelExpense has title, employee{id}, date, amount fields.
# employee.firstName works via Tripletex field expansion.
_TRAVEL_EXPENSE_FIELD_POINTS: dict[str, float] = {
    "_found": 2.0,
    "title": 2.0,
    "employee.firstName": 1.0,
}

_CREATE_TRAVEL_EXPENSE_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="create_travel_expense_nb",
        prompt=(
            "Registrer en ny reiseregning med tittel Konferansereise"
            " for ansatt med fornavn Ola"
        ),
        language=Language.NB,
        task_type=TaskType.CREATE_TRAVEL_EXPENSE,
        tier=Tier.TIER_2,
        expected_entity="travelExpense",
        expected_fields={"title": "Konferansereise", "employee.firstName": "Ola"},
        field_points=_TRAVEL_EXPENSE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_travel_expense_en",
        prompt=(
            "Register a new travel expense titled Conference Trip"
            " for employee with first name John"
        ),
        language=Language.EN,
        task_type=TaskType.CREATE_TRAVEL_EXPENSE,
        tier=Tier.TIER_2,
        expected_entity="travelExpense",
        expected_fields={"title": "Conference Trip", "employee.firstName": "John"},
        field_points=_TRAVEL_EXPENSE_FIELD_POINTS,
    ),
]

# -- UPDATE_EMPLOYEE --------------------------------------------------------

_UPDATE_EMPLOYEE_FIELD_POINTS: dict[str, float] = {
    "_found": 2.0,
    "email": 3.0,
}

_UPDATE_EMPLOYEE_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="update_employee_nb",
        prompt="Oppdater ansatt Ola Nordmann med ny e-post ola.ny@example.com",
        language=Language.NB,
        task_type=TaskType.UPDATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={"email": "ola.ny@example.com"},
        field_points=_UPDATE_EMPLOYEE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="update_employee_en",
        prompt="Update employee John Smith with new email john.new@example.com",
        language=Language.EN,
        task_type=TaskType.UPDATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={"email": "john.new@example.com"},
        field_points=_UPDATE_EMPLOYEE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="update_employee_es",
        prompt=(
            "Actualice el empleado Juan García con nuevo correo juan.nuevo@example.com"
        ),
        language=Language.ES,
        task_type=TaskType.UPDATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={"email": "juan.nuevo@example.com"},
        field_points=_UPDATE_EMPLOYEE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="update_employee_fr",
        prompt=(
            "Mettez à jour l'employé Pierre Dupont avec"
            " le nouvel email pierre.nouveau@example.com"
        ),
        language=Language.FR,
        task_type=TaskType.UPDATE_EMPLOYEE,
        tier=Tier.TIER_1,
        expected_entity="employee",
        expected_fields={"email": "pierre.nouveau@example.com"},
        field_points=_UPDATE_EMPLOYEE_FIELD_POINTS,
    ),
]

# -- CREATE_CONTACT ---------------------------------------------------------

_CONTACT_FIELD_POINTS: dict[str, float] = {
    "_found": 2.0,
    "firstName": 1.0,
    "lastName": 1.0,
    "email": 1.0,
}

_CREATE_CONTACT_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="create_contact_nb",
        prompt=(
            "Opprett en ny kontaktperson med fornavn Erik og"
            " etternavn Hansen med e-post erik@example.com"
        ),
        language=Language.NB,
        task_type=TaskType.CREATE_CONTACT,
        tier=Tier.TIER_1,
        expected_entity="contact",
        expected_fields={
            "firstName": "Erik",
            "lastName": "Hansen",
            "email": "erik@example.com",
        },
        field_points=_CONTACT_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_contact_en",
        prompt=(
            "Create a new contact with first name Alice and"
            " last name Brown with email alice@example.com"
        ),
        language=Language.EN,
        task_type=TaskType.CREATE_CONTACT,
        tier=Tier.TIER_1,
        expected_entity="contact",
        expected_fields={
            "firstName": "Alice",
            "lastName": "Brown",
            "email": "alice@example.com",
        },
        field_points=_CONTACT_FIELD_POINTS,
    ),
]

# -- UPDATE_CUSTOMER --------------------------------------------------------

_UPDATE_CUSTOMER_FIELD_POINTS: dict[str, float] = {
    "_found": 2.0,
    "email": 3.0,
}

_UPDATE_CUSTOMER_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="update_customer_nb",
        prompt="Oppdater kunden Nordisk AS med ny e-post ny@nordisk.no",
        language=Language.NB,
        task_type=TaskType.UPDATE_CUSTOMER,
        tier=Tier.TIER_1,
        expected_entity="customer",
        expected_fields={"email": "ny@nordisk.no"},
        field_points=_UPDATE_CUSTOMER_FIELD_POINTS,
    ),
    TaskDefinition(
        name="update_customer_en",
        prompt="Update customer Nordic Corp with new email new@nordic.com",
        language=Language.EN,
        task_type=TaskType.UPDATE_CUSTOMER,
        tier=Tier.TIER_1,
        expected_entity="customer",
        expected_fields={"email": "new@nordic.com"},
        field_points=_UPDATE_CUSTOMER_FIELD_POINTS,
    ),
]

# -- REGISTER_PAYMENT -------------------------------------------------------
# NOTE: Payment verification needs sandbox testing to determine the correct
# API entity and field names. The Tripletex API may use /payment, /ledger/payment,
# or verify via invoice.amountOutstanding. These expected_fields are best-effort.

_REGISTER_PAYMENT_FIELD_POINTS: dict[str, float] = {
    "_found": 3.0,
    "amountOutstanding": 2.0,
}

_REGISTER_PAYMENT_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="register_payment_nb",
        prompt="Registrer en betaling på 15000 kroner for faktura til Nordisk AS",
        language=Language.NB,
        task_type=TaskType.REGISTER_PAYMENT,
        tier=Tier.TIER_2,
        expected_entity="invoice",
        expected_fields={"amountOutstanding": 0.0},
        field_points=_REGISTER_PAYMENT_FIELD_POINTS,
    ),
    TaskDefinition(
        name="register_payment_en",
        prompt="Register a payment of 15000 NOK for the invoice to Nordic Corp",
        language=Language.EN,
        task_type=TaskType.REGISTER_PAYMENT,
        tier=Tier.TIER_2,
        expected_entity="invoice",
        expected_fields={"amountOutstanding": 0.0},
        field_points=_REGISTER_PAYMENT_FIELD_POINTS,
    ),
]

# -- DELETE_TRAVEL_EXPENSE --------------------------------------------------

_DELETE_TRAVEL_EXPENSE_FIELD_POINTS: dict[str, float] = {
    "_found": 5.0,
}

_DELETE_TRAVEL_EXPENSE_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="delete_travel_expense_nb",
        prompt="Slett reiseregningen til ansatt med fornavn Ola",
        language=Language.NB,
        task_type=TaskType.DELETE_TRAVEL_EXPENSE,
        tier=Tier.TIER_2,
        expected_entity="travelExpense",
        expected_fields={"employee.firstName": "Ola"},
        field_points=_DELETE_TRAVEL_EXPENSE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="delete_travel_expense_en",
        prompt="Delete the travel expense for employee with first name John",
        language=Language.EN,
        task_type=TaskType.DELETE_TRAVEL_EXPENSE,
        tier=Tier.TIER_2,
        expected_entity="travelExpense",
        expected_fields={"employee.firstName": "John"},
        field_points=_DELETE_TRAVEL_EXPENSE_FIELD_POINTS,
    ),
]

# -- CREATE_CREDIT_NOTE -----------------------------------------------------

_CREDIT_NOTE_FIELD_POINTS: dict[str, float] = {
    "_found": 3.0,
    "invoiceDate": 2.0,
}

_CREATE_CREDIT_NOTE_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        name="create_credit_note_nb",
        prompt=("Opprett en kreditnota for faktura til Nordisk AS datert 2026-03-20"),
        language=Language.NB,
        task_type=TaskType.CREATE_CREDIT_NOTE,
        tier=Tier.TIER_2,
        expected_entity="invoice",
        expected_fields={"invoiceDate": "2026-03-20"},
        field_points=_CREDIT_NOTE_FIELD_POINTS,
    ),
    TaskDefinition(
        name="create_credit_note_en",
        prompt="Create a credit note for the invoice to Nordic Corp dated 2026-03-20",
        language=Language.EN,
        task_type=TaskType.CREATE_CREDIT_NOTE,
        tier=Tier.TIER_2,
        expected_entity="invoice",
        expected_fields={"invoiceDate": "2026-03-20"},
        field_points=_CREDIT_NOTE_FIELD_POINTS,
    ),
]

# ---------------------------------------------------------------------------
# Aggregate registry
# ---------------------------------------------------------------------------

ALL_TASKS: list[TaskDefinition] = [
    *_CREATE_EMPLOYEE_TASKS,
    *_CREATE_CUSTOMER_TASKS,
    *_CREATE_PRODUCT_TASKS,
    *_CREATE_INVOICE_TASKS,
    *_CREATE_PROJECT_TASKS,
    *_CREATE_DEPARTMENT_TASKS,
    *_CREATE_TRAVEL_EXPENSE_TASKS,
    *_UPDATE_EMPLOYEE_TASKS,
    *_CREATE_CONTACT_TASKS,
    *_UPDATE_CUSTOMER_TASKS,
    *_REGISTER_PAYMENT_TASKS,
    *_DELETE_TRAVEL_EXPENSE_TASKS,
    *_CREATE_CREDIT_NOTE_TASKS,
]


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_tasks_by_type(task_type: TaskType) -> list[TaskDefinition]:
    """Return all tasks matching the given task type."""
    return [task for task in ALL_TASKS if task.task_type == task_type]


def get_tasks_by_language(language: Language) -> list[TaskDefinition]:
    """Return all tasks matching the given language."""
    return [task for task in ALL_TASKS if task.language == language]


def get_tasks_by_tier(tier: Tier) -> list[TaskDefinition]:
    """Return all tasks matching the given difficulty tier."""
    return [task for task in ALL_TASKS if task.tier == tier]


def get_all_task_types() -> set[TaskType]:
    """Return the set of all task types present in the library."""
    return {task.task_type for task in ALL_TASKS}


def get_all_languages() -> set[Language]:
    """Return the set of all languages present in the library."""
    return {task.language for task in ALL_TASKS}
