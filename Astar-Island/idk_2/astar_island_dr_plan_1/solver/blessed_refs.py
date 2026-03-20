from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

ReferenceRole = Literal["blessed_baseline", "last_blessed_candidate"]

_DEFAULT_BLESSED_REFERENCE_SCHEMA_VERSION = "astar-blessed-reference-v1"
_REFERENCE_FILENAME_BY_ROLE: dict[ReferenceRole, str] = {
    "blessed_baseline": "blessed-baseline.json",
    "last_blessed_candidate": "last-blessed-candidate.json",
}


@dataclass(frozen=True)
class ReferenceBundleLocator:
    candidate_id: str
    solver_id: str
    prediction_run_id: str


@dataclass(frozen=True)
class BlessedReferenceKey:
    dataset_version: str
    candidate_id: str


@dataclass(frozen=True)
class DatasetVersionCompatibility:
    dataset_version: str
    status: Literal["exact"]

    def validate_for_requested_dataset(
        self, *, requested_dataset_version: str, role: ReferenceRole
    ) -> None:
        if self.status != "exact":
            raise ValueError(
                f"{_display_role(role)} compatibility status {self.status!r} is "
                "unsupported; expected 'exact'."
            )
        if self.dataset_version != requested_dataset_version:
            raise ValueError(
                f"{_display_role(role)} is incompatible with frozen dataset version "
                f"{requested_dataset_version!r}; compatibility targets "
                f"{self.dataset_version!r}."
            )


@dataclass(frozen=True)
class BlessedReferenceRecord:
    role: ReferenceRole
    reference_key: BlessedReferenceKey
    compatibility: DatasetVersionCompatibility
    locator: ReferenceBundleLocator


@dataclass(frozen=True)
class BlessedReferences:
    blessed_baseline: BlessedReferenceRecord | None = None
    last_blessed_candidate: BlessedReferenceRecord | None = None

    def require_complete_for_promote(self) -> None:
        missing_roles = tuple(
            role
            for role, record in (
                ("blessed_baseline", self.blessed_baseline),
                ("last_blessed_candidate", self.last_blessed_candidate),
            )
            if record is None
        )
        if not missing_roles:
            return
        if len(missing_roles) == 1:
            raise ValueError(
                "Promotion requires an explicit "
                f"{_display_role(cast(ReferenceRole, missing_roles[0]))} reference."
            )
        rendered_roles = " and ".join(
            _display_role(cast(ReferenceRole, role)) for role in missing_roles
        )
        raise ValueError(
            "Promotion requires explicit blessed references for both roles; missing "
            f"{rendered_roles}."
        )


def load_blessed_references(
    cache_root: Path,
    *,
    dataset_version: str,
    require_complete_for_promote: bool = False,
) -> BlessedReferences:
    evaluation_dir = cache_root / "evaluation" / dataset_version
    references = BlessedReferences(
        blessed_baseline=_load_reference_record(
            evaluation_dir=evaluation_dir,
            requested_dataset_version=dataset_version,
            role="blessed_baseline",
        ),
        last_blessed_candidate=_load_reference_record(
            evaluation_dir=evaluation_dir,
            requested_dataset_version=dataset_version,
            role="last_blessed_candidate",
        ),
    )
    if require_complete_for_promote:
        references.require_complete_for_promote()
    return references


def _load_reference_record(
    *,
    evaluation_dir: Path,
    requested_dataset_version: str,
    role: ReferenceRole,
) -> BlessedReferenceRecord | None:
    reference_path = evaluation_dir / _REFERENCE_FILENAME_BY_ROLE[role]
    if not reference_path.exists():
        return None
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    mapping = _require_mapping(
        payload,
        context=f"{_display_role(role)} registry",
    )
    schema_version = _require_str(
        mapping.get("schema_version"),
        field_name="schema_version",
    )
    if schema_version != _DEFAULT_BLESSED_REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {_display_role(role)} schema_version {schema_version!r}."
        )
    stored_role = _require_role(
        mapping.get("reference_role"),
        field_name="reference_role",
    )
    if stored_role != role:
        raise ValueError(
            f"{_display_role(role)} file is mislabeled as {_display_role(stored_role)}."
        )
    reference_key = _parse_reference_key(mapping.get("reference_key"))
    compatibility = _parse_dataset_version_compatibility(
        mapping.get("dataset_version_compatibility")
    )
    compatibility.validate_for_requested_dataset(
        requested_dataset_version=requested_dataset_version,
        role=role,
    )
    if reference_key.dataset_version != compatibility.dataset_version:
        raise ValueError(
            f"{_display_role(role)} reference_key dataset_version must match its "
            "dataset_version_compatibility target."
        )
    locator = ReferenceBundleLocator(
        candidate_id=reference_key.candidate_id,
        solver_id=_require_str(mapping.get("solver_id"), field_name="solver_id"),
        prediction_run_id=_require_str(
            mapping.get("prediction_run_id"),
            field_name="prediction_run_id",
        ),
    )
    return BlessedReferenceRecord(
        role=role,
        reference_key=reference_key,
        compatibility=compatibility,
        locator=locator,
    )


def _parse_reference_key(payload: object) -> BlessedReferenceKey:
    mapping = _require_mapping(payload, context="reference_key")
    return BlessedReferenceKey(
        dataset_version=_require_str(
            mapping.get("dataset_version"),
            field_name="dataset_version",
        ),
        candidate_id=_require_str(
            mapping.get("candidate_id"),
            field_name="candidate_id",
        ),
    )


def _parse_dataset_version_compatibility(
    payload: object,
) -> DatasetVersionCompatibility:
    mapping = _require_mapping(payload, context="dataset_version_compatibility")
    status = _require_str(mapping.get("status"), field_name="status")
    if status != "exact":
        raise ValueError(
            "dataset_version_compatibility.status must be 'exact'."
        )
    return DatasetVersionCompatibility(
        dataset_version=_require_str(
            mapping.get("dataset_version"),
            field_name="dataset_version",
        ),
        status="exact",
    )


def _display_role(role: ReferenceRole) -> str:
    return role.replace("_", " ")


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    return cast(dict[str, object], value)


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_role(value: object, *, field_name: str) -> ReferenceRole:
    role = _require_str(value, field_name=field_name)
    if role not in _REFERENCE_FILENAME_BY_ROLE:
        raise ValueError(
            f"{field_name} must be one of {sorted(_REFERENCE_FILENAME_BY_ROLE)}."
        )
    return cast(ReferenceRole, role)


__all__ = [
    "BlessedReferenceKey",
    "BlessedReferenceRecord",
    "BlessedReferences",
    "DatasetVersionCompatibility",
    "ReferenceBundleLocator",
    "ReferenceRole",
    "load_blessed_references",
]
