"""Guarded local-nonproduction Warehouse intake and human-review lifecycle.

This module is deliberately API-independent.  It consumes active records from
the R454 repository, persists only bounded identifiers/classifications in a
task-local SQLite database, and has no Library or promotion adapter.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from admin.f13_analytics_improvement_candidate_contract import (
    validate_analytics_improvement_candidate,
)
from admin.f13_local_nonprod_analytics_repository import validate_local_repository_path


REPOSITORY_SCHEMA_VERSION = 1
SCHEMA_VERSION = 1
CONTRACT_VERSION = "warehouse.local_nonproduction.review.v1"
ITEM_TYPE = "analytics_improvement_candidate"

ITEM_STATUSES = {
    "captured",
    "classified",
    "source_verified",
    "review_ready",
    "reviewed",
    "approved_for_warehouse",
    "hold_source_missing",
    "hold_sensitive",
    "hold_copyright",
    "hold_review_needed",
    "rejected",
    "quarantined",
}
FINAL_STATUSES = {
    "approved_for_warehouse",
    "hold_source_missing",
    "hold_sensitive",
    "hold_copyright",
    "hold_review_needed",
    "rejected",
    "quarantined",
}
TRANSITIONS = {
    "captured": {"classified"},
    "classified": {"source_verified"},
    "source_verified": {"review_ready"},
    "reviewed": {"approved_for_warehouse"},
}
DECISIONS = {
    "APPROVE_WAREHOUSE",
    "REQUEST_MORE_EVIDENCE",
    "REQUEST_RIGHTS_REVIEW",
    "REQUEST_DOMAIN_REVIEW",
    "REJECT",
    "QUARANTINE",
}
REVIEWER_ROLES = {"CURATOR", "DOMAIN_EXPERT", "RIGHTS_REVIEWER", "OWNER"}
ROLE_DECISIONS = {
    "CURATOR": {"APPROVE_WAREHOUSE", "REQUEST_MORE_EVIDENCE", "REJECT", "QUARANTINE"},
    "DOMAIN_EXPERT": {"REQUEST_DOMAIN_REVIEW", "REJECT", "QUARANTINE"},
    "RIGHTS_REVIEWER": {"REQUEST_RIGHTS_REVIEW", "REJECT", "QUARANTINE"},
    "OWNER": DECISIONS,
}
RIGHTS_STATUSES = {
    "owned",
    "licensed",
    "permission_granted",
    "public_reference",
    "internal_only",
    "no_export",
    "unknown",
}
SENSITIVITIES = {"public", "internal", "restricted", "private", "secret"}
APPROVABLE_RIGHTS = {"owned", "licensed", "permission_granted", "public_reference"}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,159}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDEMPOTENCY_RE = re.compile(r"^idem:warehouse:[0-9a-f]{64}$")
_SQLITE_HEADER = b"SQLite format 3\x00"
_PROHIBITED_FIELDS = {
    "raw_query", "query", "raw_body", "body", "prompt", "raw_prompt", "raw_text",
    "answer", "answer_text", "safe_short_answer", "hold_source", "evidence",
    "evidence_text", "standard_text", "paid_standard_text", "user_id", "user_id_hash",
    "personal_name", "email", "phone", "address", "secret", "api_key", "access_token",
    "refresh_token", "password", "credential", "cookie", "authorization", "internal_path",
    "local_path", "file_path", "db_path", "dsn", "connection_string", "library_id",
    "promoted_library_id", "promotion_trace_id", "promotion_dry_run_result",
}

_ITEM_FIELDS = (
    "schema_version", "contract_version", "warehouse_item_id", "source_candidate_id",
    "source_candidate_record_id", "tenant_id", "organization_id", "cohort_id",
    "source_event_id", "source_request_id", "source_trace_id", "query_hash", "item_type",
    "provenance", "rights_status", "sensitivity", "classification", "current_status",
    "previous_status", "revision", "review_required", "approved_for_library", "auto_promote",
    "idempotency_key", "created_at", "updated_at",
)

_SCHEMA_SQL = """
CREATE TABLE warehouse_repository_metadata (
    metadata_key TEXT PRIMARY KEY,
    metadata_value TEXT NOT NULL
);
CREATE TABLE warehouse_intake_items (
    warehouse_item_id TEXT PRIMARY KEY,
    source_candidate_id TEXT NOT NULL,
    source_candidate_record_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    organization_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    current_status TEXT NOT NULL CHECK (current_status IN (
        'captured','classified','source_verified','review_ready','reviewed',
        'approved_for_warehouse','hold_source_missing','hold_sensitive',
        'hold_copyright','hold_review_needed','rejected','quarantined'
    )),
    revision INTEGER NOT NULL CHECK (revision > 0),
    intake_idempotency_key TEXT NOT NULL,
    intake_request_hash TEXT NOT NULL,
    last_transition_idempotency_key TEXT,
    last_transition_request_hash TEXT,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (tenant_id, organization_id, source_candidate_id),
    UNIQUE (tenant_id, organization_id, intake_idempotency_key)
);
CREATE TABLE warehouse_review_events (
    review_event_id TEXT PRIMARY KEY,
    warehouse_item_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    organization_id TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    reviewer_role TEXT NOT NULL CHECK (reviewer_role IN ('CURATOR','DOMAIN_EXPERT','RIGHTS_REVIEWER','OWNER')),
    decision TEXT NOT NULL CHECK (decision IN (
        'APPROVE_WAREHOUSE','REQUEST_MORE_EVIDENCE','REQUEST_RIGHTS_REVIEW',
        'REQUEST_DOMAIN_REVIEW','REJECT','QUARANTINE'
    )),
    decision_reason_code TEXT NOT NULL,
    previous_status TEXT NOT NULL,
    next_status TEXT NOT NULL,
    expected_revision INTEGER NOT NULL CHECK (expected_revision > 0),
    new_revision INTEGER NOT NULL CHECK (new_revision = expected_revision + 1),
    approval_event_id TEXT,
    reviewed_at TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    FOREIGN KEY (warehouse_item_id) REFERENCES warehouse_intake_items (warehouse_item_id),
    UNIQUE (tenant_id, organization_id, idempotency_key),
    CHECK ((decision = 'APPROVE_WAREHOUSE' AND approval_event_id IS NOT NULL)
        OR (decision <> 'APPROVE_WAREHOUSE' AND approval_event_id IS NULL))
);
CREATE INDEX idx_warehouse_items_scope_status
    ON warehouse_intake_items (tenant_id, organization_id, current_status);
CREATE INDEX idx_warehouse_reviews_scope_item
    ON warehouse_review_events (tenant_id, organization_id, warehouse_item_id);
"""
_SCHEMA_FINGERPRINT = hashlib.sha256(_SCHEMA_SQL.encode("utf-8")).hexdigest()


class WarehouseLifecycleError(RuntimeError):
    """Fail-closed local lifecycle error containing a controlled reason code."""


def _now(clock: Callable[[], datetime] | None) -> datetime:
    value = clock() if clock else datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise WarehouseLifecycleError("TIMEZONE_AWARE_CLOCK_REQUIRED")
    return value.astimezone(timezone.utc)


def _timestamp(value: Any, reason: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise WarehouseLifecycleError(reason) from exc
    else:
        raise WarehouseLifecycleError(reason)
    if parsed.tzinfo is None:
        raise WarehouseLifecycleError(reason)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_identifier(value: Any, reason: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise WarehouseLifecycleError(reason)
    return value


def _safe_scope(tenant_id: Any, organization_id: Any) -> tuple[str, str]:
    return (
        _safe_identifier(tenant_id, "TENANT_SCOPE_REQUIRED"),
        _safe_identifier(organization_id, "ORGANIZATION_SCOPE_REQUIRED"),
    )


def _safe_idempotency(value: Any) -> str:
    if not isinstance(value, str) or _IDEMPOTENCY_RE.fullmatch(value) is None:
        raise WarehouseLifecycleError("INVALID_IDEMPOTENCY_KEY")
    return value


def canonical_payload(payload: Mapping[str, Any]) -> tuple[str, str]:
    serialized = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return serialized, "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_warehouse_idempotency_key(*parts: str) -> str:
    if not parts or any(_IDENTIFIER_RE.fullmatch(part) is None for part in parts):
        raise WarehouseLifecycleError("INVALID_IDEMPOTENCY_INPUT")
    return "idem:warehouse:" + hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _prohibited_fields(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key).lower()
            path = f"{prefix}.{name}" if prefix else name
            if name in _PROHIBITED_FIELDS:
                found.append(path)
            found.extend(_prohibited_fields(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_prohibited_fields(nested, f"{prefix}[{index}]"))
    return sorted(set(found))


def validate_warehouse_item(payload: Any) -> dict[str, Any]:
    invalid: list[str] = []
    if not isinstance(payload, Mapping):
        return {"valid": False, "reason_code": "ITEM_NOT_MAPPING", "invalid_fields": []}
    if set(payload) != set(_ITEM_FIELDS):
        invalid.append("FIELD_SET")
    prohibited = _prohibited_fields(payload)
    if prohibited:
        invalid.append("PROHIBITED_FIELD")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "item_type": ITEM_TYPE,
        "review_required": True,
        "approved_for_library": False,
        "auto_promote": False,
    }
    for field, expected in fixed.items():
        if payload.get(field) != expected:
            invalid.append(field)
    for field in (
        "warehouse_item_id", "source_candidate_id", "source_candidate_record_id", "tenant_id",
        "organization_id", "cohort_id", "source_event_id", "source_request_id", "source_trace_id",
    ):
        if not isinstance(payload.get(field), str) or _IDENTIFIER_RE.fullmatch(payload[field]) is None:
            invalid.append(field)
    if not isinstance(payload.get("query_hash"), str) or _HASH_RE.fullmatch(payload["query_hash"]) is None:
        invalid.append("query_hash")
    if payload.get("current_status") not in ITEM_STATUSES:
        invalid.append("current_status")
    if payload.get("previous_status") is not None and payload.get("previous_status") not in ITEM_STATUSES:
        invalid.append("previous_status")
    if not isinstance(payload.get("revision"), int) or payload["revision"] < 1:
        invalid.append("revision")
    if payload.get("rights_status") not in RIGHTS_STATUSES:
        invalid.append("rights_status")
    if payload.get("sensitivity") not in SENSITIVITIES:
        invalid.append("sensitivity")
    if not isinstance(payload.get("classification"), Mapping) or not payload["classification"]:
        invalid.append("classification")
    provenance = payload.get("provenance")
    required_provenance = {"provider_type", "provider_ref", "source_event_id", "source_trace_id", "collection_reason"}
    if not isinstance(provenance, Mapping) or set(provenance) != required_provenance:
        invalid.append("provenance")
    if not isinstance(payload.get("idempotency_key"), str) or _IDEMPOTENCY_RE.fullmatch(payload["idempotency_key"]) is None:
        invalid.append("idempotency_key")
    for field in ("created_at", "updated_at"):
        try:
            _timestamp(payload.get(field), "INVALID_TIMESTAMP")
        except WarehouseLifecycleError:
            invalid.append(field)
    return {
        "valid": not invalid,
        "reason_code": "ITEM_VALID" if not invalid else "ITEM_INVALID",
        "invalid_fields": sorted(set(invalid)),
    }


def validate_review_event(payload: Any) -> dict[str, Any]:
    required = {
        "review_event_id", "warehouse_item_id", "reviewer_id", "reviewer_role", "decision",
        "decision_reason_code", "previous_status", "next_status", "expected_revision",
        "new_revision", "approval_event_id", "reviewed_at", "idempotency_key",
    }
    invalid: list[str] = []
    if not isinstance(payload, Mapping):
        return {"valid": False, "reason_code": "REVIEW_NOT_MAPPING", "invalid_fields": []}
    if set(payload) != required:
        invalid.append("FIELD_SET")
    for field in ("review_event_id", "warehouse_item_id", "reviewer_id", "decision_reason_code"):
        if not isinstance(payload.get(field), str) or _IDENTIFIER_RE.fullmatch(payload[field]) is None:
            invalid.append(field)
    if payload.get("reviewer_role") not in REVIEWER_ROLES:
        invalid.append("reviewer_role")
    if payload.get("decision") not in DECISIONS:
        invalid.append("decision")
    if payload.get("previous_status") not in ITEM_STATUSES or payload.get("next_status") not in ITEM_STATUSES:
        invalid.append("status")
    expected = payload.get("expected_revision")
    if not isinstance(expected, int) or expected < 1 or payload.get("new_revision") != expected + 1:
        invalid.append("revision")
    approval = payload.get("approval_event_id")
    if payload.get("decision") == "APPROVE_WAREHOUSE":
        if not isinstance(approval, str) or _IDENTIFIER_RE.fullmatch(approval) is None:
            invalid.append("approval_event_id")
    elif approval is not None:
        invalid.append("approval_event_id")
    if not isinstance(payload.get("idempotency_key"), str) or _IDEMPOTENCY_RE.fullmatch(payload["idempotency_key"]) is None:
        invalid.append("idempotency_key")
    try:
        _timestamp(payload.get("reviewed_at"), "INVALID_REVIEWED_AT")
    except WarehouseLifecycleError:
        invalid.append("reviewed_at")
    return {
        "valid": not invalid,
        "reason_code": "REVIEW_VALID" if not invalid else "REVIEW_INVALID",
        "invalid_fields": sorted(set(invalid)),
    }


class LocalNonprodWarehouseReviewLifecycle:
    def __init__(
        self,
        database_path: str | Path,
        *,
        approved_local_root: str | Path,
        environment: str,
        explicit_allow_local_durable: bool,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.database_path, self.approved_local_root = validate_local_repository_path(
            database_path,
            approved_local_root=approved_local_root,
            environment=environment,
            explicit_allow_local_durable=explicit_allow_local_durable,
        )
        self._clock = clock
        self._connection: sqlite3.Connection | None = None
        existed = self.database_path.exists()
        if existed:
            with self.database_path.open("rb") as handle:
                if handle.read(len(_SQLITE_HEADER)) != _SQLITE_HEADER:
                    raise WarehouseLifecycleError("CORRUPT_OR_NON_SQLITE_STORAGE")
        else:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(self.database_path, timeout=5, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            self._connection = connection
            if existed:
                self._verify_existing_schema()
            else:
                self._initialize_schema()
        except (sqlite3.DatabaseError, OSError, WarehouseLifecycleError) as exc:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if not existed and self.database_path.exists():
                self.database_path.unlink()
            if isinstance(exc, WarehouseLifecycleError):
                raise
            raise WarehouseLifecycleError("WAREHOUSE_REPOSITORY_OPEN_FAILED") from exc

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise WarehouseLifecycleError("WAREHOUSE_REPOSITORY_CLOSED")
        return self._connection

    def _initialize_schema(self) -> None:
        try:
            self.connection.executescript(
                "BEGIN IMMEDIATE;\n" + _SCHEMA_SQL
                + "\nINSERT INTO warehouse_repository_metadata VALUES ('repository_schema_version','1');"
                + f"\nINSERT INTO warehouse_repository_metadata VALUES ('repository_schema_fingerprint','{_SCHEMA_FINGERPRINT}');\nCOMMIT;"
            )
        except Exception:
            if self.connection.in_transaction:
                self.connection.rollback()
            raise

    def _verify_existing_schema(self) -> None:
        row = self.connection.execute("PRAGMA integrity_check").fetchone()
        if row is None or row[0] != "ok":
            raise WarehouseLifecycleError("WAREHOUSE_INTEGRITY_CHECK_FAILED")
        try:
            version = self.connection.execute(
                "SELECT metadata_value FROM warehouse_repository_metadata WHERE metadata_key='repository_schema_version'"
            ).fetchone()
            fingerprint = self.connection.execute(
                "SELECT metadata_value FROM warehouse_repository_metadata WHERE metadata_key='repository_schema_fingerprint'"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise WarehouseLifecycleError("WAREHOUSE_SCHEMA_METADATA_INVALID") from exc
        if version is None or version[0] != str(REPOSITORY_SCHEMA_VERSION):
            raise WarehouseLifecycleError("WAREHOUSE_SCHEMA_VERSION_UNSUPPORTED")
        if fingerprint is None or fingerprint[0] != _SCHEMA_FINGERPRINT:
            raise WarehouseLifecycleError("WAREHOUSE_SCHEMA_FINGERPRINT_INVALID")
        required = {"warehouse_repository_metadata", "warehouse_intake_items", "warehouse_review_events"}
        actual = {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not required <= actual:
            raise WarehouseLifecycleError("WAREHOUSE_SCHEMA_INCOMPLETE")

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "LocalNonprodWarehouseReviewLifecycle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def integrity_check(self) -> dict[str, Any]:
        row = self.connection.execute("PRAGMA integrity_check").fetchone()
        ok = row is not None and row[0] == "ok"
        return {"integrity_ok": ok, "reason_code": "INTEGRITY_OK" if ok else "INTEGRITY_FAILED"}

    def _load_item_row(self, warehouse_item_id: Any, tenant_id: Any, organization_id: Any) -> sqlite3.Row | None:
        item_id = _safe_identifier(warehouse_item_id, "INVALID_WAREHOUSE_ITEM_ID")
        tenant, organization = _safe_scope(tenant_id, organization_id)
        return self.connection.execute(
            "SELECT * FROM warehouse_intake_items WHERE warehouse_item_id=? AND tenant_id=? AND organization_id=?",
            (item_id, tenant, organization),
        ).fetchone()

    @staticmethod
    def _verified_item(row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_json"])
        serialized, digest = canonical_payload(payload)
        if serialized != row["payload_json"] or digest != row["payload_hash"]:
            raise WarehouseLifecycleError("WAREHOUSE_ITEM_INTEGRITY_FAILED")
        if not validate_warehouse_item(payload)["valid"]:
            raise WarehouseLifecycleError("WAREHOUSE_ITEM_INVALID")
        return payload

    def read_item(self, warehouse_item_id: Any, *, tenant_id: Any, organization_id: Any) -> dict[str, Any]:
        row = self._load_item_row(warehouse_item_id, tenant_id, organization_id)
        if row is None:
            return {"found": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "item": None}
        return {"found": True, "reason_code": "RECORD_FOUND", "item": self._verified_item(row)}

    def read_review_event_by_approval(
        self,
        warehouse_item_id: Any,
        approval_event_id: Any,
        *,
        tenant_id: Any,
        organization_id: Any,
    ) -> dict[str, Any]:
        """Return one scoped Warehouse approval event without exposing storage internals."""

        item_id = _safe_identifier(warehouse_item_id, "INVALID_WAREHOUSE_ITEM_ID")
        approval_id = _safe_identifier(approval_event_id, "APPROVAL_EVENT_REQUIRED")
        tenant, organization = _safe_scope(tenant_id, organization_id)
        row = self.connection.execute(
            "SELECT payload_json, payload_hash FROM warehouse_review_events "
            "WHERE warehouse_item_id=? AND approval_event_id=? AND tenant_id=? AND organization_id=? "
            "AND decision='APPROVE_WAREHOUSE'",
            (item_id, approval_id, tenant, organization),
        ).fetchone()
        if row is None:
            return {
                "found": False,
                "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE",
                "review_event": None,
            }
        event = json.loads(row["payload_json"])
        serialized, digest = canonical_payload(event)
        if serialized != row["payload_json"] or digest != row["payload_hash"]:
            raise WarehouseLifecycleError("WAREHOUSE_REVIEW_EVENT_INTEGRITY_FAILED")
        if not validate_review_event(event)["valid"]:
            raise WarehouseLifecycleError("WAREHOUSE_REVIEW_EVENT_INVALID")
        return {"found": True, "reason_code": "RECORD_FOUND", "review_event": event}

    def list_items(self, *, tenant_id: Any, organization_id: Any) -> dict[str, Any]:
        tenant, organization = _safe_scope(tenant_id, organization_id)
        rows = self.connection.execute(
            "SELECT * FROM warehouse_intake_items WHERE tenant_id=? AND organization_id=? ORDER BY warehouse_item_id",
            (tenant, organization),
        ).fetchall()
        items = [self._verified_item(row) for row in rows]
        return {"reason_code": "RECORD_LIST", "count": len(items), "items": items}

    def count_items(self, *, tenant_id: Any, organization_id: Any) -> int:
        tenant, organization = _safe_scope(tenant_id, organization_id)
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM warehouse_intake_items WHERE tenant_id=? AND organization_id=?",
            (tenant, organization),
        ).fetchone()[0])

    @staticmethod
    def _read_active_source(source_repository: Any, candidate_record_id: str, tenant: str, organization: str) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_result = source_repository.read_record(
            candidate_record_id, tenant_id=tenant, organization_id=organization
        )
        if not isinstance(candidate_result, Mapping) or candidate_result.get("found") is not True:
            raise WarehouseLifecycleError("CANDIDATE_NOT_ACTIVE_OR_VISIBLE")
        candidate = candidate_result.get("domain_object")
        validation = validate_analytics_improvement_candidate(candidate)
        if not validation.get("valid") or not isinstance(candidate, Mapping):
            raise WarehouseLifecycleError("CANDIDATE_INVALID")
        if _prohibited_fields(candidate):
            raise WarehouseLifecycleError("CANDIDATE_PROHIBITED_FIELDS")
        context = candidate.get("tenant_context")
        if not isinstance(context, Mapping) or context.get("tenant_id") != tenant or context.get("organization_id") != organization:
            raise WarehouseLifecycleError("CANDIDATE_SCOPE_MISMATCH")
        if candidate.get("review_required") is not True or candidate.get("auto_promote") is not False or candidate.get("approved_for_library") is not False:
            raise WarehouseLifecycleError("CANDIDATE_PROMOTION_BOUNDARY_INVALID")
        records = source_repository.list_records(tenant_id=tenant, organization_id=organization).get("records", [])
        source_record_id = next(
            (
                row.get("record_id") for row in records
                if row.get("record_type") == "ANALYTICS_EVENT"
                and row.get("domain_object_id") == candidate.get("source_event_id")
            ),
            None,
        )
        if not source_record_id:
            raise WarehouseLifecycleError("SOURCE_EVENT_NOT_ACTIVE")
        source_result = source_repository.read_record(
            source_record_id, tenant_id=tenant, organization_id=organization
        )
        source_event = source_result.get("domain_object") if isinstance(source_result, Mapping) else None
        if not isinstance(source_event, Mapping):
            raise WarehouseLifecycleError("SOURCE_EVENT_NOT_ACTIVE")
        continuity = {
            "source_event_id": "event_id",
            "source_request_id": "request_id",
            "source_trace_id": "trace_id",
            "query_hash": "query_hash",
        }
        if any(candidate.get(candidate_field) != source_event.get(event_field) for candidate_field, event_field in continuity.items()):
            raise WarehouseLifecycleError("SOURCE_CONTINUITY_INVALID")
        return dict(candidate), dict(source_event)

    def intake_candidate(
        self,
        source_repository: Any,
        source_candidate_record_id: Any,
        *,
        tenant_id: Any,
        organization_id: Any,
        provenance: Any,
        rights_status: Any,
        sensitivity: Any,
        classification: Any,
        idempotency_key: Any,
        created_at: Any | None = None,
        inject_failure: bool = False,
    ) -> dict[str, Any]:
        tenant, organization = _safe_scope(tenant_id, organization_id)
        record_id = _safe_identifier(source_candidate_record_id, "INVALID_CANDIDATE_RECORD_ID")
        idem = _safe_idempotency(idempotency_key)
        if provenance is None:
            raise WarehouseLifecycleError("PROVENANCE_REQUIRED")
        if rights_status is None:
            raise WarehouseLifecycleError("RIGHTS_STATUS_REQUIRED")
        if sensitivity is None:
            raise WarehouseLifecycleError("SENSITIVITY_REQUIRED")
        if classification is None:
            raise WarehouseLifecycleError("CLASSIFICATION_REQUIRED")
        if rights_status not in RIGHTS_STATUSES:
            raise WarehouseLifecycleError("RIGHTS_STATUS_INVALID")
        if sensitivity not in SENSITIVITIES:
            raise WarehouseLifecycleError("SENSITIVITY_INVALID")
        if not isinstance(classification, Mapping) or not classification:
            raise WarehouseLifecycleError("CLASSIFICATION_INVALID")
        if _prohibited_fields({"provenance": provenance, "classification": classification}):
            raise WarehouseLifecycleError("INTAKE_PROHIBITED_FIELDS")
        candidate, _source_event = self._read_active_source(
            source_repository, record_id, tenant, organization
        )
        expected_provenance = {
            "provider_type": "analytics",
            "provider_ref": candidate["candidate_id"],
            "source_event_id": candidate["source_event_id"],
            "source_trace_id": candidate["source_trace_id"],
            "collection_reason": candidate["improvement_trigger"],
        }
        if not isinstance(provenance, Mapping) or dict(provenance) != expected_provenance:
            raise WarehouseLifecycleError("PROVENANCE_INVALID")
        request = {
            "source_candidate_id": candidate["candidate_id"],
            "source_candidate_record_id": record_id,
            "tenant_id": tenant,
            "organization_id": organization,
            "provenance": dict(provenance),
            "rights_status": rights_status,
            "sensitivity": sensitivity,
            "classification": dict(classification),
        }
        _request_json, request_hash = canonical_payload(request)
        existing = self.connection.execute(
            "SELECT * FROM warehouse_intake_items WHERE tenant_id=? AND organization_id=? AND intake_idempotency_key=?",
            (tenant, organization, idem),
        ).fetchone()
        if existing is not None:
            if existing["intake_request_hash"] == request_hash:
                return {"write_performed": False, "reason_code": "IDEMPOTENT_REPLAY", "item": self._verified_item(existing)}
            return {"write_performed": False, "reason_code": "IDEMPOTENCY_CONFLICT", "item": None}
        duplicate = self.connection.execute(
            "SELECT warehouse_item_id FROM warehouse_intake_items WHERE tenant_id=? AND organization_id=? AND source_candidate_id=?",
            (tenant, organization, candidate["candidate_id"]),
        ).fetchone()
        if duplicate is not None:
            return {"write_performed": False, "reason_code": "DOMAIN_OBJECT_CONFLICT", "item": None}
        timestamp = _timestamp(created_at if created_at is not None else _now(self._clock), "INVALID_CREATED_AT")
        digest = hashlib.sha256("\x1f".join((tenant, organization, candidate["candidate_id"])).encode()).hexdigest()
        item = {
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "warehouse_item_id": f"warehouse:item:{digest[:32]}",
            "source_candidate_id": candidate["candidate_id"],
            "source_candidate_record_id": record_id,
            "tenant_id": tenant,
            "organization_id": organization,
            "cohort_id": candidate["tenant_context"]["cohort_id"],
            "source_event_id": candidate["source_event_id"],
            "source_request_id": candidate["source_request_id"],
            "source_trace_id": candidate["source_trace_id"],
            "query_hash": candidate["query_hash"],
            "item_type": ITEM_TYPE,
            "provenance": dict(provenance),
            "rights_status": rights_status,
            "sensitivity": sensitivity,
            "classification": dict(classification),
            "current_status": "captured",
            "previous_status": None,
            "revision": 1,
            "review_required": True,
            "approved_for_library": False,
            "auto_promote": False,
            "idempotency_key": idem,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        if not validate_warehouse_item(item)["valid"]:
            raise WarehouseLifecycleError("WAREHOUSE_ITEM_INVALID")
        serialized, payload_hash = canonical_payload(item)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "INSERT INTO warehouse_intake_items (warehouse_item_id,source_candidate_id,source_candidate_record_id,tenant_id,organization_id,source_event_id,current_status,revision,intake_idempotency_key,intake_request_hash,payload_json,payload_hash,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item["warehouse_item_id"], item["source_candidate_id"], record_id, tenant,
                    organization, item["source_event_id"], item["current_status"], 1, idem,
                    request_hash, serialized, payload_hash, timestamp, timestamp,
                ),
            )
            if inject_failure:
                raise WarehouseLifecycleError("INJECTED_INTAKE_FAILURE")
            self.connection.commit()
        except Exception:
            if self.connection.in_transaction:
                self.connection.rollback()
            raise
        return {"write_performed": True, "reason_code": "WAREHOUSE_ITEM_CAPTURED", "item": item}

    def transition_item(
        self,
        warehouse_item_id: Any,
        *,
        tenant_id: Any,
        organization_id: Any,
        expected_status: Any,
        next_status: Any,
        expected_revision: Any,
        actor_id: Any,
        actor_role: Any,
        reason_code: Any,
        idempotency_key: Any,
        transitioned_at: Any | None = None,
        approval_event_id: Any | None = None,
        source_repository: Any | None = None,
        inject_failure: bool = False,
    ) -> dict[str, Any]:
        tenant, organization = _safe_scope(tenant_id, organization_id)
        item_id = _safe_identifier(warehouse_item_id, "INVALID_WAREHOUSE_ITEM_ID")
        actor = _safe_identifier(actor_id, "ACTOR_ID_REQUIRED")
        reason = _safe_identifier(reason_code, "TRANSITION_REASON_REQUIRED")
        idem = _safe_idempotency(idempotency_key)
        if actor_role not in {"CURATOR", "OWNER"}:
            raise WarehouseLifecycleError("ACTOR_ROLE_NOT_ALLOWED")
        if expected_status not in ITEM_STATUSES or next_status not in ITEM_STATUSES:
            raise WarehouseLifecycleError("STATUS_INVALID")
        if next_status not in TRANSITIONS.get(expected_status, set()):
            raise WarehouseLifecycleError("TRANSITION_FORBIDDEN")
        request = {
            "warehouse_item_id": item_id, "expected_status": expected_status, "next_status": next_status,
            "expected_revision": expected_revision, "actor_id": actor, "actor_role": actor_role,
            "reason_code": reason, "approval_event_id": approval_event_id,
        }
        _request_json, request_hash = canonical_payload(request)
        row = self._load_item_row(item_id, tenant, organization)
        if row is None:
            return {"updated": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "item": None}
        if row["last_transition_idempotency_key"] == idem:
            if row["last_transition_request_hash"] == request_hash:
                return {"updated": False, "reason_code": "IDEMPOTENT_REPLAY", "item": self._verified_item(row)}
            return {"updated": False, "reason_code": "IDEMPOTENCY_CONFLICT", "item": None}
        item = self._verified_item(row)
        if item["current_status"] != expected_status:
            raise WarehouseLifecycleError("CURRENT_STATUS_CONFLICT")
        if not isinstance(expected_revision, int) or item["revision"] != expected_revision:
            raise WarehouseLifecycleError("REVISION_CONFLICT")
        if next_status == "source_verified":
            if source_repository is None:
                raise WarehouseLifecycleError("SOURCE_REPOSITORY_REQUIRED")
            candidate, _event = self._read_active_source(
                source_repository, item["source_candidate_record_id"], tenant, organization
            )
            if candidate["candidate_id"] != item["source_candidate_id"]:
                raise WarehouseLifecycleError("SOURCE_CANDIDATE_LINK_INVALID")
        if next_status == "review_ready":
            if item["review_required"] is not True or item["approved_for_library"] is not False or item["auto_promote"] is not False:
                raise WarehouseLifecycleError("REVIEW_READINESS_INVALID")
        if next_status == "approved_for_warehouse":
            approval = _safe_identifier(approval_event_id, "APPROVAL_EVENT_REQUIRED")
            event = self.connection.execute(
                "SELECT review_event_id FROM warehouse_review_events WHERE warehouse_item_id=? AND tenant_id=? AND organization_id=? AND decision='APPROVE_WAREHOUSE' AND approval_event_id=?",
                (item_id, tenant, organization, approval),
            ).fetchone()
            if event is None:
                raise WarehouseLifecycleError("APPROVAL_EVENT_NOT_FOUND")
            if item["rights_status"] not in APPROVABLE_RIGHTS or item["sensitivity"] in {"private", "secret"} or item["classification"].get("domain") == "ambiguous":
                raise WarehouseLifecycleError("WAREHOUSE_APPROVAL_POLICY_BLOCK")
        timestamp = _timestamp(transitioned_at if transitioned_at is not None else _now(self._clock), "INVALID_TRANSITIONED_AT")
        updated = dict(item)
        updated.update({
            "previous_status": expected_status,
            "current_status": next_status,
            "revision": expected_revision + 1,
            "updated_at": timestamp,
        })
        serialized, payload_hash = canonical_payload(updated)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.connection.execute(
                "UPDATE warehouse_intake_items SET current_status=?,revision=?,last_transition_idempotency_key=?,last_transition_request_hash=?,payload_json=?,payload_hash=?,updated_at=? WHERE warehouse_item_id=? AND tenant_id=? AND organization_id=? AND revision=?",
                (next_status, expected_revision + 1, idem, request_hash, serialized, payload_hash, timestamp, item_id, tenant, organization, expected_revision),
            )
            if cursor.rowcount != 1:
                raise WarehouseLifecycleError("REVISION_CONFLICT")
            if inject_failure:
                raise WarehouseLifecycleError("INJECTED_TRANSITION_FAILURE")
            self.connection.commit()
        except Exception:
            if self.connection.in_transaction:
                self.connection.rollback()
            raise
        return {"updated": True, "reason_code": "TRANSITION_ACCEPTED", "item": updated}

    def review_item(
        self,
        warehouse_item_id: Any,
        *,
        tenant_id: Any,
        organization_id: Any,
        review_event_id: Any,
        reviewer_id: Any,
        reviewer_role: Any,
        decision: Any,
        decision_reason_code: Any,
        expected_revision: Any,
        approval_event_id: Any | None,
        reviewed_at: Any,
        idempotency_key: Any,
        inject_failure: bool = False,
    ) -> dict[str, Any]:
        tenant, organization = _safe_scope(tenant_id, organization_id)
        item_id = _safe_identifier(warehouse_item_id, "INVALID_WAREHOUSE_ITEM_ID")
        event_id = _safe_identifier(review_event_id, "REVIEW_EVENT_ID_REQUIRED")
        reviewer = _safe_identifier(reviewer_id, "REVIEWER_ID_REQUIRED")
        reason = _safe_identifier(decision_reason_code, "DECISION_REASON_REQUIRED")
        idem = _safe_idempotency(idempotency_key)
        if reviewer_role not in REVIEWER_ROLES:
            raise WarehouseLifecycleError("REVIEWER_ROLE_INVALID")
        if decision not in DECISIONS:
            raise WarehouseLifecycleError("REVIEW_DECISION_INVALID")
        if decision not in ROLE_DECISIONS[reviewer_role]:
            raise WarehouseLifecycleError("REVIEWER_ROLE_DECISION_FORBIDDEN")
        if decision == "APPROVE_WAREHOUSE":
            approval = _safe_identifier(approval_event_id, "APPROVAL_EVENT_REQUIRED")
        else:
            if approval_event_id is not None:
                raise WarehouseLifecycleError("APPROVAL_EVENT_FORBIDDEN")
            approval = None
        timestamp = _timestamp(reviewed_at, "INVALID_REVIEWED_AT")
        request = {
            "review_event_id": event_id, "warehouse_item_id": item_id, "reviewer_id": reviewer,
            "reviewer_role": reviewer_role, "decision": decision, "decision_reason_code": reason,
            "expected_revision": expected_revision, "approval_event_id": approval, "reviewed_at": timestamp,
        }
        _request_json, request_hash = canonical_payload(request)
        existing = self.connection.execute(
            "SELECT * FROM warehouse_review_events WHERE tenant_id=? AND organization_id=? AND idempotency_key=?",
            (tenant, organization, idem),
        ).fetchone()
        if existing is not None:
            if existing["request_hash"] == request_hash:
                item = self.read_item(item_id, tenant_id=tenant, organization_id=organization)["item"]
                return {"write_performed": False, "reason_code": "IDEMPOTENT_REPLAY", "review_event": json.loads(existing["payload_json"]), "item": item}
            return {"write_performed": False, "reason_code": "IDEMPOTENCY_CONFLICT", "review_event": None, "item": None}
        row = self._load_item_row(item_id, tenant, organization)
        if row is None:
            return {"write_performed": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "review_event": None, "item": None}
        item = self._verified_item(row)
        if item["current_status"] != "review_ready":
            raise WarehouseLifecycleError("ITEM_NOT_REVIEW_READY")
        if not isinstance(expected_revision, int) or item["revision"] != expected_revision:
            raise WarehouseLifecycleError("REVISION_CONFLICT")
        if decision == "APPROVE_WAREHOUSE":
            next_status = "reviewed"
        elif decision == "REQUEST_RIGHTS_REVIEW":
            next_status = "hold_copyright"
        elif decision == "REQUEST_DOMAIN_REVIEW":
            next_status = "hold_review_needed"
        elif decision == "REQUEST_MORE_EVIDENCE":
            next_status = "hold_sensitive" if item["sensitivity"] in {"private", "secret"} else "hold_review_needed"
        elif decision == "REJECT":
            next_status = "rejected"
        else:
            next_status = "quarantined"
        event = {
            "review_event_id": event_id,
            "warehouse_item_id": item_id,
            "reviewer_id": reviewer,
            "reviewer_role": reviewer_role,
            "decision": decision,
            "decision_reason_code": reason,
            "previous_status": "review_ready",
            "next_status": next_status,
            "expected_revision": expected_revision,
            "new_revision": expected_revision + 1,
            "approval_event_id": approval,
            "reviewed_at": timestamp,
            "idempotency_key": idem,
        }
        if not validate_review_event(event)["valid"]:
            raise WarehouseLifecycleError("REVIEW_EVENT_INVALID")
        updated = dict(item)
        updated.update({
            "previous_status": "review_ready", "current_status": next_status,
            "revision": expected_revision + 1, "updated_at": timestamp,
        })
        event_json, event_hash = canonical_payload(event)
        item_json, item_hash = canonical_payload(updated)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "INSERT INTO warehouse_review_events (review_event_id,warehouse_item_id,tenant_id,organization_id,reviewer_id,reviewer_role,decision,decision_reason_code,previous_status,next_status,expected_revision,new_revision,approval_event_id,reviewed_at,idempotency_key,request_hash,payload_json,payload_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, item_id, tenant, organization, reviewer, reviewer_role, decision, reason,
                 "review_ready", next_status, expected_revision, expected_revision + 1, approval,
                 timestamp, idem, request_hash, event_json, event_hash),
            )
            cursor = self.connection.execute(
                "UPDATE warehouse_intake_items SET current_status=?,revision=?,payload_json=?,payload_hash=?,updated_at=? WHERE warehouse_item_id=? AND tenant_id=? AND organization_id=? AND revision=?",
                (next_status, expected_revision + 1, item_json, item_hash, timestamp, item_id, tenant, organization, expected_revision),
            )
            if cursor.rowcount != 1:
                raise WarehouseLifecycleError("REVISION_CONFLICT")
            if inject_failure:
                raise WarehouseLifecycleError("INJECTED_REVIEW_FAILURE")
            self.connection.commit()
        except Exception:
            if self.connection.in_transaction:
                self.connection.rollback()
            raise
        return {"write_performed": True, "reason_code": "HUMAN_REVIEW_RECORDED", "review_event": event, "item": updated}

    def delete_item(self, warehouse_item_id: Any, *, tenant_id: Any, organization_id: Any) -> dict[str, Any]:
        tenant, organization = _safe_scope(tenant_id, organization_id)
        item_id = _safe_identifier(warehouse_item_id, "INVALID_WAREHOUSE_ITEM_ID")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT warehouse_item_id FROM warehouse_intake_items WHERE warehouse_item_id=? AND tenant_id=? AND organization_id=?",
                (item_id, tenant, organization),
            ).fetchone()
            if row is None:
                self.connection.rollback()
                return {"deleted_count": 0, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE"}
            self.connection.execute(
                "DELETE FROM warehouse_review_events WHERE warehouse_item_id=? AND tenant_id=? AND organization_id=?",
                (item_id, tenant, organization),
            )
            self.connection.execute(
                "DELETE FROM warehouse_intake_items WHERE warehouse_item_id=? AND tenant_id=? AND organization_id=?",
                (item_id, tenant, organization),
            )
            self.connection.commit()
        except Exception:
            if self.connection.in_transaction:
                self.connection.rollback()
            raise
        return {"deleted_count": 1, "reason_code": "TEST_RECORD_DELETED"}


__all__ = [
    "APPROVABLE_RIGHTS", "CONTRACT_VERSION", "DECISIONS", "FINAL_STATUSES", "ITEM_STATUSES",
    "ITEM_TYPE", "LocalNonprodWarehouseReviewLifecycle", "REPOSITORY_SCHEMA_VERSION",
    "REVIEWER_ROLES", "RIGHTS_STATUSES", "SCHEMA_VERSION", "SENSITIVITIES", "TRANSITIONS",
    "WarehouseLifecycleError", "build_warehouse_idempotency_key", "canonical_payload",
    "validate_review_event", "validate_warehouse_item",
]
