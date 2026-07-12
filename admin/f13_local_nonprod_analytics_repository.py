from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from admin.f13_analytics_event_contract import validate_analytics_event
from admin.f13_analytics_improvement_candidate_contract import validate_analytics_improvement_candidate


REPOSITORY_SCHEMA_VERSION = 1
RECORD_TYPE_ANALYTICS = "ANALYTICS_EVENT"
RECORD_TYPE_CANDIDATE = "IMPROVEMENT_CANDIDATE"
RECORD_TYPES = frozenset({RECORD_TYPE_ANALYTICS, RECORD_TYPE_CANDIDATE})
ENVIRONMENT_LOCAL_NONPRODUCTION = "local_nonproduction"

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9:._-]{1,220}$")
_RETENTION_RE = re.compile(r"^[A-Za-z0-9:._-]{1,80}$")
_IDEMPOTENCY_RE = re.compile(r"^idem:(?:analytics|candidate):[0-9a-f]{64}$")
_SQLITE_HEADER = b"SQLite format 3\x00"
_PRODUCTION_PATH_MARKERS = (
    "\\장기기억\\",
    "\\production\\",
    "\\prod\\",
    "\\library\\",
    "\\warehouse\\",
)

REQUIRED_ENVELOPE_FIELDS = (
    "repository_schema_version",
    "record_id",
    "record_type",
    "tenant_id",
    "organization_id",
    "cohort_id",
    "subject_hash",
    "source_event_id",
    "domain_object_id",
    "idempotency_key",
    "payload_hash",
    "payload_json",
    "retention_policy_id",
    "retention_until",
    "created_at",
    "updated_at",
    "deleted_at",
    "deletion_reason",
    "revision",
)

_SCHEMA_SQL = """
CREATE TABLE local_repository_metadata (
    metadata_key TEXT PRIMARY KEY,
    metadata_value TEXT NOT NULL
);
CREATE TABLE local_repository_records (
    repository_schema_version INTEGER NOT NULL CHECK (repository_schema_version = 1),
    record_id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL CHECK (record_type IN ('ANALYTICS_EVENT', 'IMPROVEMENT_CANDIDATE')),
    tenant_id TEXT NOT NULL,
    organization_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL,
    subject_hash TEXT,
    source_event_id TEXT,
    domain_object_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    retention_policy_id TEXT NOT NULL,
    retention_until TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    deletion_reason TEXT,
    revision INTEGER NOT NULL CHECK (revision > 0),
    CHECK (
        (record_type = 'ANALYTICS_EVENT' AND subject_hash IS NOT NULL AND source_event_id IS NULL)
        OR
        (record_type = 'IMPROVEMENT_CANDIDATE' AND subject_hash IS NULL AND source_event_id IS NOT NULL)
    ),
    CHECK (
        (deleted_at IS NULL AND deletion_reason IS NULL)
        OR
        (deleted_at IS NOT NULL AND deletion_reason IS NOT NULL)
    ),
    UNIQUE (tenant_id, organization_id, idempotency_key),
    UNIQUE (tenant_id, organization_id, record_type, domain_object_id)
);
CREATE INDEX idx_local_repository_scope
    ON local_repository_records (tenant_id, organization_id, record_type);
CREATE INDEX idx_local_repository_subject
    ON local_repository_records (tenant_id, organization_id, subject_hash);
CREATE INDEX idx_local_repository_source
    ON local_repository_records (tenant_id, organization_id, source_event_id);
CREATE INDEX idx_local_repository_retention
    ON local_repository_records (tenant_id, organization_id, retention_until);
"""
_SCHEMA_FINGERPRINT = hashlib.sha256(_SCHEMA_SQL.encode("utf-8")).hexdigest()


class LocalNonprodRepositoryError(RuntimeError):
    """Fail-closed local repository boundary error without payload echo."""


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise LocalNonprodRepositoryError("TIMESTAMP_TIMEZONE_REQUIRED")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now(clock: Callable[[], datetime] | None) -> datetime:
    value = clock() if clock else datetime.now(timezone.utc)
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise LocalNonprodRepositoryError("CLOCK_MUST_BE_TIMEZONE_AWARE")
    return value.astimezone(timezone.utc)


def _canonical_payload(payload: Mapping[str, Any]) -> tuple[str, str]:
    serialized = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return serialized, f"sha256:{digest}"


def _safe_scope(tenant_id: Any, organization_id: Any) -> tuple[str, str]:
    if not isinstance(tenant_id, str) or not tenant_id:
        raise LocalNonprodRepositoryError("TENANT_SCOPE_REQUIRED")
    if not isinstance(organization_id, str) or not organization_id:
        raise LocalNonprodRepositoryError("ORGANIZATION_SCOPE_REQUIRED")
    if tenant_id == "*" or organization_id == "*":
        raise LocalNonprodRepositoryError("WILDCARD_SCOPE_FORBIDDEN")
    if _IDENTIFIER_RE.fullmatch(tenant_id) is None or _IDENTIFIER_RE.fullmatch(organization_id) is None:
        raise LocalNonprodRepositoryError("INVALID_SCOPE")
    return tenant_id, organization_id


def _safe_token(value: Any, reason: str, *, pattern: re.Pattern[str] = _IDENTIFIER_RE) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise LocalNonprodRepositoryError(reason)
    return value


def _safe_retention(retention_policy_id: Any, retention_until: Any, created_at: datetime) -> tuple[str, str]:
    policy = _safe_token(retention_policy_id, "RETENTION_POLICY_REQUIRED", pattern=_RETENTION_RE)
    parsed = _parse_timestamp(retention_until)
    if parsed is None:
        raise LocalNonprodRepositoryError("RETENTION_UNTIL_INVALID")
    parsed = parsed.astimezone(timezone.utc)
    if parsed <= created_at:
        raise LocalNonprodRepositoryError("RETENTION_EXPIRED_AT_WRITE")
    return policy, _utc_text(parsed)


def _is_unc(path_text: str) -> bool:
    return path_text.startswith("\\\\") or path_text.startswith("//")


def validate_local_repository_path(
    database_path: str | Path,
    *,
    approved_local_root: str | Path,
    environment: str,
    explicit_allow_local_durable: bool,
) -> tuple[Path, Path]:
    if environment != ENVIRONMENT_LOCAL_NONPRODUCTION:
        raise LocalNonprodRepositoryError("LOCAL_NONPRODUCTION_ENVIRONMENT_REQUIRED")
    if explicit_allow_local_durable is not True:
        raise LocalNonprodRepositoryError("EXPLICIT_LOCAL_DURABLE_ALLOW_REQUIRED")
    raw_db = str(database_path or "")
    raw_root = str(approved_local_root or "")
    if not raw_db or not raw_root:
        raise LocalNonprodRepositoryError("EXPLICIT_DATABASE_AND_ROOT_REQUIRED")
    if _is_unc(raw_db) or _is_unc(raw_root):
        raise LocalNonprodRepositoryError("NETWORK_PATH_FORBIDDEN")
    db = Path(raw_db)
    root = Path(raw_root)
    if not db.is_absolute() or not root.is_absolute():
        raise LocalNonprodRepositoryError("ABSOLUTE_PATH_REQUIRED")
    if ".." in db.parts or ".." in root.parts:
        raise LocalNonprodRepositoryError("PARENT_TRAVERSAL_FORBIDDEN")
    resolved_root = root.resolve(strict=False)
    resolved_db = db.resolve(strict=False)
    cwd = Path.cwd().resolve(strict=False)
    if resolved_root == cwd or resolved_db == cwd or resolved_db == resolved_root:
        raise LocalNonprodRepositoryError("REPOSITORY_OR_CWD_PATH_FORBIDDEN")
    try:
        resolved_db.relative_to(cwd)
    except ValueError:
        pass
    else:
        raise LocalNonprodRepositoryError("REPOSITORY_RELATIVE_STORAGE_FORBIDDEN")
    normalized = str(resolved_db).replace("/", "\\").lower()
    if any(marker in normalized for marker in _PRODUCTION_PATH_MARKERS):
        raise LocalNonprodRepositoryError("PRODUCTION_LIKE_PATH_FORBIDDEN")
    try:
        resolved_db.relative_to(resolved_root)
    except ValueError as exc:
        raise LocalNonprodRepositoryError("DATABASE_PATH_OUTSIDE_APPROVED_ROOT") from exc
    if resolved_db.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
        raise LocalNonprodRepositoryError("SQLITE_FILENAME_REQUIRED")
    return resolved_db, resolved_root


def validate_repository_record_envelope(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"valid": False, "reason_code": "ENVELOPE_NOT_MAPPING", "invalid_fields": []}
    fields = set(payload)
    required = set(REQUIRED_ENVELOPE_FIELDS)
    if fields != required:
        return {
            "valid": False,
            "reason_code": "ENVELOPE_FIELD_MISMATCH",
            "invalid_fields": sorted(str(item) for item in fields ^ required),
        }
    invalid: list[str] = []
    record_type = payload.get("record_type")
    if payload.get("repository_schema_version") != REPOSITORY_SCHEMA_VERSION:
        invalid.append("repository_schema_version")
    for field in ("record_id", "tenant_id", "organization_id", "cohort_id", "domain_object_id"):
        if not isinstance(payload.get(field), str) or _IDENTIFIER_RE.fullmatch(payload[field]) is None:
            invalid.append(field)
    if record_type not in RECORD_TYPES:
        invalid.append("record_type")
    subject_hash = payload.get("subject_hash")
    source_event_id = payload.get("source_event_id")
    if record_type == RECORD_TYPE_ANALYTICS:
        if not isinstance(subject_hash, str) or _HASH_RE.fullmatch(subject_hash) is None or source_event_id is not None:
            invalid.extend(["subject_hash", "source_event_id"])
    elif record_type == RECORD_TYPE_CANDIDATE:
        if subject_hash is not None or not isinstance(source_event_id, str) or _IDENTIFIER_RE.fullmatch(source_event_id) is None:
            invalid.extend(["subject_hash", "source_event_id"])
    if not isinstance(payload.get("idempotency_key"), str) or _IDEMPOTENCY_RE.fullmatch(payload["idempotency_key"]) is None:
        invalid.append("idempotency_key")
    if not isinstance(payload.get("payload_hash"), str) or _HASH_RE.fullmatch(payload["payload_hash"]) is None:
        invalid.append("payload_hash")
    try:
        decoded = json.loads(payload.get("payload_json", ""))
    except (TypeError, json.JSONDecodeError):
        decoded = None
    if not isinstance(decoded, Mapping):
        invalid.append("payload_json")
    else:
        serialized, digest = _canonical_payload(decoded)
        if serialized != payload.get("payload_json") or digest != payload.get("payload_hash"):
            invalid.extend(["payload_json", "payload_hash"])
    if not isinstance(payload.get("retention_policy_id"), str) or _RETENTION_RE.fullmatch(payload["retention_policy_id"]) is None:
        invalid.append("retention_policy_id")
    for field in ("retention_until", "created_at", "updated_at"):
        if _parse_timestamp(payload.get(field)) is None:
            invalid.append(field)
    if payload.get("deleted_at") is not None or payload.get("deletion_reason") is not None:
        invalid.extend(["deleted_at", "deletion_reason"])
    if payload.get("revision") != 1:
        invalid.append("revision")
    return {"valid": not invalid, "reason_code": "VALID" if not invalid else "ENVELOPE_INVALID", "invalid_fields": sorted(set(invalid))}


class LocalNonprodAnalyticsRepository:
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
                    raise LocalNonprodRepositoryError("CORRUPT_OR_NON_SQLITE_STORAGE")
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
        except (sqlite3.DatabaseError, OSError, LocalNonprodRepositoryError) as exc:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if not existed and self.database_path.exists():
                self.database_path.unlink()
            if isinstance(exc, LocalNonprodRepositoryError):
                raise
            raise LocalNonprodRepositoryError("REPOSITORY_OPEN_FAILED") from exc

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise LocalNonprodRepositoryError("REPOSITORY_CLOSED")
        return self._connection

    def _initialize_schema(self) -> None:
        connection = self.connection
        try:
            connection.executescript(
                "BEGIN IMMEDIATE;\n"
                + _SCHEMA_SQL
                + "\nINSERT INTO local_repository_metadata (metadata_key, metadata_value) "
                + "VALUES ('repository_schema_version', '1');\n"
                + "INSERT INTO local_repository_metadata (metadata_key, metadata_value) "
                + f"VALUES ('repository_schema_fingerprint', '{_SCHEMA_FINGERPRINT}');\nCOMMIT;"
            )
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise

    def _verify_existing_schema(self) -> None:
        connection = self.connection
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise LocalNonprodRepositoryError("REPOSITORY_INTEGRITY_CHECK_FAILED")
        try:
            row = connection.execute(
                "SELECT metadata_value FROM local_repository_metadata WHERE metadata_key = ?",
                ("repository_schema_version",),
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise LocalNonprodRepositoryError("REPOSITORY_SCHEMA_METADATA_INVALID") from exc
        if row is None or row[0] != str(REPOSITORY_SCHEMA_VERSION):
            raise LocalNonprodRepositoryError("REPOSITORY_SCHEMA_VERSION_UNSUPPORTED")
        fingerprint = connection.execute(
            "SELECT metadata_value FROM local_repository_metadata WHERE metadata_key = ?",
            ("repository_schema_fingerprint",),
        ).fetchone()
        if fingerprint is None or fingerprint[0] != _SCHEMA_FINGERPRINT:
            raise LocalNonprodRepositoryError("REPOSITORY_SCHEMA_FINGERPRINT_INVALID")
        required_tables = {"local_repository_metadata", "local_repository_records"}
        actual = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        if not required_tables <= actual:
            raise LocalNonprodRepositoryError("REPOSITORY_SCHEMA_INCOMPLETE")
        columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(local_repository_records)"))
        if columns != REQUIRED_ENVELOPE_FIELDS:
            raise LocalNonprodRepositoryError("REPOSITORY_RECORD_SCHEMA_INVALID")

    def integrity_check(self) -> dict[str, Any]:
        row = self.connection.execute("PRAGMA integrity_check").fetchone()
        passed = row is not None and row[0] == "ok"
        return {"integrity_ok": passed, "reason_code": "INTEGRITY_OK" if passed else "INTEGRITY_FAILED"}

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "LocalNonprodAnalyticsRepository":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _record_id(self, record_type: str, tenant_id: str, organization_id: str, object_id: str) -> str:
        digest = hashlib.sha256("\x1f".join((record_type, tenant_id, organization_id, object_id)).encode()).hexdigest()
        return f"record:{'event' if record_type == RECORD_TYPE_ANALYTICS else 'candidate'}:{digest[:32]}"

    def _analytics_idempotency(self, tenant_id: str, organization_id: str, event_id: str) -> str:
        digest = hashlib.sha256("\x1f".join((tenant_id, organization_id, event_id)).encode()).hexdigest()
        return f"idem:analytics:{digest}"

    def _envelope(
        self,
        *,
        record_type: str,
        payload: Mapping[str, Any],
        subject_hash: str | None,
        source_event_id: str | None,
        domain_object_id: str,
        idempotency_key: str,
        retention_policy_id: str,
        retention_until: str,
        created_at: datetime,
    ) -> dict[str, Any]:
        serialized, payload_hash = _canonical_payload(payload)
        return {
            "repository_schema_version": REPOSITORY_SCHEMA_VERSION,
            "record_id": self._record_id(record_type, str(payload["tenant_id"] if record_type == RECORD_TYPE_ANALYTICS else payload["tenant_context"]["tenant_id"]), str(payload["organization_id"] if record_type == RECORD_TYPE_ANALYTICS else payload["tenant_context"]["organization_id"]), domain_object_id),
            "record_type": record_type,
            "tenant_id": str(payload["tenant_id"] if record_type == RECORD_TYPE_ANALYTICS else payload["tenant_context"]["tenant_id"]),
            "organization_id": str(payload["organization_id"] if record_type == RECORD_TYPE_ANALYTICS else payload["tenant_context"]["organization_id"]),
            "cohort_id": str(payload["cohort_id"] if record_type == RECORD_TYPE_ANALYTICS else payload["tenant_context"]["cohort_id"]),
            "subject_hash": subject_hash,
            "source_event_id": source_event_id,
            "domain_object_id": domain_object_id,
            "idempotency_key": idempotency_key,
            "payload_hash": payload_hash,
            "payload_json": serialized,
            "retention_policy_id": retention_policy_id,
            "retention_until": retention_until,
            "created_at": _utc_text(created_at),
            "updated_at": _utc_text(created_at),
            "deleted_at": None,
            "deletion_reason": None,
            "revision": 1,
        }

    def _insert(
        self,
        envelope: Mapping[str, Any],
        *,
        inject_failure: bool,
        transaction_started: bool = False,
    ) -> dict[str, Any]:
        validation = validate_repository_record_envelope(envelope)
        if not validation["valid"]:
            return {"write_performed": False, "record_id": None, "reason_code": "ENVELOPE_INVALID"}
        connection = self.connection
        if not transaction_started:
            connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT record_id, payload_hash FROM local_repository_records WHERE tenant_id = ? AND organization_id = ? AND idempotency_key = ?",
                (envelope["tenant_id"], envelope["organization_id"], envelope["idempotency_key"]),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                if existing["payload_hash"] == envelope["payload_hash"]:
                    return {"write_performed": False, "record_id": existing["record_id"], "reason_code": "IDEMPOTENT_REPLAY"}
                return {"write_performed": False, "record_id": None, "reason_code": "IDEMPOTENCY_CONFLICT"}
            existing = connection.execute(
                "SELECT record_id, payload_hash FROM local_repository_records WHERE tenant_id = ? AND organization_id = ? AND record_type = ? AND domain_object_id = ?",
                (envelope["tenant_id"], envelope["organization_id"], envelope["record_type"], envelope["domain_object_id"]),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                if existing["payload_hash"] == envelope["payload_hash"]:
                    return {"write_performed": False, "record_id": existing["record_id"], "reason_code": "IDEMPOTENT_REPLAY"}
                return {"write_performed": False, "record_id": None, "reason_code": "DOMAIN_OBJECT_CONFLICT"}
            columns = ", ".join(REQUIRED_ENVELOPE_FIELDS)
            placeholders = ", ".join(f":{field}" for field in REQUIRED_ENVELOPE_FIELDS)
            connection.execute(f"INSERT INTO local_repository_records ({columns}) VALUES ({placeholders})", dict(envelope))
            if inject_failure:
                raise LocalNonprodRepositoryError("INJECTED_WRITE_FAILURE")
            connection.commit()
            return {"write_performed": True, "record_id": envelope["record_id"], "reason_code": "RECORD_STORED"}
        except Exception:
            connection.rollback()
            raise

    def store_analytics_event(
        self,
        gated_mapping_result: Any,
        *,
        retention_policy_id: str,
        retention_until: str,
        inject_failure: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(gated_mapping_result, Mapping) or gated_mapping_result.get("policy_status") != "ALLOW" or gated_mapping_result.get("reason_code") != "CONSENT_ALLOWED" or gated_mapping_result.get("analytics_event_present") is not True:
            return {"write_performed": False, "record_id": None, "reason_code": "CONSENT_NOT_ALLOWED"}
        event = gated_mapping_result.get("analytics_event")
        validation = validate_analytics_event(event)
        if not validation.get("valid") or not isinstance(event, Mapping):
            return {"write_performed": False, "record_id": None, "reason_code": "ANALYTICS_EVENT_INVALID"}
        created_at = _now(self._clock)
        policy, until = _safe_retention(retention_policy_id, retention_until, created_at)
        envelope = self._envelope(
            record_type=RECORD_TYPE_ANALYTICS,
            payload=event,
            subject_hash=str(event["user_id_hash"]),
            source_event_id=None,
            domain_object_id=str(event["event_id"]),
            idempotency_key=self._analytics_idempotency(str(event["tenant_id"]), str(event["organization_id"]), str(event["event_id"])),
            retention_policy_id=policy,
            retention_until=until,
            created_at=created_at,
        )
        return self._insert(envelope, inject_failure=inject_failure)

    def store_improvement_candidate(
        self,
        candidate_result: Any,
        *,
        retention_policy_id: str,
        retention_until: str,
        inject_failure: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(candidate_result, Mapping) or candidate_result.get("candidate_present") is not True:
            return {"write_performed": False, "record_id": None, "reason_code": "CANDIDATE_NOT_PRESENT"}
        candidate = candidate_result.get("candidate")
        validation = validate_analytics_improvement_candidate(candidate)
        if not validation.get("valid") or not isinstance(candidate, Mapping):
            return {"write_performed": False, "record_id": None, "reason_code": "CANDIDATE_INVALID"}
        context = candidate["tenant_context"]
        tenant_id, organization_id = _safe_scope(context["tenant_id"], context["organization_id"])
        created_at = _now(self._clock)
        policy, until = _safe_retention(retention_policy_id, retention_until, created_at)
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            source = connection.execute(
                "SELECT payload_json, retention_until FROM local_repository_records WHERE tenant_id = ? AND organization_id = ? AND record_type = ? AND domain_object_id = ? AND deleted_at IS NULL",
                (tenant_id, organization_id, RECORD_TYPE_ANALYTICS, candidate["source_event_id"]),
            ).fetchone()
            if source is None:
                other_tenant = connection.execute(
                    "SELECT tenant_id, organization_id FROM local_repository_records WHERE record_type = ? AND domain_object_id = ? LIMIT 1",
                    (RECORD_TYPE_ANALYTICS, candidate["source_event_id"]),
                ).fetchone()
                connection.rollback()
                if other_tenant is not None and other_tenant["tenant_id"] != tenant_id:
                    return {"write_performed": False, "record_id": None, "reason_code": "CROSS_TENANT_LINK_DENIED"}
                if other_tenant is not None and other_tenant["organization_id"] != organization_id:
                    return {"write_performed": False, "record_id": None, "reason_code": "CROSS_ORGANIZATION_LINK_DENIED"}
                return {"write_performed": False, "record_id": None, "reason_code": "SOURCE_EVENT_NOT_ACTIVE"}
            source_until = _parse_timestamp(source["retention_until"])
            if source_until is None or source_until <= created_at:
                connection.rollback()
                return {"write_performed": False, "record_id": None, "reason_code": "SOURCE_EVENT_EXPIRED"}
            source_event = json.loads(source["payload_json"])
            if source_event.get("event_id") != candidate.get("source_event_id") or source_event.get("request_id") != candidate.get("source_request_id") or source_event.get("trace_id") != candidate.get("source_trace_id"):
                connection.rollback()
                return {"write_performed": False, "record_id": None, "reason_code": "SOURCE_EVENT_CONTINUITY_INVALID"}
            envelope = self._envelope(
                record_type=RECORD_TYPE_CANDIDATE,
                payload=candidate,
                subject_hash=None,
                source_event_id=str(candidate["source_event_id"]),
                domain_object_id=str(candidate["candidate_id"]),
                idempotency_key=str(candidate["idempotency_key"]).replace("idem:analytics:", "idem:candidate:"),
                retention_policy_id=policy,
                retention_until=until,
                created_at=created_at,
            )
            return self._insert(envelope, inject_failure=inject_failure, transaction_started=True)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise

    def read_record(self, record_id: str, *, tenant_id: str, organization_id: str) -> dict[str, Any]:
        tenant_id, organization_id = _safe_scope(tenant_id, organization_id)
        _safe_token(record_id, "INVALID_RECORD_ID")
        row = self.connection.execute(
            "SELECT * FROM local_repository_records WHERE record_id = ? AND tenant_id = ? AND organization_id = ? AND deleted_at IS NULL",
            (record_id, tenant_id, organization_id),
        ).fetchone()
        if row is None:
            return {"found": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "domain_object": None}
        if _parse_timestamp(row["retention_until"]) <= _now(self._clock):
            return {"found": False, "reason_code": "RECORD_EXPIRED", "domain_object": None}
        payload = json.loads(row["payload_json"])
        validator = validate_analytics_event if row["record_type"] == RECORD_TYPE_ANALYTICS else validate_analytics_improvement_candidate
        if not validator(payload).get("valid"):
            raise LocalNonprodRepositoryError("STORED_PAYLOAD_INVALID")
        serialized, digest = _canonical_payload(payload)
        if serialized != row["payload_json"] or digest != row["payload_hash"]:
            raise LocalNonprodRepositoryError("STORED_PAYLOAD_INTEGRITY_FAILED")
        return {"found": True, "reason_code": "RECORD_FOUND", "record_type": row["record_type"], "domain_object": payload}

    def list_records(self, *, tenant_id: str, organization_id: str) -> dict[str, Any]:
        tenant_id, organization_id = _safe_scope(tenant_id, organization_id)
        rows = self.connection.execute(
            "SELECT record_id, record_type, domain_object_id, retention_until FROM local_repository_records WHERE tenant_id = ? AND organization_id = ? AND deleted_at IS NULL ORDER BY record_id",
            (tenant_id, organization_id),
        ).fetchall()
        now = _now(self._clock)
        records = [
            {"record_id": row["record_id"], "record_type": row["record_type"], "domain_object_id": row["domain_object_id"]}
            for row in rows
            if _parse_timestamp(row["retention_until"]) > now
        ]
        return {"reason_code": "RECORD_LIST", "count": len(records), "records": records}

    def count_records(self, *, tenant_id: str, organization_id: str) -> int:
        tenant_id, organization_id = _safe_scope(tenant_id, organization_id)
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM local_repository_records WHERE tenant_id = ? AND organization_id = ?",
            (tenant_id, organization_id),
        ).fetchone()[0])

    def _delete_event_ids(self, event_ids: list[str], *, tenant_id: str, organization_id: str, inject_failure: bool) -> tuple[int, int]:
        if not event_ids:
            return 0, 0
        placeholders = ",".join("?" for _ in event_ids)
        params = [tenant_id, organization_id, *event_ids]
        candidate_cursor = self.connection.execute(
            f"DELETE FROM local_repository_records WHERE tenant_id = ? AND organization_id = ? AND record_type = '{RECORD_TYPE_CANDIDATE}' AND source_event_id IN ({placeholders})",
            params,
        )
        if inject_failure:
            raise LocalNonprodRepositoryError("INJECTED_CASCADE_FAILURE")
        event_cursor = self.connection.execute(
            f"DELETE FROM local_repository_records WHERE tenant_id = ? AND organization_id = ? AND record_type = '{RECORD_TYPE_ANALYTICS}' AND domain_object_id IN ({placeholders})",
            params,
        )
        return int(event_cursor.rowcount), int(candidate_cursor.rowcount)

    def delete_record(self, record_id: str, *, tenant_id: str, organization_id: str, inject_failure: bool = False) -> dict[str, Any]:
        tenant_id, organization_id = _safe_scope(tenant_id, organization_id)
        _safe_token(record_id, "INVALID_RECORD_ID")
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT record_type, domain_object_id FROM local_repository_records WHERE record_id = ? AND tenant_id = ? AND organization_id = ?",
                (record_id, tenant_id, organization_id),
            ).fetchone()
            if row is None:
                connection.rollback()
                return self._deletion_result(0, 0, 1, tenant_id, organization_id, "RECORD_NOT_FOUND_OR_NOT_VISIBLE")
            if row["record_type"] == RECORD_TYPE_ANALYTICS:
                events, candidates = self._delete_event_ids([row["domain_object_id"]], tenant_id=tenant_id, organization_id=organization_id, inject_failure=inject_failure)
            else:
                cursor = connection.execute("DELETE FROM local_repository_records WHERE record_id = ? AND tenant_id = ? AND organization_id = ?", (record_id, tenant_id, organization_id))
                if inject_failure:
                    raise LocalNonprodRepositoryError("INJECTED_DELETE_FAILURE")
                events, candidates = 0, int(cursor.rowcount)
            connection.commit()
            return self._deletion_result(events, candidates, 0, tenant_id, organization_id, "RECORD_DELETED")
        except Exception:
            connection.rollback()
            raise

    def delete_by_subject_hash(self, *, tenant_id: str, organization_id: str, subject_hash: str, inject_failure: bool = False) -> dict[str, Any]:
        tenant_id, organization_id = _safe_scope(tenant_id, organization_id)
        _safe_token(subject_hash, "INVALID_SUBJECT_HASH", pattern=_HASH_RE)
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            event_ids = [row[0] for row in connection.execute(
                "SELECT domain_object_id FROM local_repository_records WHERE tenant_id = ? AND organization_id = ? AND record_type = ? AND subject_hash = ?",
                (tenant_id, organization_id, RECORD_TYPE_ANALYTICS, subject_hash),
            ).fetchall()]
            events, candidates = self._delete_event_ids(event_ids, tenant_id=tenant_id, organization_id=organization_id, inject_failure=inject_failure)
            connection.commit()
            return self._deletion_result(events, candidates, 1 if not event_ids else 0, tenant_id, organization_id, "SUBJECT_RECORDS_DELETED" if event_ids else "RECORD_NOT_FOUND_OR_NOT_VISIBLE")
        except Exception:
            connection.rollback()
            raise

    def purge_expired(self, *, tenant_id: str, organization_id: str, now: str | None = None, inject_failure: bool = False) -> dict[str, Any]:
        tenant_id, organization_id = _safe_scope(tenant_id, organization_id)
        threshold = _parse_timestamp(now) if now is not None else _now(self._clock)
        if threshold is None:
            raise LocalNonprodRepositoryError("PURGE_TIME_INVALID")
        threshold_text = _utc_text(threshold)
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            event_ids = [row[0] for row in connection.execute(
                "SELECT domain_object_id FROM local_repository_records WHERE tenant_id = ? AND organization_id = ? AND record_type = ? AND retention_until <= ?",
                (tenant_id, organization_id, RECORD_TYPE_ANALYTICS, threshold_text),
            ).fetchall()]
            events, linked = self._delete_event_ids(event_ids, tenant_id=tenant_id, organization_id=organization_id, inject_failure=False)
            candidate_cursor = connection.execute(
                "DELETE FROM local_repository_records WHERE tenant_id = ? AND organization_id = ? AND record_type = ? AND retention_until <= ?",
                (tenant_id, organization_id, RECORD_TYPE_CANDIDATE, threshold_text),
            )
            candidates = linked + int(candidate_cursor.rowcount)
            if inject_failure:
                raise LocalNonprodRepositoryError("INJECTED_PURGE_FAILURE")
            connection.commit()
            return self._deletion_result(events, candidates, 0, tenant_id, organization_id, "RETENTION_PURGE_COMPLETE")
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _deletion_result(events: int, candidates: int, absent: int, tenant_id: str, organization_id: str, reason: str) -> dict[str, Any]:
        return {
            "deleted_event_count": events,
            "deleted_candidate_count": candidates,
            "already_absent_count": absent,
            "scope": {"tenant_id": tenant_id, "organization_id": organization_id},
            "reason_code": reason,
        }


__all__ = [
    "ENVIRONMENT_LOCAL_NONPRODUCTION",
    "LocalNonprodAnalyticsRepository",
    "LocalNonprodRepositoryError",
    "RECORD_TYPE_ANALYTICS",
    "RECORD_TYPE_CANDIDATE",
    "RECORD_TYPES",
    "REPOSITORY_SCHEMA_VERSION",
    "REQUIRED_ENVELOPE_FIELDS",
    "validate_local_repository_path",
    "validate_repository_record_envelope",
]
