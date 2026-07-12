"""Guarded local-nonproduction Library promotion and rollback transaction.

This module deliberately has no production adapter.  It persists pointer-only
Reference metadata beneath a caller-approved local root and requires separate,
caller-supplied approval and rollback authorization records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from admin.f13_analytics_event_contract import validate_analytics_event
from admin.f13_analytics_improvement_candidate_contract import validate_analytics_improvement_candidate
from admin.f13_local_nonprod_promotion_dry_run import validate_promotion_plan
from admin.f13_local_nonprod_warehouse_review_lifecycle import validate_review_event


SCHEMA_VERSION = 1
CONTRACT_VERSION = "library.promotion.local_nonproduction.pointer_only.v1"
ENVIRONMENT_LOCAL_NONPRODUCTION = "local_nonproduction"
STATUS_ACTIVE = "ACTIVE"
STATUS_ROLLED_BACK = "ROLLED_BACK"
TRACE_PROMOTED = "PROMOTED_LOCAL_NONPRODUCTION"
TRACE_ROLLED_BACK = "rolled_back"
RAW_TEXT_POLICY_POINTER_ONLY = "POINTER_ONLY"
DOC_KIND_REFERENCE = "REFERENCE"
CANONICAL_LANG = "EN"
SOURCE_LANG = "EN"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._@-]{0,199}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROHIBITED_FIELDS = {
    "raw_query", "query", "raw_body", "body", "prompt", "raw_prompt", "raw_text",
    "answer", "answer_text", "safe_short_answer", "hold_source", "evidence", "evidence_text",
    "standard_text", "paid_standard_text", "user_id", "user_id_hash", "personal_name", "email",
    "phone", "address", "secret", "api_key", "access_token", "refresh_token", "password",
    "credential", "cookie", "authorization", "internal_path", "local_path", "file_path", "db_path",
    "dsn", "connection_string", "production_path", "library_write_target",
}
_APPROVAL_FIELDS = {
    "approval_record_id", "target_object_type", "target_object_id", "approval_type", "approver_id",
    "approver_role", "approved_at", "approval_scope", "approval_comment_code", "evidence_id", "proofpack_id",
}
_ROLLBACK_FIELDS = {
    "rollback_authorization_id", "promotion_trace_id", "library_record_id", "actor_id", "actor_role",
    "reason_code", "approved_at", "proofpack_id",
}
_AUTOMATED_ROLES = {"SYSTEM", "LLM", "AUTOMATION_AGENT", "AGENT"}
_ROLLBACK_ROLES = {"OWNER", "ADMIN"}
_ALLOWED_RIGHTS = {"owned", "licensed", "permission_granted", "public_reference", "internal_only"}
_ALLOWED_SENSITIVITY = {"public", "internal", "restricted"}


class LocalLibraryPromotionError(RuntimeError):
    """Controlled fail-closed error containing a reason code only."""


def canonical_payload(payload: Mapping[str, Any]) -> tuple[str, str]:
    text = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return text, "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _identifier(value: Any, reason: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise LocalLibraryPromotionError(reason)
    return value


def _scope(tenant_id: Any, organization_id: Any) -> tuple[str, str]:
    return _identifier(tenant_id, "TENANT_SCOPE_REQUIRED"), _identifier(
        organization_id, "ORGANIZATION_SCOPE_REQUIRED"
    )


def _timestamp(value: Any, reason: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise LocalLibraryPromotionError(reason) from exc
    else:
        raise LocalLibraryPromotionError(reason)
    if parsed.tzinfo is None:
        raise LocalLibraryPromotionError(reason)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now(clock: Callable[[], datetime] | None) -> str:
    return _timestamp(clock() if clock else datetime.now(timezone.utc), "TIMEZONE_AWARE_CLOCK_REQUIRED")


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


def validate_local_library_root(
    library_root: str | Path,
    *,
    approved_local_root: str | Path,
    environment: str,
    explicit_allow_actual_local_promotion: bool,
) -> Path:
    if environment != ENVIRONMENT_LOCAL_NONPRODUCTION:
        raise LocalLibraryPromotionError("LOCAL_NONPRODUCTION_ENVIRONMENT_REQUIRED")
    if explicit_allow_actual_local_promotion is not True:
        raise LocalLibraryPromotionError("EXPLICIT_LOCAL_PROMOTION_ALLOW_REQUIRED")
    root_text, approved_text = str(library_root), str(approved_local_root)
    if not root_text or not approved_text or root_text.startswith(("\\\\", "//")):
        raise LocalLibraryPromotionError("LOCAL_LIBRARY_ROOT_INVALID")
    root_input, approved_input = Path(root_text), Path(approved_text)
    if not root_input.is_absolute() or not approved_input.is_absolute():
        raise LocalLibraryPromotionError("ABSOLUTE_LOCAL_ROOT_REQUIRED")
    if ".." in root_input.parts or ".." in approved_input.parts:
        raise LocalLibraryPromotionError("LOCAL_LIBRARY_ROOT_ESCAPE")
    root, approved = root_input.resolve(strict=False), approved_input.resolve(strict=False)
    if root == approved or approved not in root.parents:
        raise LocalLibraryPromotionError("LOCAL_LIBRARY_ROOT_OUTSIDE_APPROVED_ROOT")
    banned = {"prod", "production", "live", "canonical_library", "staging"}
    if any(part.lower() in banned for part in root.parts):
        raise LocalLibraryPromotionError("PRODUCTION_LIKE_ROOT_FORBIDDEN")
    cwd = Path.cwd().resolve(strict=False)
    if root == cwd or cwd in root.parents or root in cwd.parents:
        raise LocalLibraryPromotionError("REPOSITORY_OR_CWD_ROOT_FORBIDDEN")
    for candidate in (approved, *[parent for parent in root.parents if parent == approved or approved in parent.parents]):
        if candidate.exists() and candidate.is_symlink():
            raise LocalLibraryPromotionError("SYMLINK_ROOT_FORBIDDEN")
    return root


def validate_library_approval(approval: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    invalid: list[str] = []
    if not isinstance(approval, Mapping):
        return {"valid": False, "reason_code": "LIBRARY_APPROVAL_REQUIRED", "invalid_fields": []}
    if set(approval) != _APPROVAL_FIELDS:
        invalid.append("FIELD_SET")
    for field in _APPROVAL_FIELDS - {"approved_at"}:
        if not isinstance(approval.get(field), str) or _IDENTIFIER_RE.fullmatch(approval[field]) is None:
            invalid.append(field)
    if approval.get("target_object_type") != "promotion_plan":
        invalid.append("target_object_type")
    if approval.get("target_object_id") != plan.get("promotion_plan_id"):
        invalid.append("target_object_id")
    if approval.get("approval_type") != "APPROVE_LIBRARY_PROMOTION":
        invalid.append("approval_type")
    if approval.get("approver_role") != "OWNER" or approval.get("approver_role") in _AUTOMATED_ROLES:
        invalid.append("approver_role")
    if approval.get("evidence_id") != plan.get("evidence_pointer", {}).get("evidence_id"):
        invalid.append("evidence_id")
    try:
        approved_at = _timestamp(approval.get("approved_at"), "APPROVAL_TIMESTAMP_INVALID")
        plan_at = _timestamp(plan.get("created_at"), "PLAN_TIMESTAMP_INVALID")
        if approved_at < plan_at:
            invalid.append("approved_at")
    except LocalLibraryPromotionError:
        invalid.append("approved_at")
    if _prohibited_fields(approval):
        invalid.append("PROHIBITED_FIELD")
    return {
        "valid": not invalid,
        "reason_code": "LIBRARY_APPROVAL_VALID" if not invalid else "LIBRARY_APPROVAL_INVALID",
        "invalid_fields": sorted(set(invalid)),
    }


def validate_rollback_authorization(authorization: Any) -> dict[str, Any]:
    invalid: list[str] = []
    if not isinstance(authorization, Mapping):
        return {"valid": False, "reason_code": "ROLLBACK_AUTHORIZATION_REQUIRED", "invalid_fields": []}
    if set(authorization) != _ROLLBACK_FIELDS:
        invalid.append("FIELD_SET")
    for field in _ROLLBACK_FIELDS - {"approved_at"}:
        if not isinstance(authorization.get(field), str) or _IDENTIFIER_RE.fullmatch(authorization[field]) is None:
            invalid.append(field)
    if authorization.get("actor_role") not in _ROLLBACK_ROLES or authorization.get("actor_role") in _AUTOMATED_ROLES:
        invalid.append("actor_role")
    try:
        _timestamp(authorization.get("approved_at"), "ROLLBACK_TIMESTAMP_INVALID")
    except LocalLibraryPromotionError:
        invalid.append("approved_at")
    if _prohibited_fields(authorization):
        invalid.append("PROHIBITED_FIELD")
    return {
        "valid": not invalid,
        "reason_code": "ROLLBACK_AUTHORIZATION_VALID" if not invalid else "ROLLBACK_AUTHORIZATION_INVALID",
        "invalid_fields": sorted(set(invalid)),
    }


class LocalNonprodLibraryPromotion:
    """SQLite-backed local Reference promotion with compensated projections."""

    def __init__(
        self,
        *,
        library_root: str | Path,
        approved_local_root: str | Path,
        environment: str,
        explicit_allow_actual_local_promotion: bool,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.library_root = validate_local_library_root(
            library_root,
            approved_local_root=approved_local_root,
            environment=environment,
            explicit_allow_actual_local_promotion=explicit_allow_actual_local_promotion,
        )
        self.library_root.mkdir(parents=True, exist_ok=True)
        self.cards_root = self.library_root / "reference_cards"
        self.cards_root.mkdir(exist_ok=True)
        self.database_path = self.library_root / "local_library.sqlite"
        self.index_path = self.library_root / "library_index.json"
        self._clock = clock
        try:
            self.connection = sqlite3.connect(self.database_path, isolation_level=None)
            self.connection.row_factory = sqlite3.Row
            self._initialize()
        except (sqlite3.DatabaseError, OSError) as exc:
            raise LocalLibraryPromotionError("LOCAL_LIBRARY_OPEN_FAILED") from exc

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS library_records(
              library_record_id TEXT PRIMARY KEY, node_id TEXT NOT NULL UNIQUE,
              tenant_id TEXT NOT NULL, organization_id TEXT NOT NULL, status TEXT NOT NULL,
              revision INTEGER NOT NULL CHECK(revision > 0), payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence_pointers(
              evidence_id TEXT PRIMARY KEY, library_record_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
              organization_id TEXT NOT NULL, active INTEGER NOT NULL CHECK(active IN (0,1)),
              payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL,
              FOREIGN KEY(library_record_id) REFERENCES library_records(library_record_id));
            CREATE TABLE IF NOT EXISTS promotion_traces(
              promotion_trace_id TEXT PRIMARY KEY, promotion_plan_id TEXT NOT NULL, library_record_id TEXT NOT NULL,
              approval_record_id TEXT NOT NULL UNIQUE, tenant_id TEXT NOT NULL, organization_id TEXT NOT NULL,
              status TEXT NOT NULL, payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL,
              FOREIGN KEY(library_record_id) REFERENCES library_records(library_record_id));
            CREATE TABLE IF NOT EXISTS approval_records(
              approval_record_id TEXT PRIMARY KEY, target_object_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
              organization_id TEXT NOT NULL, payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS promotion_requests(
              tenant_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL,
              result_json TEXT NOT NULL, PRIMARY KEY(tenant_id,idempotency_key));
            CREATE TABLE IF NOT EXISTS rollback_audits(
              rollback_authorization_id TEXT PRIMARY KEY, promotion_trace_id TEXT NOT NULL,
              library_record_id TEXT NOT NULL, tenant_id TEXT NOT NULL, organization_id TEXT NOT NULL,
              reason_code TEXT NOT NULL, approved_at TEXT NOT NULL, proofpack_id TEXT NOT NULL,
              payload_hash TEXT NOT NULL);
            """
        )
        row = self.connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if row is None:
            self.connection.execute("INSERT INTO metadata(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        elif row[0] != str(SCHEMA_VERSION):
            raise LocalLibraryPromotionError("LOCAL_LIBRARY_SCHEMA_VERSION_MISMATCH")

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _source_objects(source_repository: Any, item: Mapping[str, Any], tenant: str, organization: str) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_result = source_repository.read_record(
            item["source_candidate_record_id"], tenant_id=tenant, organization_id=organization
        )
        candidate = candidate_result.get("domain_object") if candidate_result.get("found") is True else None
        if not isinstance(candidate, Mapping) or not validate_analytics_improvement_candidate(candidate).get("valid"):
            raise LocalLibraryPromotionError("SOURCE_CANDIDATE_NOT_ACTIVE_OR_VALID")
        records = source_repository.list_records(tenant_id=tenant, organization_id=organization).get("records", [])
        record_id = next((row.get("record_id") for row in records if row.get("record_type") == "ANALYTICS_EVENT" and row.get("domain_object_id") == item.get("source_event_id")), None)
        if not record_id:
            raise LocalLibraryPromotionError("SOURCE_EVENT_NOT_ACTIVE_OR_VALID")
        event_result = source_repository.read_record(record_id, tenant_id=tenant, organization_id=organization)
        event = event_result.get("domain_object") if event_result.get("found") is True else None
        if not isinstance(event, Mapping) or not validate_analytics_event(event).get("valid"):
            raise LocalLibraryPromotionError("SOURCE_EVENT_NOT_ACTIVE_OR_VALID")
        return dict(candidate), dict(event)

    def _eligibility(
        self, warehouse_repository: Any, source_repository: Any, plan: Mapping[str, Any],
        approval: Mapping[str, Any], tenant: str, organization: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        validation = validate_promotion_plan(plan)
        if not validation.get("valid"):
            raise LocalLibraryPromotionError("PROMOTION_PLAN_INVALID")
        approval_validation = validate_library_approval(approval, plan)
        if not approval_validation["valid"]:
            raise LocalLibraryPromotionError("LIBRARY_APPROVAL_INVALID")
        if _timestamp(approval["approved_at"], "APPROVAL_TIMESTAMP_INVALID") > _now(self._clock):
            raise LocalLibraryPromotionError("LIBRARY_APPROVAL_STALE_OR_FUTURE")
        item_result = warehouse_repository.read_item(plan["warehouse_item_id"], tenant_id=tenant, organization_id=organization)
        item = item_result.get("item") if item_result.get("found") is True else None
        if not isinstance(item, Mapping):
            raise LocalLibraryPromotionError("RECORD_NOT_FOUND_OR_NOT_VISIBLE")
        if item.get("current_status") != "approved_for_warehouse":
            raise LocalLibraryPromotionError("WAREHOUSE_ITEM_NOT_APPROVED")
        if item.get("revision") != plan.get("warehouse_item_revision"):
            raise LocalLibraryPromotionError("REVISION_CONFLICT")
        approval_event_result = warehouse_repository.read_review_event_by_approval(
            item["warehouse_item_id"], plan["approval_event_id"], tenant_id=tenant, organization_id=organization
        )
        approval_event = approval_event_result.get("review_event") if approval_event_result.get("found") is True else None
        if not isinstance(approval_event, Mapping) or not validate_review_event(approval_event).get("valid"):
            raise LocalLibraryPromotionError("WAREHOUSE_APPROVAL_EVENT_REQUIRED")
        if approval_event.get("decision") != "APPROVE_WAREHOUSE" or approval_event.get("approval_event_id") != plan.get("approval_event_id"):
            raise LocalLibraryPromotionError("WAREHOUSE_APPROVAL_EVENT_MISMATCH")
        if item.get("rights_status") not in _ALLOWED_RIGHTS or item.get("sensitivity") not in _ALLOWED_SENSITIVITY:
            raise LocalLibraryPromotionError("RIGHTS_OR_SENSITIVITY_NOT_ELIGIBLE")
        if item.get("approved_for_library") is not False or item.get("auto_promote") is not False:
            raise LocalLibraryPromotionError("PROMOTION_MARKER_FORBIDDEN")
        candidate, event = self._source_objects(source_repository, item, tenant, organization)
        continuity = {
            "tenant_id": tenant, "organization_id": organization, "cohort_id": event.get("cohort_id"),
            "source_candidate_id": candidate.get("candidate_id"), "source_event_id": event.get("event_id"),
            "source_request_id": event.get("request_id"), "source_trace_id": event.get("trace_id"),
            "query_hash": event.get("query_hash"),
        }
        if any(item.get(key) != value or plan.get(key) != value for key, value in continuity.items()):
            raise LocalLibraryPromotionError("SOURCE_CONTINUITY_INVALID")
        if plan.get("rights_status") != item.get("rights_status") or plan.get("raw_text_policy") != RAW_TEXT_POLICY_POINTER_ONLY:
            raise LocalLibraryPromotionError("PROMOTION_PLAN_POLICY_CONFLICT")
        if _prohibited_fields(item) or _prohibited_fields(plan) or _prohibited_fields(approval):
            raise LocalLibraryPromotionError("PROHIBITED_FIELD")
        return dict(item), candidate, event

    @staticmethod
    def _row_payload(row: sqlite3.Row | None, reason: str) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        serialized, digest = canonical_payload(payload)
        if serialized != row["payload_json"] or digest != row["payload_hash"]:
            raise LocalLibraryPromotionError(reason)
        return payload

    def _active_index(self, override: Mapping[str, Any] | None = None, remove_id: str | None = None) -> dict[str, Any]:
        rows = self.connection.execute("SELECT payload_json,payload_hash FROM library_records WHERE status=? ORDER BY node_id", (STATUS_ACTIVE,)).fetchall()
        entries = []
        for row in rows:
            record = self._row_payload(row, "LIBRARY_RECORD_INTEGRITY_FAILED")
            if record and record["library_record_id"] != remove_id:
                entries.append({
                    "library_record_id": record["library_record_id"], "node_id": record["node_id"],
                    "doc_kind": record["doc_kind"], "canonical_lang": record["canonical_lang"],
                    "evidence_id": record["evidence_id"], "promotion_trace_id": record["promotion_trace_id"],
                    "status": record["status"], "record_hash": canonical_payload(record)[1],
                })
        if override is not None:
            entries = [e for e in entries if e["library_record_id"] != override["library_record_id"]]
            entries.append({
                "library_record_id": override["library_record_id"], "node_id": override["node_id"],
                "doc_kind": override["doc_kind"], "canonical_lang": override["canonical_lang"],
                "evidence_id": override["evidence_id"], "promotion_trace_id": override["promotion_trace_id"],
                "status": override["status"], "record_hash": canonical_payload(override)[1],
            })
            entries.sort(key=lambda value: value["node_id"])
        return {"schema_version": SCHEMA_VERSION, "active_records": entries}

    @staticmethod
    def _card(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "node_id": record["node_id"], "doc_kind": DOC_KIND_REFERENCE,
            "canonical_lang": CANONICAL_LANG, "source_lang": SOURCE_LANG, "title_code": record["title_code"],
            "purpose_code": record["purpose_code"], "scope_code": record["scope_code"],
            "evidence_id": record["evidence_id"], "promotion_trace_id": record["promotion_trace_id"],
            "rights_status": record["rights_status"], "raw_text_policy": RAW_TEXT_POLICY_POINTER_ONLY,
            "source_candidate_id": record["source_candidate_id"], "source_event_id": record["source_event_id"],
            "status": record["status"], "record_hash": canonical_payload(record)[1],
        }

    def _projection_paths(self, node_id: str) -> tuple[Path, Path]:
        basename = hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:32] + ".json"
        return self.cards_root / basename, self.index_path

    @staticmethod
    def _restore(path: Path, previous: bytes | None) -> None:
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            temp = path.with_name(path.name + ".restore.tmp")
            temp.write_bytes(previous)
            os.replace(temp, path)

    def promote(
        self, warehouse_repository: Any, source_repository: Any, plan: Any, approval_record: Any, *,
        tenant_id: Any, organization_id: Any, idempotency_key: Any,
        expected_plan_hash: Any | None = None, failure_injection: str | None = None,
    ) -> dict[str, Any]:
        tenant, organization = _scope(tenant_id, organization_id)
        key = _identifier(idempotency_key, "IDEMPOTENCY_KEY_REQUIRED")
        if failure_injection == "before_transaction":
            raise LocalLibraryPromotionError("INJECTED_PROMOTION_FAILURE")
        if not isinstance(plan, Mapping) or not isinstance(approval_record, Mapping):
            raise LocalLibraryPromotionError("PROMOTION_INPUT_INVALID")
        if expected_plan_hash is not None and expected_plan_hash != plan.get("plan_hash"):
            raise LocalLibraryPromotionError("PROMOTION_PLAN_CONFLICT")
        item, _candidate, _event = self._eligibility(
            warehouse_repository, source_repository, plan, approval_record, tenant, organization
        )
        if failure_injection == "after_approval_verification":
            raise LocalLibraryPromotionError("INJECTED_PROMOTION_FAILURE")
        request_basis = {
            "plan_hash": plan["plan_hash"], "approval_record_id": approval_record["approval_record_id"],
            "tenant_id": tenant, "organization_id": organization,
        }
        request_hash = canonical_payload(request_basis)[1]
        prior = self.connection.execute(
            "SELECT request_hash,result_json FROM promotion_requests WHERE tenant_id=? AND idempotency_key=?", (tenant, key)
        ).fetchone()
        if prior:
            if prior["request_hash"] == request_hash:
                result = json.loads(prior["result_json"])
                result["reason_code"] = "IDEMPOTENT_REPLAY"
                result["promoted"] = False
                return result
            return {"promoted": False, "reason_code": "IDEMPOTENCY_CONFLICT", "library_record": None}
        if self.connection.execute("SELECT 1 FROM approval_records WHERE approval_record_id=?", (approval_record["approval_record_id"],)).fetchone():
            raise LocalLibraryPromotionError("REUSED_APPROVAL_NOT_ALLOWED")
        node_id = plan["target_library_node_id"]
        identity = hashlib.sha256((tenant + "\x1f" + organization + "\x1f" + node_id).encode("utf-8")).hexdigest()
        record_id = f"library:record:{identity[:32]}"
        evidence_id = plan["evidence_pointer"]["evidence_id"]
        trace_seed = hashlib.sha256((plan["plan_hash"] + "\x1f" + approval_record["approval_record_id"]).encode("utf-8")).hexdigest()
        trace_id = f"promotion:trace:{trace_seed[:32]}"
        existing = self.connection.execute("SELECT * FROM library_records WHERE node_id=?", (node_id,)).fetchone()
        revision = 1
        if existing:
            existing_payload = self._row_payload(existing, "LIBRARY_RECORD_INTEGRITY_FAILED")
            if existing_payload["source_candidate_id"] != plan["source_candidate_id"]:
                raise LocalLibraryPromotionError("TARGET_ID_CONFLICT")
            if existing_payload["status"] != STATUS_ROLLED_BACK:
                raise LocalLibraryPromotionError("TARGET_ID_CONFLICT")
            revision = existing_payload["revision"] + 1
        created = _now(self._clock)
        record = {
            "schema_version": SCHEMA_VERSION, "contract_version": CONTRACT_VERSION,
            "library_record_id": record_id, "node_id": node_id, "doc_kind": DOC_KIND_REFERENCE,
            "canonical_lang": CANONICAL_LANG, "source_lang": SOURCE_LANG, "tenant_id": tenant,
            "organization_id": organization, "cohort_id": plan["cohort_id"],
            "title_code": "ANALYTICS_IMPROVEMENT_REFERENCE", "purpose_code": "CURATED_IMPROVEMENT_POINTER",
            "scope_code": "LOCAL_NONPRODUCTION", "source_candidate_id": plan["source_candidate_id"],
            "source_event_id": plan["source_event_id"], "source_request_id": plan["source_request_id"],
            "source_trace_id": plan["source_trace_id"], "query_hash": plan["query_hash"],
            "warehouse_item_id": plan["warehouse_item_id"], "warehouse_item_revision": plan["warehouse_item_revision"],
            "promotion_plan_id": plan["promotion_plan_id"], "promotion_plan_hash": plan["plan_hash"],
            "approval_record_id": approval_record["approval_record_id"], "evidence_id": evidence_id,
            "promotion_trace_id": trace_id, "rights_status": plan["rights_status"],
            "raw_text_policy": RAW_TEXT_POLICY_POINTER_ONLY, "provenance": dict(plan["provenance"]),
            "status": STATUS_ACTIVE, "revision": revision, "created_at": created if revision == 1 else existing_payload["created_at"],
            "updated_at": created,
        }
        pointer = dict(plan["evidence_pointer"])
        pointer.update({"library_record_id": record_id, "node_id": node_id, "status": STATUS_ACTIVE})
        trace_basis = {
            "promotion_trace_id": trace_id, "promotion_plan_id": plan["promotion_plan_id"],
            "warehouse_item_id": plan["warehouse_item_id"], "warehouse_item_revision": plan["warehouse_item_revision"],
            "approval_record_id": approval_record["approval_record_id"], "source_candidate_id": plan["source_candidate_id"],
            "source_event_id": plan["source_event_id"], "library_record_id": record_id, "node_id": node_id,
            "evidence_id": evidence_id, "tenant_id": tenant, "organization_id": organization,
            "started_at": created, "committed_at": created, "status": TRACE_PROMOTED,
            "proofpack_id": approval_record["proofpack_id"],
        }
        trace_basis["payload_hash"] = canonical_payload({**record, "evidence_pointer": pointer})[1]
        trace = trace_basis
        if _prohibited_fields(record) or _prohibited_fields(pointer) or _prohibited_fields(trace):
            raise LocalLibraryPromotionError("PROHIBITED_FIELD")
        card_path, index_path = self._projection_paths(node_id)
        prior_card = card_path.read_bytes() if card_path.exists() else None
        prior_index = index_path.read_bytes() if index_path.exists() else None
        stage = self.library_root / (".promotion-stage-" + trace_seed[:16])
        if stage.exists():
            raise LocalLibraryPromotionError("PROMOTION_STAGE_CONFLICT")
        stage.mkdir()
        try:
            (stage / "card.json").write_bytes(_json_bytes(self._card(record)))
            (stage / "index.json").write_bytes(_json_bytes(self._active_index(override=record)))
            self.connection.execute("BEGIN IMMEDIATE")
            approval_json, approval_hash = canonical_payload(approval_record)
            record_json, record_hash = canonical_payload(record)
            pointer_json, pointer_hash = canonical_payload(pointer)
            trace_json, trace_hash = canonical_payload(trace)
            self.connection.execute(
                "INSERT INTO library_records VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(library_record_id) DO UPDATE SET status=excluded.status,revision=excluded.revision,"
                "payload_json=excluded.payload_json,payload_hash=excluded.payload_hash",
                (record_id, node_id, tenant, organization, STATUS_ACTIVE, revision, record_json, record_hash),
            )
            if failure_injection == "after_library_record": raise LocalLibraryPromotionError("INJECTED_PROMOTION_FAILURE")
            self.connection.execute(
                "INSERT OR REPLACE INTO evidence_pointers VALUES(?,?,?,?,?,?,?)",
                (evidence_id, record_id, tenant, organization, 1, pointer_json, pointer_hash),
            )
            if failure_injection == "after_evidence": raise LocalLibraryPromotionError("INJECTED_PROMOTION_FAILURE")
            self.connection.execute(
                "INSERT INTO promotion_traces VALUES(?,?,?,?,?,?,?,?,?)",
                (trace_id, plan["promotion_plan_id"], record_id, approval_record["approval_record_id"], tenant, organization, TRACE_PROMOTED, trace_json, trace_hash),
            )
            if failure_injection == "after_trace": raise LocalLibraryPromotionError("INJECTED_PROMOTION_FAILURE")
            self.connection.execute(
                "INSERT INTO approval_records VALUES(?,?,?,?,?,?)",
                (approval_record["approval_record_id"], plan["promotion_plan_id"], tenant, organization, approval_json, approval_hash),
            )
            if failure_injection == "before_card_finalization": raise LocalLibraryPromotionError("INJECTED_PROMOTION_FAILURE")
            os.replace(stage / "card.json", card_path)
            if failure_injection == "before_index_finalization": raise LocalLibraryPromotionError("INJECTED_PROMOTION_FAILURE")
            if failure_injection == "during_atomic_rename": raise OSError("injected")
            os.replace(stage / "index.json", index_path)
            if failure_injection == "during_transaction_commit": raise LocalLibraryPromotionError("INJECTED_PROMOTION_FAILURE")
            result = {
                "promoted": True, "reason_code": "PROMOTED_LOCAL_NONPRODUCTION", "library_record": record,
                "evidence_pointer": pointer, "promotion_trace": trace, "approval_record": dict(approval_record),
                "card_hash": "sha256:" + hashlib.sha256(card_path.read_bytes()).hexdigest(),
                "index_hash": "sha256:" + hashlib.sha256(index_path.read_bytes()).hexdigest(),
            }
            if failure_injection == "during_readback_verification": raise LocalLibraryPromotionError("INJECTED_PROMOTION_FAILURE")
            request_json = json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            self.connection.execute("INSERT INTO promotion_requests VALUES(?,?,?,?)", (tenant, key, request_hash, request_json))
            self.connection.execute("COMMIT")
            return result
        except Exception as exc:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            self._restore(card_path, prior_card)
            self._restore(index_path, prior_index)
            if isinstance(exc, LocalLibraryPromotionError):
                raise
            raise LocalLibraryPromotionError("PROMOTION_ATOMIC_FINALIZATION_FAILED") from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def read_library_record(self, library_record_id: Any, *, tenant_id: Any, organization_id: Any) -> dict[str, Any]:
        record_id = _identifier(library_record_id, "LIBRARY_RECORD_ID_REQUIRED")
        tenant, organization = _scope(tenant_id, organization_id)
        row = self.connection.execute(
            "SELECT * FROM library_records WHERE library_record_id=? AND tenant_id=? AND organization_id=? AND status=?",
            (record_id, tenant, organization, STATUS_ACTIVE),
        ).fetchone()
        if row is None:
            return {"found": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "library_record": None}
        return {"found": True, "reason_code": "RECORD_FOUND", "library_record": self._row_payload(row, "LIBRARY_RECORD_INTEGRITY_FAILED")}

    def read_promotion_trace(self, promotion_trace_id: Any, *, tenant_id: Any, organization_id: Any) -> dict[str, Any]:
        trace_id = _identifier(promotion_trace_id, "PROMOTION_TRACE_ID_REQUIRED")
        tenant, organization = _scope(tenant_id, organization_id)
        row = self.connection.execute(
            "SELECT payload_json,payload_hash FROM promotion_traces WHERE promotion_trace_id=? AND tenant_id=? AND organization_id=?",
            (trace_id, tenant, organization),
        ).fetchone()
        if row is None:
            return {"found": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "promotion_trace": None}
        return {"found": True, "reason_code": "RECORD_FOUND", "promotion_trace": self._row_payload(row, "PROMOTION_TRACE_INTEGRITY_FAILED")}

    def read_evidence_pointer(self, evidence_id: Any, *, tenant_id: Any, organization_id: Any) -> dict[str, Any]:
        """Return one active pointer through the same tenant/org fail-closed boundary."""

        pointer_id = _identifier(evidence_id, "EVIDENCE_ID_REQUIRED")
        tenant, organization = _scope(tenant_id, organization_id)
        row = self.connection.execute(
            "SELECT payload_json,payload_hash FROM evidence_pointers "
            "WHERE evidence_id=? AND tenant_id=? AND organization_id=? AND active=1",
            (pointer_id, tenant, organization),
        ).fetchone()
        if row is None:
            return {"found": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "evidence_pointer": None}
        return {
            "found": True,
            "reason_code": "RECORD_FOUND",
            "evidence_pointer": self._row_payload(row, "EVIDENCE_POINTER_INTEGRITY_FAILED"),
        }

    def object_counts(self) -> dict[str, int]:
        tables = ("library_records", "evidence_pointers", "promotion_traces", "approval_records", "rollback_audits")
        return {table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}

    def verify_readback(self, library_record_id: Any, *, tenant_id: Any, organization_id: Any) -> dict[str, Any]:
        result = self.read_library_record(library_record_id, tenant_id=tenant_id, organization_id=organization_id)
        if not result["found"]:
            return {"valid": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE"}
        record = result["library_record"]
        evidence = self.connection.execute(
            "SELECT payload_json,payload_hash FROM evidence_pointers WHERE evidence_id=? AND active=1", (record["evidence_id"],)
        ).fetchone()
        trace = self.connection.execute(
            "SELECT payload_json,payload_hash FROM promotion_traces WHERE promotion_trace_id=?", (record["promotion_trace_id"],)
        ).fetchone()
        card_path, _ = self._projection_paths(record["node_id"])
        try:
            pointer = self._row_payload(evidence, "EVIDENCE_POINTER_INTEGRITY_FAILED")
            trace_payload = self._row_payload(trace, "PROMOTION_TRACE_INTEGRITY_FAILED")
            card = json.loads(card_path.read_text(encoding="utf-8"))
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalLibraryPromotionError("DERIVED_PROJECTION_INTEGRITY_FAILED") from exc
        aligned = (
            pointer and trace_payload and pointer["library_record_id"] == record["library_record_id"]
            and trace_payload["library_record_id"] == record["library_record_id"]
            and card["record_hash"] == canonical_payload(record)[1]
            and any(entry["library_record_id"] == record["library_record_id"] for entry in index["active_records"])
        )
        return {"valid": bool(aligned), "reason_code": "READBACK_VALID" if aligned else "READBACK_INVALID"}

    def rollback(
        self, authorization: Any, *, tenant_id: Any, organization_id: Any,
        failure_injection: str | None = None,
    ) -> dict[str, Any]:
        tenant, organization = _scope(tenant_id, organization_id)
        validation = validate_rollback_authorization(authorization)
        if not validation["valid"]:
            raise LocalLibraryPromotionError("ROLLBACK_AUTHORIZATION_INVALID")
        auth_id = authorization["rollback_authorization_id"]
        prior = self.connection.execute("SELECT * FROM rollback_audits WHERE rollback_authorization_id=?", (auth_id,)).fetchone()
        if prior:
            if prior["promotion_trace_id"] == authorization["promotion_trace_id"] and prior["library_record_id"] == authorization["library_record_id"]:
                return {"rolled_back": False, "reason_code": "IDEMPOTENT_REPLAY", "library_record_id": authorization["library_record_id"]}
            raise LocalLibraryPromotionError("ROLLBACK_CONFLICT")
        trace_row = self.connection.execute(
            "SELECT * FROM promotion_traces WHERE promotion_trace_id=? AND tenant_id=? AND organization_id=?",
            (authorization["promotion_trace_id"], tenant, organization),
        ).fetchone()
        record_row = self.connection.execute(
            "SELECT * FROM library_records WHERE library_record_id=? AND tenant_id=? AND organization_id=? AND status=?",
            (authorization["library_record_id"], tenant, organization, STATUS_ACTIVE),
        ).fetchone()
        if trace_row is None or record_row is None:
            raise LocalLibraryPromotionError("RECORD_NOT_FOUND_OR_NOT_VISIBLE")
        trace = self._row_payload(trace_row, "PROMOTION_TRACE_INTEGRITY_FAILED")
        record = self._row_payload(record_row, "LIBRARY_RECORD_INTEGRITY_FAILED")
        if trace["library_record_id"] != record["library_record_id"] or trace["status"] != TRACE_PROMOTED:
            raise LocalLibraryPromotionError("ROLLBACK_LINKAGE_INVALID")
        card_path, index_path = self._projection_paths(record["node_id"])
        prior_card = card_path.read_bytes() if card_path.exists() else None
        prior_index = index_path.read_bytes() if index_path.exists() else None
        stage = self.library_root / (".rollback-stage-" + hashlib.sha256(auth_id.encode()).hexdigest()[:16])
        stage.mkdir()
        try:
            rolled_record = dict(record); rolled_record.update({"status": STATUS_ROLLED_BACK, "updated_at": _now(self._clock)})
            rolled_trace = dict(trace); rolled_trace.update({"status": TRACE_ROLLED_BACK, "committed_at": _now(self._clock)})
            (stage / "index.json").write_bytes(_json_bytes(self._active_index(remove_id=record["library_record_id"])))
            self.connection.execute("BEGIN IMMEDIATE")
            record_json, record_hash = canonical_payload(rolled_record)
            trace_json, trace_hash = canonical_payload(rolled_trace)
            self.connection.execute(
                "UPDATE library_records SET status=?,payload_json=?,payload_hash=? WHERE library_record_id=?",
                (STATUS_ROLLED_BACK, record_json, record_hash, record["library_record_id"]),
            )
            self.connection.execute("UPDATE evidence_pointers SET active=0 WHERE evidence_id=?", (record["evidence_id"],))
            self.connection.execute(
                "UPDATE promotion_traces SET status=?,payload_json=?,payload_hash=? WHERE promotion_trace_id=?",
                (TRACE_ROLLED_BACK, trace_json, trace_hash, trace["promotion_trace_id"]),
            )
            audit_basis = {
                "rollback_authorization_id": auth_id, "promotion_trace_id": trace["promotion_trace_id"],
                "library_record_id": record["library_record_id"], "tenant_id": tenant, "organization_id": organization,
                "reason_code": authorization["reason_code"], "approved_at": authorization["approved_at"],
                "proofpack_id": authorization["proofpack_id"],
            }
            audit_hash = canonical_payload(audit_basis)[1]
            self.connection.execute(
                "INSERT INTO rollback_audits VALUES(?,?,?,?,?,?,?,?,?)",
                (auth_id, trace["promotion_trace_id"], record["library_record_id"], tenant, organization,
                 authorization["reason_code"], authorization["approved_at"], authorization["proofpack_id"], audit_hash),
            )
            if failure_injection == "before_projection_finalization":
                raise LocalLibraryPromotionError("INJECTED_ROLLBACK_FAILURE")
            card_path.unlink(missing_ok=True)
            os.replace(stage / "index.json", index_path)
            if failure_injection in {"before_commit", "during_readback"}:
                raise LocalLibraryPromotionError("INJECTED_ROLLBACK_FAILURE")
            self.connection.execute("COMMIT")
            return {"rolled_back": True, "reason_code": "ROLLED_BACK", "library_record_id": record["library_record_id"], "rollback_audit_hash": audit_hash}
        except Exception as exc:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            self._restore(card_path, prior_card)
            self._restore(index_path, prior_index)
            if isinstance(exc, LocalLibraryPromotionError):
                raise
            raise LocalLibraryPromotionError("ROLLBACK_ATOMIC_FINALIZATION_FAILED") from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)


__all__ = [
    "CANONICAL_LANG", "CONTRACT_VERSION", "DOC_KIND_REFERENCE", "ENVIRONMENT_LOCAL_NONPRODUCTION",
    "LocalLibraryPromotionError", "LocalNonprodLibraryPromotion", "RAW_TEXT_POLICY_POINTER_ONLY",
    "SCHEMA_VERSION", "SOURCE_LANG", "canonical_payload", "validate_library_approval",
    "validate_local_library_root", "validate_rollback_authorization",
]
