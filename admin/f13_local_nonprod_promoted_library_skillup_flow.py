"""Selected local promoted-Library -> Bridge -> Skillup flow with invalidation."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from admin.f13_feedback_queue_contract import validate_feedback_queue_contract
from admin.f13_runtime_guard import RESULT_HOLD, RESULT_OK, decide_bridge_result, project_bridge_safe_evidence
from admin.f13_skillup_bridge import skillup_answer_from_bridge_response
from admin.f13_local_nonprod_library_promotion import TRACE_PROMOTED, TRACE_ROLLED_BACK


SCHEMA_VERSION = 1
CONTRACT_VERSION = "answer.release.local_nonproduction.pointer_only.v1"
ENVIRONMENT_LOCAL_NONPRODUCTION = "local_nonproduction"
STATUS_ACTIVE = "ACTIVE"
STATUS_INVALIDATED = "INVALIDATED"
INVALIDATION_REASON = "SOURCE_PROMOTION_ROLLED_BACK"
PURPOSE_SKILLUP_ANSWER = "SKILLUP_ANSWER"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._@-]{0,199}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TRACE_FIELDS = {
    "schema_version", "contract_version", "answer_release_id", "tenant_id", "organization_id",
    "cohort_id", "request_id", "library_record_id", "node_id", "evidence_id", "promotion_trace_id",
    "bridge_trace_id", "answer_hash", "answer_length", "status", "invalidation_reason", "created_at",
    "invalidated_at", "revision", "idempotency_key",
}


class PromotedLibrarySkillupFlowError(RuntimeError):
    """Controlled local-flow failure with reason-code-only messages."""


def canonical_payload(payload: Mapping[str, Any]) -> tuple[str, str]:
    text = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return text, "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identifier(value: Any, reason: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise PromotedLibrarySkillupFlowError(reason)
    return value


def _scope(tenant_id: Any, organization_id: Any) -> tuple[str, str]:
    return _identifier(tenant_id, "TENANT_SCOPE_REQUIRED"), _identifier(organization_id, "ORGANIZATION_SCOPE_REQUIRED")


def _timestamp(value: Any, reason: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PromotedLibrarySkillupFlowError(reason) from exc
    else:
        raise PromotedLibrarySkillupFlowError(reason)
    if parsed.tzinfo is None:
        raise PromotedLibrarySkillupFlowError(reason)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now(clock: Callable[[], datetime] | None) -> str:
    return _timestamp(clock() if clock else datetime.now(timezone.utc), "TIMEZONE_AWARE_CLOCK_REQUIRED")


def validate_trace_storage_path(
    storage_path: str | Path, *, approved_local_root: str | Path, environment: str,
    explicit_allow_local_trace_storage: bool,
) -> Path:
    if environment != ENVIRONMENT_LOCAL_NONPRODUCTION:
        raise PromotedLibrarySkillupFlowError("LOCAL_NONPRODUCTION_ENVIRONMENT_REQUIRED")
    if explicit_allow_local_trace_storage is not True:
        raise PromotedLibrarySkillupFlowError("EXPLICIT_LOCAL_TRACE_STORAGE_ALLOW_REQUIRED")
    text, approved_text = str(storage_path), str(approved_local_root)
    if not text or not approved_text or text.startswith(("\\\\", "//")):
        raise PromotedLibrarySkillupFlowError("TRACE_STORAGE_PATH_INVALID")
    candidate, approved_candidate = Path(text), Path(approved_text)
    if not candidate.is_absolute() or not approved_candidate.is_absolute():
        raise PromotedLibrarySkillupFlowError("ABSOLUTE_LOCAL_ROOT_REQUIRED")
    if ".." in candidate.parts or ".." in approved_candidate.parts:
        raise PromotedLibrarySkillupFlowError("TRACE_STORAGE_PATH_ESCAPE")
    resolved, approved = candidate.resolve(strict=False), approved_candidate.resolve(strict=False)
    if resolved == approved or approved not in resolved.parents:
        raise PromotedLibrarySkillupFlowError("TRACE_STORAGE_OUTSIDE_APPROVED_ROOT")
    if any(part.lower() in {"prod", "production", "live", "staging"} for part in resolved.parts):
        raise PromotedLibrarySkillupFlowError("PRODUCTION_LIKE_PATH_FORBIDDEN")
    cwd = Path.cwd().resolve(strict=False)
    if resolved == cwd or cwd in resolved.parents or resolved in cwd.parents:
        raise PromotedLibrarySkillupFlowError("REPOSITORY_OR_CWD_PATH_FORBIDDEN")
    return resolved


def validate_answer_release_trace(payload: Any) -> dict[str, Any]:
    invalid: list[str] = []
    if not isinstance(payload, Mapping):
        return {"valid": False, "reason_code": "TRACE_NOT_MAPPING", "invalid_fields": []}
    if set(payload) != _TRACE_FIELDS:
        invalid.append("FIELD_SET")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("contract_version") != CONTRACT_VERSION:
        invalid.append("VERSION")
    for field in (
        "answer_release_id", "tenant_id", "organization_id", "cohort_id", "request_id", "library_record_id",
        "node_id", "evidence_id", "promotion_trace_id", "bridge_trace_id", "idempotency_key",
    ):
        if not isinstance(payload.get(field), str) or _IDENTIFIER_RE.fullmatch(payload[field]) is None:
            invalid.append(field)
    if not isinstance(payload.get("answer_hash"), str) or _HASH_RE.fullmatch(payload["answer_hash"]) is None:
        invalid.append("answer_hash")
    if not isinstance(payload.get("answer_length"), int) or payload["answer_length"] < 1:
        invalid.append("answer_length")
    if payload.get("status") not in {STATUS_ACTIVE, STATUS_INVALIDATED}:
        invalid.append("status")
    if not isinstance(payload.get("revision"), int) or payload["revision"] < 1:
        invalid.append("revision")
    try:
        _timestamp(payload.get("created_at"), "CREATED_AT_INVALID")
    except PromotedLibrarySkillupFlowError:
        invalid.append("created_at")
    if payload.get("status") == STATUS_ACTIVE:
        if payload.get("invalidation_reason") is not None or payload.get("invalidated_at") is not None:
            invalid.append("ACTIVE_INVALIDATION_FIELDS")
    else:
        if payload.get("invalidation_reason") != INVALIDATION_REASON:
            invalid.append("invalidation_reason")
        try:
            _timestamp(payload.get("invalidated_at"), "INVALIDATED_AT_INVALID")
        except PromotedLibrarySkillupFlowError:
            invalid.append("invalidated_at")
    return {"valid": not invalid, "reason_code": "TRACE_VALID" if not invalid else "TRACE_INVALID", "invalid_fields": sorted(set(invalid))}


class LocalPromotedLibraryBridgeAdapter:
    """Scoped read-only adapter; never accepts a path or direct Skillup DB request."""

    @staticmethod
    def _hold(request_id: str, reason: str) -> dict[str, Any]:
        digest = hashlib.sha256((request_id + "\x1f" + reason).encode("utf-8")).hexdigest()[:32]
        return {
            "result_status": RESULT_HOLD, "evidence_items": [], "hold_reason": reason,
            "feedback_candidate_required": True, "raw_text_included": False,
            "internal_path_included": False, "bridge_trace_id": f"btrace:local:{digest}",
        }

    def retrieve(
        self, library_repository: Any, *, tenant_id: Any, organization_id: Any, request_id: Any,
        purpose: Any, library_record_id: Any, allowed_rights_status: Sequence[str], max_items: Any,
        role: Any, course_id: Any, module_id: Any, binding_id: Any,
        failure_injection: str | None = None,
    ) -> dict[str, Any]:
        tenant, organization = _scope(tenant_id, organization_id)
        request = _identifier(request_id, "REQUEST_ID_REQUIRED")
        record_id = _identifier(library_record_id, "LIBRARY_RECORD_ID_REQUIRED")
        if purpose != PURPOSE_SKILLUP_ANSWER:
            raise PromotedLibrarySkillupFlowError("SKILLUP_ANSWER_PURPOSE_REQUIRED")
        if not isinstance(max_items, int) or max_items < 1 or max_items > 10:
            raise PromotedLibrarySkillupFlowError("MAX_ITEMS_INVALID")
        safe_role = _identifier(role, "ROLE_REQUIRED")
        course, module, binding = (
            _identifier(course_id, "COURSE_ID_REQUIRED"), _identifier(module_id, "MODULE_ID_REQUIRED"),
            _identifier(binding_id, "BINDING_ID_REQUIRED"),
        )
        rights = {str(value).lower() for value in allowed_rights_status}
        try:
            record_result = library_repository.read_library_record(record_id, tenant_id=tenant, organization_id=organization)
        except Exception:
            return self._hold(request, "BRIDGE_SOURCE_READ_FAILED")
        record = record_result.get("library_record") if record_result.get("found") is True else None
        if not isinstance(record, Mapping):
            return self._hold(request, "EVIDENCE_INVALIDATED")
        if failure_injection == "after_record":
            return self._hold(request, "BRIDGE_SOURCE_READ_FAILED")
        if record.get("doc_kind") != "REFERENCE" or record.get("canonical_lang") != "EN" or record.get("raw_text_policy") != "POINTER_ONLY":
            return self._hold(request, "EVIDENCE_INELIGIBLE")
        if str(record.get("rights_status", "")).lower() not in rights:
            return self._hold(request, "EVIDENCE_RIGHTS_NOT_ALLOWED")
        try:
            pointer_result = library_repository.read_evidence_pointer(
                record["evidence_id"], tenant_id=tenant, organization_id=organization
            )
        except Exception:
            return self._hold(request, "BRIDGE_SOURCE_READ_FAILED")
        pointer = pointer_result.get("evidence_pointer") if pointer_result.get("found") is True else None
        if not isinstance(pointer, Mapping):
            return self._hold(request, "EVIDENCE_INVALIDATED")
        if failure_injection == "after_evidence":
            return self._hold(request, "BRIDGE_SOURCE_READ_FAILED")
        try:
            trace_result = library_repository.read_promotion_trace(
                record["promotion_trace_id"], tenant_id=tenant, organization_id=organization
            )
        except Exception:
            return self._hold(request, "BRIDGE_SOURCE_READ_FAILED")
        trace = trace_result.get("promotion_trace") if trace_result.get("found") is True else None
        if not isinstance(trace, Mapping) or trace.get("status") != TRACE_PROMOTED:
            return self._hold(request, "EVIDENCE_INVALIDATED")
        aligned = (
            pointer.get("status") == "ACTIVE" and pointer.get("library_record_id") == record["library_record_id"]
            and pointer.get("node_id") == record["node_id"] and trace.get("library_record_id") == record["library_record_id"]
            and trace.get("evidence_id") == pointer.get("evidence_id") and trace.get("tenant_id") == tenant
            and trace.get("organization_id") == organization and isinstance(record.get("cohort_id"), str)
        )
        # Cohort is authoritative on the active record; all other object links are exact.
        if not aligned or record.get("tenant_id") != tenant or record.get("organization_id") != organization:
            return self._hold(request, "EVIDENCE_LINK_MISMATCH")
        seed = "\x1f".join((request, record["library_record_id"], pointer["evidence_id"], trace["promotion_trace_id"]))
        bridge_trace_id = "btrace:local:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]
        item = project_bridge_safe_evidence({
            "evidence_id": pointer["evidence_id"], "bridge_trace_id": bridge_trace_id,
            "safe_summary": pointer["evidence_summary_code"], "pointer_uri": pointer["pointer_uri"],
            "raw_text_policy": pointer["raw_text_policy"], "rights_status": pointer["rights_status"],
            "source_doc_kind": "REFERENCE", "validation_shape_ids": pointer["validation_shape_ids"],
        })
        decision = decide_bridge_result(item, requester_module="Skillup", purpose="answer")
        if decision.get("result_status") != RESULT_OK:
            return self._hold(request, "EVIDENCE_INELIGIBLE")
        return {
            "result_status": RESULT_OK, "evidence_items": [item][:max_items], "hold_reason": None,
            "feedback_candidate_required": False, "raw_text_included": False, "internal_path_included": False,
            "bridge_trace_id": bridge_trace_id, "request_id": request, "role": safe_role,
            "evidence_depth": "student_safe" if safe_role.upper() == "STUDENT" else "instructor_safe",
            "course_id": course, "module_id": module, "binding_id": binding,
            "tenant_id": tenant, "organization_id": organization, "cohort_id": record["cohort_id"],
            "evidence_tenant_id": tenant, "evidence_organization_id": organization,
            "evidence_cohort_id": record["cohort_id"],
        }


class LocalAnswerReleaseTraceStore:
    def __init__(
        self, *, storage_path: str | Path, approved_local_root: str | Path, environment: str,
        explicit_allow_local_trace_storage: bool, clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.storage_path = validate_trace_storage_path(
            storage_path, approved_local_root=approved_local_root, environment=environment,
            explicit_allow_local_trace_storage=explicit_allow_local_trace_storage,
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.database_path = self.storage_path / "answer_release_traces.sqlite"
        self._clock = clock
        self.connection: sqlite3.Connection | None = None
        try:
            self.connection = sqlite3.connect(self.database_path, isolation_level=None)
            self.connection.row_factory = sqlite3.Row
            self._initialize()
        except PromotedLibrarySkillupFlowError:
            if self.connection is not None:
                self.connection.close()
            raise
        except (sqlite3.DatabaseError, OSError) as exc:
            if self.connection is not None:
                self.connection.close()
            raise PromotedLibrarySkillupFlowError("TRACE_STORAGE_OPEN_FAILED") from exc

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS answer_release_traces(
              answer_release_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, organization_id TEXT NOT NULL,
              promotion_trace_id TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision>0),
              payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_answer_trace_promotion ON answer_release_traces(tenant_id,organization_id,promotion_trace_id);
            CREATE TABLE IF NOT EXISTS trace_requests(
              tenant_id TEXT NOT NULL,idempotency_key TEXT NOT NULL,request_hash TEXT NOT NULL,
              answer_release_id TEXT NOT NULL,PRIMARY KEY(tenant_id,idempotency_key));
            CREATE TABLE IF NOT EXISTS invalidation_requests(
              tenant_id TEXT NOT NULL,idempotency_key TEXT NOT NULL,request_hash TEXT NOT NULL,
              result_json TEXT NOT NULL,PRIMARY KEY(tenant_id,idempotency_key));
            """
        )
        row = self.connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if row is None:
            self.connection.execute("INSERT INTO metadata VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        elif row[0] != str(SCHEMA_VERSION):
            raise PromotedLibrarySkillupFlowError("TRACE_SCHEMA_VERSION_MISMATCH")

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @staticmethod
    def _payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        text, digest = canonical_payload(payload)
        if text != row["payload_json"] or digest != row["payload_hash"] or not validate_answer_release_trace(payload)["valid"]:
            raise PromotedLibrarySkillupFlowError("ANSWER_TRACE_INTEGRITY_FAILED")
        return payload

    def create_trace(
        self, *, tenant_id: Any, organization_id: Any, cohort_id: Any, request_id: Any,
        library_record: Mapping[str, Any], bridge_result: Mapping[str, Any], released_answer: Any,
        idempotency_key: Any, failure_injection: str | None = None,
    ) -> dict[str, Any]:
        tenant, organization = _scope(tenant_id, organization_id)
        cohort, request, key = _identifier(cohort_id, "COHORT_ID_REQUIRED"), _identifier(request_id, "REQUEST_ID_REQUIRED"), _identifier(idempotency_key, "IDEMPOTENCY_KEY_REQUIRED")
        if not isinstance(released_answer, str) or not released_answer:
            raise PromotedLibrarySkillupFlowError("ANSWER_VALUE_REQUIRED")
        evidence = (bridge_result.get("evidence_items") or [None])[0]
        if bridge_result.get("result_status") != RESULT_OK or not isinstance(evidence, Mapping):
            raise PromotedLibrarySkillupFlowError("BRIDGE_OK_REQUIRED")
        answer_hash = "sha256:" + hashlib.sha256(released_answer.encode("utf-8")).hexdigest()
        basis = {
            "tenant_id": tenant, "organization_id": organization, "cohort_id": cohort, "request_id": request,
            "library_record_id": library_record["library_record_id"], "node_id": library_record["node_id"],
            "evidence_id": evidence["evidence_id"], "promotion_trace_id": library_record["promotion_trace_id"],
            "bridge_trace_id": evidence["bridge_trace_id"], "answer_hash": answer_hash,
            "answer_length": len(released_answer),
        }
        request_hash = canonical_payload(basis)[1]
        prior = self.connection.execute("SELECT * FROM trace_requests WHERE tenant_id=? AND idempotency_key=?", (tenant, key)).fetchone()
        if prior:
            if prior["request_hash"] != request_hash:
                return {"created": False, "reason_code": "IDEMPOTENCY_CONFLICT", "answer_release_trace": None}
            result = self.read_trace(prior["answer_release_id"], tenant_id=tenant, organization_id=organization)
            return {"created": False, "reason_code": "IDEMPOTENT_REPLAY", "answer_release_trace": result["answer_release_trace"]}
        identity = hashlib.sha256(canonical_payload(basis)[0].encode("utf-8")).hexdigest()[:32]
        trace = {
            "schema_version": SCHEMA_VERSION, "contract_version": CONTRACT_VERSION,
            "answer_release_id": f"answer:release:{identity}", **basis, "status": STATUS_ACTIVE,
            "invalidation_reason": None, "created_at": _now(self._clock), "invalidated_at": None,
            "revision": 1, "idempotency_key": key,
        }
        if not validate_answer_release_trace(trace)["valid"]:
            raise PromotedLibrarySkillupFlowError("ANSWER_TRACE_INVALID")
        text, digest = canonical_payload(trace)
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "INSERT INTO answer_release_traces VALUES(?,?,?,?,?,?,?,?)",
                (trace["answer_release_id"], tenant, organization, trace["promotion_trace_id"], STATUS_ACTIVE, 1, text, digest),
            )
            if failure_injection == "after_insert":
                raise PromotedLibrarySkillupFlowError("INJECTED_TRACE_FAILURE")
            self.connection.execute("INSERT INTO trace_requests VALUES(?,?,?,?)", (tenant, key, request_hash, trace["answer_release_id"]))
            self.connection.execute("COMMIT")
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise
        return {"created": True, "reason_code": "TRACE_CREATED", "answer_release_trace": trace}

    def read_trace(self, answer_release_id: Any, *, tenant_id: Any, organization_id: Any) -> dict[str, Any]:
        trace_id = _identifier(answer_release_id, "ANSWER_RELEASE_ID_REQUIRED")
        tenant, organization = _scope(tenant_id, organization_id)
        row = self.connection.execute(
            "SELECT payload_json,payload_hash FROM answer_release_traces WHERE answer_release_id=? AND tenant_id=? AND organization_id=?",
            (trace_id, tenant, organization),
        ).fetchone()
        if row is None:
            return {"found": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "answer_release_trace": None}
        return {"found": True, "reason_code": "RECORD_FOUND", "answer_release_trace": self._payload(row)}

    def invalidate_promotion(
        self, promotion_trace_id: Any, *, tenant_id: Any, organization_id: Any, expected_revision: Any,
        idempotency_key: Any, reason: str = INVALIDATION_REASON, failure_injection: str | None = None,
    ) -> dict[str, Any]:
        promotion_id = _identifier(promotion_trace_id, "PROMOTION_TRACE_ID_REQUIRED")
        tenant, organization = _scope(tenant_id, organization_id)
        key = _identifier(idempotency_key, "IDEMPOTENCY_KEY_REQUIRED")
        request_basis = {"promotion_trace_id": promotion_id, "tenant_id": tenant, "organization_id": organization, "expected_revision": expected_revision, "reason": reason}
        request_hash = canonical_payload(request_basis)[1]
        prior = self.connection.execute("SELECT * FROM invalidation_requests WHERE tenant_id=? AND idempotency_key=?", (tenant, key)).fetchone()
        if prior:
            if prior["request_hash"] != request_hash:
                return {"invalidated": False, "reason_code": "INVALIDATION_CONFLICT", "traces": []}
            result = json.loads(prior["result_json"]); result["reason_code"] = "IDEMPOTENT_REPLAY"; result["invalidated"] = False
            return result
        rows = self.connection.execute(
            "SELECT * FROM answer_release_traces WHERE tenant_id=? AND organization_id=? AND promotion_trace_id=? ORDER BY answer_release_id",
            (tenant, organization, promotion_id),
        ).fetchall()
        if not rows:
            raise PromotedLibrarySkillupFlowError("RECORD_NOT_FOUND_OR_NOT_VISIBLE")
        payloads = [self._payload(row) for row in rows]
        if reason != INVALIDATION_REASON:
            return {"invalidated": False, "reason_code": "INVALIDATION_CONFLICT", "traces": []}
        if any(payload["revision"] != expected_revision for payload in payloads):
            return {"invalidated": False, "reason_code": "REVISION_CONFLICT", "traces": []}
        updated = []
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            for payload in payloads:
                item = dict(payload)
                item.update({"status": STATUS_INVALIDATED, "invalidation_reason": reason, "invalidated_at": _now(self._clock), "revision": payload["revision"] + 1})
                text, digest = canonical_payload(item)
                self.connection.execute(
                    "UPDATE answer_release_traces SET status=?,revision=?,payload_json=?,payload_hash=? WHERE answer_release_id=?",
                    (STATUS_INVALIDATED, item["revision"], text, digest, item["answer_release_id"]),
                )
                updated.append(item)
            if failure_injection == "before_commit":
                raise PromotedLibrarySkillupFlowError("INJECTED_INVALIDATION_FAILURE")
            result = {"invalidated": True, "reason_code": "INVALIDATED", "traces": updated}
            self.connection.execute("INSERT INTO invalidation_requests VALUES(?,?,?,?)", (tenant, key, request_hash, json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":"))))
            self.connection.execute("COMMIT")
            return result
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def serve_trace(
        self, answer_release_id: Any, library_repository: Any, *, tenant_id: Any, organization_id: Any,
    ) -> dict[str, Any]:
        result = self.read_trace(answer_release_id, tenant_id=tenant_id, organization_id=organization_id)
        trace = result.get("answer_release_trace") if result.get("found") is True else None
        if not isinstance(trace, Mapping) or trace.get("status") != STATUS_ACTIVE:
            return {"reusable": False, "reason_code": "EVIDENCE_INVALIDATED", "answer_release_trace": None}
        record_result = library_repository.read_library_record(
            trace["library_record_id"], tenant_id=tenant_id, organization_id=organization_id
        )
        record = record_result.get("library_record") if record_result.get("found") is True else None
        promotion_result = library_repository.read_promotion_trace(
            trace["promotion_trace_id"], tenant_id=tenant_id, organization_id=organization_id
        )
        promotion = promotion_result.get("promotion_trace") if promotion_result.get("found") is True else None
        if not isinstance(record, Mapping) or not isinstance(promotion, Mapping) or promotion.get("status") != TRACE_PROMOTED:
            return {"reusable": False, "reason_code": "EVIDENCE_INVALIDATED", "answer_release_trace": None}
        if record.get("promotion_trace_id") != trace["promotion_trace_id"] or record.get("evidence_id") != trace["evidence_id"]:
            return {"reusable": False, "reason_code": "EVIDENCE_LINK_MISMATCH", "answer_release_trace": None}
        return {"reusable": True, "reason_code": "TRACE_ACTIVE", "answer_release_trace": trace}


def build_feedback_payload(
    *, event_type: str, answer_status: str, tenant_id: str, organization_id: str, request_id: str,
    bridge_trace_id: str, evidence_id: str | None, hold_reason: str | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256((event_type + "\x1f" + request_id + "\x1f" + bridge_trace_id).encode("utf-8")).hexdigest()[:24]
    payload: dict[str, Any] = {
        "schema_version": 1, "contract_version": "1.0.0", "feedback_id": f"feedback:R458:{digest}",
        "request_id": request_id, "tenant_context": {"tenant_id": tenant_id, "organization_id": organization_id},
        "course_context": {"course_id": "course:R458:synthetic", "module_id": "module:R458:synthetic"},
        "event_context": {"event_type": event_type}, "answer_status": answer_status,
        "bridge_trace_id": bridge_trace_id,
        "evidence_context": {"evidence_ids": [evidence_id] if evidence_id else [], "evidence_pointers": [f"urn:qlib:evidence:{digest}"] if evidence_id else [], **({"missing_evidence_reason": "EVIDENCE_INVALIDATED"} if not evidence_id else {})},
        "feedback_policy": {"user_raw_query_stored": False, "raw_answer_stored": False, "internal_path_allowed": False, "secret_surface_allowed": False, "paid_standard_raw_text_allowed": False, "feedback_text_policy": "summary_or_pointer_only", "automation_may_promote_to_library": False, "human_review_required": True},
        "curation_target": "evidence_gap_queue" if answer_status == "HOLD" else "qa_case_candidate",
        "feedback_surface": {"safe_summary": f"R458_{event_type.upper()}_METADATA", "review_pointer": f"urn:qlib:feedback:{digest}"},
    }
    if hold_reason:
        payload["hold_reason"] = hold_reason
    return payload


def validate_feedback_event(payload: Mapping[str, Any], *, failure_injection: bool = False) -> dict[str, Any]:
    candidate = dict(payload)
    if failure_injection:
        candidate["contract_version"] = ""
    result = validate_feedback_queue_contract(candidate)
    if result.get("status") != "READY":
        raise PromotedLibrarySkillupFlowError("FEEDBACK_VALIDATION_FAILED")
    return result


def execute_answer_or_hold(
    *, adapter: LocalPromotedLibraryBridgeAdapter, trace_store: LocalAnswerReleaseTraceStore,
    library_repository: Any, tenant_id: str, organization_id: str, request_id: str,
    library_record_id: str, idempotency_key: str, failure_injection: str | None = None,
) -> dict[str, Any]:
    bridge = adapter.retrieve(
        library_repository, tenant_id=tenant_id, organization_id=organization_id, request_id=request_id,
        purpose=PURPOSE_SKILLUP_ANSWER, library_record_id=library_record_id,
        allowed_rights_status=("owned", "licensed", "permission_granted", "public_reference", "internal_only"),
        max_items=1, role="STUDENT", course_id="course:R458:synthetic", module_id="module:R458:synthetic",
        binding_id="binding:R458:synthetic", failure_injection=failure_injection if failure_injection in {"after_record", "after_evidence"} else None,
    )
    if failure_injection == "skillup":
        raise PromotedLibrarySkillupFlowError("INJECTED_SKILLUP_FAILURE")
    skillup = skillup_answer_from_bridge_response(bridge)
    if skillup.get("answer_status") == "ANSWERED":
        record = library_repository.read_library_record(library_record_id, tenant_id=tenant_id, organization_id=organization_id)["library_record"]
        created = trace_store.create_trace(
            tenant_id=tenant_id, organization_id=organization_id, cohort_id=record["cohort_id"], request_id=request_id,
            library_record=record, bridge_result=bridge, released_answer=skillup["answer"], idempotency_key=idempotency_key,
            failure_injection="after_insert" if failure_injection == "trace_insert" else None,
        )
        feedback = build_feedback_payload(event_type="answer_rendered", answer_status="ANSWERED", tenant_id=tenant_id, organization_id=organization_id, request_id=request_id, bridge_trace_id=bridge["bridge_trace_id"], evidence_id=record["evidence_id"])
        feedback_result = validate_feedback_event(feedback, failure_injection=failure_injection == "feedback")
        return {"result_status": "ANSWERED", "bridge": bridge, "skillup": skillup, "trace_result": created, "feedback_result": feedback_result}
    feedback = build_feedback_payload(event_type="hold_created", answer_status="HOLD", tenant_id=tenant_id, organization_id=organization_id, request_id=request_id, bridge_trace_id=bridge["bridge_trace_id"], evidence_id=None, hold_reason="EVIDENCE_INVALIDATED")
    feedback_result = validate_feedback_event(feedback, failure_injection=failure_injection == "feedback")
    return {"result_status": "HOLD", "bridge": bridge, "skillup": skillup, "trace_result": None, "feedback_result": feedback_result}


def invalidate_after_rollback(
    *, trace_store: LocalAnswerReleaseTraceStore, library_repository: Any, promotion_trace_id: str,
    tenant_id: str, organization_id: str, expected_revision: int, idempotency_key: str,
    failure_injection: str | None = None,
) -> dict[str, Any]:
    promotion = library_repository.read_promotion_trace(promotion_trace_id, tenant_id=tenant_id, organization_id=organization_id)
    payload = promotion.get("promotion_trace") if promotion.get("found") is True else None
    if not isinstance(payload, Mapping) or payload.get("status") != TRACE_ROLLED_BACK:
        raise PromotedLibrarySkillupFlowError("ROLLED_BACK_PROMOTION_TRACE_REQUIRED")
    result = trace_store.invalidate_promotion(
        promotion_trace_id, tenant_id=tenant_id, organization_id=organization_id,
        expected_revision=expected_revision, idempotency_key=idempotency_key,
        failure_injection="before_commit" if failure_injection == "write" else None,
    )
    if result.get("traces"):
        trace = result["traces"][0]
        feedback = build_feedback_payload(event_type="invalidated_answer", answer_status="INVALIDATED", tenant_id=tenant_id, organization_id=organization_id, request_id=trace["request_id"], bridge_trace_id=trace["bridge_trace_id"], evidence_id=trace["evidence_id"])
        result["feedback_result"] = validate_feedback_event(feedback)
    return result


__all__ = [
    "CONTRACT_VERSION", "ENVIRONMENT_LOCAL_NONPRODUCTION", "INVALIDATION_REASON", "LocalAnswerReleaseTraceStore",
    "LocalPromotedLibraryBridgeAdapter", "PromotedLibrarySkillupFlowError", "STATUS_ACTIVE", "STATUS_INVALIDATED",
    "build_feedback_payload", "canonical_payload", "execute_answer_or_hold", "invalidate_after_rollback",
    "validate_answer_release_trace", "validate_feedback_event", "validate_trace_storage_path",
]
