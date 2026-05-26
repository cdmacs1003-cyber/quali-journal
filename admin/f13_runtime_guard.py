"""Static Bridge/F13 runtime guard helpers.

This module is intentionally local and data-only. It does not open files,
connect to databases, read environment variables, call networks, or execute
subprocesses. The Bridge API passes already-provided evidence into these
helpers and receives a safe decision shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


RESULT_OK = "OK"
RESULT_HOLD = "HOLD"
RESULT_DENIED = "DENIED"

RIGHTS_PUBLIC = "PUBLIC"
RIGHTS_INTERNAL = "INTERNAL"
RIGHTS_LICENSED = "LICENSED"
RIGHTS_CUSTOMER_CONFIDENTIAL = "CUSTOMER_CONFIDENTIAL"
RIGHTS_RESTRICTED = "RESTRICTED"
RIGHTS_UNKNOWN = "UNKNOWN"
RIGHTS_NOT_VERIFIED = "NOT_VERIFIED"

RAW_TEXT_POLICY_SUMMARY_ONLY = "SUMMARY_ONLY"
RAW_TEXT_POLICY_POINTER_ONLY = "POINTER_ONLY"
RAW_TEXT_POLICY_REDACTED_SUMMARY_ONLY = "REDACTED_SUMMARY_ONLY"
RAW_TEXT_POLICY_DENIED = "DENIED"
RAW_TEXT_POLICY_NOT_VERIFIED = "NOT_VERIFIED"

BRIDGE_EVIDENCE_ALLOWLIST_FIELDS = {
    "evidence_id",
    "bridge_trace_id",
    "safe_summary",
    "pointer_uri",
    "raw_text_policy",
    "rights_status",
    "source_doc_kind",
    "validation_shape_ids",
}

_REDACTED_PREFLIGHT_FIELD = "redacted_preflight_replay_evidence"

_RAW_FIELD_MARKERS = {
    "raw_text",
    "raw text",
    "raw_text_ref",
    "raw_pointer",
    "raw_source_text",
    "full_text",
    "full_source_text",
    "original_text",
    "source_text",
}

_INTERNAL_FIELD_MARKERS = {
    "internal_path",
    "file_path",
    "local_path",
    "source_uri_or_path",
    "direct_db_row",
    "warehouse_internal_object",
    "library_internal_object",
}

_SENSITIVE_FIELD_MARKERS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
}

_DB_FIELD_MARKERS = {
    "dsn",
    "database_url",
    "db_url",
    "connection_string",
    "direct_db_access_attempt",
}

_INTERNAL_PATH_VALUE_MARKERS = (
    "h:\\",
    "c:\\",
    "/mnt/",
    "/home/",
    "/tmp/",
)

_DB_VALUE_MARKERS = (
    "postgres://",
    "postgresql://",
    "mysql://",
    "sqlite://",
)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _safe_token(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def normalize_rights_status(value: Any) -> str:
    normalized = _safe_token(value)
    mapping = {
        "PUBLIC": RIGHTS_PUBLIC,
        "PUBLIC_REFERENCE": RIGHTS_PUBLIC,
        "OPEN": RIGHTS_PUBLIC,
        "OWNED": RIGHTS_INTERNAL,
        "INTERNAL": RIGHTS_INTERNAL,
        "INTERNAL_ONLY": RIGHTS_INTERNAL,
        "LICENSED": RIGHTS_LICENSED,
        "PERMISSION_GRANTED": RIGHTS_LICENSED,
        "CUSTOMER_CONFIDENTIAL": RIGHTS_CUSTOMER_CONFIDENTIAL,
        "CUSTOMER": RIGHTS_CUSTOMER_CONFIDENTIAL,
        "RESTRICTED": RIGHTS_RESTRICTED,
        "NO_EXPORT": RIGHTS_RESTRICTED,
        "UNKNOWN": RIGHTS_UNKNOWN,
    }
    if normalized == "":
        return RIGHTS_NOT_VERIFIED
    return mapping.get(normalized, RIGHTS_NOT_VERIFIED)


def normalize_raw_text_policy(value: Any) -> str:
    normalized = _safe_token(value)
    mapping = {
        "SUMMARY_ONLY": RAW_TEXT_POLICY_SUMMARY_ONLY,
        "SAFE_SUMMARY_ONLY": RAW_TEXT_POLICY_SUMMARY_ONLY,
        "POINTER_ONLY": RAW_TEXT_POLICY_POINTER_ONLY,
        "REDACTED_SUMMARY_ONLY": RAW_TEXT_POLICY_REDACTED_SUMMARY_ONLY,
        "DENIED": RAW_TEXT_POLICY_DENIED,
        "NO_RAW_TEXT": RAW_TEXT_POLICY_DENIED,
        "NOT_VERIFIED": RAW_TEXT_POLICY_NOT_VERIFIED,
    }
    if normalized == "":
        return RAW_TEXT_POLICY_NOT_VERIFIED
    return mapping.get(normalized, RAW_TEXT_POLICY_NOT_VERIFIED)


def _safe_label(value: Any, max_length: int = 240) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if len(text) > max_length:
        return None
    if any(ord(char) < 32 for char in text):
        return None
    if _value_violation_code(text) is not None:
        return None
    return text


def _field_violation_code(field_name: Any) -> str | None:
    text = str(field_name or "").strip()
    lowered = text.lower()
    if lowered == _REDACTED_PREFLIGHT_FIELD:
        return None
    for marker in _RAW_FIELD_MARKERS:
        if marker in lowered:
            return text or "raw_marker"
    for marker in _INTERNAL_FIELD_MARKERS:
        if marker in lowered:
            return text or "internal_marker"
    for marker in _DB_FIELD_MARKERS:
        if marker in lowered:
            return text or "db_marker"
    for marker in _SENSITIVE_FIELD_MARKERS:
        if marker in lowered:
            return text or "sensitive_marker"
    return None


def _value_violation_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.lower()
    if "h:\\장기기억" in lowered:
        return "h_drive_janggigieok_path"
    for marker in _INTERNAL_PATH_VALUE_MARKERS:
        if marker in lowered:
            return "internal_path_marker"
    for marker in _DB_VALUE_MARKERS:
        if marker in lowered:
            return "db_or_dsn_marker"
    return None


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _walk_forbidden(value: Any, *, skip_values: bool = False) -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) == _REDACTED_PREFLIGHT_FIELD:
                continue
            code = _field_violation_code(key)
            if code is not None:
                findings.append(code)
            findings.extend(_walk_forbidden(child, skip_values=skip_values))
        return findings
    if isinstance(value, list | tuple | set):
        for child in value:
            findings.extend(_walk_forbidden(child, skip_values=skip_values))
        return findings
    if not skip_values:
        code = _value_violation_code(value)
        if code is not None:
            findings.append(code)
    return findings


def detect_forbidden_fields(payload: Any) -> list[str]:
    """Return safe field names or reason codes for forbidden Bridge payload data."""

    return _dedupe(_walk_forbidden(payload))


def project_bridge_safe_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    field_violations = set(_walk_forbidden(evidence, skip_values=True))
    for field in BRIDGE_EVIDENCE_ALLOWLIST_FIELDS:
        if field in field_violations or field not in evidence:
            continue
        value = evidence.get(field)
        if field == "rights_status":
            projected[field] = normalize_rights_status(value)
            continue
        if field == "raw_text_policy":
            projected[field] = normalize_raw_text_policy(value)
            continue
        if field == "validation_shape_ids":
            if isinstance(value, list):
                safe_ids = [_safe_label(item, 120) for item in value]
                projected[field] = [item for item in safe_ids if item is not None]
            continue
        label = _safe_label(value)
        if label is not None:
            projected[field] = label
    return projected


def _decision(status: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "result_status": status,
        "hold_reason": None if status == RESULT_OK else reason,
        "feedback_candidate_required": status != RESULT_OK,
    }


def decide_bridge_result(
    evidence: Mapping[str, Any] | None,
    *,
    requester_module: str = "Skillup",
    purpose: str = "answer",
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        return _decision(RESULT_HOLD, "evidence payload is missing or invalid")

    violations = detect_forbidden_fields(evidence)
    if violations:
        return _decision(RESULT_DENIED, "forbidden fields or patterns detected")

    if bool(evidence.get("direct_db_access_attempt")) and str(requester_module).lower() == "skillup":
        return _decision(RESULT_DENIED, "direct DB access is denied for Bridge-only Skillup requests")

    if _is_missing(evidence.get("evidence_id")):
        return _decision(RESULT_HOLD, "missing evidence_id")
    if _is_missing(evidence.get("safe_summary")):
        return _decision(RESULT_HOLD, "missing safe_summary")

    rights = normalize_rights_status(evidence.get("rights_status"))
    raw_policy = normalize_raw_text_policy(evidence.get("raw_text_policy"))

    if rights == RIGHTS_RESTRICTED:
        return _decision(RESULT_DENIED, "RESTRICTED rights_status is not Bridge-safe")
    if rights in {RIGHTS_UNKNOWN, RIGHTS_NOT_VERIFIED}:
        return _decision(RESULT_HOLD, "rights_status is not verified")
    if raw_policy == RAW_TEXT_POLICY_DENIED:
        return _decision(RESULT_DENIED, "raw_text_policy denies Bridge output")
    if raw_policy == RAW_TEXT_POLICY_NOT_VERIFIED:
        return _decision(RESULT_HOLD, "raw_text_policy is not verified")
    if rights == RIGHTS_CUSTOMER_CONFIDENTIAL:
        if raw_policy == RAW_TEXT_POLICY_REDACTED_SUMMARY_ONLY and evidence.get("redaction_approved") is True:
            return _decision(RESULT_OK)
        return _decision(RESULT_DENIED, "customer confidential evidence requires approved redaction")
    if rights == RIGHTS_LICENSED and raw_policy == RAW_TEXT_POLICY_POINTER_ONLY:
        if _safe_label(evidence.get("pointer_uri")) is None:
            return _decision(RESULT_HOLD, "licensed pointer-only evidence requires safe pointer_uri")
    return _decision(RESULT_OK)


def validate_bridge_safe_response(response: Mapping[str, Any]) -> dict[str, Any]:
    violations = detect_forbidden_fields(response)
    if response.get("raw_text_included") is True:
        violations.append("raw_text_included")
    if response.get("internal_path_included") is True:
        violations.append("internal_path_included")
    violations = _dedupe(violations)
    return {
        "is_safe": not violations,
        "result_status": RESULT_OK if not violations else RESULT_DENIED,
        "violations": violations,
    }


def _positive(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = _safe_token(value)
    return normalized in {"YES", "Y", "TRUE", "1", "RECORDED", "ATTEMPTED"}


def _contains(value: Any, needle: str) -> bool:
    return needle.lower() in str(value or "").lower()


def validate_human_redacted_preflight_replay_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    residuals: list[str] = []
    prohibited_actions = [
        "CODEX_DB_CONNECTION",
        "CODEX_SECRET_INSPECTION",
        "CODEX_ENV_ACCESS",
        "CODEX_RUNTIME_SMOKE",
    ]

    if _positive(evidence.get("credential_material_recorded")):
        reason_codes.append("CREDENTIAL_MATERIAL_RECORDED_POSITIVE")
    if _positive(evidence.get("password_recorded")):
        reason_codes.append("PASSWORD_RECORDED_POSITIVE")
    if _positive(evidence.get("full_connection_string_recorded")):
        reason_codes.append("FULL_CONNECTION_STRING_RECORDED_POSITIVE")
    if _positive(evidence.get("secret_store_accessed")):
        reason_codes.append("SECRET_STORE_ACCESSED_POSITIVE")
    if reason_codes:
        return {
            "result_status": RESULT_DENIED,
            "ok": False,
            "status": "DENY_SECRET_BOUNDARY_RISK",
            "reason_codes": reason_codes,
            "residuals": residuals,
            "prohibited_actions": prohibited_actions,
        }

    write_codes = []
    if _positive(evidence.get("DB_write_attempted")):
        write_codes.append("DB_WRITE_ATTEMPTED_POSITIVE")
    if _positive(evidence.get("migration_executed_in_replay")):
        write_codes.append("MIGRATION_EXECUTED_IN_REPLAY_POSITIVE")
    if _positive(evidence.get("rollback_executed_in_replay")):
        write_codes.append("ROLLBACK_EXECUTED_IN_REPLAY_POSITIVE")
    if write_codes:
        return {
            "result_status": RESULT_DENIED,
            "ok": False,
            "status": "DENY_WRITE_OR_MIGRATION_BOUNDARY_RISK",
            "reason_codes": write_codes,
            "residuals": residuals,
            "prohibited_actions": prohibited_actions,
        }

    if str(evidence.get("connected_user_result") or "") != "f13_readonly":
        reason_codes.append("CONNECTED_USER_RESULT_MISMATCH")
    if str(evidence.get("target_database") or "") != "quali_journal_f13_dev":
        reason_codes.append("TARGET_DATABASE_MISMATCH")
    if str(evidence.get("connected_database_result") or "") != "quali_journal_f13_dev":
        reason_codes.append("CONNECTED_DATABASE_RESULT_MISMATCH")
    if _positive(evidence.get("can_insert")):
        reason_codes.append("CAN_INSERT_MISMATCH")
    if _positive(evidence.get("can_update")):
        reason_codes.append("CAN_UPDATE_MISMATCH")
    if _positive(evidence.get("can_delete")):
        reason_codes.append("CAN_DELETE_MISMATCH")

    table_checked = evidence.get("table_checked")
    table_exists = evidence.get("table_exists_result")
    target_confirmed = _contains(table_checked, "public.f13_feedback_queue_items") or _contains(
        table_exists,
        "public.f13_feedback_queue_items",
    )
    public_feedback_only = _contains(table_checked, "public.feedback_queue") and not target_confirmed
    if public_feedback_only:
        reason_codes.append("PUBLIC_FEEDBACK_QUEUE_PRESENT_WITHOUT_ACCEPTED_TABLE_CONFIRMATION")

    replay_time = str(evidence.get("replay_datetime_local") or "")
    if "KST" in replay_time and "exact time not recorded" not in replay_time:
        residuals.append("TIME_EXACT_NOT_RECORDED_DATE_KST_ONLY")

    if reason_codes:
        return {
            "result_status": RESULT_HOLD,
            "ok": False,
            "status": "HOLD_HUMAN_REDACTED_PREFLIGHT_REPLAY_EVIDENCE_REVIEW_REQUIRED",
            "reason_codes": reason_codes,
            "residuals": residuals,
            "prohibited_actions": prohibited_actions,
        }

    return {
        "result_status": RESULT_OK,
        "ok": True,
        "status": "PASS_HUMAN_REDACTED_PREFLIGHT_REPLAY_EVIDENCE_ACCEPTED_FOR_LOCAL_REPLAY_GATE_ONLY",
        "reason_codes": [],
        "residuals": residuals,
        "accepted_scope": {
            "target_table": "public.f13_feedback_queue_items",
            "codex_live_db_verification": "NOT_EXECUTED",
            "runtime_smoke": "NOT_EXECUTED",
        },
        "prohibited_actions": prohibited_actions,
    }


__all__ = [
    "BRIDGE_EVIDENCE_ALLOWLIST_FIELDS",
    "RAW_TEXT_POLICY_DENIED",
    "RAW_TEXT_POLICY_NOT_VERIFIED",
    "RAW_TEXT_POLICY_POINTER_ONLY",
    "RAW_TEXT_POLICY_REDACTED_SUMMARY_ONLY",
    "RAW_TEXT_POLICY_SUMMARY_ONLY",
    "RESULT_DENIED",
    "RESULT_HOLD",
    "RESULT_OK",
    "RIGHTS_CUSTOMER_CONFIDENTIAL",
    "RIGHTS_INTERNAL",
    "RIGHTS_LICENSED",
    "RIGHTS_NOT_VERIFIED",
    "RIGHTS_PUBLIC",
    "RIGHTS_RESTRICTED",
    "RIGHTS_UNKNOWN",
    "detect_forbidden_fields",
    "decide_bridge_result",
    "normalize_raw_text_policy",
    "normalize_rights_status",
    "project_bridge_safe_evidence",
    "validate_bridge_safe_response",
    "validate_human_redacted_preflight_replay_evidence",
]
