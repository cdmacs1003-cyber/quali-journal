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

ROLE_STUDENT = "student"
ROLE_INSTRUCTOR = "instructor"
ROLE_REVIEWER = "reviewer"
ROLE_ADMIN = "admin"

EVIDENCE_DEPTH_STUDENT_SAFE = "student_safe"
EVIDENCE_DEPTH_INSTRUCTOR_SAFE = "instructor_safe"
EVIDENCE_DEPTH_REVIEW_TRACE_SAFE_METADATA = "review_trace_safe_metadata"
EVIDENCE_DEPTH_AUDIT_TRACE_SAFE_METADATA = "audit_trace_safe_metadata"

EVIDENCE_DEPTH_BY_ROLE = {
    ROLE_STUDENT: EVIDENCE_DEPTH_STUDENT_SAFE,
    ROLE_INSTRUCTOR: EVIDENCE_DEPTH_INSTRUCTOR_SAFE,
    ROLE_REVIEWER: EVIDENCE_DEPTH_REVIEW_TRACE_SAFE_METADATA,
    ROLE_ADMIN: EVIDENCE_DEPTH_AUDIT_TRACE_SAFE_METADATA,
}

_ROLE_ALIASES = {
    "learner": ROLE_STUDENT,
    "student": ROLE_STUDENT,
    "instructor": ROLE_INSTRUCTOR,
    "reviewer": ROLE_REVIEWER,
    "admin": ROLE_ADMIN,
}

_SUPPORTED_EVIDENCE_DEPTHS = set(EVIDENCE_DEPTH_BY_ROLE.values())

_ACTIVE_ENTITLEMENT_STATUSES = {
    "ACTIVE",
    "CURRENT",
    "VALID",
    "LICENSED",
    "ENTITLED",
}

_FORBIDDEN_OUTPUT_REQUEST_MARKERS = (
    "raw_standard_text",
    "raw_text_export",
    "raw_export",
    "paid_standard_raw",
    "raw_paid_standard",
    "raw_instructor_guide",
    "instructor_guide_raw",
    "raw_prompt",
    "internal_path",
    "internal_route",
    "local_path",
    "secret",
    "api_key",
    "private_key",
    "whole_log",
    "admin_screen",
    "private_tacit_knowledge",
)

_REVIEW_TRACE_REQUEST_MARKERS = (
    "review_trace",
    "review_trace_safe_metadata",
)

_AUDIT_TRACE_REQUEST_MARKERS = (
    "audit_trace",
    "audit_trace_safe_metadata",
)

_INSTRUCTOR_GUIDE_SAFE_REQUEST_MARKERS = (
    "instructor_guide_summary",
    "instructor_guide_metadata",
)

_ZERO_LEAK_COUNTERS = {
    "raw_text_export_count": 0,
    "internal_path_leak_count": 0,
    "raw_prompt_output_count": 0,
    "secret_leak_count": 0,
    "instructor_guide_raw_leak_count": 0,
}

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

_BRIDGE_EVIDENCE_FIELD_MAX_LENGTHS = {
    "evidence_id": 120,
    "bridge_trace_id": 160,
    "source_doc_kind": 120,
    "validation_shape_ids": 120,
}

_REDACTED_PREFLIGHT_FIELD = "redacted_preflight_replay_evidence"

_SAFE_METADATA_FIELD_NAMES = {
    "raw_text_policy",
}

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


def zero_leak_counters() -> dict[str, int]:
    return dict(_ZERO_LEAK_COUNTERS)


def normalize_role(value: Any) -> str | None:
    label = _safe_label(value, 64)
    if label is None:
        return None
    token = label.strip().lower().replace("-", "_").replace(" ", "_")
    return _ROLE_ALIASES.get(token)


def evidence_depth_for_role(role: Any) -> str | None:
    normalized = normalize_role(role)
    if normalized is None:
        return None
    return EVIDENCE_DEPTH_BY_ROLE[normalized]


def _context_value(context: Mapping[str, Any], key: str) -> Any:
    value = context.get(key)
    if not _is_missing(value):
        return value
    binding = context.get("course_library_binding")
    if isinstance(binding, Mapping):
        return binding.get(key)
    return value


def _safe_context_label(context: Mapping[str, Any], key: str, max_length: int = 160) -> str | None:
    return _safe_label(_context_value(context, key), max_length)


def _context_text(context: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    values = []
    for key in keys:
        value = _context_value(context, key)
        if not _is_missing(value):
            values.append(str(value))
    return " ".join(values).lower()


def _role_policy_decision(
    status: str,
    reason: str | None,
    *,
    role: str | None,
    evidence_depth: str | None,
) -> dict[str, Any]:
    return {
        "result_status": status,
        "hold_reason": None if status == RESULT_OK else reason,
        "feedback_candidate_required": status != RESULT_OK,
        "role": role,
        "evidence_depth": evidence_depth,
        **zero_leak_counters(),
    }


def _has_mismatch(context: Mapping[str, Any], left_key: str, right_keys: tuple[str, ...]) -> bool:
    left = _safe_context_label(context, left_key)
    if left is None:
        return False
    for key in right_keys:
        right = _safe_context_label(context, key)
        if right is not None and right != left:
            return True
    return False


def _requires_license_entitlement(context: Mapping[str, Any]) -> bool:
    rights = normalize_rights_status(_context_value(context, "rights_status"))
    if rights == RIGHTS_LICENSED:
        return True
    source_kind = _safe_token(_context_value(context, "source_doc_kind"))
    if "PAID_STANDARD" in source_kind or "LICENSED_STANDARD" in source_kind:
        return True
    return _positive(_context_value(context, "license_required")) or _positive(
        _context_value(context, "paid_standard")
    )


def _safe_requested_evidence_depth(context: Mapping[str, Any]) -> str | None:
    value = _context_value(context, "evidence_depth")
    if _is_missing(value):
        return None
    depth = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if depth in _SUPPORTED_EVIDENCE_DEPTHS:
        return depth
    return "UNSUPPORTED"


def decide_role_access_policy(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fail-closed Track A role, scope, license, and output-depth policy."""

    source = context if isinstance(context, Mapping) else {}
    role = normalize_role(_context_value(source, "role"))
    if role is None:
        return _role_policy_decision(
            RESULT_HOLD,
            "HOLD_PERMISSION: explicit supported role is required for Track A protected answer flow",
            role=None,
            evidence_depth=None,
        )

    requested_depth = _safe_requested_evidence_depth(source)
    expected_depth = EVIDENCE_DEPTH_BY_ROLE[role]
    if requested_depth == "UNSUPPORTED":
        return _role_policy_decision(
            RESULT_HOLD,
            "HOLD_POLICY: unsupported evidence_depth",
            role=role,
            evidence_depth=None,
        )
    if requested_depth is not None and requested_depth != expected_depth:
        return _role_policy_decision(
            RESULT_HOLD,
            "HOLD_POLICY: evidence_depth is not allowed for role",
            role=role,
            evidence_depth=expected_depth,
        )

    request_text = _context_text(
        source,
        (
            "requested_output_type",
            "requested_action",
            "action",
            "evidence_depth",
            "trace_view",
            "export_type",
        ),
    )
    if any(marker in request_text for marker in _FORBIDDEN_OUTPUT_REQUEST_MARKERS):
        return _role_policy_decision(
            RESULT_DENIED,
            "HOLD_POLICY: raw, internal, prompt, secret, or admin output is blocked for all roles",
            role=role,
            evidence_depth=expected_depth,
        )

    if any(marker in request_text for marker in _REVIEW_TRACE_REQUEST_MARKERS) and role not in {
        ROLE_REVIEWER,
        ROLE_ADMIN,
    }:
        return _role_policy_decision(
            RESULT_HOLD,
            "HOLD_PERMISSION: review_trace safe metadata is reviewer/admin only",
            role=role,
            evidence_depth=expected_depth,
        )

    if any(marker in request_text for marker in _AUDIT_TRACE_REQUEST_MARKERS) and role != ROLE_ADMIN:
        return _role_policy_decision(
            RESULT_HOLD,
            "HOLD_PERMISSION: audit_trace safe metadata is admin only",
            role=role,
            evidence_depth=expected_depth,
        )

    if any(marker in request_text for marker in _INSTRUCTOR_GUIDE_SAFE_REQUEST_MARKERS) and role == ROLE_STUDENT:
        return _role_policy_decision(
            RESULT_HOLD,
            "HOLD_PERMISSION: instructor guide metadata is not student visible",
            role=role,
            evidence_depth=expected_depth,
        )

    for key in ("course_id", "module_id", "binding_id"):
        if _safe_context_label(source, key) is None:
            return _role_policy_decision(
                RESULT_HOLD,
                f"HOLD_NO_BINDING: {key} is required for Track A protected answer flow",
                role=role,
                evidence_depth=expected_depth,
            )

    for key in ("tenant_id", "organization_id", "cohort_id"):
        if _safe_context_label(source, key) is None:
            return _role_policy_decision(
                RESULT_HOLD,
                f"HOLD_TENANT_BOUNDARY: {key} is required for Track A protected answer flow",
                role=role,
                evidence_depth=expected_depth,
            )

    if _has_mismatch(source, "tenant_id", ("target_tenant_id", "evidence_tenant_id", "license_tenant_id")):
        return _role_policy_decision(
            RESULT_HOLD,
            "HOLD_TENANT_BOUNDARY: tenant scope mismatch",
            role=role,
            evidence_depth=expected_depth,
        )
    if _has_mismatch(
        source,
        "organization_id",
        ("target_organization_id", "evidence_organization_id", "license_organization_id"),
    ):
        return _role_policy_decision(
            RESULT_HOLD,
            "HOLD_TENANT_BOUNDARY: organization scope mismatch",
            role=role,
            evidence_depth=expected_depth,
        )
    if _has_mismatch(source, "cohort_id", ("target_cohort_id", "evidence_cohort_id", "license_cohort_id")):
        return _role_policy_decision(
            RESULT_HOLD,
            "HOLD_TENANT_BOUNDARY: cohort scope mismatch",
            role=role,
            evidence_depth=expected_depth,
        )

    if _requires_license_entitlement(source):
        entitlement_id = _safe_context_label(source, "license_entitlement_id")
        entitlement_status = _safe_token(_context_value(source, "license_entitlement_status"))
        if entitlement_id is None:
            return _role_policy_decision(
                RESULT_HOLD,
                "HOLD_PERMISSION: license_entitlement_id is required for licensed pointer-only access",
                role=role,
                evidence_depth=expected_depth,
            )
        if entitlement_status not in _ACTIVE_ENTITLEMENT_STATUSES:
            return _role_policy_decision(
                RESULT_HOLD,
                "HOLD_LICENSE_EXPIRED: license entitlement is missing, expired, suspended, unknown, or unsupported",
                role=role,
                evidence_depth=expected_depth,
            )

    return _role_policy_decision(
        RESULT_OK,
        None,
        role=role,
        evidence_depth=expected_depth,
    )


def _field_violation_code(field_name: Any) -> str | None:
    text = str(field_name or "").strip()
    lowered = text.lower()
    if lowered == _REDACTED_PREFLIGHT_FIELD:
        return None
    if lowered in _SAFE_METADATA_FIELD_NAMES:
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
                safe_ids = [
                    _safe_label(
                        item,
                        _BRIDGE_EVIDENCE_FIELD_MAX_LENGTHS["validation_shape_ids"],
                    )
                    for item in value
                ]
                projected[field] = [item for item in safe_ids if item is not None]
            continue
        label = _safe_label(value, _BRIDGE_EVIDENCE_FIELD_MAX_LENGTHS.get(field, 240))
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
    enforce_role_access: bool = False,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        return _decision(RESULT_HOLD, "evidence payload is missing or invalid")

    violations = detect_forbidden_fields(evidence)
    if violations:
        return _decision(RESULT_DENIED, "forbidden fields or patterns detected")

    if bool(evidence.get("direct_db_access_attempt")) and str(requester_module).lower() == "skillup":
        return _decision(RESULT_DENIED, "direct DB access is denied for Bridge-only Skillup requests")

    if enforce_role_access:
        role_decision = decide_role_access_policy(evidence)
        if role_decision.get("result_status") != RESULT_OK:
            return role_decision

    current_status = _safe_token(evidence.get("current_status"))
    purpose_token = _safe_token(purpose)
    requester_token = _safe_token(requester_module)
    search_exposure_requested = (
        purpose_token == "SEARCH_EXPOSURE"
        or requester_token == "SEARCH"
        or _positive(evidence.get("search_exposure_requested"))
    )
    if current_status == "QUARANTINED" and search_exposure_requested:
        return _decision(RESULT_HOLD, "QUARANTINED evidence is not available for search exposure")

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
    decision = _decision(RESULT_OK)
    if enforce_role_access:
        decision.update(decide_role_access_policy(evidence))
    return decision


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


def _recorded_boundary_positive(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return False
    normalized = _safe_token(value)
    safe_negative_values = {
        "NO",
        "N",
        "FALSE",
        "0",
        "NONE",
        "NONE_OBSERVED",
        "NOT_RECORDED",
        "NOT_APPLICABLE",
        "ABSENT",
    }
    return normalized not in safe_negative_values


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

    if _recorded_boundary_positive(evidence.get("credential_material_recorded")):
        reason_codes.append("CREDENTIAL_MATERIAL_RECORDED_POSITIVE")
    if _recorded_boundary_positive(evidence.get("password_recorded")):
        reason_codes.append("PASSWORD_RECORDED_POSITIVE")
    if _recorded_boundary_positive(evidence.get("full_connection_string_recorded")):
        reason_codes.append("FULL_CONNECTION_STRING_RECORDED_POSITIVE")
    if _recorded_boundary_positive(evidence.get("secret_store_accessed")):
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
    "EVIDENCE_DEPTH_AUDIT_TRACE_SAFE_METADATA",
    "EVIDENCE_DEPTH_BY_ROLE",
    "EVIDENCE_DEPTH_INSTRUCTOR_SAFE",
    "EVIDENCE_DEPTH_REVIEW_TRACE_SAFE_METADATA",
    "EVIDENCE_DEPTH_STUDENT_SAFE",
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
    "decide_role_access_policy",
    "decide_bridge_result",
    "evidence_depth_for_role",
    "normalize_role",
    "normalize_raw_text_policy",
    "normalize_rights_status",
    "project_bridge_safe_evidence",
    "validate_bridge_safe_response",
    "validate_human_redacted_preflight_replay_evidence",
    "zero_leak_counters",
]
