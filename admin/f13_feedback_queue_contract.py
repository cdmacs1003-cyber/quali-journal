from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RESULT_READY = "READY"
RESULT_HOLD = "HOLD"
RESULT_INVALID = "INVALID"

_ALLOWED_EVENT_TYPES = {
    "answer_rendered",
    "hold_created",
    "feedback_submitted",
    "correction_requested",
    "evidence_gap_reported",
    "invalidated_answer",
    "redacted_answer",
}
_ALLOWED_ANSWER_STATUSES = {
    "ANSWERED",
    "HOLD",
    "REDACTED",
    "INVALIDATED",
}
_ALLOWED_CURATION_TARGETS = {
    "intake_candidate",
    "review_queue",
    "evidence_gap_queue",
    "qa_case_candidate",
    "failure_record_candidate",
}
_NON_CLAIMS = [
    "ANSWER_QUALITY_PASS_NOT_INFERRED",
    "BRIDGE_HEALTH_PASS_NOT_INFERRED",
    "SKILLUP_MVP_PASS_NOT_INFERRED",
    "TRACK_A_PASS_NOT_INFERRED",
    "BETA_PASS_NOT_INFERRED",
    "F13_PASS_NOT_INFERRED",
    "RELEASE_READINESS_NOT_INFERRED",
    "DEPLOYMENT_READINESS_NOT_INFERRED",
    "FEEDBACK_QUEUE_PASS_NOT_INFERRED",
    "SELECTED_STATIC_FEEDBACK_QUEUE_READINESS_ONLY",
]
_SAFE_FIELD_KEYS = {
    "answer_status",
    "feedback_text_policy",
    "user_raw_query_stored",
    "raw_answer_stored",
    "internal_path_allowed",
    "secret_surface_allowed",
    "paid_standard_raw_text_allowed",
    "raw_user_query_included",
    "raw_answer_included",
    "internal_path_included",
    "secret_surface_included",
    "paid_standard_raw_text_included",
}
_RAW_QUERY_FIELD_MARKERS = (
    "raw_query",
    "raw query",
    "raw prompt",
    "raw_prompt",
    "raw_user_query",
    "user_raw_query",
    "full_query",
)
_RAW_ANSWER_FIELD_MARKERS = (
    "raw_answer",
    "raw answer",
    "full_answer",
    "answer_raw",
    "raw_response",
    "full_response",
)
_PAID_STANDARD_FIELD_MARKERS = (
    "paid_standard_raw",
    "paid standard raw",
    "raw_standard_text",
    "raw standard text",
    "paid_standard_text",
)
_INTERNAL_FIELD_MARKERS = (
    "internal_path",
    "internal path",
    "local_path",
    "local path",
    "source_uri_or_path",
    "file_path",
    "file://",
)
_SECRET_FIELD_MARKERS = (
    "api_key",
    "api key",
    "authorization",
    "bearer ",
    "credential",
    "password",
    "private_key",
    "private key",
    "secret",
    "token",
)
_INTERNAL_VALUE_MARKERS = (
    "h:\\",
    "c:\\",
    "/mnt/",
    "/home/",
    "/tmp/",
    "file://",
    "localhost",
    "127.0.0.1",
)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _has_reference(value: Any) -> bool:
    if isinstance(value, list | tuple | set):
        return any(not _is_missing(item) for item in value)
    return not _is_missing(value)


def _normal_upper(value: Any, fallback: str = "") -> str:
    if _is_missing(value):
        return fallback
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def _normal_lower(value: Any, fallback: str = "") -> str:
    if _is_missing(value):
        return fallback
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _policy_source(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("feedback_policy", "output_policy", "policy"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _policy_value(payload: Mapping[str, Any], policy: Mapping[str, Any], key: str) -> Any:
    if key in payload:
        return payload.get(key)
    return policy.get(key)


def _append_marker(findings: list[str], code: str) -> None:
    if code not in findings:
        findings.append(code)


def _field_marker_code(key: Any) -> str | None:
    lowered = str(key).lower()
    if lowered in _SAFE_FIELD_KEYS:
        return None
    if any(marker in lowered for marker in _RAW_QUERY_FIELD_MARKERS):
        return "RAW_USER_QUERY_SURFACE"
    if any(marker in lowered for marker in _RAW_ANSWER_FIELD_MARKERS):
        return "RAW_ANSWER_SURFACE"
    if any(marker in lowered for marker in _PAID_STANDARD_FIELD_MARKERS):
        return "PAID_STANDARD_RAW_TEXT_SURFACE"
    if any(marker in lowered for marker in _INTERNAL_FIELD_MARKERS):
        return "INTERNAL_PATH_SURFACE"
    if any(marker in lowered for marker in _SECRET_FIELD_MARKERS):
        return "SECRET_LIKE_SURFACE"
    return None


def _value_marker_code(value: Any) -> str | None:
    lowered = str(value or "").lower()
    if any(marker in lowered for marker in _RAW_QUERY_FIELD_MARKERS):
        return "RAW_USER_QUERY_SURFACE"
    if any(marker in lowered for marker in _RAW_ANSWER_FIELD_MARKERS):
        return "RAW_ANSWER_SURFACE"
    if any(marker in lowered for marker in _PAID_STANDARD_FIELD_MARKERS):
        return "PAID_STANDARD_RAW_TEXT_SURFACE"
    if any(marker in lowered for marker in _INTERNAL_FIELD_MARKERS):
        return "INTERNAL_PATH_SURFACE"
    if any(marker in lowered for marker in _INTERNAL_VALUE_MARKERS):
        return "INTERNAL_PATH_SURFACE"
    if any(marker in lowered for marker in _SECRET_FIELD_MARKERS):
        return "SECRET_LIKE_SURFACE"
    return None


def _surface_findings(value: Any) -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            code = _field_marker_code(key)
            if code is not None:
                _append_marker(findings, code)
            for child_code in _surface_findings(child):
                _append_marker(findings, child_code)
        return findings
    if isinstance(value, list | tuple | set):
        for child in value:
            for child_code in _surface_findings(child):
                _append_marker(findings, child_code)
        return findings
    if isinstance(value, str):
        code = _value_marker_code(value)
        if code is not None:
            _append_marker(findings, code)
    return findings


def _new_counters(findings: list[str]) -> dict[str, int]:
    return {
        "raw_user_query_surface_count": 1 if "RAW_USER_QUERY_SURFACE" in findings else 0,
        "raw_answer_surface_count": 1 if "RAW_ANSWER_SURFACE" in findings else 0,
        "internal_path_surface_count": 1 if "INTERNAL_PATH_SURFACE" in findings else 0,
        "secret_like_surface_count": 1 if "SECRET_LIKE_SURFACE" in findings else 0,
        "paid_standard_raw_text_surface_count": 1 if "PAID_STANDARD_RAW_TEXT_SURFACE" in findings else 0,
    }


def _result(
    *,
    status: str,
    errors: list[str],
    warnings: list[str],
    counters: dict[str, int],
    checks: dict[str, bool],
) -> dict[str, Any]:
    ready = status == RESULT_READY
    return {
        "status": status,
        "queue_ready": ready,
        "hold_reason": None if ready else (errors[0] if errors else None),
        "errors": errors,
        "warnings": warnings,
        "counters": counters,
        "checks": checks,
        "non_claims": list(_NON_CLAIMS),
        "raw_user_query_included": False,
        "raw_answer_included": False,
        "internal_path_included": False,
        "secret_surface_included": False,
        "paid_standard_raw_text_included": False,
        "db_access_executed": False,
        "network_access_executed": False,
        "runtime_access_executed": False,
        "file_io_executed": False,
        "env_access_executed": False,
        "subprocess_executed": False,
    }


def validate_feedback_queue_contract(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return _result(
            status=RESULT_INVALID,
            errors=["payload must be a mapping"],
            warnings=[],
            counters=_new_counters([]),
            checks={
                "payload_is_mapping": False,
                "selected_static_feedback_queue_readiness_only": True,
            },
        )

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {"payload_is_mapping": True}
    invalid = False

    schema_version = payload.get("schema_version")
    checks["schema_version_valid"] = schema_version == 1 or str(schema_version).strip() == "1"
    if _is_missing(schema_version):
        errors.append("schema_version is required")
        invalid = True
    elif not checks["schema_version_valid"]:
        errors.append("schema_version must equal 1")
        invalid = True

    checks["contract_version_present"] = not _is_missing(payload.get("contract_version"))
    if not checks["contract_version_present"]:
        errors.append("contract_version is required")
        invalid = True

    feedback_or_request_id_present = not _is_missing(payload.get("feedback_id")) or not _is_missing(
        payload.get("request_id")
    )
    checks["feedback_or_request_id_present"] = feedback_or_request_id_present
    if not feedback_or_request_id_present:
        errors.append("feedback_id or request_id is required")
        invalid = True

    tenant_context = payload.get("tenant_context")
    checks["tenant_context_present"] = isinstance(tenant_context, Mapping)
    if not isinstance(tenant_context, Mapping):
        errors.append("tenant_context is required")
        invalid = True
        tenant_id_present = False
        organization_id_present = False
    else:
        tenant_id_present = not _is_missing(tenant_context.get("tenant_id"))
        organization_id_present = not _is_missing(tenant_context.get("organization_id"))
        if not tenant_id_present:
            errors.append("tenant_context.tenant_id is required")
            invalid = True
        if not organization_id_present:
            errors.append("tenant_context.organization_id is required")
            invalid = True
    checks["tenant_id_present"] = tenant_id_present
    checks["organization_id_present"] = organization_id_present

    course_context = payload.get("course_context")
    checks["course_context_present"] = isinstance(course_context, Mapping)
    if not isinstance(course_context, Mapping):
        errors.append("HOLD_COURSE_CONTEXT_REQUIRED")
        course_traceable = False
    else:
        course_traceable = not _is_missing(course_context.get("course_id")) or not _is_missing(
            course_context.get("module_id")
        )
        if not course_traceable:
            errors.append("HOLD_COURSE_OR_MODULE_REQUIRED")
    checks["course_or_module_present"] = course_traceable

    event_context = payload.get("event_context")
    checks["event_context_present"] = isinstance(event_context, Mapping)
    if not isinstance(event_context, Mapping):
        errors.append("event_context is required")
        invalid = True
        event_type = ""
    else:
        event_type = _normal_lower(event_context.get("event_type"))
        if not event_type:
            errors.append("event_context.event_type is required")
            invalid = True
        elif event_type not in _ALLOWED_EVENT_TYPES:
            errors.append("event_context.event_type is not supported")
            invalid = True
    checks["event_type_supported"] = event_type in _ALLOWED_EVENT_TYPES

    event_answer_status = event_context.get("answer_status") if isinstance(event_context, Mapping) else None
    answer_status = _normal_upper(payload.get("answer_status") or event_answer_status)
    checks["answer_status_supported"] = answer_status in _ALLOWED_ANSWER_STATUSES
    if not answer_status:
        errors.append("answer_status is required")
        invalid = True
    elif answer_status not in _ALLOWED_ANSWER_STATUSES:
        errors.append("answer_status is not supported")
        invalid = True

    trace_id_present = not _is_missing(payload.get("bridge_trace_id")) or not _is_missing(payload.get("trace_id"))
    checks["trace_present"] = trace_id_present
    if not trace_id_present:
        errors.append("HOLD_TRACE_REQUIRED_FOR_FEEDBACK")

    evidence_context = payload.get("evidence_context")
    checks["evidence_context_present"] = isinstance(evidence_context, Mapping)
    if not isinstance(evidence_context, Mapping):
        errors.append("evidence_context is required")
        invalid = True
        evidence_or_reason_present = False
    else:
        evidence_or_reason_present = (
            _has_reference(evidence_context.get("evidence_ids"))
            or _has_reference(evidence_context.get("evidence_pointers"))
            or not _is_missing(evidence_context.get("missing_evidence_reason"))
        )
        if not evidence_or_reason_present:
            errors.append("HOLD_EVIDENCE_OR_MISSING_REASON_REQUIRED")
    checks["evidence_or_missing_reason_present"] = evidence_or_reason_present

    hold_reason_present = not _is_missing(payload.get("hold_reason")) or (
        isinstance(event_context, Mapping) and not _is_missing(event_context.get("hold_reason"))
    )
    checks["hold_reason_present_when_required"] = answer_status != "HOLD" or hold_reason_present
    if answer_status == "HOLD" and not hold_reason_present:
        errors.append("HOLD_REASON_REQUIRED_FOR_HOLD_STATUS")

    policy = _policy_source(payload)
    user_raw_query_blocked = _policy_value(payload, policy, "user_raw_query_stored") is False
    raw_answer_blocked = _policy_value(payload, policy, "raw_answer_stored") is False
    internal_path_blocked = _policy_value(payload, policy, "internal_path_allowed") is False
    secret_surface_blocked = _policy_value(payload, policy, "secret_surface_allowed") is False
    paid_standard_raw_blocked = _policy_value(payload, policy, "paid_standard_raw_text_allowed") is False
    feedback_text_policy = _normal_lower(_policy_value(payload, policy, "feedback_text_policy"))
    automation_promotion_blocked = _policy_value(payload, policy, "automation_may_promote_to_library") is False
    human_review_required = _policy_value(payload, policy, "human_review_required") is True
    checks.update(
        {
            "user_raw_query_stored_false": user_raw_query_blocked,
            "raw_answer_stored_false": raw_answer_blocked,
            "internal_path_allowed_false": internal_path_blocked,
            "secret_surface_allowed_false": secret_surface_blocked,
            "paid_standard_raw_text_allowed_false": paid_standard_raw_blocked,
            "feedback_text_policy_summary_or_pointer_only": feedback_text_policy == "summary_or_pointer_only",
            "automation_may_promote_to_library_false": automation_promotion_blocked,
            "human_review_required": human_review_required,
        }
    )
    if not user_raw_query_blocked:
        errors.append("HOLD_USER_RAW_QUERY_STORAGE_BLOCK_REQUIRED")
    if not raw_answer_blocked:
        errors.append("HOLD_RAW_ANSWER_STORAGE_BLOCK_REQUIRED")
    if not internal_path_blocked:
        errors.append("HOLD_INTERNAL_PATH_BLOCK_REQUIRED")
    if not secret_surface_blocked:
        errors.append("HOLD_SECRET_SURFACE_BLOCK_REQUIRED")
    if not paid_standard_raw_blocked:
        errors.append("HOLD_PAID_STANDARD_RAW_TEXT_BLOCK_REQUIRED")
    if feedback_text_policy != "summary_or_pointer_only":
        errors.append("HOLD_FEEDBACK_TEXT_POLICY_SUMMARY_OR_POINTER_ONLY_REQUIRED")
    if not automation_promotion_blocked:
        errors.append("HOLD_AUTOMATION_LIBRARY_PROMOTION_BLOCK_REQUIRED")
    if not human_review_required:
        errors.append("HOLD_HUMAN_REVIEW_REQUIRED")

    curation_context = payload.get("curation_context")
    curation_context_target = curation_context.get("target") if isinstance(curation_context, Mapping) else None
    curation_target = _normal_lower(payload.get("curation_target") or curation_context_target)
    checks["curation_target_supported"] = curation_target in _ALLOWED_CURATION_TARGETS
    if not curation_target:
        errors.append("curation_target is required")
        invalid = True
    elif curation_target not in _ALLOWED_CURATION_TARGETS:
        errors.append("curation_target is not supported")
        invalid = True

    output_findings = _surface_findings(payload)
    checks["raw_user_query_absent"] = "RAW_USER_QUERY_SURFACE" not in output_findings
    checks["raw_answer_absent"] = "RAW_ANSWER_SURFACE" not in output_findings
    checks["internal_path_absent"] = "INTERNAL_PATH_SURFACE" not in output_findings
    checks["secret_like_absent"] = "SECRET_LIKE_SURFACE" not in output_findings
    checks["paid_standard_raw_text_absent"] = "PAID_STANDARD_RAW_TEXT_SURFACE" not in output_findings
    for finding in output_findings:
        errors.append(f"HOLD_UNSAFE_FEEDBACK_SURFACE_{finding}")

    checks["selected_static_feedback_queue_readiness_only"] = True
    checks["no_file_io"] = True
    checks["no_env_access"] = True
    checks["no_subprocess"] = True
    checks["no_network"] = True
    checks["no_db"] = True
    checks["no_runtime"] = True

    if invalid:
        status = RESULT_INVALID
    elif errors:
        status = RESULT_HOLD
    else:
        status = RESULT_READY

    return _result(
        status=status,
        errors=errors,
        warnings=warnings,
        counters=_new_counters(output_findings),
        checks=checks,
    )


__all__ = [
    "RESULT_HOLD",
    "RESULT_INVALID",
    "RESULT_READY",
    "validate_feedback_queue_contract",
]
