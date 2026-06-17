from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


RESULT_BOUND = "BOUND"
RESULT_DENIED = "DENIED"
RESULT_HOLD = "HOLD"
RAW_TEXT_POLICY_SUMMARY_ONLY = "SUMMARY_ONLY"
RAW_TEXT_POLICY_POINTER_ONLY = "POINTER_ONLY"
CREATED_AT = "1970-01-01T00:00:00Z"

_DENIED_RIGHTS = {"DENIED", "RESTRICTED", "PRIVATE", "PROPRIETARY"}
_HOLD_RIGHTS = {"", "UNKNOWN", "NOT_VERIFIED"}
_ACTIVE_ENTITLEMENT_STATUSES = {"ACTIVE", "CURRENT", "VALID", "LICENSED", "ENTITLED"}
_LIBRARY_BINDABLE_STATUS = "APPROVED_FOR_LIBRARY"
_UNSAFE_FIELD_MARKERS = (
    "raw_text",
    "raw_prompt",
    "raw_query",
    "full_source_text",
    "internal_path",
    "local_path",
    "secret",
    "token",
    "credential",
)
_UNSAFE_VALUE_MARKERS = (
    "raw text",
    "raw prompt",
    "raw query",
    "full source text",
    "h:\\",
    "c:\\",
    "file://",
    "secret",
    "token",
    "credential",
)


def _safe_text(value: Any, fallback: str, max_length: int = 160) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text[:max_length]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _safe_token(value: Any, fallback: str, max_length: int = 120) -> str:
    text = _safe_text(value, fallback, max_length=max_length)
    token = "".join(ch for ch in text if ch.isalnum() or ch in ":._-")
    return token or fallback


def _stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(_safe_text(part, "", max_length=240) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _contains_unsafe_surface(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered_key = str(key).lower()
            if lowered_key in {"raw_text_policy", "raw_text_included", "internal_path_included"}:
                continue
            if any(marker in lowered_key for marker in _UNSAFE_FIELD_MARKERS):
                return True
            if _contains_unsafe_surface(child):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_unsafe_surface(child) for child in value)
    if isinstance(value, str):
        lowered_value = value.lower()
        return any(marker in lowered_value for marker in _UNSAFE_VALUE_MARKERS)
    return False


def _normal_rights(value: Any) -> str:
    return str(value or "UNKNOWN").strip().upper() or "UNKNOWN"


def _normal_status(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _safe_scope_value(source: Mapping[str, Any], key: str) -> str:
    return _safe_token(source.get(key), "", max_length=120)


def _scope_mismatch(source: Mapping[str, Any], key: str, compare_key: str) -> bool:
    left = _safe_scope_value(source, key)
    right = _safe_scope_value(source, compare_key)
    return bool(left and right and left != right)


def _has_shape_pass(source: Mapping[str, Any]) -> bool:
    shape_ids = source.get("validation_shape_ids")
    if isinstance(shape_ids, list | tuple | set):
        return any(_safe_token(item, "") for item in shape_ids)
    shape_status = _normal_status(source.get("shape_validation_status") or source.get("shape_status"))
    return shape_status == "PASS" or source.get("shape_pass") is True


def _feedback_queue_item(
    *,
    course_id: str,
    module_ref: str,
    evidence_ref: str,
    feedback_type: str,
    suspected_issue: str,
) -> dict[str, Any]:
    digest = _stable_digest(course_id, module_ref, evidence_ref, feedback_type, suspected_issue)
    return {
        "feedback_id": f"fbq:course-library:{digest}",
        "origin_module": "course_library_binding",
        "origin_event_id": f"binding:{course_id}:{module_ref}",
        "feedback_type": feedback_type,
        "user_visible_text_policy": RAW_TEXT_POLICY_SUMMARY_ONLY,
        "linked_answer_id": "answer:pending",
        "linked_evidence_id": evidence_ref,
        "suspected_issue": suspected_issue,
        "proposed_candidate_type": "COURSE_LIBRARY_BINDING_REVIEW",
        "current_status": "review_required",
        "created_at": CREATED_AT,
        "dedup_key": f"course_library_binding:{feedback_type}:{digest}",
        "raw_text_included": False,
        "internal_path_included": False,
        "db_access_executed": False,
    }


def bind_course_library_reference(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, Mapping) else {}
    course_id = _safe_token(source.get("course_id"), "course:missing")
    module_ref = _safe_token(source.get("module_id") or source.get("lesson_id"), "module:missing")
    evidence_ref = _safe_token(
        source.get("evidence_id") or source.get("library_node_id"),
        "missing_evidence",
    )
    bridge_trace_id = _safe_token(
        source.get("bridge_trace_id") or source.get("trace_id") or source.get("trace_ref"),
        "btrace:pending",
    )
    trace_id = _safe_token(source.get("trace_id") or bridge_trace_id, bridge_trace_id, max_length=160)
    request_id = _safe_scope_value(source, "request_id")
    bridge_family = _safe_scope_value(source, "bridge_family")
    bridge_id = _safe_scope_value(source, "bridge_id")
    standard_pack_id = _safe_scope_value(source, "standard_pack_id")
    rights_status = _normal_rights(source.get("rights_status"))
    raw_text_policy = _safe_token(source.get("raw_text_policy"), RAW_TEXT_POLICY_SUMMARY_ONLY)
    current_status = _normal_status(source.get("current_status"))
    approval_ref = _safe_token(source.get("approval_record_id"), "", max_length=120)
    tenant_id = _safe_scope_value(source, "tenant_id")
    organization_id = _safe_scope_value(source, "organization_id")
    cohort_id = _safe_scope_value(source, "cohort_id")
    entitlement_id = _safe_scope_value(source, "license_entitlement_id")
    entitlement_status = _normal_status(source.get("license_entitlement_status"))
    shape_pass = _has_shape_pass(source)
    binding_digest = _stable_digest(
        bridge_family,
        bridge_id,
        standard_pack_id,
        tenant_id,
        organization_id,
        cohort_id,
        course_id,
        module_ref,
        evidence_ref,
        bridge_trace_id,
    )
    binding_id = f"binding:{binding_digest}"

    base = {
        "binding_id": binding_id,
        "bridge_family": bridge_family,
        "bridge_id": bridge_id,
        "standard_pack_id": standard_pack_id,
        "request_id": request_id,
        "trace_id": trace_id,
        "course_id": course_id,
        "module_id": module_ref,
        "tenant_id": tenant_id,
        "organization_id": organization_id,
        "cohort_id": cohort_id,
        "library_node_id": _safe_token(source.get("library_node_id"), "", max_length=120),
        "evidence_id": "" if evidence_ref == "missing_evidence" else evidence_ref,
        "bridge_trace_id": bridge_trace_id,
        "rights_status": rights_status,
        "raw_text_policy": raw_text_policy,
        "license_entitlement_id": entitlement_id,
        "created_at": CREATED_AT,
        "raw_text_included": False,
        "internal_path_included": False,
        "db_access_executed": False,
        "skillup_use_allowed": False,
    }

    unsafe_payload = _contains_unsafe_surface(source)
    missing_binding_scope = _is_missing(source.get("course_id")) or _is_missing(
        source.get("module_id") or source.get("lesson_id")
    )
    missing_tenant_scope = not tenant_id or not organization_id or not cohort_id
    tenant_mismatch = (
        _scope_mismatch(source, "tenant_id", "target_tenant_id")
        or _scope_mismatch(source, "tenant_id", "evidence_tenant_id")
        or _scope_mismatch(source, "tenant_id", "license_tenant_id")
        or _scope_mismatch(source, "organization_id", "target_organization_id")
        or _scope_mismatch(source, "organization_id", "evidence_organization_id")
        or _scope_mismatch(source, "organization_id", "license_organization_id")
        or _scope_mismatch(source, "cohort_id", "target_cohort_id")
        or _scope_mismatch(source, "cohort_id", "evidence_cohort_id")
        or _scope_mismatch(source, "cohort_id", "license_cohort_id")
    )
    missing_bridge_scope = not bridge_family or not bridge_id or not standard_pack_id
    missing_trace_scope = (
        _is_missing(source.get("bridge_trace_id"))
        and _is_missing(source.get("trace_id"))
        and _is_missing(source.get("trace_ref"))
    )
    missing_evidence = evidence_ref == "missing_evidence"
    if unsafe_payload:
        status = RESULT_DENIED
        issue = "unsafe course library binding payload blocked by no-DB safety boundary"
        feedback_type = "BINDING_POLICY_REVIEW"
    elif missing_binding_scope:
        status = RESULT_HOLD
        issue = "HOLD_NO_BINDING: course_library_binding requires course_id and module_id"
        feedback_type = "BINDING_POLICY_REVIEW"
    elif missing_tenant_scope:
        status = RESULT_HOLD
        issue = "HOLD_TENANT_BOUNDARY: course_library_binding requires tenant_id, organization_id, and cohort_id"
        feedback_type = "BINDING_POLICY_REVIEW"
    elif tenant_mismatch:
        status = RESULT_HOLD
        issue = "HOLD_TENANT_BOUNDARY: course_library_binding scope mismatch"
        feedback_type = "BINDING_POLICY_REVIEW"
    elif missing_bridge_scope:
        status = RESULT_HOLD
        issue = "HOLD_BRIDGE_BOUNDARY: course_library_binding requires bridge_family, bridge_id, and standard_pack_id"
        feedback_type = "BINDING_POLICY_REVIEW"
    elif missing_trace_scope:
        status = RESULT_HOLD
        issue = "HOLD_TRACE_REQUIRED: course_library_binding requires trace_id or bridge_trace_id"
        feedback_type = "EVIDENCE_GAP"
    elif missing_evidence:
        status = RESULT_HOLD
        issue = "course_library_binding requires evidence_id or library_node_id"
        feedback_type = "EVIDENCE_GAP"
    elif rights_status in _DENIED_RIGHTS:
        status = RESULT_DENIED
        issue = "rights_status blocks Skillup course library binding use"
        feedback_type = "RIGHTS_POLICY_REVIEW"
    elif rights_status in _HOLD_RIGHTS:
        status = RESULT_HOLD
        issue = "rights_status requires review before Skillup use"
        feedback_type = "RIGHTS_POLICY_REVIEW"
    elif rights_status == "LICENSED" and raw_text_policy != RAW_TEXT_POLICY_POINTER_ONLY:
        status = RESULT_HOLD
        issue = "HOLD_POLICY: licensed course_library_binding requires pointer-only raw text policy"
        feedback_type = "RIGHTS_POLICY_REVIEW"
    elif rights_status == "LICENSED" and not entitlement_id:
        status = RESULT_HOLD
        issue = "HOLD_PERMISSION: license_entitlement_id is required before Skillup use"
        feedback_type = "RIGHTS_POLICY_REVIEW"
    elif rights_status == "LICENSED" and entitlement_status not in _ACTIVE_ENTITLEMENT_STATUSES:
        status = RESULT_HOLD
        issue = "HOLD_LICENSE_EXPIRED: license entitlement is missing, expired, suspended, unknown, or unsupported"
        feedback_type = "RIGHTS_POLICY_REVIEW"
    elif current_status != _LIBRARY_BINDABLE_STATUS:
        status = RESULT_HOLD
        issue = "course_library_binding requires APPROVED_FOR_LIBRARY status before Skillup use"
        feedback_type = "STATE_TRANSITION_REVIEW"
    elif not approval_ref:
        status = RESULT_HOLD
        issue = "course_library_binding requires approval_record_id before Skillup use"
        feedback_type = "APPROVAL_GAP"
    elif not shape_pass:
        status = RESULT_HOLD
        issue = "course_library_binding requires shape PASS evidence before Skillup use"
        feedback_type = "SHAPE_VALIDATION_GAP"
    else:
        return {
            **base,
            "binding_status": RESULT_BOUND,
            "feedback_candidate_required": False,
            "feedback_queue_item": None,
            "skillup_use_allowed": True,
        }

    feedback = _feedback_queue_item(
        course_id=course_id,
        module_ref=module_ref,
        evidence_ref=evidence_ref,
        feedback_type=feedback_type,
        suspected_issue=issue,
    )
    return {
        **base,
        "binding_status": status,
        "hold_reason": issue,
        "feedback_candidate_required": True,
        "feedback_queue_item": feedback,
        "skillup_use_allowed": False,
    }


__all__ = [
    "CREATED_AT",
    "RAW_TEXT_POLICY_POINTER_ONLY",
    "RAW_TEXT_POLICY_SUMMARY_ONLY",
    "RESULT_BOUND",
    "RESULT_DENIED",
    "RESULT_HOLD",
    "bind_course_library_reference",
]
