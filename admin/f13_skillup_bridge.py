from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from admin.f13_runtime_guard import (
    RESULT_DENIED,
    RESULT_HOLD,
    RESULT_OK,
    decide_role_access_policy,
    detect_forbidden_fields,
    project_bridge_safe_evidence,
    zero_leak_counters,
)


ANSWER_STATUS_ANSWERED = "ANSWERED"
ANSWER_STATUS_DENIED = "DENIED"
ANSWER_STATUS_HOLD = "HOLD"
NOT_GRANTED = "NOT_GRANTED"
FEEDBACK_QUEUE_CREATED_AT = "1970-01-01T00:00:00Z"

_UNSAFE_FEEDBACK_FIELD_MARKERS = (
    "raw_text",
    "raw_prompt",
    "raw_query",
    "raw_answer",
    "full_answer",
    "internal_path",
    "local_route",
    "secret",
    "token",
    "credential",
    "paid_standard",
)
_UNSAFE_FEEDBACK_VALUE_MARKERS = (
    "raw text",
    "raw prompt",
    "raw query",
    "internal path",
    "h:\\",
    "c:\\",
    "file://",
    "localhost",
    "127.0.0.1",
    "secret",
    "token",
    "credential",
)
_SAFE_FEEDBACK_COUNTER_KEYS = {
    "raw_text_export_count",
    "internal_path_leak_count",
    "raw_prompt_output_count",
    "secret_leak_count",
    "instructor_guide_raw_leak_count",
}
_ROLE_CONTEXT_FIELDS = (
    "role",
    "evidence_depth",
    "requested_output_type",
    "requested_action",
    "action",
    "trace_view",
    "export_type",
    "course_id",
    "module_id",
    "binding_id",
    "course_library_binding",
    "tenant_id",
    "organization_id",
    "cohort_id",
    "target_tenant_id",
    "target_organization_id",
    "target_cohort_id",
    "evidence_tenant_id",
    "evidence_organization_id",
    "evidence_cohort_id",
    "license_tenant_id",
    "license_organization_id",
    "license_cohort_id",
    "license_required",
    "license_entitlement_id",
    "license_entitlement_status",
    "paid_standard",
    "source_doc_kind",
    "rights_status",
)


def _safe_text(value: Any, fallback: str, max_length: int = 240) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text[:max_length]


def _safe_token(value: Any, fallback: str, max_length: int = 120) -> str:
    text = _safe_text(value, fallback, max_length=max_length)
    token = "".join(ch for ch in text if ch.isalnum() or ch in ":._-")
    return token or fallback


def _stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(_safe_text(part, "", max_length=240) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _contains_unsafe_feedback_surface(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered_key = str(key).lower()
            if lowered_key in _SAFE_FEEDBACK_COUNTER_KEYS:
                continue
            if lowered_key in {"raw_text_included", "internal_path_included"}:
                continue
            if any(marker in lowered_key for marker in _UNSAFE_FEEDBACK_FIELD_MARKERS):
                return True
            if _contains_unsafe_feedback_surface(child):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_unsafe_feedback_surface(child) for child in value)
    if isinstance(value, str):
        lowered_value = value.lower()
        return any(marker in lowered_value for marker in _UNSAFE_FEEDBACK_VALUE_MARKERS)
    return False


def _safe_feedback_issue(reason: Any, unsafe_payload: bool) -> str:
    if unsafe_payload:
        return "Unsafe feedback payload blocked by no-DB safety boundary."
    return _safe_text(reason, "Bridge evidence gap requires curation before Skillup can answer.")


def _status_base() -> dict[str, Any]:
    return {
        "raw_text_included": False,
        "internal_path_included": False,
        "db_access_executed": False,
        "f13_pass": NOT_GRANTED,
        "track_a_pass": NOT_GRANTED,
        "beta_pass": NOT_GRANTED,
        **zero_leak_counters(),
    }


def _role_context_from_bridge_response(bridge_response: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key in _ROLE_CONTEXT_FIELDS:
        value = bridge_response.get(key)
        if value not in (None, ""):
            context[key] = value
    evidence_items = bridge_response.get("evidence_items") or []
    if evidence_items and isinstance(evidence_items[0], Mapping):
        for key in _ROLE_CONTEXT_FIELDS:
            value = evidence_items[0].get(key)
            if value not in (None, "") and key not in context:
                context[key] = value
    return context


def _role_policy_fields(decision: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "role": decision.get("role"),
        "evidence_depth": decision.get("evidence_depth"),
    }
    for key, default in zero_leak_counters().items():
        fields[key] = int(decision.get(key, default) or 0)
    return fields


def _feedback_candidate(reason: Any, bridge_trace_id: Any = None) -> dict[str, Any]:
    candidate = {
        "candidate_type": "SKILLUP_BRIDGE_HOLD_FEEDBACK",
        "reason": _safe_text(reason, "Bridge evidence is required before Skillup can answer."),
    }
    trace_id = _safe_text(bridge_trace_id, "", max_length=160)
    if trace_id:
        candidate["bridge_trace_id"] = trace_id
    return candidate


def _blocked(status: str, reason: Any, bridge_trace_id: Any = None) -> dict[str, Any]:
    safe_status = RESULT_DENIED if status == RESULT_DENIED else RESULT_HOLD
    answer_status = ANSWER_STATUS_DENIED if safe_status == RESULT_DENIED else ANSWER_STATUS_HOLD
    safe_reason = _safe_text(reason, "Bridge evidence is required before Skillup can answer.")
    return {
        **_status_base(),
        "result_status": safe_status,
        "answer_status": answer_status,
        "hold_reason": safe_reason,
        "feedback_candidate_required": True,
        "feedback_candidate": _feedback_candidate(safe_reason, bridge_trace_id),
    }


def skillup_answer_from_bridge_response(bridge_response: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(bridge_response, Mapping):
        return _blocked(RESULT_HOLD, "Bridge response is missing.")

    if bridge_response.get("raw_text_included") is True:
        return _blocked(RESULT_DENIED, "Bridge response included raw text.")
    if bridge_response.get("internal_path_included") is True:
        return _blocked(RESULT_DENIED, "Bridge response included an internal path.")

    status = str(bridge_response.get("result_status") or RESULT_HOLD).strip().upper()
    if status != RESULT_OK:
        evidence_items = bridge_response.get("evidence_items") or []
        trace_id = None
        if evidence_items and isinstance(evidence_items[0], Mapping):
            trace_id = evidence_items[0].get("bridge_trace_id")
        return _blocked(
            RESULT_DENIED if status == RESULT_DENIED else RESULT_HOLD,
            bridge_response.get("hold_reason"),
            trace_id,
        )

    role_decision = decide_role_access_policy(_role_context_from_bridge_response(bridge_response))
    if role_decision.get("result_status") != RESULT_OK:
        evidence_items_for_trace = bridge_response.get("evidence_items") or []
        trace_id = None
        if evidence_items_for_trace and isinstance(evidence_items_for_trace[0], Mapping):
            trace_id = evidence_items_for_trace[0].get("bridge_trace_id")
        blocked = _blocked(
            RESULT_DENIED if role_decision.get("result_status") == RESULT_DENIED else RESULT_HOLD,
            role_decision.get("hold_reason"),
            trace_id,
        )
        blocked.update(_role_policy_fields(role_decision))
        return blocked

    evidence_items = bridge_response.get("evidence_items") or []
    if not evidence_items or not isinstance(evidence_items[0], Mapping):
        return _blocked(RESULT_HOLD, "Bridge OK response did not include safe evidence.")

    if detect_forbidden_fields(evidence_items[0]):
        return _blocked(RESULT_DENIED, "Bridge evidence item contained forbidden fields.")

    projected = project_bridge_safe_evidence(evidence_items[0])
    required = ("evidence_id", "bridge_trace_id", "safe_summary")
    if any(not projected.get(field) for field in required):
        return _blocked(RESULT_HOLD, "Bridge evidence item is missing Skillup answer fields.")

    safe_summary = _safe_text(projected["safe_summary"], "Bridge safe summary unavailable.", max_length=2000)
    return {
        **_status_base(),
        "result_status": RESULT_OK,
        "answer_status": ANSWER_STATUS_ANSWERED,
        "hold_reason": None,
        "feedback_candidate_required": False,
        "feedback_candidate": None,
        "answer": safe_summary,
        "safe_summary": safe_summary,
        "evidence_id": projected["evidence_id"],
        "bridge_trace_id": projected["bridge_trace_id"],
        **_role_policy_fields(role_decision),
    }


def skillup_answer_from_request(request_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(request_payload, Mapping):
        return _blocked(RESULT_HOLD, "Skillup request payload is missing.")

    violations = detect_forbidden_fields(request_payload)
    if violations:
        return _blocked(RESULT_DENIED, "Skillup Bridge request was blocked by the no-DB safety boundary.")

    return _blocked(RESULT_HOLD, "Bridge response is required before Skillup can answer.")


def skillup_feedback_queue_item_from_hold(hold_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    source = hold_payload if isinstance(hold_payload, Mapping) else {}
    candidate = source.get("feedback_candidate") if isinstance(source.get("feedback_candidate"), Mapping) else source
    unsafe_payload = _contains_unsafe_feedback_surface(source)
    reason = candidate.get("reason") if isinstance(candidate, Mapping) else source.get("hold_reason")
    suspected_issue = _safe_feedback_issue(reason or source.get("hold_reason"), unsafe_payload)

    origin_module = _safe_token(source.get("origin_module") or source.get("requester_module"), "Skillup")
    proposed_candidate_type = _safe_token(
        candidate.get("candidate_type") if isinstance(candidate, Mapping) else None,
        "SKILLUP_BRIDGE_HOLD_FEEDBACK",
    )
    linked_evidence_id = _safe_token(
        source.get("linked_evidence_id") or source.get("evidence_id"),
        "missing_evidence",
    )
    linked_answer_id = _safe_token(
        source.get("linked_answer_id") or source.get("answer_id"),
        "answer:pending",
    )
    feedback_type = "EVIDENCE_GAP" if linked_evidence_id == "missing_evidence" and not unsafe_payload else "HOLD_CASE"
    dedup_basis = linked_evidence_id if linked_evidence_id != "missing_evidence" else suspected_issue
    dedup_digest = _stable_digest(origin_module, feedback_type, dedup_basis)
    origin_event_id = _safe_token(
        source.get("origin_event_id")
        or source.get("bridge_trace_id")
        or (candidate.get("bridge_trace_id") if isinstance(candidate, Mapping) else None),
        f"hold:{dedup_digest}",
    )

    return {
        "feedback_id": f"fbq:{dedup_digest}",
        "origin_module": origin_module,
        "origin_event_id": origin_event_id,
        "feedback_type": feedback_type,
        "user_visible_text_policy": "SUMMARY_ONLY",
        "linked_answer_id": linked_answer_id,
        "linked_evidence_id": linked_evidence_id,
        "suspected_issue": suspected_issue,
        "proposed_candidate_type": proposed_candidate_type,
        "current_status": "review_required" if unsafe_payload else "queued",
        "created_at": FEEDBACK_QUEUE_CREATED_AT,
        "dedup_key": f"{origin_module}:{feedback_type}:{dedup_digest}",
        "result_status": RESULT_HOLD,
        "raw_text_included": False,
        "internal_path_included": False,
        "db_access_executed": False,
    }


__all__ = [
    "ANSWER_STATUS_ANSWERED",
    "ANSWER_STATUS_DENIED",
    "ANSWER_STATUS_HOLD",
    "FEEDBACK_QUEUE_CREATED_AT",
    "NOT_GRANTED",
    "skillup_answer_from_bridge_response",
    "skillup_feedback_queue_item_from_hold",
    "skillup_answer_from_request",
]
