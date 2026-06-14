from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from admin.f13_runtime_guard import RESULT_DENIED, RESULT_HOLD, RESULT_OK


SCHEMA_VERSION = "1"
CONTRACT_VERSION = "R9ZKY-2026-06-13"

ANSWER_STATUS_ANSWERED = "ANSWERED"
ANSWER_STATUS_HOLD = "HOLD"
ANSWER_STATUS_INVALIDATED = "INVALIDATED"

RESULT_ERROR = "ERROR"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "contract_version",
    "trace_id",
    "request_id",
    "course_id",
    "module_id",
    "binding_id",
    "answer_status",
    "result_status",
    "answer",
    "hold_reason_code",
    "hold_reason",
    "evidence_required",
    "evidence",
    "policy",
    "raw_text_included",
    "internal_path_included",
    "warnings",
    "review_required",
}
_EVIDENCE_FIELDS = {
    "evidence_id",
    "node_id",
    "pointer",
    "source_label",
    "rights_status",
    "sensitivity",
}
_POLICY_FIELDS = {
    "raw_leak_check_passed",
    "rights_check_passed",
    "sensitivity_check_passed",
    "evidence_check_passed",
}
_UNSAFE_STRING_MARKERS = (
    "raw_text",
    "raw prompt",
    "raw query",
    "raw answer",
    "full answer",
    "internal_path",
    "file://",
    "localhost",
    "127.0.0.1",
    "secret",
    "token",
    "credential",
    "h:\\",
    "c:\\",
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _safe_string(value: Any, *, max_length: int) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text or len(text) > max_length:
        return None
    if any(ord(char) < 32 for char in text):
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in _UNSAFE_STRING_MARKERS):
        return None
    return text


def _safe_optional(source: Mapping[str, Any], key: str, *, max_length: int = 160) -> str | None:
    return _safe_string(source.get(key), max_length=max_length)


def _first_safe(*values: Any, max_length: int = 160) -> str | None:
    for value in values:
        safe = _safe_string(value, max_length=max_length)
        if safe is not None:
            return safe
    return None


def _safe_context_value(
    key: str,
    *,
    response: Mapping[str, Any],
    request_context: Mapping[str, Any],
    bridge_payload: Mapping[str, Any],
) -> str | None:
    nested_request = _as_mapping(request_context.get("request_payload"))
    evidence_items = bridge_payload.get("evidence_items")
    first_evidence = evidence_items[0] if isinstance(evidence_items, list) and evidence_items else {}
    evidence = _as_mapping(first_evidence)
    return _first_safe(
        response.get(key),
        request_context.get(key),
        nested_request.get(key),
        bridge_payload.get(key),
        evidence.get(key),
    )


def _stable_trace_fallback(
    response: Mapping[str, Any],
    request_context: Mapping[str, Any],
    bridge_payload: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256(
        "\x1f".join(
            str(part or "")
            for part in (
                response.get("result_status"),
                response.get("answer_status"),
                request_context.get("requester_module"),
                bridge_payload.get("result_status"),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"skillup-answer-hold:{digest}"


def _trace_id(
    response: Mapping[str, Any],
    request_context: Mapping[str, Any],
    bridge_payload: Mapping[str, Any],
    warnings: list[str],
) -> str:
    feedback_queue_item = _as_mapping(response.get("feedback_queue_item"))
    feedback_candidate = _as_mapping(response.get("feedback_candidate"))
    evidence_items = bridge_payload.get("evidence_items")
    first_evidence = evidence_items[0] if isinstance(evidence_items, list) and evidence_items else {}
    evidence = _as_mapping(first_evidence)
    trace = _first_safe(
        response.get("bridge_trace_id"),
        evidence.get("bridge_trace_id"),
        feedback_queue_item.get("origin_event_id"),
        request_context.get("origin_event_id"),
        feedback_candidate.get("bridge_trace_id"),
        max_length=160,
    )
    if trace is not None:
        return trace
    warnings.append("TRACE_ID_FALLBACK_USED")
    return _stable_trace_fallback(response, request_context, bridge_payload)


def _normalize_statuses(response: Mapping[str, Any], warnings: list[str]) -> tuple[str, str]:
    source_result = str(response.get("result_status") or RESULT_HOLD).strip().upper()
    source_answer = str(response.get("answer_status") or ANSWER_STATUS_HOLD).strip().upper()

    if source_result == RESULT_OK:
        return RESULT_OK, ANSWER_STATUS_ANSWERED
    if source_result == RESULT_DENIED or source_answer == RESULT_DENIED:
        warnings.append("SOURCE_DENIED_NORMALIZED_TO_ERROR")
        return RESULT_ERROR, ANSWER_STATUS_INVALIDATED
    return RESULT_HOLD, ANSWER_STATUS_HOLD


def _hold_reason_code(result_status: str, response: Mapping[str, Any], warnings: list[str]) -> str | None:
    if result_status == RESULT_OK:
        return None

    reason = str(response.get("hold_reason") or "").strip().lower()
    if result_status == RESULT_ERROR:
        if "raw text" in reason or "raw_text" in reason:
            return "RAW_TEXT_BLOCKED"
        if "internal path" in reason or "internal_path" in reason:
            return "INTERNAL_PATH_BLOCKED"
        if "db" in reason or "no-db" in reason:
            return "NO_DB_BOUNDARY"
        if "role" in reason or "access" in reason:
            return "ROLE_ACCESS_DENIED"
        return "DENIED_POLICY_BOUNDARY"

    if "bridge response is required" in reason:
        return "BRIDGE_RESPONSE_REQUIRED"
    if "safe evidence" in reason or "evidence" in reason:
        return "EVIDENCE_REQUIRED"
    if "unsupported" in reason:
        return "UNSUPPORTED_STATUS_HOLD"
    if "policy" in reason:
        return "HOLD_REVIEW_REQUIRED"
    warnings.append("EVIDENCE_ARRAY_EMPTY_FOR_HOLD")
    return "HOLD_REVIEW_REQUIRED"


def _evidence_items(response: Mapping[str, Any], bridge_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_items = bridge_payload.get("evidence_items")
    items = raw_items if isinstance(raw_items, list) else []
    adapted: list[dict[str, Any]] = []

    for item in items[:10]:
        source = _as_mapping(item)
        projected: dict[str, Any] = {}
        evidence_id = _safe_optional(source, "evidence_id")
        if evidence_id is not None:
            projected["evidence_id"] = evidence_id
        node_id = _safe_optional(source, "node_id")
        if node_id is not None:
            projected["node_id"] = node_id
        pointer = _first_safe(source.get("pointer"), source.get("pointer_uri"), max_length=512)
        if pointer is not None and pointer.startswith("pointer://"):
            projected["pointer"] = pointer
        source_label = _first_safe(source.get("source_label"), "Skillup Bridge safe evidence", max_length=240)
        if source_label is not None:
            projected["source_label"] = source_label
        rights_status = _safe_optional(source, "rights_status", max_length=80)
        if rights_status is not None:
            projected["rights_status"] = rights_status
        sensitivity = _safe_optional(source, "sensitivity", max_length=80)
        if sensitivity is not None:
            projected["sensitivity"] = sensitivity
        if projected:
            adapted.append({key: value for key, value in projected.items() if key in _EVIDENCE_FIELDS})

    if not adapted:
        evidence_id = _safe_optional(response, "evidence_id")
        pointer = _safe_optional(response, "pointer_uri", max_length=512)
        projected = {}
        if evidence_id is not None:
            projected["evidence_id"] = evidence_id
        if pointer is not None and pointer.startswith("pointer://"):
            projected["pointer"] = pointer
        if projected:
            source_label = _safe_string("Skillup Bridge safe evidence", max_length=240)
            if source_label is not None:
                projected["source_label"] = source_label
            adapted.append(projected)

    return adapted


def _bool_from_mapping(source: Mapping[str, Any], key: str) -> bool | None:
    if key not in source:
        return None
    value = source.get(key)
    return value if isinstance(value, bool) else None


def _policy(
    result_status: str,
    response: Mapping[str, Any],
    bridge_payload: Mapping[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, bool]:
    policy_result = _as_mapping(bridge_payload.get("policy_result"))
    raw_text_included = response.get("raw_text_included") is True or bridge_payload.get("raw_text_included") is True
    internal_path_included = (
        response.get("internal_path_included") is True
        or bridge_payload.get("internal_path_included") is True
    )

    raw_leak = _bool_from_mapping(policy_result, "raw_leak_pass")
    rights = _bool_from_mapping(policy_result, "rights_pass")
    sensitivity = _bool_from_mapping(policy_result, "sensitivity_pass")
    evidence_pass = _bool_from_mapping(policy_result, "evidence_required_pass")

    default_ok = result_status == RESULT_OK
    policy = {
        "raw_leak_check_passed": bool(raw_leak) if raw_leak is not None else not raw_text_included and not internal_path_included,
        "rights_check_passed": bool(rights) if rights is not None else default_ok,
        "sensitivity_check_passed": bool(sensitivity) if sensitivity is not None else result_status != RESULT_ERROR,
        "evidence_check_passed": bool(evidence_pass) if evidence_pass is not None else default_ok and bool(evidence),
    }
    return {key: bool(policy[key]) for key in _POLICY_FIELDS}


def adapt_skillup_answer_hold_response(
    helper_response: Mapping[str, Any] | None,
    *,
    request_context: Mapping[str, Any] | None = None,
    bridge_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    response = _as_mapping(helper_response)
    request = _as_mapping(request_context)
    bridge = _as_mapping(bridge_payload)
    warnings: list[str] = []

    result_status, answer_status = _normalize_statuses(response, warnings)
    evidence = _evidence_items(response, bridge)
    hold_reason_code = _hold_reason_code(result_status, response, warnings)

    adapted: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "trace_id": _trace_id(response, request, bridge, warnings),
        "answer_status": answer_status,
        "result_status": result_status,
        "evidence_required": result_status != RESULT_OK,
        "evidence": evidence,
        "policy": _policy(result_status, response, bridge, evidence),
        "raw_text_included": False,
        "internal_path_included": False,
        "review_required": result_status != RESULT_OK,
    }

    for key in ("request_id", "course_id", "module_id", "binding_id"):
        value = _safe_context_value(key, response=response, request_context=request, bridge_payload=bridge)
        if value is not None:
            adapted[key] = value

    if result_status == RESULT_OK:
        answer = _first_safe(response.get("answer"), response.get("safe_summary"), max_length=4000)
        if answer is not None:
            adapted["answer"] = answer
    else:
        if hold_reason_code is not None:
            adapted["hold_reason_code"] = hold_reason_code
        hold_reason = _safe_optional(response, "hold_reason", max_length=1000)
        if hold_reason is not None:
            adapted["hold_reason"] = hold_reason

    safe_warnings = []
    for warning in warnings:
        safe_warning = _safe_string(warning, max_length=400)
        if safe_warning is not None and safe_warning not in safe_warnings:
            safe_warnings.append(safe_warning)
    if safe_warnings:
        adapted["warnings"] = safe_warnings

    return {key: value for key, value in adapted.items() if key in _TOP_LEVEL_FIELDS}


__all__ = [
    "CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "adapt_skillup_answer_hold_response",
]
