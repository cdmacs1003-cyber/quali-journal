from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from admin.f13_runtime_guard import (
    RESULT_DENIED,
    RESULT_HOLD,
    RESULT_OK,
    detect_forbidden_fields,
    project_bridge_safe_evidence,
)


ANSWER_STATUS_ANSWERED = "ANSWERED"
ANSWER_STATUS_DENIED = "DENIED"
ANSWER_STATUS_HOLD = "HOLD"
NOT_GRANTED = "NOT_GRANTED"


def _safe_text(value: Any, fallback: str, max_length: int = 240) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text[:max_length]


def _status_base() -> dict[str, Any]:
    return {
        "raw_text_included": False,
        "internal_path_included": False,
        "db_access_executed": False,
        "f13_pass": NOT_GRANTED,
        "track_a_pass": NOT_GRANTED,
        "beta_pass": NOT_GRANTED,
    }


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
    }


def skillup_answer_from_request(request_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(request_payload, Mapping):
        return _blocked(RESULT_HOLD, "Skillup request payload is missing.")

    violations = detect_forbidden_fields(request_payload)
    if violations:
        return _blocked(RESULT_DENIED, "Skillup Bridge request was blocked by the no-DB safety boundary.")

    return _blocked(RESULT_HOLD, "Bridge response is required before Skillup can answer.")


__all__ = [
    "ANSWER_STATUS_ANSWERED",
    "ANSWER_STATUS_DENIED",
    "ANSWER_STATUS_HOLD",
    "NOT_GRANTED",
    "skillup_answer_from_bridge_response",
    "skillup_answer_from_request",
]
