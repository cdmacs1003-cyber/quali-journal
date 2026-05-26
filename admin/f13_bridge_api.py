"""F13 no-DB Bridge API route.

This router exposes the existing in-memory Bridge guard utility over a
provided-evidence-only API boundary. It does not query DB, Warehouse, Library,
Skillup runtime, files, network, or runtime indexes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from admin.f13_runtime_guard import (
    RESULT_DENIED,
    RESULT_HOLD,
    RESULT_OK,
    detect_forbidden_fields,
    decide_bridge_result,
    normalize_rights_status,
    project_bridge_safe_evidence,
    validate_human_redacted_preflight_replay_evidence,
)


router = APIRouter(prefix="/api/f13/bridge", tags=["f13-bridge"])

_MAX_RETURN_ITEMS = 10
_REDACTED_PREFLIGHT_REPLAY_EVIDENCE_FIELD = "redacted_preflight_replay_evidence"
_SCHEMA_REQUIRED_EVIDENCE_FIELDS = (
    "evidence_id",
    "bridge_trace_id",
    "safe_summary",
    "pointer_uri",
    "raw_text_policy",
    "rights_status",
)


class BridgeEvidenceRequest(BaseModel):
    query: Optional[str] = None
    purpose: str = "answer"
    requester_module: str = "Skillup"
    allowed_rights_status: List[str] = Field(default_factory=list)
    max_items: Optional[int] = Field(default=None, ge=1)
    evidence_items: Optional[List[Dict[str, Any]]] = None
    redacted_preflight_replay_evidence: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class BridgePolicyResult(BaseModel):
    evidence_required_pass: bool
    raw_leak_pass: bool
    rights_pass: bool
    sensitivity_pass: bool


class BridgeEvidenceResponse(BaseModel):
    result_status: str
    evidence_items: List[Dict[str, Any]]
    hold_reason: Optional[str]
    feedback_candidate_required: bool
    raw_text_included: bool
    internal_path_included: bool
    policy_result: BridgePolicyResult
    created_at: str


class BridgePolicyCheckRequest(BaseModel):
    evidence: Optional[Dict[str, Any]] = None
    evidence_items: Optional[List[Dict[str, Any]]] = None
    evidence_id: Optional[str] = None
    bridge_trace_id: Optional[str] = None
    safe_summary: Optional[str] = None
    pointer_uri: Optional[str] = None
    raw_text_policy: Optional[str] = None
    rights_status: Optional[str] = None
    role: str = "Learner"
    requested_output_type: str = "safe_summary"
    requester_module: str = "Skillup"
    purpose: str = "answer"

    class Config:
        extra = "allow"


class BridgePolicyCheckResponse(BaseModel):
    result_status: str
    bridge_trace_id: Optional[str]
    policy_result: str
    hold_reason: Optional[str]
    output_constraints: List[str]
    blocked_fields: List[str]
    feedback_candidate_required: bool
    raw_text_included: bool
    internal_path_included: bool
    created_at: str


class BridgeTraceExplainRequest(BaseModel):
    bridge_trace_id: Optional[str] = None
    role: str = "Learner"
    request_id: Optional[str] = None
    course_id: Optional[str] = None
    module_id: Optional[str] = None
    binding_id: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None
    evidence_items: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"


class BridgeTraceExplainResponse(BaseModel):
    result_status: str
    request_id: Optional[str]
    bridge_trace_id: Optional[str]
    course_id: Optional[str]
    module_id: Optional[str]
    binding_id: Optional[str]
    evidence_ids: List[str]
    policy_result: str
    hold_reason: Optional[str]
    visible_trace_summary: str
    raw_text_included: bool
    internal_path_included: bool
    created_at: str



def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()



def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")



def _bounded_return_limit(value: Optional[int]) -> int:
    if value is None:
        return _MAX_RETURN_ITEMS
    return min(max(int(value), 1), _MAX_RETURN_ITEMS)



def _has_schema_required_fields(item: Dict[str, Any]) -> bool:
    for field in _SCHEMA_REQUIRED_EVIDENCE_FIELDS:
        value = item.get(field)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return False
    return True


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _safe_label(value: object, *, max_length: int = 96) -> Optional[str]:
    if _is_missing(value) or detect_forbidden_fields(value):
        return None
    text = str(value).strip()
    if len(text) > max_length or any(ord(char) < 32 for char in text):
        return None
    return text


def _safe_role(value: object) -> str:
    role = (_safe_label(value, max_length=32) or "Learner").lower()
    if role in {"learner", "student"}:
        return "Learner"
    if role == "instructor":
        return "Instructor"
    if role == "reviewer":
        return "Reviewer"
    if role == "admin":
        return "Admin"
    return "Learner"


def _safe_trace_id(value: object) -> Optional[str]:
    trace_id = _safe_label(value, max_length=120)
    if trace_id is None:
        return None
    if not trace_id.startswith("btrace:"):
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-")
    if any(char not in allowed for char in trace_id):
        return None
    return trace_id


def _bounded_blocked_fields(fields: List[str]) -> List[str]:
    safe_fields: List[str] = []
    for field in fields[:10]:
        label = _safe_label(field, max_length=120)
        if label is not None:
            safe_fields.append(label)
    return safe_fields


def _policy_constraints(status: str) -> List[str]:
    if status == RESULT_OK:
        return [
            "SAFE_SUMMARY_ONLY",
            "NO_RAW_TEXT",
            "NO_INTERNAL_PATH",
            "BRIDGE_TRACE_REQUIRED",
        ]
    if status == RESULT_DENIED:
        return ["BLOCK_OUTPUT", "NO_RAW_TEXT", "NO_INTERNAL_PATH"]
    return [
        "HOLD_UNTIL_EVIDENCE_TRACE_RIGHTS_POLICY_PASS",
        "NO_RAW_TEXT",
        "NO_INTERNAL_PATH",
    ]


def _policy_label(status: str) -> str:
    if status == RESULT_OK:
        return "PASS"
    if status == RESULT_DENIED:
        return "DENIED"
    return "HOLD"


def _evidence_from_policy_request(payload: BridgePolicyCheckRequest) -> List[Dict[str, Any]]:
    evidence_items: List[Dict[str, Any]] = []
    if isinstance(payload.evidence, dict):
        evidence_items.append(dict(payload.evidence))
    for item in payload.evidence_items or []:
        if isinstance(item, dict):
            evidence_items.append(dict(item))

    if not evidence_items:
        evidence_items.append({})

    top_level = _model_to_dict(payload)
    for item in evidence_items:
        for key in (
            "evidence_id",
            "bridge_trace_id",
            "safe_summary",
            "pointer_uri",
            "raw_text_policy",
            "rights_status",
        ):
            if _is_missing(item.get(key)) and not _is_missing(top_level.get(key)):
                item[key] = top_level[key]
        if not _is_missing(top_level.get("requested_output_type")):
            item.setdefault("requested_output_type", top_level["requested_output_type"])
        if not _is_missing(top_level.get("role")):
            item.setdefault("role", top_level["role"])

    return evidence_items


def _safe_evidence_ids(trace: Dict[str, Any], evidence_items: List[Dict[str, Any]]) -> List[str]:
    raw_ids: List[object] = []
    for key in ("evidence_ids", "promoted_evidence_ids"):
        value = trace.get(key)
        if isinstance(value, list):
            raw_ids.extend(value)
    for item in evidence_items:
        raw_ids.append(item.get("evidence_id"))

    safe_ids: List[str] = []
    for value in raw_ids:
        label = _safe_label(value, max_length=96)
        if label is not None and label not in safe_ids:
            safe_ids.append(label)
    return safe_ids[:20]


def _first_safe_value(*values: object) -> Optional[str]:
    for value in values:
        label = _safe_label(value)
        if label is not None:
            return label
    return None


def _safe_reason_code_token(value: Any) -> Optional[str]:
    token = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if token and all(char.isalnum() or char == "_" for char in token):
        return token
    return None


def _bounded_preflight_validation_reason(validation: Dict[str, Any]) -> str:
    result_status = str(validation.get("result_status") or "").strip().upper()
    status = str(validation.get("status") or "").strip().upper()
    safe_codes = [
        token
        for token in (
            _safe_reason_code_token(value)
            for value in validation.get("reason_codes", [])
        )
        if token is not None
    ]
    code_suffix = f": {safe_codes[0]}" if safe_codes else ""
    if result_status == RESULT_DENIED or status.startswith("DENY_"):
        return f"redacted preflight replay evidence denied by safety boundary{code_suffix}"
    return f"redacted preflight replay evidence requires review{code_suffix}"


def _preflight_validation_gate_response(request_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    redacted_evidence = request_payload.get(_REDACTED_PREFLIGHT_REPLAY_EVIDENCE_FIELD)
    if redacted_evidence is None:
        return None

    validation = validate_human_redacted_preflight_replay_evidence(redacted_evidence)
    result_status = str(validation.get("result_status") or "").strip().upper()
    if result_status == RESULT_OK and validation.get("ok") is True:
        return None

    if result_status == RESULT_DENIED:
        return _response(
            RESULT_DENIED,
            [],
            _bounded_preflight_validation_reason(validation),
            evidence_required_pass=False,
            raw_leak_pass=False,
            rights_pass=True,
            sensitivity_pass=False,
        )

    return _response(
        RESULT_HOLD,
        [],
        _bounded_preflight_validation_reason(validation),
        evidence_required_pass=False,
        raw_leak_pass=True,
        rights_pass=True,
        sensitivity_pass=False,
    )



def _response(
    status: str,
    evidence_items: List[Dict[str, Any]],
    hold_reason: Optional[str],
    *,
    evidence_required_pass: bool,
    raw_leak_pass: bool,
    rights_pass: bool,
    sensitivity_pass: bool = True,
) -> Dict[str, Any]:
    safe_status = status if status in {RESULT_OK, RESULT_HOLD, RESULT_DENIED} else RESULT_HOLD
    safe_reason = None if safe_status == RESULT_OK else (hold_reason or "Bridge evidence request requires review")
    return {
        "result_status": safe_status,
        "evidence_items": evidence_items,
        "hold_reason": safe_reason,
        "feedback_candidate_required": safe_status != RESULT_OK,
        "raw_text_included": False,
        "internal_path_included": False,
        "policy_result": {
            "evidence_required_pass": bool(evidence_required_pass),
            "raw_leak_pass": bool(raw_leak_pass),
            "rights_pass": bool(rights_pass),
            "sensitivity_pass": bool(sensitivity_pass),
        },
        "created_at": _created_at(),
    }


@router.post("/retrieve-evidence", response_model=BridgeEvidenceResponse)
def retrieve_bridge_evidence(payload: BridgeEvidenceRequest) -> Dict[str, Any]:
    request_payload = _model_to_dict(payload)
    if detect_forbidden_fields(request_payload):
        return _response(
            RESULT_DENIED,
            [],
            "forbidden fields or patterns detected",
            evidence_required_pass=False,
            raw_leak_pass=False,
            rights_pass=False,
        )

    preflight_gate_response = _preflight_validation_gate_response(request_payload)
    if preflight_gate_response is not None:
        return preflight_gate_response

    if payload.query is not None and payload.query.strip() == "":
        return _response(
            RESULT_HOLD,
            [],
            "query is empty",
            evidence_required_pass=False,
            raw_leak_pass=True,
            rights_pass=True,
        )

    evidence_items = payload.evidence_items or []
    if not evidence_items:
        return _response(
            RESULT_HOLD,
            [],
            "evidence_items are required for no-DB Bridge evaluation",
            evidence_required_pass=False,
            raw_leak_pass=True,
            rights_pass=True,
        )

    allowed_rights = {
        normalize_rights_status(value)
        for value in payload.allowed_rights_status
        if str(value).strip()
    }
    return_limit = _bounded_return_limit(payload.max_items)
    ok_items: List[Dict[str, Any]] = []
    hold_reasons: List[str] = []
    denied_reasons: List[str] = []

    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            hold_reasons.append("evidence payload is missing or invalid")
            continue

        rights = normalize_rights_status(evidence.get("rights_status"))
        if allowed_rights and rights not in allowed_rights:
            hold_reasons.append("rights_status is outside allowed_rights_status")
            continue

        decision = decide_bridge_result(
            evidence,
            requester_module=payload.requester_module,
            purpose=payload.purpose,
        )
        status = decision.get("result_status")
        reason = decision.get("hold_reason") or "Bridge evidence request requires review"

        if status == RESULT_DENIED:
            denied_reasons.append(str(reason))
            continue
        if status == RESULT_OK:
            projected = project_bridge_safe_evidence(evidence)
            if _has_schema_required_fields(projected):
                ok_items.append(projected)
            else:
                hold_reasons.append("projected evidence is missing Bridge schema required fields")
            continue

        hold_reasons.append(str(reason))

    if denied_reasons:
        return _response(
            RESULT_DENIED,
            [],
            denied_reasons[0],
            evidence_required_pass=False,
            raw_leak_pass=True,
            rights_pass=False,
        )

    if ok_items:
        return _response(
            RESULT_OK,
            ok_items[:return_limit],
            None,
            evidence_required_pass=True,
            raw_leak_pass=True,
            rights_pass=True,
        )

    return _response(
        RESULT_HOLD,
        [],
        hold_reasons[0] if hold_reasons else "no Bridge-safe evidence item was accepted",
        evidence_required_pass=False,
        raw_leak_pass=True,
        rights_pass=True,
    )


@router.post("/check-policy", response_model=BridgePolicyCheckResponse)
def check_bridge_policy(payload: BridgePolicyCheckRequest) -> Dict[str, Any]:
    request_payload = _model_to_dict(payload)
    forbidden = detect_forbidden_fields(request_payload)
    if forbidden:
        return {
            "result_status": RESULT_DENIED,
            "bridge_trace_id": None,
            "policy_result": "DENIED",
            "hold_reason": "forbidden fields or patterns detected",
            "output_constraints": _policy_constraints(RESULT_DENIED),
            "blocked_fields": _bounded_blocked_fields(forbidden),
            "feedback_candidate_required": True,
            "raw_text_included": False,
            "internal_path_included": False,
            "created_at": _created_at(),
        }

    evidence_items = _evidence_from_policy_request(payload)
    decisions: List[Dict[str, Any]] = []
    trace_ids: List[str] = []
    for evidence in evidence_items:
        trace_id = _safe_trace_id(evidence.get("bridge_trace_id"))
        if trace_id is not None:
            trace_ids.append(trace_id)
        decisions.append(
            decide_bridge_result(
                evidence,
                requester_module=payload.requester_module,
                purpose=payload.purpose,
            )
        )

    denied = [item for item in decisions if item.get("result_status") == RESULT_DENIED]
    holds = [item for item in decisions if item.get("result_status") == RESULT_HOLD]
    if denied:
        final_status = RESULT_DENIED
        hold_reason = str(denied[0].get("hold_reason") or "Bridge policy denied")
    elif holds:
        final_status = RESULT_HOLD
        hold_reason = str(holds[0].get("hold_reason") or "Bridge policy requires review")
    else:
        final_status = RESULT_OK
        hold_reason = None

    return {
        "result_status": final_status,
        "bridge_trace_id": trace_ids[0] if trace_ids else None,
        "policy_result": _policy_label(final_status),
        "hold_reason": hold_reason,
        "output_constraints": _policy_constraints(final_status),
        "blocked_fields": [],
        "feedback_candidate_required": final_status != RESULT_OK,
        "raw_text_included": False,
        "internal_path_included": False,
        "created_at": _created_at(),
    }


@router.post("/explain-trace", response_model=BridgeTraceExplainResponse)
def explain_bridge_trace(payload: BridgeTraceExplainRequest) -> Dict[str, Any]:
    request_payload = _model_to_dict(payload)
    forbidden = detect_forbidden_fields(request_payload)
    if forbidden:
        return {
            "result_status": RESULT_DENIED,
            "request_id": None,
            "bridge_trace_id": None,
            "course_id": None,
            "module_id": None,
            "binding_id": None,
            "evidence_ids": [],
            "policy_result": "DENIED",
            "hold_reason": "forbidden fields or patterns detected",
            "visible_trace_summary": "Trace explanation denied by Bridge safety boundary.",
            "raw_text_included": False,
            "internal_path_included": False,
            "created_at": _created_at(),
        }

    trace = dict(payload.trace or {})
    evidence_items = [dict(item) for item in payload.evidence_items or [] if isinstance(item, dict)]
    bridge_trace_id = _safe_trace_id(
        payload.bridge_trace_id
        or trace.get("bridge_trace_id")
        or trace.get("promotion_trace_id")
    )
    if bridge_trace_id is None:
        return {
            "result_status": RESULT_HOLD,
            "request_id": _safe_label(payload.request_id),
            "bridge_trace_id": None,
            "course_id": _first_safe_value(payload.course_id, trace.get("course_id")),
            "module_id": _first_safe_value(payload.module_id, trace.get("module_id")),
            "binding_id": _first_safe_value(payload.binding_id, trace.get("binding_id")),
            "evidence_ids": [],
            "policy_result": "HOLD",
            "hold_reason": "bridge_trace_id is required for no-DB trace explanation",
            "visible_trace_summary": "Trace explanation is on HOLD because bridge_trace_id is missing.",
            "raw_text_included": False,
            "internal_path_included": False,
            "created_at": _created_at(),
        }

    evidence_ids = _safe_evidence_ids(trace, evidence_items)
    course_id = _first_safe_value(payload.course_id, trace.get("course_id"))
    module_id = _first_safe_value(payload.module_id, trace.get("module_id"))
    binding_id = _first_safe_value(payload.binding_id, trace.get("binding_id"))
    request_id = _first_safe_value(payload.request_id, trace.get("request_id"))
    role = _safe_role(payload.role or trace.get("role"))

    if not evidence_ids:
        status = RESULT_HOLD
        policy_result = "HOLD"
        hold_reason = "trace evidence_ids are required for no-DB trace explanation"
        summary = f"Trace {bridge_trace_id} is on HOLD for {role}: no safe evidence_ids were provided."
    else:
        status = RESULT_OK
        policy_result = "PASS"
        hold_reason = None
        summary_parts = [
            f"Trace {bridge_trace_id} is visible to {role} as a safe summary only.",
            f"Evidence count: {len(evidence_ids)}.",
            "Raw text included: false.",
            "Internal path included: false.",
        ]
        if course_id is not None:
            summary_parts.append(f"Course: {course_id}.")
        if module_id is not None:
            summary_parts.append(f"Module: {module_id}.")
        if binding_id is not None:
            summary_parts.append(f"Binding: {binding_id}.")
        summary = " ".join(summary_parts)

    return {
        "result_status": status,
        "request_id": request_id,
        "bridge_trace_id": bridge_trace_id,
        "course_id": course_id,
        "module_id": module_id,
        "binding_id": binding_id,
        "evidence_ids": evidence_ids,
        "policy_result": policy_result,
        "hold_reason": hold_reason,
        "visible_trace_summary": summary,
        "raw_text_included": False,
        "internal_path_included": False,
        "created_at": _created_at(),
    }


__all__ = [
    "router",
    "retrieve_bridge_evidence",
    "check_bridge_policy",
    "explain_bridge_trace",
    "BridgeEvidenceRequest",
    "BridgeEvidenceResponse",
    "BridgePolicyCheckRequest",
    "BridgePolicyCheckResponse",
    "BridgeTraceExplainRequest",
    "BridgeTraceExplainResponse",
]
