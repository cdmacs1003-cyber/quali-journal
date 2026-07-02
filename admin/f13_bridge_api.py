"""F13 no-DB Bridge API route.

This router exposes the existing in-memory Bridge guard utility over a
provided-evidence-only API boundary. It does not query DB, Warehouse, Library,
Skillup runtime, files, network, or runtime indexes.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from admin.f13_skillup_answer_hold_adapter import adapt_skillup_answer_hold_response
from admin.f13_runtime_guard import (
    RESULT_DENIED,
    RESULT_HOLD,
    RESULT_OK,
    detect_forbidden_fields,
    decide_bridge_result,
    decide_role_access_policy,
    normalize_rights_status,
    project_bridge_safe_evidence,
    validate_human_redacted_preflight_replay_evidence,
    zero_leak_counters,
)
from admin.f13_skillup_bridge import (
    skillup_answer_from_bridge_response,
    skillup_answer_from_request,
    skillup_feedback_queue_item_from_hold,
)


router = APIRouter(prefix="/api/f13/bridge", tags=["f13-bridge"])

_MAX_RETURN_ITEMS = 10
_REDACTED_PREFLIGHT_REPLAY_EVIDENCE_FIELD = "redacted_preflight_replay_evidence"
_SAFE_SHORT_ANSWER_FIELD = "safe_short_answer"
_SAFE_SHORT_ANSWER_MAX_CHARS = 800
_BETA_HOLD_SHORT_ANSWER = (
    "안전 검토 중입니다. 베타에서는 확정 답변이 아닌 상태 안내가 먼저 표시될 수 있습니다."
)
_UNSAFE_SHORT_ANSWER_MARKERS = (
    "raw_text",
    "raw text",
    "raw_prompt",
    "raw prompt",
    "raw_query",
    "raw query",
    "raw_answer",
    "raw answer",
    "full_json",
    "full json",
    "full_answer",
    "full answer",
    "internal_path",
    "internal path",
    "file://",
    "localhost",
    "127.0.0.1",
    "secret",
    "token",
    "credential",
    "cookie",
    "authorization",
    "h:\\",
    "c:\\",
)
_SCHEMA_REQUIRED_EVIDENCE_FIELDS = (
    "evidence_id",
    "bridge_trace_id",
    "safe_summary",
    "pointer_uri",
    "raw_text_policy",
    "rights_status",
)
_ROLE_POLICY_CONTEXT_FIELDS = (
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
)


class BridgeEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: Optional[str] = None
    purpose: str = "answer"
    requester_module: str = "Skillup"
    allowed_rights_status: List[str] = Field(default_factory=list)
    max_items: Optional[int] = Field(default=None, ge=1)
    evidence_items: Optional[List[Dict[str, Any]]] = None
    redacted_preflight_replay_evidence: Optional[Dict[str, Any]] = None


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
    model_config = ConfigDict(extra="allow")

    evidence: Optional[Dict[str, Any]] = None
    evidence_items: Optional[List[Dict[str, Any]]] = None
    evidence_id: Optional[str] = None
    bridge_trace_id: Optional[str] = None
    safe_summary: Optional[str] = None
    pointer_uri: Optional[str] = None
    raw_text_policy: Optional[str] = None
    rights_status: Optional[str] = None
    role: Optional[str] = None
    evidence_depth: Optional[str] = None
    course_id: Optional[str] = None
    module_id: Optional[str] = None
    binding_id: Optional[str] = None
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None
    cohort_id: Optional[str] = None
    license_entitlement_id: Optional[str] = None
    license_entitlement_status: Optional[str] = None
    requested_output_type: str = "safe_summary"
    requester_module: str = "Skillup"
    purpose: str = "answer"


class BridgePolicyCheckResponse(BaseModel):
    result_status: str
    bridge_trace_id: Optional[str]
    policy_result: str
    hold_reason: Optional[str]
    output_constraints: List[str]
    blocked_fields: List[str]
    role: Optional[str]
    evidence_depth: Optional[str]
    raw_text_export_count: int
    internal_path_leak_count: int
    raw_prompt_output_count: int
    secret_leak_count: int
    instructor_guide_raw_leak_count: int
    feedback_candidate_required: bool
    raw_text_included: bool
    internal_path_included: bool
    created_at: str


class BridgeTraceExplainRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    bridge_trace_id: Optional[str] = None
    role: Optional[str] = None
    evidence_depth: Optional[str] = None
    request_id: Optional[str] = None
    course_id: Optional[str] = None
    module_id: Optional[str] = None
    binding_id: Optional[str] = None
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None
    cohort_id: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None
    evidence_items: Optional[List[Dict[str, Any]]] = None


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
    role: Optional[str]
    evidence_depth: Optional[str]
    review_trace: Optional[Dict[str, Any]]
    audit_trace: Optional[Dict[str, Any]]
    raw_text_export_count: int
    internal_path_leak_count: int
    raw_prompt_output_count: int
    secret_leak_count: int
    instructor_guide_raw_leak_count: int
    feedback_candidate_required: bool
    feedback_candidate: Optional[Dict[str, Any]]
    visible_trace_summary: str
    raw_text_included: bool
    internal_path_included: bool
    created_at: str


class SkillupBridgeAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    bridge_response: Optional[Dict[str, Any]] = None
    request_payload: Optional[Dict[str, Any]] = None
    requester_module: str = "Skillup"



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


def _role_policy_context(*sources: Mapping[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in _ROLE_POLICY_CONTEXT_FIELDS:
            value = source.get(key)
            if not _is_missing(value):
                context[key] = value
    return context


def _role_policy_response_fields(decision: Mapping[str, Any] | None) -> Dict[str, Any]:
    source = decision if isinstance(decision, Mapping) else {}
    fields: Dict[str, Any] = {
        "role": source.get("role"),
        "evidence_depth": source.get("evidence_depth"),
    }
    counters = zero_leak_counters()
    for key, default in counters.items():
        fields[key] = int(source.get(key, default) or 0)
    return fields


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


def _trace_feedback_candidate(
    status: str,
    hold_reason: Optional[str],
    bridge_trace_id: Optional[str],
) -> Optional[Dict[str, str]]:
    if status == RESULT_OK:
        return None

    reason = _safe_label(hold_reason, max_length=160) or "Bridge trace explanation requires review"
    candidate = {
        "candidate_type": "BRIDGE_TRACE_REVIEW",
        "reason": reason,
        "next_action": "REVIEW_EVIDENCE_TRACE_POLICY",
    }
    safe_trace_id = _safe_trace_id(bridge_trace_id)
    if safe_trace_id is not None:
        candidate["bridge_trace_id"] = safe_trace_id
    return candidate


def _bounded_blocked_fields(fields: List[str]) -> List[str]:
    safe_fields: List[str] = []
    for field in fields[:10]:
        label = _safe_label(field, max_length=120)
        if label is not None:
            safe_fields.append(label)
    return safe_fields


def _policy_constraints(
    status: str,
    *,
    role: Optional[str] = None,
    evidence_depth: Optional[str] = None,
) -> List[str]:
    if status == RESULT_OK:
        constraints = [
            "SAFE_SUMMARY_ONLY",
            "NO_RAW_EXPORT",
            "NO_RAW_TEXT",
            "NO_INTERNAL_PATH",
            "BRIDGE_TRACE_REQUIRED",
            "ZERO_ROLE_LEAK_COUNTERS",
        ]
        if role:
            constraints.append(f"ROLE_{role.upper()}")
        if evidence_depth:
            constraints.append(f"EVIDENCE_DEPTH_{evidence_depth.upper()}")
        return constraints
    if status == RESULT_DENIED:
        return ["BLOCK_OUTPUT", "NO_RAW_EXPORT", "NO_RAW_TEXT", "NO_INTERNAL_PATH"]
    return [
        "HOLD_UNTIL_EVIDENCE_TRACE_RIGHTS_POLICY_PASS",
        "NO_RAW_EXPORT",
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
            *_ROLE_POLICY_CONTEXT_FIELDS,
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


def _safe_review_trace_metadata(
    role: Optional[str],
    evidence_ids: List[str],
    status: str,
    hold_reason: Optional[str],
) -> Optional[Dict[str, Any]]:
    if role not in {"reviewer", "admin"}:
        return None
    return {
        "visibility": "review_trace_safe_metadata",
        "evidence_match_status": "MATCHED" if evidence_ids else "MISSING",
        "hold_queue_status": "not_required" if status == RESULT_OK else "review_required",
        "policy_block_summary": None if status == RESULT_OK else (hold_reason or "Trace policy requires review"),
    }


def _safe_audit_trace_metadata(role: Optional[str]) -> Optional[Dict[str, Any]]:
    if role != "admin":
        return None
    return {
        "visibility": "audit_trace_safe_metadata",
        "approval_metadata_visible": True,
        "course_assignment_metadata_visible": True,
        "role_assignment_metadata_visible": True,
        "audit_metadata_visible": True,
        "raw_export_allowed": False,
    }


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


def _safe_skillup_pointer_uri(value: object) -> Optional[str]:
    pointer = _safe_label(value, max_length=200)
    if pointer is None or not pointer.startswith("pointer://"):
        return None
    lowered = pointer.lower()
    if any(marker in lowered for marker in ("file://", "h:\\", "c:\\", "localhost", "127.0.0.1")):
        return None
    return pointer


def _skillup_bridge_response_payload(request_payload: Dict[str, Any]) -> Dict[str, Any]:
    bridge_response = request_payload.get("bridge_response")
    if isinstance(bridge_response, dict):
        return dict(bridge_response)
    if any(
        key in request_payload
        for key in ("result_status", "evidence_items", "hold_reason", "feedback_candidate_required")
    ):
        return dict(request_payload)
    return {}


def _without_pass_claim_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"f13_pass", "track_a_pass", "beta_pass"}
    }


def _safe_short_answer_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    if not text or len(text) > _SAFE_SHORT_ANSWER_MAX_CHARS:
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in _UNSAFE_SHORT_ANSWER_MARKERS):
        return None
    return text


def _with_safe_short_answer(payload: Dict[str, Any]) -> Dict[str, Any]:
    result_status = str(payload.get("result_status") or "")
    answer_status = str(payload.get("answer_status") or "")
    if result_status == RESULT_OK and answer_status == "ANSWERED":
        safe_answer = _safe_short_answer_text(payload.get("answer"))
        if safe_answer is not None:
            payload[_SAFE_SHORT_ANSWER_FIELD] = safe_answer
    elif result_status == RESULT_HOLD or answer_status == "HOLD":
        payload[_SAFE_SHORT_ANSWER_FIELD] = _BETA_HOLD_SHORT_ANSWER
    return payload


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


@router.post("/skillup/bridge-answer")
def skillup_bridge_answer(payload: SkillupBridgeAnswerRequest) -> Dict[str, Any]:
    request_payload = _model_to_dict(payload)
    bridge_payload = _skillup_bridge_response_payload(request_payload)
    if bridge_payload:
        helper_result = skillup_answer_from_bridge_response(bridge_payload)
    else:
        helper_result = skillup_answer_from_request(request_payload.get("request_payload") or request_payload)

    response = _without_pass_claim_fields(dict(helper_result))
    response["created_at"] = _created_at()

    if response.get("result_status") == RESULT_OK:
        evidence_items = bridge_payload.get("evidence_items") or []
        if evidence_items and isinstance(evidence_items[0], dict):
            pointer_uri = _safe_skillup_pointer_uri(evidence_items[0].get("pointer_uri"))
            if pointer_uri is not None:
                response["pointer_uri"] = pointer_uri
        adapted = adapt_skillup_answer_hold_response(
            response,
            request_context=request_payload,
            bridge_payload=bridge_payload,
        )
        return _with_safe_short_answer(adapted)

    queue_source = {
        **response,
        "origin_module": request_payload.get("requester_module") or "Skillup",
        "origin_event_id": request_payload.get("origin_event_id") or response.get("bridge_trace_id"),
    }
    response["feedback_queue_item"] = skillup_feedback_queue_item_from_hold(queue_source)
    adapted = adapt_skillup_answer_hold_response(
        response,
        request_context=request_payload,
        bridge_payload=bridge_payload,
    )
    return _with_safe_short_answer(adapted)


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
            **_role_policy_response_fields(None),
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
                {**evidence, **_role_policy_context(request_payload, evidence)},
                requester_module=payload.requester_module,
                purpose=payload.purpose,
                enforce_role_access=True,
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
    selected_decision = (denied or holds or decisions or [{}])[0]
    selected_role = selected_decision.get("role")
    selected_depth = selected_decision.get("evidence_depth")

    return {
        "result_status": final_status,
        "bridge_trace_id": trace_ids[0] if trace_ids else None,
        "policy_result": _policy_label(final_status),
        "hold_reason": hold_reason,
        "output_constraints": _policy_constraints(
            final_status,
            role=selected_role if isinstance(selected_role, str) else None,
            evidence_depth=selected_depth if isinstance(selected_depth, str) else None,
        ),
        "blocked_fields": [],
        **_role_policy_response_fields(selected_decision),
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
        hold_reason = "forbidden fields or patterns detected"
        return {
            "result_status": RESULT_DENIED,
            "request_id": None,
            "bridge_trace_id": None,
            "course_id": None,
            "module_id": None,
            "binding_id": None,
            "evidence_ids": [],
            "policy_result": "DENIED",
            "hold_reason": hold_reason,
            **_role_policy_response_fields(None),
            "review_trace": None,
            "audit_trace": None,
            "feedback_candidate_required": True,
            "feedback_candidate": _trace_feedback_candidate(RESULT_DENIED, hold_reason, None),
            "visible_trace_summary": "Trace explanation denied by Bridge safety boundary.",
            "raw_text_included": False,
            "internal_path_included": False,
            "created_at": _created_at(),
        }

    trace = dict(payload.trace or {})
    evidence_items = [dict(item) for item in payload.evidence_items or [] if isinstance(item, dict)]
    first_evidence = evidence_items[0] if evidence_items else {}
    role_decision = decide_role_access_policy(_role_policy_context(request_payload, trace, first_evidence))
    if role_decision.get("result_status") != RESULT_OK:
        hold_reason = str(role_decision.get("hold_reason") or "HOLD_POLICY: trace role access requires review")
        status = str(role_decision.get("result_status") or RESULT_HOLD)
        safe_status = RESULT_DENIED if status == RESULT_DENIED else RESULT_HOLD
        return {
            "result_status": safe_status,
            "request_id": None,
            "bridge_trace_id": None,
            "course_id": None,
            "module_id": None,
            "binding_id": None,
            "evidence_ids": [],
            "policy_result": _policy_label(safe_status),
            "hold_reason": hold_reason,
            **_role_policy_response_fields(role_decision),
            "review_trace": None,
            "audit_trace": None,
            "feedback_candidate_required": True,
            "feedback_candidate": _trace_feedback_candidate(safe_status, hold_reason, None),
            "visible_trace_summary": "Trace explanation is on HOLD by role access policy.",
            "raw_text_included": False,
            "internal_path_included": False,
            "created_at": _created_at(),
        }

    bridge_trace_id = _safe_trace_id(
        payload.bridge_trace_id
        or trace.get("bridge_trace_id")
        or trace.get("promotion_trace_id")
    )
    if bridge_trace_id is None:
        hold_reason = "bridge_trace_id is required for no-DB trace explanation"
        return {
            "result_status": RESULT_HOLD,
            "request_id": _safe_label(payload.request_id),
            "bridge_trace_id": None,
            "course_id": _first_safe_value(payload.course_id, trace.get("course_id")),
            "module_id": _first_safe_value(payload.module_id, trace.get("module_id")),
            "binding_id": _first_safe_value(payload.binding_id, trace.get("binding_id")),
            "evidence_ids": [],
            "policy_result": "HOLD",
            "hold_reason": hold_reason,
            **_role_policy_response_fields(role_decision),
            "review_trace": _safe_review_trace_metadata(
                role_decision.get("role"),
                [],
                RESULT_HOLD,
                hold_reason,
            ),
            "audit_trace": _safe_audit_trace_metadata(role_decision.get("role")),
            "feedback_candidate_required": True,
            "feedback_candidate": _trace_feedback_candidate(RESULT_HOLD, hold_reason, None),
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
    role = role_decision.get("role")
    evidence_depth = role_decision.get("evidence_depth")

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
            f"Trace {bridge_trace_id} is visible to {role} as {evidence_depth} metadata only.",
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
        **_role_policy_response_fields(role_decision),
        "review_trace": _safe_review_trace_metadata(
            role if isinstance(role, str) else None,
            evidence_ids,
            status,
            hold_reason,
        ),
        "audit_trace": _safe_audit_trace_metadata(role if isinstance(role, str) else None),
        "feedback_candidate_required": status != RESULT_OK,
        "feedback_candidate": _trace_feedback_candidate(status, hold_reason, bridge_trace_id),
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
    "SkillupBridgeAnswerRequest",
    "skillup_bridge_answer",
]
