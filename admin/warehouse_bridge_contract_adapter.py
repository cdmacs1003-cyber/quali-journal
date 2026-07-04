from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from admin.f13_runtime_guard import (
    RAW_TEXT_POLICY_POINTER_ONLY,
    RAW_TEXT_POLICY_SUMMARY_ONLY,
    RESULT_DENIED,
    RESULT_HOLD,
    RESULT_OK,
    RIGHTS_LICENSED,
    RIGHTS_PUBLIC,
    detect_forbidden_fields,
    decide_bridge_result,
    normalize_raw_text_policy,
    normalize_rights_status,
    project_bridge_safe_evidence,
)


CREATED_AT = "1970-01-01T00:00:00Z"
WAREHOUSE_SOURCE_DOC_KIND = "WAREHOUSE_PROMOTION"
BRIDGE_TRACE_PREFIX = "btrace:warehouse:"

REQUIRED_CONTEXT_FIELDS = (
    "tenant_id",
    "organization_id",
    "cohort_id",
    "course_id",
    "module_id",
    "binding_id",
)

OPTIONAL_CONTEXT_FIELDS = (
    "actor_id",
    "reviewer_id",
    "role",
    "evidence_depth",
    "requested_output_type",
    "requested_action",
    "action",
    "trace_view",
    "export_type",
    "bridge_family",
    "bridge_id",
    "standard_pack_id",
    "request_id",
    "license_entitlement_id",
    "license_entitlement_status",
    "validation_shape_ids",
)

_DEFAULT_CONTEXT = {
    "role": "student",
    "evidence_depth": "student_safe",
    "requested_output_type": "safe_summary",
}

_SLUG_RE = re.compile(r"[^a-z0-9:._-]+")
_UNSAFE_VALUE_MARKERS = (
    "raw source text",
    "raw standard text",
    "full source text",
    "full standard text",
    "source_uri_or_path",
    "internal path",
    "proofpack",
    "backup path",
    "file://",
    "localhost",
    "127.0.0.1",
    "h:\\",
    "c:\\",
    "/home/",
    "/mnt/",
    "/tmp/",
    "brain.db",
    "graph.db",
    ".env",
    "secret",
    "token",
    "credential",
    "postgres://",
    "postgresql://",
    "mysql://",
    "sqlite://",
)
_STATUS_PROMOTED_TO_LIBRARY = {
    "PROMOTED",
    "APPROVED_FOR_LIBRARY",
    "APPROVED_FOR_LIBRARY_EVIDENCE",
}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _first_present(*values: Any) -> Any:
    for value in values:
        if isinstance(value, list | tuple):
            for item in value:
                if not _is_missing(item):
                    return item
            continue
        if not _is_missing(value):
            return value
    return None


def _has_unsafe_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(marker in lowered for marker in _UNSAFE_VALUE_MARKERS)


def _safe_label(value: Any, *, max_length: int = 160) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text or len(text) > max_length:
        return None
    if any(ord(char) < 32 for char in text):
        return None
    if _has_unsafe_value(text):
        return None
    if detect_forbidden_fields({"value": text}):
        return None
    return text


def _safe_summary(*values: Any) -> str | None:
    for value in values:
        text = _safe_label(value, max_length=2000)
        if text is not None:
            return text
    return None


def _safe_digest(*parts: Any) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _safe_slug_or_hash(value: Any) -> str:
    text = str(value or "").strip().lower()
    slug = _SLUG_RE.sub("-", text).strip("-._:")
    if not slug or _has_unsafe_value(slug):
        slug = _safe_digest(value)
    return slug[:96]


def _bridge_trace_id(promotion_trace_id: Any) -> str:
    return f"{BRIDGE_TRACE_PREFIX}{_safe_slug_or_hash(promotion_trace_id)}"


def _safe_context_value(context: Mapping[str, Any], key: str) -> Any:
    if key == "validation_shape_ids":
        value = context.get(key)
        if not isinstance(value, list | tuple | set):
            return []
        safe_ids = [_safe_label(item, max_length=120) for item in value]
        return [item for item in safe_ids if item]
    return _safe_label(context.get(key), max_length=160)


def validate_warehouse_bridge_context(caller_context: Mapping[str, Any] | None) -> dict[str, Any]:
    source = _as_mapping(caller_context)
    context: dict[str, Any] = {}
    missing: list[str] = []

    for key in REQUIRED_CONTEXT_FIELDS:
        value = _safe_context_value(source, key)
        if value is None:
            missing.append(key)
        else:
            context[key] = value

    for key, value in _DEFAULT_CONTEXT.items():
        context[key] = _safe_context_value(source, key) or value

    for key in OPTIONAL_CONTEXT_FIELDS:
        if key in _DEFAULT_CONTEXT:
            continue
        value = _safe_context_value(source, key)
        if value not in (None, [], ""):
            context[key] = value

    if missing:
        return {
            "result_status": RESULT_HOLD,
            "hold_reason": "Warehouse Bridge context requires explicit downstream scope.",
            "missing_fields": missing,
            "context": context,
        }
    return {
        "result_status": RESULT_OK,
        "hold_reason": None,
        "missing_fields": [],
        "context": context,
    }


def _promotion_surfaces(promotion: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = _as_mapping(promotion)
    trace = _as_mapping(
        _first_present(
            source.get("trace"),
            source.get("promotion_trace"),
            source.get("promotion"),
            source,
        )
    )
    item = _as_mapping(_first_present(source.get("item"), source.get("warehouse_item"), source))
    return trace, item


def _item_promotion(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(item.get("promotion"))


def _item_approval(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(_first_present(item.get("approval"), item.get("review"), {}))


def _safe_identifier(*values: Any, max_length: int = 160) -> str | None:
    for value in values:
        if isinstance(value, list | tuple):
            for item in value:
                label = _safe_label(item, max_length=max_length)
                if label is not None:
                    return label
            continue
        label = _safe_label(value, max_length=max_length)
        if label is not None:
            return label
    return None


def _evidence_id(trace: Mapping[str, Any], item_promotion: Mapping[str, Any]) -> str | None:
    return _safe_identifier(
        trace.get("evidence_id"),
        trace.get("promoted_evidence_id"),
        trace.get("promoted_evidence_ids"),
        item_promotion.get("evidence_id"),
        item_promotion.get("promoted_evidence_id"),
        item_promotion.get("promoted_evidence_ids"),
        max_length=120,
    )


def _library_id(trace: Mapping[str, Any], item_promotion: Mapping[str, Any]) -> str | None:
    return _safe_identifier(
        trace.get("promoted_library_id"),
        item_promotion.get("promoted_library_id"),
        trace.get("library_id"),
        item_promotion.get("library_id"),
        max_length=120,
    )


def _approval_record_id(trace: Mapping[str, Any], item: Mapping[str, Any]) -> str | None:
    approval = _item_approval(item)
    return _safe_identifier(
        trace.get("approval_record_id"),
        trace.get("approval_event_id"),
        approval.get("approval_record_id"),
        approval.get("approval_event_id"),
        item.get("approval_record_id"),
        item.get("approval_event_id"),
        max_length=120,
    )


def _safe_current_status(trace: Mapping[str, Any], item: Mapping[str, Any]) -> str:
    status = str(_first_present(item.get("status"), trace.get("source_item_status"), "")).strip()
    normalized = status.upper().replace("-", "_").replace(" ", "_")
    if normalized in _STATUS_PROMOTED_TO_LIBRARY or trace.get("promoted_library_id"):
        return "APPROVED_FOR_LIBRARY"
    return normalized or "HOLD_REVIEW_REQUIRED"


def _safe_raw_text_policy(
    trace: Mapping[str, Any],
    item: Mapping[str, Any],
    item_promotion: Mapping[str, Any],
    rights_status: str,
) -> str:
    explicit = _first_present(
        trace.get("raw_text_policy"),
        item_promotion.get("raw_text_policy"),
        item.get("raw_text_policy"),
    )
    if explicit:
        return normalize_raw_text_policy(explicit)

    visibility = str(item.get("visibility") or "").strip().lower()
    if rights_status == RIGHTS_PUBLIC and visibility == "public_summary_allowed":
        return RAW_TEXT_POLICY_SUMMARY_ONLY
    return RAW_TEXT_POLICY_POINTER_ONLY


def _policy_flags(
    *,
    bridge_decision: Mapping[str, Any],
    evidence_id: str | None,
    safe_summary: str | None,
    rights_status: str,
    sensitivity: str,
) -> dict[str, bool]:
    return {
        "evidence_required_pass": bool(evidence_id and safe_summary),
        "raw_leak_pass": True,
        "rights_pass": bridge_decision.get("result_status") != RESULT_DENIED,
        "sensitivity_pass": sensitivity not in {"PRIVATE", "RESTRICTED", "SECRET"},
        "pointer_only_policy": True,
        "metadata_only": True,
    }


def _bridge_response(
    *,
    status: str,
    hold_reason: str | None,
    evidence_items: list[dict[str, Any]],
    policy_result: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "result_status": status,
        "evidence_items": evidence_items if status == RESULT_OK else [],
        "hold_reason": None if status == RESULT_OK else hold_reason,
        "feedback_candidate_required": status != RESULT_OK,
        "raw_text_included": False,
        "internal_path_included": False,
        "policy_result": dict(policy_result),
        "created_at": CREATED_AT,
    }


def _overall_status(
    *,
    bridge_status: str,
    context_status: str,
    sensitivity: str,
    promotion_trace_id: str | None,
) -> tuple[str, str | None]:
    if sensitivity == "SECRET":
        return RESULT_DENIED, "Warehouse promotion sensitivity is not Bridge-safe."
    if not promotion_trace_id:
        return RESULT_HOLD, "Warehouse promotion_trace_id is required."
    if bridge_status == RESULT_DENIED:
        return RESULT_DENIED, "Warehouse promotion failed Bridge safety projection."
    if bridge_status == RESULT_HOLD:
        return RESULT_HOLD, "Warehouse promotion is missing Bridge-required evidence."
    if context_status != RESULT_OK:
        return RESULT_HOLD, "Warehouse Bridge context requires explicit downstream scope."
    if sensitivity in {"PRIVATE", "RESTRICTED"}:
        return RESULT_HOLD, "Warehouse promotion sensitivity requires review."
    return RESULT_OK, None


def map_warehouse_promotion_to_bridge_payload(
    promotion: Mapping[str, Any] | None,
    caller_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    trace, item = _promotion_surfaces(promotion)
    item_promotion = _item_promotion(item)
    context_validation = validate_warehouse_bridge_context(caller_context)
    context = context_validation["context"]

    promotion_trace_id = _safe_identifier(
        trace.get("promotion_trace_id"),
        item_promotion.get("promotion_trace_id"),
        max_length=120,
    )
    bridge_trace_id = _bridge_trace_id(promotion_trace_id or "missing-promotion-trace")
    warehouse_item_id = _safe_identifier(
        trace.get("warehouse_item_id"),
        item.get("warehouse_item_id"),
        max_length=120,
    )
    evidence_id = _evidence_id(trace, item_promotion)
    library_id = _library_id(trace, item_promotion)
    raw_hash = _safe_identifier(trace.get("raw_hash"), item.get("raw_hash"), max_length=160)
    safe_summary = _safe_summary(item.get("summary"), trace.get("summary"), item.get("title"))
    rights_status = normalize_rights_status(_first_present(item.get("rights_status"), trace.get("rights_status")))
    sensitivity = str(_safe_identifier(item.get("sensitivity"), trace.get("sensitivity")) or "NOT_VERIFIED")
    sensitivity = sensitivity.upper().replace("-", "_").replace(" ", "_")
    raw_text_policy = _safe_raw_text_policy(trace, item, item_promotion, rights_status)
    approval_record_id = _approval_record_id(trace, item)
    current_status = _safe_current_status(trace, item)
    pointer_uri = f"pointer://warehouse/evidence/{evidence_id}" if evidence_id else None

    bridge_evidence_item = {
        "evidence_id": evidence_id or "",
        "bridge_trace_id": bridge_trace_id,
        "safe_summary": safe_summary or "",
        "pointer_uri": pointer_uri or "",
        "raw_text_policy": raw_text_policy,
        "rights_status": rights_status,
        "source_doc_kind": WAREHOUSE_SOURCE_DOC_KIND,
        "validation_shape_ids": context.get("validation_shape_ids", []),
    }
    bridge_evidence_item = project_bridge_safe_evidence(bridge_evidence_item)
    bridge_decision = decide_bridge_result(bridge_evidence_item)
    policy_result = _policy_flags(
        bridge_decision=bridge_decision,
        evidence_id=evidence_id,
        safe_summary=safe_summary,
        rights_status=rights_status,
        sensitivity=sensitivity,
    )
    status, hold_reason = _overall_status(
        bridge_status=str(bridge_decision.get("result_status") or RESULT_HOLD),
        context_status=str(context_validation.get("result_status") or RESULT_HOLD),
        sensitivity=sensitivity,
        promotion_trace_id=promotion_trace_id,
    )

    bridge_response = _bridge_response(
        status=str(bridge_decision.get("result_status") or RESULT_HOLD),
        hold_reason=bridge_decision.get("hold_reason"),
        evidence_items=[bridge_evidence_item],
        policy_result=policy_result,
    )

    skillup_evidence_item = {
        **bridge_evidence_item,
        **{key: value for key, value in context.items() if key in REQUIRED_CONTEXT_FIELDS + OPTIONAL_CONTEXT_FIELDS},
    }
    skillup_bridge_response = _bridge_response(
        status=status,
        hold_reason=hold_reason,
        evidence_items=[skillup_evidence_item],
        policy_result=policy_result,
    )

    course_binding_payload = {
        "course_id": context.get("course_id", ""),
        "module_id": context.get("module_id", ""),
        "tenant_id": context.get("tenant_id", ""),
        "organization_id": context.get("organization_id", ""),
        "cohort_id": context.get("cohort_id", ""),
        "bridge_family": context.get("bridge_family", "warehouse"),
        "bridge_id": context.get("bridge_id", "bridge:warehouse"),
        "standard_pack_id": context.get("standard_pack_id", "SPK_WAREHOUSE_BRIDGE"),
        "request_id": context.get("request_id", ""),
        "trace_id": bridge_trace_id,
        "bridge_trace_id": bridge_trace_id,
        "library_node_id": library_id or "",
        "evidence_id": evidence_id or "",
        "approval_record_id": approval_record_id or "",
        "current_status": current_status,
        "rights_status": rights_status,
        "raw_text_policy": raw_text_policy,
        "validation_shape_ids": context.get("validation_shape_ids", []),
        "license_entitlement_id": context.get("license_entitlement_id", ""),
        "license_entitlement_status": context.get("license_entitlement_status", ""),
    }

    return {
        "result_status": status,
        "hold_reason": hold_reason,
        "context_validation": context_validation,
        "bridge_trace_id": bridge_trace_id,
        "bridge_evidence_item": bridge_evidence_item,
        "bridge_response": bridge_response,
        "skillup_bridge_response": skillup_bridge_response,
        "course_binding_payload": course_binding_payload,
        "safe_metadata": {
            "warehouse_item_id": warehouse_item_id,
            "promotion_trace_id": promotion_trace_id,
            "promoted_library_id": library_id,
            "evidence_id": evidence_id,
            "raw_hash": raw_hash,
            "rights_status": rights_status,
            "sensitivity": sensitivity,
            "review_status": current_status,
            "approval_record_id": approval_record_id,
        },
        "raw_text_included": False,
        "internal_path_included": False,
        "db_access_executed": False,
        "network_access_executed": False,
        "runtime_access_executed": False,
    }


def build_bridge_contract_from_warehouse_promotion(
    promotion: Mapping[str, Any] | None,
    caller_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return map_warehouse_promotion_to_bridge_payload(promotion, caller_context)


__all__ = [
    "BRIDGE_TRACE_PREFIX",
    "REQUIRED_CONTEXT_FIELDS",
    "build_bridge_contract_from_warehouse_promotion",
    "map_warehouse_promotion_to_bridge_payload",
    "validate_warehouse_bridge_context",
]
