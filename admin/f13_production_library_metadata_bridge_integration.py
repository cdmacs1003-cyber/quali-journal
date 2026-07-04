"""Pure integration projection for production Library metadata adapter records.

This helper stays outside route/runtime wiring. It consumes in-memory metadata
fixture records or already-adapted R9ZNW-332 adapter records and emits a
conservative Bridge evidence projection that remains HOLD until rights and
semantic summaries are verified elsewhere.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from admin.f13_production_library_metadata_bridge_adapter import (
    ADAPTER_POLICY,
    ADAPTER_SOURCE,
    DUPLICATE_STATUS,
    RIGHTS_STATUS_DECISION,
    SUMMARY_SOURCE,
    adapt_metadata_fixture_records,
    bridge_evidence_item_from_adapter_record,
    validate_bridge_skeleton_record,
)
from admin.f13_runtime_guard import (
    RAW_TEXT_POLICY_POINTER_ONLY,
    RESULT_HOLD,
    RIGHTS_NOT_VERIFIED,
    decide_bridge_result,
    project_bridge_safe_evidence,
)


INTEGRATION_SOURCE = "PRODUCTION_LIBRARY_METADATA_BRIDGE_INTEGRATION_DRAFT"
INTEGRATION_POLICY = "HOLD_BRIDGE_PROJECTION_UNTIL_RIGHTS_VERIFIED"

_BRIDGE_REQUIRED_FIELDS = (
    "evidence_id",
    "bridge_trace_id",
    "safe_summary",
    "pointer_uri",
    "raw_text_policy",
    "rights_status",
)
_RAW_BODY_FIELD_NAMES = {
    "raw_text",
    "full_text",
    "source_text",
    "paid_standard_text",
}
_ALLOWED_RAW_POLICY_FIELD_NAMES = {"raw_text_policy", "raw_text_exposed"}
_ALLOWED_PATH_AUDIT_FIELD_NAMES = {"production_path_exposed"}
_SECRET_MARKERS = (
    ".env",
    "credential",
    "service-account",
    "service_account",
    "secret",
    "token",
    "bearer",
)
_KEY_MARKER_RE = re.compile(r"(^|[_\-.])key($|[_\-.])", re.IGNORECASE)
_DB_MARKERS = (
    "brain.db",
    "graph.db",
    ".sqlite",
    ".sqlite3",
    "sqlite://",
    "postgres://",
    "postgresql://",
)
_READINESS_MARKERS = (
    "f13_pass=true",
    "track_a_pass=true",
    "beta_pass=true",
    "release_ready=true",
    "production_ready=true",
)
_ABSOLUTE_PATH_RE = re.compile(r"(?i)\b[a-z]:\\")
_PRODUCTION_LIBRARY_RE = re.compile(
    r"(?i)h:[\\/](?:장기기억|janggigieok)[\\/]library"
)
_PRODUCTION_DB_RE = re.compile(
    r"(?i)h:[\\/](?:장기기억|janggigieok)[\\/](?:brain|graph)\.db"
)
_PROMOTED_RIGHTS = {
    "PUBLIC",
    "INTERNAL",
    "LICENSED",
    "CUSTOMER_CONFIDENTIAL",
    "RESTRICTED",
    "UNKNOWN",
    "PASS",
    "OK",
    "VERIFIED",
    "APPROVED",
}


def integrate_adapter_records_to_bridge_evidence(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project adapter or fixture-like records into conservative Bridge evidence."""

    adapter_records, adapter_rejections = _ensure_adapter_records(records)
    projected_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = list(adapter_rejections)

    for index, record in enumerate(adapter_records):
        try:
            projected = build_bridge_hold_projection_from_adapter_record(record)
        except ValueError as exc:
            rejected_records.append(
                {
                    "record_index": index,
                    "evidence_id": _safe_string(record.get("evidence_id")),
                    "reason_codes": _reason_codes_from_exception(exc),
                }
            )
            continue
        validation = validate_integration_projection(projected)
        if validation["is_valid"]:
            projected_records.append(projected)
        else:
            rejected_records.append(
                {
                    "record_index": index,
                    "evidence_id": _safe_string(record.get("evidence_id")),
                    "reason_codes": validation["reason_codes"],
                }
            )

    return {
        "result_status": RESULT_HOLD,
        "integration_source": INTEGRATION_SOURCE,
        "integration_policy": INTEGRATION_POLICY,
        "processed_count": len(records),
        "adapter_record_count": len(adapter_records),
        "projected_count": len(projected_records),
        "rejected_count": len(rejected_records),
        "hold_reason": "rights_status is not verified",
        "feedback_candidate_required": True,
        "raw_text_included": False,
        "internal_path_included": False,
        "evidence_items": projected_records,
        "rejected_records": rejected_records,
    }


def build_bridge_hold_projection_from_adapter_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one HOLD projection from a validated adapter record."""

    adapter_validation = validate_bridge_skeleton_record(record)
    if not adapter_validation["is_valid"]:
        raise ValueError(",".join(adapter_validation["reason_codes"]))

    bridge_input = bridge_evidence_item_from_adapter_record(record)
    for optional_field in ("source_doc_kind", "validation_shape_ids"):
        if optional_field in record:
            bridge_input[optional_field] = record[optional_field]

    bridge_projected_item = project_bridge_safe_evidence(bridge_input)
    if not _has_bridge_required_fields(bridge_projected_item):
        raise ValueError("BRIDGE_PROJECTED_ITEM_MISSING_REQUIRED_FIELDS")

    bridge_decision = decide_bridge_result(bridge_projected_item)
    if bridge_decision.get("result_status") != RESULT_HOLD:
        raise ValueError("BRIDGE_PROJECTED_ITEM_DID_NOT_HOLD")

    projection: dict[str, Any] = {
        "evidence_id": bridge_projected_item["evidence_id"],
        "bridge_trace_id": bridge_projected_item["bridge_trace_id"],
        "safe_summary": bridge_projected_item["safe_summary"],
        "pointer_uri": bridge_projected_item["pointer_uri"],
        "raw_text_policy": RAW_TEXT_POLICY_POINTER_ONLY,
        "rights_status": RIGHTS_NOT_VERIFIED,
        "rights_status_decision": RIGHTS_STATUS_DECISION,
        "summary_source": SUMMARY_SOURCE,
        "semantic_summary_verified": False,
        "raw_text_exposed": False,
        "production_path_exposed": False,
        "adapter_policy": ADAPTER_POLICY,
        "integration_source": INTEGRATION_SOURCE,
        "integration_policy": INTEGRATION_POLICY,
        "bridge_policy_result": RESULT_HOLD,
        "bridge_hold_reason": bridge_decision.get("hold_reason"),
        "feedback_candidate_required": True,
        "primary_adapter_candidate": record.get("primary_adapter_candidate") is not False,
        "duplicate_status": _safe_string(record.get("duplicate_status")),
        "duplicate_decision": _safe_string(record.get("duplicate_decision")),
        "bridge_projected_item": bridge_projected_item,
    }

    for optional_field in (
        "source_label",
        "standard_family",
        "doc_id",
        "revision",
        "section_label",
        "page_hint",
        "tags",
        "validation_shape_ids",
    ):
        if optional_field in record:
            projection[optional_field] = _safe_optional_value(record.get(optional_field))

    return projection


def validate_integration_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate integration projection without treating it as a route response."""

    reason_codes = _collect_unsafe_reason_codes(record)

    for field in _BRIDGE_REQUIRED_FIELDS:
        if not _safe_string(record.get(field)):
            reason_codes.append(f"MISSING_REQUIRED_FIELD:{field}")
    if record.get("rights_status") != RIGHTS_NOT_VERIFIED:
        reason_codes.append("RIGHTS_STATUS_NOT_HELD")
    if _safe_string(record.get("rights_status")).upper() in _PROMOTED_RIGHTS:
        reason_codes.append("RIGHTS_STATUS_PROMOTED")
    if record.get("rights_status_decision") != RIGHTS_STATUS_DECISION:
        reason_codes.append("RIGHTS_STATUS_DECISION_NOT_HOLD")
    if record.get("raw_text_policy") != RAW_TEXT_POLICY_POINTER_ONLY:
        reason_codes.append("RAW_TEXT_POLICY_NOT_POINTER_ONLY")
    if record.get("summary_source") != SUMMARY_SOURCE:
        reason_codes.append("SUMMARY_SOURCE_NOT_METADATA_ONLY")
    if record.get("semantic_summary_verified") is not False:
        reason_codes.append("SEMANTIC_SUMMARY_VERIFIED_NOT_FALSE")
    if record.get("raw_text_exposed") is not False:
        reason_codes.append("RAW_TEXT_EXPOSED_NOT_FALSE")
    if record.get("production_path_exposed") is not False:
        reason_codes.append("PRODUCTION_PATH_EXPOSED_NOT_FALSE")
    if record.get("adapter_policy") != ADAPTER_POLICY:
        reason_codes.append("ADAPTER_POLICY_NOT_PRESERVED")
    if record.get("integration_policy") != INTEGRATION_POLICY:
        reason_codes.append("INTEGRATION_POLICY_NOT_HOLD")
    if record.get("bridge_policy_result") != RESULT_HOLD:
        reason_codes.append("BRIDGE_POLICY_RESULT_NOT_HOLD")

    bridge_projected_item = record.get("bridge_projected_item")
    if not isinstance(bridge_projected_item, Mapping):
        reason_codes.append("MISSING_BRIDGE_PROJECTED_ITEM")
    elif not _has_bridge_required_fields(bridge_projected_item):
        reason_codes.append("BRIDGE_PROJECTED_ITEM_MISSING_REQUIRED_FIELDS")
    elif decide_bridge_result(bridge_projected_item).get("result_status") != RESULT_HOLD:
        reason_codes.append("BRIDGE_PROJECTED_ITEM_DID_NOT_HOLD")

    if record.get("duplicate_status") == DUPLICATE_STATUS and record.get("primary_adapter_candidate") is not False:
        reason_codes.append("DUPLICATE_PROMOTED_TO_PRIMARY")

    return {
        "is_valid": not reason_codes,
        "result_status": RESULT_HOLD if not reason_codes else "REJECT",
        "reason_codes": sorted(set(reason_codes)),
    }


def summarize_integration_rejections(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return only rejection summary data for caller diagnostics."""

    result = integrate_adapter_records_to_bridge_evidence(records)
    return {
        "result_status": result["result_status"],
        "processed_count": result["processed_count"],
        "projected_count": result["projected_count"],
        "rejected_count": result["rejected_count"],
        "rejected_records": result["rejected_records"],
    }


def _ensure_adapter_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    copied_records = [dict(record) for record in records]
    if all(_looks_like_adapter_record(record) for record in copied_records):
        return copied_records, []

    adapted_result = adapt_metadata_fixture_records(copied_records)
    return list(adapted_result["records"]), list(adapted_result["rejected_records"])


def _looks_like_adapter_record(record: Mapping[str, Any]) -> bool:
    return (
        record.get("adapter_source") == ADAPTER_SOURCE
        and record.get("adapter_policy") == ADAPTER_POLICY
        and isinstance(record.get("bridge_evidence_item"), Mapping)
    )


def _has_bridge_required_fields(record: Mapping[str, Any]) -> bool:
    return all(_safe_string(record.get(field)) for field in _BRIDGE_REQUIRED_FIELDS)


def _collect_unsafe_reason_codes(value: Any) -> list[str]:
    reasons: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, raw_value in node.items():
                key = str(raw_key)
                lowered_key = key.lower()
                if lowered_key in _RAW_BODY_FIELD_NAMES:
                    reasons.append(f"RAW_BODY_FIELD_PRESENT:{key}")
                elif "raw_text" in lowered_key and lowered_key not in _ALLOWED_RAW_POLICY_FIELD_NAMES:
                    reasons.append(f"RAW_TEXT_FIELD_PRESENT:{key}")
                elif "path" in lowered_key and lowered_key not in _ALLOWED_PATH_AUDIT_FIELD_NAMES:
                    reasons.append(f"PATH_FIELD_PRESENT:{key}")
                if _is_secret_like_marker(lowered_key):
                    reasons.append(f"SECRET_MARKER_FIELD:{key}")
                visit(raw_value)
            return
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if isinstance(node, str):
            lowered = node.lower()
            if _ABSOLUTE_PATH_RE.search(node):
                reasons.append("ABSOLUTE_FILESYSTEM_PATH_VALUE")
            if _PRODUCTION_LIBRARY_RE.search(node):
                reasons.append("PRODUCTION_LIBRARY_PATH_VALUE")
            if _PRODUCTION_DB_RE.search(node) or any(marker in lowered for marker in _DB_MARKERS):
                reasons.append("DB_MARKER_VALUE")
            if _is_secret_like_marker(lowered):
                reasons.append("SECRET_MARKER_VALUE")
            for marker in _READINESS_MARKERS:
                if marker in lowered:
                    reasons.append(f"BROAD_READINESS_MARKER:{marker}")

    visit(value)
    return reasons


def _is_secret_like_marker(value: str) -> bool:
    return any(marker in value for marker in _SECRET_MARKERS) or bool(_KEY_MARKER_RE.search(value))


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_optional_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_safe_string(item) for item in value if _safe_string(item)]
    if value is None:
        return "NOT_VERIFIED"
    return _safe_string(value)


def _reason_codes_from_exception(exc: ValueError) -> list[str]:
    reason = str(exc).strip()
    if not reason:
        return ["INTEGRATION_REJECTION"]
    return [item for item in reason.split(",") if item]


__all__ = [
    "INTEGRATION_POLICY",
    "INTEGRATION_SOURCE",
    "build_bridge_hold_projection_from_adapter_record",
    "integrate_adapter_records_to_bridge_evidence",
    "summarize_integration_rejections",
    "validate_integration_projection",
]
