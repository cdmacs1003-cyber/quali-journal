"""No-DB adapter for production Library metadata fixture records.

This module only adapts metadata-only fixture records produced by the
R9ZNW-331 normalization packet. It does not read the production Library root,
open databases, start runtime code, or verify rights/semantic summaries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from admin.f13_runtime_guard import (
    RAW_TEXT_POLICY_POINTER_ONLY,
    RESULT_HOLD,
    RIGHTS_NOT_VERIFIED,
    decide_bridge_result,
    normalize_raw_text_policy,
    normalize_rights_status,
)


ADAPTER_SOURCE = "PRODUCTION_LIBRARY_METADATA_FIXTURE"
ADAPTER_POLICY = "HOLD_UNTIL_RIGHTS_VERIFIED"
SUMMARY_SOURCE = "METADATA_DERIVED_NOT_SEMANTIC"
RIGHTS_STATUS_DECISION = "HOLD_RIGHTS_NOT_VERIFIED"
PRIMARY_STATUS = "PRIMARY_CANDIDATE_HOLD_UNTIL_RIGHTS_VERIFIED"
DUPLICATE_STATUS = "DUPLICATE_HOLD_NOT_PRIMARY"

_BRIDGE_REQUIRED_FIELDS = (
    "evidence_id",
    "bridge_trace_id",
    "safe_summary",
    "pointer_uri",
    "raw_text_policy",
    "rights_status",
)
_ADAPTER_REQUIRED_FIELDS = (
    *_BRIDGE_REQUIRED_FIELDS,
    "source_label",
    "standard_family",
    "doc_id",
)
_OPTIONAL_SAFE_FIELDS = ("revision", "section_label", "page_hint", "tags")
_RAW_BODY_FIELD_NAMES = {
    "raw_text",
    "full_text",
    "source_text",
    "paid_standard_text",
}
_ALLOWED_RAW_POLICY_FIELD_NAMES = {"raw_text_policy", "raw_text_exposed"}
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
_FORBIDDEN_PRODUCTION_LIBRARY_RE = re.compile(
    r"(?i)h:[\\/](?:장기기억|janggigieok)[\\/]library"
)
_FORBIDDEN_DB_PATH_RE = re.compile(
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


def load_metadata_fixture(path: str | Path) -> list[dict[str, Any]]:
    """Load an explicit R9ZNW-331 metadata fixture path.

    The caller supplies the fixture path. Production Library roots, DB-like
    paths, and secret-like filenames are rejected before open.
    """

    fixture_path = Path(path)
    _raise_for_forbidden_fixture_path(str(fixture_path))
    with fixture_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _extract_records(payload)


def adapt_metadata_fixture_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Adapt metadata-only fixture records into conservative HOLD records."""

    adapted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        rejection = classify_adapter_rejection(record)
        if rejection["accepted"]:
            adapted_records.append(_adapt_single_record(record, index))
        else:
            rejected_records.append(
                {
                    "record_index": index,
                    "evidence_id": _safe_string(record.get("evidence_id")),
                    "reason_codes": rejection["reason_codes"],
                }
            )

    return {
        "result_status": RESULT_HOLD,
        "adapter_source": ADAPTER_SOURCE,
        "adapter_policy": ADAPTER_POLICY,
        "processed_count": len(records),
        "adapted_count": len(adapted_records),
        "held_count": len(adapted_records),
        "rejected_count": len(rejected_records),
        "records": adapted_records,
        "rejected_records": rejected_records,
    }


def validate_bridge_skeleton_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an adapted record without promoting rights or summaries."""

    reason_codes = _collect_unsafe_reason_codes(record)

    for field in _ADAPTER_REQUIRED_FIELDS:
        if not _safe_string(record.get(field)):
            reason_codes.append(f"MISSING_REQUIRED_FIELD:{field}")

    if normalize_rights_status(record.get("rights_status")) != RIGHTS_NOT_VERIFIED:
        reason_codes.append("RIGHTS_STATUS_NOT_HELD")
    if _safe_string(record.get("rights_status")).upper() in _PROMOTED_RIGHTS:
        reason_codes.append("RIGHTS_STATUS_PROMOTED")
    if record.get("rights_status_decision") != RIGHTS_STATUS_DECISION:
        reason_codes.append("RIGHTS_STATUS_DECISION_NOT_HOLD")
    if normalize_raw_text_policy(record.get("raw_text_policy")) != RAW_TEXT_POLICY_POINTER_ONLY:
        reason_codes.append("RAW_TEXT_POLICY_NOT_POINTER_ONLY")
    if record.get("summary_source") != SUMMARY_SOURCE:
        reason_codes.append("SUMMARY_SOURCE_NOT_METADATA_ONLY")
    if record.get("semantic_summary_verified") is not False:
        reason_codes.append("SEMANTIC_SUMMARY_VERIFIED_NOT_FALSE")
    if record.get("raw_text_exposed") is not False:
        reason_codes.append("RAW_TEXT_EXPOSED_NOT_FALSE")
    if record.get("production_path_exposed") is not False:
        reason_codes.append("PRODUCTION_PATH_EXPOSED_NOT_FALSE")
    if record.get("adapter_source") != ADAPTER_SOURCE:
        reason_codes.append("ADAPTER_SOURCE_MISMATCH")
    if record.get("adapter_policy") != ADAPTER_POLICY:
        reason_codes.append("ADAPTER_POLICY_MISMATCH")
    if record.get("bridge_policy_result") != RESULT_HOLD:
        reason_codes.append("BRIDGE_POLICY_RESULT_NOT_HOLD")

    bridge_item = record.get("bridge_evidence_item")
    if not isinstance(bridge_item, Mapping):
        reason_codes.append("MISSING_BRIDGE_EVIDENCE_ITEM")
    else:
        for field in _BRIDGE_REQUIRED_FIELDS:
            if not _safe_string(bridge_item.get(field)):
                reason_codes.append(f"MISSING_BRIDGE_FIELD:{field}")
        if set(bridge_item) != set(_BRIDGE_REQUIRED_FIELDS):
            reason_codes.append("BRIDGE_EVIDENCE_ITEM_SCHEMA_FIELD_SET_MISMATCH")
        if decide_bridge_result(dict(bridge_item))["result_status"] != RESULT_HOLD:
            reason_codes.append("BRIDGE_GUARD_DID_NOT_HOLD")

    if record.get("duplicate_status") == DUPLICATE_STATUS and record.get("primary_adapter_candidate") is not False:
        reason_codes.append("DUPLICATE_PROMOTED_TO_PRIMARY")

    return {
        "is_valid": not reason_codes,
        "result_status": RESULT_HOLD if not reason_codes else "REJECT",
        "reason_codes": sorted(set(reason_codes)),
    }


def classify_adapter_rejection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify whether a fixture record can be adapted safely."""

    reason_codes = _collect_unsafe_reason_codes(record)

    for field in _ADAPTER_REQUIRED_FIELDS:
        if not _safe_string(record.get(field)):
            reason_codes.append(f"MISSING_REQUIRED_FIELD:{field}")

    rights_status = _safe_string(record.get("rights_status")).upper()
    if normalize_rights_status(record.get("rights_status")) != RIGHTS_NOT_VERIFIED:
        reason_codes.append("RIGHTS_STATUS_NOT_HELD")
    if rights_status in _PROMOTED_RIGHTS:
        reason_codes.append("RIGHTS_STATUS_PROMOTED")
    if _safe_string(record.get("rights_status_decision")) not in {
        RIGHTS_STATUS_DECISION,
        "",
    }:
        reason_codes.append("RIGHTS_STATUS_DECISION_NOT_HOLD")
    if normalize_raw_text_policy(record.get("raw_text_policy")) != RAW_TEXT_POLICY_POINTER_ONLY:
        reason_codes.append("RAW_TEXT_POLICY_NOT_POINTER_ONLY")
    if record.get("semantic_summary_verified") is True:
        reason_codes.append("SEMANTIC_SUMMARY_VERIFIED_TRUE")
    if _safe_string(record.get("summary_source")) not in {SUMMARY_SOURCE, ""}:
        reason_codes.append("SUMMARY_SOURCE_NOT_METADATA_ONLY")
    if _safe_string(record.get("body_text_exposure")) not in {
        "",
        "NONE_POINTER_ONLY",
        "NO_RAW_TEXT",
    }:
        reason_codes.append("BODY_TEXT_EXPOSURE_NOT_NONE")

    return {
        "accepted": not reason_codes,
        "reason_codes": sorted(set(reason_codes)),
    }


def bridge_evidence_item_from_adapter_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema-facing Bridge evidence item for a valid adapter record."""

    bridge_item = record.get("bridge_evidence_item")
    if not isinstance(bridge_item, Mapping):
        raise ValueError("adapter record does not contain bridge_evidence_item")
    return {field: _safe_string(bridge_item.get(field)) for field in _BRIDGE_REQUIRED_FIELDS}


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw_records = payload
    elif isinstance(payload, Mapping):
        raw_records = payload.get("records")
    else:
        raise ValueError("fixture payload must be an object or list")

    if not isinstance(raw_records, list):
        raise ValueError("fixture payload does not contain a records list")

    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"fixture record {index} is not an object")
        records.append(dict(raw_record))
    return records


def _adapt_single_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    bridge_item = {
        "evidence_id": _safe_string(record.get("evidence_id")),
        "bridge_trace_id": _safe_string(record.get("bridge_trace_id")),
        "safe_summary": _safe_string(record.get("safe_summary")),
        "pointer_uri": _safe_string(record.get("pointer_uri")),
        "raw_text_policy": RAW_TEXT_POLICY_POINTER_ONLY,
        "rights_status": RIGHTS_NOT_VERIFIED,
    }
    is_duplicate = _safe_string(record.get("candidate_role")).upper() == "DUPLICATE_CANDIDATE_HOLD"
    adapted: dict[str, Any] = {
        **bridge_item,
        "source_label": _safe_string(record.get("source_label")),
        "standard_family": _safe_string(record.get("standard_family")),
        "doc_id": _safe_string(record.get("doc_id")),
        "rights_status_decision": RIGHTS_STATUS_DECISION,
        "summary_source": SUMMARY_SOURCE,
        "semantic_summary_verified": False,
        "raw_text_exposed": False,
        "production_path_exposed": False,
        "adapter_source": ADAPTER_SOURCE,
        "adapter_policy": ADAPTER_POLICY,
        "adapter_record_index": index,
        "bridge_policy_result": decide_bridge_result(bridge_item)["result_status"],
        "primary_adapter_candidate": not is_duplicate,
        "duplicate_status": DUPLICATE_STATUS if is_duplicate else PRIMARY_STATUS,
        "duplicate_decision": DUPLICATE_STATUS if is_duplicate else PRIMARY_STATUS,
        "unsafe_field_exclusion_count": _safe_len(record.get("unsafe_fields_excluded")),
        "bridge_evidence_item": bridge_item,
    }

    for field in _OPTIONAL_SAFE_FIELDS:
        if field in record:
            adapted[field] = _safe_optional_value(record.get(field))

    validation_shape_ids = record.get("validation_shape_ids")
    if isinstance(validation_shape_ids, list):
        adapted["validation_shape_ids"] = [
            _safe_string(item)
            for item in validation_shape_ids
            if _safe_string(item)
        ]

    validation = validate_bridge_skeleton_record(adapted)
    if not validation["is_valid"]:
        raise ValueError(
            "adapted record failed validation: "
            + ",".join(validation["reason_codes"])
        )
    return adapted


def _raise_for_forbidden_fixture_path(path_value: str) -> None:
    lowered = path_value.lower()
    filename = Path(path_value).name.lower()

    if _FORBIDDEN_PRODUCTION_LIBRARY_RE.search(path_value):
        raise ValueError("fixture path points at production Library root")
    if _FORBIDDEN_DB_PATH_RE.search(path_value) or any(marker in lowered for marker in _DB_MARKERS):
        raise ValueError("fixture path is DB-like or references a DB marker")
    if any(marker in filename for marker in _SECRET_MARKERS) or _KEY_MARKER_RE.search(filename):
        raise ValueError("fixture path filename is secret-like")


def _collect_unsafe_reason_codes(value: Any) -> list[str]:
    reasons: list[str] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            for raw_key, raw_value in node.items():
                key = str(raw_key)
                lowered_key = key.lower()
                if lowered_key in _RAW_BODY_FIELD_NAMES:
                    reasons.append(f"RAW_BODY_FIELD_PRESENT:{key}")
                elif "raw_text" in lowered_key and lowered_key not in _ALLOWED_RAW_POLICY_FIELD_NAMES:
                    reasons.append(f"RAW_TEXT_FIELD_PRESENT:{key}")
                if _is_secret_like_marker(lowered_key):
                    reasons.append(f"SECRET_MARKER_FIELD:{key}")
                visit(raw_value, (*path, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, (*path, str(index)))
        elif isinstance(node, str):
            lowered = node.lower()
            if _ABSOLUTE_PATH_RE.search(node):
                reasons.append("ABSOLUTE_FILESYSTEM_PATH_VALUE")
            if _FORBIDDEN_PRODUCTION_LIBRARY_RE.search(node):
                reasons.append("PRODUCTION_LIBRARY_PATH_VALUE")
            if _FORBIDDEN_DB_PATH_RE.search(node) or any(marker in lowered for marker in _DB_MARKERS):
                reasons.append("DB_MARKER_VALUE")
            if _is_secret_like_marker(lowered):
                reasons.append("SECRET_MARKER_VALUE")
            for marker in _READINESS_MARKERS:
                if marker in lowered:
                    reasons.append(f"BROAD_READINESS_MARKER:{marker}")

    visit(value, ())
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


def _safe_len(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0
