from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from admin.f13_feedback_queue_contract import RESULT_READY, validate_feedback_queue_contract


SCHEMA_VERSION = 1
CONTRACT_VERSION = "1.0.0"
QUERY_SUMMARY_MAX_LENGTH = 64

ANALYTICS_EVENT_TYPES = frozenset(
    {
        "question_asked",
        "answer_rendered",
        "hold_created",
        "evidence_viewed",
        "assessment_viewed",
    }
)
FEEDBACK_TO_ANALYTICS_EVENT_TYPES = {
    "answer_rendered": "answer_rendered",
    "hold_created": "hold_created",
}
FEEDBACK_TO_ANALYTICS_SUMMARIES = {
    "answer_rendered": "answer_event",
    "hold_created": "hold_event",
}
TRACE_REQUIRED_EVENT_TYPES = frozenset(
    {
        "answer_rendered",
        "hold_created",
        "evidence_viewed",
    }
)
RISK_FLAG_VOCABULARY = frozenset(
    {
        "consent_excluded",
        "evidence_missing",
        "paid_standard_raw_request",
        "policy_hold",
    }
)
REQUIRED_FIELDS = (
    "schema_version",
    "contract_version",
    "event_id",
    "tenant_id",
    "organization_id",
    "cohort_id",
    "user_id_hash",
    "event_type",
    "request_id",
    "trace_id",
    "query_hash",
    "query_summary",
    "raw_query_stored",
    "risk_flags",
    "occurred_at",
)
PROHIBITED_FIELD_NAMES = frozenset(
    {
        "raw_query",
        "query",
        "raw_body",
        "body",
        "prompt",
        "raw_prompt",
        "raw_text",
        "evidence_text",
        "standard_text",
        "paid_standard_text",
        "secret",
        "api_key",
        "access_token",
        "refresh_token",
        "password",
        "credential",
        "cookie",
        "authorization",
        "internal_path",
        "local_path",
        "file_path",
        "db_path",
        "dsn",
        "connection_string",
    }
)
QUERY_SUMMARY_DISALLOWED_MARKERS = (
    "api_key",
    "access_token",
    "authorization",
    "credential",
    "evidence_text",
    "paid_standard",
    "password",
    "raw_evidence",
    "raw_text",
    "secret",
    "standard_text",
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9:._-]{1,160}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/]")
_POSIX_PATH_RE = re.compile(r"(?:^|\s)/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+")
_URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)


def _normalize_field_name(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _prohibited_fields(value: Any) -> list[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalize_field_name(key)
            if normalized in PROHIBITED_FIELD_NAMES:
                found.add(normalized)
            found.update(_prohibited_fields(child))
    elif isinstance(value, list | tuple | set):
        for child in value:
            found.update(_prohibited_fields(child))
    return sorted(found)


def _execution_flags() -> dict[str, bool]:
    return {
        "db_access_executed": False,
        "durable_write_executed": False,
        "file_io_executed": False,
        "network_access_executed": False,
        "environment_access_executed": False,
        "subprocess_executed": False,
    }


def _validation_result(
    *,
    valid: bool,
    reason_codes: list[str],
    invalid_fields: list[str] | None = None,
    prohibited_fields: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": "VALID" if valid else "INVALID",
        "valid": valid,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "invalid_fields": sorted(set(invalid_fields or [])),
        "prohibited_fields": sorted(set(prohibited_fields or [])),
        **_execution_flags(),
    }


def _valid_identifier(value: Any) -> bool:
    return isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) is not None


def _valid_hash(value: Any) -> bool:
    return value is None or (isinstance(value, str) and _HASH_RE.fullmatch(value) is not None)


def _valid_summary(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str) or not 1 <= len(value) <= QUERY_SUMMARY_MAX_LENGTH:
        return False
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    lowered = value.lower()
    if _URL_RE.search(value) or _WINDOWS_PATH_RE.search(value) or _POSIX_PATH_RE.search(value):
        return False
    return not any(marker in lowered for marker in QUERY_SUMMARY_DISALLOWED_MARKERS)


def _valid_occurred_at(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def validate_analytics_event(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return _validation_result(valid=False, reason_codes=["PAYLOAD_NOT_MAPPING"])

    prohibited = _prohibited_fields(payload)
    if prohibited:
        return _validation_result(
            valid=False,
            reason_codes=["PROHIBITED_FIELD"],
            prohibited_fields=prohibited,
        )

    payload_fields = set(payload)
    required = set(REQUIRED_FIELDS)
    missing = sorted(required - payload_fields)
    unexpected = sorted(str(field) for field in payload_fields - required)
    reasons: list[str] = []
    invalid_fields: list[str] = []
    if missing:
        reasons.append("MISSING_REQUIRED_FIELD")
        invalid_fields.extend(missing)
    if unexpected:
        reasons.append("UNEXPECTED_PROPERTY")
        invalid_fields.extend(unexpected)
    if missing or unexpected:
        return _validation_result(
            valid=False,
            reason_codes=reasons,
            invalid_fields=invalid_fields,
        )

    if payload.get("schema_version") != SCHEMA_VERSION:
        reasons.append("INVALID_SCHEMA_VERSION")
        invalid_fields.append("schema_version")
    if payload.get("contract_version") != CONTRACT_VERSION:
        reasons.append("INVALID_CONTRACT_VERSION")
        invalid_fields.append("contract_version")

    for field in ("event_id", "tenant_id", "organization_id", "cohort_id", "request_id"):
        if not _valid_identifier(payload.get(field)):
            reasons.append(f"INVALID_{field.upper()}")
            invalid_fields.append(field)

    event_type = payload.get("event_type")
    if event_type not in ANALYTICS_EVENT_TYPES:
        reasons.append("UNCONTROLLED_EVENT_TYPE")
        invalid_fields.append("event_type")

    trace_id = payload.get("trace_id")
    if trace_id is not None and not _valid_identifier(trace_id):
        reasons.append("INVALID_TRACE_ID")
        invalid_fields.append("trace_id")
    if event_type in TRACE_REQUIRED_EVENT_TYPES and not _valid_identifier(trace_id):
        reasons.append("TRACE_ID_REQUIRED")
        invalid_fields.append("trace_id")

    if not _valid_hash(payload.get("user_id_hash")) or payload.get("user_id_hash") is None:
        reasons.append("INVALID_USER_ID_HASH")
        invalid_fields.append("user_id_hash")
    if not _valid_hash(payload.get("query_hash")):
        reasons.append("INVALID_QUERY_HASH")
        invalid_fields.append("query_hash")
    if not _valid_summary(payload.get("query_summary")):
        reasons.append("UNSAFE_QUERY_SUMMARY")
        invalid_fields.append("query_summary")
    if payload.get("raw_query_stored") is not False:
        reasons.append("RAW_QUERY_STORAGE_FORBIDDEN")
        invalid_fields.append("raw_query_stored")

    risk_flags = payload.get("risk_flags")
    risk_flags_are_strings = isinstance(risk_flags, list) and all(
        isinstance(flag, str) for flag in risk_flags
    )
    if (
        not risk_flags_are_strings
        or len(risk_flags) != len(set(risk_flags))
        or any(flag not in RISK_FLAG_VOCABULARY for flag in risk_flags)
    ):
        reasons.append("INVALID_RISK_FLAGS")
        invalid_fields.append("risk_flags")
    if not _valid_occurred_at(payload.get("occurred_at")):
        reasons.append("INVALID_OCCURRED_AT")
        invalid_fields.append("occurred_at")

    return _validation_result(
        valid=not reasons,
        reason_codes=reasons,
        invalid_fields=invalid_fields,
    )


def _mapping_result(
    *,
    mapped: bool,
    reason_codes: list[str],
    event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "MAPPED" if mapped else "NOT_MAPPED",
        "mapped": mapped,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "event": event if mapped else None,
        **_execution_flags(),
    }


def map_feedback_to_analytics_event(
    feedback_payload: Any,
    *,
    user_id_hash: str,
    event_id: str | None = None,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    feedback_validation = validate_feedback_queue_contract(feedback_payload)
    if feedback_validation.get("status") != RESULT_READY or not isinstance(feedback_payload, Mapping):
        return _mapping_result(mapped=False, reason_codes=["FEEDBACK_CONTRACT_INVALID"])

    event_context = feedback_payload.get("event_context")
    event_type = event_context.get("event_type") if isinstance(event_context, Mapping) else None
    mapped_event_type = FEEDBACK_TO_ANALYTICS_EVENT_TYPES.get(str(event_type or ""))
    if mapped_event_type is None:
        return _mapping_result(mapped=False, reason_codes=["ANALYTICS_EVENT_NOT_MAPPED"])

    tenant_context = feedback_payload.get("tenant_context")
    tenant = tenant_context if isinstance(tenant_context, Mapping) else {}
    missing_context = [
        field
        for field in ("tenant_id", "organization_id", "cohort_id")
        if not _valid_identifier(tenant.get(field))
    ]
    if not _valid_identifier(feedback_payload.get("request_id")):
        missing_context.append("request_id")
    trace_id = feedback_payload.get("bridge_trace_id") or feedback_payload.get("trace_id")
    if not _valid_identifier(trace_id):
        missing_context.append("trace_id")
    query_hash = feedback_payload.get("query_hash")
    if not isinstance(query_hash, str) or not _valid_hash(query_hash):
        missing_context.append("query_hash")
    if missing_context:
        return _mapping_result(
            mapped=False,
            reason_codes=[f"MISSING_OR_INVALID_{field.upper()}" for field in missing_context],
        )

    risk_flags = feedback_payload.get("risk_flags", [])
    event = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "event_id": event_id or f"EVT-{uuid.uuid4().hex}",
        "tenant_id": tenant["tenant_id"],
        "organization_id": tenant["organization_id"],
        "cohort_id": tenant["cohort_id"],
        "user_id_hash": user_id_hash,
        "event_type": mapped_event_type,
        "request_id": feedback_payload["request_id"],
        "trace_id": trace_id,
        "query_hash": query_hash,
        "query_summary": FEEDBACK_TO_ANALYTICS_SUMMARIES[mapped_event_type],
        "raw_query_stored": False,
        "risk_flags": list(risk_flags) if isinstance(risk_flags, list) else risk_flags,
        "occurred_at": occurred_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    validation = validate_analytics_event(event)
    if not validation["valid"]:
        return _mapping_result(mapped=False, reason_codes=validation["reason_codes"])
    return _mapping_result(mapped=True, reason_codes=[], event=event)


__all__ = [
    "ANALYTICS_EVENT_TYPES",
    "CONTRACT_VERSION",
    "FEEDBACK_TO_ANALYTICS_EVENT_TYPES",
    "FEEDBACK_TO_ANALYTICS_SUMMARIES",
    "PROHIBITED_FIELD_NAMES",
    "QUERY_SUMMARY_DISALLOWED_MARKERS",
    "QUERY_SUMMARY_MAX_LENGTH",
    "REQUIRED_FIELDS",
    "RISK_FLAG_VOCABULARY",
    "SCHEMA_VERSION",
    "TRACE_REQUIRED_EVENT_TYPES",
    "map_feedback_to_analytics_event",
    "validate_analytics_event",
]
