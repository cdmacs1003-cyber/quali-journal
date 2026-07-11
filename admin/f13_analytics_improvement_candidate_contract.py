from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from admin.f13_analytics_event_contract import RISK_FLAG_VOCABULARY, validate_analytics_event


SCHEMA_VERSION = 1
CONTRACT_VERSION = "1.0.0"
SUMMARY_CODE_MAX_LENGTH = 64

CANDIDATE_TYPE = "analytics_improvement_candidate"
CANDIDATE_STATUS = "captured"
IMPROVEMENT_TRIGGERS = frozenset({"evidence_hold", "review_needed", "content_gap"})
ELIGIBLE_EVENT_TRIGGER_PAIRS = frozenset(
    {
        ("hold_created", "evidence_hold"),
        ("hold_created", "content_gap"),
        ("answer_rendered", "review_needed"),
    }
)
SUMMARY_CODES = {
    "evidence_hold": "analytics_evidence_hold",
    "review_needed": "analytics_review_needed",
    "content_gap": "analytics_content_gap",
}

REQUIRED_FIELDS = (
    "schema_version",
    "contract_version",
    "candidate_id",
    "candidate_type",
    "tenant_context",
    "source_event_id",
    "source_event_type",
    "source_request_id",
    "source_trace_id",
    "query_hash",
    "improvement_trigger",
    "summary_code",
    "risk_flags",
    "provenance",
    "classification",
    "status",
    "review_required",
    "auto_promote",
    "approved_for_library",
    "raw_query_stored",
    "raw_body_stored",
    "raw_answer_stored",
    "raw_evidence_text_stored",
    "idempotency_key",
    "created_at",
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
        "answer",
        "answer_text",
        "safe_short_answer",
        "hold_source",
        "evidence",
        "evidence_text",
        "standard_text",
        "paid_standard_text",
        "query_summary",
        "user_id",
        "user_id_hash",
        "personal_name",
        "email",
        "phone",
        "address",
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
        "library_write",
        "promoted_library_id",
        "approver_id",
        "approval_event_id",
        "promotion_trace_id",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9:._-]{1,160}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate:analytics:[0-9a-f]{32}$")
_IDEMPOTENCY_RE = re.compile(r"^idem:analytics:[0-9a-f]{64}$")


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


def _result(
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
    }


def _valid_identifier(value: Any) -> bool:
    return isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) is not None


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def build_improvement_idempotency_key(
    *,
    tenant_id: str,
    organization_id: str,
    source_event_id: str,
    source_trace_id: str,
    improvement_trigger: str,
) -> str:
    parts = (tenant_id, organization_id, source_event_id, source_trace_id, improvement_trigger)
    if not all(_valid_identifier(part) for part in parts):
        raise ValueError("INVALID_SAFE_IDEMPOTENCY_INPUT")
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"idem:analytics:{digest}"


def validate_analytics_improvement_candidate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return _result(valid=False, reason_codes=["CANDIDATE_PAYLOAD_NOT_MAPPING"])

    prohibited = _prohibited_fields(payload)
    if prohibited:
        return _result(
            valid=False,
            reason_codes=["PROHIBITED_FIELD"],
            prohibited_fields=prohibited,
        )

    fields = set(payload)
    required = set(REQUIRED_FIELDS)
    missing = sorted(required - fields)
    unexpected = sorted(str(field) for field in fields - required)
    reasons: list[str] = []
    invalid_fields: list[str] = []
    if missing:
        reasons.append("MISSING_REQUIRED_FIELD")
        invalid_fields.extend(missing)
    if unexpected:
        reasons.append("UNEXPECTED_PROPERTY")
        invalid_fields.extend(unexpected)
    if missing or unexpected:
        return _result(valid=False, reason_codes=reasons, invalid_fields=invalid_fields)

    if payload.get("schema_version") != SCHEMA_VERSION:
        reasons.append("INVALID_SCHEMA_VERSION")
        invalid_fields.append("schema_version")
    if payload.get("contract_version") != CONTRACT_VERSION:
        reasons.append("INVALID_CONTRACT_VERSION")
        invalid_fields.append("contract_version")
    if not isinstance(payload.get("candidate_id"), str) or _CANDIDATE_ID_RE.fullmatch(payload["candidate_id"]) is None:
        reasons.append("INVALID_CANDIDATE_ID")
        invalid_fields.append("candidate_id")
    if payload.get("candidate_type") != CANDIDATE_TYPE:
        reasons.append("INVALID_CANDIDATE_TYPE")
        invalid_fields.append("candidate_type")

    tenant_context = payload.get("tenant_context")
    if not isinstance(tenant_context, Mapping) or set(tenant_context) != {
        "tenant_id",
        "organization_id",
        "cohort_id",
    } or any(not _valid_identifier(tenant_context.get(field)) for field in tenant_context):
        reasons.append("INVALID_TENANT_CONTEXT")
        invalid_fields.append("tenant_context")

    for field in ("source_event_id", "source_request_id", "source_trace_id"):
        if not _valid_identifier(payload.get(field)):
            reasons.append(f"INVALID_{field.upper()}")
            invalid_fields.append(field)

    source_event_type = payload.get("source_event_type")
    trigger = payload.get("improvement_trigger")
    if source_event_type not in {"answer_rendered", "hold_created"}:
        reasons.append("IMPROVEMENT_EVENT_NOT_ELIGIBLE")
        invalid_fields.append("source_event_type")
    if trigger not in IMPROVEMENT_TRIGGERS:
        reasons.append("IMPROVEMENT_TRIGGER_INVALID")
        invalid_fields.append("improvement_trigger")
    elif (source_event_type, trigger) not in ELIGIBLE_EVENT_TRIGGER_PAIRS:
        reasons.append("IMPROVEMENT_EVENT_NOT_ELIGIBLE")
        invalid_fields.extend(["source_event_type", "improvement_trigger"])
    if payload.get("summary_code") != SUMMARY_CODES.get(trigger):
        reasons.append("INVALID_SUMMARY_CODE")
        invalid_fields.append("summary_code")
    if not isinstance(payload.get("query_hash"), str) or _HASH_RE.fullmatch(payload["query_hash"]) is None:
        reasons.append("INVALID_QUERY_HASH")
        invalid_fields.append("query_hash")

    risk_flags = payload.get("risk_flags")
    if (
        not isinstance(risk_flags, list)
        or not all(isinstance(flag, str) for flag in risk_flags)
        or len(risk_flags) != len(set(risk_flags))
        or any(flag not in RISK_FLAG_VOCABULARY for flag in risk_flags)
    ):
        reasons.append("INVALID_RISK_FLAGS")
        invalid_fields.append("risk_flags")

    provenance = payload.get("provenance")
    expected_provenance = {
        "provider_type": "analytics",
        "provider_ref": payload.get("source_event_id"),
        "collection_reason": trigger,
        "rights_status": "owned",
    }
    if not isinstance(provenance, Mapping) or dict(provenance) != expected_provenance:
        reasons.append("INVALID_PROVENANCE")
        invalid_fields.append("provenance")

    expected_classification = {
        "sensitivity": "internal",
        "visibility": "internal_only",
        "domain": "quality",
    }
    classification = payload.get("classification")
    if not isinstance(classification, Mapping) or dict(classification) != expected_classification:
        reasons.append("INVALID_CLASSIFICATION")
        invalid_fields.append("classification")

    fixed_values = {
        "status": CANDIDATE_STATUS,
        "review_required": True,
        "auto_promote": False,
        "approved_for_library": False,
        "raw_query_stored": False,
        "raw_body_stored": False,
        "raw_answer_stored": False,
        "raw_evidence_text_stored": False,
    }
    for field, expected in fixed_values.items():
        if payload.get(field) is not expected and payload.get(field) != expected:
            reasons.append(f"INVALID_{field.upper()}")
            invalid_fields.append(field)

    idempotency_key = payload.get("idempotency_key")
    if not isinstance(idempotency_key, str) or _IDEMPOTENCY_RE.fullmatch(idempotency_key) is None:
        reasons.append("INVALID_IDEMPOTENCY_KEY")
        invalid_fields.append("idempotency_key")
    elif isinstance(tenant_context, Mapping):
        try:
            expected_key = build_improvement_idempotency_key(
                tenant_id=str(tenant_context.get("tenant_id", "")),
                organization_id=str(tenant_context.get("organization_id", "")),
                source_event_id=str(payload.get("source_event_id", "")),
                source_trace_id=str(payload.get("source_trace_id", "")),
                improvement_trigger=str(trigger or ""),
            )
        except ValueError:
            expected_key = None
        if idempotency_key != expected_key:
            reasons.append("IDEMPOTENCY_KEY_MISMATCH")
            invalid_fields.append("idempotency_key")

    if not _valid_timestamp(payload.get("created_at")):
        reasons.append("INVALID_CREATED_AT")
        invalid_fields.append("created_at")

    return _result(valid=not reasons, reason_codes=reasons, invalid_fields=invalid_fields)


def _candidate_mapping_result(
    *,
    candidate_present: bool,
    reason_code: str,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_present": candidate_present,
        "status": "CREATED" if candidate_present else "NOT_CREATED",
        "reason_code": reason_code,
        "candidate": candidate if candidate_present else None,
    }


def _build_candidate(
    analytics_event: Mapping[str, Any],
    *,
    improvement_trigger: str,
    created_at: str | None,
) -> dict[str, Any]:
    idempotency_key = build_improvement_idempotency_key(
        tenant_id=str(analytics_event["tenant_id"]),
        organization_id=str(analytics_event["organization_id"]),
        source_event_id=str(analytics_event["event_id"]),
        source_trace_id=str(analytics_event["trace_id"]),
        improvement_trigger=improvement_trigger,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "candidate_id": f"candidate:analytics:{idempotency_key.rsplit(':', 1)[-1][:32]}",
        "candidate_type": CANDIDATE_TYPE,
        "tenant_context": {
            "tenant_id": analytics_event["tenant_id"],
            "organization_id": analytics_event["organization_id"],
            "cohort_id": analytics_event["cohort_id"],
        },
        "source_event_id": analytics_event["event_id"],
        "source_event_type": analytics_event["event_type"],
        "source_request_id": analytics_event["request_id"],
        "source_trace_id": analytics_event["trace_id"],
        "query_hash": analytics_event["query_hash"],
        "improvement_trigger": improvement_trigger,
        "summary_code": SUMMARY_CODES[improvement_trigger],
        "risk_flags": list(analytics_event["risk_flags"]),
        "provenance": {
            "provider_type": "analytics",
            "provider_ref": analytics_event["event_id"],
            "collection_reason": improvement_trigger,
            "rights_status": "owned",
        },
        "classification": {
            "sensitivity": "internal",
            "visibility": "internal_only",
            "domain": "quality",
        },
        "status": CANDIDATE_STATUS,
        "review_required": True,
        "auto_promote": False,
        "approved_for_library": False,
        "raw_query_stored": False,
        "raw_body_stored": False,
        "raw_answer_stored": False,
        "raw_evidence_text_stored": False,
        "idempotency_key": idempotency_key,
        "created_at": created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def map_consent_analytics_to_warehouse_candidate(
    consent_analytics_result: Any,
    *,
    improvement_trigger: str | None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if not isinstance(consent_analytics_result, Mapping):
        return _candidate_mapping_result(candidate_present=False, reason_code="CONSENT_NOT_ALLOWED")
    if (
        consent_analytics_result.get("policy_status") != "ALLOW"
        or consent_analytics_result.get("reason_code") != "CONSENT_ALLOWED"
        or consent_analytics_result.get("analytics_event_present") is not True
    ):
        return _candidate_mapping_result(candidate_present=False, reason_code="CONSENT_NOT_ALLOWED")
    if not improvement_trigger:
        return _candidate_mapping_result(
            candidate_present=False,
            reason_code="IMPROVEMENT_TRIGGER_REQUIRED",
        )
    if improvement_trigger not in IMPROVEMENT_TRIGGERS:
        return _candidate_mapping_result(
            candidate_present=False,
            reason_code="IMPROVEMENT_TRIGGER_INVALID",
        )

    analytics_event = consent_analytics_result.get("analytics_event")
    validation = validate_analytics_event(analytics_event)
    if not validation.get("valid") or not isinstance(analytics_event, Mapping):
        return _candidate_mapping_result(candidate_present=False, reason_code="ANALYTICS_EVENT_INVALID")
    if (analytics_event.get("event_type"), improvement_trigger) not in ELIGIBLE_EVENT_TRIGGER_PAIRS:
        return _candidate_mapping_result(
            candidate_present=False,
            reason_code="IMPROVEMENT_EVENT_NOT_ELIGIBLE",
        )

    candidate = _build_candidate(
        analytics_event,
        improvement_trigger=improvement_trigger,
        created_at=created_at,
    )
    candidate_validation = validate_analytics_improvement_candidate(candidate)
    if not candidate_validation["valid"]:
        return _candidate_mapping_result(
            candidate_present=False,
            reason_code="WAREHOUSE_CANDIDATE_INVALID",
        )
    return _candidate_mapping_result(
        candidate_present=True,
        reason_code="IMPROVEMENT_CANDIDATE_CREATED",
        candidate=candidate,
    )


__all__ = [
    "CANDIDATE_STATUS",
    "CANDIDATE_TYPE",
    "CONTRACT_VERSION",
    "ELIGIBLE_EVENT_TRIGGER_PAIRS",
    "IMPROVEMENT_TRIGGERS",
    "PROHIBITED_FIELD_NAMES",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "SUMMARY_CODES",
    "SUMMARY_CODE_MAX_LENGTH",
    "build_improvement_idempotency_key",
    "map_consent_analytics_to_warehouse_candidate",
    "validate_analytics_improvement_candidate",
]
