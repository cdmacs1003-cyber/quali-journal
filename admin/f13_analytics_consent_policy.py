from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from admin.f13_analytics_event_contract import map_feedback_to_analytics_event


CONSENT_ALLOWED = "CONSENT_ALLOWED"
CONSENT_REQUIRED = "CONSENT_REQUIRED"
CONSENT_REVOKED = "CONSENT_REVOKED"
CONSENT_SCOPE_DENIED = "CONSENT_SCOPE_DENIED"
ANALYTICS_EXCLUDED = "ANALYTICS_EXCLUDED"
CONSENT_USER_MISMATCH = "CONSENT_USER_MISMATCH"
CONSENT_RECORD_INVALID = "CONSENT_RECORD_INVALID"

CONSENT_SCOPE_FIELDS = (
    "learning_analytics",
    "marketing_aggregate",
    "personalized_feedback",
)
REQUIRED_CONSENT_FIELDS = (
    "consent_id",
    "user_id_hash",
    "consent_scope",
    "consent_version",
    "granted_at",
    "revoked_at",
    "retention_policy_id",
)
PROHIBITED_CONSENT_FIELDS = frozenset(
    {
        "raw_query",
        "query",
        "raw_body",
        "body",
        "prompt",
        "raw_prompt",
        "raw_text",
        "answer_text",
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
        "personal_name",
        "email",
        "phone",
        "address",
    }
)

_CONSENT_ID_RE = re.compile(r"^CST-[A-Za-z0-9][A-Za-z0-9._-]{0,155}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
_RETENTION_ID_RE = re.compile(r"^[A-Za-z0-9:._-]{1,80}$")


def _normalize_field_name(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _prohibited_fields(value: Any) -> list[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalize_field_name(key)
            if normalized in PROHIBITED_CONSENT_FIELDS:
                found.add(normalized)
            found.update(_prohibited_fields(child))
    elif isinstance(value, list | tuple | set):
        for child in value:
            found.update(_prohibited_fields(child))
    return sorted(found)


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
    }


def _parse_aware_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def validate_analytics_consent_record(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return _validation_result(valid=False, reason_codes=["CONSENT_PAYLOAD_NOT_MAPPING"])

    prohibited = _prohibited_fields(payload)
    if prohibited:
        return _validation_result(
            valid=False,
            reason_codes=["PROHIBITED_FIELD"],
            prohibited_fields=prohibited,
        )

    required = set(REQUIRED_CONSENT_FIELDS)
    payload_fields = set(payload)
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

    if not isinstance(payload.get("consent_id"), str) or _CONSENT_ID_RE.fullmatch(payload["consent_id"]) is None:
        reasons.append("INVALID_CONSENT_ID")
        invalid_fields.append("consent_id")
    if not isinstance(payload.get("user_id_hash"), str) or _HASH_RE.fullmatch(payload["user_id_hash"]) is None:
        reasons.append("INVALID_USER_ID_HASH")
        invalid_fields.append("user_id_hash")

    scope = payload.get("consent_scope")
    if not isinstance(scope, Mapping):
        reasons.append("INVALID_CONSENT_SCOPE")
        invalid_fields.append("consent_scope")
    else:
        scope_fields = set(scope)
        expected_scope_fields = set(CONSENT_SCOPE_FIELDS)
        if scope_fields != expected_scope_fields:
            reasons.append("INVALID_CONSENT_SCOPE_FIELDS")
            invalid_fields.append("consent_scope")
        elif any(type(scope[field]) is not bool for field in CONSENT_SCOPE_FIELDS):
            reasons.append("INVALID_CONSENT_SCOPE_VALUE")
            invalid_fields.append("consent_scope")

    consent_version = payload.get("consent_version")
    if not isinstance(consent_version, str) or _VERSION_RE.fullmatch(consent_version) is None:
        reasons.append("INVALID_CONSENT_VERSION")
        invalid_fields.append("consent_version")

    granted_at = _parse_aware_datetime(payload.get("granted_at"))
    if granted_at is None:
        reasons.append("INVALID_GRANTED_AT")
        invalid_fields.append("granted_at")

    revoked_value = payload.get("revoked_at")
    revoked_at = None if revoked_value is None else _parse_aware_datetime(revoked_value)
    if revoked_value is not None and revoked_at is None:
        reasons.append("INVALID_REVOKED_AT")
        invalid_fields.append("revoked_at")
    elif granted_at is not None and revoked_at is not None and revoked_at < granted_at:
        reasons.append("REVOKED_AT_PRECEDES_GRANTED_AT")
        invalid_fields.append("revoked_at")

    retention_policy_id = payload.get("retention_policy_id")
    if not isinstance(retention_policy_id, str) or _RETENTION_ID_RE.fullmatch(retention_policy_id) is None:
        reasons.append("INVALID_RETENTION_POLICY_ID")
        invalid_fields.append("retention_policy_id")

    return _validation_result(
        valid=not reasons,
        reason_codes=reasons,
        invalid_fields=invalid_fields,
    )


def _decision(*, allowed: bool, status: str, reason_code: str, consent_id: str | None) -> dict[str, Any]:
    return {
        "allowed": allowed,
        "status": status,
        "reason_code": reason_code,
        "consent_id": consent_id,
    }


def evaluate_analytics_consent_policy(
    consent_payload: Any,
    *,
    target_user_id_hash: str,
    analytics_exclusion: bool,
) -> dict[str, Any]:
    if consent_payload is None:
        return _decision(
            allowed=False,
            status="EXCLUDE",
            reason_code=CONSENT_REQUIRED,
            consent_id=None,
        )

    validation = validate_analytics_consent_record(consent_payload)
    if not validation["valid"] or not isinstance(consent_payload, Mapping) or type(analytics_exclusion) is not bool:
        return _decision(
            allowed=False,
            status="EXCLUDE",
            reason_code=CONSENT_RECORD_INVALID,
            consent_id=None,
        )

    consent_id = str(consent_payload["consent_id"])
    if consent_payload["user_id_hash"] != target_user_id_hash:
        return _decision(
            allowed=False,
            status="EXCLUDE",
            reason_code=CONSENT_USER_MISMATCH,
            consent_id=consent_id,
        )
    if consent_payload["revoked_at"] is not None:
        return _decision(
            allowed=False,
            status="EXCLUDE",
            reason_code=CONSENT_REVOKED,
            consent_id=consent_id,
        )
    if analytics_exclusion:
        return _decision(
            allowed=False,
            status="EXCLUDE",
            reason_code=ANALYTICS_EXCLUDED,
            consent_id=consent_id,
        )
    if consent_payload["consent_scope"]["learning_analytics"] is not True:
        return _decision(
            allowed=False,
            status="EXCLUDE",
            reason_code=CONSENT_SCOPE_DENIED,
            consent_id=consent_id,
        )
    return _decision(
        allowed=True,
        status="ALLOW",
        reason_code=CONSENT_ALLOWED,
        consent_id=consent_id,
    )


def map_feedback_to_analytics_with_consent(
    feedback_payload: Any,
    *,
    consent_payload: Any,
    user_id_hash: str,
    analytics_exclusion: bool = False,
    event_id: str | None = None,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    decision = evaluate_analytics_consent_policy(
        consent_payload,
        target_user_id_hash=user_id_hash,
        analytics_exclusion=analytics_exclusion,
    )
    if not decision["allowed"]:
        return {
            "policy_status": decision["status"],
            "reason_code": decision["reason_code"],
            "analytics_event_present": False,
            "analytics_event": None,
        }

    mapping = map_feedback_to_analytics_event(
        feedback_payload,
        user_id_hash=user_id_hash,
        event_id=event_id,
        occurred_at=occurred_at,
    )
    if not mapping.get("mapped") or not isinstance(mapping.get("event"), dict):
        reason_codes = mapping.get("reason_codes")
        reason_code = reason_codes[0] if isinstance(reason_codes, list) and reason_codes else "ANALYTICS_MAPPING_REJECTED"
        return {
            "policy_status": "MAPPING_REJECTED",
            "reason_code": reason_code,
            "analytics_event_present": False,
            "analytics_event": None,
        }
    return {
        "policy_status": decision["status"],
        "reason_code": decision["reason_code"],
        "analytics_event_present": True,
        "analytics_event": mapping["event"],
    }


__all__ = [
    "ANALYTICS_EXCLUDED",
    "CONSENT_ALLOWED",
    "CONSENT_RECORD_INVALID",
    "CONSENT_REQUIRED",
    "CONSENT_REVOKED",
    "CONSENT_SCOPE_DENIED",
    "CONSENT_SCOPE_FIELDS",
    "CONSENT_USER_MISMATCH",
    "PROHIBITED_CONSENT_FIELDS",
    "REQUIRED_CONSENT_FIELDS",
    "evaluate_analytics_consent_policy",
    "map_feedback_to_analytics_with_consent",
    "validate_analytics_consent_record",
]
