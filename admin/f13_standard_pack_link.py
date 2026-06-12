from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RESULT_VALID = "VALID"
RESULT_HOLD = "HOLD"
RESULT_INVALID = "INVALID"

_ALLOWED_PACK_FAMILIES = {"IPC", "ECSS", "NASA", "INTERNAL", "QUALI"}
_ALLOWED_PACK_STATUSES = {"draft", "approved", "deprecated"}
_SAFE_SURFACE_KEYS = {
    "paid_standard_pointer_only",
    "raw_export_allowed",
    "student_summary_allowed",
    "raw_text_included",
    "internal_path_included",
    "db_access_executed",
    "network_access_executed",
    "runtime_access_executed",
}
_UNSAFE_FIELD_MARKERS = (
    "raw_text",
    "raw_prompt",
    "raw_query",
    "raw_standard_text",
    "full_source_text",
    "paid_standard_raw",
    "internal_path",
    "local_path",
    "source_uri_or_path",
    "secret",
    "token",
    "credential",
    "api_key",
    "private_key",
    "password",
    "authorization",
)
_UNSAFE_VALUE_MARKERS = (
    "raw text",
    "raw prompt",
    "raw query",
    "raw standard text",
    "full source text",
    "paid standard raw",
    "internal path",
    "h:\\",
    "c:\\",
    "file://",
    "localhost",
    "127.0.0.1",
    "secret",
    "token",
    "credential",
    "api key",
    "private key",
    "bearer ",
)


def _safe_text(value: Any, fallback: str, max_length: int = 160) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text[:max_length]


def _safe_token(value: Any, fallback: str, max_length: int = 120) -> str:
    text = _safe_text(value, fallback, max_length=max_length)
    token = "".join(ch for ch in text if ch.isalnum() or ch in ":._-")
    return token or fallback


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _contains_unsafe_surface(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered_key = str(key).lower()
            if lowered_key in _SAFE_SURFACE_KEYS:
                continue
            if any(marker in lowered_key for marker in _UNSAFE_FIELD_MARKERS):
                return True
            if _contains_unsafe_surface(child):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_unsafe_surface(child) for child in value)
    if isinstance(value, str):
        lowered_value = value.lower()
        return any(marker in lowered_value for marker in _UNSAFE_VALUE_MARKERS)
    return False


def _count_values(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    return sum(1 for item in value if not _is_missing(item))


def _result(
    *,
    status: str,
    active_link_ready: bool,
    errors: list[str],
    warnings: list[str],
    standard_pack_id: str,
    pack_version: str,
    pack_family: str,
    standard_count: int,
    library_count: int,
    evidence_count: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "active_link_ready": active_link_ready,
        "hold_reason": errors[0] if errors else None,
        "errors": errors,
        "warnings": warnings,
        "standard_pack_id": standard_pack_id,
        "pack_version": pack_version,
        "pack_family": pack_family,
        "standard_count": standard_count,
        "library_count": library_count,
        "evidence_count": evidence_count,
        "raw_text_included": False,
        "internal_path_included": False,
        "db_access_executed": False,
        "network_access_executed": False,
        "runtime_access_executed": False,
    }


def validate_standard_pack_link(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return _result(
            status=RESULT_INVALID,
            active_link_ready=False,
            errors=["payload must be a mapping"],
            warnings=[],
            standard_pack_id="",
            pack_version="",
            pack_family="",
            standard_count=0,
            library_count=0,
            evidence_count=0,
        )

    source = payload
    standard_pack_id = _safe_token(source.get("standard_pack_id"), "")
    pack_version = _safe_token(source.get("pack_version"), "")
    pack_family = _safe_token(source.get("pack_family"), "").upper()
    standard_count = _count_values(source.get("standard_node_ids"))
    library_count = _count_values(source.get("library_ids"))
    evidence_count = _count_values(source.get("evidence_ids"))
    errors: list[str] = []
    warnings: list[str] = []
    invalid = False

    if _contains_unsafe_surface(source):
        return _result(
            status=RESULT_HOLD,
            active_link_ready=False,
            errors=["unsafe standard pack link payload blocked by static safety boundary"],
            warnings=[],
            standard_pack_id=standard_pack_id,
            pack_version=pack_version,
            pack_family=pack_family,
            standard_count=standard_count,
            library_count=library_count,
            evidence_count=evidence_count,
        )

    schema_version = source.get("schema_version")
    if _is_missing(schema_version):
        errors.append("schema_version is required")
    elif schema_version != 1 and str(schema_version).strip() != "1":
        errors.append("schema_version must equal 1")
        invalid = True

    if _is_missing(source.get("contract_version")):
        errors.append("contract_version is required")

    tenant_context = source.get("tenant_context")
    if not isinstance(tenant_context, Mapping):
        errors.append("tenant_context is required")
    else:
        if _is_missing(tenant_context.get("tenant_id")):
            errors.append("tenant_context.tenant_id is required")
        if _is_missing(tenant_context.get("organization_id")):
            errors.append("tenant_context.organization_id is required")

    if not standard_pack_id:
        errors.append("standard_pack_id is required")
    elif not standard_pack_id.startswith("SPK-"):
        errors.append("standard_pack_id must start with SPK-")
        invalid = True

    if not pack_family:
        errors.append("pack_family is required")
    elif pack_family not in _ALLOWED_PACK_FAMILIES:
        errors.append("pack_family is not supported")
        invalid = True

    if _is_missing(source.get("pack_title")):
        errors.append("pack_title is required")

    if not pack_version:
        errors.append("pack_version is required")

    pack_status = str(source.get("status") or "").strip().lower()
    if not pack_status:
        errors.append("status is required")
    elif pack_status not in _ALLOWED_PACK_STATUSES:
        errors.append("status is not supported")
        invalid = True

    if not isinstance(source.get("standard_node_ids"), list):
        errors.append("standard_node_ids must be a list")
    if not isinstance(source.get("library_ids"), list):
        errors.append("library_ids must be a list")
    if not isinstance(source.get("evidence_ids"), list):
        errors.append("evidence_ids must be a list")

    policy = source.get("policy")
    if not isinstance(policy, Mapping):
        errors.append("policy is required")
    else:
        if policy.get("paid_standard_pointer_only") is not True:
            errors.append("policy.paid_standard_pointer_only must be true")
        if policy.get("raw_export_allowed") is not False:
            errors.append("policy.raw_export_allowed must be false")
        if policy.get("student_summary_allowed") is not True:
            errors.append("policy.student_summary_allowed must be true")

    if pack_status == "approved":
        if standard_count == 0:
            errors.append("approved standard pack link requires standard_node_ids")
        if library_count == 0:
            errors.append("approved standard pack link requires library_ids")
        if evidence_count == 0:
            errors.append("approved standard pack link requires evidence_ids")

    if errors:
        return _result(
            status=RESULT_INVALID if invalid else RESULT_HOLD,
            active_link_ready=False,
            errors=errors,
            warnings=warnings,
            standard_pack_id=standard_pack_id,
            pack_version=pack_version,
            pack_family=pack_family,
            standard_count=standard_count,
            library_count=library_count,
            evidence_count=evidence_count,
        )

    if pack_status == "approved":
        return _result(
            status=RESULT_VALID,
            active_link_ready=True,
            errors=[],
            warnings=[],
            standard_pack_id=standard_pack_id,
            pack_version=pack_version,
            pack_family=pack_family,
            standard_count=standard_count,
            library_count=library_count,
            evidence_count=evidence_count,
        )

    warnings.append(f"{pack_status} standard pack link is structurally valid but not active-link-ready")
    return _result(
        status=RESULT_VALID,
        active_link_ready=False,
        errors=[],
        warnings=warnings,
        standard_pack_id=standard_pack_id,
        pack_version=pack_version,
        pack_family=pack_family,
        standard_count=standard_count,
        library_count=library_count,
        evidence_count=evidence_count,
    )


__all__ = [
    "RESULT_HOLD",
    "RESULT_INVALID",
    "RESULT_VALID",
    "validate_standard_pack_link",
]
