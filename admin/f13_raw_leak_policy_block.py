from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RESULT_READY = "READY"
RESULT_HOLD = "HOLD"
RESULT_INVALID = "INVALID"

_COUNTER_KEYS = (
    "raw_standard_text_export_count",
    "internal_path_leak_count",
    "secret_like_surface_count",
)
_NON_CLAIMS = [
    "BRIDGE_HEALTH_PASS_NOT_INFERRED",
    "ANSWER_QUALITY_PASS_NOT_INFERRED",
    "SKILLUP_MVP_PASS_NOT_INFERRED",
    "TRACK_A_PASS_NOT_INFERRED",
    "BETA_PASS_NOT_INFERRED",
    "F13_PASS_NOT_INFERRED",
    "RAW_LEAK_POLICY_BLOCK_PASS_FULL_STATUS_NOT_INFERRED",
    "SELECTED_STATIC_POLICY_BLOCK_READINESS_ONLY",
]
_SAFE_FIELD_KEYS = {
    "raw_text_included",
    "internal_path_included",
    "raw_export_allowed",
    "student_raw_text_allowed",
    "secret_surface_allowed",
    "internal_path_allowed",
    "pointer_only_required",
}
_RAW_FIELD_MARKERS = (
    "raw_text",
    "raw text",
    "raw_standard_text",
    "raw standard text",
    "paid_standard_raw",
    "paid standard raw",
    "raw_source_text",
    "full_source_text",
    "full source text",
    "source_text",
)
_INTERNAL_FIELD_MARKERS = (
    "internal_path",
    "internal path",
    "local_path",
    "local path",
    "source_uri_or_path",
    "file_path",
    "file://",
)
_SECRET_FIELD_MARKERS = (
    "api_key",
    "api key",
    "authorization",
    "bearer ",
    "credential",
    "password",
    "private_key",
    "private key",
    "secret",
    "token",
)
_INTERNAL_VALUE_MARKERS = (
    "h:\\",
    "c:\\",
    "/mnt/",
    "/home/",
    "/tmp/",
    "file://",
    "localhost",
    "127.0.0.1",
)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _normal_token(value: Any, fallback: str = "") -> str:
    if _is_missing(value):
        return fallback
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def _has_reference(value: Any) -> bool:
    if isinstance(value, list | tuple | set):
        return any(not _is_missing(item) for item in value)
    return not _is_missing(value)


def _safe_counter(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _new_counters(source: Mapping[str, Any] | None) -> dict[str, int | None]:
    if not isinstance(source, Mapping):
        return {key: None for key in _COUNTER_KEYS}
    return {key: _safe_counter(source.get(key)) for key in _COUNTER_KEYS}


def _append_marker(findings: list[str], code: str) -> None:
    if code not in findings:
        findings.append(code)


def _field_marker_code(key: Any) -> str | None:
    lowered = str(key).lower()
    if lowered in _SAFE_FIELD_KEYS:
        return None
    if any(marker in lowered for marker in _RAW_FIELD_MARKERS):
        return "RAW_STANDARD_TEXT_SURFACE"
    if any(marker in lowered for marker in _INTERNAL_FIELD_MARKERS):
        return "INTERNAL_PATH_SURFACE"
    if any(marker in lowered for marker in _SECRET_FIELD_MARKERS):
        return "SECRET_LIKE_SURFACE"
    return None


def _value_marker_code(value: Any) -> str | None:
    lowered = str(value or "").lower()
    if any(marker in lowered for marker in _RAW_FIELD_MARKERS):
        return "RAW_STANDARD_TEXT_SURFACE"
    if any(marker in lowered for marker in _INTERNAL_FIELD_MARKERS):
        return "INTERNAL_PATH_SURFACE"
    if any(marker in lowered for marker in _INTERNAL_VALUE_MARKERS):
        return "INTERNAL_PATH_SURFACE"
    if any(marker in lowered for marker in _SECRET_FIELD_MARKERS):
        return "SECRET_LIKE_SURFACE"
    return None


def _surface_findings(value: Any) -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            code = _field_marker_code(key)
            if code is not None:
                _append_marker(findings, code)
            for child_code in _surface_findings(child):
                _append_marker(findings, child_code)
        return findings
    if isinstance(value, list | tuple | set):
        for child in value:
            for child_code in _surface_findings(child):
                _append_marker(findings, child_code)
        return findings
    if isinstance(value, str):
        code = _value_marker_code(value)
        if code is not None:
            _append_marker(findings, code)
    return findings


def _result(
    *,
    status: str,
    errors: list[str],
    warnings: list[str],
    counters: dict[str, int | None],
    checks: dict[str, bool],
) -> dict[str, Any]:
    ready = status == RESULT_READY
    return {
        "status": status,
        "policy_block_ready": ready,
        "hold_reason": None if ready else (errors[0] if errors else None),
        "errors": errors,
        "warnings": warnings,
        "counters": counters,
        "checks": checks,
        "non_claims": list(_NON_CLAIMS),
        "raw_text_included": False,
        "internal_path_included": False,
        "secret_surface_included": False,
        "db_access_executed": False,
        "network_access_executed": False,
        "runtime_access_executed": False,
        "file_io_executed": False,
        "env_access_executed": False,
        "subprocess_executed": False,
    }


def validate_raw_leak_policy_block(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        counters = {key: None for key in _COUNTER_KEYS}
        return _result(
            status=RESULT_INVALID,
            errors=["payload must be a mapping"],
            warnings=[],
            counters=counters,
            checks={
                "payload_is_mapping": False,
                "selected_static_policy_block_readiness_only": True,
            },
        )

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {"payload_is_mapping": True}
    invalid = False

    schema_version = payload.get("schema_version")
    checks["schema_version_valid"] = schema_version == 1 or str(schema_version).strip() == "1"
    if _is_missing(schema_version):
        errors.append("schema_version is required")
        invalid = True
    elif not checks["schema_version_valid"]:
        errors.append("schema_version must equal 1")
        invalid = True

    checks["contract_version_present"] = not _is_missing(payload.get("contract_version"))
    if not checks["contract_version_present"]:
        errors.append("contract_version is required")
        invalid = True

    tenant_context = payload.get("tenant_context")
    checks["tenant_context_present"] = isinstance(tenant_context, Mapping)
    if not isinstance(tenant_context, Mapping):
        errors.append("tenant_context is required")
        invalid = True
        tenant_id_present = False
        organization_id_present = False
    else:
        tenant_id_present = not _is_missing(tenant_context.get("tenant_id"))
        organization_id_present = not _is_missing(tenant_context.get("organization_id"))
        if not tenant_id_present:
            errors.append("tenant_context.tenant_id is required")
            invalid = True
        if not organization_id_present:
            errors.append("tenant_context.organization_id is required")
            invalid = True
    checks["tenant_id_present"] = tenant_id_present
    checks["organization_id_present"] = organization_id_present

    evidence_context = payload.get("evidence_context")
    checks["evidence_context_present"] = isinstance(evidence_context, Mapping)
    if not isinstance(evidence_context, Mapping):
        errors.append("evidence_context is required")
        invalid = True
        evidence_linked = False
    else:
        evidence_linked = _has_reference(evidence_context.get("evidence_ids")) or _has_reference(
            evidence_context.get("evidence_pointers")
        )
        if not evidence_linked:
            errors.append("HOLD_EVIDENCE_REQUIRED")
    checks["evidence_linked"] = evidence_linked

    rights_context = payload.get("rights_context")
    checks["rights_context_present"] = isinstance(rights_context, Mapping)
    if not isinstance(rights_context, Mapping):
        errors.append("rights_context is required")
        invalid = True
        rights_status = "NOT_VERIFIED"
    else:
        rights_status = _normal_token(rights_context.get("rights_status"), "NOT_VERIFIED")
        if rights_status in {"UNKNOWN", "NOT_VERIFIED", ""}:
            errors.append("HOLD_RIGHTS_STATUS_NOT_VERIFIED")
    checks["rights_status_known"] = rights_status not in {"UNKNOWN", "NOT_VERIFIED", ""}

    output_policy = payload.get("output_policy")
    checks["output_policy_present"] = isinstance(output_policy, Mapping)
    if not isinstance(output_policy, Mapping):
        errors.append("output_policy is required")
        invalid = True
        output_policy = {}

    pointer_only_required = output_policy.get("pointer_only_required") is True
    raw_export_blocked = output_policy.get("raw_export_allowed") is False
    student_raw_text_blocked = output_policy.get("student_raw_text_allowed") is False
    internal_path_blocked = output_policy.get("internal_path_allowed") is False
    secret_surface_blocked = output_policy.get("secret_surface_allowed") is False
    checks.update(
        {
            "pointer_only_required": pointer_only_required,
            "raw_export_blocked": raw_export_blocked,
            "student_raw_text_blocked": student_raw_text_blocked,
            "internal_path_blocked": internal_path_blocked,
            "secret_surface_blocked": secret_surface_blocked,
        }
    )
    if not pointer_only_required:
        errors.append("HOLD_POINTER_ONLY_REQUIRED")
    if not raw_export_blocked:
        errors.append("HOLD_RAW_EXPORT_BLOCKED_REQUIRED")
    if not student_raw_text_blocked:
        errors.append("HOLD_STUDENT_RAW_TEXT_BLOCKED_REQUIRED")
    if not internal_path_blocked:
        errors.append("HOLD_INTERNAL_PATH_BLOCKED_REQUIRED")
    if not secret_surface_blocked:
        errors.append("HOLD_SECRET_SURFACE_BLOCKED_REQUIRED")

    counters_source = payload.get("counters")
    counters = _new_counters(counters_source if isinstance(counters_source, Mapping) else None)
    checks["counters_present"] = isinstance(counters_source, Mapping)
    if not isinstance(counters_source, Mapping):
        errors.append("counters are required")
        invalid = True
    for key in _COUNTER_KEYS:
        value = counters[key]
        check_key = f"{key}_zero"
        checks[check_key] = value == 0
        if value != 0:
            errors.append(f"HOLD_{key.upper()}_MUST_EQUAL_ZERO")

    output_findings = _surface_findings(payload.get("output_surface", {}))
    checks["output_surface_raw_standard_text_absent"] = "RAW_STANDARD_TEXT_SURFACE" not in output_findings
    checks["output_surface_internal_path_absent"] = "INTERNAL_PATH_SURFACE" not in output_findings
    checks["output_surface_secret_like_absent"] = "SECRET_LIKE_SURFACE" not in output_findings
    for finding in output_findings:
        errors.append(f"HOLD_UNSAFE_OUTPUT_SURFACE_{finding}")

    checks["selected_static_policy_block_readiness_only"] = True
    checks["no_file_io"] = True
    checks["no_env_access"] = True
    checks["no_subprocess"] = True
    checks["no_network"] = True
    checks["no_db"] = True
    checks["no_runtime"] = True

    if invalid:
        status = RESULT_INVALID
    elif errors:
        status = RESULT_HOLD
    else:
        status = RESULT_READY

    return _result(
        status=status,
        errors=errors,
        warnings=warnings,
        counters=counters,
        checks=checks,
    )


__all__ = [
    "RESULT_HOLD",
    "RESULT_INVALID",
    "RESULT_READY",
    "validate_raw_leak_policy_block",
]
