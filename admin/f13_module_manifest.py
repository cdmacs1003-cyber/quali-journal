from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RESULT_VALID = "VALID"
RESULT_HOLD = "HOLD"
RESULT_INVALID = "INVALID"

_ALLOWED_MODULE_FAMILIES = {"QUALI", "IPC", "ECSS", "NASA", "INTERNAL"}
_ALLOWED_MANIFEST_STATUSES = {"draft", "active", "deprecated"}
_REQUIRED_SCOPE_KEYS = ("library_ids", "graph_node_ids", "evidence_ids")
_SAFE_SURFACE_KEYS = {
    "raw_standard_text_allowed_for_student",
    "internal_path_allowed",
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


def _non_empty_list(value: Any) -> bool:
    return isinstance(value, list) and any(not _is_missing(item) for item in value)


def _scope_has_reference(scope: Any) -> bool:
    if not isinstance(scope, Mapping):
        return False
    return any(_non_empty_list(scope.get(key)) for key in _REQUIRED_SCOPE_KEYS)


def _result(
    *,
    status: str,
    active_ready: bool,
    errors: list[str],
    warnings: list[str],
    module_id: str,
    module_version: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "active_ready": active_ready,
        "hold_reason": errors[0] if errors else None,
        "errors": errors,
        "warnings": warnings,
        "module_id": module_id,
        "module_version": module_version,
        "raw_text_included": False,
        "internal_path_included": False,
        "db_access_executed": False,
        "network_access_executed": False,
        "runtime_access_executed": False,
    }


def validate_module_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return _result(
            status=RESULT_INVALID,
            active_ready=False,
            errors=["payload must be a mapping"],
            warnings=[],
            module_id="",
            module_version="",
        )

    source = payload
    module_id = _safe_token(source.get("module_id"), "")
    module_version = _safe_token(source.get("module_version"), "")
    errors: list[str] = []
    warnings: list[str] = []
    invalid = False

    if _contains_unsafe_surface(source):
        return _result(
            status=RESULT_HOLD,
            active_ready=False,
            errors=["unsafe module manifest payload blocked by static safety boundary"],
            warnings=[],
            module_id=module_id,
            module_version=module_version,
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

    if not module_id:
        errors.append("module_id is required")

    module_family = _safe_token(source.get("module_family"), "").upper()
    if not module_family:
        errors.append("module_family is required")
    elif module_family not in _ALLOWED_MODULE_FAMILIES:
        errors.append("module_family is not supported")
        invalid = True

    if _is_missing(source.get("module_title")):
        errors.append("module_title is required")

    if not module_version:
        errors.append("module_version is required")

    manifest_status = str(source.get("status") or "").strip().lower()
    if not manifest_status:
        errors.append("status is required")
    elif manifest_status not in _ALLOWED_MANIFEST_STATUSES:
        errors.append("status is not supported")
        invalid = True

    if _is_missing(source.get("owner")):
        errors.append("owner is required")

    objective_ids: set[str] = set()
    learning_objectives = source.get("learning_objectives")
    if not isinstance(learning_objectives, list) or not learning_objectives:
        errors.append("learning_objectives must be a non-empty list")
    else:
        for index, objective in enumerate(learning_objectives, start=1):
            if not isinstance(objective, Mapping):
                errors.append(f"learning_objectives[{index}] must be a mapping")
                invalid = True
                continue
            objective_id = _safe_token(objective.get("objective_id"), "")
            if not objective_id:
                errors.append(f"learning_objectives[{index}].objective_id is required")
            else:
                objective_ids.add(objective_id)
            if _is_missing(objective.get("title")):
                errors.append(f"learning_objectives[{index}].title is required")
            if not isinstance(objective.get("linked_library_scope"), Mapping):
                errors.append(f"learning_objectives[{index}].linked_library_scope is required")

    required_library_scope = source.get("required_library_scope")
    if not isinstance(required_library_scope, Mapping):
        errors.append("required_library_scope is required")
    elif not _scope_has_reference(required_library_scope):
        errors.append("required_library_scope must include at least one library, graph, or evidence reference")

    evidence_policy = source.get("evidence_policy")
    if not isinstance(evidence_policy, Mapping):
        errors.append("evidence_policy is required")
    else:
        if evidence_policy.get("evidence_required") is not True:
            errors.append("evidence_policy.evidence_required must be true")
        if str(evidence_policy.get("missing_evidence_action") or "").strip().upper() != RESULT_HOLD:
            errors.append("evidence_policy.missing_evidence_action must be HOLD")
        if evidence_policy.get("raw_standard_text_allowed_for_student") is not False:
            errors.append("student standard-text exposure policy must be false")
        if evidence_policy.get("internal_path_allowed") is not False:
            errors.append("location exposure policy must be false")

    assessment_map = source.get("assessment_map")
    if not isinstance(assessment_map, list) or not assessment_map:
        errors.append("assessment_map must be a non-empty list")
    else:
        for index, assessment in enumerate(assessment_map, start=1):
            if not isinstance(assessment, Mapping):
                errors.append(f"assessment_map[{index}] must be a mapping")
                invalid = True
                continue
            objective_ref = _safe_token(assessment.get("objective_id"), "")
            if not objective_ref:
                errors.append(f"assessment_map[{index}].objective_id is required")
            elif objective_ref not in objective_ids:
                errors.append(f"assessment_map[{index}].objective_id must reference a learning objective")

    if not isinstance(source.get("telemetry_policy"), Mapping):
        errors.append("telemetry_policy is required")

    if errors:
        return _result(
            status=RESULT_INVALID if invalid else RESULT_HOLD,
            active_ready=False,
            errors=errors,
            warnings=warnings,
            module_id=module_id,
            module_version=module_version,
        )

    if manifest_status == "active":
        return _result(
            status=RESULT_VALID,
            active_ready=True,
            errors=[],
            warnings=[],
            module_id=module_id,
            module_version=module_version,
        )

    warnings.append(f"{manifest_status} module manifest is structurally valid but not active-ready")
    return _result(
        status=RESULT_VALID,
        active_ready=False,
        errors=[],
        warnings=warnings,
        module_id=module_id,
        module_version=module_version,
    )


__all__ = [
    "RESULT_HOLD",
    "RESULT_INVALID",
    "RESULT_VALID",
    "validate_module_manifest",
]
