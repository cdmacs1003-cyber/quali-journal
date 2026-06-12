from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


CREATED_AT = "1970-01-01T00:00:00Z"
RESULT_READY = "READY"
RESULT_HOLD = "HOLD"
RESULT_INVALID = "INVALID"
NOT_GRANTED = "NOT_GRANTED"
NOT_VERIFIED = "NOT_VERIFIED"
NOT_EXECUTED = "NOT_EXECUTED"

_REQUIRED_GATES = (
    "bridge_policy_boundary",
    "skillup_answer_hold_flow",
    "course_library_binding",
    "module_manifest",
    "standard_pack_link",
    "raw_leak_policy_block",
    "feedback_queue",
)
_ALLOWED_GATE_STATUSES = {
    "PASS",
    "HOLD",
    NOT_EXECUTED,
    NOT_VERIFIED,
    NOT_GRANTED,
    "REVIEW_REQUIRED",
}
_NON_CLAIMS = [
    "TRACK_A_PASS_NOT_INFERRED",
    "BETA_PASS_NOT_INFERRED",
    "F13_PASS_NOT_INFERRED",
    "RELEASE_READINESS_NOT_INFERRED",
    "DEPLOYMENT_READINESS_NOT_INFERRED",
    "PRODUCTION_READINESS_NOT_INFERRED",
    "ANSWER_QUALITY_PASS_NOT_INFERRED",
    "BRIDGE_HEALTH_PASS_NOT_INFERRED",
    "BETA_RELEASE_BOARD_PASS_NOT_INFERRED",
    "SELECTED_STATIC_BETA_RELEASE_BOARD_READINESS_ONLY",
]
_OPEN_ITEM_LABELS = {
    "db_behavior": "DB_BEHAVIOR_NOT_VERIFIED",
    "production_raw_leak_safety": "PRODUCTION_RAW_LEAK_SAFETY_NOT_VERIFIED",
    "full_regression_safety": "FULL_REGRESSION_SAFETY_NOT_VERIFIED",
    "proofpack_status": "PROOFPACK_NOT_EXECUTED",
    "gate_matrix_status": "GATE_MATRIX_NOT_COMPLETE",
}
_SAFE_FIELD_KEYS = {
    "raw_leak_zero",
    "raw_leak_zero_evidence",
    "raw_leak_policy_block",
    "raw_export_allowed",
    "raw_text_policy",
}
_STANDARD_TEXT_FIELD_MARKERS = (
    "raw_standard_text",
    "raw standard text",
    "paid_standard_raw",
    "paid standard raw",
    "full_source_text",
    "full source text",
)
_QUERY_FIELD_MARKERS = (
    "raw_query",
    "raw query",
    "raw_user_query",
    "raw user query",
    "raw_prompt",
    "raw prompt",
)
_INTERNAL_FIELD_MARKERS = (
    "internal_path",
    "internal path",
    "file_path",
    "file://",
    "local_path",
    "source_uri_or_path",
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


def _normal_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _has_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_reference(child) for child in value.values())
    if isinstance(value, list | tuple | set):
        return any(not _is_missing(item) for item in value)
    return not _is_missing(value)


def _stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _append_unique(values: list[str], code: str) -> None:
    if code not in values:
        values.append(code)


def _safe_text(value: Any, fallback: str, max_length: int = 180) -> str:
    text = str(value or "").strip() or fallback
    if _surface_findings(text):
        return "redacted_safety_summary"
    return text[:max_length]


def _safe_token(value: Any, fallback: str, max_length: int = 96) -> str:
    text = _safe_text(value, fallback, max_length=max_length)
    token = "".join(ch for ch in text if ch.isalnum() or ch in ":._-")
    return token or fallback


def _field_marker_code(key: Any) -> str | None:
    lowered = str(key or "").lower()
    if lowered in _SAFE_FIELD_KEYS:
        return None
    if any(marker in lowered for marker in _STANDARD_TEXT_FIELD_MARKERS):
        return "STANDARD_TEXT_SURFACE"
    if any(marker in lowered for marker in _QUERY_FIELD_MARKERS):
        return "QUERY_SURFACE"
    if any(marker in lowered for marker in _INTERNAL_FIELD_MARKERS):
        return "INTERNAL_LOCATOR_SURFACE"
    if any(marker in lowered for marker in _SECRET_FIELD_MARKERS):
        return "SECRET_LIKE_SURFACE"
    return None


def _value_marker_code(value: Any) -> str | None:
    lowered = str(value or "").lower()
    if any(marker in lowered for marker in _STANDARD_TEXT_FIELD_MARKERS):
        return "STANDARD_TEXT_SURFACE"
    if any(marker in lowered for marker in _QUERY_FIELD_MARKERS):
        return "QUERY_SURFACE"
    if any(marker in lowered for marker in _INTERNAL_FIELD_MARKERS):
        return "INTERNAL_LOCATOR_SURFACE"
    if any(marker in lowered for marker in _INTERNAL_VALUE_MARKERS):
        return "INTERNAL_LOCATOR_SURFACE"
    if any(marker in lowered for marker in _SECRET_FIELD_MARKERS):
        return "SECRET_LIKE_SURFACE"
    return None


def _surface_findings(value: Any) -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            code = _field_marker_code(key)
            if code is not None:
                _append_unique(findings, code)
            for child_code in _surface_findings(child):
                _append_unique(findings, child_code)
        return findings
    if isinstance(value, list | tuple | set):
        for child in value:
            for child_code in _surface_findings(child):
                _append_unique(findings, child_code)
        return findings
    if isinstance(value, str):
        code = _value_marker_code(value)
        if code is not None:
            _append_unique(findings, code)
    return findings


def _status_label(value: Any) -> str:
    status = _normal_token(value, "RECORDED")
    if status in {"PASS", "PASSED", "SUCCESS"} or "PASS" in status:
        return "BOUNDED_EVIDENCE_RECORDED"
    if status in {"FAIL", "FAILED"}:
        return "REVIEW_REQUIRED"
    return _safe_token(status, "RECORDED")


def _evidence_summary(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []

    summary: list[dict[str, Any]] = []
    for index, record in enumerate(records[:20], start=1):
        if not isinstance(record, Mapping):
            continue
        safe_record = {
            "record_id": _safe_token(record.get("record_id") or record.get("gate") or f"record:{index}", f"record:{index}"),
            "status": _status_label(record.get("status") or record.get("result")),
            "commit_id": _safe_token(record.get("commit_id") or record.get("commit"), "commit:not_recorded"),
            "test_count": int(record.get("test_count") or record.get("tests") or 0),
            "summary": _safe_text(record.get("summary"), "bounded evidence recorded"),
        }
        if _surface_findings(record):
            safe_record["summary"] = "redacted_safety_summary"
        summary.append(safe_record)
    return summary


def _legacy_open_items(payload: Mapping[str, Any]) -> list[str]:
    items: list[str] = []
    if _normal_token(payload.get("db_behavior")) in {NOT_EXECUTED, NOT_VERIFIED, ""}:
        items.append(_OPEN_ITEM_LABELS["db_behavior"])
    if _normal_token(payload.get("production_raw_leak_safety")) in {NOT_VERIFIED, ""}:
        items.append(_OPEN_ITEM_LABELS["production_raw_leak_safety"])
    if _normal_token(payload.get("full_regression_safety")) in {NOT_VERIFIED, ""}:
        items.append(_OPEN_ITEM_LABELS["full_regression_safety"])
    if _normal_token(payload.get("proofpack_status")) in {NOT_EXECUTED, NOT_VERIFIED, ""}:
        items.append(_OPEN_ITEM_LABELS["proofpack_status"])
    if _normal_token(payload.get("gate_matrix_status")) in {NOT_EXECUTED, NOT_VERIFIED, ""}:
        items.append(_OPEN_ITEM_LABELS["gate_matrix_status"])
    return items


def _gate_entries(value: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, Mapping):
        out: dict[str, Mapping[str, Any]] = {}
        for key, gate in value.items():
            if isinstance(gate, Mapping):
                out[_normal_key(key)] = gate
        return out
    if isinstance(value, list):
        out = {}
        for gate in value:
            if not isinstance(gate, Mapping):
                continue
            gate_name = _normal_key(gate.get("gate") or gate.get("gate_id") or gate.get("name"))
            if gate_name:
                out[gate_name] = gate
        return out
    return {}


def _gate_has_actual_evidence(gate: Mapping[str, Any]) -> bool:
    return (
        _has_reference(gate.get("evidence_ref"))
        or _has_reference(gate.get("evidence_ids"))
        or _has_reference(gate.get("proofpack_ref"))
    )


def _gate_has_evidence_representation(gate: Mapping[str, Any]) -> bool:
    return _gate_has_actual_evidence(gate) or not _is_missing(gate.get("missing_evidence_reason"))


def _accepted_out_of_scope(gate: Mapping[str, Any]) -> bool:
    return gate.get("accepted_out_of_scope_for_limited_beta") is True and (
        not _is_missing(gate.get("accepted_out_of_scope_reason"))
        or not _is_missing(gate.get("missing_evidence_reason"))
    )


def _beta_scope_allowed(value: Any) -> bool:
    token = _normal_token(value)
    if not token:
        return False
    if "PRODUCTION" in token or token in {"FULL_RELEASE", "RELEASE", "DEPLOYMENT_RELEASE"}:
        return False
    return "BETA" in token and ("LIMITED" in token or "STATIC" in token or "REVIEW" in token)


def _production_scope_requested(value: Any) -> bool:
    token = _normal_token(value)
    return "PRODUCTION" in token or token in {"FULL_RELEASE", "RELEASE", "DEPLOYMENT_RELEASE"}


def _plan_present(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_reference(child) for child in value.values())
    return _has_reference(value)


def _raw_leak_zero_represented(source: Mapping[str, Any]) -> bool:
    if source.get("raw_leak_zero") is True:
        return True
    if _has_reference(source.get("raw_leak_zero_evidence")):
        return True
    raw_context = source.get("raw_leak_context")
    if isinstance(raw_context, Mapping):
        return raw_context.get("raw_leak_zero") is True or _has_reference(raw_context.get("evidence_ref"))
    return False


def _feedback_queue_represented(source: Mapping[str, Any], gates: Mapping[str, Mapping[str, Any]]) -> bool:
    if _has_reference(source.get("feedback_queue_evidence")):
        return True
    gate = gates.get("feedback_queue")
    return isinstance(gate, Mapping) and _gate_has_actual_evidence(gate)


def _bridge_trace_represented(source: Mapping[str, Any], gates: Mapping[str, Mapping[str, Any]]) -> bool:
    if _has_reference(source.get("bridge_trace_id")) or _has_reference(source.get("bridge_trace_evidence")):
        return True
    gate = gates.get("bridge_policy_boundary")
    return isinstance(gate, Mapping) and (
        _has_reference(gate.get("bridge_trace_id")) or _has_reference(gate.get("trace_id"))
    )


def _answer_or_hold_represented(source: Mapping[str, Any]) -> bool:
    if _has_reference(source.get("evidence_based_answer_evidence")):
        return True
    answer = source.get("evidence_based_answer")
    if isinstance(answer, Mapping) and _has_reference(answer):
        return True
    hold = source.get("hold_fallback_evidence") or source.get("hold_fallback")
    if isinstance(hold, Mapping):
        return _has_reference(hold.get("hold_reason")) or _has_reference(hold.get("evidence_ref"))
    return _has_reference(hold)


def _new_counters(
    *,
    gates: Mapping[str, Mapping[str, Any]],
    missing_gates: int,
    missing_gate_evidence: int,
    accepted_out_of_scope: int,
    surface_findings: list[str],
) -> dict[str, int]:
    return {
        "required_gate_count": len(_REQUIRED_GATES),
        "represented_gate_count": len([gate for gate in _REQUIRED_GATES if gate in gates]),
        "missing_required_gate_count": missing_gates,
        "missing_gate_evidence_count": missing_gate_evidence,
        "accepted_out_of_scope_count": accepted_out_of_scope,
        "unsafe_surface_count": len(surface_findings),
    }


def _result(
    *,
    status: str,
    errors: list[str],
    warnings: list[str],
    counters: dict[str, int],
    checks: dict[str, bool],
    required_gates: dict[str, dict[str, Any]],
    open_items: list[str],
    release_board_id: str,
    beta_scope: str,
) -> dict[str, Any]:
    ready = status == RESULT_READY
    return {
        "status": status,
        "board_ready": ready,
        "hold_reason": None if ready else (errors[0] if errors else (open_items[0] if open_items else None)),
        "errors": errors,
        "warnings": warnings,
        "counters": counters,
        "checks": checks,
        "required_gates": required_gates,
        "open_items": open_items,
        "non_claims": list(_NON_CLAIMS),
        "release_board_id": release_board_id,
        "beta_scope": beta_scope,
        "not_granted_claims": {
            "track_a_pass": NOT_GRANTED,
            "beta_pass": NOT_GRANTED,
            "f13_pass": NOT_GRANTED,
            "release_readiness": NOT_GRANTED,
            "deployment_readiness": NOT_GRANTED,
            "production_readiness": NOT_GRANTED,
            "answer_quality_pass": NOT_GRANTED,
            "bridge_health_pass": NOT_GRANTED,
        },
        "db_access_executed": False,
        "network_access_executed": False,
        "runtime_access_executed": False,
        "file_io_executed": False,
        "env_access_executed": False,
        "subprocess_executed": False,
    }


def validate_beta_release_board_contract(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return _result(
            status=RESULT_INVALID,
            errors=["payload must be a mapping"],
            warnings=[],
            counters=_new_counters(
                gates={},
                missing_gates=len(_REQUIRED_GATES),
                missing_gate_evidence=0,
                accepted_out_of_scope=0,
                surface_findings=[],
            ),
            checks={
                "payload_is_mapping": False,
                "selected_static_beta_release_board_readiness_only": True,
            },
            required_gates={},
            open_items=[],
            release_board_id="",
            beta_scope="",
        )

    errors: list[str] = []
    warnings: list[str] = []
    open_items: list[str] = []
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

    release_board_id_present = not _is_missing(payload.get("release_board_id")) or not _is_missing(
        payload.get("beta_gate_id")
    )
    checks["release_board_or_beta_gate_id_present"] = release_board_id_present
    if not release_board_id_present:
        errors.append("release_board_id or beta_gate_id is required")
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

    beta_scope = _safe_text(payload.get("beta_scope"), "")
    checks["beta_scope_limited_static_review"] = _beta_scope_allowed(payload.get("beta_scope"))
    checks["production_scope_not_requested"] = not _production_scope_requested(payload.get("beta_scope"))
    if _is_missing(payload.get("beta_scope")):
        errors.append("beta_scope is required")
        invalid = True
    elif _production_scope_requested(payload.get("beta_scope")):
        errors.append("HOLD_PRODUCTION_RELEASE_SCOPE_NOT_ALLOWED")
    elif not checks["beta_scope_limited_static_review"]:
        errors.append("HOLD_LIMITED_BETA_STATIC_REVIEW_SCOPE_REQUIRED")

    gates = _gate_entries(payload.get("required_gates"))
    checks["required_gates_present"] = bool(gates)
    if not gates:
        errors.append("required_gates are required")
    sanitized_gates: dict[str, dict[str, Any]] = {}
    missing_gates = 0
    missing_gate_evidence = 0
    accepted_out_of_scope_count = 0

    for gate_name in _REQUIRED_GATES:
        gate = gates.get(gate_name)
        if not isinstance(gate, Mapping):
            missing_gates += 1
            errors.append(f"MISSING_REQUIRED_GATE_{gate_name.upper()}")
            open_items.append(f"MISSING_REQUIRED_GATE_{gate_name.upper()}")
            sanitized_gates[gate_name] = {
                "present": False,
                "status": "MISSING",
                "evidence_represented": False,
                "accepted_out_of_scope_for_limited_beta": False,
            }
            continue

        status = _normal_token(gate.get("status"))
        accepted_out = _accepted_out_of_scope(gate)
        if accepted_out:
            accepted_out_of_scope_count += 1
            warnings.append(f"ACCEPTED_OUT_OF_SCOPE_FOR_LIMITED_BETA_{gate_name.upper()}")
        evidence_represented = _gate_has_evidence_representation(gate)
        actual_evidence = _gate_has_actual_evidence(gate)
        if not status:
            errors.append(f"required gate {gate_name} status is required")
            invalid = True
        elif status not in _ALLOWED_GATE_STATUSES:
            errors.append(f"required gate {gate_name} status is not supported")
            invalid = True
        if not evidence_represented:
            missing_gate_evidence += 1
            errors.append(f"HOLD_REQUIRED_GATE_EVIDENCE_REQUIRED_{gate_name.upper()}")
            open_items.append(f"REQUIRED_GATE_EVIDENCE_REQUIRED_{gate_name.upper()}")
        elif not actual_evidence and not accepted_out:
            missing_gate_evidence += 1
            errors.append(f"HOLD_REQUIRED_GATE_EVIDENCE_MISSING_{gate_name.upper()}")
            open_items.append(f"REQUIRED_GATE_EVIDENCE_MISSING_{gate_name.upper()}")
        if status in {NOT_EXECUTED, NOT_VERIFIED} and not accepted_out:
            errors.append(f"HOLD_REQUIRED_GATE_{status}_{gate_name.upper()}")
            open_items.append(f"REQUIRED_GATE_{status}_{gate_name.upper()}")
        elif status in {"HOLD", "REVIEW_REQUIRED", NOT_GRANTED}:
            open_items.append(f"REQUIRED_GATE_{status}_{gate_name.upper()}")
        sanitized_gates[gate_name] = {
            "present": True,
            "status": status or "MISSING",
            "evidence_represented": evidence_represented,
            "accepted_out_of_scope_for_limited_beta": accepted_out,
        }

    raw_leak_zero = _raw_leak_zero_represented(payload)
    feedback_queue_evidence = _feedback_queue_represented(payload, gates)
    bridge_trace_evidence = _bridge_trace_represented(payload, gates)
    answer_or_hold_evidence = _answer_or_hold_represented(payload)
    checks.update(
        {
            "raw_leak_zero_represented": raw_leak_zero,
            "feedback_queue_evidence_represented": feedback_queue_evidence,
            "bridge_trace_evidence_represented": bridge_trace_evidence,
            "evidence_answer_or_hold_fallback_represented": answer_or_hold_evidence,
            "rollback_plan_present": _plan_present(payload.get("rollback_plan")),
            "incident_log_plan_present": _plan_present(payload.get("incident_log_plan")),
            "daily_beta_summary_plan_present": _plan_present(payload.get("daily_beta_summary_plan")),
            "instructor_operator_handover_note_present": _plan_present(
                payload.get("instructor_operator_handover_note")
            ),
            "deploy_release_approval_not_granted": _normal_token(payload.get("deploy_release_approval")) == NOT_GRANTED,
            "production_readiness_not_granted": _normal_token(payload.get("production_readiness")) == NOT_GRANTED,
        }
    )
    if not raw_leak_zero:
        errors.append("HOLD_RAW_LEAK_ZERO_EVIDENCE_REQUIRED")
        open_items.append("RAW_LEAK_ZERO_EVIDENCE_REQUIRED")
    if not feedback_queue_evidence:
        errors.append("HOLD_FEEDBACK_QUEUE_EVIDENCE_REQUIRED")
        open_items.append("FEEDBACK_QUEUE_EVIDENCE_REQUIRED")
    if not bridge_trace_evidence:
        errors.append("HOLD_BRIDGE_TRACE_EVIDENCE_REQUIRED")
        open_items.append("BRIDGE_TRACE_EVIDENCE_REQUIRED")
    if not answer_or_hold_evidence:
        errors.append("HOLD_EVIDENCE_ANSWER_OR_HOLD_FALLBACK_REQUIRED")
        open_items.append("EVIDENCE_ANSWER_OR_HOLD_FALLBACK_REQUIRED")
    for key, check_key, reason in (
        ("rollback_plan", "rollback_plan_present", "ROLLBACK_PLAN_REQUIRED"),
        ("incident_log_plan", "incident_log_plan_present", "INCIDENT_LOG_PLAN_REQUIRED"),
        ("daily_beta_summary_plan", "daily_beta_summary_plan_present", "DAILY_BETA_SUMMARY_PLAN_REQUIRED"),
        (
            "instructor_operator_handover_note",
            "instructor_operator_handover_note_present",
            "INSTRUCTOR_OPERATOR_HANDOVER_NOTE_REQUIRED",
        ),
    ):
        if not checks[check_key]:
            errors.append(f"HOLD_{reason}")
            open_items.append(reason)
    if not checks["deploy_release_approval_not_granted"]:
        errors.append("HOLD_DEPLOY_RELEASE_APPROVAL_MUST_REMAIN_NOT_GRANTED")
        open_items.append("DEPLOY_RELEASE_APPROVAL_MUST_REMAIN_NOT_GRANTED")
    if not checks["production_readiness_not_granted"]:
        errors.append("HOLD_PRODUCTION_READINESS_MUST_REMAIN_NOT_GRANTED")
        open_items.append("PRODUCTION_READINESS_MUST_REMAIN_NOT_GRANTED")

    surface_findings = _surface_findings(payload)
    checks["standard_text_surface_absent"] = "STANDARD_TEXT_SURFACE" not in surface_findings
    checks["query_surface_absent"] = "QUERY_SURFACE" not in surface_findings
    checks["internal_locator_surface_absent"] = "INTERNAL_LOCATOR_SURFACE" not in surface_findings
    checks["secret_like_surface_absent"] = "SECRET_LIKE_SURFACE" not in surface_findings
    for finding in surface_findings:
        errors.append(f"HOLD_UNSAFE_BETA_RELEASE_BOARD_SURFACE_{finding}")
        open_items.append(f"UNSAFE_BETA_RELEASE_BOARD_SURFACE_{finding}")

    checks["selected_static_beta_release_board_readiness_only"] = True
    checks["no_file_io"] = True
    checks["no_env_access"] = True
    checks["no_subprocess"] = True
    checks["no_network"] = True
    checks["no_db"] = True
    checks["no_runtime"] = True

    if invalid:
        status_result = RESULT_INVALID
    elif errors:
        status_result = RESULT_HOLD
    else:
        status_result = RESULT_READY

    release_board_id = _safe_token(
        payload.get("release_board_id") or payload.get("beta_gate_id"),
        f"beta-board:{_stable_digest(payload.get('contract_version'), payload.get('beta_scope'))}",
    )
    return _result(
        status=status_result,
        errors=errors,
        warnings=warnings,
        counters=_new_counters(
            gates=gates,
            missing_gates=missing_gates,
            missing_gate_evidence=missing_gate_evidence,
            accepted_out_of_scope=accepted_out_of_scope_count,
            surface_findings=surface_findings,
        ),
        checks=checks,
        required_gates=sanitized_gates,
        open_items=open_items,
        release_board_id=release_board_id,
        beta_scope=beta_scope,
    )


def build_beta_release_board(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, Mapping) else {}
    validation = validate_beta_release_board_contract(source)
    evidence_summary = _evidence_summary(source.get("evidence_records"))
    legacy_open_items = _legacy_open_items(source)
    required_open_items = list(validation["open_items"])
    for item in legacy_open_items:
        _append_unique(required_open_items, item)
    scope = _safe_text(source.get("scope") or source.get("beta_scope"), "Track A Skillup Beta")
    board_digest = _stable_digest(
        validation.get("release_board_id"),
        scope,
        ",".join(sorted(required_open_items)),
        validation.get("status"),
    )
    release_board_id = validation.get("release_board_id") or f"beta-board:{board_digest}"
    gate_status = "READY_FOR_REVIEW" if validation["board_ready"] else "REVIEW_REQUIRED"
    board = dict(validation)
    board.update(
        {
            "release_board_id": release_board_id,
            "scope": scope,
            "gate_status": gate_status,
            "evidence_summary": evidence_summary,
            "required_open_items": required_open_items,
            "not_verified_items": [
                "DB_BEHAVIOR",
                "PRODUCTION_RAW_LEAK_SAFETY",
                "FULL_REGRESSION_SAFETY",
            ],
            "approval_status": "NOT_APPROVED",
            "recommendation": gate_status,
            "created_at": CREATED_AT,
        }
    )
    return board


__all__ = [
    "CREATED_AT",
    "NOT_EXECUTED",
    "NOT_GRANTED",
    "NOT_VERIFIED",
    "RESULT_HOLD",
    "RESULT_INVALID",
    "RESULT_READY",
    "build_beta_release_board",
    "validate_beta_release_board_contract",
]
