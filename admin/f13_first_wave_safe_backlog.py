from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PACKET_VERSION = "R416-FIRST-WAVE-SAFE-BACKLOG-LOCAL-NONPROD-1"
RESULT_READY = "READY_FOR_LOCAL_NONPROD_REVIEW"
RESULT_HOLD = "HOLD_FOR_BOUNDARY_REVIEW"
EXPECTED_FIRST_WAVE_ALIAS_COUNT = 8

SELECTED_BACKLOG_IDS = ("R414-BL-001", "R414-BL-002", "R414-BL-003")
PREFLIGHT_CLARIFICATION_BACKLOG_ID = "R414-BL-004"
EXCLUDED_BACKLOG_IDS = ("R414-BL-005", "R414-BL-006", "R414-BL-007", "R414-BL-008")
SECOND_WAVE_SELECTED_IMPROVEMENT_BACKLOG_IDS = ("R421-SW-BL-001", "R421-SW-BL-002")

_DEFAULT_SAFE_SUMMARY = "Safe-summary feedback capture only."
_SECOND_WAVE_EVIDENCE_SCOPE_GUIDANCE = {
    "backlog_id": "R421-SW-BL-001",
    "source_policy": "SAFE_SUMMARY_AND_PROOFPACK_ONLY",
    "evidence_scope": "Use only safe summary evidence and proofpack pointers.",
    "hold_wording": (
        "Return HOLD_FOR_BOUNDARY_REVIEW when evidence scope is incomplete or "
        "a boundary assertion is not explicitly false."
    ),
    "forbidden_fields": (
        "raw_prompt_body",
        "raw_paid_standard_text",
        "secret_like_content",
        "real_participant_contact",
    ),
}
_SECOND_WAVE_BOUNDARY_REMINDERS = (
    "alias-only local/internal/nonprod scope",
    "no deploy",
    "no production DB/root",
    "no Production Library root",
    "no raw prompt/body",
    "no raw paid standard text",
    "no secret-like content inspection",
    "no backend/storage creation",
    "no durable feedback write",
    "no public URL",
    "no real participant contact/onboarding",
)
_SELECTED_CANDIDATES = (
    {
        "backlog_id": "R414-BL-001",
        "source_evidence_file": "field_use_safe_feedback_summary.md",
        "safe_summary": _DEFAULT_SAFE_SUMMARY,
        "r416_requirement": "local_nonprod_safe_summary_feedback_triage",
    },
    {
        "backlog_id": "R414-BL-002",
        "source_evidence_file": "field_use_observation_record.md",
        "safe_summary": "Eight first-wave aliases had bounded observations and no stop condition.",
        "r416_requirement": "first_wave_observation_criteria",
    },
    {
        "backlog_id": "R414-BL-003",
        "source_evidence_file": "field_use_access_boundary_verification.md",
        "safe_summary": "Runtime/browser/HTTP/backend/storage/prod roots were not used.",
        "r416_requirement": "local_boundary_checklist_carry_forward",
    },
)
_EXCLUDED_CANDIDATES = (
    {
        "backlog_id": "R414-BL-005",
        "reason": "backend_storage_durable_write_boundary",
    },
    {
        "backlog_id": "R414-BL-006",
        "reason": "sensitive_filename_content_inspection_boundary",
    },
    {
        "backlog_id": "R414-BL-007",
        "reason": "second_wave_not_ready",
    },
    {
        "backlog_id": "R414-BL-008",
        "reason": "release_deploy_production_readiness_non_grant",
    },
)
_BOUNDARY_CHECKLIST = {
    "runtime_server_executed": False,
    "browser_executed": False,
    "loopback_http_executed": False,
    "external_http_executed": False,
    "provider_cloud_executed": False,
    "deploy_executed": False,
    "public_url_created": False,
    "production_db_root_accessed": False,
    "production_library_root_accessed": False,
    "backend_storage_created": False,
    "durable_feedback_write_executed": False,
    "production_feedback_write_executed": False,
    "raw_paid_standard_text_inspected": False,
    "learner_prompt_body_inspected": False,
    "sensitive_content_inspected": False,
    "second_wave_onboarded": False,
    "release_readiness_claimed": False,
}
_UNSAFE_TEXT_MARKERS = (
    "raw text",
    "raw prompt",
    "raw query",
    "raw answer",
    "raw standard",
    "paid standard text",
    "full source",
    "full answer",
    "internal path",
    "file://",
    "localhost",
    "127.0.0.1",
    "h:\\",
    "c:\\",
    "credential",
    "password=",
    "authorization:",
    "api_key",
    "token",
    "secret",
    "dsn=",
)


def _safe_text(value: Any, fallback: str, *, max_length: int = 500) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text[:max_length]


def _unsafe_text_marker_present(value: Any) -> bool:
    lowered = str(value or "").lower()
    return any(marker in lowered for marker in _UNSAFE_TEXT_MARKERS)


def _boundary_checklist(boundary_assertions: Mapping[str, Any] | None) -> tuple[dict[str, bool], list[str]]:
    checklist = dict(_BOUNDARY_CHECKLIST)
    errors: list[str] = []
    if boundary_assertions is None:
        return checklist, errors
    if not isinstance(boundary_assertions, Mapping):
        return checklist, ["BOUNDARY_ASSERTIONS_MUST_BE_MAPPING"]

    for key, value in boundary_assertions.items():
        if key not in checklist:
            errors.append("UNKNOWN_BOUNDARY_ASSERTION")
            continue
        if value is not False:
            errors.append(f"BOUNDARY_REQUIRES_HIGHER_RISK_GATE:{key}")
            continue
        checklist[key] = False
    return checklist, errors


def selected_candidate_mapping() -> list[dict[str, str]]:
    return [dict(candidate) for candidate in _SELECTED_CANDIDATES]


def excluded_candidate_preservation() -> list[dict[str, str]]:
    return [dict(candidate) for candidate in _EXCLUDED_CANDIDATES]


def second_wave_selected_improvement_guidance() -> dict[str, Any]:
    return {
        "selected_backlog_ids": list(SECOND_WAVE_SELECTED_IMPROVEMENT_BACKLOG_IDS),
        "evidence_scope_guidance": dict(_SECOND_WAVE_EVIDENCE_SCOPE_GUIDANCE),
        "boundary_reminders": {
            "backlog_id": "R421-SW-BL-002",
            "visibility": "SHOW_BEFORE_FUTURE_ALIAS_OR_FIELD_USE_DECISION",
            "reminders": list(_SECOND_WAVE_BOUNDARY_REMINDERS),
        },
        "implementation_limit": "LOCAL_NONPROD_IN_MEMORY_GUIDANCE_ONLY",
        "implementation_does_not_cover": (
            "R421-SW-BL-003",
            "R421-SW-BL-004",
        ),
    }


def build_first_wave_safe_backlog_packet(
    safe_feedback_summary: Any = _DEFAULT_SAFE_SUMMARY,
    *,
    first_wave_alias_count: int = EXPECTED_FIRST_WAVE_ALIAS_COUNT,
    stop_condition_triggered: bool = False,
    participant_notice_acknowledgement_reuse_confirmed: bool | None = None,
    boundary_assertions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    safe_summary = _safe_text(safe_feedback_summary, _DEFAULT_SAFE_SUMMARY)
    if _unsafe_text_marker_present(safe_summary):
        safe_summary = _DEFAULT_SAFE_SUMMARY
        errors.append("SAFE_SUMMARY_UNSAFE_SURFACE_BLOCKED")

    if isinstance(first_wave_alias_count, bool) or not isinstance(first_wave_alias_count, int):
        observed_alias_count = 0
        errors.append("FIRST_WAVE_ALIAS_COUNT_MUST_BE_INTEGER")
    else:
        observed_alias_count = first_wave_alias_count
        if observed_alias_count != EXPECTED_FIRST_WAVE_ALIAS_COUNT:
            errors.append("FIRST_WAVE_ALIAS_COUNT_MUST_MATCH_REVIEWED_R412_EVIDENCE")

    if stop_condition_triggered:
        errors.append("STOP_CONDITION_REQUIRES_REVIEW")

    boundary_checklist, boundary_errors = _boundary_checklist(boundary_assertions)
    errors.extend(boundary_errors)

    clarification_status = "CARRY_FORWARD_CLARIFICATION_REQUIRED_BEFORE_SCOPE_EXPANSION"
    if participant_notice_acknowledgement_reuse_confirmed is False:
        clarification_status = "NOT_REUSED_FOR_SCOPE_EXPANSION"
    elif participant_notice_acknowledgement_reuse_confirmed is True:
        clarification_status = "RECORDED_WITH_LIMITS_FOR_CURRENT_REVIEW_ONLY"

    status = RESULT_HOLD if errors else RESULT_READY
    return {
        "packet_version": PACKET_VERSION,
        "status": status,
        "errors": errors,
        "selected_backlog_ids": list(SELECTED_BACKLOG_IDS),
        "excluded_backlog_ids": list(EXCLUDED_BACKLOG_IDS),
        "selected_candidates": selected_candidate_mapping(),
        "excluded_candidates": excluded_candidate_preservation(),
        "feedback_triage": {
            "backlog_id": "R414-BL-001",
            "source_policy": "SAFE_SUMMARY_ONLY",
            "safe_summary": safe_summary,
            "raw_text_included": False,
            "raw_prompt_included": False,
            "paid_standard_raw_text_included": False,
            "sensitive_content_included": False,
            "durable_write_executed": False,
        },
        "observation_criteria": {
            "backlog_id": "R414-BL-002",
            "expected_first_wave_alias_count": EXPECTED_FIRST_WAVE_ALIAS_COUNT,
            "observed_first_wave_alias_count": observed_alias_count,
            "stop_condition_triggered": bool(stop_condition_triggered),
            "field_feedback_executed_again": False,
            "participant_scope_expanded": False,
        },
        "boundary_checklist": {
            "backlog_id": "R414-BL-003",
            "checks": boundary_checklist,
        },
        "second_wave_safe_improvements": second_wave_selected_improvement_guidance(),
        "preflight_clarification": {
            "backlog_id": PREFLIGHT_CLARIFICATION_BACKLOG_ID,
            "status": clarification_status,
            "blocks_selected_implementation": False,
            "allowed_handling": "Carry forward before broader scope decisions.",
            "forbidden_handling": "Do not expand participants or approve second-wave onboarding.",
        },
        "execution_flags": {
            "db_access_executed": False,
            "file_io_executed": False,
            "network_access_executed": False,
            "runtime_access_executed": False,
            "browser_access_executed": False,
            "subprocess_executed": False,
            "durable_write_executed": False,
        },
        "non_authorizations": {
            "implementation_scope": "R416_SELECTED_LOCAL_NONPROD_HELPER_ONLY",
            "deploy": "NOT_AUTHORIZED",
            "production_db_root": "NOT_AUTHORIZED",
            "production_library_root": "NOT_AUTHORIZED",
            "public_url": "NOT_AUTHORIZED",
            "backend_storage": "NOT_AUTHORIZED",
            "durable_feedback_write": "NOT_AUTHORIZED",
            "second_wave": "NOT_AUTHORIZED",
        },
    }


__all__ = [
    "EXCLUDED_BACKLOG_IDS",
    "EXPECTED_FIRST_WAVE_ALIAS_COUNT",
    "PACKET_VERSION",
    "PREFLIGHT_CLARIFICATION_BACKLOG_ID",
    "RESULT_HOLD",
    "RESULT_READY",
    "SELECTED_BACKLOG_IDS",
    "SECOND_WAVE_SELECTED_IMPROVEMENT_BACKLOG_IDS",
    "build_first_wave_safe_backlog_packet",
    "excluded_candidate_preservation",
    "second_wave_selected_improvement_guidance",
    "selected_candidate_mapping",
]
