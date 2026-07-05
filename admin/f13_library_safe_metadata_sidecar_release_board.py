"""Release-board draft governance for safe metadata sidecar placement rehearsal.

This module records a production-adjacent placement candidate policy and a
release-board draft for task-owned rehearsal evidence only. It does not write
to production Library roots, open production DBs, expose public pointers, or
grant live placement.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from admin.f13_library_safe_metadata_sidecar_registry import (
    ARTIFACT_STATE_APPROVED_SOURCE,
    ARTIFACT_STATE_CANONICAL_CANDIDATE,
    ARTIFACT_STATE_PROOFPACKED,
    REFRESH_MODE_EXPLICIT_REVIEW,
)
from admin.f13_runtime_guard import RESULT_HOLD, RESULT_OK


GO_HOLD_CUT_HOLD = "HOLD_LIVE_PRODUCTION_ADJACENT_PLACEMENT_NOT_GRANTED"
RELEASE_BOARD_DRAFT_FINAL_RECOMMENDATION = (
    "APPROVE_REHEARSAL_ONLY_WITH_LIMITS_LIVE_USE_NOT_GRANTED"
)

REVIEW_STATUS_DRAFT_REQUIRED = "DRAFT_REVIEW_REQUIRED_BEFORE_LIVE_USE"

NOT_GRANTED_CLAIMS = (
    "PRODUCTION_ADJACENT_PLACEMENT_PASS",
    "DB_BACKED_RETRIEVAL_PASS",
    "PRODUCTION_DB_OK_RETRIEVAL_PASS",
    "PRODUCTION_LIBRARY_ROOT_PASS",
    "PUBLIC_API_PASS",
    "FULL_E2E_PASS",
    "BETA_PASS",
    "TRACK_A_PASS",
    "F13_PASS",
    "BROWSER_PASS",
    "SKILLUP_MVP_PASS",
    "RELEASE_READY",
    "DEPLOYMENT_READY",
    "PRODUCTION_READY",
)

_REQUIRED_POLICY_FIELDS = (
    "sidecar_root_candidate",
    "manifest_root_candidate",
    "ownership",
    "review_status_required",
    "allowed_artifact_states",
    "hash_validation_required",
    "rollback_requirement",
    "refresh_requirement",
    "expiry_or_review_date_requirement",
    "production_db_mutation_forbidden",
    "production_raw_text_read_forbidden",
    "public_pointer_exposure_forbidden",
    "skillup_direct_db_access_forbidden",
    "bridge_only_retrieval_required",
)

_REQUIRED_RELEASE_BOARD_FIELDS = (
    "release_board_id",
    "task_id",
    "source_commit",
    "sidecar_id",
    "sidecar_manifest_hash",
    "sidecar_sqlite_hash",
    "sidecar_json_hash",
    "resolver_validation",
    "bridge_retrieval_validation",
    "skillup_public_exposure_check",
    "rollback_plan",
    "refresh_plan",
    "review_status",
    "approval_required_before_live_use",
    "NOT_GRANTED claims preserved",
    "go_hold_cut_decision",
    "final_recommendation",
)

_ALLOWED_ARTIFACT_STATES = {
    ARTIFACT_STATE_APPROVED_SOURCE,
    ARTIFACT_STATE_PROOFPACKED,
    ARTIFACT_STATE_CANONICAL_CANDIDATE,
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _safe_text(value: object, max_length: int = 240) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > max_length:
        return None
    if any(ord(char) < 32 for char in text):
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in (".env", "secret", "token", "credential", "api_key")):
        return None
    return text


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _ok_validation(summary: str) -> dict[str, Any]:
    return {
        "result_status": RESULT_OK,
        "ok": True,
        "summary": summary,
    }


def _validate_result_mapping(value: object, field: str) -> list[str]:
    errors: list[str] = []
    mapping = _as_mapping(value)
    if not mapping:
        return [f"{field} must be a validation object"]
    if mapping.get("result_status") != RESULT_OK or mapping.get("ok") is not True:
        errors.append(f"{field} must be OK")
    return errors


def _validate_sha256(value: object, field: str) -> list[str]:
    text = _safe_text(value, 64)
    if text is None or not _HASH_RE.fullmatch(text.lower()):
        return [f"{field} must be a SHA-256 hex digest"]
    return []


def build_production_adjacent_placement_candidate_policy(
    *,
    sidecar_root_candidate: str,
    manifest_root_candidate: str,
    ownership: str = "F13 Bridge safe metadata sidecar governance",
    review_status_required: str = "REVIEW_REQUIRED_BEFORE_LIVE_USE",
    expiry_or_review_date_requirement: str = "NEXT_PLACEMENT_GATE_REVIEW_REQUIRED",
    allowed_artifact_states: Sequence[str] = (
        ARTIFACT_STATE_APPROVED_SOURCE,
        ARTIFACT_STATE_PROOFPACKED,
        ARTIFACT_STATE_CANONICAL_CANDIDATE,
    ),
) -> dict[str, Any]:
    """Build a policy-only production-adjacent placement candidate."""

    return {
        "sidecar_root_candidate": sidecar_root_candidate,
        "manifest_root_candidate": manifest_root_candidate,
        "ownership": ownership,
        "review_status_required": review_status_required,
        "allowed_artifact_states": list(allowed_artifact_states),
        "hash_validation_required": True,
        "rollback_requirement": "PRESERVE_PRIOR_SIDECAR_AND_REPOINT_MANIFEST_AFTER_REVIEW",
        "refresh_requirement": REFRESH_MODE_EXPLICIT_REVIEW,
        "expiry_or_review_date_requirement": expiry_or_review_date_requirement,
        "production_db_mutation_forbidden": True,
        "production_raw_text_read_forbidden": True,
        "public_pointer_exposure_forbidden": True,
        "skillup_direct_db_access_forbidden": True,
        "bridge_only_retrieval_required": True,
        "production_root_write_allowed_for_rehearsal": False,
        "live_production_adjacent_placement_granted": False,
    }


def validate_placement_candidate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a candidate policy is non-mutating and review-gated."""

    errors: list[str] = []
    for field in _REQUIRED_POLICY_FIELDS:
        if field not in policy:
            errors.append(f"{field} is missing")

    for field in (
        "sidecar_root_candidate",
        "manifest_root_candidate",
        "ownership",
        "review_status_required",
        "rollback_requirement",
        "refresh_requirement",
        "expiry_or_review_date_requirement",
    ):
        if field in policy and _safe_text(policy.get(field), 320) is None:
            errors.append(f"{field} must be a safe non-empty label")

    allowed_states = {str(item) for item in _as_list(policy.get("allowed_artifact_states"))}
    if not allowed_states:
        errors.append("allowed_artifact_states must be non-empty")
    if allowed_states - _ALLOWED_ARTIFACT_STATES:
        errors.append("allowed_artifact_states contains unapproved states")

    required_true_fields = (
        "hash_validation_required",
        "production_db_mutation_forbidden",
        "production_raw_text_read_forbidden",
        "public_pointer_exposure_forbidden",
        "skillup_direct_db_access_forbidden",
        "bridge_only_retrieval_required",
    )
    for field in required_true_fields:
        if policy.get(field) is not True:
            errors.append(f"{field} must be explicit true")

    required_false_fields = (
        "production_root_write_allowed_for_rehearsal",
        "live_production_adjacent_placement_granted",
    )
    for field in required_false_fields:
        if policy.get(field) is not False:
            errors.append(f"{field} must be explicit false")

    if policy.get("refresh_requirement") != REFRESH_MODE_EXPLICIT_REVIEW:
        errors.append("refresh_requirement must remain explicit review")
    if "REVIEW" not in str(policy.get("review_status_required") or "").upper():
        errors.append("review_status_required must require review")

    status = RESULT_OK if not errors else RESULT_HOLD
    return {
        "result_status": status,
        "ok": status == RESULT_OK,
        "errors": errors,
    }


def build_sidecar_release_board_draft(
    *,
    release_board_id: str,
    task_id: str,
    source_commit: str,
    sidecar_id: str,
    sidecar_manifest_hash: str,
    sidecar_sqlite_hash: str,
    sidecar_json_hash: str,
    resolver_validation: Mapping[str, Any],
    bridge_retrieval_validation: Mapping[str, Any],
    skillup_public_exposure_check: Mapping[str, Any],
    rollback_plan: str,
    refresh_plan: str,
    review_status: str = REVIEW_STATUS_DRAFT_REQUIRED,
    not_granted_claims: Sequence[str] = NOT_GRANTED_CLAIMS,
) -> dict[str, Any]:
    """Build a release-board draft for rehearsal evidence only."""

    return {
        "release_board_id": release_board_id,
        "task_id": task_id,
        "source_commit": source_commit,
        "sidecar_id": sidecar_id,
        "sidecar_manifest_hash": sidecar_manifest_hash.lower(),
        "sidecar_sqlite_hash": sidecar_sqlite_hash.lower(),
        "sidecar_json_hash": sidecar_json_hash.lower(),
        "resolver_validation": dict(resolver_validation),
        "bridge_retrieval_validation": dict(bridge_retrieval_validation),
        "skillup_public_exposure_check": dict(skillup_public_exposure_check),
        "rollback_plan": rollback_plan,
        "refresh_plan": refresh_plan,
        "review_status": review_status,
        "approval_required_before_live_use": True,
        "NOT_GRANTED claims preserved": list(not_granted_claims),
        "go_hold_cut_decision": GO_HOLD_CUT_HOLD,
        "final_recommendation": RELEASE_BOARD_DRAFT_FINAL_RECOMMENDATION,
    }


def validate_sidecar_release_board_draft(board: Mapping[str, Any]) -> dict[str, Any]:
    """Validate release-board draft fields and prevent readiness escalation."""

    errors: list[str] = []
    for field in _REQUIRED_RELEASE_BOARD_FIELDS:
        if field not in board:
            errors.append(f"{field} is missing")

    for field in (
        "release_board_id",
        "task_id",
        "source_commit",
        "sidecar_id",
        "rollback_plan",
        "refresh_plan",
        "review_status",
        "go_hold_cut_decision",
        "final_recommendation",
    ):
        if field in board and _safe_text(board.get(field), 800) is None:
            errors.append(f"{field} must be a safe non-empty label")

    for field in ("sidecar_manifest_hash", "sidecar_sqlite_hash", "sidecar_json_hash"):
        if field in board:
            errors.extend(_validate_sha256(board.get(field), field))

    for field in (
        "resolver_validation",
        "bridge_retrieval_validation",
        "skillup_public_exposure_check",
    ):
        if field in board:
            errors.extend(_validate_result_mapping(board.get(field), field))

    if board.get("approval_required_before_live_use") is not True:
        errors.append("approval_required_before_live_use must be explicit true")
    if board.get("go_hold_cut_decision") != GO_HOLD_CUT_HOLD:
        errors.append("go_hold_cut_decision must remain HOLD")
    if board.get("final_recommendation") != RELEASE_BOARD_DRAFT_FINAL_RECOMMENDATION:
        errors.append("final_recommendation must be rehearsal-only")
    if str(board.get("review_status") or "") not in {
        REVIEW_STATUS_DRAFT_REQUIRED,
        "REVIEW_REQUIRED_BEFORE_LIVE_USE",
        "HOLD_REVIEW_REQUIRED_BEFORE_LIVE_USE",
    }:
        errors.append("review_status must require review before live use")

    preserved = set(str(item) for item in _as_list(board.get("NOT_GRANTED claims preserved")))
    missing_claims = set(NOT_GRANTED_CLAIMS) - preserved
    if missing_claims:
        errors.append("NOT_GRANTED claims preserved is missing required claims")

    exposure = _as_mapping(board.get("skillup_public_exposure_check"))
    hits = _as_list(exposure.get("forbidden_marker_hits"))
    if hits:
        errors.append("skillup_public_exposure_check contains forbidden marker hits")

    status = RESULT_OK if not errors else RESULT_HOLD
    return {
        "result_status": status,
        "ok": status == RESULT_OK,
        "errors": errors,
        "required_fields": list(_REQUIRED_RELEASE_BOARD_FIELDS),
    }


__all__ = [
    "GO_HOLD_CUT_HOLD",
    "NOT_GRANTED_CLAIMS",
    "RELEASE_BOARD_DRAFT_FINAL_RECOMMENDATION",
    "REVIEW_STATUS_DRAFT_REQUIRED",
    "build_production_adjacent_placement_candidate_policy",
    "build_sidecar_release_board_draft",
    "validate_placement_candidate_policy",
    "validate_sidecar_release_board_draft",
]
