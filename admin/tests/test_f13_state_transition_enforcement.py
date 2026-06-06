from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from admin.f13_course_library_binding import bind_course_library_reference


ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_GUIDEBOOK = ROOT / "ONTOLOGY_SEMANTIC_WORKFLOW_ENHANCEMENT_GUIDEBOOK_20260516_v1_0_FINAL.md"
CURATION_DECISION_SCHEMA = ROOT / "ImplementationCompletion/F13/schemas/curation_decision.schema.json"
GAP_MAP_CLOSED_EVIDENCE = ROOT / "ImplementationCompletion/F13/F13_gap_map_closed_evidence.md"

STATE_NAMES = {
    "DRAFT",
    "AUTO_SUGGESTED",
    "CURATION_REQUIRED",
    "APPROVED_FOR_WAREHOUSE",
    "APPROVED_FOR_LIBRARY",
    "REQUEST_RIGHTS_REVIEW",
    "REQUEST_DOMAIN_REVIEW",
    "REJECTED",
    "QUARANTINED",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _allowed_transition_rows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in _read_text(ONTOLOGY_GUIDEBOOK).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        from_state, to_state, condition = cells[0], cells[1], cells[2]
        if from_state in STATE_NAMES and to_state in STATE_NAMES:
            rows.append((from_state, to_state, condition))
    assert rows, "committed state transition matrix rows were not parsed"
    return rows


def _assert_forbidden_transition_is_not_allowed(from_state: str, to_state: str) -> None:
    allowed_pairs = {(source, target) for source, target, _condition in _allowed_transition_rows()}
    assert (from_state, to_state) not in allowed_pairs
    assert f"{from_state} -> {to_state}" in _read_text(ONTOLOGY_GUIDEBOOK)


def _binding_payload(current_status: str, **overrides: Any) -> dict[str, Any]:
    payload = {
        "course_id": "course:state-transition",
        "module_id": "module:state-transition",
        "library_node_id": "lib:state-transition",
        "evidence_id": "ev:state-transition",
        "approval_record_id": "approval:state-transition",
        "bridge_trace_id": "btrace:state-transition",
        "current_status": current_status,
        "rights_status": "PUBLIC",
        "raw_text_policy": "SUMMARY_ONLY",
        "validation_shape_ids": ["SH-F13-CURATION-001"],
    }
    payload.update(overrides)
    return payload


def _assert_binding_is_not_skillup_library_use(payload: dict[str, Any]) -> None:
    result = bind_course_library_reference(payload)

    assert result["binding_status"] in {"DENIED", "HOLD"}
    assert result["skillup_use_allowed"] is False
    assert result["feedback_candidate_required"] is True


def test_draft_to_library_approval_is_forbidden_by_committed_transition_contract():
    _assert_forbidden_transition_is_not_allowed("DRAFT", "APPROVED_FOR_LIBRARY")


def test_auto_suggested_to_library_approval_is_forbidden_by_committed_transition_contract():
    _assert_forbidden_transition_is_not_allowed("AUTO_SUGGESTED", "APPROVED_FOR_LIBRARY")


def test_rejected_to_library_direct_approval_is_forbidden_by_committed_transition_contract():
    _assert_forbidden_transition_is_not_allowed("REJECTED", "APPROVED_FOR_LIBRARY")


def test_draft_candidate_cannot_be_bound_as_skillup_library_use():
    _assert_binding_is_not_skillup_library_use(_binding_payload("DRAFT"))


def test_auto_suggested_candidate_cannot_be_bound_as_skillup_library_use():
    _assert_binding_is_not_skillup_library_use(_binding_payload("AUTO_SUGGESTED"))


def test_rejected_candidate_cannot_be_bound_as_skillup_library_use():
    _assert_binding_is_not_skillup_library_use(_binding_payload("REJECTED"))


def test_library_promotion_schema_requires_approval_record_id():
    schema = json.loads(_read_text(CURATION_DECISION_SCHEMA))

    assert "approval_record_id" in schema["required"]
    assert "APPROVE_LIBRARY_PROMOTION" in schema["properties"]["decision"]["enum"]

    promotion_rule = next(
        rule
        for rule in schema["allOf"]
        if (
            rule.get("if", {})
            .get("properties", {})
            .get("decision", {})
            .get("const")
            == "APPROVE_LIBRARY_PROMOTION"
        )
    )
    approval_record_rule = promotion_rule["then"]["properties"]["approval_record_id"]
    assert approval_record_rule["type"] == "string"
    assert approval_record_rule["pattern"].startswith("^approval:")


def test_library_promotion_contract_requires_evidence_approval_and_shape_pass():
    conditions = [
        condition
        for source, target, condition in _allowed_transition_rows()
        if source == "CURATION_REQUIRED" and target == "APPROVED_FOR_LIBRARY"
    ]
    assert conditions, "CURATION_REQUIRED -> APPROVED_FOR_LIBRARY contract row missing"

    rendered_condition = " ".join(conditions)
    assert "evidence_id" in rendered_condition
    assert "approval_record_id" in rendered_condition
    assert "shape PASS" in rendered_condition


def test_not_executed_not_verified_not_granted_boundaries_are_not_silent_pass():
    artifact = _read_text(GAP_MAP_CLOSED_EVIDENCE)

    required_boundaries = (
        "F13 PASS = NOT_GRANTED",
        "Track A PASS = NOT_GRANTED",
        "Beta PASS = NOT_GRANTED",
        "DB behavior = NOT_VERIFIED",
        "Runtime behavior = NOT_VERIFIED",
        "HTTP behavior = NOT_VERIFIED",
        "Full regression = NOT_EXECUTED",
    )
    for boundary in required_boundaries:
        assert boundary in artifact

    forbidden_escalations = (
        "F13 PASS = PASS",
        "Track A PASS = PASS",
        "Beta PASS = PASS",
        "DB behavior = PASS",
        "Runtime behavior = PASS",
        "HTTP behavior = PASS",
        "Full regression = PASS",
    )
    for escalation in forbidden_escalations:
        assert escalation not in artifact
