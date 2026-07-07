from admin.f13_first_wave_safe_backlog import (
    EXCLUDED_BACKLOG_IDS,
    EXPECTED_FIRST_WAVE_ALIAS_COUNT,
    RESULT_HOLD,
    RESULT_READY,
    SECOND_WAVE_SELECTED_IMPROVEMENT_BACKLOG_IDS,
    SELECTED_BACKLOG_IDS,
    build_first_wave_safe_backlog_packet,
    excluded_candidate_preservation,
    second_wave_selected_improvement_guidance,
    selected_candidate_mapping,
)


def _public_values(value):
    if isinstance(value, dict):
        out = []
        for child in value.values():
            out.extend(_public_values(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_public_values(child))
        return out
    return [str(value)]


def _assert_no_execution_flags(packet):
    flags = packet["execution_flags"]
    assert flags["db_access_executed"] is False
    assert flags["file_io_executed"] is False
    assert flags["network_access_executed"] is False
    assert flags["runtime_access_executed"] is False
    assert flags["browser_access_executed"] is False
    assert flags["subprocess_executed"] is False
    assert flags["durable_write_executed"] is False


def test_selected_candidates_are_exact_and_exclusions_are_preserved():
    selected = selected_candidate_mapping()
    excluded = excluded_candidate_preservation()

    assert tuple(item["backlog_id"] for item in selected) == SELECTED_BACKLOG_IDS
    assert tuple(item["backlog_id"] for item in excluded) == EXCLUDED_BACKLOG_IDS
    assert {item["r416_requirement"] for item in selected} == {
        "local_nonprod_safe_summary_feedback_triage",
        "first_wave_observation_criteria",
        "local_boundary_checklist_carry_forward",
    }


def test_packet_is_ready_for_review_with_r412_safe_summary_defaults():
    packet = build_first_wave_safe_backlog_packet()

    assert packet["status"] == RESULT_READY
    assert packet["errors"] == []
    assert tuple(packet["selected_backlog_ids"]) == SELECTED_BACKLOG_IDS
    assert packet["feedback_triage"]["source_policy"] == "SAFE_SUMMARY_ONLY"
    assert packet["feedback_triage"]["raw_text_included"] is False
    assert packet["feedback_triage"]["raw_prompt_included"] is False
    assert packet["feedback_triage"]["paid_standard_raw_text_included"] is False
    assert packet["observation_criteria"]["observed_first_wave_alias_count"] == EXPECTED_FIRST_WAVE_ALIAS_COUNT
    assert packet["observation_criteria"]["stop_condition_triggered"] is False
    assert all(value is False for value in packet["boundary_checklist"]["checks"].values())
    assert packet["preflight_clarification"]["blocks_selected_implementation"] is False
    _assert_no_execution_flags(packet)


def test_unsafe_safe_summary_is_blocked_without_echoing_input():
    unsafe_summary = "raw prompt with H:\\internal\\path and credential marker"

    packet = build_first_wave_safe_backlog_packet(unsafe_summary)
    rendered_values = "\n".join(_public_values(packet)).lower()

    assert packet["status"] == RESULT_HOLD
    assert "SAFE_SUMMARY_UNSAFE_SURFACE_BLOCKED" in packet["errors"]
    assert packet["feedback_triage"]["safe_summary"] == "Safe-summary feedback capture only."
    assert unsafe_summary.lower() not in rendered_values
    _assert_no_execution_flags(packet)


def test_boundary_override_attempt_holds_without_flipping_checklist_flags():
    packet = build_first_wave_safe_backlog_packet(
        boundary_assertions={
            "deploy_executed": True,
            "runtime_server_executed": False,
        }
    )

    assert packet["status"] == RESULT_HOLD
    assert "BOUNDARY_REQUIRES_HIGHER_RISK_GATE:deploy_executed" in packet["errors"]
    assert packet["boundary_checklist"]["checks"]["deploy_executed"] is False
    assert packet["boundary_checklist"]["checks"]["runtime_server_executed"] is False
    _assert_no_execution_flags(packet)


def test_alias_count_or_stop_condition_changes_hold_for_review():
    alias_packet = build_first_wave_safe_backlog_packet(first_wave_alias_count=7)
    stop_packet = build_first_wave_safe_backlog_packet(stop_condition_triggered=True)

    assert alias_packet["status"] == RESULT_HOLD
    assert "FIRST_WAVE_ALIAS_COUNT_MUST_MATCH_REVIEWED_R412_EVIDENCE" in alias_packet["errors"]
    assert stop_packet["status"] == RESULT_HOLD
    assert "STOP_CONDITION_REQUIRES_REVIEW" in stop_packet["errors"]
    _assert_no_execution_flags(alias_packet)
    _assert_no_execution_flags(stop_packet)


def test_bl_004_preflight_clarification_does_not_block_selected_implementation():
    packet = build_first_wave_safe_backlog_packet(
        participant_notice_acknowledgement_reuse_confirmed=None
    )

    assert packet["preflight_clarification"]["backlog_id"] == "R414-BL-004"
    assert packet["preflight_clarification"]["blocks_selected_implementation"] is False
    assert packet["preflight_clarification"]["status"] == (
        "CARRY_FORWARD_CLARIFICATION_REQUIRED_BEFORE_SCOPE_EXPANSION"
    )


def test_second_wave_guidance_makes_evidence_scope_and_hold_wording_explicit():
    guidance = second_wave_selected_improvement_guidance()
    packet = build_first_wave_safe_backlog_packet()

    assert tuple(guidance["selected_backlog_ids"]) == SECOND_WAVE_SELECTED_IMPROVEMENT_BACKLOG_IDS
    assert packet["second_wave_safe_improvements"] == guidance
    assert guidance["evidence_scope_guidance"]["backlog_id"] == "R421-SW-BL-001"
    assert guidance["evidence_scope_guidance"]["source_policy"] == "SAFE_SUMMARY_AND_PROOFPACK_ONLY"
    assert "safe summary evidence" in guidance["evidence_scope_guidance"]["evidence_scope"].lower()
    assert RESULT_HOLD in guidance["evidence_scope_guidance"]["hold_wording"]
    assert "boundary assertion is not explicitly false" in guidance["evidence_scope_guidance"]["hold_wording"]
    assert guidance["implementation_does_not_cover"] == ("R421-SW-BL-003", "R421-SW-BL-004")


def test_second_wave_boundary_reminders_are_visible_and_non_authorizing():
    guidance = second_wave_selected_improvement_guidance()
    reminders = guidance["boundary_reminders"]["reminders"]

    assert guidance["boundary_reminders"]["backlog_id"] == "R421-SW-BL-002"
    assert "alias-only local/internal/nonprod scope" in reminders
    assert "no deploy" in reminders
    assert "no production DB/root" in reminders
    assert "no Production Library root" in reminders
    assert "no raw prompt/body" in reminders
    assert "no raw paid standard text" in reminders
    assert "no secret-like content inspection" in reminders
    assert "no backend/storage creation" in reminders
    assert "no durable feedback write" in reminders
    assert "no public URL" in reminders
    assert "no real participant contact/onboarding" in reminders
    assert guidance["implementation_limit"] == "LOCAL_NONPROD_IN_MEMORY_GUIDANCE_ONLY"
