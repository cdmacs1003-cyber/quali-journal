from typing import Any

from admin.f13_beta_release_board import (
    build_beta_release_board,
    validate_beta_release_board_contract,
)


_REQUIRED_GATES = (
    "bridge_policy_boundary",
    "skillup_answer_hold_flow",
    "course_library_binding",
    "module_manifest",
    "standard_pack_link",
    "raw_leak_policy_block",
    "feedback_queue",
)


def _gate(name: str) -> dict[str, Any]:
    gate = {
        "status": "PASS",
        "evidence_ref": f"urn:qlib:proof:{name}",
        "proofpack_ref": f"proofpack:{name}",
    }
    if name == "bridge_policy_boundary":
        gate["bridge_trace_id"] = "btrace:beta-board-bridge"
    if name == "raw_leak_policy_block":
        gate["raw_leak_zero"] = True
    return gate


def _ready_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "release_board_id": "BRB-track-a-limited-beta-1",
        "tenant_context": {
            "tenant_id": "TEN-diagnostic",
            "organization_id": "ORGUNIT-diagnostic",
        },
        "beta_scope": "limited_beta_static_readiness_review",
        "required_gates": {name: _gate(name) for name in _REQUIRED_GATES},
        "raw_leak_zero": True,
        "raw_leak_zero_evidence": "urn:qlib:evidence:raw-leak-zero",
        "feedback_queue_evidence": "urn:qlib:evidence:feedback-queue",
        "bridge_trace_id": "btrace:beta-board-bridge",
        "evidence_based_answer": {
            "evidence_ids": ["EVD-answer-1"],
            "answer_status": "ANSWERED",
        },
        "rollback_plan": {"plan_ref": "urn:qlib:ops:rollback"},
        "incident_log_plan": {"plan_ref": "urn:qlib:ops:incident-log"},
        "daily_beta_summary_plan": {"plan_ref": "urn:qlib:ops:daily-summary"},
        "instructor_operator_handover_note": {"note_ref": "urn:qlib:ops:handover"},
        "deploy_release_approval": "NOT_GRANTED",
        "production_readiness": "NOT_GRANTED",
        "evidence_records": [
            {
                "record_id": "feedback_queue_selected_test",
                "status": "PASS",
                "commit_id": "b0d821f",
                "test_count": 30,
                "summary": "Selected static feedback queue evidence recorded.",
            },
        ],
    }


def _assert_hold_or_invalid(result: dict[str, Any]) -> None:
    assert result["status"] in {"HOLD", "INVALID"}
    assert result["board_ready"] is False
    assert result["hold_reason"]
    assert result["errors"]
    _assert_no_execution_flags(result)


def _assert_no_execution_flags(result: dict[str, Any]) -> None:
    assert result["db_access_executed"] is False
    assert result["network_access_executed"] is False
    assert result["runtime_access_executed"] is False
    assert result["file_io_executed"] is False
    assert result["env_access_executed"] is False
    assert result["subprocess_executed"] is False


def _walk_public_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(_walk_public_values(child))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for child in value:
            out.extend(_walk_public_values(child))
        return out
    return [str(value)]


def test_valid_limited_beta_board_payload_returns_ready_and_board_ready_true():
    result = validate_beta_release_board_contract(_ready_payload())

    assert result["status"] == "READY"
    assert result["board_ready"] is True
    assert result["hold_reason"] is None
    assert result["errors"] == []
    assert result["checks"]["beta_scope_limited_static_review"] is True
    _assert_no_execution_flags(result)


def test_production_release_scope_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["beta_scope"] = "production_release"

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_non_dict_payload_returns_invalid():
    result = validate_beta_release_board_contract(["not", "a", "mapping"])

    assert result["status"] == "INVALID"
    assert result["board_ready"] is False
    _assert_no_execution_flags(result)


def test_missing_schema_version_returns_invalid():
    payload = _ready_payload()
    payload.pop("schema_version")

    result = validate_beta_release_board_contract(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_missing_contract_version_returns_invalid():
    payload = _ready_payload()
    payload.pop("contract_version")

    result = validate_beta_release_board_contract(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_missing_release_board_id_and_beta_gate_id_returns_invalid():
    payload = _ready_payload()
    payload.pop("release_board_id")

    result = validate_beta_release_board_contract(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_missing_tenant_context_returns_invalid_or_hold():
    payload = _ready_payload()
    payload.pop("tenant_context")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_tenant_id_returns_invalid_or_hold():
    payload = _ready_payload()
    payload["tenant_context"].pop("tenant_id")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_organization_id_returns_invalid_or_hold():
    payload = _ready_payload()
    payload["tenant_context"].pop("organization_id")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_required_gates_returns_hold_or_invalid():
    payload = _ready_payload()
    payload.pop("required_gates")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_bridge_policy_boundary_gate_blocks_board_ready():
    payload = _ready_payload()
    payload["required_gates"].pop("bridge_policy_boundary")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_skillup_answer_hold_flow_gate_blocks_board_ready():
    payload = _ready_payload()
    payload["required_gates"].pop("skillup_answer_hold_flow")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_course_library_binding_gate_blocks_board_ready():
    payload = _ready_payload()
    payload["required_gates"].pop("course_library_binding")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_module_manifest_gate_blocks_board_ready():
    payload = _ready_payload()
    payload["required_gates"].pop("module_manifest")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_standard_pack_link_gate_blocks_board_ready():
    payload = _ready_payload()
    payload["required_gates"].pop("standard_pack_link")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_raw_leak_policy_block_gate_blocks_board_ready():
    payload = _ready_payload()
    payload["required_gates"].pop("raw_leak_policy_block")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_feedback_queue_gate_blocks_board_ready():
    payload = _ready_payload()
    payload["required_gates"].pop("feedback_queue")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_evidence_for_required_gate_blocks_board_ready():
    payload = _ready_payload()
    payload["required_gates"]["course_library_binding"] = {"status": "PASS"}

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_not_executed_gate_blocks_board_ready_unless_limited_beta_out_of_scope_reason_exists():
    payload = _ready_payload()
    payload["required_gates"]["standard_pack_link"]["status"] = "NOT_EXECUTED"

    blocked = validate_beta_release_board_contract(payload)
    _assert_hold_or_invalid(blocked)

    payload["required_gates"]["standard_pack_link"]["accepted_out_of_scope_for_limited_beta"] = True
    payload["required_gates"]["standard_pack_link"]["accepted_out_of_scope_reason"] = "Static beta board notes gap."

    accepted = validate_beta_release_board_contract(payload)
    assert accepted["status"] == "READY"
    assert accepted["board_ready"] is True


def test_not_verified_gate_blocks_board_ready_unless_limited_beta_out_of_scope_reason_exists():
    payload = _ready_payload()
    payload["required_gates"]["module_manifest"]["status"] = "NOT_VERIFIED"

    blocked = validate_beta_release_board_contract(payload)
    _assert_hold_or_invalid(blocked)

    payload["required_gates"]["module_manifest"]["accepted_out_of_scope_for_limited_beta"] = True
    payload["required_gates"]["module_manifest"]["accepted_out_of_scope_reason"] = "Static beta board notes gap."

    accepted = validate_beta_release_board_contract(payload)
    assert accepted["status"] == "READY"
    assert accepted["board_ready"] is True


def test_raw_leak_zero_evidence_missing_blocks_board_ready():
    payload = _ready_payload()
    payload.pop("raw_leak_zero")
    payload.pop("raw_leak_zero_evidence")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_feedback_queue_evidence_missing_blocks_board_ready():
    payload = _ready_payload()
    payload.pop("feedback_queue_evidence")
    payload["required_gates"]["feedback_queue"] = {"status": "PASS"}

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_bridge_trace_evidence_missing_blocks_board_ready():
    payload = _ready_payload()
    payload.pop("bridge_trace_id")
    payload["required_gates"]["bridge_policy_boundary"].pop("bridge_trace_id")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_evidence_answer_or_hold_fallback_missing_blocks_board_ready():
    payload = _ready_payload()
    payload.pop("evidence_based_answer")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_rollback_plan_missing_blocks_board_ready():
    payload = _ready_payload()
    payload.pop("rollback_plan")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_incident_log_plan_missing_blocks_board_ready():
    payload = _ready_payload()
    payload.pop("incident_log_plan")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_daily_beta_summary_plan_missing_blocks_board_ready():
    payload = _ready_payload()
    payload.pop("daily_beta_summary_plan")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_instructor_operator_handover_note_missing_blocks_board_ready():
    payload = _ready_payload()
    payload.pop("instructor_operator_handover_note")

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_deploy_release_approval_granted_incorrectly_blocks_or_warns():
    payload = _ready_payload()
    payload["deploy_release_approval"] = "APPROVED"

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_production_readiness_granted_incorrectly_blocks_or_warns():
    payload = _ready_payload()
    payload["production_readiness"] = "APPROVED"

    result = validate_beta_release_board_contract(payload)

    _assert_hold_or_invalid(result)


def test_unsafe_raw_standard_text_is_not_echoed():
    payload = _ready_payload()
    unsafe_value = "raw standard text must not surface"
    payload["board_surface"] = {"student_copy": unsafe_value}

    result = validate_beta_release_board_contract(payload)
    rendered = "\n".join(_walk_public_values(result)).lower()

    _assert_hold_or_invalid(result)
    assert unsafe_value not in rendered


def test_unsafe_raw_user_query_is_not_echoed():
    payload = _ready_payload()
    unsafe_value = "raw user query must not surface"
    payload["board_surface"] = {"learner_query": unsafe_value}

    result = validate_beta_release_board_contract(payload)
    rendered = "\n".join(_walk_public_values(result)).lower()

    _assert_hold_or_invalid(result)
    assert unsafe_value not in rendered


def test_internal_path_marker_is_not_echoed():
    payload = _ready_payload()
    unsafe_value = "H:\\internal\\beta-release-board.txt"
    payload["board_surface"] = {"storage_note": unsafe_value}

    result = validate_beta_release_board_contract(payload)
    rendered = "\n".join(_walk_public_values(result)).lower()

    _assert_hold_or_invalid(result)
    assert unsafe_value.lower() not in rendered


def test_secret_like_marker_is_not_echoed():
    payload = _ready_payload()
    unsafe_value = "api token value"
    payload["board_surface"] = {"operator_note": unsafe_value}

    result = validate_beta_release_board_contract(payload)
    rendered = "\n".join(_walk_public_values(result)).lower()

    _assert_hold_or_invalid(result)
    assert unsafe_value not in rendered


def test_non_claims_include_no_track_a_beta_f13_release_or_deployment_escalation():
    result = validate_beta_release_board_contract(_ready_payload())

    assert result["non_claims"] == [
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
    assert result["not_granted_claims"]["track_a_pass"] == "NOT_GRANTED"
    assert result["not_granted_claims"]["beta_pass"] == "NOT_GRANTED"
    assert result["not_granted_claims"]["f13_pass"] == "NOT_GRANTED"


def test_no_db_network_runtime_file_env_or_subprocess_flags_are_present_or_false():
    ready_result = validate_beta_release_board_contract(_ready_payload())
    hold_payload = _ready_payload()
    hold_payload.pop("rollback_plan")
    hold_result = validate_beta_release_board_contract(hold_payload)

    for result in (ready_result, hold_result):
        _assert_no_execution_flags(result)


def test_build_beta_release_board_preserves_existing_api_and_wraps_validation():
    board = build_beta_release_board(_ready_payload())

    assert board["status"] == "READY"
    assert board["board_ready"] is True
    assert board["release_board_id"] == "BRB-track-a-limited-beta-1"
    assert board["gate_status"] == "READY_FOR_REVIEW"
    assert board["recommendation"] == "READY_FOR_REVIEW"
    assert board["approval_status"] == "NOT_APPROVED"
    assert board["evidence_summary"]
    assert board["not_granted_claims"]["deployment_readiness"] == "NOT_GRANTED"
