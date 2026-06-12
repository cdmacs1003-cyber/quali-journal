from admin.f13_feedback_queue_contract import validate_feedback_queue_contract


def _ready_payload():
    return {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "feedback_id": "FBQ-diagnostic-answer-1",
        "request_id": "REQ-diagnostic-answer-1",
        "tenant_context": {
            "tenant_id": "TEN-diagnostic",
            "organization_id": "ORGUNIT-diagnostic",
        },
        "course_context": {
            "course_id": "CRS-diagnostic",
            "module_id": "MOD-diagnostic",
        },
        "event_context": {
            "event_type": "answer_rendered",
        },
        "answer_status": "ANSWERED",
        "bridge_trace_id": "btrace:diagnostic-answer-1",
        "evidence_context": {
            "evidence_ids": ["EVD-diagnostic-1"],
            "evidence_pointers": ["urn:qlib:evidence:diagnostic-1"],
        },
        "feedback_policy": {
            "user_raw_query_stored": False,
            "raw_answer_stored": False,
            "internal_path_allowed": False,
            "secret_surface_allowed": False,
            "paid_standard_raw_text_allowed": False,
            "feedback_text_policy": "summary_or_pointer_only",
            "automation_may_promote_to_library": False,
            "human_review_required": True,
        },
        "curation_target": "qa_case_candidate",
        "feedback_surface": {
            "safe_summary": "Answer feedback captured as safe summary metadata.",
            "review_pointer": "urn:qlib:feedback:diagnostic-answer-1",
        },
    }


def _hold_payload():
    payload = _ready_payload()
    payload["feedback_id"] = "FBQ-diagnostic-hold-1"
    payload["event_context"]["event_type"] = "hold_created"
    payload["answer_status"] = "HOLD"
    payload["hold_reason"] = "Evidence gap requires curation review."
    payload["bridge_trace_id"] = "btrace:diagnostic-hold-1"
    payload["evidence_context"] = {
        "evidence_ids": [],
        "evidence_pointers": [],
        "missing_evidence_reason": "No safe evidence pointer was available.",
    }
    payload["curation_target"] = "evidence_gap_queue"
    return payload


def _assert_hold_or_invalid(result):
    assert result["status"] in {"HOLD", "INVALID"}
    assert result["queue_ready"] is False
    assert result["hold_reason"]
    assert result["errors"]
    _assert_no_execution_flags(result)


def _assert_no_execution_flags(result):
    assert result["db_access_executed"] is False
    assert result["network_access_executed"] is False
    assert result["runtime_access_executed"] is False
    assert result["file_io_executed"] is False
    assert result["env_access_executed"] is False
    assert result["subprocess_executed"] is False
    assert result["raw_user_query_included"] is False
    assert result["raw_answer_included"] is False
    assert result["internal_path_included"] is False
    assert result["secret_surface_included"] is False
    assert result["paid_standard_raw_text_included"] is False


def _walk_public_values(value):
    if isinstance(value, dict):
        out = []
        for child in value.values():
            out.extend(_walk_public_values(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_walk_public_values(child))
        return out
    return [str(value)]


def test_valid_answer_feedback_payload_returns_ready_and_queue_ready_true():
    result = validate_feedback_queue_contract(_ready_payload())

    assert result["status"] == "READY"
    assert result["queue_ready"] is True
    assert result["hold_reason"] is None
    assert result["errors"] == []
    assert result["checks"]["trace_present"] is True
    assert result["checks"]["human_review_required"] is True
    _assert_no_execution_flags(result)


def test_valid_hold_feedback_payload_returns_ready_and_queue_ready_true():
    result = validate_feedback_queue_contract(_hold_payload())

    assert result["status"] == "READY"
    assert result["queue_ready"] is True
    assert result["hold_reason"] is None
    assert result["errors"] == []
    assert result["checks"]["hold_reason_present_when_required"] is True
    assert result["checks"]["evidence_or_missing_reason_present"] is True
    _assert_no_execution_flags(result)


def test_non_dict_payload_returns_invalid():
    result = validate_feedback_queue_contract(["not", "a", "mapping"])

    assert result["status"] == "INVALID"
    assert result["queue_ready"] is False
    _assert_no_execution_flags(result)


def test_missing_schema_version_returns_invalid():
    payload = _ready_payload()
    payload.pop("schema_version")

    result = validate_feedback_queue_contract(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_missing_contract_version_returns_invalid():
    payload = _ready_payload()
    payload.pop("contract_version")

    result = validate_feedback_queue_contract(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_missing_feedback_id_and_request_id_returns_invalid():
    payload = _ready_payload()
    payload.pop("feedback_id")
    payload.pop("request_id")

    result = validate_feedback_queue_contract(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_missing_tenant_context_returns_invalid_or_hold():
    payload = _ready_payload()
    payload.pop("tenant_context")

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_tenant_id_returns_invalid_or_hold():
    payload = _ready_payload()
    payload["tenant_context"].pop("tenant_id")

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_organization_id_returns_invalid_or_hold():
    payload = _ready_payload()
    payload["tenant_context"].pop("organization_id")

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_event_type_returns_invalid():
    payload = _ready_payload()
    payload["event_context"].pop("event_type")

    result = validate_feedback_queue_contract(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_unsupported_event_type_returns_invalid():
    payload = _ready_payload()
    payload["event_context"]["event_type"] = "runtime_queue_inserted"

    result = validate_feedback_queue_contract(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_missing_trace_for_answer_related_feedback_returns_hold_or_invalid():
    payload = _ready_payload()
    payload.pop("bridge_trace_id")

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_missing_evidence_and_missing_evidence_reason_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["evidence_context"] = {
        "evidence_ids": [],
        "evidence_pointers": [],
    }

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_hold_without_hold_reason_returns_hold_or_invalid():
    payload = _hold_payload()
    payload.pop("hold_reason")

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_user_raw_query_stored_true_blocks_readiness():
    payload = _ready_payload()
    payload["feedback_policy"]["user_raw_query_stored"] = True

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_raw_answer_stored_true_blocks_readiness():
    payload = _ready_payload()
    payload["feedback_policy"]["raw_answer_stored"] = True

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_internal_path_allowed_true_blocks_readiness():
    payload = _ready_payload()
    payload["feedback_policy"]["internal_path_allowed"] = True

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_secret_surface_allowed_true_blocks_readiness():
    payload = _ready_payload()
    payload["feedback_policy"]["secret_surface_allowed"] = True

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_paid_standard_raw_text_allowed_true_blocks_readiness():
    payload = _ready_payload()
    payload["feedback_policy"]["paid_standard_raw_text_allowed"] = True

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_feedback_text_policy_not_summary_or_pointer_only_blocks_readiness():
    payload = _ready_payload()
    payload["feedback_policy"]["feedback_text_policy"] = "raw_text_allowed"

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_unsupported_curation_target_returns_invalid():
    payload = _ready_payload()
    payload["curation_target"] = "library_promotion"

    result = validate_feedback_queue_contract(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_automation_may_promote_to_library_true_blocks_readiness():
    payload = _ready_payload()
    payload["feedback_policy"]["automation_may_promote_to_library"] = True

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_human_review_required_false_blocks_readiness():
    payload = _ready_payload()
    payload["feedback_policy"]["human_review_required"] = False

    result = validate_feedback_queue_contract(payload)

    _assert_hold_or_invalid(result)


def test_unsafe_raw_query_marker_is_not_echoed():
    payload = _ready_payload()
    unsafe_value = "raw query should not be retained"
    payload["feedback_surface"]["learner_note"] = unsafe_value

    result = validate_feedback_queue_contract(payload)
    rendered_values = "\n".join(_walk_public_values(result)).lower()

    _assert_hold_or_invalid(result)
    assert unsafe_value not in rendered_values


def test_unsafe_raw_answer_marker_is_not_echoed():
    payload = _ready_payload()
    unsafe_value = "raw answer should not be retained"
    payload["feedback_surface"]["answer_note"] = unsafe_value

    result = validate_feedback_queue_contract(payload)
    rendered_values = "\n".join(_walk_public_values(result)).lower()

    _assert_hold_or_invalid(result)
    assert unsafe_value not in rendered_values


def test_unsafe_internal_path_marker_is_not_echoed():
    payload = _ready_payload()
    unsafe_value = "H:\\restricted\\feedback.txt"
    payload["feedback_surface"]["storage_note"] = unsafe_value

    result = validate_feedback_queue_contract(payload)
    rendered_values = "\n".join(_walk_public_values(result)).lower()

    _assert_hold_or_invalid(result)
    assert unsafe_value.lower() not in rendered_values


def test_unsafe_secret_like_marker_is_not_echoed():
    payload = _ready_payload()
    unsafe_value = "api token value"
    payload["feedback_surface"]["operator_note"] = unsafe_value

    result = validate_feedback_queue_contract(payload)
    rendered_values = "\n".join(_walk_public_values(result)).lower()

    _assert_hold_or_invalid(result)
    assert unsafe_value not in rendered_values


def test_paid_standard_raw_text_marker_is_not_echoed():
    payload = _ready_payload()
    unsafe_value = "paid standard raw excerpt"
    payload["feedback_surface"]["student_copy"] = unsafe_value

    result = validate_feedback_queue_contract(payload)
    rendered_values = "\n".join(_walk_public_values(result)).lower()

    _assert_hold_or_invalid(result)
    assert unsafe_value not in rendered_values


def test_non_claims_include_no_broader_pass_escalation():
    result = validate_feedback_queue_contract(_ready_payload())

    assert result["non_claims"] == [
        "ANSWER_QUALITY_PASS_NOT_INFERRED",
        "BRIDGE_HEALTH_PASS_NOT_INFERRED",
        "SKILLUP_MVP_PASS_NOT_INFERRED",
        "TRACK_A_PASS_NOT_INFERRED",
        "BETA_PASS_NOT_INFERRED",
        "F13_PASS_NOT_INFERRED",
        "RELEASE_READINESS_NOT_INFERRED",
        "DEPLOYMENT_READINESS_NOT_INFERRED",
        "FEEDBACK_QUEUE_PASS_NOT_INFERRED",
        "SELECTED_STATIC_FEEDBACK_QUEUE_READINESS_ONLY",
    ]


def test_no_db_network_runtime_file_env_or_subprocess_flags_are_true():
    ready_result = validate_feedback_queue_contract(_ready_payload())
    hold_payload = _ready_payload()
    hold_payload["feedback_policy"]["raw_answer_stored"] = True
    hold_result = validate_feedback_queue_contract(hold_payload)

    for result in (ready_result, hold_result):
        _assert_no_execution_flags(result)
