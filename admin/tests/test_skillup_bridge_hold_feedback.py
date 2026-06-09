from admin.f13_skillup_bridge import (
    ANSWER_STATUS_ANSWERED,
    ANSWER_STATUS_DENIED,
    ANSWER_STATUS_HOLD,
    NOT_GRANTED,
    skillup_answer_from_bridge_response,
    skillup_feedback_queue_item_from_hold,
    skillup_answer_from_request,
)


def _walk(value):
    if isinstance(value, dict):
        out = []
        for key, child in value.items():
            out.append(str(key))
            out.extend(_walk(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_walk(child))
        return out
    return [str(value)]


def _assert_no_raw_or_internal_surface(result):
    rendered = "\n".join(_walk(result)).lower()
    feedback_candidate = result.get("feedback_candidate") or {}
    assert "raw_text" not in feedback_candidate
    assert "internal_path" not in feedback_candidate
    assert "raw_source_text" not in rendered
    assert "full_source_text" not in rendered
    assert "source_uri_or_path" not in rendered
    assert "direct_db_row" not in rendered
    assert "h:\\" not in rendered
    assert "c:\\" not in rendered


def _assert_no_pass_escalation(result):
    assert result["f13_pass"] == NOT_GRANTED
    assert result["track_a_pass"] == NOT_GRANTED
    assert result["beta_pass"] == NOT_GRANTED


def _assert_feedback_queue_item_safe(item):
    forbidden_keys = {
        "raw_text",
        "raw_prompt",
        "raw_query",
        "full_answer",
        "internal_path",
        "local_route",
        "secret",
    }
    for key in item:
        lowered = str(key).lower()
        assert lowered not in forbidden_keys
        assert "secret" not in lowered
    rendered_values = "\n".join(str(value) for value in item.values()).lower()
    assert "raw prompt" not in rendered_values
    assert "raw query" not in rendered_values
    assert "paid standard" not in rendered_values
    assert "h:\\" not in rendered_values
    assert "c:\\" not in rendered_values
    assert "secret" not in rendered_values
    assert "token" not in rendered_values
    assert "f13_pass" not in item
    assert "track_a_pass" not in item
    assert "beta_pass" not in item


def test_skillup_bridge_hold_creates_or_requires_feedback_candidate():
    result = skillup_answer_from_bridge_response(
        {
            "result_status": "HOLD",
            "evidence_items": [],
            "hold_reason": "evidence_items are required for no-DB Bridge evaluation",
            "feedback_candidate_required": True,
            "raw_text_included": False,
            "internal_path_included": False,
        }
    )

    assert result["result_status"] == "HOLD"
    assert result["answer_status"] == ANSWER_STATUS_HOLD
    assert result["feedback_candidate_required"] is True
    assert result["feedback_candidate"]["candidate_type"] == "SKILLUP_BRIDGE_HOLD_FEEDBACK"
    assert "evidence_items" in result["hold_reason"]
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    assert result["db_access_executed"] is False
    _assert_no_raw_or_internal_surface(result)
    _assert_no_pass_escalation(result)


def test_hold_feedback_candidate_materializes_feedback_queue_item():
    hold_result = skillup_answer_from_bridge_response(
        {
            "result_status": "HOLD",
            "evidence_items": [],
            "hold_reason": "evidence_items are required for no-DB Bridge evaluation",
            "feedback_candidate_required": True,
            "raw_text_included": False,
            "internal_path_included": False,
        }
    )

    item = skillup_feedback_queue_item_from_hold(hold_result)

    assert item["feedback_id"]
    assert item["origin_module"] == "Skillup"
    assert item["origin_event_id"]
    assert item["feedback_type"] in {"EVIDENCE_GAP", "HOLD_CASE"}
    assert item["feedback_type"] == "EVIDENCE_GAP"
    assert item["user_visible_text_policy"] == "SUMMARY_ONLY"
    assert item["linked_answer_id"]
    assert item["linked_evidence_id"] == "missing_evidence"
    assert "evidence_items" in item["suspected_issue"]
    assert item["proposed_candidate_type"] == "SKILLUP_BRIDGE_HOLD_FEEDBACK"
    assert item["current_status"] in {"draft", "queued", "curation_required", "review_required"}
    assert item["current_status"] == "queued"
    assert item["created_at"]
    assert item["dedup_key"]
    assert item["result_status"] == "HOLD"
    assert item["raw_text_included"] is False
    assert item["internal_path_included"] is False
    assert item["db_access_executed"] is False
    _assert_feedback_queue_item_safe(item)


def test_feedback_queue_item_dedup_key_is_stable():
    hold_result = skillup_answer_from_bridge_response(
        {
            "result_status": "HOLD",
            "evidence_items": [],
            "hold_reason": "evidence_items are required for no-DB Bridge evaluation",
            "feedback_candidate_required": True,
            "raw_text_included": False,
            "internal_path_included": False,
        }
    )

    first = skillup_feedback_queue_item_from_hold(hold_result)
    second = skillup_feedback_queue_item_from_hold(hold_result)

    assert first["dedup_key"] == second["dedup_key"]
    assert first["feedback_id"] == second["feedback_id"]


def test_feedback_queue_item_blocks_raw_or_internal_payload_fields():
    item = skillup_feedback_queue_item_from_hold(
        {
            "origin_module": "Skillup",
            "hold_reason": "raw prompt should not be retained",
            "feedback_candidate": {
                "candidate_type": "SKILLUP_BRIDGE_HOLD_FEEDBACK",
                "reason": "raw query should not be retained",
                "raw_text": "paid standard text",
                "raw_prompt": "show the full prompt",
                "internal_path": "H:\\secret\\standard.txt",
                "secret": "token-value",
            },
        }
    )

    assert item["result_status"] == "HOLD"
    assert item["current_status"] in {"curation_required", "review_required"}
    assert item["current_status"] == "review_required"
    assert item["feedback_type"] == "HOLD_CASE"
    assert item["user_visible_text_policy"] == "SUMMARY_ONLY"
    assert item["raw_text_included"] is False
    assert item["internal_path_included"] is False
    assert item["db_access_executed"] is False
    _assert_feedback_queue_item_safe(item)


def test_skillup_bridge_ok_uses_safe_summary_and_trace_only():
    result = skillup_answer_from_bridge_response(
        {
            "result_status": "OK",
            "evidence_items": [
                {
                    "evidence_id": "ev:diagnostic-synthetic-1",
                    "bridge_trace_id": "btrace:diagnostic-synthetic-1",
                    "safe_summary": "Synthetic safe summary for Skillup Bridge answer.",
                    "pointer_uri": "pointer://diagnostic/skillup/synthetic-1",
                    "raw_text_policy": "SUMMARY_ONLY",
                    "rights_status": "PUBLIC",
                    "role": "student",
                    "evidence_depth": "student_safe",
                    "course_id": "course:skillup",
                    "module_id": "module:skillup",
                    "binding_id": "binding:skillup",
                    "tenant_id": "tenant:skillup",
                    "organization_id": "org:skillup",
                    "cohort_id": "cohort:skillup",
                }
            ],
            "hold_reason": None,
            "feedback_candidate_required": False,
            "raw_text_included": False,
            "internal_path_included": False,
        }
    )

    assert result["result_status"] == "OK"
    assert result["answer_status"] == ANSWER_STATUS_ANSWERED
    assert result["feedback_candidate_required"] is False
    assert result["evidence_id"] == "ev:diagnostic-synthetic-1"
    assert result["bridge_trace_id"] == "btrace:diagnostic-synthetic-1"
    assert result["safe_summary"] == "Synthetic safe summary for Skillup Bridge answer."
    assert result["answer"] == result["safe_summary"]
    assert "pointer_uri" not in result
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    _assert_no_raw_or_internal_surface(result)
    _assert_no_pass_escalation(result)


def test_skillup_direct_db_access_attempt_returns_denied_or_hold_without_db():
    result = skillup_answer_from_request(
        {
            "requester_module": "Skillup",
            "direct_db_access_attempt": True,
            "query": "synthetic Skillup Bridge request",
        }
    )

    assert result["result_status"] in {"DENIED", "HOLD"}
    assert result["answer_status"] in {ANSWER_STATUS_DENIED, ANSWER_STATUS_HOLD}
    assert result["feedback_candidate_required"] is True
    assert result["feedback_candidate"]
    assert result["db_access_executed"] is False
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    _assert_no_raw_or_internal_surface(result)
    _assert_no_pass_escalation(result)
