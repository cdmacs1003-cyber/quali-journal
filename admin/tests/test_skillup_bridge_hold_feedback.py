from admin.f13_skillup_bridge import (
    ANSWER_STATUS_ANSWERED,
    ANSWER_STATUS_DENIED,
    ANSWER_STATUS_HOLD,
    NOT_GRANTED,
    skillup_answer_from_bridge_response,
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
