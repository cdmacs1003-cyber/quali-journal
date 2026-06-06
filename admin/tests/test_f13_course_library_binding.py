from typing import Any

from admin.f13_course_library_binding import bind_course_library_reference


def _walk(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            out.append(str(key))
            out.extend(_walk(child))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for child in value:
            out.extend(_walk(child))
        return out
    return [str(value)]


def _assert_no_raw_internal_or_secret_surface(result: dict[str, Any]) -> None:
    rendered = "\n".join(_walk(result)).lower()
    forbidden_keys = {
        "raw_text",
        "raw_prompt",
        "raw_query",
        "full_source_text",
        "internal_path",
        "secret",
    }
    assert forbidden_keys.isdisjoint(set(result))
    assert "raw prompt" not in rendered
    assert "raw query" not in rendered
    assert "full source text" not in rendered
    assert "h:\\" not in rendered
    assert "c:\\" not in rendered
    assert "file://" not in rendered
    assert "secret" not in rendered
    assert "token" not in rendered


def _assert_no_pass_escalation(result: dict[str, Any]) -> None:
    assert "f13_pass" not in result
    assert "track_a_pass" not in result
    assert "beta_pass" not in result


def test_course_library_binding_safe_ok_contract():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            "evidence_id": "ev:diagnostic-1",
            "bridge_trace_id": "btrace:diagnostic-1",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )

    assert result["binding_status"] == "BOUND"
    assert result["course_id"] == "course:diagnostic-1"
    assert result["module_id"] == "module:diagnostic-1"
    assert result["evidence_id"] == "ev:diagnostic-1"
    assert result["bridge_trace_id"] == "btrace:diagnostic-1"
    assert result["rights_status"] == "PUBLIC"
    assert result["raw_text_policy"] == "SUMMARY_ONLY"
    assert result["feedback_candidate_required"] is False
    assert result["feedback_queue_item"] is None
    assert result["skillup_use_allowed"] is True
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    assert result["db_access_executed"] is False
    _assert_no_raw_internal_or_secret_surface(result)
    _assert_no_pass_escalation(result)


def test_course_library_binding_missing_evidence_holds_with_feedback():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            "bridge_trace_id": "btrace:diagnostic-1",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )

    queue_item = result["feedback_queue_item"]
    assert result["binding_status"] == "HOLD"
    assert result["feedback_candidate_required"] is True
    assert queue_item["feedback_id"]
    assert queue_item["dedup_key"]
    assert queue_item["feedback_type"] == "EVIDENCE_GAP"
    assert result["skillup_use_allowed"] is False
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    assert queue_item["raw_text_included"] is False
    assert queue_item["internal_path_included"] is False
    _assert_no_raw_internal_or_secret_surface(result)
    _assert_no_pass_escalation(result)


def test_course_library_binding_unknown_rights_blocks_skillup_use():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "lesson_id": "lesson:diagnostic-1",
            "evidence_id": "ev:diagnostic-1",
            "bridge_trace_id": "btrace:diagnostic-1",
            "rights_status": "UNKNOWN",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )

    assert result["binding_status"] in {"DENIED", "HOLD"}
    assert result["binding_status"] == "HOLD"
    assert result["module_id"] == "lesson:diagnostic-1"
    assert result["evidence_id"] == "ev:diagnostic-1"
    assert result["rights_status"] == "UNKNOWN"
    assert result["skillup_use_allowed"] is False
    assert result["feedback_candidate_required"] is True
    assert result["feedback_queue_item"]["feedback_type"] == "RIGHTS_POLICY_REVIEW"
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    _assert_no_raw_internal_or_secret_surface(result)
    _assert_no_pass_escalation(result)


def test_course_library_binding_warehouse_status_blocks_skillup_canonical_use():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            "library_node_id": "lib:diagnostic-1",
            "evidence_id": "ev:diagnostic-1",
            "approval_record_id": "approval:diagnostic-1",
            "bridge_trace_id": "btrace:diagnostic-1",
            "current_status": "APPROVED_FOR_WAREHOUSE",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
            "validation_shape_ids": ["SH-F13-CURATION-001"],
        }
    )

    assert result["binding_status"] in {"DENIED", "HOLD"}
    assert result["skillup_use_allowed"] is False
    assert result["feedback_candidate_required"] is True
    _assert_no_raw_internal_or_secret_surface(result)
    _assert_no_pass_escalation(result)


def test_course_library_binding_dedup_key_stable_for_missing_evidence():
    payload = {
        "course_id": "course:diagnostic-1",
        "module_id": "module:diagnostic-1",
        "bridge_trace_id": "btrace:diagnostic-1",
        "rights_status": "PUBLIC",
        "raw_text_policy": "SUMMARY_ONLY",
    }

    first = bind_course_library_reference(payload)
    second = bind_course_library_reference(dict(payload))

    assert first["feedback_queue_item"]["dedup_key"] == second["feedback_queue_item"]["dedup_key"]
    assert first["feedback_queue_item"]["feedback_id"] == second["feedback_queue_item"]["feedback_id"]
