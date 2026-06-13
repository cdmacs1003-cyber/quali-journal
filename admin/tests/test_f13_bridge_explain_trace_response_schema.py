import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_bridge_explain_trace_response.schema.json"

FORBIDDEN_RESPONSE_FIELDS = {
    "raw_text_ref",
    "raw_pointer",
    "raw_source_text",
    "full_source_text",
    "source_uri_or_path",
    "direct_db_row",
    "warehouse_internal_object",
    "library_internal_object",
    "internal_path",
    "local_path",
    "secret",
    "token",
    "api_key",
    "dsn",
    "database_url",
}


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _safe_ok_trace_response(**overrides: Any) -> dict[str, Any]:
    response = {
        "result_status": "OK",
        "request_id": "req:trace-safe-1",
        "bridge_trace_id": "btrace:trace-safe-1",
        "course_id": "CRS-track-a-safe",
        "module_id": "MOD-bridge-runtime-mvp-v1",
        "binding_id": "BND-bridge-runtime-mvp-v1",
        "evidence_ids": ["ev:trace-safe-1"],
        "policy_result": "PASS",
        "hold_reason": None,
        "role": "student",
        "evidence_depth": "student_safe",
        "review_trace": None,
        "audit_trace": None,
        "raw_text_export_count": 0,
        "internal_path_leak_count": 0,
        "raw_prompt_output_count": 0,
        "secret_leak_count": 0,
        "instructor_guide_raw_leak_count": 0,
        "feedback_candidate_required": False,
        "feedback_candidate": None,
        "visible_trace_summary": (
            "Trace btrace:trace-safe-1 is visible as student_safe metadata only. "
            "Evidence count: 1. Raw text included: false. Internal path included: false."
        ),
        "raw_text_included": False,
        "internal_path_included": False,
        "created_at": "2026-06-12T00:00:00Z",
    }
    response.update(overrides)
    return response


def _hold_trace_response() -> dict[str, Any]:
    return _safe_ok_trace_response(
        result_status="HOLD",
        bridge_trace_id=None,
        evidence_ids=[],
        policy_result="HOLD",
        hold_reason="bridge_trace_id is required for no-DB trace explanation",
        role=None,
        evidence_depth=None,
        feedback_candidate_required=True,
        feedback_candidate={
            "candidate_type": "BRIDGE_TRACE_REVIEW",
            "reason": "bridge_trace_id is required for no-DB trace explanation",
            "next_action": "REVIEW_EVIDENCE_TRACE_POLICY",
        },
        visible_trace_summary="Trace explanation is on HOLD because bridge_trace_id is missing.",
    )


def _assert_payload_matches_schema_contract(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    properties = schema["properties"]
    required = set(schema["required"])

    assert set(payload) == required
    assert set(payload).issubset(properties)
    assert payload["result_status"] in properties["result_status"]["enum"]
    assert payload["policy_result"] in properties["policy_result"]["enum"]
    assert payload["role"] in properties["role"]["enum"]
    assert payload["evidence_depth"] in properties["evidence_depth"]["enum"]
    assert isinstance(payload["evidence_ids"], list)
    assert isinstance(payload["feedback_candidate_required"], bool)
    assert payload["raw_text_included"] is properties["raw_text_included"]["const"]
    assert payload["internal_path_included"] is properties["internal_path_included"]["const"]

    for counter in (
        "raw_text_export_count",
        "internal_path_leak_count",
        "raw_prompt_output_count",
        "secret_leak_count",
        "instructor_guide_raw_leak_count",
    ):
        assert payload[counter] == properties[counter]["const"]


def test_explain_trace_response_schema_status_and_required_fields():
    schema = _schema()
    properties = schema["properties"]
    required = set(schema["required"])
    unexpected_root_payload = _safe_ok_trace_response(unexpected_root_field="not allowed")

    assert schema.get("type") == "object"
    assert schema["additionalProperties"] is False
    assert "unexpected_root_field" not in properties
    assert set(unexpected_root_payload) != required
    assert not set(unexpected_root_payload).issubset(properties)
    assert set(properties["result_status"]["enum"]) == {"OK", "HOLD", "DENIED"}
    assert {
        "result_status",
        "request_id",
        "bridge_trace_id",
        "course_id",
        "module_id",
        "binding_id",
        "evidence_ids",
        "policy_result",
        "hold_reason",
        "role",
        "evidence_depth",
        "review_trace",
        "audit_trace",
        "raw_text_export_count",
        "internal_path_leak_count",
        "raw_prompt_output_count",
        "secret_leak_count",
        "instructor_guide_raw_leak_count",
        "feedback_candidate_required",
        "feedback_candidate",
        "visible_trace_summary",
        "raw_text_included",
        "internal_path_included",
        "created_at",
    }.issubset(required)


def test_explain_trace_representative_ok_response_matches_schema_contract():
    schema = _schema()
    payload = _safe_ok_trace_response()

    _assert_payload_matches_schema_contract(payload, schema)
    assert payload["bridge_trace_id"].startswith("btrace:")
    assert payload["evidence_ids"] == ["ev:trace-safe-1"]
    assert payload["feedback_candidate_required"] is False
    assert payload["feedback_candidate"] is None
    assert FORBIDDEN_RESPONSE_FIELDS.isdisjoint(payload)


def test_explain_trace_representative_hold_response_matches_feedback_candidate_contract():
    schema = _schema()
    payload = _hold_trace_response()
    feedback_schema = schema["properties"]["feedback_candidate"]
    candidate = payload["feedback_candidate"]

    _assert_payload_matches_schema_contract(payload, schema)
    assert payload["result_status"] == "HOLD"
    assert payload["feedback_candidate_required"] is True
    assert isinstance(candidate, dict)
    assert set(feedback_schema["required"]).issubset(candidate)
    assert set(candidate).issubset(feedback_schema["properties"])
    assert candidate["candidate_type"] == "BRIDGE_TRACE_REVIEW"
    assert candidate["next_action"] == "REVIEW_EVIDENCE_TRACE_POLICY"
    assert FORBIDDEN_RESPONSE_FIELDS.isdisjoint(payload)


def test_explain_trace_feedback_candidate_top_level_shape():
    properties = _schema()["properties"]
    feedback_candidate = properties["feedback_candidate"]
    candidate_properties = feedback_candidate["properties"]

    assert properties["feedback_candidate_required"]["type"] == "boolean"
    assert set(feedback_candidate["type"]) == {"object", "null"}
    assert feedback_candidate["additionalProperties"] is False
    assert set(feedback_candidate["required"]) == {
        "candidate_type",
        "reason",
        "next_action",
    }
    assert candidate_properties["candidate_type"]["enum"] == ["BRIDGE_TRACE_REVIEW"]
    assert candidate_properties["next_action"]["enum"] == ["REVIEW_EVIDENCE_TRACE_POLICY"]


def test_explain_trace_response_schema_blocks_raw_and_internal_flags():
    properties = _schema()["properties"]

    assert properties["raw_text_included"]["const"] is False
    assert properties["internal_path_included"]["const"] is False
    assert set(properties["role"]["enum"]) == {"student", "instructor", "reviewer", "admin", None}
    assert set(properties["evidence_depth"]["enum"]) == {
        "student_safe",
        "instructor_safe",
        "review_trace_safe_metadata",
        "audit_trace_safe_metadata",
        None,
    }
    for counter in (
        "raw_text_export_count",
        "internal_path_leak_count",
        "raw_prompt_output_count",
        "secret_leak_count",
        "instructor_guide_raw_leak_count",
    ):
        assert properties[counter]["const"] == 0


def test_explain_trace_schema_safe_review_and_audit_metadata_shapes():
    properties = _schema()["properties"]
    review_trace = properties["review_trace"]
    audit_trace = properties["audit_trace"]

    assert set(review_trace["type"]) == {"object", "null"}
    assert review_trace["additionalProperties"] is False
    assert set(review_trace["required"]) == {
        "visibility",
        "evidence_match_status",
        "hold_queue_status",
        "policy_block_summary",
    }
    assert review_trace["properties"]["visibility"]["enum"] == ["review_trace_safe_metadata"]
    assert set(audit_trace["type"]) == {"object", "null"}
    assert audit_trace["additionalProperties"] is False
    assert audit_trace["properties"]["raw_export_allowed"]["const"] is False

