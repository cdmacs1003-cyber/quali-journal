import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_bridge_explain_trace_response.schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_explain_trace_response_schema_status_and_required_fields():
    schema = _schema()
    properties = schema["properties"]
    required = set(schema["required"])

    assert schema["additionalProperties"] is False
    assert set(properties["result_status"]["enum"]) == {"OK", "HOLD", "DENIED"}
    assert {
        "result_status",
        "request_id",
        "bridge_trace_id",
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
