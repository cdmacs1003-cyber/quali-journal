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
