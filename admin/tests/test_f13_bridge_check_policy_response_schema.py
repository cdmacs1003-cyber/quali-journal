import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_bridge_check_policy_response.schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_check_policy_response_schema_status_and_required_fields():
    schema = _schema()
    properties = schema["properties"]
    required = set(schema["required"])

    assert schema["additionalProperties"] is False
    assert set(properties["result_status"]["enum"]) == {"OK", "HOLD", "DENIED"}
    assert {
        "result_status",
        "bridge_trace_id",
        "policy_result",
        "hold_reason",
        "output_constraints",
        "blocked_fields",
        "feedback_candidate_required",
        "raw_text_included",
        "internal_path_included",
        "created_at",
    }.issubset(required)


def test_check_policy_response_schema_blocks_raw_and_internal_flags():
    properties = _schema()["properties"]

    assert properties["raw_text_included"]["const"] is False
    assert properties["internal_path_included"]["const"] is False
    assert properties["feedback_candidate_required"]["type"] == "boolean"
    assert properties["blocked_fields"]["type"] == "array"
    assert properties["output_constraints"]["type"] == "array"
