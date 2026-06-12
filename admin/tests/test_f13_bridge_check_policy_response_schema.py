import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_bridge_check_policy_response.schema.json"

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


def _safe_ok_response(**overrides: Any) -> dict[str, Any]:
    response = {
        "result_status": "OK",
        "bridge_trace_id": "btrace:check-policy-safe-1",
        "policy_result": "PASS",
        "hold_reason": None,
        "output_constraints": [
            "SAFE_SUMMARY_ONLY",
            "NO_RAW_EXPORT",
            "NO_RAW_TEXT",
            "NO_INTERNAL_PATH",
            "BRIDGE_TRACE_REQUIRED",
            "ZERO_ROLE_LEAK_COUNTERS",
            "ROLE_STUDENT",
            "EVIDENCE_DEPTH_STUDENT_SAFE",
        ],
        "blocked_fields": [],
        "role": "student",
        "evidence_depth": "student_safe",
        "raw_text_export_count": 0,
        "internal_path_leak_count": 0,
        "raw_prompt_output_count": 0,
        "secret_leak_count": 0,
        "instructor_guide_raw_leak_count": 0,
        "feedback_candidate_required": False,
        "raw_text_included": False,
        "internal_path_included": False,
        "created_at": "2026-06-12T00:00:00Z",
    }
    response.update(overrides)
    return response


def _hold_response() -> dict[str, Any]:
    return _safe_ok_response(
        result_status="HOLD",
        bridge_trace_id=None,
        policy_result="HOLD",
        hold_reason="HOLD_PERMISSION: explicit supported role is required for Track A protected answer flow",
        output_constraints=[
            "HOLD_UNTIL_EVIDENCE_TRACE_RIGHTS_POLICY_PASS",
            "NO_RAW_EXPORT",
            "NO_RAW_TEXT",
            "NO_INTERNAL_PATH",
        ],
        role=None,
        evidence_depth=None,
        feedback_candidate_required=True,
    )


def _denied_response() -> dict[str, Any]:
    return _safe_ok_response(
        result_status="DENIED",
        bridge_trace_id=None,
        policy_result="DENIED",
        hold_reason="forbidden fields or patterns detected",
        output_constraints=["BLOCK_OUTPUT", "NO_RAW_EXPORT", "NO_RAW_TEXT", "NO_INTERNAL_PATH"],
        blocked_fields=["raw_text_ref"],
        role=None,
        evidence_depth=None,
        feedback_candidate_required=True,
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
    assert isinstance(payload["output_constraints"], list)
    assert isinstance(payload["blocked_fields"], list)
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
        "role",
        "evidence_depth",
        "raw_text_export_count",
        "internal_path_leak_count",
        "raw_prompt_output_count",
        "secret_leak_count",
        "instructor_guide_raw_leak_count",
        "feedback_candidate_required",
        "raw_text_included",
        "internal_path_included",
        "created_at",
    }.issubset(required)


def test_check_policy_representative_ok_response_matches_schema_contract():
    schema = _schema()
    payload = _safe_ok_response()

    _assert_payload_matches_schema_contract(payload, schema)
    assert payload["bridge_trace_id"].startswith("btrace:")
    assert payload["hold_reason"] is None
    assert FORBIDDEN_RESPONSE_FIELDS.isdisjoint(payload)


def test_check_policy_representative_hold_and_denied_responses_match_schema_contract():
    schema = _schema()

    for payload in (_hold_response(), _denied_response()):
        _assert_payload_matches_schema_contract(payload, schema)
        assert payload["result_status"] in {"HOLD", "DENIED"}
        assert payload["hold_reason"]
        assert payload["feedback_candidate_required"] is True
        assert FORBIDDEN_RESPONSE_FIELDS.isdisjoint(payload)


def test_check_policy_response_schema_blocks_raw_and_internal_flags():
    properties = _schema()["properties"]

    assert properties["raw_text_included"]["const"] is False
    assert properties["internal_path_included"]["const"] is False
    assert properties["feedback_candidate_required"]["type"] == "boolean"
    assert properties["blocked_fields"]["type"] == "array"
    assert properties["output_constraints"]["type"] == "array"
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

