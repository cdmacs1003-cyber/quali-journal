import inspect
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

import admin.f13_analytics_event_contract as analytics_contract
from admin.f13_analytics_event_contract import (
    ANALYTICS_EVENT_TYPES,
    CONTRACT_VERSION,
    FEEDBACK_TO_ANALYTICS_EVENT_TYPES,
    FEEDBACK_TO_ANALYTICS_SUMMARIES,
    PROHIBITED_FIELD_NAMES,
    QUERY_SUMMARY_DISALLOWED_MARKERS,
    QUERY_SUMMARY_MAX_LENGTH,
    REQUIRED_FIELDS,
    RISK_FLAG_VOCABULARY,
    SCHEMA_VERSION,
    map_feedback_to_analytics_event,
    validate_analytics_event,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_analytics_event.schema.json"
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _event(event_type="answer_rendered"):
    return {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "event_id": "EVT-R451-TEST-1",
        "tenant_id": "TEN-R451-SYNTHETIC",
        "organization_id": "ORGUNIT-R451-SYNTHETIC",
        "cohort_id": "COH-R451-SYNTHETIC",
        "user_id_hash": HASH_B,
        "event_type": event_type,
        "request_id": "REQ-R451-TEST-1",
        "trace_id": "btrace:R451:TEST:1" if event_type != "question_asked" else None,
        "query_hash": HASH_A,
        "query_summary": "answer_event" if event_type == "answer_rendered" else "hold_event",
        "raw_query_stored": False,
        "risk_flags": [],
        "occurred_at": "2026-07-12T00:00:00Z",
    }


def _feedback(event_type="answer_rendered"):
    is_hold = event_type == "hold_created"
    payload = {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "feedback_id": "FBQ-R451-TEST-1",
        "request_id": "REQ-R451-TEST-1",
        "query_hash": HASH_A,
        "tenant_context": {
            "tenant_id": "TEN-R451-SYNTHETIC",
            "organization_id": "ORGUNIT-R451-SYNTHETIC",
            "cohort_id": "COH-R451-SYNTHETIC",
        },
        "course_context": {"course_id": "CRS-R451-SYNTHETIC", "module_id": "MOD-R451-SYNTHETIC"},
        "event_context": {"event_type": event_type},
        "answer_status": "HOLD" if is_hold else "ANSWERED",
        "bridge_trace_id": "btrace:R451:TEST:1",
        "evidence_context": (
            {"evidence_ids": [], "evidence_pointers": [], "missing_evidence_reason": "EVIDENCE_NOT_AVAILABLE"}
            if is_hold
            else {"evidence_ids": ["EVD-R451-TEST-1"], "evidence_pointers": ["urn:qlib:evidence:r451-test-1"]}
        ),
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
        "curation_target": "evidence_gap_queue" if is_hold else "qa_case_candidate",
        "feedback_surface": {"safe_summary": "R451_CONTROLLED_FEEDBACK_METADATA"},
    }
    if is_hold:
        payload["hold_reason"] = "HOLD_REVIEW_REQUIRED"
    return payload


def _schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _assert_no_execution(result):
    assert result["db_access_executed"] is False
    assert result["durable_write_executed"] is False
    assert result["file_io_executed"] is False
    assert result["network_access_executed"] is False
    assert result["environment_access_executed"] is False
    assert result["subprocess_executed"] is False


@pytest.mark.parametrize("event_type", ["answer_rendered", "hold_created"])
def test_valid_answer_and_hold_events_pass_python_and_json_schema(event_type):
    payload = _event(event_type)
    assert validate_analytics_event(payload)["valid"] is True
    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(payload)


@pytest.mark.parametrize(
    ("feedback_type", "summary"),
    [("answer_rendered", "answer_event"), ("hold_created", "hold_event")],
)
def test_valid_feedback_mapping_preserves_context_trace_and_hash(feedback_type, summary):
    source = _feedback(feedback_type)
    result = map_feedback_to_analytics_event(
        source,
        user_id_hash=HASH_B,
        event_id=f"EVT-R451-{feedback_type}",
        occurred_at="2026-07-12T00:00:00Z",
    )

    assert result["mapped"] is True
    event = result["event"]
    assert event["event_type"] == feedback_type
    assert event["tenant_id"] == source["tenant_context"]["tenant_id"]
    assert event["organization_id"] == source["tenant_context"]["organization_id"]
    assert event["cohort_id"] == source["tenant_context"]["cohort_id"]
    assert event["request_id"] == source["request_id"]
    assert event["trace_id"] == source["bridge_trace_id"]
    assert event["query_hash"] == source["query_hash"]
    assert event["query_summary"] == summary
    assert event["raw_query_stored"] is False
    assert set(event) == set(REQUIRED_FIELDS)
    _assert_no_execution(result)


def test_schema_and_python_validator_constants_are_aligned():
    schema = _schema()
    properties = schema["properties"]

    assert set(schema["required"]) == set(REQUIRED_FIELDS)
    assert schema["additionalProperties"] is False
    assert properties["schema_version"]["const"] == SCHEMA_VERSION
    assert properties["contract_version"]["const"] == CONTRACT_VERSION
    assert set(properties["event_type"]["enum"]) == set(ANALYTICS_EVENT_TYPES)
    assert properties["query_summary"]["maxLength"] == QUERY_SUMMARY_MAX_LENGTH
    assert tuple(properties["query_summary"]["x-qlib-disallowed-markers"]) == QUERY_SUMMARY_DISALLOWED_MARKERS
    assert properties["raw_query_stored"]["const"] is False
    assert set(properties["risk_flags"]["items"]["enum"]) == set(RISK_FLAG_VOCABULARY)
    Draft202012Validator.check_schema(schema)


def test_mapper_and_validator_are_pure_no_persistence_surfaces():
    source = inspect.getsource(analytics_contract)
    forbidden_imports = ("import sqlite", "import socket", "import requests", "import httpx", "import subprocess")

    assert not any(item in source for item in forbidden_imports)
    _assert_no_execution(validate_analytics_event(_event()))
    _assert_no_execution(map_feedback_to_analytics_event(_feedback(), user_id_hash=HASH_B))


def test_uncontrolled_analytics_event_type_fails_closed():
    payload = _event()
    payload["event_type"] = "uncontrolled_event"
    result = validate_analytics_event(payload)
    assert result["valid"] is False
    assert "UNCONTROLLED_EVENT_TYPE" in result["reason_codes"]


def test_unmapped_feedback_event_type_fails_closed():
    result = map_feedback_to_analytics_event(_feedback("feedback_submitted"), user_id_hash=HASH_B)
    assert result["mapped"] is False
    assert result["reason_codes"] == ["ANALYTICS_EVENT_NOT_MAPPED"]


def test_raw_query_stored_true_is_rejected():
    payload = _event()
    payload["raw_query_stored"] = True
    result = validate_analytics_event(payload)
    assert result["valid"] is False
    assert "RAW_QUERY_STORAGE_FORBIDDEN" in result["reason_codes"]


@pytest.mark.parametrize(
    "field_name",
    [
        "raw_query", "raw_body", "prompt", "raw_prompt", "evidence_text", "standard_text",
        "api_key", "access_token", "internal_path", "local_path", "file_path", "connection_string",
    ],
)
def test_prohibited_fields_are_rejected_without_value_echo(field_name, capsys, caplog):
    payload = _event()
    sentinel = "UNRETAINED_SENTINEL_R451"
    payload[field_name] = sentinel
    result = validate_analytics_event(payload)
    public_result = json.dumps(result, sort_keys=True)

    assert result["valid"] is False
    assert result["reason_codes"] == ["PROHIBITED_FIELD"]
    assert field_name in result["prohibited_fields"]
    assert sentinel not in public_result
    assert sentinel not in capsys.readouterr().out
    assert sentinel not in caplog.text


@pytest.mark.parametrize(
    ("field_name", "normalized"),
    [("rawQuery", "raw_query"), ("apiKey", "api_key"), ("filePath", "file_path")],
)
def test_normalized_prohibited_field_equivalents_fail_closed(field_name, normalized):
    payload = _event()
    payload[field_name] = "UNRETAINED_SENTINEL_R451"
    result = validate_analytics_event(payload)

    assert result["valid"] is False
    assert result["reason_codes"] == ["PROHIBITED_FIELD"]
    assert result["prohibited_fields"] == [normalized]


@pytest.mark.parametrize("field", ["tenant_id", "organization_id", "cohort_id", "request_id"])
def test_missing_required_context_is_rejected(field):
    payload = _event()
    payload.pop(field)
    result = validate_analytics_event(payload)
    assert result["valid"] is False
    assert "MISSING_REQUIRED_FIELD" in result["reason_codes"]
    assert field in result["invalid_fields"]


def test_invalid_query_hash_is_rejected():
    payload = _event()
    payload["query_hash"] = "sha256:invalid"
    result = validate_analytics_event(payload)
    assert result["valid"] is False
    assert "INVALID_QUERY_HASH" in result["reason_codes"]


def test_mapper_rejects_missing_cohort_context():
    payload = _feedback()
    payload["tenant_context"].pop("cohort_id")
    result = map_feedback_to_analytics_event(payload, user_id_hash=HASH_B)

    assert result["mapped"] is False
    assert result["reason_codes"] == ["MISSING_OR_INVALID_COHORT_ID"]


def test_trace_required_event_rejects_missing_trace():
    payload = _event()
    payload["trace_id"] = None
    result = validate_analytics_event(payload)

    assert result["valid"] is False
    assert "TRACE_ID_REQUIRED" in result["reason_codes"]


@pytest.mark.parametrize("risk_flags", [["uncontrolled_flag"], ["policy_hold", "policy_hold"], [{}]])
def test_risk_flags_fail_closed_when_uncontrolled_duplicate_or_wrong_type(risk_flags):
    payload = _event()
    payload["risk_flags"] = risk_flags
    result = validate_analytics_event(payload)

    assert result["valid"] is False
    assert "INVALID_RISK_FLAGS" in result["reason_codes"]


@pytest.mark.parametrize(
    "summary",
    [
        "x" * 65,
        "line_one\nline_two",
        "https://invalid.example",
        "C:\\internal\\location",
        "/internal/location",
        "credential_marker",
        "standard_text_marker",
    ],
)
def test_unsafe_query_summary_is_rejected(summary):
    payload = _event()
    payload["query_summary"] = summary
    result = validate_analytics_event(payload)
    assert result["valid"] is False
    assert "UNSAFE_QUERY_SUMMARY" in result["reason_codes"]


def test_query_summary_may_be_null_for_general_event():
    payload = _event("question_asked")
    payload["query_summary"] = None
    payload["query_hash"] = None
    assert validate_analytics_event(payload)["valid"] is True


def test_invalid_user_id_hash_is_rejected():
    payload = _event()
    payload["user_id_hash"] = "USH-not-a-one-way-hash"
    result = validate_analytics_event(payload)
    assert result["valid"] is False
    assert "INVALID_USER_ID_HASH" in result["reason_codes"]


def test_unexpected_extra_property_is_rejected():
    payload = _event()
    payload["safe_extra"] = True
    result = validate_analytics_event(payload)
    assert result["valid"] is False
    assert "UNEXPECTED_PROPERTY" in result["reason_codes"]


def test_feedback_mapping_contract_is_exactly_two_events():
    assert FEEDBACK_TO_ANALYTICS_EVENT_TYPES == {
        "answer_rendered": "answer_rendered",
        "hold_created": "hold_created",
    }
    assert FEEDBACK_TO_ANALYTICS_SUMMARIES == {
        "answer_rendered": "answer_event",
        "hold_created": "hold_event",
    }


def test_prohibited_field_policy_contains_mandatory_names():
    assert {
        "raw_query", "query", "raw_body", "body", "prompt", "raw_prompt", "raw_text",
        "evidence_text", "standard_text", "paid_standard_text", "secret", "api_key",
        "access_token", "refresh_token", "password", "credential", "cookie", "authorization",
        "internal_path", "local_path", "file_path", "db_path", "dsn", "connection_string",
    } <= set(PROHIBITED_FIELD_NAMES)
