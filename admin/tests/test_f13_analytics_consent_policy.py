import copy
import inspect
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

import admin.f13_analytics_consent_policy as consent_policy
from admin.f13_analytics_consent_policy import (
    ANALYTICS_EXCLUDED,
    CONSENT_ALLOWED,
    CONSENT_RECORD_INVALID,
    CONSENT_REQUIRED,
    CONSENT_REVOKED,
    CONSENT_SCOPE_DENIED,
    CONSENT_SCOPE_FIELDS,
    CONSENT_USER_MISMATCH,
    PROHIBITED_CONSENT_FIELDS,
    REQUIRED_CONSENT_FIELDS,
    evaluate_analytics_consent_policy,
    map_feedback_to_analytics_with_consent,
    validate_analytics_consent_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_analytics_consent_record.schema.json"
USER_HASH = "sha256:" + "a" * 64
OTHER_USER_HASH = "sha256:" + "b" * 64
QUERY_HASH = "sha256:" + "c" * 64


def _consent(**overrides):
    record = {
        "consent_id": "CST-R452-SYNTHETIC-1",
        "user_id_hash": USER_HASH,
        "consent_scope": {
            "learning_analytics": True,
            "marketing_aggregate": False,
            "personalized_feedback": False,
        },
        "consent_version": "v1",
        "granted_at": "2026-07-12T00:00:00Z",
        "revoked_at": None,
        "retention_policy_id": "RET-R452-v1",
    }
    record.update(overrides)
    return record


def _feedback(event_type="answer_rendered"):
    is_hold = event_type == "hold_created"
    payload = {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "feedback_id": "FBQ-R452-SYNTHETIC-1",
        "request_id": "REQ-R452-SYNTHETIC-1",
        "query_hash": QUERY_HASH,
        "tenant_context": {
            "tenant_id": "TEN-R452-SYNTHETIC",
            "organization_id": "ORGUNIT-R452-SYNTHETIC",
            "cohort_id": "COH-R452-SYNTHETIC",
        },
        "course_context": {"course_id": "CRS-R452-SYNTHETIC", "module_id": "MOD-R452-SYNTHETIC"},
        "event_context": {"event_type": event_type},
        "answer_status": "HOLD" if is_hold else "ANSWERED",
        "bridge_trace_id": "btrace:R452:SYNTHETIC:1",
        "evidence_context": (
            {"evidence_ids": [], "evidence_pointers": [], "missing_evidence_reason": "EVIDENCE_NOT_AVAILABLE"}
            if is_hold
            else {"evidence_ids": ["EVD-R452-SYNTHETIC-1"], "evidence_pointers": ["urn:qlib:evidence:r452-1"]}
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
        "feedback_surface": {"safe_summary": "R452_CONTROLLED_FEEDBACK_METADATA"},
    }
    if is_hold:
        payload["hold_reason"] = "HOLD_REVIEW_REQUIRED"
    return payload


def _schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_valid_consent_record_passes_python_and_json_schema():
    record = _consent()
    assert validate_analytics_consent_record(record)["valid"] is True
    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(record)


def test_schema_self_check_and_python_constants_align():
    schema = _schema()
    properties = schema["properties"]

    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(REQUIRED_CONSENT_FIELDS)
    assert set(properties["consent_scope"]["required"]) == set(CONSENT_SCOPE_FIELDS)
    assert properties["consent_scope"]["additionalProperties"] is False
    assert all(properties["consent_scope"]["properties"][field]["type"] == "boolean" for field in CONSENT_SCOPE_FIELDS)


@pytest.mark.parametrize(
    "scopes",
    [
        {"learning_analytics": True, "marketing_aggregate": False, "personalized_feedback": False},
        {"learning_analytics": True, "marketing_aggregate": True, "personalized_feedback": False},
        {"learning_analytics": True, "marketing_aggregate": False, "personalized_feedback": True},
    ],
)
def test_learning_consent_is_independent_of_other_scopes(scopes):
    decision = evaluate_analytics_consent_policy(
        _consent(consent_scope=scopes), target_user_id_hash=USER_HASH, analytics_exclusion=False
    )
    assert decision["allowed"] is True
    assert decision["reason_code"] == CONSENT_ALLOWED


def test_null_revoked_at_is_accepted():
    assert validate_analytics_consent_record(_consent(revoked_at=None))["valid"] is True


def test_valid_revoked_chronology_is_parsed_then_policy_denied():
    record = _consent(revoked_at="2026-07-13T00:00:00Z")
    assert validate_analytics_consent_record(record)["valid"] is True
    assert evaluate_analytics_consent_policy(
        record, target_user_id_hash=USER_HASH, analytics_exclusion=False
    )["reason_code"] == CONSENT_REVOKED


def test_valid_consent_without_exclusion_is_allowed_and_deterministic():
    first = evaluate_analytics_consent_policy(_consent(), target_user_id_hash=USER_HASH, analytics_exclusion=False)
    second = evaluate_analytics_consent_policy(_consent(), target_user_id_hash=USER_HASH, analytics_exclusion=False)
    assert first == second == {
        "allowed": True,
        "status": "ALLOW",
        "reason_code": CONSENT_ALLOWED,
        "consent_id": "CST-R452-SYNTHETIC-1",
    }


@pytest.mark.parametrize("event_type", ["answer_rendered", "hold_created"])
def test_answer_and_hold_feedback_map_only_after_consent_allow(event_type):
    result = map_feedback_to_analytics_with_consent(
        _feedback(event_type),
        consent_payload=_consent(),
        user_id_hash=USER_HASH,
        event_id=f"EVT-R452-{event_type}",
        occurred_at="2026-07-12T00:00:00Z",
    )
    event = result["analytics_event"]

    assert result["policy_status"] == "ALLOW"
    assert result["reason_code"] == CONSENT_ALLOWED
    assert result["analytics_event_present"] is True
    assert event["event_type"] == event_type
    assert event["tenant_id"] == "TEN-R452-SYNTHETIC"
    assert event["organization_id"] == "ORGUNIT-R452-SYNTHETIC"
    assert event["cohort_id"] == "COH-R452-SYNTHETIC"
    assert event["request_id"] == "REQ-R452-SYNTHETIC-1"
    assert event["trace_id"] == "btrace:R452:SYNTHETIC:1"
    assert event["query_hash"] == QUERY_HASH
    assert event["raw_query_stored"] is False


def test_missing_consent_is_required_and_event_is_absent():
    decision = evaluate_analytics_consent_policy(None, target_user_id_hash=USER_HASH, analytics_exclusion=False)
    result = map_feedback_to_analytics_with_consent(
        _feedback(), consent_payload=None, user_id_hash=USER_HASH
    )
    assert decision["reason_code"] == CONSENT_REQUIRED
    assert result == {
        "policy_status": "EXCLUDE",
        "reason_code": CONSENT_REQUIRED,
        "analytics_event_present": False,
        "analytics_event": None,
    }


def test_invalid_consent_fails_closed():
    decision = evaluate_analytics_consent_policy([], target_user_id_hash=USER_HASH, analytics_exclusion=False)
    assert decision["allowed"] is False
    assert decision["reason_code"] == CONSENT_RECORD_INVALID


@pytest.mark.parametrize(
    "scopes",
    [
        {"learning_analytics": False, "marketing_aggregate": False, "personalized_feedback": False},
        {"learning_analytics": False, "marketing_aggregate": True, "personalized_feedback": False},
        {"learning_analytics": False, "marketing_aggregate": False, "personalized_feedback": True},
    ],
)
def test_other_scopes_do_not_substitute_for_learning_analytics(scopes):
    decision = evaluate_analytics_consent_policy(
        _consent(consent_scope=scopes), target_user_id_hash=USER_HASH, analytics_exclusion=False
    )
    assert decision["allowed"] is False
    assert decision["reason_code"] == CONSENT_SCOPE_DENIED


def test_revoked_consent_excludes_event():
    result = map_feedback_to_analytics_with_consent(
        _feedback("hold_created"),
        consent_payload=_consent(revoked_at="2026-07-13T00:00:00Z"),
        user_id_hash=USER_HASH,
    )
    assert result["reason_code"] == CONSENT_REVOKED
    assert result["analytics_event_present"] is False
    assert result["analytics_event"] is None


def test_revoked_at_before_granted_at_is_invalid_in_python_but_schema_format_is_valid():
    record = _consent(revoked_at="2026-07-11T00:00:00Z")
    assert list(Draft202012Validator(_schema(), format_checker=FormatChecker()).iter_errors(record)) == []
    result = validate_analytics_consent_record(record)
    assert result["valid"] is False
    assert "REVOKED_AT_PRECEDES_GRANTED_AT" in result["reason_codes"]


def test_explicit_analytics_exclusion_excludes_event():
    result = map_feedback_to_analytics_with_consent(
        _feedback(), consent_payload=_consent(), user_id_hash=USER_HASH, analytics_exclusion=True
    )
    assert result["reason_code"] == ANALYTICS_EXCLUDED
    assert result["analytics_event_present"] is False


def test_user_hash_mismatch_excludes_event():
    result = map_feedback_to_analytics_with_consent(
        _feedback("hold_created"), consent_payload=_consent(), user_id_hash=OTHER_USER_HASH
    )
    assert result["reason_code"] == CONSENT_USER_MISMATCH
    assert result["analytics_event_present"] is False


def test_missing_consent_scope_is_rejected():
    record = _consent()
    record.pop("consent_scope")
    result = validate_analytics_consent_record(record)
    assert result["valid"] is False
    assert "consent_scope" in result["invalid_fields"]


def test_additional_consent_scope_key_is_rejected():
    record = _consent()
    record["consent_scope"]["uncontrolled_scope"] = True
    result = validate_analytics_consent_record(record)
    assert result["valid"] is False
    assert "INVALID_CONSENT_SCOPE_FIELDS" in result["reason_codes"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("consent_id", "BAD-R452", "INVALID_CONSENT_ID"),
        ("user_id_hash", "sha256:invalid", "INVALID_USER_ID_HASH"),
        ("consent_version", "x" * 33, "INVALID_CONSENT_VERSION"),
        ("retention_policy_id", "RET invalid", "INVALID_RETENTION_POLICY_ID"),
        ("granted_at", "2026-07-12T00:00:00", "INVALID_GRANTED_AT"),
        ("revoked_at", "not-a-datetime", "INVALID_REVOKED_AT"),
    ],
)
def test_invalid_consent_fields_fail_closed(field, value, reason):
    result = validate_analytics_consent_record(_consent(**{field: value}))
    assert result["valid"] is False
    assert reason in result["reason_codes"]


def test_unexpected_top_level_property_is_rejected():
    record = _consent()
    record["safe_extra"] = True
    result = validate_analytics_consent_record(record)
    assert result["valid"] is False
    assert result["reason_codes"] == ["UNEXPECTED_PROPERTY"]


@pytest.mark.parametrize(
    "field_name",
    [
        "raw_query", "raw_body", "prompt", "answer_text", "evidence_text", "api_key",
        "credential", "internal_path", "file_path", "personal_name", "email", "address",
    ],
)
def test_prohibited_fields_fail_before_validation_without_echo(field_name, capsys, caplog):
    record = _consent()
    sentinel = "UNRETAINED_SENTINEL_R452"
    record[field_name] = sentinel
    result = validate_analytics_consent_record(record)
    rendered = json.dumps(result, sort_keys=True)

    assert result["valid"] is False
    assert result["reason_codes"] == ["PROHIBITED_FIELD"]
    assert field_name in result["prohibited_fields"]
    assert sentinel not in rendered
    assert sentinel not in capsys.readouterr().out
    assert sentinel not in caplog.text


@pytest.mark.parametrize(
    ("field_name", "normalized"),
    [("rawQuery", "raw_query"), ("apiKey", "api_key"), ("personalName", "personal_name")],
)
def test_normalized_prohibited_equivalents_fail_closed(field_name, normalized):
    record = _consent()
    record[field_name] = "UNRETAINED_SENTINEL_R452"
    result = validate_analytics_consent_record(record)
    assert result["prohibited_fields"] == [normalized]


def test_denial_does_not_invoke_r451_mapper(monkeypatch):
    def forbidden_mapper(*args, **kwargs):
        raise AssertionError("R451 mapper must not run for denied consent")

    monkeypatch.setattr(consent_policy, "map_feedback_to_analytics_event", forbidden_mapper)
    result = map_feedback_to_analytics_with_consent(
        _feedback(), consent_payload=None, user_id_hash=USER_HASH
    )
    assert result["analytics_event"] is None


def test_consent_denial_does_not_alter_application_answer():
    application_result = {"answer_status": "ANSWERED", "safe_answer_present": True, "safe_answer_length": 19}
    before = copy.deepcopy(application_result)
    result = map_feedback_to_analytics_with_consent(
        _feedback(), consent_payload=None, user_id_hash=USER_HASH
    )
    assert application_result == before
    assert result["reason_code"] == CONSENT_REQUIRED


def test_consent_denial_does_not_alter_application_hold_reason():
    application_result = {"answer_status": "HOLD", "hold_reason_code": "EVIDENCE_REQUIRED"}
    before = copy.deepcopy(application_result)
    result = map_feedback_to_analytics_with_consent(
        _feedback("hold_created"),
        consent_payload=_consent(revoked_at="2026-07-13T00:00:00Z"),
        user_id_hash=USER_HASH,
    )
    assert application_result == before
    assert result["reason_code"] == CONSENT_REVOKED


def test_policy_module_is_pure_without_persistence_db_network_or_file_writes():
    source = inspect.getsource(consent_policy)
    forbidden_imports = (
        "import sqlite", "import socket", "import requests", "import httpx", "import subprocess",
        "import pathlib", "persistence", "repository", "open(", "write_text", "write_bytes",
    )
    assert not any(marker in source for marker in forbidden_imports)


def test_prohibited_policy_contains_all_mandatory_names():
    assert {
        "raw_query", "query", "raw_body", "body", "prompt", "raw_prompt", "raw_text",
        "answer_text", "evidence_text", "standard_text", "paid_standard_text", "secret",
        "api_key", "access_token", "refresh_token", "password", "credential", "cookie",
        "authorization", "internal_path", "local_path", "file_path", "db_path", "dsn",
        "connection_string", "personal_name", "email", "phone", "address",
    } <= set(PROHIBITED_CONSENT_FIELDS)
