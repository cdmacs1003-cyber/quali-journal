import copy
import inspect
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

import admin.f13_analytics_improvement_candidate_contract as candidate_contract
from admin.f13_analytics_improvement_candidate_contract import (
    CANDIDATE_STATUS,
    CANDIDATE_TYPE,
    ELIGIBLE_EVENT_TRIGGER_PAIRS,
    IMPROVEMENT_TRIGGERS,
    PROHIBITED_FIELD_NAMES,
    REQUIRED_FIELDS,
    SUMMARY_CODES,
    build_improvement_idempotency_key,
    map_consent_analytics_to_warehouse_candidate,
    validate_analytics_improvement_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_analytics_improvement_candidate.schema.json"
CREATED_AT = "2026-07-12T00:00:00Z"
QUERY_HASH = "sha256:" + "a" * 64


def _analytics_event(event_type="hold_created", **overrides):
    event = {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "event_id": "EVT-R453-SYNTHETIC-1",
        "tenant_id": "TEN-R453-SYNTHETIC",
        "organization_id": "ORGUNIT-R453-SYNTHETIC",
        "cohort_id": "COH-R453-SYNTHETIC",
        "user_id_hash": "sha256:" + "b" * 64,
        "event_type": event_type,
        "request_id": "REQ-R453-SYNTHETIC-1",
        "trace_id": "btrace:R453:SYNTHETIC:1",
        "query_hash": QUERY_HASH,
        "query_summary": "hold_event" if event_type == "hold_created" else "answer_event",
        "raw_query_stored": False,
        "risk_flags": ["evidence_missing"] if event_type == "hold_created" else [],
        "occurred_at": CREATED_AT,
    }
    event.update(overrides)
    return event


def _allowed_result(event_type="hold_created", **event_overrides):
    return {
        "policy_status": "ALLOW",
        "reason_code": "CONSENT_ALLOWED",
        "analytics_event_present": True,
        "analytics_event": _analytics_event(event_type, **event_overrides),
    }


def _mapped(event_type="hold_created", trigger="evidence_hold", **event_overrides):
    return map_consent_analytics_to_warehouse_candidate(
        _allowed_result(event_type, **event_overrides),
        improvement_trigger=trigger,
        created_at=CREATED_AT,
    )


def _candidate(event_type="hold_created", trigger="evidence_hold"):
    result = _mapped(event_type, trigger)
    assert result["candidate_present"] is True
    return result["candidate"]


def _schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("event_type", "trigger", "summary"),
    [
        ("hold_created", "evidence_hold", "analytics_evidence_hold"),
        ("hold_created", "content_gap", "analytics_content_gap"),
        ("answer_rendered", "review_needed", "analytics_review_needed"),
    ],
)
def test_allowed_event_trigger_pairs_create_valid_candidate(event_type, trigger, summary):
    result = _mapped(event_type, trigger)
    candidate = result["candidate"]

    assert result["reason_code"] == "IMPROVEMENT_CANDIDATE_CREATED"
    assert validate_analytics_improvement_candidate(candidate)["valid"] is True
    assert candidate["summary_code"] == summary
    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(candidate)


def test_candidate_fixed_intake_and_human_review_boundary():
    candidate = _candidate()

    assert candidate["candidate_type"] == CANDIDATE_TYPE
    assert candidate["status"] == CANDIDATE_STATUS == "captured"
    assert candidate["review_required"] is True
    assert candidate["auto_promote"] is False
    assert candidate["approved_for_library"] is False
    assert all(
        candidate[field] is False
        for field in (
            "raw_query_stored",
            "raw_body_stored",
            "raw_answer_stored",
            "raw_evidence_text_stored",
        )
    )


def test_candidate_provenance_and_classification_are_controlled():
    candidate = _candidate()

    assert candidate["provenance"] == {
        "provider_type": "analytics",
        "provider_ref": candidate["source_event_id"],
        "collection_reason": "evidence_hold",
        "rights_status": "owned",
    }
    assert candidate["classification"] == {
        "sensitivity": "internal",
        "visibility": "internal_only",
        "domain": "quality",
    }


def test_context_identifiers_trace_and_hash_are_preserved():
    event = _analytics_event()
    candidate = map_consent_analytics_to_warehouse_candidate(
        {
            "policy_status": "ALLOW",
            "reason_code": "CONSENT_ALLOWED",
            "analytics_event_present": True,
            "analytics_event": event,
        },
        improvement_trigger="evidence_hold",
        created_at=CREATED_AT,
    )["candidate"]

    assert candidate["tenant_context"] == {
        "tenant_id": event["tenant_id"],
        "organization_id": event["organization_id"],
        "cohort_id": event["cohort_id"],
    }
    assert candidate["source_event_id"] == event["event_id"]
    assert candidate["source_request_id"] == event["request_id"]
    assert candidate["source_trace_id"] == event["trace_id"]
    assert candidate["query_hash"] == event["query_hash"]


def test_candidate_excludes_subject_and_analytics_summary():
    candidate = _candidate()

    assert "user_id_hash" not in candidate
    assert "query_summary" not in candidate
    assert "consent" not in candidate


def test_idempotency_is_stable_and_uses_safe_metadata():
    kwargs = {
        "tenant_id": "TEN-R453-SYNTHETIC",
        "organization_id": "ORGUNIT-R453-SYNTHETIC",
        "source_event_id": "EVT-R453-SYNTHETIC-1",
        "source_trace_id": "btrace:R453:SYNTHETIC:1",
        "improvement_trigger": "evidence_hold",
    }
    first = build_improvement_idempotency_key(**kwargs)
    second = build_improvement_idempotency_key(**kwargs)

    assert first == second
    assert first.startswith("idem:analytics:")


def test_changed_trigger_changes_idempotency_key():
    first = _candidate("hold_created", "evidence_hold")
    second = _candidate("hold_created", "content_gap")
    assert first["idempotency_key"] != second["idempotency_key"]


def test_changed_source_event_changes_idempotency_key():
    first = _candidate()
    second = _mapped("hold_created", "evidence_hold", event_id="EVT-R453-SYNTHETIC-2")["candidate"]
    assert first["idempotency_key"] != second["idempotency_key"]


def test_schema_self_check_and_python_contract_alignment():
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(REQUIRED_FIELDS)
    assert set(schema["properties"]["improvement_trigger"]["enum"]) == set(IMPROVEMENT_TRIGGERS)
    assert schema["properties"]["candidate_type"]["const"] == CANDIDATE_TYPE


@pytest.mark.parametrize(
    "reason_code",
    [
        "CONSENT_REQUIRED",
        "CONSENT_RECORD_INVALID",
        "CONSENT_REVOKED",
        "ANALYTICS_EXCLUDED",
        "CONSENT_USER_MISMATCH",
        "CONSENT_SCOPE_DENIED",
    ],
)
def test_denied_or_excluded_consent_creates_no_candidate(reason_code, monkeypatch):
    calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("candidate construction must not run")

    monkeypatch.setattr(candidate_contract, "_build_candidate", fail_if_called)
    result = map_consent_analytics_to_warehouse_candidate(
        {
            "policy_status": "EXCLUDE",
            "reason_code": reason_code,
            "analytics_event_present": False,
            "analytics_event": None,
        },
        improvement_trigger="evidence_hold",
    )
    assert result == {
        "candidate_present": False,
        "status": "NOT_CREATED",
        "reason_code": "CONSENT_NOT_ALLOWED",
        "candidate": None,
    }
    assert calls == 0


def test_denial_does_not_mutate_answer_application_result():
    application_result = {"status": "ANSWERED", "content_present": True}
    before = copy.deepcopy(application_result)
    map_consent_analytics_to_warehouse_candidate(
        {"policy_status": "EXCLUDE", "reason_code": "CONSENT_REQUIRED", "analytics_event_present": False, "analytics_event": None},
        improvement_trigger="review_needed",
    )
    assert application_result == before


def test_denial_does_not_mutate_hold_application_result_or_reason():
    application_result = {"status": "HOLD", "reason_code": "HOLD_REVIEW_REQUIRED"}
    before = copy.deepcopy(application_result)
    map_consent_analytics_to_warehouse_candidate(
        {"policy_status": "EXCLUDE", "reason_code": "CONSENT_REVOKED", "analytics_event_present": False, "analytics_event": None},
        improvement_trigger="evidence_hold",
    )
    assert application_result == before


def test_missing_trigger_creates_no_candidate(monkeypatch):
    monkeypatch.setattr(candidate_contract, "_build_candidate", lambda *args, **kwargs: pytest.fail("unexpected build"))
    result = map_consent_analytics_to_warehouse_candidate(
        _allowed_result("answer_rendered"), improvement_trigger=None
    )
    assert result["candidate_present"] is False
    assert result["reason_code"] == "IMPROVEMENT_TRIGGER_REQUIRED"


def test_uncontrolled_trigger_creates_no_candidate():
    result = map_consent_analytics_to_warehouse_candidate(
        _allowed_result("answer_rendered"), improvement_trigger="uncontrolled"
    )
    assert result["candidate_present"] is False
    assert result["reason_code"] == "IMPROVEMENT_TRIGGER_INVALID"


@pytest.mark.parametrize(
    ("event_type", "trigger"),
    [
        ("answer_rendered", "evidence_hold"),
        ("answer_rendered", "content_gap"),
        ("hold_created", "review_needed"),
        ("question_asked", "review_needed"),
        ("evidence_viewed", "review_needed"),
        ("assessment_viewed", "review_needed"),
    ],
)
def test_ineligible_event_trigger_pair_creates_no_candidate(event_type, trigger):
    result = map_consent_analytics_to_warehouse_candidate(
        _allowed_result(event_type), improvement_trigger=trigger
    )
    assert result["candidate_present"] is False
    assert result["reason_code"] in {"ANALYTICS_EVENT_INVALID", "IMPROVEMENT_EVENT_NOT_ELIGIBLE"}


def test_null_analytics_event_is_rejected():
    result = map_consent_analytics_to_warehouse_candidate(
        {"policy_status": "ALLOW", "reason_code": "CONSENT_ALLOWED", "analytics_event_present": True, "analytics_event": None},
        improvement_trigger="evidence_hold",
    )
    assert result["reason_code"] == "ANALYTICS_EVENT_INVALID"


def test_invalid_analytics_event_is_rejected():
    event = _analytics_event(raw_query_stored=True)
    result = map_consent_analytics_to_warehouse_candidate(
        {"policy_status": "ALLOW", "reason_code": "CONSENT_ALLOWED", "analytics_event_present": True, "analytics_event": event},
        improvement_trigger="evidence_hold",
    )
    assert result["reason_code"] == "ANALYTICS_EVENT_INVALID"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approved_for_library", True),
        ("auto_promote", True),
        ("review_required", False),
        ("status", "approved_for_library"),
        ("status", "promotion_dry_run_pass"),
        ("status", "promoted"),
    ],
)
def test_approval_and_promotion_state_is_rejected(field, value):
    candidate = _candidate()
    candidate[field] = value
    result = validate_analytics_improvement_candidate(candidate)
    assert result["valid"] is False
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema()).validate(candidate)


@pytest.mark.parametrize(
    "field",
    ["approval_event_id", "approver_id", "promoted_library_id", "library_write", "promotion_trace_id"],
)
def test_promotion_or_library_write_fields_are_rejected(field):
    candidate = _candidate()
    candidate[field] = "R453_REJECTED_MARKER"
    result = validate_analytics_improvement_candidate(candidate)
    assert result["valid"] is False
    assert result["reason_codes"] == ["PROHIBITED_FIELD"]
    assert result["prohibited_fields"] == [field]


@pytest.mark.parametrize(
    "field",
    [
        "raw_query", "raw_body", "prompt", "answer", "answer_text", "evidence",
        "evidence_text", "standard_text", "user_id_hash", "personal_name", "email",
        "secret", "credential", "internal_path", "dsn", "connection_string",
    ],
)
def test_raw_sensitive_personal_and_path_fields_fail_closed_without_echo(field):
    candidate = _candidate()
    marker = "R453_REJECTED_MARKER"
    candidate[field] = marker
    result = validate_analytics_improvement_candidate(candidate)
    assert result["valid"] is False
    assert result["reason_codes"] == ["PROHIBITED_FIELD"]
    assert field in result["prohibited_fields"]
    assert marker not in repr(result)


def test_unexpected_property_is_rejected_by_python_and_schema():
    candidate = _candidate()
    candidate["uncontrolled_metadata"] = "CONTROLLED_MARKER"
    result = validate_analytics_improvement_candidate(candidate)
    assert result["valid"] is False
    assert result["reason_codes"] == ["UNEXPECTED_PROPERTY"]
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema()).validate(candidate)


def test_mapper_does_not_mutate_analytics_event():
    allowed = _allowed_result()
    before = copy.deepcopy(allowed)
    _mapped()
    map_consent_analytics_to_warehouse_candidate(
        allowed, improvement_trigger="evidence_hold", created_at=CREATED_AT
    )
    assert allowed == before


def test_mapper_output_is_deterministic_for_identical_safe_input():
    first = _mapped()
    second = _mapped()
    assert first == second


def test_module_is_pure_and_has_no_persistence_network_or_warehouse_runtime_imports():
    source = inspect.getsource(candidate_contract).lower()
    forbidden = (
        "import sqlite",
        "import socket",
        "import requests",
        "import httpx",
        "import subprocess",
        "warehouse_core",
        "open(",
        ".write(",
        "save(",
        "enqueue(",
        "publish(",
    )
    assert all(token not in source for token in forbidden)


def test_schema_rejects_event_trigger_mismatch():
    candidate = _candidate()
    candidate["improvement_trigger"] = "review_needed"
    candidate["summary_code"] = SUMMARY_CODES["review_needed"]
    candidate["provenance"]["collection_reason"] = "review_needed"
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema()).validate(candidate)


def test_prohibited_field_vocabulary_covers_required_boundaries():
    required = {
        "raw_query", "raw_body", "answer", "evidence", "user_id_hash", "personal_name",
        "secret", "api_key", "internal_path", "db_path", "connection_string",
        "library_write", "promoted_library_id", "approver_id", "approval_event_id",
    }
    assert required <= PROHIBITED_FIELD_NAMES
    assert ELIGIBLE_EVENT_TRIGGER_PAIRS == {
        ("hold_created", "evidence_hold"),
        ("hold_created", "content_gap"),
        ("answer_rendered", "review_needed"),
    }
