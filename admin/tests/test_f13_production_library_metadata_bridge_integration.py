import json
from copy import deepcopy
from pathlib import Path

import pytest

from admin.f13_production_library_metadata_bridge_adapter import (
    ADAPTER_POLICY,
    DUPLICATE_STATUS,
    RIGHTS_STATUS_DECISION,
    SUMMARY_SOURCE,
    adapt_metadata_fixture_records,
    load_metadata_fixture,
)
from admin.f13_production_library_metadata_bridge_integration import (
    INTEGRATION_POLICY,
    build_bridge_hold_projection_from_adapter_record,
    integrate_adapter_records_to_bridge_evidence,
    summarize_integration_rejections,
    validate_integration_projection,
)
from admin.f13_runtime_guard import (
    RAW_TEXT_POLICY_POINTER_ONLY,
    RESULT_HOLD,
    RIGHTS_NOT_VERIFIED,
    decide_bridge_result,
)


FIXTURE_PATH = Path("H:/tmp/R9ZNW-331/production_library_bridge_skeleton_fixture.json")
EXPECTED_FIXTURE_RECORD_COUNT = 39


def _synthetic_fixture_record(**overrides):
    record = {
        "body_text_exposure": "NONE_POINTER_ONLY",
        "bridge_trace_id": "btrace:prodlib:test.synthetic",
        "candidate_role": "PRIMARY_CANDIDATE",
        "doc_id": "SYNTH-001",
        "evidence_id": "prodlib.evd.test.synthetic",
        "fixture_status": "HOLD_FOR_RIGHTS_AND_SUMMARY_REVIEW",
        "page_hint": "NOT_VERIFIED",
        "page_hint_status": "NOT_VERIFIED",
        "pointer_uri": "qlib://production-library/exports/reference_cards/SYNTH-001.md",
        "raw_text_policy": RAW_TEXT_POLICY_POINTER_ONLY,
        "relative_path": "exports/reference_cards/SYNTH-001.md",
        "revision": "NOT_VERIFIED",
        "revision_status": "NOT_VERIFIED",
        "rights_status": RIGHTS_NOT_VERIFIED,
        "rights_status_decision": RIGHTS_STATUS_DECISION,
        "safe_summary": "Metadata-only pointer for synthetic record; semantic summary not verified.",
        "section_label": "NOT_VERIFIED",
        "section_label_status": "NOT_VERIFIED",
        "semantic_summary_verified": False,
        "source_doc_kind": "REFERENCE_CARD_POINTER",
        "source_label": "Synthetic Reference Card",
        "standard_family": "SYNTH",
        "summary_source": SUMMARY_SOURCE,
        "tags": ["SYNTH", "REFERENCE_CARD_POINTER"],
        "unsafe_fields_excluded": [],
        "validation_shape_ids": ["R9ZNW-333_SYNTHETIC_METADATA_ONLY_FIXTURE"],
    }
    record.update(overrides)
    return record


def _walk_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_integration_consumes_synthetic_safe_adapter_records():
    adapter_result = adapt_metadata_fixture_records([_synthetic_fixture_record()])
    integration_result = integrate_adapter_records_to_bridge_evidence(adapter_result["records"])

    assert integration_result["result_status"] == RESULT_HOLD
    assert integration_result["processed_count"] == 1
    assert integration_result["projected_count"] == 1
    assert integration_result["rejected_count"] == 0

    projection = integration_result["evidence_items"][0]
    validation = validate_integration_projection(projection)

    assert validation["is_valid"], validation
    assert projection["rights_status"] == RIGHTS_NOT_VERIFIED
    assert projection["rights_status_decision"] == RIGHTS_STATUS_DECISION
    assert projection["raw_text_policy"] == RAW_TEXT_POLICY_POINTER_ONLY
    assert projection["summary_source"] == SUMMARY_SOURCE
    assert projection["semantic_summary_verified"] is False
    assert projection["raw_text_exposed"] is False
    assert projection["production_path_exposed"] is False
    assert projection["adapter_policy"] == ADAPTER_POLICY
    assert projection["integration_policy"] == INTEGRATION_POLICY
    assert decide_bridge_result(projection["bridge_projected_item"])["result_status"] == RESULT_HOLD


def test_integration_can_adapt_synthetic_metadata_fixture_record_directly():
    integration_result = integrate_adapter_records_to_bridge_evidence([_synthetic_fixture_record()])

    assert integration_result["adapter_record_count"] == 1
    assert integration_result["projected_count"] == 1
    assert integration_result["evidence_items"][0]["evidence_id"] == "prodlib.evd.test.synthetic"


def test_optional_r9znw331_fixture_smoke_processes_all_records_if_present():
    if not FIXTURE_PATH.exists():
        pytest.skip("R9ZNW-331 fixture missing; optional fixture smoke NOT_EXECUTED")

    fixture_records = load_metadata_fixture(FIXTURE_PATH)
    integration_result = integrate_adapter_records_to_bridge_evidence(fixture_records)

    assert len(fixture_records) == EXPECTED_FIXTURE_RECORD_COUNT
    assert integration_result["processed_count"] == EXPECTED_FIXTURE_RECORD_COUNT
    assert integration_result["projected_count"] == EXPECTED_FIXTURE_RECORD_COUNT
    assert integration_result["rejected_count"] == 0
    assert all(
        item["rights_status"] == RIGHTS_NOT_VERIFIED
        for item in integration_result["evidence_items"]
    )


def test_projected_records_exclude_forbidden_paths_db_secret_raw_body_and_readiness_markers():
    adapter_result = adapt_metadata_fixture_records([_synthetic_fixture_record()])
    projection = integrate_adapter_records_to_bridge_evidence(adapter_result["records"])["evidence_items"][0]
    allowed_raw_policy_keys = {"raw_text_policy", "raw_text_exposed"}
    allowed_path_audit_keys = {"production_path_exposed"}
    forbidden_exact_keys = {"raw_text", "full_text", "source_text", "paid_standard_text"}
    forbidden_value_markers = [
        r"H:\장기기억",
        "brain.db",
        "graph.db",
        ".env",
        "credential",
        "token",
        "key",
        "service-account",
        "bearer",
        "secret",
        "f13_pass=true",
        "track_a_pass=true",
        "beta_pass=true",
        "release_ready=true",
        "production_ready=true",
        "semantic_summary_verified=true",
    ]

    keys = set(_walk_keys(projection))
    assert forbidden_exact_keys.isdisjoint(keys)
    assert {key for key in keys if "raw_text" in str(key)}.issubset(allowed_raw_policy_keys)
    assert {key for key in keys if "path" in str(key)}.issubset(allowed_path_audit_keys)

    rendered_values = "\n".join(str(value).lower() for value in _walk_values(projection))
    for marker in forbidden_value_markers:
        assert marker.lower() not in rendered_values


def test_duplicate_hold_records_are_not_promoted_when_fixture_is_available():
    if not FIXTURE_PATH.exists():
        pytest.skip("R9ZNW-331 fixture missing; duplicate smoke NOT_EXECUTED")

    integration_result = integrate_adapter_records_to_bridge_evidence(load_metadata_fixture(FIXTURE_PATH))
    duplicate_records = [
        item for item in integration_result["evidence_items"] if item["duplicate_status"] == DUPLICATE_STATUS
    ]

    assert len(duplicate_records) == 10
    assert all(item["primary_adapter_candidate"] is False for item in duplicate_records)
    assert all(item["duplicate_decision"] == DUPLICATE_STATUS for item in duplicate_records)


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda record: record.update({"rights_status": "PUBLIC"}), "RIGHTS_STATUS_PROMOTED"),
        (
            lambda record: record.update({"semantic_summary_verified": True}),
            "SEMANTIC_SUMMARY_VERIFIED_TRUE",
        ),
        (lambda record: record.update({"raw_text": "not allowed"}), "RAW_BODY_FIELD_PRESENT:raw_text"),
        (
            lambda record: record.update({"pointer_uri": r"H:\장기기억\LIBRARY\unsafe.md"}),
            "ABSOLUTE_FILESYSTEM_PATH_VALUE",
        ),
        (lambda record: record.update({"status": "release_ready=true"}), "BROAD_READINESS_MARKER:release_ready=true"),
    ],
)
def test_unsafe_fixture_records_are_rejected_or_excluded(mutation, expected_reason):
    record = _synthetic_fixture_record()
    mutation(record)

    summary = summarize_integration_rejections([record])

    assert summary["projected_count"] == 0
    assert summary["rejected_count"] == 1
    assert expected_reason in summary["rejected_records"][0]["reason_codes"]


def test_unsafe_adapter_record_is_rejected_before_projection():
    adapter_record = adapt_metadata_fixture_records([_synthetic_fixture_record()])["records"][0]
    unsafe_adapter_record = deepcopy(adapter_record)
    unsafe_adapter_record["bridge_evidence_item"]["pointer_uri"] = r"H:\장기기억\LIBRARY\unsafe.md"

    summary = summarize_integration_rejections([unsafe_adapter_record])

    assert summary["projected_count"] == 0
    assert summary["rejected_count"] == 1
    assert "ABSOLUTE_FILESYSTEM_PATH_VALUE" in summary["rejected_records"][0]["reason_codes"]


def test_build_projection_rejects_promoted_rights_adapter_record():
    adapter_record = adapt_metadata_fixture_records([_synthetic_fixture_record()])["records"][0]
    adapter_record["rights_status"] = "PUBLIC"

    with pytest.raises(ValueError):
        build_bridge_hold_projection_from_adapter_record(adapter_record)


def test_integration_module_public_functions_have_no_db_runtime_browser_or_http_surface():
    from admin import f13_production_library_metadata_bridge_integration as integration

    public_functions = [
        integration.integrate_adapter_records_to_bridge_evidence,
        integration.build_bridge_hold_projection_from_adapter_record,
        integration.validate_integration_projection,
        integration.summarize_integration_rejections,
    ]
    names = set()
    for function in public_functions:
        names.update(function.__code__.co_names)

    forbidden_names = {
        "connect",
        "execute",
        "subprocess",
        "run",
        "popen",
        "requests",
        "urllib",
        "socket",
        "open",
        "write",
        "environ",
        "getenv",
        "dotenv",
        "playwright",
        "selenium",
    }
    assert names.isdisjoint(forbidden_names)


def test_bridge_projected_item_remains_existing_schema_evidence_shape():
    schema = json.loads(Path("schemas/f13_bridge_evidence_response.schema.json").read_text(encoding="utf-8"))
    evidence_schema = schema["properties"]["evidence_items"]["items"]
    allowed_fields = set(evidence_schema["properties"])
    required_fields = set(evidence_schema["required"])
    projection = integrate_adapter_records_to_bridge_evidence([_synthetic_fixture_record()])["evidence_items"][0]
    bridge_projected_item = projection["bridge_projected_item"]

    assert set(bridge_projected_item).issubset(allowed_fields)
    assert required_fields.issubset(set(bridge_projected_item))
    assert bridge_projected_item["rights_status"] == RIGHTS_NOT_VERIFIED
    assert bridge_projected_item["raw_text_policy"] == RAW_TEXT_POLICY_POINTER_ONLY
