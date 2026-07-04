import json
from copy import deepcopy
from pathlib import Path

import pytest

from admin.f13_production_library_metadata_bridge_adapter import (
    ADAPTER_POLICY,
    ADAPTER_SOURCE,
    DUPLICATE_STATUS,
    RIGHTS_STATUS_DECISION,
    SUMMARY_SOURCE,
    adapt_metadata_fixture_records,
    bridge_evidence_item_from_adapter_record,
    classify_adapter_rejection,
    load_metadata_fixture,
    validate_bridge_skeleton_record,
)
from admin.f13_runtime_guard import (
    RAW_TEXT_POLICY_POINTER_ONLY,
    RESULT_HOLD,
    RIGHTS_NOT_VERIFIED,
    decide_bridge_result,
)


FIXTURE_PATH = Path("H:/tmp/R9ZNW-331/production_library_bridge_skeleton_fixture.json")
SCHEMA_PATH = Path("schemas/f13_bridge_evidence_response.schema.json")
EXPECTED_FIXTURE_RECORD_COUNT = 39


def _records():
    return load_metadata_fixture(FIXTURE_PATH)


def _adapted_result():
    return adapt_metadata_fixture_records(_records())


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


def test_loads_r9znw331_metadata_fixture():
    records = _records()

    assert len(records) == EXPECTED_FIXTURE_RECORD_COUNT
    assert records[0]["rights_status"] == RIGHTS_NOT_VERIFIED
    assert records[0]["raw_text_policy"] == RAW_TEXT_POLICY_POINTER_ONLY
    assert records[0]["summary_source"] == SUMMARY_SOURCE


def test_adapts_all_fixture_records_as_conservative_hold_records():
    result = _adapted_result()

    assert result["result_status"] == RESULT_HOLD
    assert result["adapter_source"] == ADAPTER_SOURCE
    assert result["adapter_policy"] == ADAPTER_POLICY
    assert result["processed_count"] == EXPECTED_FIXTURE_RECORD_COUNT
    assert result["adapted_count"] == EXPECTED_FIXTURE_RECORD_COUNT
    assert result["held_count"] == EXPECTED_FIXTURE_RECORD_COUNT
    assert result["rejected_count"] == 0
    assert result["rejected_records"] == []

    for record in result["records"]:
        validation = validate_bridge_skeleton_record(record)
        bridge_item = bridge_evidence_item_from_adapter_record(record)

        assert validation["is_valid"], validation
        assert record["rights_status"] == RIGHTS_NOT_VERIFIED
        assert record["rights_status_decision"] == RIGHTS_STATUS_DECISION
        assert record["summary_source"] == SUMMARY_SOURCE
        assert record["semantic_summary_verified"] is False
        assert record["raw_text_exposed"] is False
        assert record["production_path_exposed"] is False
        assert record["adapter_source"] == ADAPTER_SOURCE
        assert record["adapter_policy"] == ADAPTER_POLICY
        assert decide_bridge_result(bridge_item)["result_status"] == RESULT_HOLD


def test_adapter_output_excludes_forbidden_paths_db_secret_raw_body_and_readiness_markers():
    result = _adapted_result()
    allowed_raw_policy_keys = {"raw_text_policy", "raw_text_exposed"}
    forbidden_exact_keys = {"raw_text", "full_text", "source_text", "paid_standard_text"}
    forbidden_value_markers = [
        r"H:\장기기억",
        "brain.db",
        "graph.db",
        ".env",
        "credential",
        "service-account",
        "bearer token",
        "f13_pass=true",
        "track_a_pass=true",
        "beta_pass=true",
        "release_ready=true",
        "production_ready=true",
    ]

    for record in result["records"]:
        assert forbidden_exact_keys.isdisjoint(set(_walk_keys(record)))
        assert {
            key for key in _walk_keys(record) if "raw_text" in str(key)
        }.issubset(allowed_raw_policy_keys)
        rendered_values = "\n".join(str(value).lower() for value in _walk_values(record))
        for marker in forbidden_value_markers:
            assert marker.lower() not in rendered_values


def test_duplicate_hold_records_are_not_promoted_to_primary_records():
    result = _adapted_result()
    duplicate_records = [
        record for record in result["records"] if record["duplicate_status"] == DUPLICATE_STATUS
    ]

    assert len(duplicate_records) == 10
    assert all(record["primary_adapter_candidate"] is False for record in duplicate_records)
    assert all(record["duplicate_decision"] == DUPLICATE_STATUS for record in duplicate_records)


def test_unsafe_marker_examples_are_excluded_not_projected():
    result = _adapted_result()
    records_with_exclusions = [
        record for record in result["records"] if record["unsafe_field_exclusion_count"] > 0
    ]

    assert records_with_exclusions
    for record in records_with_exclusions:
        assert "unsafe_fields_excluded" not in record
        assert "library_raw_path" not in json.dumps(record, ensure_ascii=False)
        assert "source_text_window" not in json.dumps(record, ensure_ascii=False)


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
        (lambda record: record.update({"service_account": "service-account marker"}), "SECRET_MARKER_FIELD:service_account"),
        (lambda record: record.update({"api_key": "blocked"}), "SECRET_MARKER_FIELD:api_key"),
        (lambda record: record.update({"status": "release_ready=true"}), "BROAD_READINESS_MARKER:release_ready=true"),
    ],
)
def test_unsafe_or_promoted_fixture_records_fail_closed(mutation, expected_reason):
    record = deepcopy(_records()[0])
    mutation(record)

    classification = classify_adapter_rejection(record)

    assert classification["accepted"] is False
    assert expected_reason in classification["reason_codes"]


def test_loader_rejects_direct_production_library_and_db_like_paths_before_open():
    with pytest.raises(ValueError):
        load_metadata_fixture(r"H:\장기기억\LIBRARY\exports\reference_cards\unsafe.json")

    with pytest.raises(ValueError):
        load_metadata_fixture(r"H:\장기기억\brain.db")


def test_bridge_evidence_item_matches_existing_schema_field_and_enum_contracts():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    evidence_schema = schema["properties"]["evidence_items"]["items"]
    allowed_fields = set(evidence_schema["properties"])
    required_fields = set(evidence_schema["required"])
    raw_text_policy_enum = set(evidence_schema["properties"]["raw_text_policy"]["enum"])
    rights_status_enum = set(evidence_schema["properties"]["rights_status"]["enum"])

    for record in _adapted_result()["records"]:
        bridge_item = bridge_evidence_item_from_adapter_record(record)
        assert set(bridge_item).issubset(allowed_fields)
        assert set(bridge_item) == required_fields
        assert bridge_item["raw_text_policy"] in raw_text_policy_enum
        assert bridge_item["rights_status"] in rights_status_enum
        assert bridge_item["rights_status"] == RIGHTS_NOT_VERIFIED
        assert bridge_item["raw_text_policy"] == RAW_TEXT_POLICY_POINTER_ONLY
