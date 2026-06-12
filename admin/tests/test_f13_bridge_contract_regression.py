import json
from pathlib import Path

from admin.f13_runtime_guard import (
    RAW_TEXT_POLICY_SUMMARY_ONLY,
    RESULT_DENIED,
    RESULT_HOLD,
    RESULT_OK,
    RIGHTS_PUBLIC,
    RIGHTS_RESTRICTED,
    decide_bridge_result,
    detect_forbidden_fields,
    project_bridge_safe_evidence,
    validate_bridge_safe_response,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13" / "bridge_evidence_response.schema.json"

FORBIDDEN_TOP_LEVEL_PROPERTIES = {
    "raw_text_ref",
    "raw_pointer",
    "raw_source_text",
    "full_source_text",
    "source_uri_or_path",
    "direct_db_row",
    "warehouse_internal_object",
    "library_internal_object",
}

REQUIRED_EVIDENCE_ITEM_FIELDS = {
    "evidence_id",
    "bridge_trace_id",
    "safe_summary",
    "pointer_uri",
    "raw_text_policy",
    "rights_status",
}


def _load_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def _schema_status_values(schema: dict) -> set[str]:
    return set(schema["properties"]["result_status"]["enum"])


def _safe_evidence(**overrides: object) -> dict:
    evidence = {
        "evidence_id": "ev:regression-1",
        "bridge_trace_id": "btrace:regression-1",
        "safe_summary": "Synthetic safe summary for Bridge contract regression.",
        "pointer_uri": "pointer://safe/regression-1",
        "raw_text_policy": RAW_TEXT_POLICY_SUMMARY_ONLY,
        "rights_status": RIGHTS_PUBLIC,
    }
    evidence.update(overrides)
    return evidence


def test_schema_and_utility_status_vocabulary_are_aligned():
    schema = _load_schema()
    properties = schema["properties"]
    schema_status_values = _schema_status_values(schema)
    utility_status_values = {RESULT_OK, RESULT_HOLD, RESULT_DENIED}

    assert "result_status" in properties
    assert schema_status_values == {"OK", "HOLD", "DENIED"}
    assert utility_status_values == schema_status_values
    assert "answer_status" not in properties
    assert "ANSWERABLE" not in schema_status_values
    assert "DENIED" in schema_status_values


def test_ok_decision_aligns_with_schema_evidence_items_contract():
    schema = _load_schema()
    properties = schema["properties"]
    item_properties = properties["evidence_items"]["items"]["properties"]
    evidence = _safe_evidence()

    decision = decide_bridge_result(evidence)
    projected = project_bridge_safe_evidence(evidence)

    assert decision["result_status"] == RESULT_OK
    assert "evidence_items" in properties
    assert REQUIRED_EVIDENCE_ITEM_FIELDS.issubset(item_properties)
    assert REQUIRED_EVIDENCE_ITEM_FIELDS.issubset(projected)
    assert detect_forbidden_fields(projected) == []


def test_projected_evidence_lengths_do_not_exceed_schema_contract():
    schema = _load_schema()
    item_properties = schema["properties"]["evidence_items"]["items"]["properties"]

    projected = project_bridge_safe_evidence(
        _safe_evidence(
            evidence_id="e" * (item_properties["evidence_id"]["maxLength"] + 1),
            bridge_trace_id="b" * (item_properties["bridge_trace_id"]["maxLength"] + 1),
            source_doc_kind="s" * (item_properties["source_doc_kind"]["maxLength"] + 1),
        )
    )
    boundary_projected = project_bridge_safe_evidence(
        _safe_evidence(
            evidence_id="e" * item_properties["evidence_id"]["maxLength"],
            bridge_trace_id="b" * item_properties["bridge_trace_id"]["maxLength"],
            source_doc_kind="s" * item_properties["source_doc_kind"]["maxLength"],
        )
    )

    assert "evidence_id" not in projected
    assert "bridge_trace_id" not in projected
    assert "source_doc_kind" not in projected
    assert len(boundary_projected["evidence_id"]) == item_properties["evidence_id"]["maxLength"]
    assert len(boundary_projected["bridge_trace_id"]) == item_properties["bridge_trace_id"]["maxLength"]
    assert len(boundary_projected["source_doc_kind"]) == item_properties["source_doc_kind"]["maxLength"]


def test_hold_decision_aligns_with_schema_hold_contract():
    schema = _load_schema()
    properties = schema["properties"]
    evidence = _safe_evidence(evidence_id="")

    decision = decide_bridge_result(evidence)

    assert decision["result_status"] == RESULT_HOLD
    assert decision["hold_reason"]
    assert "hold_reason" in properties
    assert "feedback_candidate_required" in properties


def test_denied_decision_aligns_with_schema_denied_contract():
    schema = _load_schema()
    properties = schema["properties"]
    evidence = _safe_evidence(rights_status=RIGHTS_RESTRICTED)

    decision = decide_bridge_result(evidence)

    assert decision["result_status"] == RESULT_DENIED
    assert "DENIED" in _schema_status_values(schema)
    assert "hold_reason" in properties
    assert "feedback_candidate_required" in properties


def test_raw_leak_fields_remain_blocked_at_contract_and_utility_level():
    schema = _load_schema()
    properties = schema["properties"]
    payload = _safe_evidence(raw_text_ref="synthetic forbidden reference")
    response = {**payload, "result_status": RESULT_OK}

    assert FORBIDDEN_TOP_LEVEL_PROPERTIES.isdisjoint(properties)
    assert schema.get("additionalProperties") is False
    assert "raw_text_ref" in detect_forbidden_fields(payload)

    validation = validate_bridge_safe_response(response)

    assert validation["is_safe"] is False
    assert validation["result_status"] == RESULT_DENIED
