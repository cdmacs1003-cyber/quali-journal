import json
from pathlib import Path


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


def _load_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def test_bridge_evidence_response_schema_uses_result_status_contract():
    schema = _load_schema()
    properties = schema["properties"]
    required = set(schema.get("required", []))

    assert "result_status" in properties
    assert set(properties["result_status"]["enum"]) == {"OK", "HOLD", "DENIED"}
    assert "answer_status" not in required
    assert "answer_status" not in properties
    assert "DENIED" in properties["result_status"]["enum"]


def test_bridge_evidence_response_schema_exposes_safe_evidence_items():
    schema = _load_schema()
    properties = schema["properties"]
    item_properties = properties["evidence_items"]["items"]["properties"]

    assert "evidence_items" in properties
    assert "evidence_id" in item_properties
    assert "bridge_trace_id" in item_properties
    assert "safe_summary" in item_properties
    assert "pointer_uri" in item_properties
    assert "raw_text_policy" in item_properties
    assert "rights_status" in item_properties
    assert "validation_shape_ids" in item_properties
    assert set(properties["evidence_items"]["items"]["required"]) >= {
        "evidence_id",
        "bridge_trace_id",
        "safe_summary",
        "pointer_uri",
        "raw_text_policy",
        "rights_status",
    }


def test_bridge_evidence_response_schema_supports_hold_feedback_and_blocks_raw_leak_fields():
    schema = _load_schema()
    properties = schema["properties"]

    assert "hold_reason" in properties
    assert "feedback_candidate_required" in properties
    assert properties["feedback_candidate_required"]["type"] == "boolean"
    assert FORBIDDEN_TOP_LEVEL_PROPERTIES.isdisjoint(properties)
    assert schema.get("additionalProperties") is False
    assert properties["evidence_items"]["items"].get("additionalProperties") is False
