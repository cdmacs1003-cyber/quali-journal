from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from admin.f13_skillup_answer_hold_adapter import adapt_skillup_answer_hold_response
from admin.f13_skillup_bridge import (
    skillup_answer_from_bridge_response,
    skillup_answer_from_request,
    skillup_feedback_queue_item_from_hold,
)
from admin.f13_skillup_feedback_queue_persistence import (
    SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS,
    durable_feedback_queue_item_from_hold,
)


# R9ZMZ creates this bounded test surface only. The file is not executed by
# R9ZMZ, and no validator execution evidence is granted by its creation.
# FULL_JSON_SCHEMA_CONFORMANCE_PASS remains NOT_GRANTED until a later approved
# execution packet runs bounded node IDs and records evidence.
# R9ZN3 adds future adapter-produced synthetic payload nodes only. R9ZN3 does
# not execute helper calls, pytest, or JSON Schema validation.

REPO_ROOT = Path(__file__).resolve().parents[2]

ANSWER_HOLD_RESPONSE_SCHEMA = "schemas/skillup_answer_hold_response.schema.json"
ROUTE_MAPPING_DOCUMENT = "schemas/skillup_answer_hold_route_mapping.schema.json"
FEEDBACK_QUEUE_ITEM_SCHEMA = "schemas/skillup_feedback_queue_item.schema.json"
FEEDBACK_QUEUE_DB_ROW_SCHEMA = "schemas/skillup_feedback_queue_db_row.schema.json"

_TRACKED_JSON_INPUTS = {
    ANSWER_HOLD_RESPONSE_SCHEMA,
    ROUTE_MAPPING_DOCUMENT,
    FEEDBACK_QUEUE_ITEM_SCHEMA,
    FEEDBACK_QUEUE_DB_ROW_SCHEMA,
}


def _tracked_json_path(relative_path: str) -> Path:
    assert relative_path in _TRACKED_JSON_INPUTS
    return REPO_ROOT / relative_path


def _load_json(relative_path: str) -> dict[str, Any]:
    with _tracked_json_path(relative_path).open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _load_schema(relative_path: str) -> dict[str, Any]:
    schema = _load_json(relative_path)
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    return schema


def _json_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _validation_errors(schema_relative_path: str, payload: dict[str, Any]) -> list[Any]:
    schema = _load_schema(schema_relative_path)
    validator = Draft202012Validator(schema)
    return sorted(
        validator.iter_errors(_json_clone(payload)),
        key=lambda error: tuple(error.path),
    )


def _validate(schema_relative_path: str, payload: dict[str, Any]) -> None:
    assert _validation_errors(schema_relative_path, payload) == []


def _sample_ok_answer_payload() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "contract_version": "R9ZKY-2026-06-13",
        "trace_id": "btrace:skillup-jsonschema-static-ok-1",
        "request_id": "req:skillup-jsonschema-static-ok-1",
        "course_id": "course:skillup-jsonschema",
        "module_id": "module:skillup-jsonschema",
        "binding_id": "binding:skillup-jsonschema",
        "answer_status": "ANSWERED",
        "result_status": "OK",
        "answer": "Synthetic safe summary for JSON Schema conformance.",
        "evidence_required": False,
        "evidence": [
            {
                "evidence_id": "ev:skillup-jsonschema-static-1",
                "node_id": "node:skillup-jsonschema-static-1",
                "pointer": "pointer://diagnostic/skillup-jsonschema/static-1",
                "source_label": "Skillup Bridge safe evidence",
                "rights_status": "PUBLIC",
                "sensitivity": "LOW",
            }
        ],
        "policy": {
            "raw_leak_check_passed": True,
            "rights_check_passed": True,
            "sensitivity_check_passed": True,
            "evidence_check_passed": True,
        },
        "raw_text_included": False,
        "internal_path_included": False,
        "warnings": [],
        "review_required": False,
    }


def _sample_hold_answer_payload() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "contract_version": "R9ZKY-2026-06-13",
        "trace_id": "btrace:skillup-jsonschema-static-hold-1",
        "request_id": "req:skillup-jsonschema-static-hold-1",
        "course_id": "course:skillup-jsonschema",
        "module_id": "module:skillup-jsonschema",
        "binding_id": "binding:skillup-jsonschema",
        "answer_status": "HOLD",
        "result_status": "HOLD",
        "hold_reason_code": "EVIDENCE_REQUIRED",
        "hold_reason": "Evidence review is required before Skillup can answer.",
        "evidence_required": True,
        "evidence": [],
        "policy": {
            "raw_leak_check_passed": True,
            "rights_check_passed": False,
            "sensitivity_check_passed": True,
            "evidence_check_passed": False,
        },
        "raw_text_included": False,
        "internal_path_included": False,
        "warnings": ["EVIDENCE_ARRAY_EMPTY_FOR_HOLD"],
        "review_required": True,
    }


def _sample_denied_or_error_payload() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "contract_version": "R9ZKY-2026-06-13",
        "trace_id": "btrace:skillup-jsonschema-static-error-1",
        "request_id": "req:skillup-jsonschema-static-error-1",
        "course_id": "course:skillup-jsonschema",
        "module_id": "module:skillup-jsonschema",
        "binding_id": "binding:skillup-jsonschema",
        "answer_status": "INVALIDATED",
        "result_status": "ERROR",
        "hold_reason_code": "SOURCE_CONTENT_BLOCKED",
        "hold_reason": "Unsafe source content was blocked.",
        "evidence_required": True,
        "evidence": [],
        "policy": {
            "raw_leak_check_passed": True,
            "rights_check_passed": False,
            "sensitivity_check_passed": False,
            "evidence_check_passed": False,
        },
        "raw_text_included": False,
        "internal_path_included": False,
        "warnings": ["SOURCE_DENIED_NORMALIZED_TO_ERROR"],
        "review_required": True,
    }


def _sample_feedback_queue_item_payload() -> dict[str, Any]:
    return {
        "contract_version": "R9ZMH-2026-06-14",
        "persistence_mechanism": "DB_BACKED_QUEUE_DEFERRED",
        "feedback_id": "fbq:skillup_jsonschema_static_1",
        "origin_event_id": "hold:skillup_jsonschema_static_1",
        "current_status": "queued",
        "dedup_key": "Skillup:EVIDENCE_GAP:skillup_jsonschema_static_1",
        "created_at": "1970-01-01T00:00:00Z",
        "review_reason_code": "EVIDENCE_REQUIRED",
        "safe_summary": "Synthetic safe feedback queue summary.",
        "result_status": "HOLD",
        "answer_status": "HOLD",
        "evidence_required": True,
        "review_required": True,
        "evidence_count": 0,
        "warning_codes": ["EVIDENCE_ARRAY_EMPTY_FOR_HOLD"],
        "trace_id": "btrace:skillup-jsonschema-static-hold-1",
        "request_id": "req:skillup-jsonschema-static-hold-1",
        "raw_text_included": False,
        "internal_path_included": False,
        "db_access_executed": False,
    }


def _sample_feedback_queue_db_row_payload() -> dict[str, Any]:
    return _json_clone(_sample_feedback_queue_item_payload())


def _payload_with_internal_queue_field_exposed() -> dict[str, Any]:
    payload = _json_clone(_sample_hold_answer_payload())
    payload["feedback_queue_item"] = _sample_feedback_queue_item_payload()
    return payload


def _payload_missing_required_shape_field() -> dict[str, Any]:
    payload = _json_clone(_sample_ok_answer_payload())
    payload.pop("trace_id")
    return payload


def _adapter_request_context() -> dict[str, Any]:
    return {
        "request_id": "req:skillup-jsonschema-adapter-1",
        "course_id": "course:skillup-jsonschema",
        "module_id": "module:skillup-jsonschema",
        "binding_id": "binding:skillup-jsonschema",
        "origin_event_id": "hold:skillup_jsonschema_adapter_1",
        "requester_module": "Skillup",
    }


def _adapter_bridge_ok_response() -> dict[str, Any]:
    return {
        "result_status": "OK",
        "role": "student",
        "evidence_depth": "student_safe",
        "course_id": "course:skillup-jsonschema",
        "module_id": "module:skillup-jsonschema",
        "binding_id": "binding:skillup-jsonschema",
        "tenant_id": "tenant:skillup-jsonschema",
        "organization_id": "org:skillup-jsonschema",
        "cohort_id": "cohort:skillup-jsonschema",
        "policy_result": {
            "raw_leak_pass": True,
            "rights_pass": True,
            "sensitivity_pass": True,
            "evidence_required_pass": True,
        },
        "evidence_items": [
            {
                "evidence_id": "ev:skillup-jsonschema-adapter-1",
                "node_id": "node:skillup-jsonschema-adapter-1",
                "bridge_trace_id": "btrace:skillup-jsonschema-adapter-ok-1",
                "safe_summary": "Synthetic adapter-produced safe summary.",
                "pointer_uri": "pointer://diagnostic/skillup-jsonschema/adapter-ok-1",
                "source_label": "Skillup Bridge safe evidence",
                "rights_status": "PUBLIC",
                "sensitivity": "LOW",
                "raw_text_policy": "SUMMARY_ONLY",
                "source_doc_kind": "PUBLIC_REFERENCE",
            }
        ],
    }


def _adapter_produced_ok_payload() -> dict[str, Any]:
    bridge_response = _adapter_bridge_ok_response()
    helper_response = skillup_answer_from_bridge_response(bridge_response)
    return adapt_skillup_answer_hold_response(
        helper_response,
        request_context=_adapter_request_context(),
        bridge_payload=bridge_response,
    )


def _adapter_produced_hold_payload() -> dict[str, Any]:
    request_context = _adapter_request_context()
    helper_response = skillup_answer_from_request(request_context)
    return adapt_skillup_answer_hold_response(
        helper_response,
        request_context=request_context,
        bridge_payload={},
    )


def _adapter_denied_bridge_response() -> dict[str, Any]:
    bridge_response = _adapter_bridge_ok_response()
    bridge_response["result_status"] = "DENIED"
    bridge_response["hold_reason"] = "Bridge response included raw text."
    bridge_response["raw_text_included"] = True
    bridge_response["policy_result"] = {
        "raw_leak_pass": False,
        "rights_pass": False,
        "sensitivity_pass": False,
        "evidence_required_pass": False,
    }
    return bridge_response


def _adapter_produced_denied_error_payload() -> dict[str, Any]:
    bridge_response = _adapter_denied_bridge_response()
    helper_response = skillup_answer_from_bridge_response(bridge_response)
    return adapt_skillup_answer_hold_response(
        helper_response,
        request_context=_adapter_request_context(),
        bridge_payload=bridge_response,
    )


def _adapter_produced_no_db_boundary_payload() -> dict[str, Any]:
    direct_db_attempt = {
        **_adapter_request_context(),
        "direct_db_access_attempt": True,
    }
    helper_response = skillup_answer_from_request(direct_db_attempt)
    return adapt_skillup_answer_hold_response(
        helper_response,
        request_context=_adapter_request_context(),
        bridge_payload={},
    )


def _adapter_source_with_queue_internal_payload() -> dict[str, Any]:
    request_context = _adapter_request_context()
    helper_response = skillup_answer_from_request(request_context)
    queue_source = {
        **helper_response,
        "origin_module": request_context["requester_module"],
        "origin_event_id": request_context["origin_event_id"],
    }
    return {
        **queue_source,
        "feedback_queue_item": skillup_feedback_queue_item_from_hold(queue_source),
    }


def _adapter_produced_queue_internal_omission_payload() -> dict[str, Any]:
    return adapt_skillup_answer_hold_response(
        _adapter_source_with_queue_internal_payload(),
        request_context=_adapter_request_context(),
        bridge_payload={},
    )


def _adapter_produced_durable_queue_payload() -> dict[str, Any]:
    request_context = _adapter_request_context()
    hold_response = skillup_answer_from_request(request_context)
    durable_item = durable_feedback_queue_item_from_hold(
        {
            "feedback_id": "fbq:skillup_jsonschema_adapter_1",
            "origin_event_id": request_context["origin_event_id"],
            "current_status": "queued",
            "dedup_key": "Skillup:EVIDENCE_GAP:skillup_jsonschema_adapter_1",
            "created_at": "1970-01-01T00:00:00Z",
            "review_reason_code": "BRIDGE_RESPONSE_REQUIRED",
            "safe_summary": hold_response["hold_reason"],
            "result_status": "HOLD",
            "answer_status": "HOLD",
            "evidence_required": True,
            "review_required": True,
            "evidence_count": 0,
            "warning_codes": ["EVIDENCE_ARRAY_EMPTY_FOR_HOLD"],
            "trace_id": "btrace:skillup-jsonschema-adapter-hold-1",
            "request_id": request_context["request_id"],
            "raw_text_included": False,
            "internal_path_included": False,
            "db_access_executed": False,
        }
    )
    # This is schema-only contract validation, not persistence proof.
    return durable_item.to_persistence_dict()


def test_skillup_answer_hold_response_schema_accepts_static_ok_payload() -> None:
    _validate(ANSWER_HOLD_RESPONSE_SCHEMA, _sample_ok_answer_payload())


def test_skillup_answer_hold_response_schema_accepts_static_hold_payload() -> None:
    _validate(ANSWER_HOLD_RESPONSE_SCHEMA, _sample_hold_answer_payload())


def test_skillup_answer_hold_response_schema_accepts_static_denied_error_payload() -> None:
    _validate(ANSWER_HOLD_RESPONSE_SCHEMA, _sample_denied_or_error_payload())


def test_skillup_answer_hold_response_schema_rejects_queue_internal_fields() -> None:
    errors = _validation_errors(
        ANSWER_HOLD_RESPONSE_SCHEMA,
        _payload_with_internal_queue_field_exposed(),
    )
    assert any(error.validator == "additionalProperties" for error in errors)


def test_skillup_answer_hold_response_schema_rejects_missing_required_field() -> None:
    errors = _validation_errors(
        ANSWER_HOLD_RESPONSE_SCHEMA,
        _payload_missing_required_shape_field(),
    )
    assert any(error.validator == "required" for error in errors)


def test_skillup_feedback_queue_item_schema_accepts_static_contract_payload() -> None:
    _validate(FEEDBACK_QUEUE_ITEM_SCHEMA, _sample_feedback_queue_item_payload())


def test_skillup_feedback_queue_db_row_schema_accepts_static_fixture_row_payload() -> None:
    _validate(FEEDBACK_QUEUE_DB_ROW_SCHEMA, _sample_feedback_queue_db_row_payload())


def test_skillup_feedback_queue_item_schema_requires_minimal_bridge_answer_metadata() -> None:
    payload = _sample_feedback_queue_item_payload()
    payload.pop("evidence_count")

    errors = _validation_errors(FEEDBACK_QUEUE_ITEM_SCHEMA, payload)

    assert any(error.validator == "required" for error in errors)


def test_skillup_answer_hold_response_schema_accepts_adapter_produced_ok_payload() -> None:
    payload = _adapter_produced_ok_payload()
    assert payload["result_status"] == "OK"
    assert payload["answer_status"] == "ANSWERED"
    _validate(ANSWER_HOLD_RESPONSE_SCHEMA, payload)


def test_skillup_answer_hold_response_schema_accepts_adapter_produced_hold_payload() -> None:
    payload = _adapter_produced_hold_payload()
    assert payload["result_status"] == "HOLD"
    assert payload["answer_status"] == "HOLD"
    assert payload["hold_reason_code"] == "BRIDGE_RESPONSE_REQUIRED"
    _validate(ANSWER_HOLD_RESPONSE_SCHEMA, payload)


def test_skillup_answer_hold_response_schema_accepts_adapter_produced_denied_error_payload() -> None:
    payload = _adapter_produced_denied_error_payload()
    assert payload["result_status"] == "ERROR"
    assert payload["answer_status"] == "INVALIDATED"
    assert payload["hold_reason_code"] == "SOURCE_CONTENT_BLOCKED"
    _validate(ANSWER_HOLD_RESPONSE_SCHEMA, payload)


def test_skillup_answer_hold_response_schema_accepts_adapter_produced_no_db_boundary_payload() -> None:
    payload = _adapter_produced_no_db_boundary_payload()
    assert payload["result_status"] == "ERROR"
    assert payload["answer_status"] == "INVALIDATED"
    assert payload["hold_reason_code"] == "NO_DB_BOUNDARY"
    _validate(ANSWER_HOLD_RESPONSE_SCHEMA, payload)


def test_skillup_answer_hold_response_schema_omits_adapter_source_queue_internals() -> None:
    payload = _adapter_produced_queue_internal_omission_payload()
    assert SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS.isdisjoint(payload)
    _validate(ANSWER_HOLD_RESPONSE_SCHEMA, payload)


def test_skillup_answer_hold_response_schema_rejects_unadapted_queue_internal_payload() -> None:
    errors = _validation_errors(
        ANSWER_HOLD_RESPONSE_SCHEMA,
        _adapter_source_with_queue_internal_payload(),
    )
    assert any(error.validator == "additionalProperties" for error in errors)


def test_skillup_feedback_queue_item_schema_accepts_adapter_produced_durable_queue_payload() -> None:
    _validate(FEEDBACK_QUEUE_ITEM_SCHEMA, _adapter_produced_durable_queue_payload())


def test_skillup_route_mapping_references_existing_schema_surfaces() -> None:
    mapping = _load_json(ROUTE_MAPPING_DOCUMENT)
    assert mapping["mapping_status"] == "CANDIDATE_WITH_LIMITS"
    assert mapping["source_schema"] == ANSWER_HOLD_RESPONSE_SCHEMA
    assert mapping["feedback_queue_persistence_contract"]["durable_schema"] == FEEDBACK_QUEUE_ITEM_SCHEMA
    assert _tracked_json_path(mapping["source_schema"]).is_file()
    assert _tracked_json_path(
        mapping["feedback_queue_persistence_contract"]["durable_schema"]
    ).is_file()
