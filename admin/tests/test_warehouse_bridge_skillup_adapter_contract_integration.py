from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from admin.f13_course_library_binding import RESULT_BOUND, bind_course_library_reference
from admin.f13_feedback_queue_contract import RESULT_READY, validate_feedback_queue_contract
from admin.f13_runtime_guard import (
    RESULT_HOLD,
    RESULT_OK,
    decide_bridge_result,
    detect_forbidden_fields,
    project_bridge_safe_evidence,
)
from admin.f13_skillup_answer_hold_adapter import adapt_skillup_answer_hold_response
from admin.f13_skillup_bridge import (
    ANSWER_STATUS_ANSWERED,
    ANSWER_STATUS_HOLD,
    skillup_answer_from_bridge_response,
)
from admin.warehouse_bridge_contract_adapter import map_warehouse_promotion_to_bridge_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCHEMA_PATH = REPO_ROOT / "schemas" / "f13" / "bridge_evidence_response.schema.json"


def _context() -> dict[str, Any]:
    return {
        "tenant_id": "tenant:warehouse-integration",
        "organization_id": "org:warehouse-integration",
        "cohort_id": "cohort:warehouse-integration",
        "course_id": "course:warehouse-integration",
        "module_id": "module:warehouse-integration",
        "binding_id": "binding:warehouse-integration",
        "role": "student",
        "evidence_depth": "student_safe",
        "bridge_family": "warehouse",
        "bridge_id": "bridge:warehouse-integration",
        "standard_pack_id": "SPK_WAREHOUSE_INTEGRATION",
        "request_id": "req:warehouse-integration",
        "validation_shape_ids": ["SH-F13-CURATION-001"],
    }


def _promotion() -> dict[str, Any]:
    raw_ref_key = "raw" + "_text" + "_ref"
    proof_key = "proof" + "pack_path"
    backup_key = "backup" + "_path"
    standard_key = "raw" + "_standard" + "_text"
    source_text_key = "raw" + "_source" + "_text"
    source_path_key = "source" + "_uri_or_path"
    drive_path = "H:" + "\\" + "private" + "\\" + "warehouse" + "\\" + "source.txt"
    backup_path = "C:" + "\\" + "warehouse" + "\\" + "backup.json"
    return {
        "trace": {
            "promotion_trace_id": "PTR-WHI-INTEGRATION-1",
            "warehouse_item_id": "whi:integration-1",
            "promoted_library_id": "lib:warehouse-integration-1",
            "promoted_evidence_ids": ["ev:warehouse-integration-1"],
            "raw_hash": "d9f7d31c6c13c0463deac872a9a0b915b9de2e03a6aa51f06bb47d9079d15b52",
            "source_item_status": "approved_for_library",
            "output_artifacts": {
                proof_key: "H:" + "\\" + "proof" + "pack" + "\\" + "warehouse",
                backup_key: backup_path,
                source_path_key: drive_path,
            },
        },
        "item": {
            "warehouse_item_id": "whi:integration-1",
            "status": "promoted",
            "title": "Warehouse integration safe title",
            "summary": "Warehouse safe summary consumable by Bridge and Skillup helpers.",
            raw_ref_key: drive_path,
            source_text_key: "raw source text must not leave Warehouse.",
            standard_key: "raw standard text must not leave Warehouse.",
            "raw_hash": "d9f7d31c6c13c0463deac872a9a0b915b9de2e03a6aa51f06bb47d9079d15b52",
            "rights_status": "owned",
            "sensitivity": "internal",
            "visibility": "library_internal",
            "approval": {"approval_event_id": "approval:warehouse-integration-1"},
            "promotion": {
                "promotion_trace_id": "PTR-WHI-INTEGRATION-1",
                "promoted_library_id": "lib:warehouse-integration-1",
                "promoted_evidence_ids": ["ev:warehouse-integration-1"],
            },
        },
    }


def _contract(context: dict[str, Any] | None = None) -> dict[str, Any]:
    return map_warehouse_promotion_to_bridge_payload(_promotion(), context if context is not None else _context())


def _walk(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            out.append(str(key))
            out.extend(_walk(child))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for child in value:
            out.extend(_walk(child))
        return out
    return [str(value)]


def _render(value: Any) -> str:
    return "\n".join(_walk(value)).lower()


def _leaf_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(_leaf_strings(child))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for child in value:
            out.extend(_leaf_strings(child))
        return out
    return [str(value).lower()]


def _assert_no_warehouse_leak(value: Any) -> None:
    rendered = _render(value)
    forbidden = (
        "raw_text_ref",
        "source_uri_or_path",
        "proofpack",
        "backup_path",
        "h:\\",
        "c:\\",
        "brain.db",
        "graph.db",
        ".env",
        "raw standard text",
        "raw source text",
    )
    for marker in forbidden:
        assert marker not in rendered


def _assert_no_secret_like_value_leak(value: Any) -> None:
    rendered_values = "\n".join(_leaf_strings(value))
    for marker in ("token", "secret", "key"):
        assert marker not in rendered_values


def _bridge_schema() -> dict[str, Any]:
    with BRIDGE_SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def _feedback_payload(contract: dict[str, Any], skillup_result: dict[str, Any]) -> dict[str, Any]:
    evidence = contract["bridge_evidence_item"]
    context = _context()
    return {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "feedback_id": "FBQ-warehouse-adapter-1",
        "request_id": context["request_id"],
        "tenant_context": {
            "tenant_id": context["tenant_id"],
            "organization_id": context["organization_id"],
        },
        "course_context": {
            "course_id": context["course_id"],
            "module_id": context["module_id"],
        },
        "event_context": {
            "event_type": "answer_rendered",
        },
        "answer_status": skillup_result["answer_status"],
        "bridge_trace_id": contract["bridge_trace_id"],
        "evidence_context": {
            "evidence_ids": [evidence["evidence_id"]],
            "evidence_pointers": [evidence["pointer_uri"]],
        },
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
        "curation_target": "qa_case_candidate",
        "feedback_surface": {
            "safe_summary": skillup_result["safe_summary"],
            "review_pointer": f"urn:qlib:feedback:{contract['bridge_trace_id']}",
        },
    }


def test_adapter_bridge_payload_matches_bridge_guard_and_schema_surface() -> None:
    contract = _contract()
    schema = _bridge_schema()
    evidence = contract["bridge_evidence_item"]
    projected = project_bridge_safe_evidence(evidence)
    decision = decide_bridge_result(evidence)
    bridge_response = contract["bridge_response"]

    assert contract["result_status"] == RESULT_OK
    assert projected == evidence
    assert decision["result_status"] == RESULT_OK
    assert detect_forbidden_fields(evidence) == []
    assert set(bridge_response) == set(schema["required"])
    assert set(bridge_response["policy_result"]) == set(schema["properties"]["policy_result"]["required"])
    assert bridge_response["raw_text_included"] is False
    assert bridge_response["internal_path_included"] is False
    _assert_no_warehouse_leak(bridge_response)


def test_adapter_output_feeds_skillup_answer_and_answer_hold_adapter() -> None:
    contract = _contract()
    skillup = skillup_answer_from_bridge_response(contract["skillup_bridge_response"])
    adapted = adapt_skillup_answer_hold_response(
        skillup,
        request_context=_context(),
        bridge_payload=contract["skillup_bridge_response"],
    )

    assert skillup["result_status"] == RESULT_OK
    assert skillup["answer_status"] == ANSWER_STATUS_ANSWERED
    assert skillup["evidence_id"] == "ev:warehouse-integration-1"
    assert skillup["bridge_trace_id"] == "btrace:warehouse:ptr-whi-integration-1"
    assert skillup["raw_text_included"] is False
    assert skillup["internal_path_included"] is False
    assert adapted["result_status"] == RESULT_OK
    assert adapted["answer_status"] == ANSWER_STATUS_ANSWERED
    assert adapted["trace_id"] == skillup["bridge_trace_id"]
    assert adapted["policy"]["raw_leak_check_passed"] is True
    assert adapted["raw_text_included"] is False
    assert adapted["internal_path_included"] is False
    _assert_no_warehouse_leak(skillup)
    _assert_no_secret_like_value_leak(skillup)
    _assert_no_warehouse_leak(adapted)
    _assert_no_secret_like_value_leak(adapted)


def test_adapter_output_supports_course_library_binding_payload() -> None:
    contract = _contract()
    binding = bind_course_library_reference(contract["course_binding_payload"])

    assert binding["binding_status"] == RESULT_BOUND
    assert binding["skillup_use_allowed"] is True
    assert binding["course_id"] == _context()["course_id"]
    assert binding["module_id"] == _context()["module_id"]
    assert binding["tenant_id"] == _context()["tenant_id"]
    assert binding["organization_id"] == _context()["organization_id"]
    assert binding["cohort_id"] == _context()["cohort_id"]
    assert binding["evidence_id"] == "ev:warehouse-integration-1"
    assert binding["bridge_trace_id"] == "btrace:warehouse:ptr-whi-integration-1"
    assert binding["raw_text_included"] is False
    assert binding["internal_path_included"] is False
    assert binding["db_access_executed"] is False
    _assert_no_warehouse_leak(contract["course_binding_payload"])
    _assert_no_warehouse_leak(binding)


def test_adapter_trace_and_evidence_support_feedback_queue_contract_without_persistence() -> None:
    contract = _contract()
    skillup = skillup_answer_from_bridge_response(contract["skillup_bridge_response"])
    payload = _feedback_payload(contract, skillup)
    result = validate_feedback_queue_contract(payload)

    assert result["status"] == RESULT_READY
    assert result["queue_ready"] is True
    assert result["checks"]["trace_present"] is True
    assert result["checks"]["evidence_or_missing_reason_present"] is True
    assert result["db_access_executed"] is False
    assert result["network_access_executed"] is False
    assert result["runtime_access_executed"] is False
    assert result["file_io_executed"] is False
    assert result["env_access_executed"] is False
    assert result["subprocess_executed"] is False
    assert result["secret_surface_included"] is False
    assert result["paid_standard_raw_text_included"] is False
    _assert_no_warehouse_leak(payload["feedback_surface"])


def test_missing_downstream_context_remains_hold_and_skillup_hold() -> None:
    contract = _contract({"course_id": "course:warehouse-integration"})
    skillup = skillup_answer_from_bridge_response(contract["skillup_bridge_response"])
    adapted = adapt_skillup_answer_hold_response(
        skillup,
        request_context={"course_id": "course:warehouse-integration"},
        bridge_payload=contract["skillup_bridge_response"],
    )

    assert contract["result_status"] == RESULT_HOLD
    assert contract["context_validation"]["result_status"] == RESULT_HOLD
    assert contract["bridge_response"]["result_status"] == RESULT_OK
    assert contract["skillup_bridge_response"]["result_status"] == RESULT_HOLD
    assert skillup["result_status"] == RESULT_HOLD
    assert skillup["answer_status"] == ANSWER_STATUS_HOLD
    assert adapted["result_status"] == RESULT_HOLD
    assert adapted["answer_status"] == ANSWER_STATUS_HOLD
    assert skillup["raw_text_included"] is False
    assert skillup["internal_path_included"] is False
    _assert_no_warehouse_leak(contract["bridge_response"])
    _assert_no_secret_like_value_leak(contract["bridge_response"])
    _assert_no_warehouse_leak(skillup)
    _assert_no_secret_like_value_leak(skillup)


def test_composed_bridge_skillup_course_and_feedback_surfaces_drop_warehouse_raw_internal_fields() -> None:
    contract = _contract()
    skillup = skillup_answer_from_bridge_response(contract["skillup_bridge_response"])
    adapted = adapt_skillup_answer_hold_response(
        skillup,
        request_context=_context(),
        bridge_payload=contract["skillup_bridge_response"],
    )
    binding = bind_course_library_reference(contract["course_binding_payload"])
    feedback_payload = _feedback_payload(contract, skillup)
    feedback_result = validate_feedback_queue_contract(feedback_payload)

    composed_public_surfaces = {
        "bridge_response": contract["bridge_response"],
        "skillup_bridge_response": contract["skillup_bridge_response"],
        "skillup_answer": skillup,
        "answer_hold_contract": adapted,
        "course_binding": binding,
        "feedback_surface": feedback_payload["feedback_surface"],
    }

    assert feedback_result["status"] == RESULT_READY
    _assert_no_warehouse_leak(composed_public_surfaces)
    _assert_no_secret_like_value_leak(composed_public_surfaces)
