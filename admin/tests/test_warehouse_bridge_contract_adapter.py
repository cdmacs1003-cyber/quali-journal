from __future__ import annotations

from typing import Any

from admin.f13_course_library_binding import RESULT_BOUND, bind_course_library_reference
from admin.f13_runtime_guard import RESULT_HOLD, RESULT_OK, detect_forbidden_fields
from admin.f13_skillup_bridge import ANSWER_STATUS_ANSWERED, skillup_answer_from_bridge_response
from admin.warehouse_bridge_contract_adapter import (
    build_bridge_contract_from_warehouse_promotion,
    map_warehouse_promotion_to_bridge_payload,
    validate_warehouse_bridge_context,
)


def _safe_context() -> dict[str, Any]:
    return {
        "tenant_id": "tenant:warehouse-selected",
        "organization_id": "org:warehouse-selected",
        "cohort_id": "cohort:warehouse-selected",
        "course_id": "course:warehouse-selected",
        "module_id": "module:warehouse-selected",
        "binding_id": "binding:warehouse-selected",
        "role": "student",
        "evidence_depth": "student_safe",
        "bridge_family": "warehouse",
        "bridge_id": "bridge:warehouse-selected",
        "standard_pack_id": "SPK_WAREHOUSE_SELECTED",
        "request_id": "req:warehouse-selected",
        "validation_shape_ids": ["SH-F13-CURATION-001"],
    }


def _promotion() -> dict[str, Any]:
    hidden_ref_key = "raw" + "_text" + "_ref"
    source_path_key = "source" + "_uri_or_path"
    proof_bundle_key = "proof" + "pack_path"
    backup_path_key = "backup" + "_path"
    standard_text_key = "raw" + "_standard" + "_text"
    drive_path = "H:" + "\\" + "private" + "\\" + "warehouse" + "\\" + "source.txt"
    local_path = "C:" + "\\" + "workspace" + "\\" + "warehouse" + "\\" + "backup.json"
    return {
        "trace": {
            "promotion_trace_id": "PTR-WHI-ABC-1",
            "warehouse_item_id": "whi:selected-1",
            "promoted_library_id": "lib:warehouse-selected-1",
            "promoted_evidence_ids": ["ev:warehouse-selected-1"],
            "raw_hash": "7f5a4a5db7f65f98f6c7fc2d7c85598ac338e5fd9a03c64f84203a5c78f020f5",
            "source_item_status": "approved_for_library",
            "output_artifacts": {
                source_path_key: drive_path,
                proof_bundle_key: "H:" + "\\" + "proof" + "pack" + "\\" + "warehouse",
                backup_path_key: local_path,
            },
        },
        "item": {
            "warehouse_item_id": "whi:selected-1",
            "status": "promoted",
            "title": "Selected warehouse summary",
            "summary": "Selected safe summary for Bridge and Skillup.",
            hidden_ref_key: drive_path,
            standard_text_key: "paid standard source material must stay out of adapter output",
            "raw_hash": "7f5a4a5db7f65f98f6c7fc2d7c85598ac338e5fd9a03c64f84203a5c78f020f5",
            "rights_status": "owned",
            "sensitivity": "internal",
            "visibility": "library_internal",
            "approval": {
                "approval_event_id": "approval:warehouse-selected-1",
                "reviewer_id": "reviewer:warehouse-selected",
            },
            "promotion": {
                "promotion_trace_id": "PTR-WHI-ABC-1",
                "promoted_library_id": "lib:warehouse-selected-1",
                "promoted_evidence_ids": ["ev:warehouse-selected-1"],
            },
        },
    }


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


def test_maps_promotion_trace_id_to_deterministic_bridge_trace_id() -> None:
    first = map_warehouse_promotion_to_bridge_payload(_promotion(), _safe_context())
    second = build_bridge_contract_from_warehouse_promotion(_promotion(), _safe_context())

    assert first["bridge_trace_id"] == "btrace:warehouse:ptr-whi-abc-1"
    assert second["bridge_trace_id"] == first["bridge_trace_id"]
    assert first["bridge_response"]["evidence_items"][0]["bridge_trace_id"] == first["bridge_trace_id"]


def test_preserves_safe_identifiers_and_supports_bridge_skillup_binding_contracts() -> None:
    contract = map_warehouse_promotion_to_bridge_payload(_promotion(), _safe_context())
    metadata = contract["safe_metadata"]

    assert contract["result_status"] == RESULT_OK
    assert metadata["warehouse_item_id"] == "whi:selected-1"
    assert metadata["promotion_trace_id"] == "PTR-WHI-ABC-1"
    assert metadata["promoted_library_id"] == "lib:warehouse-selected-1"
    assert metadata["evidence_id"] == "ev:warehouse-selected-1"
    assert metadata["raw_hash"].startswith("7f5a4a5d")
    assert contract["bridge_evidence_item"]["evidence_id"] == "ev:warehouse-selected-1"
    assert contract["bridge_evidence_item"]["pointer_uri"] == "pointer://warehouse/evidence/ev:warehouse-selected-1"
    assert detect_forbidden_fields(contract["bridge_evidence_item"]) == []

    skillup = skillup_answer_from_bridge_response(contract["skillup_bridge_response"])
    assert skillup["result_status"] == RESULT_OK
    assert skillup["answer_status"] == ANSWER_STATUS_ANSWERED
    assert skillup["evidence_id"] == "ev:warehouse-selected-1"
    assert skillup["bridge_trace_id"] == "btrace:warehouse:ptr-whi-abc-1"
    assert skillup["raw_text_included"] is False
    assert skillup["internal_path_included"] is False

    binding = bind_course_library_reference(contract["course_binding_payload"])
    assert binding["binding_status"] == RESULT_BOUND
    assert binding["skillup_use_allowed"] is True
    assert binding["raw_text_included"] is False
    assert binding["internal_path_included"] is False


def test_drops_raw_internal_and_backup_surfaces_from_contract_output() -> None:
    contract = map_warehouse_promotion_to_bridge_payload(_promotion(), _safe_context())
    rendered = _render(contract)

    assert "raw_text_ref" not in rendered
    assert "source_uri_or_path" not in rendered
    assert "proofpack" not in rendered
    assert "backup_path" not in rendered
    assert "raw_standard_text" not in rendered
    assert "h:\\" not in rendered
    assert "c:\\" not in rendered
    assert "brain.db" not in rendered
    assert "graph.db" not in rendered
    assert "secret" not in rendered
    assert "token" not in rendered
    assert contract["raw_text_included"] is False
    assert contract["internal_path_included"] is False


def test_missing_required_context_returns_hold_not_fake_success() -> None:
    context_result = validate_warehouse_bridge_context({"course_id": "course:warehouse-selected"})
    contract = map_warehouse_promotion_to_bridge_payload(_promotion(), {"course_id": "course:warehouse-selected"})
    skillup = skillup_answer_from_bridge_response(contract["skillup_bridge_response"])

    assert context_result["result_status"] == RESULT_HOLD
    assert set(context_result["missing_fields"]) == {
        "tenant_id",
        "organization_id",
        "cohort_id",
        "module_id",
        "binding_id",
    }
    assert contract["result_status"] == RESULT_HOLD
    assert contract["context_validation"]["result_status"] == RESULT_HOLD
    assert contract["skillup_bridge_response"]["result_status"] == RESULT_HOLD
    assert skillup["result_status"] == RESULT_HOLD
    assert skillup["raw_text_included"] is False
    assert skillup["internal_path_included"] is False


def test_maps_warehouse_approval_and_promoted_status_to_safe_downstream_fields() -> None:
    contract = map_warehouse_promotion_to_bridge_payload(_promotion(), _safe_context())
    course_payload = contract["course_binding_payload"]
    metadata = contract["safe_metadata"]

    assert metadata["review_status"] == "APPROVED_FOR_LIBRARY"
    assert metadata["approval_record_id"] == "approval:warehouse-selected-1"
    assert course_payload["current_status"] == "APPROVED_FOR_LIBRARY"
    assert course_payload["approval_record_id"] == "approval:warehouse-selected-1"
    assert course_payload["raw_text_policy"] == "POINTER_ONLY"
    assert course_payload["rights_status"] == "INTERNAL"


def test_unknown_rights_hold_without_db_runtime_or_provider_dependency() -> None:
    promotion = _promotion()
    promotion["item"]["rights_status"] = "unknown"

    contract = map_warehouse_promotion_to_bridge_payload(promotion, _safe_context())

    assert contract["result_status"] == RESULT_HOLD
    assert contract["bridge_response"]["result_status"] == RESULT_HOLD
    assert contract["raw_text_included"] is False
    assert contract["internal_path_included"] is False
    assert contract["db_access_executed"] is False
    assert contract["network_access_executed"] is False
    assert contract["runtime_access_executed"] is False
