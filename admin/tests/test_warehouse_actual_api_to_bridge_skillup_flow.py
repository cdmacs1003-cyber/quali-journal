from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

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
from admin.warehouse_bridge_contract_adapter import build_bridge_contract_from_warehouse_promotion
from server_quali import app, authorize


def _context() -> dict[str, Any]:
    return {
        "tenant_id": "tenant:warehouse-actual-api",
        "organization_id": "org:warehouse-actual-api",
        "cohort_id": "cohort:warehouse-actual-api",
        "course_id": "course:warehouse-actual-api",
        "module_id": "module:warehouse-actual-api",
        "binding_id": "binding:warehouse-actual-api",
        "role": "student",
        "evidence_depth": "student_safe",
        "bridge_family": "warehouse",
        "bridge_id": "bridge:warehouse-actual-api",
        "standard_pack_id": "SPK_WAREHOUSE_ACTUAL_API",
        "request_id": "req:warehouse-actual-api",
        "validation_shape_ids": ["SH-F13-CURATION-001"],
    }


@pytest.fixture
def warehouse_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    for name in list(os.environ):
        if name.startswith("QUALI_") or name == "LTM_ROOT":
            monkeypatch.delenv(name, raising=False)

    process_tmp = tmp_path.parent.parent / "process_tmp"
    process_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("QUALIJOURNAL_SKIP_DOTENV", "1")
    monkeypatch.setenv("TEMP", str(process_tmp))
    monkeypatch.setenv("TMP", str(process_tmp))
    monkeypatch.setenv("QUALI_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("QUALI_WAREHOUSE_ROOT", str(tmp_path / "data" / "warehouse"))
    monkeypatch.setenv("QUALI_LIBRARY_ROOT", str(tmp_path / "data" / "library"))
    monkeypatch.setenv("QUALI_WAREHOUSE_BACKUP_ROOT", str(tmp_path / "backup" / "warehouse"))
    monkeypatch.setenv("QUALI_WAREHOUSE_PROOFPACK_ROOT", str(tmp_path / "reports" / "proofpacks" / "warehouse"))
    monkeypatch.setenv("QUALI_WAREHOUSE_RELEASE_ROOT", str(tmp_path / "releases" / "warehouse"))

    async def _ok() -> bool:
        return True

    app.dependency_overrides[authorize] = _ok
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(authorize, None)


def _create_actual_item(client: TestClient) -> str:
    raw_text = (
        "Warehouse actual API sample summary. "
        "raw source text and raw standard text are fixture-only sentinels."
    )
    resp = client.post(
        "/api/warehouse/items",
        json={
            "item_type": "expert_knowhow",
            "title": "Warehouse actual API bridge Skillup sample",
            "summary": "Safe actual Warehouse API summary for Bridge and Skillup flow tests.",
            "raw_text": raw_text,
            "raw_mime_type": "text/plain",
            "provenance": {
                "source_type": "expert",
                "source_title": "Internal actual API field note",
                "source_author": "reviewer-a",
                "source_org": "Quali",
                "source_date": "2026-07-04",
                "captured_by": "capturer-a",
                "source_locator": "internal://warehouse-actual-api/001",
            },
            "rights_status": "owned",
            "sensitivity": "internal",
            "visibility": "library_candidate",
            "tags": ["warehouse", "bridge", "skillup"],
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["item"]["raw_hash"].startswith("sha256:")
    assert "raw_text_ref" in body["item"]
    return body["item"]["warehouse_item_id"]


def _move_to_pending_review(client: TestClient, item_id: str) -> None:
    for status in ("untriaged", "triaged", "pending_review"):
        resp = client.patch(
            f"/api/warehouse/items/{item_id}/status",
            json={"status": status, "actor_id": "tester", "reason": "actual-output selected flow"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["item"]["status"] == status


def _review_approve_and_promote(client: TestClient, item_id: str) -> dict[str, Any]:
    _move_to_pending_review(client, item_id)
    review = client.post(
        f"/api/warehouse/items/{item_id}/reviews",
        json={
            "reviewer_id": "subject-reviewer-1",
            "reviewer_role": "Subject Reviewer",
            "review_decision": "approved_for_library",
            "review_note": "Actual API output selected flow reviewed.",
            "quality_score": 88,
            "confidence_score": 0.92,
            "rights_status_confirmed": True,
            "sensitivity_confirmed": True,
            "promotion_recommendation": "library_reference_card",
        },
    )
    assert review.status_code == 200, review.text

    approval = client.post(
        f"/api/warehouse/items/{item_id}/approve",
        json={"approver_id": "approver-1", "approval_note": "Approved for actual-output flow test."},
    )
    assert approval.status_code == 200, approval.text
    assert approval.json()["item"]["status"] == "approved_for_library"

    dry_run = client.post(
        f"/api/warehouse/items/{item_id}/promotion-dry-run",
        json={"promotion_target": "library_reference_card", "created_by": "librarian-1"},
    )
    assert dry_run.status_code == 200, dry_run.text
    dry_body = dry_run.json()
    assert dry_body["ok"] is True
    assert dry_body["dry_run"]["decision"] == "PASS"
    assert dry_body["dry_run"]["library_engine"]["enabled"] is False

    promoted = client.post(
        f"/api/warehouse/items/{item_id}/promote",
        json={"promotion_target": "library_reference_card", "created_by": "librarian-1"},
    )
    assert promoted.status_code == 200, promoted.text
    promoted_body = promoted.json()
    assert promoted_body["ok"] is True
    assert promoted_body["item"]["status"] == "promoted"

    trace_id = promoted_body["trace"]["promotion_trace_id"]
    trace = client.get(f"/api/warehouse/traces/{trace_id}")
    assert trace.status_code == 200, trace.text
    assert trace.json()["trace"]["warehouse_item_id"] == item_id
    assert trace.json()["trace"] == promoted_body["trace"]
    return promoted_body


def _actual_contract(client: TestClient) -> dict[str, Any]:
    item_id = _create_actual_item(client)
    promoted_body = _review_approve_and_promote(client, item_id)
    return build_bridge_contract_from_warehouse_promotion(promoted_body, _context())


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


def _render(value: Any) -> str:
    return "\n".join(_walk(value)).lower()


def _assert_no_raw_internal_leak(value: Any) -> None:
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


def _assert_public_payload_safe(value: Any) -> None:
    _assert_no_raw_internal_leak(value)
    _assert_no_secret_like_value_leak(value)


def _feedback_payload(contract: dict[str, Any], skillup_result: dict[str, Any]) -> dict[str, Any]:
    evidence = contract["bridge_evidence_item"]
    context = _context()
    return {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "feedback_id": "FBQ-warehouse-actual-api-1",
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


def test_actual_warehouse_api_output_feeds_bridge_skillup_course_feedback_flow(
    warehouse_client: TestClient,
) -> None:
    contract = _actual_contract(warehouse_client)
    evidence = contract["bridge_evidence_item"]
    projected = project_bridge_safe_evidence(evidence)
    bridge_decision = decide_bridge_result(evidence)
    skillup = skillup_answer_from_bridge_response(contract["skillup_bridge_response"])
    adapted = adapt_skillup_answer_hold_response(
        skillup,
        request_context=_context(),
        bridge_payload=contract["skillup_bridge_response"],
    )
    binding = bind_course_library_reference(contract["course_binding_payload"])
    feedback_payload = _feedback_payload(contract, skillup)
    feedback_result = validate_feedback_queue_contract(feedback_payload)

    assert contract["result_status"] == RESULT_OK
    assert contract["safe_metadata"]["warehouse_item_id"].startswith("WHI-")
    assert contract["safe_metadata"]["promotion_trace_id"].startswith("PTR-")
    assert contract["safe_metadata"]["promoted_library_id"].startswith("LIB-")
    assert contract["safe_metadata"]["evidence_id"].startswith("EVD-")
    assert contract["safe_metadata"]["review_status"] == "APPROVED_FOR_LIBRARY"
    assert projected == evidence
    assert detect_forbidden_fields(evidence) == []
    assert bridge_decision["result_status"] == RESULT_OK

    assert skillup["result_status"] == RESULT_OK
    assert skillup["answer_status"] == ANSWER_STATUS_ANSWERED
    assert skillup["bridge_trace_id"] == contract["bridge_trace_id"]
    assert adapted["result_status"] == RESULT_OK
    assert adapted["answer_status"] == ANSWER_STATUS_ANSWERED

    assert binding["binding_status"] == RESULT_BOUND
    assert binding["skillup_use_allowed"] is True
    assert binding["tenant_id"] == _context()["tenant_id"]
    assert binding["course_id"] == _context()["course_id"]
    assert binding["module_id"] == _context()["module_id"]
    assert binding["evidence_id"] == contract["safe_metadata"]["evidence_id"]
    assert binding["bridge_trace_id"] == contract["bridge_trace_id"]

    assert feedback_result["status"] == RESULT_READY
    assert feedback_result["queue_ready"] is True
    assert feedback_result["db_access_executed"] is False
    assert feedback_result["network_access_executed"] is False
    assert feedback_result["runtime_access_executed"] is False
    assert feedback_result["file_io_executed"] is False

    public_surfaces = {
        "bridge_response": contract["bridge_response"],
        "skillup_bridge_response": contract["skillup_bridge_response"],
        "skillup_answer": skillup,
        "answer_hold_contract": adapted,
        "course_binding": binding,
        "feedback_surface": feedback_payload["feedback_surface"],
        "safe_metadata": contract["safe_metadata"],
    }
    _assert_public_payload_safe(public_surfaces)
    assert "raw_text_ref" not in json.dumps(public_surfaces, sort_keys=True)


def test_actual_warehouse_api_output_missing_context_remains_hold(
    warehouse_client: TestClient,
) -> None:
    item_id = _create_actual_item(warehouse_client)
    promoted_body = _review_approve_and_promote(warehouse_client, item_id)
    contract = build_bridge_contract_from_warehouse_promotion(
        promoted_body,
        {"course_id": "course:warehouse-actual-api"},
    )
    skillup = skillup_answer_from_bridge_response(contract["skillup_bridge_response"])
    adapted = adapt_skillup_answer_hold_response(
        skillup,
        request_context={"course_id": "course:warehouse-actual-api"},
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
    assert contract["raw_text_included"] is False
    assert contract["internal_path_included"] is False

    public_surfaces = {
        "bridge_response": contract["bridge_response"],
        "skillup_bridge_response": contract["skillup_bridge_response"],
        "skillup_answer": skillup,
        "answer_hold_contract": adapted,
        "safe_metadata": contract["safe_metadata"],
    }
    _assert_public_payload_safe(public_surfaces)
