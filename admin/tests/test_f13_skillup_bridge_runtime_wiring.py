from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin.f13_bridge_api as bridge_api


ROUTE = "/api/f13/bridge/skillup/bridge-answer"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(bridge_api.router)
    with TestClient(app) as test_client:
        yield test_client


def _safe_evidence(**overrides: Any) -> dict[str, Any]:
    evidence = {
        "evidence_id": "ev:skillup-bridge-safe-1",
        "bridge_trace_id": "btrace:skillup-bridge-safe-1",
        "safe_summary": "Synthetic safe summary for Skillup route wiring.",
        "pointer_uri": "pointer://diagnostic/skillup-route/safe-1",
        "raw_text_policy": "SUMMARY_ONLY",
        "rights_status": "PUBLIC",
        "role": "student",
        "evidence_depth": "student_safe",
        "course_id": "course:skillup-route",
        "module_id": "module:skillup-route",
        "binding_id": "binding:skillup-route",
        "tenant_id": "tenant:skillup",
        "organization_id": "org:skillup",
        "cohort_id": "cohort:skillup",
    }
    evidence.update(overrides)
    return evidence


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


def _assert_no_pass_fields(body: dict[str, Any]) -> None:
    assert "f13_pass" not in body
    assert "track_a_pass" not in body
    assert "beta_pass" not in body
    if isinstance(body.get("feedback_queue_item"), dict):
        assert "f13_pass" not in body["feedback_queue_item"]
        assert "track_a_pass" not in body["feedback_queue_item"]
        assert "beta_pass" not in body["feedback_queue_item"]


def _assert_no_raw_internal_or_secret_echo(body: dict[str, Any]) -> None:
    allowed_counter_keys = {
        "raw_text_export_count",
        "internal_path_leak_count",
        "raw_prompt_output_count",
        "secret_leak_count",
        "instructor_guide_raw_leak_count",
    }
    rendered = "\n".join(
        item for item in _walk(body) if item not in allowed_counter_keys
    ).lower()
    assert "raw_prompt" not in rendered
    assert "raw_query" not in rendered
    assert "raw_source_text" not in rendered
    assert "full_source_text" not in rendered
    assert "source_uri_or_path" not in rendered
    assert "internal_path" not in (body.get("feedback_candidate") or {})
    assert "raw_text" not in (body.get("feedback_candidate") or {})
    assert "h:\\" not in rendered
    assert "c:\\" not in rendered
    assert "file://" not in rendered
    assert "secret" not in rendered
    assert "token" not in rendered


def test_skillup_bridge_route_hold_returns_feedback_queue_item(client: TestClient):
    response = client.post(
        ROUTE,
        json={
            "result_status": "HOLD",
            "evidence_items": [],
            "hold_reason": "evidence_items are required for no-DB Bridge evaluation",
            "feedback_candidate_required": True,
            "raw_text_included": False,
            "internal_path_included": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    queue_item = body["feedback_queue_item"]
    assert body["result_status"] == "HOLD"
    assert body["answer_status"] == "HOLD"
    assert body["feedback_candidate_required"] is True
    assert queue_item["feedback_id"]
    assert queue_item["dedup_key"]
    assert queue_item["feedback_type"] in {"EVIDENCE_GAP", "HOLD_CASE"}
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    assert queue_item["raw_text_included"] is False
    assert queue_item["internal_path_included"] is False
    _assert_no_pass_fields(body)
    _assert_no_raw_internal_or_secret_echo(body)


def test_skillup_bridge_route_ok_uses_safe_summary_and_trace(client: TestClient):
    response = client.post(
        ROUTE,
        json={
            "result_status": "OK",
            "evidence_items": [_safe_evidence()],
            "feedback_candidate_required": False,
            "raw_text_included": False,
            "internal_path_included": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_status"] == "OK"
    assert body["answer_status"] == "ANSWERED"
    assert body["evidence_id"] == "ev:skillup-bridge-safe-1"
    assert body["bridge_trace_id"] == "btrace:skillup-bridge-safe-1"
    assert body["safe_summary"] == "Synthetic safe summary for Skillup route wiring."
    assert body["answer"] == body["safe_summary"]
    assert body["pointer_uri"] == "pointer://diagnostic/skillup-route/safe-1"
    assert "feedback_queue_item" not in body
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    _assert_no_pass_fields(body)
    _assert_no_raw_internal_or_secret_echo(body)


def test_skillup_bridge_route_direct_db_attempt_denied_without_db(client: TestClient):
    response = client.post(
        ROUTE,
        json={
            "requester_module": "Skillup",
            "direct_db_access_attempt": True,
            "raw_query": "do not echo this query",
            "internal_path": "H:\\secret\\source.txt",
            "api_token": "do-not-echo",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_status"] in {"DENIED", "HOLD"}
    assert body["answer_status"] in {"DENIED", "HOLD"}
    assert body["db_access_executed"] is False
    assert body["feedback_candidate_required"] is True
    assert body["feedback_queue_item"]["feedback_id"]
    assert body["feedback_queue_item"]["dedup_key"]
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    _assert_no_pass_fields(body)
    _assert_no_raw_internal_or_secret_echo(body)
