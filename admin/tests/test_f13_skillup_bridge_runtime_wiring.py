from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin.f13_bridge_api as bridge_api
from admin.server_quali import app as full_app


ROUTE = "/api/f13/bridge/skillup/bridge-answer"
TEST_AUTH_ENV_KEY = "ADMIN_TOKEN"
TEST_AUTH_ALT_ENV_KEY = "API_TOKEN"
TEST_AUTH_HEADER = "X-Admin-Token"
TEST_AUTH_PLACEHOLDER = "test-scope-placeholder-auth"

_SCHEMA_REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "contract_version",
    "trace_id",
    "answer_status",
    "result_status",
    "evidence_required",
    "evidence",
    "policy",
    "raw_text_included",
    "internal_path_included",
    "review_required",
}
_SCHEMA_ALLOWED_TOP_LEVEL_FIELDS = _SCHEMA_REQUIRED_TOP_LEVEL_FIELDS | {
    "request_id",
    "course_id",
    "module_id",
    "binding_id",
    "answer",
    "safe_short_answer",
    "hold_reason_code",
    "hold_reason",
    "warnings",
}
_POLICY_FIELDS = {
    "raw_leak_check_passed",
    "rights_check_passed",
    "sensitivity_check_passed",
    "evidence_check_passed",
}
_LEGACY_SELECTED_ROUTE_TOP_LEVEL_FIELDS = {
    "safe_summary",
    "evidence_id",
    "bridge_trace_id",
    "feedback_queue_item",
    "feedback_candidate",
    "feedback_candidate_required",
    "created_at",
    "db_access_executed",
    "pointer_uri",
}


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(bridge_api.router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def full_app_client() -> TestClient:
    with TestClient(full_app) as test_client:
        yield test_client


def _configure_expected_test_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TEST_AUTH_ENV_KEY, TEST_AUTH_PLACEHOLDER)
    monkeypatch.delenv(TEST_AUTH_ALT_ENV_KEY, raising=False)


def _test_auth_headers(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    _configure_expected_test_auth(monkeypatch)
    return {TEST_AUTH_HEADER: TEST_AUTH_PLACEHOLDER}


def _safe_evidence(**overrides: Any) -> dict[str, Any]:
    evidence = {
        "evidence_id": "ev:skillup-bridge-safe-1",
        "bridge_trace_id": "btrace:skillup-bridge-safe-1",
        "safe_summary": "Synthetic safe summary for Skillup route wiring.",
        "pointer_uri": "pointer://diagnostic/skillup-route/safe-1",
        "raw_text_policy": "SUMMARY_ONLY",
        "rights_status": "PUBLIC",
        "request_id": "req:skillup-route-safe-1",
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


def _assert_schema_shaped_response(body: dict[str, Any]) -> None:
    assert _SCHEMA_REQUIRED_TOP_LEVEL_FIELDS <= set(body)
    assert set(body) <= _SCHEMA_ALLOWED_TOP_LEVEL_FIELDS
    assert not (_LEGACY_SELECTED_ROUTE_TOP_LEVEL_FIELDS & set(body))
    assert isinstance(body["trace_id"], str)
    assert body["trace_id"]
    assert isinstance(body["evidence"], list)
    assert isinstance(body["policy"], dict)
    assert set(body["policy"]) == _POLICY_FIELDS
    assert all(isinstance(value, bool) for value in body["policy"].values())
    assert isinstance(body.get("warnings", []), list)
    assert all(isinstance(value, str) for value in body.get("warnings", []))
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False


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


def _assert_no_forbidden_reason_label_tokens(*values: Any) -> None:
    rendered = "\n".join(str(value) for value in values if value is not None).lower()
    for token in (
        "raw_text",
        "raw text",
        "raw_query",
        "raw query",
        "internal_path",
        "internal path",
        "api_token",
        "secret",
        "credential",
        ".env",
        "h:\\",
        "c:\\",
        "file://",
    ):
        assert token not in rendered


def test_skillup_bridge_route_hold_returns_schema_shaped_review_response(client: TestClient):
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
    _assert_schema_shaped_response(body)
    assert body["result_status"] == "HOLD"
    assert body["answer_status"] == "HOLD"
    assert body["evidence_required"] is True
    assert body["review_required"] is True
    assert body["evidence"] == []
    assert body["hold_reason_code"] == "EVIDENCE_REQUIRED"
    assert "evidence_items" in body["hold_reason"]
    _assert_no_forbidden_reason_label_tokens(body["hold_reason_code"], body["hold_reason"])
    assert body["policy"]["raw_leak_check_passed"] is True
    assert body["policy"]["evidence_check_passed"] is False
    assert "answer" not in body
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    _assert_no_pass_fields(body)
    _assert_no_raw_internal_or_secret_echo(body)


def test_skillup_bridge_full_app_route_requires_auth_without_token(
    monkeypatch: pytest.MonkeyPatch,
    full_app_client: TestClient,
) -> None:
    _configure_expected_test_auth(monkeypatch)

    response = full_app_client.post(
        ROUTE,
        json={
            "result_status": "HOLD",
            "evidence_items": [],
            "hold_reason": "selected full-app auth boundary check",
            "feedback_candidate_required": False,
            "raw_text_included": False,
            "internal_path_included": False,
        },
    )

    assert response.status_code == 401


def test_skillup_bridge_full_app_route_ok_uses_schema_answer_evidence_and_trace_with_test_auth(
    monkeypatch: pytest.MonkeyPatch,
    full_app_client: TestClient,
) -> None:
    payload = {
        "request_id": "req:skillup-full-app-selected-ok",
        "result_status": "OK",
        "evidence_items": [_safe_evidence()],
        "feedback_candidate_required": False,
        "raw_text_included": False,
        "internal_path_included": False,
    }

    response = full_app_client.post(
        ROUTE,
        headers=_test_auth_headers(monkeypatch),
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()
    _assert_schema_shaped_response(body)
    assert body["result_status"] == "OK"
    assert body["answer_status"] == "ANSWERED"
    assert body["trace_id"] == "btrace:skillup-bridge-safe-1"
    assert isinstance(body["answer"], str)
    assert body["evidence"] == [
        {
            "evidence_id": "ev:skillup-bridge-safe-1",
            "pointer": "pointer://diagnostic/skillup-route/safe-1",
            "source_label": "Skillup Bridge safe evidence",
            "rights_status": "PUBLIC",
        }
    ]
    assert body["policy"] == {
        "raw_leak_check_passed": True,
        "rights_check_passed": True,
        "sensitivity_check_passed": True,
        "evidence_check_passed": True,
    }
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    _assert_no_pass_fields(body)
    _assert_no_raw_internal_or_secret_echo(body)


def test_skillup_bridge_full_app_route_sanitizes_unsafe_source_content_with_test_auth(
    monkeypatch: pytest.MonkeyPatch,
    full_app_client: TestClient,
) -> None:
    response = full_app_client.post(
        ROUTE,
        headers=_test_auth_headers(monkeypatch),
        json={
            "requester_module": "full-app-selected-sanitization",
            "result_status": "OK",
            "evidence_items": [
                _safe_evidence(
                    safe_summary="unsafe source content withheld",
                    pointer_uri="withheld-by-policy",
                    source_label="withheld-source-label",
                    secret=True,
                )
            ],
            "raw_text_included": True,
            "internal_path_included": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    _assert_schema_shaped_response(body)
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    _assert_no_forbidden_reason_label_tokens(body["hold_reason_code"], body["hold_reason"])
    _assert_no_pass_fields(body)
    _assert_no_raw_internal_or_secret_echo(body)


def test_skillup_bridge_full_app_route_direct_db_attempt_denied_without_db_with_test_auth(
    monkeypatch: pytest.MonkeyPatch,
    full_app_client: TestClient,
) -> None:
    response = full_app_client.post(
        ROUTE,
        headers=_test_auth_headers(monkeypatch),
        json={
            "requester_module": "full-app-selected-db-denial",
            "direct_db_access_attempt": True,
            "raw_query": True,
            "internal_path": True,
            "api_token": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    _assert_schema_shaped_response(body)
    assert body["result_status"] == "ERROR"
    assert body["answer_status"] == "INVALIDATED"
    assert body["evidence_required"] is True
    assert body["review_required"] is True
    assert body["evidence"] == []
    _assert_no_forbidden_reason_label_tokens(body["hold_reason_code"], body["hold_reason"])
    assert "SOURCE_DENIED_NORMALIZED_TO_ERROR" in body.get("warnings", [])
    assert body["policy"] == {
        "raw_leak_check_passed": False,
        "rights_check_passed": False,
        "sensitivity_check_passed": True,
        "evidence_check_passed": False,
    }
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    _assert_no_pass_fields(body)
    _assert_no_raw_internal_or_secret_echo(body)


def test_skillup_bridge_route_ok_uses_schema_answer_evidence_and_trace(client: TestClient):
    response = client.post(
        ROUTE,
        json={
            "request_id": "req:skillup-route-safe-1",
            "result_status": "OK",
            "evidence_items": [_safe_evidence()],
            "feedback_candidate_required": False,
            "raw_text_included": False,
            "internal_path_included": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    _assert_schema_shaped_response(body)
    assert body["result_status"] == "OK"
    assert body["answer_status"] == "ANSWERED"
    assert body["trace_id"] == "btrace:skillup-bridge-safe-1"
    assert body["request_id"] == "req:skillup-route-safe-1"
    assert body["course_id"] == "course:skillup-route"
    assert body["module_id"] == "module:skillup-route"
    assert body["binding_id"] == "binding:skillup-route"
    assert body["answer"] == "Synthetic safe summary for Skillup route wiring."
    assert body["evidence_required"] is False
    assert body["review_required"] is False
    assert body["evidence"] == [
        {
            "evidence_id": "ev:skillup-bridge-safe-1",
            "pointer": "pointer://diagnostic/skillup-route/safe-1",
            "source_label": "Skillup Bridge safe evidence",
            "rights_status": "PUBLIC",
        }
    ]
    assert body["policy"] == {
        "raw_leak_check_passed": True,
        "rights_check_passed": True,
        "sensitivity_check_passed": True,
        "evidence_check_passed": True,
    }
    assert body.get("warnings", []) == []
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    _assert_no_pass_fields(body)
    _assert_no_raw_internal_or_secret_echo(body)


def test_skillup_bridge_route_sanitizes_unsafe_source_content_reason_labels(client: TestClient):
    response = client.post(
        ROUTE,
        json={
            "requester_module": "Skillup",
            "result_status": "OK",
            "evidence_items": [
                _safe_evidence(
                    safe_summary="synthetic raw_text marker should not echo",
                    pointer_uri="file://synthetic/internal/source.txt",
                    source_label="synthetic credential marker should not echo",
                    secret="synthetic-secret-marker",
                )
            ],
            "raw_text_included": True,
            "internal_path_included": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    _assert_schema_shaped_response(body)
    assert body["result_status"] == "ERROR"
    assert body["answer_status"] == "INVALIDATED"
    assert body["evidence_required"] is True
    assert body["review_required"] is True
    assert body["hold_reason_code"] == "SOURCE_CONTENT_BLOCKED"
    assert body["hold_reason"] == "Unsafe source content was blocked."
    _assert_no_forbidden_reason_label_tokens(body["hold_reason_code"], body["hold_reason"])
    assert "SOURCE_DENIED_NORMALIZED_TO_ERROR" in body.get("warnings", [])
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
    _assert_schema_shaped_response(body)
    assert body["result_status"] == "ERROR"
    assert body["answer_status"] == "INVALIDATED"
    assert body["evidence_required"] is True
    assert body["review_required"] is True
    assert body["evidence"] == []
    assert body["hold_reason_code"] == "DENIED_POLICY_BOUNDARY"
    assert body["hold_reason"] == "forbidden fields or patterns detected"
    _assert_no_forbidden_reason_label_tokens(body["hold_reason_code"], body["hold_reason"])
    assert "SOURCE_DENIED_NORMALIZED_TO_ERROR" in body.get("warnings", [])
    assert body["policy"] == {
        "raw_leak_check_passed": False,
        "rights_check_passed": False,
        "sensitivity_check_passed": True,
        "evidence_check_passed": False,
    }
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    _assert_no_pass_fields(body)
    _assert_no_raw_internal_or_secret_echo(body)
