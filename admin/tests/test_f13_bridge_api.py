import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin.f13_bridge_api as bridge_api


ROUTE = "/api/f13/bridge/retrieve-evidence"
SKILLUP_ROUTE = "/api/f13/bridge/skillup/bridge-answer"
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13" / "bridge_evidence_response.schema.json"
ALLOWED_STATUSES = {"OK", "HOLD", "DENIED"}
FORBIDDEN_KEYS = {
    "raw_text_ref",
    "raw_pointer",
    "raw_source_text",
    "full_source_text",
    "source_uri_or_path",
    "direct_db_row",
    "warehouse_internal_object",
    "library_internal_object",
}


@pytest.fixture
def local_app() -> FastAPI:
    app = FastAPI()
    app.include_router(bridge_api.router)
    return app


@pytest.fixture
def client(local_app: FastAPI) -> TestClient:
    with TestClient(local_app) as test_client:
        yield test_client


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _safe_evidence(**overrides: Any) -> dict[str, Any]:
    evidence = {
        "evidence_id": "ev:t16-1-safe-1",
        "bridge_trace_id": "btrace:t16-1-safe-1",
        "safe_summary": "Synthetic public summary for the no-DB Bridge API route.",
        "pointer_uri": "pointer://f13/t16-1/safe-1",
        "raw_text_policy": "SUMMARY_ONLY",
        "rights_status": "PUBLIC",
    }
    evidence.update(overrides)
    return evidence


def _redacted_preflight_evidence(**overrides: Any) -> dict[str, Any]:
    evidence = {
        "replay_datetime_local": "2026-05-17 KST, exact time not recorded",
        "human_operator_role": "HUMAN_OPERATOR",
        "db_server_label": "local PostgreSQL 18 / f13_readonly_test",
        "db_engine": "PostgreSQL 18",
        "target_database": "quali_journal_f13_dev",
        "read_only_role_observed": "f13_readonly",
        "table_checked": "public.f13_feedback_queue_items",
        "connected_user_result": "f13_readonly",
        "connected_database_result": "quali_journal_f13_dev",
        "table_exists_result": "public.f13_feedback_queue_items observed",
        "select_count_status": "EXECUTED_SUCCESSFULLY_REDACTED_STATUS",
        "can_select": True,
        "can_insert": False,
        "can_update": False,
        "can_delete": False,
        "original_preflight_script_replay_status": (
            "PASS_HUMAN_OPERATOR_READ_ONLY_PREFLIGHT_REPLAY_REDACTED_EVIDENCE"
        ),
        "error_status_if_any": "NONE_OBSERVED",
        "credential_material_recorded": "NO",
        "password_recorded": "NO",
        "full_connection_string_recorded": "NO",
        ".env_accessed": "NO",
        "environment_values_recorded": "NO",
        "secret_store_accessed": "NO",
        "DB_write_attempted": "NO",
        "migration_executed_in_replay": "NO",
        "rollback_executed_in_replay": "NO",
        "operator_final_status": "PASS_HUMAN_OPERATOR_READ_ONLY_PREFLIGHT_REPLAY_REDACTED_EVIDENCE",
    }
    evidence.update(overrides)
    return evidence


def _payload(evidence_items: list[dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
    payload = {
        "query": "synthetic bridge request",
        "purpose": "answer",
        "requester_module": "Skillup",
        "evidence_items": evidence_items if evidence_items is not None else [_safe_evidence()],
    }
    payload.update(overrides)
    return payload


def _walk_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            out.append(str(key))
            out.extend(_walk_values(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_walk_values(child))
        return out
    return [str(value)]


def _assert_no_forbidden_echo(body: dict[str, Any]) -> None:
    haystack = "\n".join(_walk_values(body))
    for forbidden in FORBIDDEN_KEYS:
        assert forbidden not in haystack


def _assert_bridge_shape(body: dict[str, Any]) -> None:
    schema = _schema()
    schema_properties = set(schema["properties"])
    required = set(schema["required"])

    assert set(body).issubset(schema_properties)
    assert required.issubset(body)
    assert body["result_status"] in ALLOWED_STATUSES
    assert isinstance(body["evidence_items"], list)
    assert "hold_reason" in body
    assert isinstance(body["feedback_candidate_required"], bool)
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    assert set(body["policy_result"]) == set(schema["properties"]["policy_result"]["required"])

    item_required = set(schema["properties"]["evidence_items"]["items"]["required"])
    item_properties = set(schema["properties"]["evidence_items"]["items"]["properties"])
    for item in body["evidence_items"]:
        assert item_required.issubset(item)
        assert set(item).issubset(item_properties)


def test_route_exists_and_accepts_post(client: TestClient, local_app: FastAPI):
    assert ROUTE in {getattr(route, "path", "") for route in local_app.routes}

    response = client.post(ROUTE, json=_payload())

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "OK"


def test_ok_response_with_public_summary_only_safe_evidence(client: TestClient):
    response = client.post(ROUTE, json=_payload([_safe_evidence()]))

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "OK"
    assert body["hold_reason"] is None
    assert body["feedback_candidate_required"] is False
    assert len(body["evidence_items"]) == 1
    assert body["evidence_items"][0]["rights_status"] == "PUBLIC"
    assert body["evidence_items"][0]["raw_text_policy"] == "SUMMARY_ONLY"
    _assert_no_forbidden_echo(body)


def test_ok_response_with_canonical_soldering_seed_for_matching_query(client: TestClient):
    response = client.post(
        ROUTE,
        json={"query": "솔더링이란?", "purpose": "answer", "requester_module": "Skillup"},
    )

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "OK"
    assert body["hold_reason"] is None
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    assert len(body["evidence_items"]) == 1
    evidence = body["evidence_items"][0]
    assert evidence["evidence_id"] == "ev-soldering-safe-summary-v1"
    assert evidence["bridge_trace_id"] == "btrace:library-seed:soldering-safe-summary-v1"
    assert "솔더링" in evidence["safe_summary"]
    assert evidence["raw_text_policy"] == "SUMMARY_ONLY"
    assert evidence["rights_status"] == "INTERNAL"
    assert "raw_text_excluded" not in evidence
    assert "standard_raw_text_not_included" not in evidence
    assert "data/library" not in "\n".join(_walk_values(body))
    _assert_no_forbidden_echo(body)


def test_hold_response_for_unknown_query_does_not_invent_seed_answer(client: TestClient):
    response = client.post(
        ROUTE,
        json={"query": "무관한질문", "purpose": "answer", "requester_module": "Skillup"},
    )

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "HOLD"
    assert body["evidence_items"] == []
    assert body["feedback_candidate_required"] is True
    assert "approved Library Evidence seed was not found" in body["hold_reason"]
    assert "솔더링" not in "\n".join(_walk_values(body))
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False


def test_hold_response_for_matching_seed_without_approval(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    unsafe_seed = {
        "evidence_id": "ev-soldering-unapproved",
        "evidence_type": "SAFE_SUMMARY_ONLY",
        "created_for_query_domain": ["솔더링이란?"],
        "safe_summary": "Synthetic safe summary that must not be returned.",
        "pointer_uri": "qlib://library/evidence_seeds/soldering/ev-soldering-unapproved",
        "raw_text_policy": "SAFE_SUMMARY_ONLY",
        "rights_status": "INTERNAL",
        "source_doc_kind": "REFERENCE",
        "review_status": "APPROVED_FOR_LIBRARY_EVIDENCE",
        "approval_status": "NOT_APPROVED",
        "bridge_trace_seed": "btrace-seed-soldering-unapproved",
        "raw_text_excluded": True,
        "standard_raw_text_not_included": True,
        "validation_shape_ids": ["SH-F13-EVIDENCE-001"],
    }
    monkeypatch.setattr(bridge_api, "_load_library_evidence_seed_records", lambda: [unsafe_seed])

    response = client.post(
        ROUTE,
        json={"query": "솔더링이란?", "purpose": "answer", "requester_module": "Skillup"},
    )

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "HOLD"
    assert body["evidence_items"] == []
    assert "not approved" in body["hold_reason"]
    assert "Synthetic safe summary that must not be returned." not in "\n".join(_walk_values(body))


def test_skillup_route_uses_bridge_seed_without_direct_skillup_file_access(client: TestClient):
    response = client.post(
        SKILLUP_ROUTE,
        json={
            "requester_module": "Skillup",
            "ui_mode": "test_minimal",
            "request_payload": {"question": "솔더링이란?"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_status"] == "OK"
    assert body["answer_status"] == "ANSWERED"
    assert body["trace_id"] == "btrace:library-seed:soldering-safe-summary-v1"
    assert body["safe_short_answer"] == body["answer"]
    assert "솔더링" in body["safe_short_answer"]
    assert body["evidence"][0]["evidence_id"] == "ev-soldering-safe-summary-v1"
    assert body["raw_text_included"] is False
    assert body["internal_path_included"] is False
    answer_surface = "\n".join(
        str(value)
        for value in (
            body.get("answer"),
            body.get("safe_short_answer"),
            body.get("hold_reason"),
        )
        if value is not None
    ).lower()
    for forbidden in ("full_json", "full json", "file://", "h:\\", "c:\\", "secret", "token", "credential"):
        assert forbidden not in answer_surface


def test_skillup_seed_adapter_does_not_add_direct_skillup_file_or_db_access():
    import admin.f13_skillup_bridge as skillup_bridge

    forbidden_module_attrs = {
        "Path",
        "open",
        "read_text",
        "glob",
        "rglob",
        "connect",
        "execute",
        "get_session",
        "make_engine",
    }
    forbidden_code_names = forbidden_module_attrs | {"psycopg", "psycopg2", "sqlalchemy"}

    assert forbidden_module_attrs.isdisjoint(skillup_bridge.__dict__)
    for helper in (
        skillup_bridge.skillup_answer_from_bridge_response,
        skillup_bridge.skillup_answer_from_request,
    ):
        assert forbidden_code_names.isdisjoint(set(helper.__code__.co_names))


def test_hold_response_when_evidence_items_missing_or_empty(client: TestClient):
    missing_response = client.post(
        ROUTE,
        json={"query": "synthetic bridge request", "purpose": "answer", "requester_module": "Skillup"},
    )
    empty_response = client.post(ROUTE, json=_payload([]))

    for response in (missing_response, empty_response):
        assert response.status_code == 200
        body = response.json()
        _assert_bridge_shape(body)
        assert body["result_status"] == "HOLD"
        assert body["evidence_items"] == []
        assert body["hold_reason"]
        assert body["feedback_candidate_required"] is True
        assert body["policy_result"]["evidence_required_pass"] is False


def test_hold_response_when_evidence_id_is_missing(client: TestClient):
    response = client.post(ROUTE, json=_payload([_safe_evidence(evidence_id="")]))

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "HOLD"
    assert body["evidence_items"] == []
    assert "missing evidence_id" in body["hold_reason"]
    assert body["feedback_candidate_required"] is True


@pytest.mark.parametrize(
    ("field", "length"),
    [
        ("evidence_id", 121),
        ("bridge_trace_id", 161),
    ],
)
def test_hold_response_when_required_projected_field_exceeds_schema_cap(
    client: TestClient,
    field: str,
    length: int,
):
    response = client.post(ROUTE, json=_payload([_safe_evidence(**{field: field[0] * length})]))

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "HOLD"
    assert body["evidence_items"] == []
    assert "projected evidence is missing Bridge schema required fields" in body["hold_reason"]


def test_optional_source_doc_kind_exceeding_schema_cap_is_not_returned(client: TestClient):
    item_schema = _schema()["properties"]["evidence_items"]["items"]["properties"]
    source_doc_kind_cap = item_schema["source_doc_kind"]["maxLength"]

    response = client.post(
        ROUTE,
        json=_payload([_safe_evidence(source_doc_kind="s" * (source_doc_kind_cap + 1))]),
    )

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "OK"
    assert "source_doc_kind" not in body["evidence_items"][0]


def test_denied_response_for_restricted_rights(client: TestClient):
    response = client.post(ROUTE, json=_payload([_safe_evidence(rights_status="RESTRICTED")]))

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "DENIED"
    assert body["evidence_items"] == []
    assert "RESTRICTED" in body["hold_reason"]
    assert body["feedback_candidate_required"] is True
    assert body["policy_result"]["rights_pass"] is False


@pytest.mark.parametrize("forbidden_field", ["raw_text_ref", "raw_pointer"])
def test_denied_response_for_forbidden_raw_leak_fields_and_no_echo(
    client: TestClient,
    forbidden_field: str,
):
    response = client.post(
        ROUTE,
        json=_payload([_safe_evidence(**{forbidden_field: "synthetic forbidden raw pointer"})]),
    )

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "DENIED"
    assert body["evidence_items"] == []
    assert body["feedback_candidate_required"] is True
    assert body["policy_result"]["raw_leak_pass"] is False
    _assert_no_forbidden_echo(body)


def test_response_status_values_are_limited_to_schema_vocabulary(client: TestClient):
    responses = [
        client.post(ROUTE, json=_payload([_safe_evidence()])),
        client.post(ROUTE, json=_payload([])),
        client.post(ROUTE, json=_payload([_safe_evidence(rights_status="RESTRICTED")])),
    ]

    for response in responses:
        assert response.status_code == 200
        body = response.json()
        _assert_bridge_shape(body)
        assert body["result_status"] in ALLOWED_STATUSES


def test_route_does_not_require_db_access(client: TestClient):
    assert not hasattr(bridge_api, "get_session")
    assert not hasattr(bridge_api, "make_engine")
    assert not hasattr(bridge_api, "Session")

    response = client.post(ROUTE, json=_payload([_safe_evidence()]))

    assert response.status_code == 200
    assert response.json()["result_status"] == "OK"


def test_optional_redacted_preflight_evidence_absent_preserves_bridge_behavior(client: TestClient):
    response = client.post(ROUTE, json=_payload([_safe_evidence()]))

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "OK"
    assert body["feedback_candidate_required"] is False
    assert len(body["evidence_items"]) == 1


def test_accepted_redacted_preflight_evidence_allows_normal_bridge_response(client: TestClient):
    response = client.post(
        ROUTE,
        json=_payload(
            [_safe_evidence()],
            redacted_preflight_replay_evidence=_redacted_preflight_evidence(),
        ),
    )

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "OK"
    assert body["hold_reason"] is None
    assert body["feedback_candidate_required"] is False
    rendered = "\n".join(_walk_values(body)).lower()
    assert "codex live db verification" not in rendered
    assert "production_runtime_readiness" not in rendered
    assert "production/runtime readiness" not in rendered


@pytest.mark.parametrize(
    ("field", "raw_value"),
    [
        ("connected_user_result", "postgres_superuser_raw_value"),
        ("target_database", "production_database_raw_value"),
    ],
)
def test_mismatched_redacted_preflight_evidence_returns_hold_without_raw_echo(
    client: TestClient,
    field: str,
    raw_value: str,
):
    response = client.post(
        ROUTE,
        json=_payload(
            [_safe_evidence()],
            redacted_preflight_replay_evidence=_redacted_preflight_evidence(**{field: raw_value}),
        ),
    )

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "HOLD"
    assert body["evidence_items"] == []
    assert body["feedback_candidate_required"] is True
    assert "redacted preflight replay evidence requires review" in body["hold_reason"]
    assert raw_value not in "\n".join(_walk_values(body))


def test_secret_boundary_redacted_preflight_evidence_returns_denied_without_raw_echo(client: TestClient):
    raw_secret_like_value = "credential-material-value-123"
    response = client.post(
        ROUTE,
        json=_payload(
            [_safe_evidence()],
            redacted_preflight_replay_evidence=_redacted_preflight_evidence(
                credential_material_recorded="YES",
                operator_raw_note=raw_secret_like_value,
            ),
        ),
    )

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "DENIED"
    assert body["evidence_items"] == []
    assert body["feedback_candidate_required"] is True
    assert "redacted preflight replay evidence denied by safety boundary" in body["hold_reason"]
    assert raw_secret_like_value not in "\n".join(_walk_values(body))


def test_write_boundary_redacted_preflight_evidence_returns_denied_and_feedback_required(client: TestClient):
    response = client.post(
        ROUTE,
        json=_payload(
            [_safe_evidence()],
            redacted_preflight_replay_evidence=_redacted_preflight_evidence(DB_write_attempted="YES"),
        ),
    )

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "DENIED"
    assert body["feedback_candidate_required"] is True
    assert "DB_WRITE_ATTEMPTED_POSITIVE" in body["hold_reason"]


def test_preflight_validation_gate_keeps_public_feedback_queue_nonblocking_when_target_table_confirmed(
    client: TestClient,
):
    response = client.post(
        ROUTE,
        json=_payload(
            [_safe_evidence()],
            redacted_preflight_replay_evidence=_redacted_preflight_evidence(
                table_checked="public.feedback_queue null; public.f13_feedback_queue_items confirmed",
            ),
        ),
    )

    assert response.status_code == 200
    body = response.json()
    _assert_bridge_shape(body)
    assert body["result_status"] == "OK"
    assert body["feedback_candidate_required"] is False


def test_preflight_validation_gate_preserves_schema_shape_for_hold_and_denied(client: TestClient):
    hold_response = client.post(
        ROUTE,
        json=_payload(
            [_safe_evidence()],
            redacted_preflight_replay_evidence=_redacted_preflight_evidence(can_insert=True),
        ),
    )
    denied_response = client.post(
        ROUTE,
        json=_payload(
            [_safe_evidence()],
            redacted_preflight_replay_evidence=_redacted_preflight_evidence(rollback_executed_in_replay="YES"),
        ),
    )

    for response in (hold_response, denied_response):
        assert response.status_code == 200
        body = response.json()
        _assert_bridge_shape(body)
        assert set(body).issubset(set(_schema()["properties"]))
        assert body["feedback_candidate_required"] is True


def test_preflight_validation_gate_introduces_no_execution_or_secret_surface():
    forbidden_module_attrs = {
        "get_session",
        "make_engine",
        "Session",
        "psycopg",
        "psycopg2",
        "sqlalchemy",
        "subprocess",
        "requests",
        "httpx",
        "dotenv",
    }
    forbidden_code_names = {
        "connect",
        "execute",
        "run",
        "Popen",
        "subprocess",
        "requests",
        "httpx",
        "environ",
        "getenv",
        "dotenv",
        "open",
        "write",
        "write_text",
        "write_bytes",
    }

    assert forbidden_module_attrs.isdisjoint(bridge_api.__dict__)
    gate_names = set(bridge_api._preflight_validation_gate_response.__code__.co_names)
    route_names = set(bridge_api.retrieve_bridge_evidence.__code__.co_names)
    assert forbidden_code_names.isdisjoint(gate_names)
    assert forbidden_code_names.isdisjoint(route_names)
