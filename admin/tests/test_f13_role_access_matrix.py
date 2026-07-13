from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin.f13_bridge_api as bridge_api
from admin.f13_course_library_binding import bind_course_library_reference
from admin.f13_skillup_bridge import skillup_answer_from_bridge_response


CHECK_POLICY_ROUTE = "/api/f13/bridge/check-policy"
EXPLAIN_TRACE_ROUTE = "/api/f13/bridge/explain-trace"

ROLE_DEPTH = {
    "student": "student_safe",
    "learner": "student_safe",
    "instructor": "instructor_safe",
    "reviewer": "review_trace_safe_metadata",
    "admin": "audit_trace_safe_metadata",
}


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(bridge_api.router)
    with TestClient(app) as test_client:
        yield test_client


def _scope(**overrides: Any) -> dict[str, Any]:
    data = {
        "course_id": "course:role-matrix",
        "module_id": "module:role-matrix",
        "binding_id": "binding:role-matrix",
        "bridge_family": "SKILLUP",
        "bridge_id": "bridge:role-matrix",
        "standard_pack_id": "pack:role-matrix",
        "tenant_id": "tenant:role-matrix",
        "organization_id": "org:role-matrix",
        "cohort_id": "cohort:role-matrix",
    }
    data.update(overrides)
    return data


def _policy_payload(role: str | None = "student", **overrides: Any) -> dict[str, Any]:
    data = {
        **_scope(),
        "evidence_id": "ev:role-matrix",
        "bridge_trace_id": "btrace:role-matrix",
        "safe_summary": "Synthetic student safe summary for role matrix testing.",
        "pointer_uri": "pointer://role-matrix/safe",
        "raw_text_policy": "SUMMARY_ONLY",
        "rights_status": "PUBLIC",
        "requested_output_type": "safe_summary",
        "purpose": "answer",
        "requester_module": "Skillup",
    }
    if role is not None:
        data["role"] = role
        data["evidence_depth"] = ROLE_DEPTH.get(role, "student_safe")
    data.update(overrides)
    return data


def _safe_bridge_response(role: str = "student", **overrides: Any) -> dict[str, Any]:
    evidence = {
        "evidence_id": "ev:skillup-role-matrix",
        "bridge_trace_id": "btrace:skillup-role-matrix",
        "safe_summary": "Synthetic safe Skillup answer.",
        "pointer_uri": "pointer://role-matrix/skillup",
        "raw_text_policy": "SUMMARY_ONLY",
        "rights_status": "PUBLIC",
        "role": role,
        "evidence_depth": ROLE_DEPTH[role],
        **_scope(),
    }
    response = {
        "result_status": "OK",
        "evidence_items": [evidence],
        "hold_reason": None,
        "feedback_candidate_required": False,
        "raw_text_included": False,
        "internal_path_included": False,
    }
    response.update(overrides)
    return response


def _assert_zero_leak_counters(body: dict[str, Any]) -> None:
    assert body["raw_text_export_count"] == 0
    assert body["internal_path_leak_count"] == 0
    assert body["raw_prompt_output_count"] == 0
    assert body["secret_leak_count"] == 0
    assert body["instructor_guide_raw_leak_count"] == 0


def _assert_no_forbidden_surface(body: dict[str, Any]) -> None:
    rendered = repr(body).lower()
    assert "raw standard text" not in rendered
    assert "raw_prompt_value" not in rendered
    assert "h:\\" not in rendered
    assert "file://" not in rendered
    assert "api_key_value" not in rendered


def test_missing_or_unknown_role_fails_closed_for_protected_check_policy(client: TestClient):
    missing = client.post(CHECK_POLICY_ROUTE, json=_policy_payload(role=None))
    unknown = client.post(CHECK_POLICY_ROUTE, json=_policy_payload(role="guest"))

    for response in (missing, unknown):
        assert response.status_code == 200
        body = response.json()
        assert body["result_status"] == "HOLD"
        assert body["policy_result"] == "HOLD"
        assert "HOLD_PERMISSION" in body["hold_reason"]
        assert body["role"] is None
        _assert_zero_leak_counters(body)


@pytest.mark.parametrize("missing_field", ["course_id", "module_id", "binding_id"])
def test_missing_binding_or_course_scope_holds_no_binding(client: TestClient, missing_field: str):
    payload = _policy_payload()
    payload.pop(missing_field)

    response = client.post(CHECK_POLICY_ROUTE, json=payload)

    body = response.json()
    assert body["result_status"] == "HOLD"
    assert "HOLD_NO_BINDING" in body["hold_reason"]
    assert body["feedback_candidate_required"] is True
    _assert_zero_leak_counters(body)


@pytest.mark.parametrize(
    ("role", "expected_depth"),
    [
        ("student", "student_safe"),
        ("learner", "student_safe"),
        ("instructor", "instructor_safe"),
        ("reviewer", "review_trace_safe_metadata"),
        ("admin", "audit_trace_safe_metadata"),
    ],
)
def test_evidence_depth_is_mapped_by_role(client: TestClient, role: str, expected_depth: str):
    response = client.post(CHECK_POLICY_ROUTE, json=_policy_payload(role=role, evidence_depth=expected_depth))

    body = response.json()
    assert body["result_status"] == "OK"
    assert body["role"] == ("student" if role == "learner" else role)
    assert body["evidence_depth"] == expected_depth
    _assert_zero_leak_counters(body)


def test_unsupported_evidence_depth_holds_policy(client: TestClient):
    response = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="student", evidence_depth="raw_full_text"),
    )

    body = response.json()
    assert body["result_status"] == "HOLD"
    assert "HOLD_POLICY" in body["hold_reason"]
    _assert_zero_leak_counters(body)


def test_student_safe_summary_allowed_but_raw_and_trace_surfaces_blocked(client: TestClient):
    safe = client.post(CHECK_POLICY_ROUTE, json=_policy_payload(role="student"))
    raw = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="student", requested_output_type="raw_standard_text_export"),
    )
    review_trace = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="student", requested_output_type="review_trace_safe_metadata"),
    )

    assert safe.json()["result_status"] == "OK"
    assert raw.json()["result_status"] == "DENIED"
    assert "HOLD_POLICY" in raw.json()["hold_reason"]
    assert review_trace.json()["result_status"] == "HOLD"
    assert "HOLD_PERMISSION" in review_trace.json()["hold_reason"]
    _assert_zero_leak_counters(raw.json())


def test_instructor_safe_view_allowed_and_raw_exports_blocked(client: TestClient):
    guide_summary = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="instructor", requested_output_type="instructor_guide_summary"),
    )
    raw_standard = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="instructor", requested_output_type="raw_standard_text_export"),
    )
    raw_guide = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="instructor", requested_output_type="raw_instructor_guide_export"),
    )

    assert guide_summary.json()["result_status"] == "OK"
    assert guide_summary.json()["evidence_depth"] == "instructor_safe"
    assert raw_standard.json()["result_status"] == "DENIED"
    assert raw_guide.json()["result_status"] == "DENIED"
    _assert_zero_leak_counters(raw_guide.json())


def test_reviewer_review_trace_metadata_allowed_and_raw_export_blocked(client: TestClient):
    review = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="reviewer", requested_output_type="review_trace_safe_metadata"),
    )
    raw = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="reviewer", requested_output_type="raw_export"),
    )

    assert review.json()["result_status"] == "OK"
    assert review.json()["evidence_depth"] == "review_trace_safe_metadata"
    assert raw.json()["result_status"] == "DENIED"
    _assert_zero_leak_counters(raw.json())


def test_admin_safe_metadata_allowed_and_sensitive_outputs_blocked(client: TestClient):
    audit = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="admin", requested_output_type="audit_trace_safe_metadata"),
    )
    raw_prompt = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="admin", requested_output_type="raw_prompt"),
    )
    private_tacit = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="admin", requested_output_type="private_tacit_knowledge"),
    )

    assert audit.json()["result_status"] == "OK"
    assert audit.json()["evidence_depth"] == "audit_trace_safe_metadata"
    assert raw_prompt.json()["result_status"] == "DENIED"
    assert private_tacit.json()["result_status"] == "DENIED"
    _assert_zero_leak_counters(raw_prompt.json())


def test_explain_trace_safe_review_and_admin_audit_metadata(client: TestClient):
    reviewer = client.post(
        EXPLAIN_TRACE_ROUTE,
        json={
            **_policy_payload(role="reviewer", requested_output_type="review_trace_safe_metadata"),
            "trace": {
                **_scope(),
                "bridge_trace_id": "btrace:role-matrix",
                "evidence_ids": ["ev:role-matrix"],
            },
        },
    ).json()
    admin = client.post(
        EXPLAIN_TRACE_ROUTE,
        json={
            **_policy_payload(role="admin", requested_output_type="audit_trace_safe_metadata"),
            "trace": {
                **_scope(),
                "bridge_trace_id": "btrace:role-matrix",
                "evidence_ids": ["ev:role-matrix"],
            },
        },
    ).json()

    assert reviewer["result_status"] == "OK"
    assert reviewer["review_trace"]["visibility"] == "review_trace_safe_metadata"
    assert reviewer["audit_trace"] is None
    assert admin["result_status"] == "OK"
    assert admin["review_trace"]["visibility"] == "review_trace_safe_metadata"
    assert admin["audit_trace"]["visibility"] == "audit_trace_safe_metadata"
    assert admin["audit_trace"]["role_assignment_metadata_visible"] is True
    assert admin["audit_trace"]["raw_export_allowed"] is False
    _assert_zero_leak_counters(admin)
    _assert_no_forbidden_surface(admin)


def test_license_entitlement_pointer_only_and_hold_states(client: TestClient):
    active = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(
            role="instructor",
            rights_status="LICENSED",
            raw_text_policy="POINTER_ONLY",
            license_entitlement_id="lic:role-matrix",
            license_entitlement_status="ACTIVE",
        ),
    ).json()
    missing = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(role="instructor", rights_status="LICENSED", raw_text_policy="POINTER_ONLY"),
    ).json()
    expired = client.post(
        CHECK_POLICY_ROUTE,
        json=_policy_payload(
            role="instructor",
            rights_status="LICENSED",
            raw_text_policy="POINTER_ONLY",
            license_entitlement_id="lic:role-matrix",
            license_entitlement_status="EXPIRED",
        ),
    ).json()

    assert active["result_status"] == "OK"
    assert missing["result_status"] == "HOLD"
    assert "HOLD_PERMISSION" in missing["hold_reason"]
    assert expired["result_status"] == "HOLD"
    assert "HOLD_LICENSE_EXPIRED" in expired["hold_reason"]
    _assert_zero_leak_counters(expired)


def test_tenant_org_cohort_required_and_cross_tenant_denied(client: TestClient):
    missing_tenant = _policy_payload(role="instructor")
    missing_tenant.pop("tenant_id")
    same_scope = _policy_payload(
        role="instructor",
        target_tenant_id="tenant:role-matrix",
        target_organization_id="org:role-matrix",
        target_cohort_id="cohort:role-matrix",
    )
    cross_tenant = _policy_payload(role="instructor", target_tenant_id="tenant:other")

    missing = client.post(CHECK_POLICY_ROUTE, json=missing_tenant).json()
    same = client.post(CHECK_POLICY_ROUTE, json=same_scope).json()
    cross = client.post(CHECK_POLICY_ROUTE, json=cross_tenant).json()

    assert missing["result_status"] == "HOLD"
    assert "HOLD_TENANT_BOUNDARY" in missing["hold_reason"]
    assert same["result_status"] == "OK"
    assert cross["result_status"] == "HOLD"
    assert "HOLD_TENANT_BOUNDARY" in cross["hold_reason"]


def test_skillup_ok_answer_requires_protected_role_scope_and_keeps_student_safe_summary():
    missing_role = skillup_answer_from_bridge_response(_safe_bridge_response(role="student", evidence_items=[
        {
            "evidence_id": "ev:skillup-role-matrix",
            "bridge_trace_id": "btrace:skillup-role-matrix",
            "safe_summary": "Synthetic safe Skillup answer.",
            "pointer_uri": "pointer://role-matrix/skillup",
            "raw_text_policy": "SUMMARY_ONLY",
            "rights_status": "PUBLIC",
            **_scope(),
        }
    ]))
    answered = skillup_answer_from_bridge_response(_safe_bridge_response(role="student"))

    assert missing_role["result_status"] == "HOLD"
    assert "HOLD_PERMISSION" in missing_role["hold_reason"]
    assert answered["result_status"] == "OK"
    assert answered["answer"] == "Synthetic safe Skillup answer."
    assert answered["role"] == "student"
    assert answered["evidence_depth"] == "student_safe"
    _assert_zero_leak_counters(answered)


def test_course_binding_scope_license_and_cross_tenant_policy_are_fail_closed():
    missing_scope = bind_course_library_reference(
        {
            "course_id": "course:role-matrix",
            "module_id": "module:role-matrix",
            "evidence_id": "ev:role-matrix",
            "bridge_trace_id": "btrace:role-matrix",
            "current_status": "APPROVED_FOR_LIBRARY",
            "approval_record_id": "approval:role-matrix",
            "shape_validation_status": "PASS",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )
    cross_tenant = bind_course_library_reference(
        {
            **_scope(),
            "target_tenant_id": "tenant:other",
            "evidence_id": "ev:role-matrix",
            "bridge_trace_id": "btrace:role-matrix",
            "current_status": "APPROVED_FOR_LIBRARY",
            "approval_record_id": "approval:role-matrix",
            "shape_validation_status": "PASS",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )
    active_license = bind_course_library_reference(
        {
            **_scope(),
            "evidence_id": "ev:role-matrix",
            "bridge_trace_id": "btrace:role-matrix",
            "current_status": "APPROVED_FOR_LIBRARY",
            "approval_record_id": "approval:role-matrix",
            "shape_validation_status": "PASS",
            "rights_status": "LICENSED",
            "raw_text_policy": "POINTER_ONLY",
            "license_entitlement_id": "lic:role-matrix",
            "license_entitlement_status": "ACTIVE",
        }
    )

    assert missing_scope["binding_status"] == "HOLD"
    assert "HOLD_TENANT_BOUNDARY" in missing_scope["hold_reason"]
    assert cross_tenant["binding_status"] == "HOLD"
    assert "HOLD_TENANT_BOUNDARY" in cross_tenant["hold_reason"]
    assert active_license["binding_status"] == "BOUND"
