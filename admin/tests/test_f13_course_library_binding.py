from typing import Any

from admin.f13_course_library_binding import bind_course_library_reference


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


def _scope() -> dict[str, str]:
    return {
        "tenant_id": "tenant:diagnostic",
        "organization_id": "org:diagnostic",
        "cohort_id": "cohort:diagnostic",
    }


def _bridge_scope(
    *,
    bridge_family: str = "IPC",
    bridge_id: str = "bridge:ipc-core-5",
    standard_pack_id: str = "SPK_IPC_CORE_5",
    request_id: str = "req:diagnostic-1",
    trace_id: str = "btrace:diagnostic-1",
) -> dict[str, str]:
    return {
        "bridge_family": bridge_family,
        "bridge_id": bridge_id,
        "standard_pack_id": standard_pack_id,
        "request_id": request_id,
        "trace_id": trace_id,
        "bridge_trace_id": trace_id,
    }


def _assert_no_raw_internal_or_secret_surface(result: dict[str, Any]) -> None:
    rendered = "\n".join(_walk(result)).lower()
    forbidden_keys = {
        "raw_text",
        "raw_prompt",
        "raw_query",
        "full_source_text",
        "internal_path",
        "secret",
    }
    assert forbidden_keys.isdisjoint(set(result))
    assert "raw prompt" not in rendered
    assert "raw query" not in rendered
    assert "full source text" not in rendered
    assert "h:\\" not in rendered
    assert "c:\\" not in rendered
    assert "file://" not in rendered
    assert "secret" not in rendered
    assert "token" not in rendered


def _assert_no_pass_escalation(result: dict[str, Any]) -> None:
    assert "f13_pass" not in result
    assert "track_a_pass" not in result
    assert "beta_pass" not in result


def test_course_library_binding_safe_ok_contract():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            **_scope(),
            **_bridge_scope(),
            "evidence_id": "ev:diagnostic-1",
            "approval_record_id": "approval:diagnostic-1",
            "current_status": "APPROVED_FOR_LIBRARY",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
            "validation_shape_ids": ["SH-F13-CURATION-001"],
        }
    )

    assert result["binding_status"] == "BOUND"
    assert result["bridge_family"] == "IPC"
    assert result["bridge_id"] == "bridge:ipc-core-5"
    assert result["standard_pack_id"] == "SPK_IPC_CORE_5"
    assert result["request_id"] == "req:diagnostic-1"
    assert result["trace_id"] == "btrace:diagnostic-1"
    assert result["course_id"] == "course:diagnostic-1"
    assert result["module_id"] == "module:diagnostic-1"
    assert result["tenant_id"] == "tenant:diagnostic"
    assert result["organization_id"] == "org:diagnostic"
    assert result["cohort_id"] == "cohort:diagnostic"
    assert result["evidence_id"] == "ev:diagnostic-1"
    assert result["bridge_trace_id"] == "btrace:diagnostic-1"
    assert result["rights_status"] == "PUBLIC"
    assert result["raw_text_policy"] == "SUMMARY_ONLY"
    assert result["feedback_candidate_required"] is False
    assert result["feedback_queue_item"] is None
    assert result["skillup_use_allowed"] is True
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    assert result["db_access_executed"] is False
    _assert_no_raw_internal_or_secret_surface(result)
    _assert_no_pass_escalation(result)


def test_course_library_binding_missing_evidence_holds_with_feedback():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            **_scope(),
            **_bridge_scope(),
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )

    queue_item = result["feedback_queue_item"]
    assert result["binding_status"] == "HOLD"
    assert result["feedback_candidate_required"] is True
    assert queue_item["feedback_id"]
    assert queue_item["dedup_key"]
    assert queue_item["feedback_type"] == "EVIDENCE_GAP"
    assert result["skillup_use_allowed"] is False
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    assert queue_item["raw_text_included"] is False
    assert queue_item["internal_path_included"] is False
    _assert_no_raw_internal_or_secret_surface(result)
    _assert_no_pass_escalation(result)


def test_course_library_binding_unknown_rights_blocks_skillup_use():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "lesson_id": "lesson:diagnostic-1",
            **_scope(),
            **_bridge_scope(),
            "evidence_id": "ev:diagnostic-1",
            "rights_status": "UNKNOWN",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )

    assert result["binding_status"] in {"DENIED", "HOLD"}
    assert result["binding_status"] == "HOLD"
    assert result["module_id"] == "lesson:diagnostic-1"
    assert result["evidence_id"] == "ev:diagnostic-1"
    assert result["rights_status"] == "UNKNOWN"
    assert result["skillup_use_allowed"] is False
    assert result["feedback_candidate_required"] is True
    assert result["feedback_queue_item"]["feedback_type"] == "RIGHTS_POLICY_REVIEW"
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    _assert_no_raw_internal_or_secret_surface(result)
    _assert_no_pass_escalation(result)


def test_course_library_binding_warehouse_status_blocks_skillup_canonical_use():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            **_scope(),
            **_bridge_scope(),
            "library_node_id": "lib:diagnostic-1",
            "evidence_id": "ev:diagnostic-1",
            "approval_record_id": "approval:diagnostic-1",
            "current_status": "APPROVED_FOR_WAREHOUSE",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
            "validation_shape_ids": ["SH-F13-CURATION-001"],
        }
    )

    assert result["binding_status"] in {"DENIED", "HOLD"}
    assert result["skillup_use_allowed"] is False
    assert result["feedback_candidate_required"] is True
    _assert_no_raw_internal_or_secret_surface(result)
    _assert_no_pass_escalation(result)


def test_course_library_binding_dedup_key_stable_for_missing_evidence():
    payload = {
        "course_id": "course:diagnostic-1",
        "module_id": "module:diagnostic-1",
        **_scope(),
        **_bridge_scope(),
        "rights_status": "PUBLIC",
        "raw_text_policy": "SUMMARY_ONLY",
    }

    first = bind_course_library_reference(payload)
    second = bind_course_library_reference(dict(payload))

    assert first["feedback_queue_item"]["dedup_key"] == second["feedback_queue_item"]["dedup_key"]
    assert first["feedback_queue_item"]["feedback_id"] == second["feedback_queue_item"]["feedback_id"]


def test_course_library_binding_missing_course_or_module_holds_no_binding():
    result = bind_course_library_reference(
        {
            **_scope(),
            **_bridge_scope(trace_id="btrace:diagnostic-missing-scope"),
            "evidence_id": "ev:diagnostic-missing-scope",
            "current_status": "APPROVED_FOR_LIBRARY",
            "approval_record_id": "approval:diagnostic-missing-scope",
            "shape_validation_status": "PASS",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )

    assert result["binding_status"] == "HOLD"
    assert "HOLD_NO_BINDING" in result["hold_reason"]
    assert result["skillup_use_allowed"] is False
    _assert_no_raw_internal_or_secret_surface(result)


def test_course_library_binding_missing_tenant_scope_holds_boundary():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            **_bridge_scope(trace_id="btrace:diagnostic-missing-tenant"),
            "evidence_id": "ev:diagnostic-missing-tenant",
            "current_status": "APPROVED_FOR_LIBRARY",
            "approval_record_id": "approval:diagnostic-missing-tenant",
            "shape_validation_status": "PASS",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )

    assert result["binding_status"] == "HOLD"
    assert "HOLD_TENANT_BOUNDARY" in result["hold_reason"]
    assert result["skillup_use_allowed"] is False
    _assert_no_raw_internal_or_secret_surface(result)


def test_course_library_binding_missing_bridge_scope_holds_boundary():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            **_scope(),
            "evidence_id": "ev:diagnostic-missing-bridge",
            "bridge_trace_id": "btrace:diagnostic-missing-bridge",
            "current_status": "APPROVED_FOR_LIBRARY",
            "approval_record_id": "approval:diagnostic-missing-bridge",
            "shape_validation_status": "PASS",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )

    assert result["binding_status"] == "HOLD"
    assert "HOLD_BRIDGE_BOUNDARY" in result["hold_reason"]
    assert result["skillup_use_allowed"] is False
    _assert_no_raw_internal_or_secret_surface(result)


def test_course_library_binding_missing_trace_holds_boundary():
    result = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            **_scope(),
            "bridge_family": "IPC",
            "bridge_id": "bridge:ipc-core-5",
            "standard_pack_id": "SPK_IPC_CORE_5",
            "request_id": "req:diagnostic-missing-trace",
            "evidence_id": "ev:diagnostic-missing-trace",
            "current_status": "APPROVED_FOR_LIBRARY",
            "approval_record_id": "approval:diagnostic-missing-trace",
            "shape_validation_status": "PASS",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
        }
    )

    assert result["binding_status"] == "HOLD"
    assert "HOLD_TRACE_REQUIRED" in result["hold_reason"]
    assert result["skillup_use_allowed"] is False
    _assert_no_raw_internal_or_secret_surface(result)


def test_course_library_binding_preserves_multi_bridge_multi_course_identity():
    ipc = bind_course_library_reference(
        {
            "course_id": "course:ipc-manufacturing",
            "module_id": "module:ipc-a-610",
            **_scope(),
            **_bridge_scope(
                bridge_family="IPC",
                bridge_id="bridge:ipc-core-5",
                standard_pack_id="SPK_IPC_CORE_5",
                request_id="req:ipc-1",
                trace_id="btrace:ipc-1",
            ),
            "evidence_id": "ev:ipc-1",
            "approval_record_id": "approval:ipc-1",
            "current_status": "APPROVED_FOR_LIBRARY",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
            "validation_shape_ids": ["SH-F13-CURATION-IPC"],
        }
    )
    space = bind_course_library_reference(
        {
            "course_id": "course:space-standards",
            "module_id": "module:esa-nasa-jaxa",
            "tenant_id": "tenant:space",
            "organization_id": "org:space",
            "cohort_id": "cohort:space",
            **_bridge_scope(
                bridge_family="SPACE_AEROSPACE",
                bridge_id="bridge:space-esa-nasa-jaxa-ipc",
                standard_pack_id="SPK_SPACE_ESA_NASA_JAXA_IPC",
                request_id="req:space-1",
                trace_id="btrace:space-1",
            ),
            "evidence_id": "ev:space-1",
            "approval_record_id": "approval:space-1",
            "current_status": "APPROVED_FOR_LIBRARY",
            "rights_status": "PUBLIC",
            "raw_text_policy": "SUMMARY_ONLY",
            "validation_shape_ids": ["SH-F13-CURATION-SPACE"],
        }
    )

    assert ipc["binding_status"] == "BOUND"
    assert space["binding_status"] == "BOUND"
    assert ipc["binding_id"] != space["binding_id"]
    assert ipc["bridge_family"] == "IPC"
    assert space["bridge_family"] == "SPACE_AEROSPACE"
    assert ipc["standard_pack_id"] == "SPK_IPC_CORE_5"
    assert space["standard_pack_id"] == "SPK_SPACE_ESA_NASA_JAXA_IPC"
    assert ipc["course_id"] != space["course_id"]
    assert ipc["module_id"] != space["module_id"]
    assert ipc["tenant_id"] != space["tenant_id"]
    assert ipc["evidence_id"] != space["evidence_id"]
    assert ipc["trace_id"] != space["trace_id"]


def test_course_library_binding_licensed_entitlement_requires_active_pointer_only_scope():
    missing_entitlement = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            **_scope(),
            **_bridge_scope(trace_id="btrace:diagnostic-license"),
            "evidence_id": "ev:diagnostic-license",
            "current_status": "APPROVED_FOR_LIBRARY",
            "approval_record_id": "approval:diagnostic-license",
            "shape_validation_status": "PASS",
            "rights_status": "LICENSED",
            "raw_text_policy": "POINTER_ONLY",
        }
    )
    active_entitlement = bind_course_library_reference(
        {
            "course_id": "course:diagnostic-1",
            "module_id": "module:diagnostic-1",
            **_scope(),
            **_bridge_scope(trace_id="btrace:diagnostic-license"),
            "evidence_id": "ev:diagnostic-license",
            "current_status": "APPROVED_FOR_LIBRARY",
            "approval_record_id": "approval:diagnostic-license",
            "shape_validation_status": "PASS",
            "rights_status": "LICENSED",
            "raw_text_policy": "POINTER_ONLY",
            "license_entitlement_id": "lic:diagnostic",
            "license_entitlement_status": "ACTIVE",
        }
    )

    assert missing_entitlement["binding_status"] == "HOLD"
    assert "license_entitlement_id" in missing_entitlement["hold_reason"]
    assert active_entitlement["binding_status"] == "BOUND"
    assert active_entitlement["skillup_use_allowed"] is True
    _assert_no_raw_internal_or_secret_surface(active_entitlement)
