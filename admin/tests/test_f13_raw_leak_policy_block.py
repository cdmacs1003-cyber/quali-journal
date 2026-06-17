from admin.f13_raw_leak_policy_block import validate_raw_leak_policy_block


def _ready_payload():
    return {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "tenant_context": {
            "tenant_id": "TEN-diagnostic",
            "organization_id": "ORGUNIT-diagnostic",
        },
        "evidence_context": {
            "evidence_ids": ["EVD-20260511-0001"],
            "evidence_pointers": ["urn:qlib:evidence:20260511-0001"],
        },
        "rights_context": {
            "rights_status": "LICENSED",
        },
        "output_policy": {
            "pointer_only_required": True,
            "raw_export_allowed": False,
            "student_raw_text_allowed": False,
            "internal_path_allowed": False,
            "secret_surface_allowed": False,
        },
        "counters": {
            "raw_standard_text_export_count": 0,
            "internal_path_leak_count": 0,
            "secret_like_surface_count": 0,
        },
        "output_surface": {
            "safe_summary": "Use a bounded training summary with evidence metadata.",
            "evidence_label": "IPC training evidence pointer",
        },
    }


def _assert_hold_or_invalid(result):
    assert result["status"] in {"HOLD", "INVALID"}
    assert result["policy_block_ready"] is False
    assert result["hold_reason"]
    assert result["errors"]
    _assert_no_execution_flags(result)


def _assert_no_execution_flags(result):
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    assert result["secret_surface_included"] is False
    assert result["db_access_executed"] is False
    assert result["network_access_executed"] is False
    assert result["runtime_access_executed"] is False
    assert result["file_io_executed"] is False
    assert result["env_access_executed"] is False
    assert result["subprocess_executed"] is False


def _walk_public_values(value):
    if isinstance(value, dict):
        out = []
        for child in value.values():
            out.extend(_walk_public_values(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_walk_public_values(child))
        return out
    return [str(value)]


def test_ready_payload_returns_ready_and_policy_block_ready_true():
    result = validate_raw_leak_policy_block(_ready_payload())

    assert result["status"] == "READY"
    assert result["policy_block_ready"] is True
    assert result["hold_reason"] is None
    assert result["errors"] == []
    assert result["counters"] == {
        "raw_standard_text_export_count": 0,
        "internal_path_leak_count": 0,
        "secret_like_surface_count": 0,
    }
    assert result["checks"]["evidence_linked"] is True
    assert result["checks"]["output_surface_internal_data_absent"] is True
    _assert_no_execution_flags(result)


def test_non_dict_payload_returns_invalid():
    result = validate_raw_leak_policy_block(["not", "a", "mapping"])

    assert result["status"] == "INVALID"
    assert result["policy_block_ready"] is False
    _assert_no_execution_flags(result)


def test_missing_schema_version_returns_invalid():
    payload = _ready_payload()
    payload.pop("schema_version")

    result = validate_raw_leak_policy_block(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_missing_contract_version_returns_invalid():
    payload = _ready_payload()
    payload.pop("contract_version")

    result = validate_raw_leak_policy_block(payload)

    assert result["status"] == "INVALID"
    _assert_no_execution_flags(result)


def test_missing_tenant_context_returns_invalid_or_hold():
    payload = _ready_payload()
    payload.pop("tenant_context")

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_missing_tenant_id_returns_invalid_or_hold():
    payload = _ready_payload()
    payload["tenant_context"].pop("tenant_id")

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_missing_organization_id_returns_invalid_or_hold():
    payload = _ready_payload()
    payload["tenant_context"].pop("organization_id")

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_missing_evidence_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["evidence_context"] = {
        "evidence_ids": [],
        "evidence_pointers": [],
    }

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_rights_status_unknown_returns_hold():
    payload = _ready_payload()
    payload["rights_context"]["rights_status"] = "UNKNOWN"

    result = validate_raw_leak_policy_block(payload)

    assert result["status"] == "HOLD"
    assert result["policy_block_ready"] is False
    _assert_no_execution_flags(result)


def test_pointer_only_required_false_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["output_policy"]["pointer_only_required"] = False

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_raw_export_allowed_true_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["output_policy"]["raw_export_allowed"] = True

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_student_raw_text_allowed_true_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["output_policy"]["student_raw_text_allowed"] = True

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_internal_path_allowed_true_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["output_policy"]["internal_path_allowed"] = True

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_secret_surface_allowed_true_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["output_policy"]["secret_surface_allowed"] = True

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_raw_standard_text_export_count_nonzero_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["counters"]["raw_standard_text_export_count"] = 1

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_internal_path_leak_count_nonzero_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["counters"]["internal_path_leak_count"] = 1

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_secret_like_surface_count_nonzero_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["counters"]["secret_like_surface_count"] = 1

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_unsafe_output_surface_raw_text_marker_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["output_surface"]["student_copy"] = "paid standard raw excerpt"

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_forbidden_raw_field_key_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["output_surface"]["raw_text"] = "synthetic raw standard text should be blocked"

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_unsafe_output_surface_internal_path_marker_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["output_surface"]["storage_hint"] = "H:\\restricted\\standard.txt"

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_unsafe_output_surface_internal_data_markers_return_hold_or_invalid():
    payload = _ready_payload()
    payload["output_surface"] = {
        "direct_db_row": {"id": "row-1"},
        "warehouse_internal_object": "warehouse internal object",
        "library_internal_object": "library internal object",
    }

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)
    assert result["checks"]["output_surface_internal_data_absent"] is False


def test_unsafe_output_surface_secret_like_marker_returns_hold_or_invalid():
    payload = _ready_payload()
    payload["output_surface"]["operator_note"] = "api token value"

    result = validate_raw_leak_policy_block(payload)

    _assert_hold_or_invalid(result)


def test_public_result_does_not_echo_unsafe_raw_internal_or_secret_surface():
    payload = _ready_payload()
    unsafe_values = [
        "paid standard raw excerpt",
        "H:\\restricted\\standard.txt",
        "api token value",
    ]
    payload["output_surface"] = {
        "student_copy": unsafe_values[0],
        "storage_hint": unsafe_values[1],
        "operator_note": unsafe_values[2],
    }

    result = validate_raw_leak_policy_block(payload)
    rendered_values = "\n".join(_walk_public_values(result)).lower()

    _assert_hold_or_invalid(result)
    for unsafe_value in unsafe_values:
        assert unsafe_value.lower() not in rendered_values


def test_no_db_network_runtime_file_env_or_subprocess_flags_are_ever_true():
    ready_result = validate_raw_leak_policy_block(_ready_payload())
    hold_payload = _ready_payload()
    hold_payload["output_policy"]["raw_export_allowed"] = True
    hold_result = validate_raw_leak_policy_block(hold_payload)

    for result in (ready_result, hold_result):
        _assert_no_execution_flags(result)


def test_non_claims_include_no_broader_pass_escalation():
    result = validate_raw_leak_policy_block(_ready_payload())

    assert result["non_claims"] == [
        "BRIDGE_HEALTH_PASS_NOT_INFERRED",
        "ANSWER_QUALITY_PASS_NOT_INFERRED",
        "SKILLUP_MVP_PASS_NOT_INFERRED",
        "TRACK_A_PASS_NOT_INFERRED",
        "BETA_PASS_NOT_INFERRED",
        "F13_PASS_NOT_INFERRED",
        "RAW_LEAK_POLICY_BLOCK_PASS_FULL_STATUS_NOT_INFERRED",
        "SELECTED_STATIC_POLICY_BLOCK_READINESS_ONLY",
    ]
