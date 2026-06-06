from admin.f13_runtime_guard import (
    BRIDGE_EVIDENCE_ALLOWLIST_FIELDS,
    RAW_TEXT_POLICY_DENIED,
    RAW_TEXT_POLICY_NOT_VERIFIED,
    RAW_TEXT_POLICY_POINTER_ONLY,
    RAW_TEXT_POLICY_REDACTED_SUMMARY_ONLY,
    RAW_TEXT_POLICY_SUMMARY_ONLY,
    RESULT_DENIED,
    RESULT_HOLD,
    RESULT_OK,
    RIGHTS_CUSTOMER_CONFIDENTIAL,
    RIGHTS_INTERNAL,
    RIGHTS_LICENSED,
    RIGHTS_NOT_VERIFIED,
    RIGHTS_PUBLIC,
    RIGHTS_RESTRICTED,
    RIGHTS_UNKNOWN,
    decide_bridge_result,
    detect_forbidden_fields,
    normalize_raw_text_policy,
    normalize_rights_status,
    project_bridge_safe_evidence,
    validate_human_redacted_preflight_replay_evidence,
    validate_bridge_safe_response,
)


def _base_evidence(**overrides):
    evidence = {
        "evidence_id": "ev:synthetic-1",
        "bridge_trace_id": "btrace:synthetic-1",
        "safe_summary": "Synthetic safe summary for policy testing.",
        "raw_text_policy": RAW_TEXT_POLICY_SUMMARY_ONLY,
        "rights_status": RIGHTS_PUBLIC,
    }
    evidence.update(overrides)
    return evidence


def test_tc1_public_summary_only_with_required_fields_returns_ok():
    result = decide_bridge_result(_base_evidence())

    assert result["result_status"] == RESULT_OK
    assert result["feedback_candidate_required"] is False


def test_tc2_missing_evidence_id_returns_hold():
    evidence = _base_evidence(evidence_id="")

    result = decide_bridge_result(evidence)

    assert result["result_status"] == RESULT_HOLD
    assert "evidence_id" in result["hold_reason"]


def test_tc3_missing_safe_summary_returns_hold():
    evidence = _base_evidence(safe_summary=None)

    result = decide_bridge_result(evidence)

    assert result["result_status"] == RESULT_HOLD
    assert "safe_summary" in result["hold_reason"]


def test_tc4_unknown_rights_returns_hold():
    evidence = _base_evidence(rights_status=RIGHTS_UNKNOWN)

    result = decide_bridge_result(evidence)

    assert result["result_status"] == RESULT_HOLD


def test_tc5_restricted_rights_returns_denied():
    evidence = _base_evidence(rights_status=RIGHTS_RESTRICTED)

    result = decide_bridge_result(evidence)

    assert result["result_status"] == RESULT_DENIED


def test_tc6_licensed_pointer_only_with_safe_pointer_returns_ok():
    evidence = _base_evidence(
        rights_status=RIGHTS_LICENSED,
        raw_text_policy=RAW_TEXT_POLICY_POINTER_ONLY,
        pointer_uri="pointer://safe/synthetic-1",
    )

    result = decide_bridge_result(evidence)

    assert result["result_status"] == RESULT_OK


def test_tc7_licensed_raw_text_policy_denied_returns_denied():
    evidence = _base_evidence(
        rights_status=RIGHTS_LICENSED,
        raw_text_policy=RAW_TEXT_POLICY_DENIED,
        pointer_uri="pointer://safe/synthetic-1",
    )

    result = decide_bridge_result(evidence)

    assert result["result_status"] == RESULT_DENIED


def test_tc8_customer_confidential_summary_only_does_not_return_ok():
    evidence = _base_evidence(
        rights_status=RIGHTS_CUSTOMER_CONFIDENTIAL,
        raw_text_policy=RAW_TEXT_POLICY_SUMMARY_ONLY,
    )

    result = decide_bridge_result(evidence)

    assert result["result_status"] in {RESULT_DENIED, RESULT_HOLD}
    assert result["result_status"] != RESULT_OK


def test_tc9_customer_confidential_redacted_summary_with_approval_returns_ok():
    evidence = _base_evidence(
        rights_status=RIGHTS_CUSTOMER_CONFIDENTIAL,
        raw_text_policy=RAW_TEXT_POLICY_REDACTED_SUMMARY_ONLY,
        redaction_approved=True,
    )

    result = decide_bridge_result(evidence)

    assert result["result_status"] == RESULT_OK


def test_tc10_raw_text_ref_is_detected_and_not_projected():
    evidence = _base_evidence(raw_text_ref="internal-only-synthetic-pointer")

    violations = detect_forbidden_fields(evidence)
    projected = project_bridge_safe_evidence(evidence)

    assert "raw_text_ref" in violations
    assert "raw_text_ref" not in projected


def test_tc11_raw_pointer_or_internal_h_drive_path_is_detected_and_denied():
    evidence = _base_evidence(pointer_uri=r"H:\장기기억\synthetic\raw.txt")

    violations = detect_forbidden_fields(evidence)
    result = decide_bridge_result(evidence)
    projected = project_bridge_safe_evidence(evidence)

    assert any("h_drive_janggigieok" in item for item in violations)
    assert result["result_status"] == RESULT_DENIED
    assert "pointer_uri" not in projected


def test_tc12_skillup_direct_db_access_attempt_returns_denied():
    evidence = _base_evidence(direct_db_access_attempt=True)

    result = decide_bridge_result(evidence, requester_module="Skillup")

    assert result["result_status"] == RESULT_DENIED


def test_tc13_project_bridge_safe_evidence_returns_only_allowlisted_fields():
    evidence = _base_evidence(
        pointer_uri="pointer://safe/synthetic-1",
        source_doc_kind="synthetic_note",
        extra_field="not allowed",
        warehouse_internal_object={"id": "synthetic"},
    )

    projected = project_bridge_safe_evidence(evidence)

    assert set(projected).issubset(BRIDGE_EVIDENCE_ALLOWLIST_FIELDS)
    assert "extra_field" not in projected
    assert "warehouse_internal_object" not in projected


def test_tc14_validate_bridge_safe_response_fails_for_forbidden_fields():
    response = {
        **_base_evidence(),
        "result_status": RESULT_OK,
        "raw_text_ref": "internal-only-synthetic-pointer",
    }

    validation = validate_bridge_safe_response(response)

    assert validation["is_safe"] is False
    assert validation["result_status"] == RESULT_DENIED
    assert validation["violations"]


def test_tc15_normalize_rights_status_maps_existing_values():
    assert normalize_rights_status("public_reference") == RIGHTS_PUBLIC
    assert normalize_rights_status("owned") == RIGHTS_INTERNAL
    assert normalize_rights_status("licensed") == RIGHTS_LICENSED
    assert normalize_rights_status("permission_granted") == RIGHTS_LICENSED
    assert normalize_rights_status("internal_only") == RIGHTS_INTERNAL
    assert normalize_rights_status("no_export") == RIGHTS_RESTRICTED
    assert normalize_rights_status("unknown") == RIGHTS_UNKNOWN
    assert normalize_rights_status(None) == RIGHTS_NOT_VERIFIED


def test_tc16_normalize_raw_text_policy_maps_unknown_or_none_to_not_verified():
    assert normalize_raw_text_policy("SUMMARY_ONLY") == RAW_TEXT_POLICY_SUMMARY_ONLY
    assert normalize_raw_text_policy("pointer_only") == RAW_TEXT_POLICY_POINTER_ONLY
    assert normalize_raw_text_policy("not-a-policy") == RAW_TEXT_POLICY_NOT_VERIFIED
    assert normalize_raw_text_policy(None) == RAW_TEXT_POLICY_NOT_VERIFIED


def _redacted_preflight_evidence(**overrides):
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


def _assert_no_raw_value_echo(result, *raw_values):
    rendered = repr(result)
    for raw_value in raw_values:
        assert raw_value not in rendered


def test_tc17_redacted_preflight_replay_evidence_returns_local_gate_pass():
    result = validate_human_redacted_preflight_replay_evidence(_redacted_preflight_evidence())

    assert result["result_status"] == RESULT_OK
    assert result["ok"] is True
    assert (
        result["status"]
        == "PASS_HUMAN_REDACTED_PREFLIGHT_REPLAY_EVIDENCE_ACCEPTED_FOR_LOCAL_REPLAY_GATE_ONLY"
    )
    assert result["accepted_scope"]["target_table"] == "public.f13_feedback_queue_items"
    assert result["accepted_scope"]["codex_live_db_verification"] == "NOT_EXECUTED"


def test_tc18_redacted_preflight_wrong_user_holds_without_echoing_raw_value():
    raw_wrong_user = "postgres_superuser_raw_value"
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(connected_user_result=raw_wrong_user)
    )

    assert result["result_status"] == RESULT_HOLD
    assert "CONNECTED_USER_RESULT_MISMATCH" in result["reason_codes"]
    _assert_no_raw_value_echo(result, raw_wrong_user)


def test_tc19_redacted_preflight_wrong_database_holds_without_db_access():
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(target_database="other_local_database")
    )

    assert result["result_status"] == RESULT_HOLD
    assert "TARGET_DATABASE_MISMATCH" in result["reason_codes"]
    assert "CODEX_DB_CONNECTION" in result["prohibited_actions"]


def test_tc20_redacted_preflight_wrong_table_holds_but_public_feedback_queue_is_nonblocking_when_target_confirmed():
    wrong_only = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(
            table_checked="public.feedback_queue",
            table_exists_result="public.feedback_queue returned null",
        )
    )
    nonblocking_mismatch = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(table_checked="public.feedback_queue")
    )

    assert wrong_only["result_status"] == RESULT_HOLD
    assert "PUBLIC_FEEDBACK_QUEUE_PRESENT_WITHOUT_ACCEPTED_TABLE_CONFIRMATION" in wrong_only["reason_codes"]
    assert nonblocking_mismatch["result_status"] == RESULT_OK


def test_tc21_redacted_preflight_can_insert_true_holds_or_denies():
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(can_insert=True)
    )

    assert result["result_status"] in {RESULT_HOLD, RESULT_DENIED}
    assert "CAN_INSERT_MISMATCH" in result["reason_codes"]


def test_tc22_redacted_preflight_can_update_true_holds_or_denies():
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(can_update=True)
    )

    assert result["result_status"] in {RESULT_HOLD, RESULT_DENIED}
    assert "CAN_UPDATE_MISMATCH" in result["reason_codes"]


def test_tc23_redacted_preflight_can_delete_true_holds_or_denies():
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(can_delete=True)
    )

    assert result["result_status"] in {RESULT_HOLD, RESULT_DENIED}
    assert "CAN_DELETE_MISMATCH" in result["reason_codes"]


def test_tc24_redacted_preflight_db_write_attempted_yes_denies_boundary_risk():
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(DB_write_attempted="YES")
    )

    assert result["result_status"] == RESULT_DENIED
    assert result["status"] == "DENY_WRITE_OR_MIGRATION_BOUNDARY_RISK"
    assert "DB_WRITE_ATTEMPTED_POSITIVE" in result["reason_codes"]


def test_tc25_redacted_preflight_migration_executed_yes_denies_boundary_risk():
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(migration_executed_in_replay="YES")
    )

    assert result["result_status"] == RESULT_DENIED
    assert result["status"] == "DENY_WRITE_OR_MIGRATION_BOUNDARY_RISK"
    assert "MIGRATION_EXECUTED_IN_REPLAY_POSITIVE" in result["reason_codes"]


def test_tc26_redacted_preflight_rollback_executed_yes_denies_boundary_risk():
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(rollback_executed_in_replay="YES")
    )

    assert result["result_status"] == RESULT_DENIED
    assert result["status"] == "DENY_WRITE_OR_MIGRATION_BOUNDARY_RISK"
    assert "ROLLBACK_EXECUTED_IN_REPLAY_POSITIVE" in result["reason_codes"]


def test_tc27_redacted_preflight_credential_material_yes_denies_without_echo():
    raw_marker = "unsafe credential marker"
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(credential_material_recorded=raw_marker)
    )

    assert result["result_status"] == RESULT_DENIED
    assert result["status"] == "DENY_SECRET_BOUNDARY_RISK"
    _assert_no_raw_value_echo(result, raw_marker)


def test_tc28_redacted_preflight_password_recorded_yes_denies_without_echo():
    raw_marker = "unsafe password marker"
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(password_recorded=raw_marker)
    )

    assert result["result_status"] == RESULT_DENIED
    assert result["status"] == "DENY_SECRET_BOUNDARY_RISK"
    _assert_no_raw_value_echo(result, raw_marker)


def test_tc29_redacted_preflight_full_connection_string_recorded_yes_denies_without_echo():
    raw_marker = "unsafe full connection string marker"
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(full_connection_string_recorded=raw_marker)
    )

    assert result["result_status"] == RESULT_DENIED
    assert result["status"] == "DENY_SECRET_BOUNDARY_RISK"
    _assert_no_raw_value_echo(result, raw_marker)


def test_tc30_redacted_preflight_missing_exact_time_is_nonblocking_with_date_and_kst_context():
    result = validate_human_redacted_preflight_replay_evidence(
        _redacted_preflight_evidence(replay_datetime_local="2026-05-17 KST")
    )

    assert result["result_status"] == RESULT_OK
    assert "TIME_EXACT_NOT_RECORDED_DATE_KST_ONLY" in result["residuals"]


def test_tc31_redacted_preflight_helper_has_no_db_subprocess_environment_or_filesystem_write_surface():
    names = set(validate_human_redacted_preflight_replay_evidence.__code__.co_names)

    forbidden_names = {
        "connect",
        "execute",
        "psql",
        "subprocess",
        "run",
        "popen",
        "requests",
        "urllib",
        "socket",
        "open",
        "write",
        "environ",
        "getenv",
        "dotenv",
    }
    assert names.isdisjoint(forbidden_names)


def test_tc32_quarantined_search_exposure_returns_hold_or_denied():
    evidence = _base_evidence(
        current_status="QUARANTINED",
        source_doc_kind="library_item",
        search_exposure_requested=True,
    )

    result = decide_bridge_result(evidence, requester_module="Search", purpose="search_exposure")

    assert result["result_status"] in {RESULT_HOLD, RESULT_DENIED}
