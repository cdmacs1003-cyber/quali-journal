from admin.f13_module_manifest import validate_module_manifest


def _valid_manifest():
    return {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "tenant_context": {
            "tenant_id": "TEN-diagnostic",
            "organization_id": "ORGUNIT-diagnostic",
        },
        "module_id": "MOD-QUALI-PCB-REWORK-V1",
        "module_family": "QUALI",
        "module_title": "PCB Rework and Repair",
        "module_version": "v1",
        "status": "active",
        "owner": "QLIB",
        "learning_objectives": [
            {
                "objective_id": "OBJ-001",
                "title": "Understand approved repair boundaries",
                "linked_library_scope": {
                    "library_ids": ["LIB-ref-pcb-rework-approval-v1"],
                    "graph_node_ids": ["GND-risk-repair-without-approval-v1"],
                    "evidence_ids": ["EVD-20260511-0001"],
                },
            }
        ],
        "standard_pack_refs": ["SPK-QUALI-REWORK-V1"],
        "required_library_scope": {
            "library_ids": ["LIB-ref-pcb-rework-approval-v1"],
            "graph_node_ids": ["GND-risk-repair-without-approval-v1"],
            "evidence_ids": ["EVD-20260511-0001"],
        },
        "evidence_policy": {
            "evidence_required": True,
            "missing_evidence_action": "HOLD",
            "raw_standard_text_allowed_for_student": False,
            "internal_path_allowed": False,
        },
        "assessment_map": [
            {
                "assessment_id": "ASM-REWORK-001",
                "objective_id": "OBJ-001",
                "evidence_required": True,
            }
        ],
        "telemetry_policy": {
            "event_logging_required": True,
            "learning_analytics_allowed": True,
            "marketing_use_default": "aggregated_only",
        },
    }


def _walk_public_result(value):
    safe_flag_keys = {
        "raw_text_included",
        "internal_path_included",
        "db_access_executed",
        "network_access_executed",
        "runtime_access_executed",
    }
    if isinstance(value, dict):
        out = []
        for key, child in value.items():
            if key in safe_flag_keys:
                out.append(str(child))
                continue
            out.append(str(key))
            out.extend(_walk_public_result(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_walk_public_result(child))
        return out
    return [str(value)]


def _assert_hold_or_invalid(result):
    assert result["status"] in {"HOLD", "INVALID"}
    assert result["active_ready"] is False
    assert result["hold_reason"]
    assert result["errors"]
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    assert result["db_access_executed"] is False
    assert result["network_access_executed"] is False
    assert result["runtime_access_executed"] is False


def _assert_no_public_unsafe_surface(result):
    rendered = "\n".join(_walk_public_result(result)).lower()
    assert result["raw_text_included"] is False
    assert result["internal_path_included"] is False
    assert result["db_access_executed"] is False
    assert result["network_access_executed"] is False
    assert result["runtime_access_executed"] is False
    assert "raw prompt" not in rendered
    assert "raw query" not in rendered
    assert "full source text" not in rendered
    assert "paid standard raw" not in rendered
    assert "h:\\" not in rendered
    assert "c:\\" not in rendered
    assert "file://" not in rendered
    assert "secret" not in rendered
    assert "token" not in rendered
    assert "credential" not in rendered


def test_valid_active_module_manifest_returns_valid_active_ready():
    result = validate_module_manifest(_valid_manifest())

    assert result["status"] == "VALID"
    assert result["active_ready"] is True
    assert result["hold_reason"] is None
    assert result["errors"] == []
    assert result["module_id"] == "MOD-QUALI-PCB-REWORK-V1"
    assert result["module_version"] == "v1"
    _assert_no_public_unsafe_surface(result)


def test_missing_required_library_scope_returns_hold_or_invalid():
    payload = _valid_manifest()
    payload.pop("required_library_scope")

    result = validate_module_manifest(payload)

    _assert_hold_or_invalid(result)
    _assert_no_public_unsafe_surface(result)


def test_evidence_required_false_returns_hold_or_invalid():
    payload = _valid_manifest()
    payload["evidence_policy"]["evidence_required"] = False

    result = validate_module_manifest(payload)

    _assert_hold_or_invalid(result)


def test_raw_standard_text_allowed_for_student_true_returns_hold_or_invalid():
    payload = _valid_manifest()
    payload["evidence_policy"]["raw_standard_text_allowed_for_student"] = True

    result = validate_module_manifest(payload)

    _assert_hold_or_invalid(result)


def test_internal_path_allowed_true_returns_hold_or_invalid():
    payload = _valid_manifest()
    payload["evidence_policy"]["internal_path_allowed"] = True

    result = validate_module_manifest(payload)

    _assert_hold_or_invalid(result)


def test_invalid_status_returns_hold_or_invalid():
    payload = _valid_manifest()
    payload["status"] = "published"

    result = validate_module_manifest(payload)

    _assert_hold_or_invalid(result)


def test_missing_tenant_context_returns_hold_or_invalid():
    payload = _valid_manifest()
    payload.pop("tenant_context")

    result = validate_module_manifest(payload)

    _assert_hold_or_invalid(result)


def test_active_status_without_evidence_scope_returns_hold_or_invalid():
    payload = _valid_manifest()
    payload["required_library_scope"] = {
        "library_ids": [],
        "graph_node_ids": [],
        "evidence_ids": [],
    }

    result = validate_module_manifest(payload)

    _assert_hold_or_invalid(result)


def test_public_result_has_no_raw_internal_or_secret_surface():
    payload = _valid_manifest()
    payload["unsafe_notes"] = {
        "source_uri_or_path": "H:\\restricted\\module.txt",
        "api_token": "do-not-echo",
        "student_copy": "paid standard raw excerpt",
    }

    result = validate_module_manifest(payload)

    _assert_hold_or_invalid(result)
    _assert_no_public_unsafe_surface(result)


def test_deprecated_manifest_is_classified_but_not_active_ready():
    payload = _valid_manifest()
    payload["status"] = "deprecated"

    result = validate_module_manifest(payload)

    assert result["status"] == "VALID"
    assert result["active_ready"] is False
    assert result["hold_reason"] is None
    assert result["errors"] == []
    assert result["warnings"]
    _assert_no_public_unsafe_surface(result)
