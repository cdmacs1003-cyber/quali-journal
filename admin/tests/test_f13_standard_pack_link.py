from admin.f13_standard_pack_link import validate_standard_pack_link


def _valid_pack_link():
    return {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "tenant_context": {
            "tenant_id": "TEN-diagnostic",
            "organization_id": "ORGUNIT-diagnostic",
        },
        "standard_pack_id": "SPK-IPC-SOLDERING-V1",
        "pack_family": "IPC",
        "pack_title": "IPC Soldering Standards Pack",
        "pack_version": "v1",
        "status": "approved",
        "standard_node_ids": ["IPC:J-STD-001@RevH"],
        "library_ids": ["LIB-ipc-jstd-001-revh-v1"],
        "graph_node_ids": ["GND-ipc-jstd-001-revh-v1"],
        "evidence_ids": ["EVD-20260511-0001"],
        "tailoring_pack_ids": ["TPK-IPC-SOLDERING-V1"],
        "policy": {
            "paid_standard_pointer_only": True,
            "raw_export_allowed": False,
            "student_summary_allowed": True,
        },
        "linked_module_ids": ["MOD-IPC-SOLDERING-V1"],
        "linked_binding_ids": ["BND-CRS-IPC-SOLDERING-MOD-IPC-SOLDERING-V1-V1"],
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
    assert result["active_link_ready"] is False
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
    assert "raw standard text" not in rendered
    assert "full source text" not in rendered
    assert "paid standard raw" not in rendered
    assert "h:\\" not in rendered
    assert "c:\\" not in rendered
    assert "file://" not in rendered
    assert "secret" not in rendered
    assert "token" not in rendered
    assert "credential" not in rendered


def test_valid_approved_standard_pack_link_returns_valid_active_link_ready():
    result = validate_standard_pack_link(_valid_pack_link())

    assert result["status"] == "VALID"
    assert result["active_link_ready"] is True
    assert result["hold_reason"] is None
    assert result["errors"] == []
    assert result["standard_pack_id"] == "SPK-IPC-SOLDERING-V1"
    assert result["pack_version"] == "v1"
    assert result["pack_family"] == "IPC"
    assert result["standard_count"] == 1
    assert result["library_count"] == 1
    assert result["evidence_count"] == 1
    _assert_no_public_unsafe_surface(result)


def test_missing_tenant_context_returns_hold_or_invalid():
    payload = _valid_pack_link()
    payload.pop("tenant_context")

    result = validate_standard_pack_link(payload)

    _assert_hold_or_invalid(result)


def test_missing_standard_node_ids_returns_hold_or_invalid():
    payload = _valid_pack_link()
    payload["standard_node_ids"] = []

    result = validate_standard_pack_link(payload)

    _assert_hold_or_invalid(result)


def test_missing_library_ids_returns_hold_or_invalid():
    payload = _valid_pack_link()
    payload["library_ids"] = []

    result = validate_standard_pack_link(payload)

    _assert_hold_or_invalid(result)


def test_missing_evidence_ids_returns_hold_or_invalid():
    payload = _valid_pack_link()
    payload["evidence_ids"] = []

    result = validate_standard_pack_link(payload)

    _assert_hold_or_invalid(result)


def test_raw_export_allowed_true_returns_hold_or_invalid():
    payload = _valid_pack_link()
    payload["policy"]["raw_export_allowed"] = True

    result = validate_standard_pack_link(payload)

    _assert_hold_or_invalid(result)


def test_paid_standard_pointer_only_false_returns_hold_or_invalid():
    payload = _valid_pack_link()
    payload["policy"]["paid_standard_pointer_only"] = False

    result = validate_standard_pack_link(payload)

    _assert_hold_or_invalid(result)


def test_invalid_pack_family_returns_hold_or_invalid():
    payload = _valid_pack_link()
    payload["pack_family"] = "MIL"

    result = validate_standard_pack_link(payload)

    _assert_hold_or_invalid(result)


def test_invalid_status_returns_hold_or_invalid():
    payload = _valid_pack_link()
    payload["status"] = "published"

    result = validate_standard_pack_link(payload)

    _assert_hold_or_invalid(result)


def test_draft_pack_is_classified_but_not_active_link_ready():
    payload = _valid_pack_link()
    payload["status"] = "draft"

    result = validate_standard_pack_link(payload)

    assert result["status"] == "VALID"
    assert result["active_link_ready"] is False
    assert result["hold_reason"] is None
    assert result["errors"] == []
    assert result["warnings"]
    _assert_no_public_unsafe_surface(result)


def test_deprecated_pack_is_classified_but_not_active_link_ready():
    payload = _valid_pack_link()
    payload["status"] = "deprecated"

    result = validate_standard_pack_link(payload)

    assert result["status"] == "VALID"
    assert result["active_link_ready"] is False
    assert result["hold_reason"] is None
    assert result["errors"] == []
    assert result["warnings"]
    _assert_no_public_unsafe_surface(result)


def test_public_result_has_no_raw_internal_or_secret_surface():
    payload = _valid_pack_link()
    payload["unsafe_notes"] = {
        "source_uri_or_path": "H:\\restricted\\standard-pack.txt",
        "api_token": "do-not-echo",
        "student_copy": "paid standard raw excerpt",
    }

    result = validate_standard_pack_link(payload)

    _assert_hold_or_invalid(result)
    _assert_no_public_unsafe_surface(result)


def test_no_db_network_or_runtime_flags_are_ever_true():
    ok_result = validate_standard_pack_link(_valid_pack_link())
    hold_payload = _valid_pack_link()
    hold_payload["evidence_ids"] = []
    hold_result = validate_standard_pack_link(hold_payload)

    for result in (ok_result, hold_result):
        assert result["db_access_executed"] is False
        assert result["network_access_executed"] is False
        assert result["runtime_access_executed"] is False
        assert result["raw_text_included"] is False
        assert result["internal_path_included"] is False
