from typing import Any

from admin.f13_gap_map import build_f13_gap_map


def _selected_surface(contract: str, status: str = "PASS") -> dict[str, Any]:
    return {
        "surface_id": contract.lower(),
        "contract": contract,
        "status": status,
        "evidence_ref": f"gate:{contract.lower()}",
        "commit_id": "3f1f35e",
        "test_count": 1,
        "summary": f"Selected bounded evidence for {contract}.",
        "repo_relative_path": f"reports/track_a/{contract.lower()}.md",
    }


def _base_payload() -> dict[str, Any]:
    return {
        "project": "QLIB Track A",
        "scope": "F13 Gap Map",
        "source_snapshot_ref": "commit:3f1f35e",
        "observed_surfaces": [
            _selected_surface("F13_SPEC_MATERIALIZED"),
            _selected_surface("JSON_SCHEMAS_MATERIALIZED"),
            _selected_surface("GAP_MAP_MATERIALIZED"),
            _selected_surface("BRIDGE_BOUNDARY_ENFORCED"),
            _selected_surface("EVIDENCE_REQUIRED_ENFORCED"),
            _selected_surface("RAW_LEAK_ENFORCED"),
            _selected_surface("FEEDBACK_LOOP_ENFORCED"),
            _selected_surface("PROOFPACK_MANIFEST_PRESENT"),
            _selected_surface("RELEASE_BOARD_PRESENT"),
            _selected_surface("GATE_RESULTS_PRESENT"),
        ],
        "db_behavior": "NOT_EXECUTED",
        "production_raw_leak_safety": "NOT_VERIFIED",
        "full_regression_safety": "NOT_VERIFIED",
        "actual_proofpack_generation": "NOT_EXECUTED",
        "final_approval": "NOT_RECORDED",
    }


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


def _assert_no_raw_internal_or_secret_surface(gap_map: dict[str, Any]) -> None:
    rendered = "\n".join(_walk(gap_map)).lower()
    assert "raw_text" not in rendered
    assert "raw_prompt" not in rendered
    assert "raw_query" not in rendered
    assert "full_source_text" not in rendered
    assert "internal_path" not in rendered
    assert "h:\\" not in rendered
    assert "c:\\" not in rendered
    assert "file://" not in rendered
    assert "secret" not in rendered
    assert "api_key" not in rendered
    assert "credential" not in rendered
    assert "dsn" not in rendered
    assert "customer confidential" not in rendered
    assert "paid standard" not in rendered


def _assert_no_pass_escalation(gap_map: dict[str, Any]) -> None:
    assert gap_map["not_granted_claims"] == {
        "f13_pass": "NOT_GRANTED",
        "track_a_pass": "NOT_GRANTED",
        "beta_pass": "NOT_GRANTED",
    }


def test_gap_map_preserves_review_required_when_open_items_remain():
    gap_map = build_f13_gap_map(_base_payload())

    assert gap_map["final_recommendation"] == "REVIEW_REQUIRED"
    assert "DB_BEHAVIOR_NOT_VERIFIED" in gap_map["open_items"]
    assert "PRODUCTION_RAW_LEAK_SAFETY_NOT_VERIFIED" in gap_map["open_items"]
    assert "FULL_REGRESSION_SAFETY_NOT_VERIFIED" in gap_map["open_items"]
    assert "ACTUAL_PROOFPACK_GENERATION_NOT_EXECUTED" in gap_map["open_items"]
    assert "FINAL_APPROVAL_NOT_RECORDED" in gap_map["open_items"]
    _assert_no_pass_escalation(gap_map)


def test_gap_map_marks_selected_no_db_surfaces_limited_closed_only():
    payload = _base_payload()
    payload["observed_surfaces"].extend(
        [
            _selected_surface("RETRIEVE_EVIDENCE_SELECTED_CONTRACT"),
            _selected_surface("SKILLUP_BRIDGE_HOLD_FEEDBACK"),
            _selected_surface("FEEDBACK_QUEUE_ITEM"),
            _selected_surface("COURSE_LIBRARY_BINDING"),
            _selected_surface("BETA_RELEASE_BOARD"),
            _selected_surface("F13_PROOFPACK_MANIFEST"),
        ]
    )

    gap_map = build_f13_gap_map(payload)
    selected = {
        item["surface_id"]: item for item in gap_map["observed_surfaces"]
        if item["surface_id"]
        in {
            "RETRIEVE_EVIDENCE_SELECTED_CONTRACT",
            "SKILLUP_BRIDGE_HOLD_FEEDBACK",
            "FEEDBACK_QUEUE_ITEM",
            "COURSE_LIBRARY_BINDING",
            "BETA_RELEASE_BOARD",
            "F13_PROOFPACK_MANIFEST",
        }
    }

    assert selected
    assert {item["status"] for item in selected.values()} <= {"CLOSED_LIMITED", "VERIFIED_LIMITED"}
    assert "PASS" not in {item["status"] for item in selected.values()}
    _assert_no_pass_escalation(gap_map)


def test_gap_map_blocks_pass_when_required_contract_missing():
    gap_map = build_f13_gap_map(
        {
            "project": "QLIB Track A",
            "scope": "F13 Gap Map",
            "source_snapshot_ref": "commit:3f1f35e",
            "observed_surfaces": [
                _selected_surface("GAP_MAP_MATERIALIZED"),
                _selected_surface("BRIDGE_BOUNDARY_ENFORCED"),
            ],
            "db_behavior": "NOT_EXECUTED",
            "production_raw_leak_safety": "NOT_VERIFIED",
            "full_regression_safety": "NOT_VERIFIED",
            "actual_proofpack_generation": "NOT_EXECUTED",
            "final_approval": "NOT_RECORDED",
        }
    )
    reasons = {item["reason"] for item in gap_map["gap_items"]}

    assert "MISSING_CONTRACT:F13_SPEC_MATERIALIZED" in reasons
    assert "MISSING_CONTRACT:JSON_SCHEMAS_MATERIALIZED" in reasons
    assert "MISSING_CONTRACT:RELEASE_BOARD_PRESENT" in reasons
    assert "MISSING_CONTRACT:GATE_RESULTS_PRESENT" in reasons
    assert gap_map["final_recommendation"] == "REVIEW_REQUIRED"
    _assert_no_pass_escalation(gap_map)


def test_gap_map_sanitizes_raw_internal_secret_fields():
    payload = _base_payload()
    payload["observed_surfaces"].append(
        {
            "surface_id": "unsafe_surface",
            "contract": "UNSAFE_SURFACE",
            "status": "PASS",
            "evidence_ref": "gate:unsafe",
            "commit_id": "abc1234",
            "test_count": 1,
            "summary": "customer confidential paid standard raw text from H:\\internal\\source.txt",
            "repo_relative_path": "H:\\internal\\source.txt",
            "raw_text": "do not echo",
            "internal_path": "H:\\internal\\source.txt",
            "api_key": "do-not-echo",
            "dsn": "do-not-echo",
        }
    )
    payload["observed_surfaces"].append(
        {
            "surface_id": "safe_surface",
            "contract": "SAFE_SURFACE",
            "status": "PASS",
            "evidence_ref": "gate:safe",
            "commit_id": "abc1234",
            "test_count": 1,
            "summary": "Safe bounded evidence summary.",
            "repo_relative_path": "reports/track_a/safe_surface.md",
        }
    )

    gap_map = build_f13_gap_map(payload)
    unsafe_surface = [
        item for item in gap_map["observed_surfaces"] if item["surface_id"] == "UNSAFE_SURFACE"
    ][0]
    safe_surface = [
        item for item in gap_map["observed_surfaces"] if item["surface_id"] == "SAFE_SURFACE"
    ][0]

    assert unsafe_surface["summary"] == "redacted_safety_summary"
    assert unsafe_surface["repo_relative_path"] == ""
    assert safe_surface["repo_relative_path"] == "reports/track_a/safe_surface.md"
    _assert_no_raw_internal_or_secret_surface(gap_map)
    _assert_no_pass_escalation(gap_map)


def test_gap_map_id_is_stable_for_same_safe_inputs():
    first = build_f13_gap_map(_base_payload())
    second = build_f13_gap_map(_base_payload())

    assert first["gap_map_id"] == second["gap_map_id"]
    assert first["gap_map_version"] == second["gap_map_version"]
    assert first["closed_items"] == second["closed_items"]
