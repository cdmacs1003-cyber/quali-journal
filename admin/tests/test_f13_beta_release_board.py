from typing import Any

from admin.f13_beta_release_board import build_beta_release_board


def _bounded_evidence() -> list[dict[str, Any]]:
    return [
        {
            "record_id": "retrieve_evidence_contract_tests",
            "status": "PASS",
            "commit_id": "37c86a4",
            "test_count": 24,
            "summary": "Selected no-DB retrieve-evidence contract checks recorded.",
        },
        {
            "record_id": "skillup_bridge_route_runtime",
            "status": "PASS",
            "commit_id": "24f5b53",
            "test_count": 1,
            "summary": "Localhost Skillup Bridge HOLD diagnostic returned feedback queue item.",
        },
        {
            "record_id": "course_library_binding",
            "status": "PASS",
            "commit_id": "37c86a4",
            "test_count": 4,
            "summary": "No-DB course library binding selected contract checks recorded.",
        },
    ]


def _base_payload() -> dict[str, Any]:
    return {
        "scope": "Track A Skillup Beta",
        "evidence_records": _bounded_evidence(),
        "db_behavior": "NOT_EXECUTED",
        "production_raw_leak_safety": "NOT_VERIFIED",
        "full_regression_safety": "NOT_VERIFIED",
        "proofpack_status": "NOT_EXECUTED",
        "gate_matrix_status": "NOT_EXECUTED",
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


def _assert_no_raw_internal_or_secret_surface(board: dict[str, Any]) -> None:
    rendered = "\n".join(_walk(board)).lower()
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
    assert "customer confidential" not in rendered


def test_beta_release_board_preserves_not_granted_when_open_items_remain():
    board = build_beta_release_board(_base_payload())

    assert board["not_granted_claims"]["f13_pass"] == "NOT_GRANTED"
    assert board["not_granted_claims"]["track_a_pass"] == "NOT_GRANTED"
    assert board["not_granted_claims"]["beta_pass"] == "NOT_GRANTED"
    assert board["recommendation"] in {"REVIEW_REQUIRED", "READY_FOR_REVIEW"}
    assert board["recommendation"] != "PASS"
    assert board["approval_status"] == "NOT_APPROVED"
    assert "DB_BEHAVIOR_NOT_VERIFIED" in board["required_open_items"]
    assert "PRODUCTION_RAW_LEAK_SAFETY_NOT_VERIFIED" in board["required_open_items"]
    assert "FULL_REGRESSION_SAFETY_NOT_VERIFIED" in board["required_open_items"]


def test_beta_release_board_summarizes_bounded_evidence_without_raw_leak():
    payload = _base_payload()
    payload["evidence_records"].append(
        {
            "record_id": "unsafe_record",
            "status": "PASS",
            "commit_id": "abc1234",
            "test_count": 1,
            "summary": "customer confidential raw text from H:\\internal\\source.txt",
            "raw_text": "do not echo",
            "internal_path": "H:\\internal\\source.txt",
            "api_key": "do-not-echo",
        }
    )

    board = build_beta_release_board(payload)

    assert board["evidence_summary"]
    assert any(item["summary"] == "redacted_safety_summary" for item in board["evidence_summary"])
    _assert_no_raw_internal_or_secret_surface(board)


def test_beta_release_board_requires_proofpack_for_f13_or_track_a_pass():
    payload = _base_payload()
    payload["f13_pass"] = "PASS"
    payload["track_a_pass"] = "PASS"
    payload["beta_pass"] = "PASS"
    payload["proofpack_status"] = "NOT_EXECUTED"
    payload["gate_matrix_status"] = "NOT_EXECUTED"

    board = build_beta_release_board(payload)

    assert board["not_granted_claims"] == {
        "f13_pass": "NOT_GRANTED",
        "track_a_pass": "NOT_GRANTED",
        "beta_pass": "NOT_GRANTED",
    }
    assert "PROOFPACK_NOT_EXECUTED" in board["required_open_items"]
    assert "GATE_MATRIX_NOT_COMPLETE" in board["required_open_items"]
    assert board["recommendation"] != "PASS"


def test_beta_release_board_dedup_or_stable_id_for_same_scope():
    first = build_beta_release_board(_base_payload())
    second = build_beta_release_board(_base_payload())

    assert first["release_board_id"] == second["release_board_id"]
    assert first["scope"] == second["scope"]
