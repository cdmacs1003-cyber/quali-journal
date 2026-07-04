import sqlite3
from pathlib import Path
from typing import Any

from admin.f13_library_db_evidence_retrieval import (
    probe_sqlite_metadata,
    retrieve_bridge_evidence_from_sqlite,
)
from admin.f13_skillup_answer_hold_adapter import adapt_skillup_answer_hold_response
from admin.f13_skillup_bridge import skillup_answer_from_bridge_response


def _walk(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            out.append(str(key))
            out.extend(_walk(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_walk(child))
        return out
    return [str(value)]


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "library_evidence.sqlite"


def _create_safe_metadata_db(db_path: Path, *, rights_status: str = "INTERNAL") -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE bridge_evidence (
                evidence_id TEXT NOT NULL,
                bridge_trace_id TEXT NOT NULL,
                safe_summary TEXT NOT NULL,
                pointer_uri TEXT,
                raw_text_policy TEXT,
                rights_status TEXT,
                raw_text TEXT,
                path TEXT,
                content TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO bridge_evidence (
                evidence_id,
                bridge_trace_id,
                safe_summary,
                pointer_uri,
                raw_text_policy,
                rights_status,
                raw_text,
                path,
                content
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ev:db-safe-1",
                "btrace:db-safe-1",
                "Synthetic DB-backed safe summary.",
                "qlib://library-db/safe/ev-db-safe-1",
                "SUMMARY_ONLY",
                rights_status,
                "SYNTHETIC_RAW_SENTINEL_MUST_NOT_APPEAR",
                "SYNTHETIC_PATH_SENTINEL_MUST_NOT_APPEAR",
                "SYNTHETIC_CONTENT_SENTINEL_MUST_NOT_APPEAR",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def test_db_backed_bridge_retrieval_returns_safe_metadata_only(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    _create_safe_metadata_db(db_path)

    metadata = probe_sqlite_metadata(db_path)
    table = next(item for item in metadata["tables"] if item["name"] == "bridge_evidence")
    classified = {column["name"]: column["classification"] for column in table["columns"]}
    assert classified["raw_text"] == "FORBIDDEN_RAW_TEXT_PATH_OR_BLOB_NAME"
    assert classified["path"] == "FORBIDDEN_RAW_TEXT_PATH_OR_BLOB_NAME"
    assert classified["content"] == "FORBIDDEN_RAW_TEXT_PATH_OR_BLOB_NAME"
    assert metadata["readonly_uri_mode"] == "mode=ro&immutable=1"

    response = retrieve_bridge_evidence_from_sqlite(db_path, table_name="bridge_evidence")

    assert response["result_status"] == "OK"
    assert response["raw_text_included"] is False
    assert response["internal_path_included"] is False
    assert response["policy_result"]["raw_leak_pass"] is True
    assert response["policy_result"]["rights_pass"] is True
    assert len(response["evidence_items"]) == 1
    evidence = response["evidence_items"][0]
    assert evidence == {
        "evidence_id": "ev:db-safe-1",
        "bridge_trace_id": "btrace:db-safe-1",
        "safe_summary": "Synthetic DB-backed safe summary.",
        "pointer_uri": "qlib://library-db/safe/ev-db-safe-1",
        "raw_text_policy": "SUMMARY_ONLY",
        "rights_status": "INTERNAL",
    }
    rendered = "\n".join(_walk(response))
    assert "SYNTHETIC_RAW_SENTINEL_MUST_NOT_APPEAR" not in rendered
    assert "SYNTHETIC_PATH_SENTINEL_MUST_NOT_APPEAR" not in rendered
    assert "SYNTHETIC_CONTENT_SENTINEL_MUST_NOT_APPEAR" not in rendered


def test_skillup_consumes_db_bridge_response_without_direct_db_or_public_pointer(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    _create_safe_metadata_db(db_path)

    bridge_response = retrieve_bridge_evidence_from_sqlite(db_path, table_name="bridge_evidence")
    bridge_response_with_context = {
        **bridge_response,
        "role": "student",
        "evidence_depth": "student_safe",
        "course_id": "course:skillup-db",
        "module_id": "module:skillup-db",
        "binding_id": "binding:skillup-db",
        "tenant_id": "tenant:skillup-db",
        "organization_id": "org:skillup-db",
        "cohort_id": "cohort:skillup-db",
    }
    helper_response = skillup_answer_from_bridge_response(bridge_response_with_context)
    adapted = adapt_skillup_answer_hold_response(
        helper_response,
        request_context={"requester_module": "Skillup"},
        bridge_payload=bridge_response_with_context,
    )

    assert adapted["result_status"] == "OK"
    assert adapted["answer_status"] == "ANSWERED"
    assert adapted["answer"] == "Synthetic DB-backed safe summary."
    assert adapted["raw_text_included"] is False
    assert adapted["internal_path_included"] is False
    assert adapted["policy"]["raw_leak_check_passed"] is True
    assert adapted["policy"]["rights_check_passed"] is True
    assert adapted["evidence"][0]["evidence_id"] == "ev:db-safe-1"
    assert "pointer" not in adapted["evidence"][0]
    rendered = "\n".join(_walk(adapted)).lower()
    assert "qlib://" not in rendered
    assert "SYNTHETIC_RAW_SENTINEL".lower() not in rendered
    assert "SYNTHETIC_PATH_SENTINEL".lower() not in rendered
    assert "SYNTHETIC_CONTENT_SENTINEL".lower() not in rendered


def test_not_verified_db_metadata_returns_hold_without_public_summary(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    _create_safe_metadata_db(db_path, rights_status="NOT_VERIFIED")

    response = retrieve_bridge_evidence_from_sqlite(db_path, table_name="bridge_evidence")

    assert response["result_status"] == "HOLD"
    assert response["evidence_items"] == []
    assert response["hold_reason"] == "rights_status is not verified"
    assert response["policy_result"]["rights_pass"] is False
    rendered = "\n".join(_walk(response))
    assert "Synthetic DB-backed safe summary." not in rendered
    assert "qlib://" not in rendered
    assert "SYNTHETIC_RAW_SENTINEL_MUST_NOT_APPEAR" not in rendered
    assert "SYNTHETIC_PATH_SENTINEL_MUST_NOT_APPEAR" not in rendered
    assert "SYNTHETIC_CONTENT_SENTINEL_MUST_NOT_APPEAR" not in rendered


def test_raw_index_schema_holds_without_selecting_body_or_path_columns(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("CREATE TABLE pages (text TEXT, doc_id TEXT, page INTEGER, path TEXT)")
        connection.execute(
            "INSERT INTO pages (text, doc_id, page, path) VALUES (?, ?, ?, ?)",
            (
                "SYNTHETIC_RAW_INDEX_TEXT_MUST_NOT_APPEAR",
                "SYNTHETIC-DOC-1",
                1,
                "SYNTHETIC_INDEX_PATH_MUST_NOT_APPEAR",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    metadata = probe_sqlite_metadata(db_path)
    table = metadata["tables"][0]
    classified = {column["name"]: column["classification"] for column in table["columns"]}
    assert classified["text"] == "FORBIDDEN_RAW_TEXT_PATH_OR_BLOB_NAME"
    assert classified["path"] == "FORBIDDEN_RAW_TEXT_PATH_OR_BLOB_NAME"

    response = retrieve_bridge_evidence_from_sqlite(db_path, table_name="pages")

    assert response["result_status"] == "HOLD"
    assert response["evidence_items"] == []
    assert "safe_summary" in response["hold_reason"]
    rendered = "\n".join(_walk(response))
    assert "SYNTHETIC_RAW_INDEX_TEXT_MUST_NOT_APPEAR" not in rendered
    assert "SYNTHETIC_INDEX_PATH_MUST_NOT_APPEAR" not in rendered
