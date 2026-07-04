from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from admin.f13_library_db_evidence_retrieval import retrieve_bridge_evidence_from_sqlite
from admin.f13_library_safe_metadata_materializer import (
    ACCEPTED_OK,
    HOLD_ONLY_EXCLUDED,
    HOLD_ONLY_INCLUDED,
    REJECTED,
    SAFE_SIDECAR_SUMMARY_SOURCE,
    load_safe_metadata_records_from_json_files,
    materialize_safe_metadata_sidecar,
)
from admin.f13_skillup_answer_hold_adapter import adapt_skillup_answer_hold_response
from admin.f13_skillup_bridge import skillup_answer_from_bridge_response


REPO_ROOT = Path(__file__).resolve().parents[2]
SOLDERING_SEED_PATH = (
    REPO_ROOT
    / "data"
    / "library"
    / "evidence_seeds"
    / "soldering"
    / "ev-soldering-safe-summary-v1.json"
)


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


def _direct_record(**overrides: Any) -> dict[str, Any]:
    record = {
        "evidence_id": "ev:materializer-safe-1",
        "bridge_trace_id": "btrace:materializer:safe-1",
        "safe_summary": "Synthetic approved sidecar summary.",
        "pointer_uri": "qlib://library/evidence_seeds/materializer/ev-materializer-safe-1",
        "raw_text_policy": "SUMMARY_ONLY",
        "rights_status": "INTERNAL",
        "summary_source": SAFE_SIDECAR_SUMMARY_SOURCE,
        "semantic_summary_verified": False,
        "raw_text_exposed": False,
        "production_path_exposed": False,
    }
    record.update(overrides)
    return record


def _row_count(db_path: Path) -> int:
    with sqlite3.connect(db_path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM bridge_evidence").fetchone()[0])


def _skillup_context(bridge_response: dict[str, Any]) -> dict[str, Any]:
    return {
        **bridge_response,
        "role": "student",
        "evidence_depth": "student_safe",
        "course_id": "course:r342-materializer",
        "module_id": "module:r342-materializer",
        "binding_id": "binding:r342-materializer",
        "tenant_id": "tenant:r342-materializer",
        "organization_id": "org:r342-materializer",
        "cohort_id": "cohort:r342-materializer",
    }


def test_materializes_approved_seed_to_sqlite_json_and_bridge_ok(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "materialized_safe_metadata_sidecar.sqlite"
    json_path = tmp_path / "materialized_safe_metadata_sidecar.json"
    seed_records = load_safe_metadata_records_from_json_files([SOLDERING_SEED_PATH])

    result = materialize_safe_metadata_sidecar(
        seed_records,
        sqlite_path=sqlite_path,
        json_path=json_path,
    )

    assert result["result_status"] == "OK"
    assert result["accepted_count"] == 1
    assert result["hold_only_count"] == 0
    assert result["rejected_count"] == 0
    assert result["rows_written"] == 1
    assert result["input_matrix"][0]["status"] == ACCEPTED_OK
    assert _row_count(sqlite_path) == 1

    sidecar_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar_json["table_name"] == "bridge_evidence"
    assert len(sidecar_json["records"]) == 1
    materialized = sidecar_json["records"][0]
    for field in (
        "evidence_id",
        "bridge_trace_id",
        "safe_summary",
        "rights_status",
        "raw_text_policy",
        "summary_source",
        "semantic_summary_verified",
        "raw_text_exposed",
        "production_path_exposed",
    ):
        assert field in materialized
    assert materialized["summary_source"] == SAFE_SIDECAR_SUMMARY_SOURCE
    assert materialized["raw_text_exposed"] is False
    assert materialized["production_path_exposed"] is False

    bridge_response = retrieve_bridge_evidence_from_sqlite(sqlite_path)
    assert bridge_response["result_status"] == "OK"
    assert len(bridge_response["evidence_items"]) == 1
    evidence = bridge_response["evidence_items"][0]
    assert evidence["evidence_id"] == "ev-soldering-safe-summary-v1"
    assert evidence["bridge_trace_id"] == "btrace:library-seed:soldering-safe-summary-v1"
    assert evidence["rights_status"] == "INTERNAL"
    assert evidence["raw_text_policy"] == "SUMMARY_ONLY"

    bridge_context = _skillup_context(bridge_response)
    helper_response = skillup_answer_from_bridge_response(bridge_context)
    adapted = adapt_skillup_answer_hold_response(
        helper_response,
        request_context={"requester_module": "Skillup"},
        bridge_payload=bridge_context,
    )

    assert adapted["result_status"] == "OK"
    assert adapted["answer_status"] == "ANSWERED"
    assert adapted["raw_text_included"] is False
    assert adapted["internal_path_included"] is False
    assert adapted["evidence"][0]["evidence_id"] == "ev-soldering-safe-summary-v1"
    rendered = "\n".join(_walk(adapted)).lower()
    for forbidden in ("qlib://", "file://", "h:\\", "c:\\", "secret", "token", "credential"):
        assert forbidden not in rendered


def test_not_verified_record_excluded_by_default_and_can_be_hold_only(tmp_path: Path) -> None:
    hold_summary = "Hold-only safe summary must not become public."
    record = _direct_record(
        evidence_id="ev:materializer-hold-1",
        bridge_trace_id="btrace:materializer:hold-1",
        safe_summary=hold_summary,
        rights_status="NOT_VERIFIED",
    )

    default_sqlite = tmp_path / "default_excludes_not_verified.sqlite"
    default_result = materialize_safe_metadata_sidecar([record], sqlite_path=default_sqlite)

    assert default_result["result_status"] == "HOLD"
    assert default_result["accepted_count"] == 0
    assert default_result["hold_only_count"] == 1
    assert default_result["input_matrix"][0]["status"] == HOLD_ONLY_EXCLUDED
    assert _row_count(default_sqlite) == 0

    included_sqlite = tmp_path / "included_hold_only.sqlite"
    included_result = materialize_safe_metadata_sidecar(
        [record],
        sqlite_path=included_sqlite,
        include_hold_only=True,
    )

    assert included_result["accepted_count"] == 0
    assert included_result["rows_written"] == 1
    assert included_result["input_matrix"][0]["status"] == HOLD_ONLY_INCLUDED

    bridge_response = retrieve_bridge_evidence_from_sqlite(included_sqlite)
    assert bridge_response["result_status"] == "HOLD"
    assert bridge_response["evidence_items"] == []
    assert bridge_response["policy_result"]["rights_pass"] is False
    rendered = "\n".join(_walk(bridge_response))
    assert hold_summary not in rendered
    assert "qlib://" not in rendered


def test_materializer_rejects_missing_fields_and_exposure_flags(tmp_path: Path) -> None:
    missing_summary = _direct_record(evidence_id="ev:missing-summary")
    missing_summary.pop("safe_summary")
    raw_exposed = _direct_record(
        evidence_id="ev:raw-exposed",
        raw_text_exposed=True,
    )
    path_exposed = _direct_record(
        evidence_id="ev:path-exposed",
        production_path_exposed=True,
    )

    result = materialize_safe_metadata_sidecar(
        [missing_summary, raw_exposed, path_exposed],
        sqlite_path=tmp_path / "rejects.sqlite",
    )

    assert result["result_status"] == "HOLD"
    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 3
    assert {item["status"] for item in result["input_matrix"]} == {REJECTED}
    assert _row_count(tmp_path / "rejects.sqlite") == 0


def test_materializer_holds_metadata_derived_and_pointer_only_records(tmp_path: Path) -> None:
    metadata_derived = _direct_record(
        evidence_id="ev:metadata-derived",
        summary_source="METADATA_DERIVED_NOT_SEMANTIC",
        semantic_summary_verified=False,
    )
    pointer_only = _direct_record(
        evidence_id="ev:pointer-only",
        raw_text_policy="POINTER_ONLY",
    )

    result = materialize_safe_metadata_sidecar(
        [metadata_derived, pointer_only],
        sqlite_path=tmp_path / "hold_only.sqlite",
    )

    assert result["accepted_count"] == 0
    assert result["hold_only_count"] == 2
    assert {item["status"] for item in result["input_matrix"]} == {HOLD_ONLY_EXCLUDED}
    assert _row_count(tmp_path / "hold_only.sqlite") == 0


def test_materializer_rejects_unsafe_output_values(tmp_path: Path) -> None:
    unsafe_summary = _direct_record(
        evidence_id="ev:unsafe-value",
        safe_summary="This summary includes C:\\unsafe\\path and must be rejected.",
    )

    result = materialize_safe_metadata_sidecar(
        [unsafe_summary],
        sqlite_path=tmp_path / "unsafe_value.sqlite",
    )

    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 1
    assert result["input_matrix"][0]["status"] == REJECTED
    assert "safe_summary" in result["input_matrix"][0]["reason"]
    assert _row_count(tmp_path / "unsafe_value.sqlite") == 0
