"""Materialize Bridge-safe Library metadata sidecars.

This module writes task-owned SQLite/JSON sidecars from already-approved safe
metadata records. It does not inspect production Library bodies, config, DSNs,
or production DB sources.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from admin.f13_runtime_guard import (
    RAW_TEXT_POLICY_REDACTED_SUMMARY_ONLY,
    RAW_TEXT_POLICY_SUMMARY_ONLY,
    RESULT_HOLD,
    RESULT_OK,
    RIGHTS_INTERNAL,
    RIGHTS_LICENSED,
    RIGHTS_NOT_VERIFIED,
    RIGHTS_PUBLIC,
    RIGHTS_UNKNOWN,
    normalize_raw_text_policy,
    normalize_rights_status,
)


DEFAULT_TABLE_NAME = "bridge_evidence"
SAFE_SIDECAR_SUMMARY_SOURCE = "SAFE_SIDECAR_APPROVED_SUMMARY"
METADATA_DERIVED_NOT_SEMANTIC = "METADATA_DERIVED_NOT_SEMANTIC"

ACCEPTED_OK = "ACCEPTED_OK"
HOLD_ONLY_EXCLUDED = "HOLD_ONLY_EXCLUDED"
HOLD_ONLY_INCLUDED = "HOLD_ONLY_INCLUDED"
REJECTED = "REJECTED"

_APPROVED_SEED_REVIEW_STATUS = "APPROVED_FOR_LIBRARY_EVIDENCE"
_APPROVED_SEED_STATUS = "APPROVED_WITH_LIMITS"
_SAFE_SUMMARY_ONLY = "SAFE_SUMMARY_ONLY"
_MAX_RECORDS = 100

_REQUIRED_OUTPUT_FIELDS = (
    "evidence_id",
    "bridge_trace_id",
    "safe_summary",
    "rights_status",
    "raw_text_policy",
    "summary_source",
    "semantic_summary_verified",
    "raw_text_exposed",
    "production_path_exposed",
)

_SQLITE_COLUMNS = (
    "evidence_id",
    "bridge_trace_id",
    "safe_summary",
    "pointer_uri",
    "raw_text_policy",
    "rights_status",
    "summary_source",
    "semantic_summary_verified",
    "raw_text_exposed",
    "production_path_exposed",
    "source_doc_kind",
)

_OK_RIGHTS = {RIGHTS_PUBLIC, RIGHTS_INTERNAL, RIGHTS_LICENSED}
_OK_RAW_POLICIES = {RAW_TEXT_POLICY_SUMMARY_ONLY, RAW_TEXT_POLICY_REDACTED_SUMMARY_ONLY}
_APPROVED_SUMMARY_SOURCES = {
    "APPROVED_SAFE_SUMMARY",
    "APPROVED_SAFE_SHORT_ANSWER",
    "CURATED_SAFE_SUMMARY",
    SAFE_SIDECAR_SUMMARY_SOURCE,
    "SYNTHETIC_SAFE_SUMMARY",
    "VERIFIED_SEMANTIC_SUMMARY",
}

_FORBIDDEN_OUTPUT_VALUE_MARKERS = (
    "h:\\",
    "c:\\",
    "file://",
    "/mnt/",
    "/home/",
    "/tmp/",
    ".env",
    "api_key",
    "credential",
    "private_key",
    "secret",
    "service-account",
    "token",
    "raw standard",
    "raw_standard",
    "paid standard raw",
    "raw paid standard",
    "full_text",
    "pdf_text",
    "clause_text",
)


def _resolved_lower(path: Path) -> str:
    return str(path.resolve()).lower()


def _assert_sidecar_output_path(path: Path) -> None:
    lowered = _resolved_lower(path)
    if "\\library\\" in lowered:
        raise ValueError("safe metadata sidecar output must not be under a Library root")
    if path.name.lower() in {"ripple_index.sqlite", "chat.db"}:
        raise ValueError("safe metadata sidecar output must not target a production DB name")


def _can_overwrite_temp_sidecar(path: Path) -> bool:
    lowered = _resolved_lower(path)
    return "\\tmp\\" in lowered or "\\temp\\" in lowered


def _safe_token(value: object) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in {0, 1}:
            return bool(value)
        return None
    token = _safe_token(value)
    if token in {"TRUE", "YES", "Y", "1"}:
        return True
    if token in {"FALSE", "NO", "N", "0"}:
        return False
    return None


def _safe_text(value: object, max_length: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > max_length:
        return None
    if any(ord(char) < 32 for char in text):
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in _FORBIDDEN_OUTPUT_VALUE_MARKERS):
        return None
    return text


def _bridge_trace_id_from_seed(record: Mapping[str, Any]) -> str | None:
    trace_seed = _safe_text(record.get("bridge_trace_seed"), 120)
    if trace_seed is None:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-")
    if any(char not in allowed for char in trace_seed):
        return None
    if trace_seed.startswith("btrace:"):
        return _safe_text(trace_seed, 160)
    if trace_seed.startswith("btrace-seed-"):
        suffix = trace_seed[len("btrace-seed-") :]
        return _safe_text(f"btrace:library-seed:{suffix}", 160)
    return None


def _bridge_trace_id_from_record(record: Mapping[str, Any]) -> str | None:
    trace_id = _safe_text(record.get("bridge_trace_id"), 160)
    if trace_id and trace_id.startswith("btrace:"):
        return trace_id
    return _bridge_trace_id_from_seed(record)


def _safe_pointer_uri(record: Mapping[str, Any]) -> str | None:
    pointer_uri = _safe_text(record.get("pointer_uri") or record.get("pointer"), 512)
    if pointer_uri is None:
        return None
    if pointer_uri.startswith(("qlib://", "pointer://")):
        return pointer_uri
    return None


def _is_approved_seed(record: Mapping[str, Any]) -> bool:
    return (
        _safe_token(record.get("evidence_type")) == _SAFE_SUMMARY_ONLY
        and _safe_token(record.get("review_status")) == _APPROVED_SEED_REVIEW_STATUS
        and _safe_token(record.get("approval_status")) == _APPROVED_SEED_STATUS
        and record.get("raw_text_excluded") is True
        and record.get("standard_raw_text_not_included") is True
    )


def _candidate_from_record(record: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    evidence_id = _safe_text(record.get("evidence_id"), 120)
    bridge_trace_id = _bridge_trace_id_from_record(record)
    safe_summary = _safe_text(record.get("safe_summary") or record.get("safe_short_answer"), 2000)
    pointer_uri = _safe_pointer_uri(record)

    rights_status_source = record.get("rights_status")
    raw_text_policy_source = record.get("raw_text_policy")
    summary_source_source = record.get("summary_source")
    semantic_verified_source = record.get("semantic_summary_verified")
    raw_text_exposed_source = record.get("raw_text_exposed")
    production_path_exposed_source = record.get("production_path_exposed")

    if _is_approved_seed(record):
        summary_source_source = summary_source_source or SAFE_SIDECAR_SUMMARY_SOURCE
        semantic_verified_source = False if semantic_verified_source is None else semantic_verified_source
        raw_text_exposed_source = False
        production_path_exposed_source = False

    missing = []
    required_sources = {
        "evidence_id": evidence_id,
        "bridge_trace_id": bridge_trace_id,
        "safe_summary": safe_summary,
        "rights_status": rights_status_source,
        "raw_text_policy": raw_text_policy_source,
        "summary_source": summary_source_source,
        "semantic_summary_verified": semantic_verified_source,
        "raw_text_exposed": raw_text_exposed_source,
        "production_path_exposed": production_path_exposed_source,
    }
    for field, value in required_sources.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    if missing:
        return {}, "missing required safe sidecar field: " + ",".join(sorted(missing))

    semantic_verified = _bool_or_none(semantic_verified_source)
    raw_text_exposed = _bool_or_none(raw_text_exposed_source)
    production_path_exposed = _bool_or_none(production_path_exposed_source)
    if semantic_verified is None:
        return {}, "semantic_summary_verified must be explicit true/false"
    if raw_text_exposed is None or production_path_exposed is None:
        return {}, "exposure audit flags must be explicit true/false"

    raw_policy = normalize_raw_text_policy(raw_text_policy_source)
    rights_status = normalize_rights_status(rights_status_source)
    summary_source = _safe_token(summary_source_source)
    source_doc_kind = _safe_text(record.get("source_doc_kind"), 120)

    candidate: dict[str, Any] = {
        "evidence_id": evidence_id,
        "bridge_trace_id": bridge_trace_id,
        "safe_summary": safe_summary,
        "pointer_uri": pointer_uri,
        "raw_text_policy": raw_policy,
        "rights_status": rights_status,
        "summary_source": summary_source,
        "semantic_summary_verified": semantic_verified,
        "raw_text_exposed": raw_text_exposed,
        "production_path_exposed": production_path_exposed,
    }
    if source_doc_kind is not None:
        candidate["source_doc_kind"] = source_doc_kind
    return candidate, None


def _record_status(candidate: Mapping[str, Any]) -> tuple[str, str]:
    if candidate.get("raw_text_exposed") is True:
        return REJECTED, "raw_text_exposed=true"
    if candidate.get("production_path_exposed") is True:
        return REJECTED, "production_path_exposed=true"

    rights_status = str(candidate.get("rights_status") or "")
    if rights_status in {RIGHTS_NOT_VERIFIED, RIGHTS_UNKNOWN}:
        return HOLD_ONLY_EXCLUDED, "rights_status requires HOLD"

    raw_policy = str(candidate.get("raw_text_policy") or "")
    if raw_policy not in _OK_RAW_POLICIES:
        return HOLD_ONLY_EXCLUDED, "raw_text_policy is not safe-summary answer policy"

    summary_source = _safe_token(candidate.get("summary_source"))
    semantic_verified = candidate.get("semantic_summary_verified") is True
    if summary_source == METADATA_DERIVED_NOT_SEMANTIC and not semantic_verified:
        return HOLD_ONLY_EXCLUDED, "metadata-derived summary is not semantic verified"
    if summary_source not in _APPROVED_SUMMARY_SOURCES and not semantic_verified:
        return HOLD_ONLY_EXCLUDED, "summary_source is not approved for answer"

    if rights_status not in _OK_RIGHTS:
        return HOLD_ONLY_EXCLUDED, "rights_status is not approved for OK sidecar"

    return ACCEPTED_OK, "approved safe summary record"


def _matrix_item(
    record: Mapping[str, Any],
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    evidence_id = _safe_text(record.get("evidence_id"), 120)
    if evidence_id is None:
        evidence_id = "MISSING_EVIDENCE_ID"
    return {
        "evidence_id": evidence_id,
        "status": status,
        "reason": reason,
        "rights_status": normalize_rights_status(record.get("rights_status")),
        "raw_text_policy": normalize_raw_text_policy(record.get("raw_text_policy")),
        "summary_source": _safe_token(record.get("summary_source")),
    }


def _quote_identifier(identifier: str) -> str:
    if "\x00" in identifier:
        raise ValueError("SQLite identifier contains NUL")
    return '"' + identifier.replace('"', '""') + '"'


def _create_sqlite_sidecar(
    sqlite_path: Path,
    table_name: str,
    rows: list[dict[str, Any]],
) -> None:
    _assert_sidecar_output_path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        if not _can_overwrite_temp_sidecar(sqlite_path):
            raise FileExistsError("existing safe sidecar output is not under a temp/task path")
        sqlite_path.unlink()
    quoted_table = _quote_identifier(table_name)
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            f"""
            CREATE TABLE {quoted_table} (
                evidence_id TEXT NOT NULL,
                bridge_trace_id TEXT NOT NULL,
                safe_summary TEXT NOT NULL,
                pointer_uri TEXT,
                raw_text_policy TEXT NOT NULL,
                rights_status TEXT NOT NULL,
                summary_source TEXT NOT NULL,
                semantic_summary_verified INTEGER NOT NULL,
                raw_text_exposed INTEGER NOT NULL,
                production_path_exposed INTEGER NOT NULL,
                source_doc_kind TEXT
            )
            """
        )
        for row in rows:
            connection.execute(
                f"""
                INSERT INTO {quoted_table} (
                    evidence_id,
                    bridge_trace_id,
                    safe_summary,
                    pointer_uri,
                    raw_text_policy,
                    rights_status,
                    summary_source,
                    semantic_summary_verified,
                    raw_text_exposed,
                    production_path_exposed,
                    source_doc_kind
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(
                    row.get(column)
                    if column
                    not in {
                        "semantic_summary_verified",
                        "raw_text_exposed",
                        "production_path_exposed",
                    }
                    else int(bool(row.get(column)))
                    for column in _SQLITE_COLUMNS
                ),
            )
        connection.commit()


def _write_json_sidecar(
    json_path: Path,
    *,
    table_name: str,
    rows: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
) -> None:
    _assert_sidecar_output_path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if json_path.exists() and not _can_overwrite_temp_sidecar(json_path):
        raise FileExistsError("existing safe sidecar output is not under a temp/task path")
    payload = {
        "sidecar_contract_version": "R9ZNW-342",
        "table_name": table_name,
        "required_fields": list(_REQUIRED_OUTPUT_FIELDS),
        "records": rows,
        "input_matrix": matrix,
        "raw_text_included": False,
        "production_path_included": False,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialize_safe_metadata_sidecar(
    records: Iterable[Mapping[str, Any]],
    *,
    sqlite_path: str | Path,
    json_path: str | Path | None = None,
    table_name: str = DEFAULT_TABLE_NAME,
    include_hold_only: bool = False,
) -> dict[str, Any]:
    """Materialize accepted safe metadata records into SQLite and optional JSON."""

    input_records = list(records)[:_MAX_RECORDS]
    rows: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    accepted_count = 0
    hold_only_count = 0
    rejected_count = 0

    for record in input_records:
        candidate, error = _candidate_from_record(record)
        if error is not None:
            rejected_count += 1
            matrix.append(_matrix_item(record, status=REJECTED, reason=error))
            continue

        status, reason = _record_status(candidate)
        if status == ACCEPTED_OK:
            accepted_count += 1
            rows.append(candidate)
            matrix.append(_matrix_item(candidate, status=status, reason=reason))
            continue
        if status == REJECTED:
            rejected_count += 1
            matrix.append(_matrix_item(candidate, status=status, reason=reason))
            continue

        hold_only_count += 1
        if include_hold_only:
            rows.append(candidate)
            matrix.append(_matrix_item(candidate, status=HOLD_ONLY_INCLUDED, reason=reason))
        else:
            matrix.append(_matrix_item(candidate, status=HOLD_ONLY_EXCLUDED, reason=reason))

    sqlite_target = Path(sqlite_path)
    _create_sqlite_sidecar(sqlite_target, table_name, rows)
    if json_path is not None:
        _write_json_sidecar(Path(json_path), table_name=table_name, rows=rows, matrix=matrix)

    result_status = RESULT_OK if accepted_count else RESULT_HOLD
    return {
        "result_status": result_status,
        "table_name": table_name,
        "sqlite_path": str(sqlite_target),
        "json_path": str(json_path) if json_path is not None else None,
        "input_count": len(input_records),
        "accepted_count": accepted_count,
        "hold_only_count": hold_only_count,
        "rejected_count": rejected_count,
        "rows_written": len(rows),
        "include_hold_only": include_hold_only,
        "input_matrix": matrix,
        "raw_text_included": False,
        "production_path_included": False,
    }


def load_safe_metadata_records_from_json_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load explicit JSON metadata records from caller-approved file paths."""

    records: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8-sig") as input_file:
            payload = json.load(input_file)
        if isinstance(payload, list):
            records.extend(dict(item) for item in payload if isinstance(item, Mapping))
        elif isinstance(payload, Mapping):
            records.append(dict(payload))
    return records


__all__ = [
    "ACCEPTED_OK",
    "DEFAULT_TABLE_NAME",
    "HOLD_ONLY_EXCLUDED",
    "HOLD_ONLY_INCLUDED",
    "REJECTED",
    "SAFE_SIDECAR_SUMMARY_SOURCE",
    "load_safe_metadata_records_from_json_files",
    "materialize_safe_metadata_sidecar",
]
