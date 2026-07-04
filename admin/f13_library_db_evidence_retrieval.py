"""Bridge-only SQLite Library Evidence retrieval helper.

The helper accepts an explicit SQLite path from its caller, opens it in
read-only URI mode, and returns the existing Bridge evidence/HOLD response
shape. It never reads config, environment DSNs, raw body columns, or paths.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from admin.f13_runtime_guard import (
    RAW_TEXT_POLICY_POINTER_ONLY,
    RESULT_DENIED,
    RESULT_HOLD,
    RESULT_OK,
    RIGHTS_NOT_VERIFIED,
    decide_bridge_result,
    project_bridge_safe_evidence,
)

_DEFAULT_LIMIT = 5
_MAX_LIMIT = 10

_BRIDGE_SCHEMA_REQUIRED_FIELDS = {
    "evidence_id",
    "bridge_trace_id",
    "safe_summary",
    "pointer_uri",
    "raw_text_policy",
    "rights_status",
}

_COLUMN_ALIASES = {
    "evidence_id": ("evidence_id", "id", "node_id", "doc_id"),
    "bridge_trace_id": ("bridge_trace_id", "trace_id"),
    "safe_summary": ("safe_summary", "safe_short_answer"),
    "pointer_uri": ("pointer_uri", "pointer"),
    "raw_text_policy": ("raw_text_policy",),
    "rights_status": ("rights_status",),
    "source_doc_kind": ("source_doc_kind",),
    "summary_source": ("summary_source",),
    "semantic_summary_verified": ("semantic_summary_verified",),
    "raw_text_exposed": ("raw_text_exposed",),
    "production_path_exposed": ("production_path_exposed",),
}

_REQUIRED_SAFE_INDEX_POLICY_FIELDS = (
    "summary_source",
    "semantic_summary_verified",
    "raw_text_exposed",
    "production_path_exposed",
)

_METADATA_DERIVED_NOT_SEMANTIC = "METADATA_DERIVED_NOT_SEMANTIC"

_APPROVED_SAFE_SUMMARY_SOURCES = {
    "APPROVED_SAFE_SUMMARY",
    "APPROVED_SAFE_SHORT_ANSWER",
    "CURATED_SAFE_SUMMARY",
    "SAFE_SIDECAR_APPROVED_SUMMARY",
    "SYNTHETIC_SAFE_SUMMARY",
    "VERIFIED_SEMANTIC_SUMMARY",
}

_SAFE_MAPPING_COLUMNS = {
    "evidence_id",
    "id",
    "node_id",
    "doc_id",
    "bridge_trace_id",
    "trace_id",
    "safe_summary",
    "safe_short_answer",
    "pointer_uri",
    "pointer",
    "raw_text_policy",
    "rights_status",
    "source_doc_kind",
    "summary_source",
    "semantic_summary_verified",
    "raw_text_exposed",
    "production_path_exposed",
}

_FORBIDDEN_EXACT_COLUMNS = {
    "text",
    "path",
    "body",
    "content",
    "snippet",
    "excerpt",
    "markdown",
    "yaml",
    "block",
    "sz",
}

_FORBIDDEN_COLUMN_TOKENS = (
    "raw",
    "body",
    "full_text",
    "pdf_text",
    "clause_text",
    "content",
    "snippet",
    "excerpt",
    "markdown",
    "yaml",
    "path",
)

_SAFE_AUDIT_COLUMNS = {"raw_text_policy", "raw_text_exposed", "production_path_exposed"}


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _quote_identifier(identifier: str) -> str:
    if "\x00" in identifier:
        raise ValueError("SQLite identifier contains NUL")
    return '"' + identifier.replace('"', '""') + '"'


def _readonly_uri(db_path: str | Path) -> str:
    path = Path(db_path)
    return f"{path.resolve().as_uri()}?mode=ro&immutable=1"


def _connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError("SQLite evidence source was not found")
    connection = sqlite3.connect(_readonly_uri(path), uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        connection.execute("PRAGMA trusted_schema=OFF")
    except sqlite3.DatabaseError:
        pass
    return connection


def _bounded_limit(value: int | None) -> int:
    if value is None:
        return _DEFAULT_LIMIT
    return min(max(int(value), 1), _MAX_LIMIT)


def _is_forbidden_column_name(name: object, declared_type: object = "") -> bool:
    lowered = str(name or "").strip().lower()
    if lowered in _SAFE_AUDIT_COLUMNS:
        return False
    if str(declared_type or "").strip().upper() == "BLOB":
        return True
    if lowered in _FORBIDDEN_EXACT_COLUMNS:
        return True
    return any(token in lowered for token in _FORBIDDEN_COLUMN_TOKENS)


def _column_classification(name: object, declared_type: object = "") -> str:
    if _is_forbidden_column_name(name, declared_type):
        return "FORBIDDEN_RAW_TEXT_PATH_OR_BLOB_NAME"
    return "METADATA_NAME_ONLY_REVIEWED"


def _safe_column_name(name: object, declared_type: object = "") -> bool:
    lowered = str(name or "").strip().lower()
    return lowered in _SAFE_MAPPING_COLUMNS and not _is_forbidden_column_name(lowered, declared_type)


def _bridge_response(
    status: str,
    evidence_items: list[dict[str, Any]],
    hold_reason: str | None,
    *,
    evidence_required_pass: bool,
    raw_leak_pass: bool,
    rights_pass: bool,
    sensitivity_pass: bool = True,
) -> dict[str, Any]:
    safe_status = status if status in {RESULT_OK, RESULT_HOLD, RESULT_DENIED} else RESULT_HOLD
    safe_reason = None if safe_status == RESULT_OK else (hold_reason or "Library DB evidence requires review")
    return {
        "result_status": safe_status,
        "evidence_items": evidence_items,
        "hold_reason": safe_reason,
        "feedback_candidate_required": safe_status != RESULT_OK,
        "raw_text_included": False,
        "internal_path_included": False,
        "policy_result": {
            "evidence_required_pass": bool(evidence_required_pass),
            "raw_leak_pass": bool(raw_leak_pass),
            "rights_pass": bool(rights_pass),
            "sensitivity_pass": bool(sensitivity_pass),
        },
        "created_at": _created_at(),
    }


def _schema_required_fields_present(evidence: Mapping[str, Any]) -> bool:
    return all(bool(evidence.get(field)) for field in _BRIDGE_SCHEMA_REQUIRED_FIELDS)


def probe_sqlite_metadata(db_path: str | Path) -> dict[str, Any]:
    """Return SQLite schema metadata without selecting row values."""

    path = Path(db_path)
    tables: list[dict[str, Any]] = []
    with _connect_readonly(path) as connection:
        objects = connection.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        for obj in objects:
            table_name = str(obj["name"])
            quoted = _quote_identifier(table_name)
            columns = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
            indexes = connection.execute(f"PRAGMA index_list({quoted})").fetchall()
            row_count: int | None = None
            row_count_error: str | None = None
            try:
                row_count = int(connection.execute(f"SELECT COUNT(*) AS c FROM {quoted}").fetchone()["c"])
            except sqlite3.DatabaseError as exc:
                row_count_error = type(exc).__name__
            tables.append(
                {
                    "name": table_name,
                    "type": str(obj["type"]),
                    "row_count": row_count,
                    "row_count_error": row_count_error,
                    "columns": [
                        {
                            "name": str(column["name"]),
                            "type": str(column["type"]),
                            "notnull": bool(column["notnull"]),
                            "pk": bool(column["pk"]),
                            "classification": _column_classification(column["name"], column["type"]),
                        }
                        for column in columns
                    ],
                    "indexes": [
                        {
                            "name": str(index["name"]),
                            "unique": bool(index["unique"]),
                            "origin": str(index["origin"]),
                        }
                        for index in indexes
                    ],
                }
            )
    return {
        "selected_db_name": path.name,
        "readonly_uri_mode": "mode=ro&immutable=1",
        "query_only_requested": True,
        "table_count": len(tables),
        "tables": tables,
    }


def _column_lookup(table: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    for column in table.get("columns", []):
        if not isinstance(column, Mapping):
            continue
        name = str(column.get("name") or "")
        lowered = name.lower()
        lookup[lowered] = (name, str(column.get("type") or ""))
    return lookup


def _find_column(table: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
    lookup = _column_lookup(table)
    for alias in aliases:
        column = lookup.get(alias.lower())
        if column is None:
            continue
        if _safe_column_name(column[0], column[1]):
            return column[0]
    return None


def _table_mapping(table: Mapping[str, Any]) -> tuple[dict[str, str], str | None]:
    mapping: dict[str, str] = {}
    evidence_id = _find_column(table, _COLUMN_ALIASES["evidence_id"])
    safe_summary = _find_column(table, _COLUMN_ALIASES["safe_summary"])
    if evidence_id is None:
        return mapping, "safe metadata table has no evidence_id/id/doc_id column"
    if safe_summary is None:
        return mapping, "safe metadata table has no safe_summary column"

    mapping["evidence_id"] = evidence_id
    mapping["safe_summary"] = safe_summary
    for bridge_field in (
        "bridge_trace_id",
        "pointer_uri",
        "raw_text_policy",
        "rights_status",
        "source_doc_kind",
    ):
        column = _find_column(table, _COLUMN_ALIASES[bridge_field])
        if column is not None:
            mapping[bridge_field] = column
    for bridge_field in _REQUIRED_SAFE_INDEX_POLICY_FIELDS:
        column = _find_column(table, _COLUMN_ALIASES[bridge_field])
        if column is None:
            return mapping, f"safe metadata table has no {bridge_field} column"
        mapping[bridge_field] = column
    return mapping, None


def _first_bridge_table(
    metadata: Mapping[str, Any],
    table_name: str | None,
) -> tuple[Mapping[str, Any] | None, dict[str, str], str]:
    rejected: list[str] = []
    tables = [table for table in metadata.get("tables", []) if isinstance(table, Mapping)]
    if table_name is not None:
        tables = [table for table in tables if str(table.get("name")) == table_name]
        if not tables:
            return None, {}, f"requested table {table_name!r} was not found"

    for table in tables:
        mapping, reason = _table_mapping(table)
        if reason is None:
            return table, mapping, ""
        rejected.append(f"{table.get('name')}: {reason}")
    return None, {}, "; ".join(rejected) or "no Bridge-compatible safe metadata table was found"


def _select_rows(
    connection: sqlite3.Connection,
    table_name: str,
    mapping: Mapping[str, str],
    limit: int,
) -> list[sqlite3.Row]:
    selected_columns = sorted(set(mapping.values()))
    quoted_table = _quote_identifier(table_name)
    quoted_columns = ", ".join(_quote_identifier(column) for column in selected_columns)
    return connection.execute(
        f"SELECT {quoted_columns} FROM {quoted_table} LIMIT ?",
        (limit,),
    ).fetchall()


def _string_or_none(value: object, max_length: int = 512) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > max_length:
        return None
    if any(ord(char) < 32 for char in text):
        return None
    lowered = text.lower()
    if "h:\\" in lowered or "c:\\" in lowered or "file://" in lowered:
        return None
    return text


def _safe_token(value: object, max_length: int = 120) -> str | None:
    text = _string_or_none(value, max_length)
    if text is None:
        return None
    return text.strip().upper().replace("-", "_").replace(" ", "_")


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in {0, 1}:
            return bool(value)
        return None
    token = _safe_token(value, 40)
    if token in {"TRUE", "YES", "Y", "1"}:
        return True
    if token in {"FALSE", "NO", "N", "0"}:
        return False
    return None


def _safe_summary_source_allows_answer(summary_source: object, semantic_verified: bool) -> bool:
    source = _safe_token(summary_source, 120)
    if source is None:
        return False
    if source in _APPROVED_SAFE_SUMMARY_SOURCES:
        return True
    if semantic_verified and source != _METADATA_DERIVED_NOT_SEMANTIC:
        return True
    return False


def _safe_index_contract_decision(
    row: Mapping[str, Any],
    mapping: Mapping[str, str],
) -> tuple[str, str | None]:
    summary_source = _row_value(row, mapping.get("summary_source"))
    semantic_verified = _bool_or_none(_row_value(row, mapping.get("semantic_summary_verified")))
    raw_text_exposed = _bool_or_none(_row_value(row, mapping.get("raw_text_exposed")))
    production_path_exposed = _bool_or_none(_row_value(row, mapping.get("production_path_exposed")))

    if _safe_token(summary_source, 120) is None:
        return RESULT_HOLD, "safe metadata row is missing summary_source"
    if semantic_verified is None:
        return RESULT_HOLD, "safe metadata row is missing semantic_summary_verified"
    if raw_text_exposed is None or production_path_exposed is None:
        return RESULT_HOLD, "safe metadata exposure audit flags must be explicit false"
    if raw_text_exposed or production_path_exposed:
        return RESULT_DENIED, "safe metadata exposure audit flags are not false"
    if not _safe_summary_source_allows_answer(summary_source, semantic_verified):
        return RESULT_HOLD, "safe_summary source is not approved for user-visible answer"
    return RESULT_OK, None


def _row_value(row: Mapping[str, Any], source_column: str | None) -> Any:
    if source_column is None:
        return None
    if hasattr(row, "keys") and source_column in row.keys():
        return row[source_column]
    if isinstance(row, Mapping):
        return row.get(source_column)
    return None


def _project_row_to_evidence(
    row: Mapping[str, Any],
    mapping: Mapping[str, str],
    table_name: str,
) -> dict[str, Any]:
    evidence_id = _string_or_none(_row_value(row, mapping.get("evidence_id")), 120)
    if evidence_id is None:
        evidence_id = "library-db-evidence-unknown"
    safe_table = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in table_name)[:80]
    bridge_trace_id = _string_or_none(_row_value(row, mapping.get("bridge_trace_id")), 160)
    if bridge_trace_id is None:
        bridge_trace_id = f"btrace:library-db:{safe_table}:{evidence_id}"[:160]
    pointer_uri = _string_or_none(_row_value(row, mapping.get("pointer_uri")), 512)
    if pointer_uri is None:
        pointer_uri = f"qlib://library-db/{safe_table}/{evidence_id}"[:512]

    evidence = {
        "evidence_id": evidence_id,
        "bridge_trace_id": bridge_trace_id,
        "safe_summary": _string_or_none(_row_value(row, mapping.get("safe_summary")), 2000),
        "pointer_uri": pointer_uri,
        "raw_text_policy": _string_or_none(
            _row_value(row, mapping.get("raw_text_policy")),
            80,
        )
        or RAW_TEXT_POLICY_POINTER_ONLY,
        "rights_status": _string_or_none(
            _row_value(row, mapping.get("rights_status")),
            80,
        )
        or RIGHTS_NOT_VERIFIED,
    }
    source_doc_kind = _string_or_none(_row_value(row, mapping.get("source_doc_kind")), 120)
    if source_doc_kind is not None:
        evidence["source_doc_kind"] = source_doc_kind
    return evidence


def retrieve_bridge_evidence_from_sqlite(
    db_path: str | Path,
    *,
    table_name: str | None = None,
    limit: int | None = None,
    requester_module: str = "Bridge",
    purpose: str = "answer",
) -> dict[str, Any]:
    """Return Bridge evidence from a safe metadata table or HOLD closed."""

    try:
        metadata = probe_sqlite_metadata(db_path)
    except (OSError, sqlite3.DatabaseError, ValueError):
        return _bridge_response(
            RESULT_HOLD,
            [],
            "SQLite Library Evidence source could not be opened read-only",
            evidence_required_pass=False,
            raw_leak_pass=True,
            rights_pass=False,
        )

    selected_table, mapping, rejection_reason = _first_bridge_table(metadata, table_name)
    if selected_table is None:
        return _bridge_response(
            RESULT_HOLD,
            [],
            rejection_reason or "no Bridge-compatible safe metadata table was found",
            evidence_required_pass=False,
            raw_leak_pass=True,
            rights_pass=False,
        )

    ok_items: list[dict[str, Any]] = []
    hold_reasons: list[str] = []
    denied_reasons: list[str] = []
    bounded_limit = _bounded_limit(limit)

    try:
        with _connect_readonly(db_path) as connection:
            rows = _select_rows(connection, str(selected_table["name"]), mapping, bounded_limit)
    except (OSError, sqlite3.DatabaseError, ValueError):
        return _bridge_response(
            RESULT_HOLD,
            [],
            "SQLite Library Evidence source could not be read with safe metadata columns",
            evidence_required_pass=False,
            raw_leak_pass=True,
            rights_pass=False,
        )

    for row in rows:
        contract_status, contract_reason = _safe_index_contract_decision(row, mapping)
        if contract_status == RESULT_DENIED:
            denied_reasons.append(contract_reason or "safe metadata sidecar contract denied")
            continue
        if contract_status != RESULT_OK:
            hold_reasons.append(contract_reason or "safe metadata sidecar contract requires review")
            continue

        evidence = _project_row_to_evidence(row, mapping, str(selected_table["name"]))
        decision = decide_bridge_result(
            evidence,
            requester_module=requester_module,
            purpose=purpose,
        )
        status = decision.get("result_status")
        reason = str(decision.get("hold_reason") or "Library DB evidence requires review")
        if status == RESULT_DENIED:
            denied_reasons.append(reason)
            continue
        if status == RESULT_OK:
            projected = project_bridge_safe_evidence(evidence)
            if _schema_required_fields_present(projected):
                ok_items.append(projected)
            else:
                hold_reasons.append("projected DB evidence is missing Bridge schema required fields")
            continue
        hold_reasons.append(reason)

    if denied_reasons:
        return _bridge_response(
            RESULT_DENIED,
            [],
            denied_reasons[0],
            evidence_required_pass=False,
            raw_leak_pass=True,
            rights_pass=False,
        )
    if ok_items:
        return _bridge_response(
            RESULT_OK,
            ok_items,
            None,
            evidence_required_pass=True,
            raw_leak_pass=True,
            rights_pass=True,
        )
    return _bridge_response(
        RESULT_HOLD,
        [],
        hold_reasons[0] if hold_reasons else "no Bridge-safe DB evidence item was accepted",
        evidence_required_pass=False,
        raw_leak_pass=True,
        rights_pass=False,
    )


__all__ = [
    "probe_sqlite_metadata",
    "retrieve_bridge_evidence_from_sqlite",
]
