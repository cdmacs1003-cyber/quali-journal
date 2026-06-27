from __future__ import annotations

import sqlite3
import json
from collections.abc import Mapping, Sequence
from typing import Any

from admin.f13_skillup_feedback_queue_persistence import (
    ANSWER_STATUS_VALUES,
    CONTRACT_VERSION,
    CURRENT_STATUS_VALUES,
    DB_BACKED_QUEUE_DEFERRED,
    DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS,
    RESULT_STATUS_VALUES,
    SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS,
    DurableFeedbackQueueItem,
    FeedbackQueuePersistenceResult,
    UnsafeFeedbackQueuePayloadError,
    validate_minimized_feedback_queue_item,
)


SQLITE_FIXTURE_MIGRATION_ID = "R9ZMO_SKILLUP_FEEDBACK_QUEUE_SQLITE_FIXTURE_20260614"
SQLITE_FIXTURE_TABLE_NAME = "skillup_feedback_queue_items"
SQLITE_FIXTURE_PERSISTENCE_MECHANISM = "LOCAL_DISPOSABLE_SQLITE_FIXTURE"
DB_FIXTURE_EXECUTION_NOT_GRANTED = (
    "SQLite fixture execution, migration execution, and durable persistence PASS "
    "remain NOT_GRANTED until a later approved validation gate executes them."
)

_TABLE_NAME_PREFIX = f"{SQLITE_FIXTURE_TABLE_NAME}_"
_TABLE_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_LOOKUP_TOKEN_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:._-")
_SQLITE_ROW_COLUMNS = (
    "contract_version",
    "persistence_mechanism",
    "feedback_id",
    "origin_event_id",
    "current_status",
    "dedup_key",
    "created_at",
    "review_reason_code",
    "safe_summary",
    "result_status",
    "answer_status",
    "evidence_required",
    "review_required",
    "evidence_count",
    "warning_codes",
    "trace_id",
    "request_id",
    "raw_text_included",
    "internal_path_included",
    "db_access_executed",
)


class SQLiteFeedbackQueueFixtureError(RuntimeError):
    """Raised for local disposable SQLite fixture boundary failures."""


def validate_sqlite_fixture_table_name(table_name: str = SQLITE_FIXTURE_TABLE_NAME) -> str:
    safe_name = str(table_name or "").strip()
    if not safe_name:
        raise SQLiteFeedbackQueueFixtureError("SQLite fixture table name is required")
    if safe_name[0].isdigit() or any(char not in _TABLE_NAME_CHARS for char in safe_name):
        raise SQLiteFeedbackQueueFixtureError("SQLite fixture table name is outside the safe contract")
    if safe_name != SQLITE_FIXTURE_TABLE_NAME and not safe_name.startswith(_TABLE_NAME_PREFIX):
        raise SQLiteFeedbackQueueFixtureError("SQLite fixture table name must use the approved fixture prefix")
    return safe_name


def build_sqlite_feedback_queue_schema_sql(table_name: str = SQLITE_FIXTURE_TABLE_NAME) -> str:
    safe_table_name = validate_sqlite_fixture_table_name(table_name)
    status_literals = ", ".join(f"'{status}'" for status in sorted(CURRENT_STATUS_VALUES))
    result_status_literals = ", ".join(f"'{status}'" for status in sorted(RESULT_STATUS_VALUES))
    answer_status_literals = ", ".join(f"'{status}'" for status in sorted(ANSWER_STATUS_VALUES))
    return f"""
-- {SQLITE_FIXTURE_MIGRATION_ID}
-- Test-scoped local disposable SQLite fixture only.
-- No production/shared DB target, external DSN, network DB, or secret-backed config.
CREATE TABLE IF NOT EXISTS {safe_table_name} (
    contract_version TEXT NOT NULL,
    persistence_mechanism TEXT NOT NULL CHECK (persistence_mechanism = '{DB_BACKED_QUEUE_DEFERRED}'),
    feedback_id TEXT NOT NULL PRIMARY KEY,
    origin_event_id TEXT NOT NULL,
    current_status TEXT NOT NULL CHECK (current_status IN ({status_literals})),
    dedup_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    review_reason_code TEXT NOT NULL,
    safe_summary TEXT NOT NULL,
    result_status TEXT NOT NULL CHECK (result_status IN ({result_status_literals})),
    answer_status TEXT NOT NULL CHECK (answer_status IN ({answer_status_literals})),
    evidence_required INTEGER NOT NULL CHECK (evidence_required IN (0, 1)),
    review_required INTEGER NOT NULL CHECK (review_required IN (0, 1)),
    evidence_count INTEGER NOT NULL DEFAULT 0 CHECK (evidence_count >= 0),
    warning_codes TEXT NOT NULL DEFAULT '[]',
    trace_id TEXT,
    request_id TEXT,
    raw_text_included INTEGER NOT NULL DEFAULT 0 CHECK (raw_text_included = 0),
    internal_path_included INTEGER NOT NULL DEFAULT 0 CHECK (internal_path_included = 0),
    db_access_executed INTEGER NOT NULL DEFAULT 0 CHECK (db_access_executed = 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_{safe_table_name}_dedup_key
    ON {safe_table_name} (dedup_key);
""".strip()


SQLITE_FIXTURE_SCHEMA_SQL = build_sqlite_feedback_queue_schema_sql()


def _safe_lookup_token(value: Any, field_name: str, *, max_length: int = 220) -> str:
    text = str(value or "").strip()
    if not text:
        raise UnsafeFeedbackQueuePayloadError(f"{field_name} is required")
    if len(text) > max_length or any(ord(char) < 32 for char in text):
        raise UnsafeFeedbackQueuePayloadError(f"{field_name} is outside the safe lookup contract")
    if any(char not in _LOOKUP_TOKEN_CHARS for char in text):
        raise UnsafeFeedbackQueuePayloadError(f"{field_name} contains unsafe lookup characters")
    return text


def _sqlite_false(value: Any, field_name: str) -> bool:
    if value in (0, False):
        return False
    raise UnsafeFeedbackQueuePayloadError(f"{field_name} must remain false in SQLite fixture rows")


def _sqlite_bool(value: Any, field_name: str) -> bool:
    if value in (1, True):
        return True
    if value in (0, False):
        return False
    raise UnsafeFeedbackQueuePayloadError(f"{field_name} must be a SQLite boolean")


def _sqlite_warning_codes(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        loaded = json.loads(value)
    else:
        loaded = value
    if not isinstance(loaded, list):
        raise UnsafeFeedbackQueuePayloadError("warning_codes must decode to a safe string list")
    return tuple(str(code) for code in loaded)


def normalize_durable_feedback_queue_item(
    item: DurableFeedbackQueueItem | Mapping[str, Any],
) -> DurableFeedbackQueueItem:
    if isinstance(item, DurableFeedbackQueueItem):
        validate_minimized_feedback_queue_item(item)
        return item

    payload = dict(item)
    validate_minimized_feedback_queue_item(payload)
    extra_fields = set(payload) - DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS
    if extra_fields:
        raise UnsafeFeedbackQueuePayloadError(
            f"durable feedback queue item has unsupported DB row fields: {sorted(extra_fields)}"
        )
    return DurableFeedbackQueueItem(
        feedback_id=str(payload["feedback_id"]),
        origin_event_id=str(payload["origin_event_id"]),
        current_status=str(payload["current_status"]),
        dedup_key=str(payload["dedup_key"]),
        created_at=str(payload["created_at"]),
        review_reason_code=str(payload["review_reason_code"]),
        safe_summary=str(payload["safe_summary"]),
        result_status=str(payload["result_status"]),
        answer_status=str(payload["answer_status"]),
        evidence_required=payload["evidence_required"],
        review_required=payload["review_required"],
        evidence_count=int(payload["evidence_count"]),
        warning_codes=tuple(payload["warning_codes"]),
        trace_id=payload.get("trace_id") if payload.get("trace_id") is not None else None,
        request_id=payload.get("request_id") if payload.get("request_id") is not None else None,
        raw_text_included=payload["raw_text_included"],
        internal_path_included=payload["internal_path_included"],
        db_access_executed=payload["db_access_executed"],
        persistence_mechanism=str(payload["persistence_mechanism"]),
        contract_version=str(payload["contract_version"]),
    )


def durable_item_to_sqlite_row(
    item: DurableFeedbackQueueItem | Mapping[str, Any],
) -> dict[str, Any]:
    durable_item = normalize_durable_feedback_queue_item(item)
    row = durable_item.to_persistence_dict()
    return {
        "contract_version": row["contract_version"],
        "persistence_mechanism": row["persistence_mechanism"],
        "feedback_id": row["feedback_id"],
        "origin_event_id": row["origin_event_id"],
        "current_status": row["current_status"],
        "dedup_key": row["dedup_key"],
        "created_at": row["created_at"],
        "review_reason_code": row["review_reason_code"],
        "safe_summary": row["safe_summary"],
        "result_status": row["result_status"],
        "answer_status": row["answer_status"],
        "evidence_required": 1 if row["evidence_required"] else 0,
        "review_required": 1 if row["review_required"] else 0,
        "evidence_count": row["evidence_count"],
        "warning_codes": json.dumps(list(row["warning_codes"]), separators=(",", ":")),
        "trace_id": row.get("trace_id"),
        "request_id": row.get("request_id"),
        "raw_text_included": 0,
        "internal_path_included": 0,
        "db_access_executed": 0,
    }


def _row_to_mapping(row: sqlite3.Row | Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        return {column: row[column] for column in _SQLITE_ROW_COLUMNS}
    if isinstance(row, Mapping):
        return {column: row[column] for column in _SQLITE_ROW_COLUMNS}
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        if len(row) != len(_SQLITE_ROW_COLUMNS):
            raise SQLiteFeedbackQueueFixtureError("SQLite fixture row shape is outside the contract")
        return dict(zip(_SQLITE_ROW_COLUMNS, row))
    raise SQLiteFeedbackQueueFixtureError("SQLite fixture row is not readable")


def sqlite_row_to_durable_item(row: sqlite3.Row | Mapping[str, Any] | Sequence[Any]) -> DurableFeedbackQueueItem:
    payload = _row_to_mapping(row)
    payload["evidence_required"] = _sqlite_bool(payload.get("evidence_required"), "evidence_required")
    payload["review_required"] = _sqlite_bool(payload.get("review_required"), "review_required")
    payload["warning_codes"] = _sqlite_warning_codes(payload.get("warning_codes"))
    payload["raw_text_included"] = _sqlite_false(payload.get("raw_text_included"), "raw_text_included")
    payload["internal_path_included"] = _sqlite_false(
        payload.get("internal_path_included"),
        "internal_path_included",
    )
    payload["db_access_executed"] = _sqlite_false(payload.get("db_access_executed"), "db_access_executed")
    return normalize_durable_feedback_queue_item(payload)


def assert_selected_route_persistence_internals_absent(response: Mapping[str, Any]) -> bool:
    exposed = SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS & set(response)
    if exposed:
        raise UnsafeFeedbackQueuePayloadError(f"selected-route response exposed queue internals: {sorted(exposed)}")
    return True


class SQLiteFeedbackQueueRepository:
    """Local disposable SQLite fixture repository.

    The constructor requires an injected sqlite3.Connection. This module never
    opens a DSN, reads config, or targets a production/shared database.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        table_name: str = SQLITE_FIXTURE_TABLE_NAME,
    ) -> None:
        if not isinstance(connection, sqlite3.Connection):
            raise SQLiteFeedbackQueueFixtureError("SQLite fixture repository requires an injected connection")
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._table_name = validate_sqlite_fixture_table_name(table_name)

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def execution_boundary(self) -> str:
        return DB_FIXTURE_EXECUTION_NOT_GRANTED

    def ensure_schema(self) -> None:
        self._connection.executescript(build_sqlite_feedback_queue_schema_sql(self._table_name))

    def enqueue(
        self,
        item: DurableFeedbackQueueItem | Mapping[str, Any],
    ) -> FeedbackQueuePersistenceResult:
        durable_item = normalize_durable_feedback_queue_item(item)
        row = durable_item_to_sqlite_row(durable_item)
        placeholders = ", ".join(f":{column}" for column in _SQLITE_ROW_COLUMNS)
        columns = ", ".join(_SQLITE_ROW_COLUMNS)
        try:
            with self._connection:
                cursor = self._connection.execute(
                    f"INSERT OR IGNORE INTO {self._table_name} ({columns}) VALUES ({placeholders})",
                    row,
                )
        except sqlite3.DatabaseError as exc:
            raise SQLiteFeedbackQueueFixtureError("SQLite fixture enqueue failed") from exc

        if cursor.rowcount == 0:
            existing = self.read_by_dedup_key(durable_item.dedup_key)
            if existing is None:
                raise SQLiteFeedbackQueueFixtureError("SQLite fixture duplicate row was not readable")
            return self._result_for_item(
                existing,
                current_status="duplicate",
                reason_code="SQLITE_FIXTURE_DUPLICATE_DEDUP_KEY",
            )

        return self._result_for_item(
            durable_item,
            current_status=durable_item.current_status,
            reason_code="SQLITE_FIXTURE_ACCEPTED",
        )

    def read(self, feedback_id: str) -> DurableFeedbackQueueItem | None:
        safe_feedback_id = _safe_lookup_token(feedback_id, "feedback_id", max_length=160)
        return self._select_one("feedback_id = ?", safe_feedback_id)

    def read_by_dedup_key(self, dedup_key: str) -> DurableFeedbackQueueItem | None:
        safe_dedup_key = _safe_lookup_token(dedup_key, "dedup_key", max_length=220)
        return self._select_one("dedup_key = ?", safe_dedup_key)

    def cleanup(self) -> int:
        with self._connection:
            cursor = self._connection.execute(f"DELETE FROM {self._table_name}")
        return int(cursor.rowcount if cursor.rowcount is not None else 0)

    def drop_schema(self) -> None:
        with self._connection:
            self._connection.execute(f"DROP INDEX IF EXISTS idx_{self._table_name}_dedup_key")
            self._connection.execute(f"DROP TABLE IF EXISTS {self._table_name}")

    def dispose(self) -> None:
        self._connection.close()

    def _select_one(self, clause: str, value: str) -> DurableFeedbackQueueItem | None:
        columns = ", ".join(_SQLITE_ROW_COLUMNS)
        try:
            cursor = self._connection.execute(
                f"SELECT {columns} FROM {self._table_name} WHERE {clause} LIMIT 1",
                (value,),
            )
            row = cursor.fetchone()
        except sqlite3.DatabaseError as exc:
            raise SQLiteFeedbackQueueFixtureError("SQLite fixture read failed") from exc
        if row is None:
            return None
        return sqlite_row_to_durable_item(row)

    def _result_for_item(
        self,
        item: DurableFeedbackQueueItem,
        *,
        current_status: str,
        reason_code: str,
    ) -> FeedbackQueuePersistenceResult:
        return FeedbackQueuePersistenceResult(
            accepted=True,
            feedback_id=item.feedback_id,
            dedup_key=item.dedup_key,
            current_status=current_status,
            reason_code=reason_code,
            raw_text_included=False,
            internal_path_included=False,
            db_access_executed=True,
            persistence_executed=True,
            persistence_mechanism=SQLITE_FIXTURE_PERSISTENCE_MECHANISM,
        )


__all__ = [
    "CONTRACT_VERSION",
    "DB_FIXTURE_EXECUTION_NOT_GRANTED",
    "SQLITE_FIXTURE_MIGRATION_ID",
    "SQLITE_FIXTURE_PERSISTENCE_MECHANISM",
    "SQLITE_FIXTURE_SCHEMA_SQL",
    "SQLITE_FIXTURE_TABLE_NAME",
    "SQLiteFeedbackQueueFixtureError",
    "SQLiteFeedbackQueueRepository",
    "assert_selected_route_persistence_internals_absent",
    "build_sqlite_feedback_queue_schema_sql",
    "durable_item_to_sqlite_row",
    "normalize_durable_feedback_queue_item",
    "sqlite_row_to_durable_item",
    "validate_sqlite_fixture_table_name",
]
