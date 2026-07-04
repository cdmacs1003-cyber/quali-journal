from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from admin.f13_skillup_feedback_queue_persistence import (
    DB_ACCESS_EXECUTED,
    DB_BACKED_QUEUE_DEFERRED,
    DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS,
    INTERNAL_PATH_INCLUDED,
    RAW_TEXT_INCLUDED,
    DurableFeedbackQueueItem,
    FeedbackQueuePersistenceResult,
    UnsafeFeedbackQueuePayloadError,
    validate_minimized_feedback_queue_item,
)


LOCAL_FILE_PERSISTENCE_MECHANISM = "LOCAL_FILE_FEEDBACK_QUEUE_ADAPTER"
LOCAL_FILE_ADAPTER_BOUNDARY = (
    "Local-file feedback queue persistence is bounded to minimized durable records "
    "in caller-owned test/local paths; it is not DB-backed or production persistence evidence."
)

_DEFAULT_FILENAME = "feedback_queue_items.jsonl"
_LONG_MEMORY_ROOT_NAME = "\uc7a5\uae30\uae30\uc5b5"
_PRODUCTION_ROOT_PREFIXES = (f"h:\\{_LONG_MEMORY_ROOT_NAME}",)
_FORBIDDEN_PATH_MARKERS = ("\\brain.db", "\\graph.db", "\\library\\")
_SECRET_LIKE_FILENAME_MARKERS = (
    ".env",
    ".pem",
    ".key",
    "credential",
    "secret",
    "token",
    "key",
    "service-account",
    "service_account",
)
_SAFE_LOOKUP_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:._-")


class LocalFileFeedbackQueueRepositoryError(RuntimeError):
    """Raised when the bounded local-file adapter cannot safely read or write."""


def _safe_storage_filename(filename: str) -> str:
    name = str(filename or "").strip()
    if not name or name in {".", ".."}:
        raise LocalFileFeedbackQueueRepositoryError("local feedback queue filename is required")
    path = Path(name)
    if path.name != name or path.is_absolute():
        raise LocalFileFeedbackQueueRepositoryError("local feedback queue filename must be a plain filename")
    lowered = name.lower()
    if any(marker in lowered for marker in _SECRET_LIKE_FILENAME_MARKERS):
        raise LocalFileFeedbackQueueRepositoryError("local feedback queue filename is secret-like")
    return name


def _normalized_path_text(path: Path) -> str:
    return str(path.resolve(strict=False)).replace("/", "\\").lower()


def _reject_forbidden_storage_path(path: Path) -> None:
    normalized = _normalized_path_text(path)
    if any(normalized == prefix or normalized.startswith(f"{prefix}\\") for prefix in _PRODUCTION_ROOT_PREFIXES):
        raise LocalFileFeedbackQueueRepositoryError("production long-memory roots are forbidden")
    if any(marker in normalized for marker in _FORBIDDEN_PATH_MARKERS):
        raise LocalFileFeedbackQueueRepositoryError("production DB/library paths are forbidden")


def _safe_lookup_token(value: str, field_name: str, *, max_length: int = 220) -> str:
    token = str(value or "").strip()
    if not token:
        raise UnsafeFeedbackQueuePayloadError(f"{field_name} is required")
    if len(token) > max_length:
        raise UnsafeFeedbackQueuePayloadError(f"{field_name} is too long")
    if any(char not in _SAFE_LOOKUP_CHARS for char in token):
        raise UnsafeFeedbackQueuePayloadError(f"{field_name} contains unsafe characters")
    return token


def _item_from_payload(payload: Mapping[str, Any]) -> DurableFeedbackQueueItem:
    validate_minimized_feedback_queue_item(payload)
    extra_fields = set(payload) - DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS
    if extra_fields:
        raise UnsafeFeedbackQueuePayloadError(
            f"local feedback queue item contains unsupported fields: {sorted(extra_fields)}"
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
        evidence_count=payload["evidence_count"],
        warning_codes=tuple(payload["warning_codes"]),
        trace_id=payload.get("trace_id"),
        request_id=payload.get("request_id"),
        raw_text_included=payload["raw_text_included"],
        internal_path_included=payload["internal_path_included"],
        db_access_executed=payload["db_access_executed"],
        persistence_mechanism=payload["persistence_mechanism"],
        contract_version=payload["contract_version"],
    )


def _normalize_item(item: DurableFeedbackQueueItem | Mapping[str, Any]) -> DurableFeedbackQueueItem:
    payload = item.to_persistence_dict() if isinstance(item, DurableFeedbackQueueItem) else dict(item)
    return _item_from_payload(payload)


class LocalFileFeedbackQueueRepository:
    def __init__(self, root: str | Path, *, filename: str = _DEFAULT_FILENAME) -> None:
        self._root = Path(root)
        self._filename = _safe_storage_filename(filename)
        self._path = self._root / self._filename
        _reject_forbidden_storage_path(self._root)
        _reject_forbidden_storage_path(self._path)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def enqueue(self, item: DurableFeedbackQueueItem | Mapping[str, Any]) -> FeedbackQueuePersistenceResult:
        durable_item = _normalize_item(item)
        records = self._load_records()
        by_feedback_id = {record.feedback_id: record for record in records}
        by_dedup_key = {record.dedup_key: record for record in records}

        existing = by_feedback_id.get(durable_item.feedback_id)
        reason_code = "LOCAL_FILE_DUPLICATE_FEEDBACK_ID"
        if existing is None:
            existing = by_dedup_key.get(durable_item.dedup_key)
            reason_code = "LOCAL_FILE_DUPLICATE_DEDUP_KEY"
        if existing is not None:
            return self._result(existing, current_status="duplicate", reason_code=reason_code)

        records.append(durable_item)
        self._write_records(records)
        return self._result(durable_item, current_status=durable_item.current_status, reason_code="LOCAL_FILE_ACCEPTED")

    def read(self, feedback_id: str) -> DurableFeedbackQueueItem | None:
        safe_feedback_id = _safe_lookup_token(feedback_id, "feedback_id", max_length=160)
        for record in self._load_records():
            if record.feedback_id == safe_feedback_id:
                return record
        return None

    def read_by_dedup_key(self, dedup_key: str) -> DurableFeedbackQueueItem | None:
        safe_dedup_key = _safe_lookup_token(dedup_key, "dedup_key", max_length=220)
        for record in self._load_records():
            if record.dedup_key == safe_dedup_key:
                return record
        return None

    def cleanup(self) -> int:
        records = self._load_records()
        if self._path.exists():
            self._path.unlink()
        return len(records)

    def _result(
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
            raw_text_included=RAW_TEXT_INCLUDED,
            internal_path_included=INTERNAL_PATH_INCLUDED,
            db_access_executed=DB_ACCESS_EXECUTED,
            persistence_executed=True,
            persistence_mechanism=LOCAL_FILE_PERSISTENCE_MECHANISM,
        )

    def _load_records(self) -> list[DurableFeedbackQueueItem]:
        if not self._path.exists():
            return []
        records: list[DurableFeedbackQueueItem] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise LocalFileFeedbackQueueRepositoryError(
                        f"local feedback queue record {line_number} is not valid JSON"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise LocalFileFeedbackQueueRepositoryError(
                        f"local feedback queue record {line_number} is not an object"
                    )
                records.append(_item_from_payload(payload))
        return records

    def _write_records(self, records: list[DurableFeedbackQueueItem]) -> None:
        with self._path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                payload = record.to_persistence_dict()
                validate_minimized_feedback_queue_item(payload)
                if payload["persistence_mechanism"] != DB_BACKED_QUEUE_DEFERRED:
                    raise UnsafeFeedbackQueuePayloadError("stored feedback queue item has unexpected mechanism")
                handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
                handle.write("\n")


__all__ = [
    "LOCAL_FILE_ADAPTER_BOUNDARY",
    "LOCAL_FILE_PERSISTENCE_MECHANISM",
    "LocalFileFeedbackQueueRepository",
    "LocalFileFeedbackQueueRepositoryError",
]
