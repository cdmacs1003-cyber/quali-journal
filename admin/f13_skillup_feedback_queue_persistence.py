from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol


CONTRACT_VERSION = "R9ZMH-2026-06-14"
DB_BACKED_QUEUE_DEFERRED = "DB_BACKED_QUEUE_DEFERRED"
DB_ACCESS_EXECUTED_BOUNDARY = (
    "db_access_executed=false means this contract construction path did not "
    "execute DB access; it is not durable persistence success evidence."
)

RAW_TEXT_INCLUDED = False
INTERNAL_PATH_INCLUDED = False
DB_ACCESS_EXECUTED = False

CURRENT_STATUS_VALUES = frozenset(
    {
        "queued",
        "review_required",
        "resolved",
        "rejected",
        "duplicate",
    }
)
INITIAL_CURRENT_STATUS_VALUES = frozenset({"queued", "review_required"})

DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS = frozenset(
    {
        "contract_version",
        "persistence_mechanism",
        "feedback_id",
        "origin_event_id",
        "current_status",
        "dedup_key",
        "created_at",
        "review_reason_code",
        "safe_summary",
        "trace_id",
        "request_id",
        "raw_text_included",
        "internal_path_included",
        "db_access_executed",
    }
)

SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS = frozenset(
    {
        "feedback_queue_item",
        "feedback_candidate",
        "feedback_candidate_required",
        "created_at",
        "db_access_executed",
        "feedback_id",
        "origin_event_id",
        "current_status",
        "dedup_key",
        "review_reason_code",
        "safe_summary",
        "persistence_mechanism",
        "persistence_result",
        "queue_write_result",
        "queue_read_result",
        "durable_feedback_queue_item",
    }
)

_SAFE_FALSE_FLAG_FIELDS = frozenset(
    {
        "raw_text_included",
        "internal_path_included",
        "db_access_executed",
    }
)
_UNSAFE_FIELD_MARKERS = (
    "raw_text",
    "raw_prompt",
    "raw_query",
    "raw_answer",
    "raw_source",
    "source_text",
    "source_uri_or_path",
    "full_answer",
    "full_source",
    "internal_path",
    "local_route",
    "file_uri",
    "hostname",
    "host_name",
    "secret",
    "token",
    "credential",
    "dsn",
    "api_key",
    "private_key",
    "secret_key",
    "access_key",
    "client_key",
    "service_account",
    "service-account",
    "bridge_payload",
    "bridge_response",
    "evidence_items",
    "source_payload",
    "standard_text",
)
_UNSAFE_VALUE_MARKERS = (
    "raw standard text",
    "raw text",
    "raw prompt",
    "raw query",
    "raw answer",
    "raw source",
    "full source",
    "full answer",
    "restricted prompt",
    "internal path",
    "source_uri_or_path",
    "file://",
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "h:\\",
    "c:\\",
    "/users/",
    "/home/",
    "/var/",
    "/etc/",
    "/mnt/",
    ".env",
    "secret",
    "token",
    "credential",
    "password=",
    "authorization:",
    "api_key",
    "apikey",
    "dsn=",
    "postgres://",
    "postgresql://",
    "mysql://",
    "mongodb://",
    "redis://",
    "sqlite://",
    "service-account",
    "service_account",
)
_HOSTNAME_RE = re.compile(
    r"\b[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.(?:local|internal|example|invalid|test|com|net|org|io|dev|app|cloud|db))\b",
    re.IGNORECASE,
)


class UnsafeFeedbackQueuePayloadError(ValueError):
    """Raised when a candidate durable queue item contains unsafe payload surface."""


class FeedbackQueuePersistenceNotEnabled(RuntimeError):
    """Raised when a caller attempts persistence through the disabled boundary."""


@dataclass(frozen=True)
class DurableFeedbackQueueItem:
    feedback_id: str
    origin_event_id: str
    current_status: str
    dedup_key: str
    created_at: str
    review_reason_code: str
    safe_summary: str
    trace_id: str | None = None
    request_id: str | None = None
    raw_text_included: bool = RAW_TEXT_INCLUDED
    internal_path_included: bool = INTERNAL_PATH_INCLUDED
    db_access_executed: bool = DB_ACCESS_EXECUTED
    persistence_mechanism: str = DB_BACKED_QUEUE_DEFERRED
    contract_version: str = CONTRACT_VERSION

    def to_persistence_dict(self) -> dict[str, Any]:
        return {key: asdict(self)[key] for key in sorted(DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS)}


@dataclass(frozen=True)
class FeedbackQueuePersistenceResult:
    accepted: bool
    feedback_id: str | None
    dedup_key: str | None
    current_status: str
    reason_code: str
    raw_text_included: bool = RAW_TEXT_INCLUDED
    internal_path_included: bool = INTERNAL_PATH_INCLUDED
    db_access_executed: bool = DB_ACCESS_EXECUTED
    persistence_executed: bool = False
    persistence_mechanism: str = DB_BACKED_QUEUE_DEFERRED


class FeedbackQueueRepository(Protocol):
    def enqueue(self, item: DurableFeedbackQueueItem) -> FeedbackQueuePersistenceResult:
        ...

    def read(self, feedback_id: str) -> DurableFeedbackQueueItem | None:
        ...


def _safe_text(value: Any, fallback: str, *, max_length: int = 500) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    if any(ord(char) < 32 for char in text):
        raise UnsafeFeedbackQueuePayloadError("feedback queue text contains control characters")
    return text[:max_length]


def _safe_token(value: Any, fallback: str, *, max_length: int = 160) -> str:
    text = _safe_text(value, fallback, max_length=max_length)
    token = "".join(char for char in text if char.isalnum() or char in ":._-")
    return token[:max_length] or fallback


def _stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(_safe_text(part, "", max_length=300) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _looks_like_hostname(text: str) -> bool:
    return bool(_HOSTNAME_RE.search(text))


def _unsafe_string_reason(text: str) -> str | None:
    lowered = text.lower()
    if any(marker in lowered for marker in _UNSAFE_VALUE_MARKERS):
        return "unsafe value marker"
    if "://" in lowered and not lowered.startswith("pointer://opaque/"):
        return "uri-like value"
    if _looks_like_hostname(lowered):
        return "hostname-like value"
    return None


def _assert_no_unsafe_surface(value: Any, *, field_name: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered_key = str(key).lower()
            if lowered_key in _SAFE_FALSE_FLAG_FIELDS:
                if child is not False:
                    raise UnsafeFeedbackQueuePayloadError(
                        f"{lowered_key} must be false for durable feedback queue persistence"
                    )
                continue
            if any(marker in lowered_key for marker in _UNSAFE_FIELD_MARKERS):
                raise UnsafeFeedbackQueuePayloadError(f"{field_name}.{key} is not allowed for persistence")
            _assert_no_unsafe_surface(child, field_name=f"{field_name}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_unsafe_surface(child, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, str):
        reason = _unsafe_string_reason(value)
        if reason is not None:
            raise UnsafeFeedbackQueuePayloadError(f"{field_name} contains {reason}")


def _review_reason_code(value: Any, fallback: str = "HOLD_REVIEW_REQUIRED") -> str:
    text = str(value or fallback).strip().upper().replace("-", "_").replace(" ", "_")
    token = "".join(char for char in text if char.isalnum() or char == "_")
    return token[:80] or fallback


def _current_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in CURRENT_STATUS_VALUES:
        return status
    return "review_required"


def _optional_token(value: Any, *, max_length: int = 160) -> str | None:
    if value in (None, ""):
        return None
    return _safe_token(value, "", max_length=max_length) or None


def durable_feedback_queue_item_from_hold(hold_payload: Mapping[str, Any] | None) -> DurableFeedbackQueueItem:
    source = hold_payload if isinstance(hold_payload, Mapping) else {}
    _assert_no_unsafe_surface(source)

    summary_source = (
        source.get("safe_summary")
        or source.get("suspected_issue")
        or source.get("hold_reason")
        or "Skillup answer/HOLD response requires feedback queue review."
    )
    safe_summary = _safe_text(summary_source, "Skillup answer/HOLD response requires feedback queue review.")
    _assert_no_unsafe_surface(safe_summary, field_name="safe_summary")

    review_reason = _review_reason_code(
        source.get("review_reason_code")
        or source.get("hold_reason_code")
        or source.get("feedback_type")
        or "HOLD_REVIEW_REQUIRED"
    )
    seed = _stable_digest(
        source.get("feedback_id"),
        source.get("origin_event_id"),
        source.get("dedup_key"),
        review_reason,
        safe_summary,
    )

    feedback_id = _safe_token(source.get("feedback_id"), f"fbq:{seed}")
    origin_event_id = _safe_token(
        source.get("origin_event_id")
        or source.get("bridge_trace_id")
        or source.get("trace_id"),
        f"hold:{seed}",
    )
    dedup_key = _safe_token(
        source.get("dedup_key"),
        f"Skillup:{review_reason}:{seed}",
        max_length=220,
    )

    item = DurableFeedbackQueueItem(
        feedback_id=feedback_id,
        origin_event_id=origin_event_id,
        current_status=_current_status(source.get("current_status")),
        dedup_key=dedup_key,
        created_at=_safe_token(source.get("created_at"), "1970-01-01T00:00:00Z"),
        review_reason_code=review_reason,
        safe_summary=safe_summary,
        trace_id=_optional_token(source.get("trace_id") or source.get("bridge_trace_id")),
        request_id=_optional_token(source.get("request_id")),
    )
    validate_minimized_feedback_queue_item(item)
    return item


def validate_minimized_feedback_queue_item(item: DurableFeedbackQueueItem | Mapping[str, Any]) -> bool:
    payload = item.to_persistence_dict() if isinstance(item, DurableFeedbackQueueItem) else dict(item)
    missing = DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS - set(payload)
    if missing:
        raise UnsafeFeedbackQueuePayloadError(f"durable feedback queue item missing fields: {sorted(missing)}")
    if payload.get("raw_text_included") is not False:
        raise UnsafeFeedbackQueuePayloadError("raw_text_included must be false")
    if payload.get("internal_path_included") is not False:
        raise UnsafeFeedbackQueuePayloadError("internal_path_included must be false")
    if payload.get("db_access_executed") is not False:
        raise UnsafeFeedbackQueuePayloadError("db_access_executed must be false in this deferred contract")
    if payload.get("persistence_mechanism") != DB_BACKED_QUEUE_DEFERRED:
        raise UnsafeFeedbackQueuePayloadError("unexpected persistence mechanism")
    status = payload.get("current_status")
    if status not in CURRENT_STATUS_VALUES:
        raise UnsafeFeedbackQueuePayloadError("current_status is outside the durable queue contract")
    _assert_no_unsafe_surface(payload)
    return True


class DisabledFeedbackQueueRepository:
    def enqueue(self, item: DurableFeedbackQueueItem) -> FeedbackQueuePersistenceResult:
        validate_minimized_feedback_queue_item(item)
        raise FeedbackQueuePersistenceNotEnabled("DB-backed feedback queue persistence is deferred")

    def read(self, feedback_id: str) -> DurableFeedbackQueueItem | None:
        _safe_token(feedback_id, "feedback_id")
        raise FeedbackQueuePersistenceNotEnabled("DB-backed feedback queue persistence is deferred")


class FakeFeedbackQueueRepository:
    def __init__(self) -> None:
        self._items_by_feedback_id: dict[str, DurableFeedbackQueueItem] = {}
        self._feedback_id_by_dedup_key: dict[str, str] = {}

    def enqueue(self, item: DurableFeedbackQueueItem) -> FeedbackQueuePersistenceResult:
        validate_minimized_feedback_queue_item(item)
        existing_feedback_id = self._feedback_id_by_dedup_key.get(item.dedup_key)
        if existing_feedback_id is not None:
            existing = self._items_by_feedback_id[existing_feedback_id]
            return FeedbackQueuePersistenceResult(
                accepted=True,
                feedback_id=existing.feedback_id,
                dedup_key=existing.dedup_key,
                current_status="duplicate",
                reason_code="DUPLICATE_DEDUP_KEY",
            )
        self._items_by_feedback_id[item.feedback_id] = item
        self._feedback_id_by_dedup_key[item.dedup_key] = item.feedback_id
        return FeedbackQueuePersistenceResult(
            accepted=True,
            feedback_id=item.feedback_id,
            dedup_key=item.dedup_key,
            current_status=item.current_status,
            reason_code="FAKE_REPOSITORY_ACCEPTED",
        )

    def read(self, feedback_id: str) -> DurableFeedbackQueueItem | None:
        safe_feedback_id = _safe_token(feedback_id, "feedback_id")
        return self._items_by_feedback_id.get(safe_feedback_id)


def default_disabled_feedback_queue_repository() -> FeedbackQueueRepository:
    return DisabledFeedbackQueueRepository()


__all__ = [
    "CONTRACT_VERSION",
    "CURRENT_STATUS_VALUES",
    "DB_ACCESS_EXECUTED",
    "DB_ACCESS_EXECUTED_BOUNDARY",
    "DB_BACKED_QUEUE_DEFERRED",
    "DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS",
    "INITIAL_CURRENT_STATUS_VALUES",
    "INTERNAL_PATH_INCLUDED",
    "RAW_TEXT_INCLUDED",
    "SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS",
    "DisabledFeedbackQueueRepository",
    "DurableFeedbackQueueItem",
    "FakeFeedbackQueueRepository",
    "FeedbackQueuePersistenceNotEnabled",
    "FeedbackQueuePersistenceResult",
    "FeedbackQueueRepository",
    "UnsafeFeedbackQueuePayloadError",
    "default_disabled_feedback_queue_repository",
    "durable_feedback_queue_item_from_hold",
    "validate_minimized_feedback_queue_item",
]
