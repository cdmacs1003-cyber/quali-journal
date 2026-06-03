from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


CREATED_AT = "1970-01-01T00:00:00Z"
NOT_GRANTED = "NOT_GRANTED"
NOT_VERIFIED = "NOT_VERIFIED"
NOT_EXECUTED = "NOT_EXECUTED"

_UNSAFE_FIELD_MARKERS = (
    "raw_text",
    "raw_prompt",
    "raw_query",
    "full_source_text",
    "internal_path",
    "secret",
    "token",
    "credential",
    "api_key",
)
_UNSAFE_VALUE_MARKERS = (
    "raw text",
    "raw prompt",
    "raw query",
    "full source text",
    "h:\\",
    "c:\\",
    "file://",
    "secret",
    "token",
    "credential",
    "api key",
)
_OPEN_ITEM_LABELS = {
    "db_behavior": "DB_BEHAVIOR_NOT_VERIFIED",
    "production_raw_leak_safety": "PRODUCTION_RAW_LEAK_SAFETY_NOT_VERIFIED",
    "full_regression_safety": "FULL_REGRESSION_SAFETY_NOT_VERIFIED",
    "proofpack_status": "PROOFPACK_NOT_EXECUTED",
    "gate_matrix_status": "GATE_MATRIX_NOT_COMPLETE",
}


def _safe_text(value: Any, fallback: str, max_length: int = 180) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    if _unsafe_value(text):
        return "redacted_safety_summary"
    return text[:max_length]


def _safe_token(value: Any, fallback: str, max_length: int = 96) -> str:
    text = _safe_text(value, fallback, max_length=max_length)
    token = "".join(ch for ch in text if ch.isalnum() or ch in ":._-")
    return token or fallback


def _unsafe_value(value: Any) -> bool:
    lowered = str(value or "").lower()
    return any(marker in lowered for marker in _UNSAFE_VALUE_MARKERS)


def _unsafe_key(key: Any) -> bool:
    lowered = str(key or "").lower()
    return any(marker in lowered for marker in _UNSAFE_FIELD_MARKERS)


def _status_label(value: Any) -> str:
    status = str(value or "RECORDED").strip().upper().replace(" ", "_").replace("-", "_")
    if status in {"PASS", "PASSED", "SUCCESS"} or "PASS" in status:
        return "BOUNDED_EVIDENCE_RECORDED"
    if status in {"FAIL", "FAILED"}:
        return "REVIEW_REQUIRED"
    return _safe_token(status, "RECORDED")


def _stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _evidence_summary(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []

    summary: list[dict[str, Any]] = []
    for index, record in enumerate(records[:20], start=1):
        if not isinstance(record, Mapping):
            continue
        safe_record = {
            "record_id": _safe_token(record.get("record_id") or record.get("gate") or f"record:{index}", f"record:{index}"),
            "status": _status_label(record.get("status") or record.get("result")),
            "commit_id": _safe_token(record.get("commit_id") or record.get("commit"), "commit:not_recorded"),
            "test_count": int(record.get("test_count") or record.get("tests") or 0),
            "summary": _safe_text(record.get("summary"), "bounded evidence recorded"),
        }
        for key in record:
            if _unsafe_key(key):
                safe_record["summary"] = "redacted_safety_summary"
                break
        summary.append(safe_record)
    return summary


def _open_items(payload: Mapping[str, Any]) -> list[str]:
    items: list[str] = []
    if str(payload.get("db_behavior") or "").upper() in {NOT_EXECUTED, NOT_VERIFIED, ""}:
        items.append(_OPEN_ITEM_LABELS["db_behavior"])
    if str(payload.get("production_raw_leak_safety") or "").upper() in {NOT_VERIFIED, ""}:
        items.append(_OPEN_ITEM_LABELS["production_raw_leak_safety"])
    if str(payload.get("full_regression_safety") or "").upper() in {NOT_VERIFIED, ""}:
        items.append(_OPEN_ITEM_LABELS["full_regression_safety"])
    if str(payload.get("proofpack_status") or "").upper() in {NOT_EXECUTED, NOT_VERIFIED, ""}:
        items.append(_OPEN_ITEM_LABELS["proofpack_status"])
    if str(payload.get("gate_matrix_status") or "").upper() in {NOT_EXECUTED, NOT_VERIFIED, ""}:
        items.append(_OPEN_ITEM_LABELS["gate_matrix_status"])
    return items


def build_beta_release_board(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, Mapping) else {}
    scope = _safe_text(source.get("scope"), "Track A Skillup Beta")
    evidence_summary = _evidence_summary(source.get("evidence_records"))
    required_open_items = _open_items(source)
    signature = "|".join(
        sorted(
            f"{item['record_id']}:{item['status']}:{item['commit_id']}:{item['test_count']}:{item['summary']}"
            for item in evidence_summary
        )
    )
    board_digest = _stable_digest(scope, signature, ",".join(sorted(required_open_items)))
    gate_status = "READY_FOR_REVIEW_WITH_OPEN_ITEMS" if evidence_summary else "REVIEW_REQUIRED"
    recommendation = "REVIEW_REQUIRED"

    return {
        "release_board_id": f"beta-board:{board_digest}",
        "scope": scope,
        "gate_status": gate_status,
        "evidence_summary": evidence_summary,
        "required_open_items": required_open_items,
        "not_granted_claims": {
            "f13_pass": NOT_GRANTED,
            "track_a_pass": NOT_GRANTED,
            "beta_pass": NOT_GRANTED,
        },
        "not_verified_items": [
            "DB_BEHAVIOR",
            "PRODUCTION_RAW_LEAK_SAFETY",
            "FULL_REGRESSION_SAFETY",
        ],
        "approval_status": "NOT_APPROVED",
        "recommendation": recommendation,
        "created_at": CREATED_AT,
    }


__all__ = [
    "CREATED_AT",
    "NOT_EXECUTED",
    "NOT_GRANTED",
    "NOT_VERIFIED",
    "build_beta_release_board",
]
