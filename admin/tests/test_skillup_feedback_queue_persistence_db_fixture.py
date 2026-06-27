from __future__ import annotations

import sqlite3
from dataclasses import asdict

import pytest

from admin.f13_skillup_answer_hold_adapter import adapt_skillup_answer_hold_response
from admin.f13_skillup_bridge import (
    skillup_answer_from_bridge_response,
    skillup_feedback_queue_item_from_hold,
)
from admin.f13_skillup_feedback_queue_persistence import (
    DB_BACKED_QUEUE_DEFERRED,
    DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS,
    SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS,
    UnsafeFeedbackQueuePayloadError,
    durable_feedback_queue_item_from_hold,
    validate_minimized_feedback_queue_item,
)
from admin.f13_skillup_feedback_queue_persistence_db import (
    SQLITE_FIXTURE_PERSISTENCE_MECHANISM,
    SQLiteFeedbackQueueRepository,
    assert_selected_route_persistence_internals_absent,
)


@pytest.fixture()
def sqlite_fixture_repository():
    connection = sqlite3.connect(":memory:")
    repository = SQLiteFeedbackQueueRepository(connection)
    repository.ensure_schema()
    try:
        yield repository
    finally:
        repository.drop_schema()
        repository.dispose()


def _safe_helper_feedback_queue_item():
    hold_result = skillup_answer_from_bridge_response(
        {
            "result_status": "HOLD",
            "evidence_items": [],
            "hold_reason": "Bridge evidence gap requires curation before Skillup can answer.",
            "feedback_candidate_required": True,
            "raw_text_included": False,
            "internal_path_included": False,
        }
    )
    return skillup_feedback_queue_item_from_hold(hold_result)


def _safe_durable_item():
    return durable_feedback_queue_item_from_hold(_safe_helper_feedback_queue_item())


def _assert_minimized_durable_item(item):
    payload = item.to_persistence_dict()
    assert set(payload) == DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS
    assert payload["persistence_mechanism"] == DB_BACKED_QUEUE_DEFERRED
    assert payload["result_status"] == "HOLD"
    assert payload["answer_status"] == "HOLD"
    assert payload["evidence_required"] is True
    assert payload["review_required"] is True
    assert payload["evidence_count"] == 0
    assert payload["warning_codes"] == ("EVIDENCE_ARRAY_EMPTY_FOR_HOLD",)
    assert payload["raw_text_included"] is False
    assert payload["internal_path_included"] is False
    assert payload["db_access_executed"] is False
    assert validate_minimized_feedback_queue_item(payload) is True


def test_db_backed_repository_writes_minimized_durable_record(sqlite_fixture_repository):
    item = _safe_durable_item()

    result = sqlite_fixture_repository.enqueue(item)
    stored = sqlite_fixture_repository.read(item.feedback_id)

    assert result.accepted is True
    assert result.feedback_id == item.feedback_id
    assert result.dedup_key == item.dedup_key
    assert result.current_status == item.current_status
    assert result.reason_code == "SQLITE_FIXTURE_ACCEPTED"
    assert result.persistence_executed is True
    assert result.persistence_mechanism == SQLITE_FIXTURE_PERSISTENCE_MECHANISM
    assert result.raw_text_included is False
    assert result.internal_path_included is False
    assert stored == item
    _assert_minimized_durable_item(stored)


def test_db_backed_repository_reads_back_minimized_durable_record(sqlite_fixture_repository):
    item = _safe_durable_item()
    sqlite_fixture_repository.enqueue(item)

    by_feedback_id = sqlite_fixture_repository.read(item.feedback_id)
    by_dedup_key = sqlite_fixture_repository.read_by_dedup_key(item.dedup_key)

    assert by_feedback_id == item
    assert by_dedup_key == item
    assert by_feedback_id.safe_summary
    assert by_feedback_id.trace_id is None or isinstance(by_feedback_id.trace_id, str)
    assert by_feedback_id.request_id is None or isinstance(by_feedback_id.request_id, str)
    _assert_minimized_durable_item(by_feedback_id)


def test_db_backed_repository_preserves_dedup_idempotency(sqlite_fixture_repository):
    item = _safe_durable_item()

    first = sqlite_fixture_repository.enqueue(item)
    duplicate = sqlite_fixture_repository.enqueue(item)
    stored = sqlite_fixture_repository.read_by_dedup_key(item.dedup_key)

    assert first.reason_code == "SQLITE_FIXTURE_ACCEPTED"
    assert duplicate.reason_code == "SQLITE_FIXTURE_DUPLICATE_DEDUP_KEY"
    assert duplicate.current_status == "duplicate"
    assert duplicate.feedback_id == item.feedback_id
    assert stored == item


def test_db_backed_repository_rejects_raw_internal_secret_like_payload_before_write(sqlite_fixture_repository):
    item = _safe_durable_item()
    raw_summary_payload = {
        **item.to_persistence_dict(),
        "safe_summary": "raw standard text should not be persisted",
    }
    secret_field_payload = {
        **item.to_persistence_dict(),
        "api_token": "synthetic-token-must-not-persist",
    }

    with pytest.raises(UnsafeFeedbackQueuePayloadError):
        sqlite_fixture_repository.enqueue(raw_summary_payload)

    with pytest.raises(UnsafeFeedbackQueuePayloadError):
        sqlite_fixture_repository.enqueue(secret_field_payload)


def test_db_backed_repository_rejects_true_raw_internal_flags_before_write(sqlite_fixture_repository):
    item = _safe_durable_item()

    for field_name in ("raw_text_included", "internal_path_included", "db_access_executed"):
        payload = item.to_persistence_dict()
        payload[field_name] = True
        with pytest.raises(UnsafeFeedbackQueuePayloadError):
            sqlite_fixture_repository.enqueue(payload)


def test_db_backed_repository_cleanup_removes_fixture_records(sqlite_fixture_repository):
    item = _safe_durable_item()
    sqlite_fixture_repository.enqueue(item)

    assert sqlite_fixture_repository.read(item.feedback_id) == item
    removed_count = sqlite_fixture_repository.cleanup()

    assert removed_count == 1
    assert sqlite_fixture_repository.read(item.feedback_id) is None


def test_skillup_selected_route_persistence_hook_keeps_queue_internals_out_of_response(sqlite_fixture_repository):
    item = _safe_durable_item()
    result = sqlite_fixture_repository.enqueue(item)
    helper_response = {
        "result_status": "HOLD",
        "answer_status": "HOLD",
        "hold_reason": "Bridge evidence gap requires curation before Skillup can answer.",
        "feedback_queue_item": _safe_helper_feedback_queue_item(),
        "durable_feedback_queue_item": item.to_persistence_dict(),
        "persistence_result": asdict(result),
        "queue_write_result": {
            "feedback_id": item.feedback_id,
            "dedup_key": item.dedup_key,
            "persistence_mechanism": SQLITE_FIXTURE_PERSISTENCE_MECHANISM,
        },
        "raw_text_included": False,
        "internal_path_included": False,
    }

    selected_route_response = adapt_skillup_answer_hold_response(
        helper_response,
        request_context={"requester_module": "Skillup"},
        bridge_payload={},
    )

    assert_selected_route_persistence_internals_absent(selected_route_response)
    assert not (SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS & set(selected_route_response))
    assert selected_route_response["raw_text_included"] is False
    assert selected_route_response["internal_path_included"] is False
