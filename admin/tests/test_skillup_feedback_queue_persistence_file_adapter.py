import json
from pathlib import Path

import pytest

from admin.f13_skillup_bridge import (
    skillup_answer_from_bridge_response,
    skillup_feedback_queue_item_from_hold,
)
from admin.f13_skillup_feedback_queue_persistence import (
    DB_BACKED_QUEUE_DEFERRED,
    DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS,
    UnsafeFeedbackQueuePayloadError,
    durable_feedback_queue_item_from_hold,
    validate_minimized_feedback_queue_item,
)
from admin.f13_skillup_feedback_queue_persistence_file import (
    LOCAL_FILE_ADAPTER_BOUNDARY,
    LOCAL_FILE_PERSISTENCE_MECHANISM,
    LocalFileFeedbackQueueRepository,
    LocalFileFeedbackQueueRepositoryError,
)


def _safe_helper_feedback_queue_item():
    hold_result = skillup_answer_from_bridge_response(
        {
            "result_status": "HOLD",
            "evidence_items": [],
            "hold_reason": "evidence_items are required for no-DB Bridge evaluation",
            "feedback_candidate_required": True,
            "raw_text_included": False,
            "internal_path_included": False,
        }
    )
    return skillup_feedback_queue_item_from_hold(hold_result)


def _safe_durable_item():
    return durable_feedback_queue_item_from_hold(_safe_helper_feedback_queue_item())


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_minimized_record(payload):
    assert set(payload) == DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS
    assert payload["persistence_mechanism"] == DB_BACKED_QUEUE_DEFERRED
    assert payload["raw_text_included"] is False
    assert payload["internal_path_included"] is False
    assert payload["db_access_executed"] is False
    assert payload["result_status"] == "HOLD"
    assert payload["answer_status"] == "HOLD"
    assert "answer" not in payload
    assert "evidence" not in payload
    assert validate_minimized_feedback_queue_item(payload) is True


def test_local_file_repository_writes_minimized_record_and_reports_local_boundary(tmp_path):
    repo = LocalFileFeedbackQueueRepository(tmp_path)
    item = _safe_durable_item()

    result = repo.enqueue(item)
    stored = repo.read(item.feedback_id)
    rows = _read_jsonl(repo.path)

    assert result.accepted is True
    assert result.feedback_id == item.feedback_id
    assert result.dedup_key == item.dedup_key
    assert result.current_status == item.current_status
    assert result.reason_code == "LOCAL_FILE_ACCEPTED"
    assert result.persistence_executed is True
    assert result.persistence_mechanism == LOCAL_FILE_PERSISTENCE_MECHANISM
    assert result.db_access_executed is False
    assert "not DB-backed or production persistence evidence" in LOCAL_FILE_ADAPTER_BOUNDARY
    assert stored == item
    assert len(rows) == 1
    _assert_minimized_record(rows[0])


def test_local_file_repository_reads_back_by_feedback_id_and_dedup_key(tmp_path):
    repo = LocalFileFeedbackQueueRepository(tmp_path)
    item = _safe_durable_item()

    repo.enqueue(item)

    assert repo.read(item.feedback_id) == item
    assert repo.read_by_dedup_key(item.dedup_key) == item
    assert repo.read("fbq:missing") is None


def test_local_file_repository_preserves_dedup_idempotency(tmp_path):
    repo = LocalFileFeedbackQueueRepository(tmp_path)
    item = _safe_durable_item()

    first = repo.enqueue(item)
    duplicate = repo.enqueue(item)
    rows = _read_jsonl(repo.path)

    assert first.reason_code == "LOCAL_FILE_ACCEPTED"
    assert duplicate.accepted is True
    assert duplicate.feedback_id == item.feedback_id
    assert duplicate.dedup_key == item.dedup_key
    assert duplicate.current_status == "duplicate"
    assert duplicate.reason_code == "LOCAL_FILE_DUPLICATE_FEEDBACK_ID"
    assert duplicate.persistence_executed is True
    assert duplicate.persistence_mechanism == LOCAL_FILE_PERSISTENCE_MECHANISM
    assert duplicate.db_access_executed is False
    assert len(rows) == 1
    _assert_minimized_record(rows[0])


def test_local_file_repository_rejects_raw_internal_secret_and_extra_surfaces_before_write(tmp_path):
    repo = LocalFileFeedbackQueueRepository(tmp_path)
    payload = _safe_durable_item().to_persistence_dict()
    payload["safe_summary"] = "raw standard text should not be persisted"

    with pytest.raises(UnsafeFeedbackQueuePayloadError):
        repo.enqueue(payload)

    extra_payload = _safe_durable_item().to_persistence_dict()
    extra_payload["safe_extra_note"] = "summary only"

    with pytest.raises(UnsafeFeedbackQueuePayloadError):
        repo.enqueue(extra_payload)

    secret_payload = _safe_durable_item().to_persistence_dict()
    secret_payload["api_token"] = "synthetic-token"

    with pytest.raises(UnsafeFeedbackQueuePayloadError):
        repo.enqueue(secret_payload)

    assert not repo.path.exists()


def test_local_file_repository_rejects_true_boundary_flags_before_write(tmp_path):
    repo = LocalFileFeedbackQueueRepository(tmp_path)

    for field_name in ("raw_text_included", "internal_path_included", "db_access_executed"):
        payload = _safe_durable_item().to_persistence_dict()
        payload[field_name] = True

        with pytest.raises(UnsafeFeedbackQueuePayloadError):
            repo.enqueue(payload)

    assert not repo.path.exists()


def test_local_file_repository_cleanup_removes_test_owned_records(tmp_path):
    repo = LocalFileFeedbackQueueRepository(tmp_path)
    item = _safe_durable_item()

    repo.enqueue(item)
    assert repo.read(item.feedback_id) == item

    removed_count = repo.cleanup()

    assert removed_count == 1
    assert repo.read(item.feedback_id) is None
    assert not repo.path.exists()


def test_local_file_repository_rejects_forbidden_storage_boundaries(tmp_path):
    with pytest.raises(LocalFileFeedbackQueueRepositoryError):
        LocalFileFeedbackQueueRepository(f"H:/\uc7a5\uae30\uae30\uc5b5/R9ZNW-312")

    with pytest.raises(LocalFileFeedbackQueueRepositoryError):
        LocalFileFeedbackQueueRepository(tmp_path, filename="feedback_token_records.jsonl")
