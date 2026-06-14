import pytest

from admin.f13_skillup_bridge import (
    skillup_answer_from_bridge_response,
    skillup_feedback_queue_item_from_hold,
)
from admin.f13_skillup_feedback_queue_persistence import (
    DB_ACCESS_EXECUTED_BOUNDARY,
    DB_BACKED_QUEUE_DEFERRED,
    DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS,
    SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS,
    DisabledFeedbackQueueRepository,
    FakeFeedbackQueueRepository,
    FeedbackQueuePersistenceNotEnabled,
    UnsafeFeedbackQueuePayloadError,
    durable_feedback_queue_item_from_hold,
    validate_minimized_feedback_queue_item,
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


def _assert_minimized_persistence_payload(payload):
    assert set(payload) == DURABLE_FEEDBACK_QUEUE_ITEM_FIELDS
    assert payload["feedback_id"].startswith("fbq:")
    assert payload["origin_event_id"]
    assert payload["current_status"] in {"queued", "review_required"}
    assert payload["dedup_key"]
    assert payload["review_reason_code"]
    assert payload["safe_summary"]
    assert payload["raw_text_included"] is False
    assert payload["internal_path_included"] is False
    assert payload["db_access_executed"] is False
    assert payload["persistence_mechanism"] == DB_BACKED_QUEUE_DEFERRED
    assert "not durable persistence success evidence" in DB_ACCESS_EXECUTED_BOUNDARY
    assert validate_minimized_feedback_queue_item(payload) is True


def test_durable_feedback_queue_item_from_safe_helper_item_is_minimized_contract():
    item = durable_feedback_queue_item_from_hold(_safe_helper_feedback_queue_item())

    _assert_minimized_persistence_payload(item.to_persistence_dict())


def test_durable_feedback_queue_item_rejects_raw_internal_and_secret_like_payload():
    with pytest.raises(UnsafeFeedbackQueuePayloadError):
        durable_feedback_queue_item_from_hold(
            {
                "feedback_id": "fbq:unsafe",
                "origin_event_id": "hold:unsafe",
                "current_status": "queued",
                "dedup_key": "Skillup:HOLD_CASE:unsafe",
                "created_at": "1970-01-01T00:00:00Z",
                "review_reason_code": "HOLD_REVIEW_REQUIRED",
                "safe_summary": "raw standard text should not be persisted",
                "raw_source_payload": "full source should not persist",
                "internal_path": "H:\\secret\\standard.txt",
                "file_uri": "file://synthetic/internal/source.txt",
                "api_token": "do-not-store",
            }
        )


def test_durable_feedback_queue_item_rejects_hostnames_file_locations_and_true_flags():
    with pytest.raises(UnsafeFeedbackQueuePayloadError):
        durable_feedback_queue_item_from_hold(
            {
                **_safe_helper_feedback_queue_item(),
                "safe_summary": "review item points at internal.example.local",
            }
        )

    with pytest.raises(UnsafeFeedbackQueuePayloadError):
        durable_feedback_queue_item_from_hold(
            {
                **_safe_helper_feedback_queue_item(),
                "raw_text_included": True,
            }
        )


def test_default_disabled_repository_does_not_claim_persistence_execution():
    item = durable_feedback_queue_item_from_hold(_safe_helper_feedback_queue_item())
    repo = DisabledFeedbackQueueRepository()

    with pytest.raises(FeedbackQueuePersistenceNotEnabled):
        repo.enqueue(item)

    with pytest.raises(FeedbackQueuePersistenceNotEnabled):
        repo.read(item.feedback_id)


def test_fake_repository_accepts_only_minimized_records_and_preserves_idempotency():
    item = durable_feedback_queue_item_from_hold(_safe_helper_feedback_queue_item())
    repo = FakeFeedbackQueueRepository()

    first = repo.enqueue(item)
    duplicate = repo.enqueue(item)

    assert first.accepted is True
    assert first.feedback_id == item.feedback_id
    assert first.dedup_key == item.dedup_key
    assert first.db_access_executed is False
    assert first.persistence_executed is False
    assert duplicate.reason_code == "DUPLICATE_DEDUP_KEY"
    assert duplicate.current_status == "duplicate"
    assert repo.read(item.feedback_id) == item

    with pytest.raises(UnsafeFeedbackQueuePayloadError):
        repo.enqueue(
            durable_feedback_queue_item_from_hold(
                {
                    **_safe_helper_feedback_queue_item(),
                    "safe_summary": "synthetic credential should not persist",
                }
            )
        )


def test_selected_route_contract_keeps_queue_internals_out_of_response_surface():
    selected_route_body = {
        "schema_version": "1",
        "contract_version": "R9ZKY-2026-06-13",
        "trace_id": "btrace:skillup-bridge-safe-1",
        "answer_status": "HOLD",
        "result_status": "HOLD",
        "evidence_required": True,
        "evidence": [],
        "policy": {
            "raw_leak_check_passed": True,
            "rights_check_passed": False,
            "sensitivity_check_passed": False,
            "evidence_check_passed": False,
        },
        "raw_text_included": False,
        "internal_path_included": False,
        "review_required": True,
    }

    assert not (SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS & set(selected_route_body))
    assert {
        "feedback_queue_item",
        "feedback_candidate",
        "feedback_candidate_required",
        "created_at",
        "db_access_executed",
        "persistence_result",
    } <= SELECTED_ROUTE_FORBIDDEN_QUEUE_FIELDS
