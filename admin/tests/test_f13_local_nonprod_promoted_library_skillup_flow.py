import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from admin.f13_local_nonprod_library_promotion import canonical_payload as library_payload
from admin.f13_local_nonprod_promoted_library_skillup_flow import (
    INVALIDATION_REASON,
    LocalAnswerReleaseTraceStore,
    LocalPromotedLibraryBridgeAdapter,
    PromotedLibrarySkillupFlowError,
    build_feedback_payload,
    canonical_payload,
    execute_answer_or_hold,
    invalidate_after_rollback,
    validate_answer_release_trace,
    validate_feedback_event,
    validate_trace_storage_path,
)
from admin.f13_skillup_bridge import skillup_answer_from_request
from admin.tests.test_f13_local_nonprod_library_promotion import ORG, TENANT, _approval, _promote, _rollback


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_local_nonprod_answer_release_trace.schema.json"
NOW = datetime(2026, 7, 12, 8, 0, tzinfo=timezone.utc)


def _active(tmp_path):
    source, warehouse, item, candidate, plan, approval, library, promoted = _promote(tmp_path)
    approved = tmp_path / "trace-runtime"; approved.mkdir()
    store = LocalAnswerReleaseTraceStore(
        storage_path=approved / "answer_traces", approved_local_root=approved,
        environment="local_nonproduction", explicit_allow_local_trace_storage=True,
        clock=lambda: NOW,
    )
    return source, warehouse, item, candidate, plan, approval, library, promoted, LocalPromotedLibraryBridgeAdapter(), store


def _answer(tmp_path, *, request="request:R458:1", key="idem:R458:1"):
    *context, adapter, store = _active(tmp_path)
    library, promoted = context[6], context[7]
    flow = execute_answer_or_hold(
        adapter=adapter, trace_store=store, library_repository=library,
        tenant_id=TENANT, organization_id=ORG, request_id=request,
        library_record_id=promoted["library_record"]["library_record_id"], idempotency_key=key,
    )
    return (*context, adapter, store, flow)


def test_p1_explicit_activation_and_schema(tmp_path):
    *_, store = _active(tmp_path)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert store.database_path.is_file()


@pytest.mark.parametrize("environment,allow", [("production", True), ("prod", True), ("local_nonproduction", False)])
def test_n1_environment_or_allow_rejected(tmp_path, environment, allow):
    root = tmp_path / "root"; root.mkdir()
    with pytest.raises(PromotedLibrarySkillupFlowError):
        LocalAnswerReleaseTraceStore(storage_path=root / "answer_traces", approved_local_root=root, environment=environment, explicit_allow_local_trace_storage=allow)


@pytest.mark.parametrize("path", ["relative/path", r"\\server\share\traces"])
def test_n1_relative_and_unc_rejected(tmp_path, path):
    with pytest.raises(PromotedLibrarySkillupFlowError):
        validate_trace_storage_path(path, approved_local_root=tmp_path, environment="local_nonproduction", explicit_allow_local_trace_storage=True)


def test_n1_outside_escape_and_production_like_rejected(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    for path in (tmp_path / "outside", root / ".." / "escape", root / "prod" / "traces"):
        with pytest.raises(PromotedLibrarySkillupFlowError):
            validate_trace_storage_path(path, approved_local_root=root, environment="local_nonproduction", explicit_allow_local_trace_storage=True)


def test_p3_to_p8_active_library_bridge_pointer_and_continuity(tmp_path):
    *_, library, promoted, adapter, store = _active(tmp_path)
    result = adapter.retrieve(
        library, tenant_id=TENANT, organization_id=ORG, request_id="request:R458:bridge",
        purpose="SKILLUP_ANSWER", library_record_id=promoted["library_record"]["library_record_id"],
        allowed_rights_status=("owned",), max_items=1, role="STUDENT",
        course_id="course:R458:synthetic", module_id="module:R458:synthetic", binding_id="binding:R458:synthetic",
    )
    assert result["result_status"] == "OK"
    assert result["raw_text_included"] is False and result["internal_path_included"] is False
    item = result["evidence_items"][0]
    assert item["raw_text_policy"] == "POINTER_ONLY"
    assert item["source_doc_kind"] == "REFERENCE"
    assert item["bridge_trace_id"].startswith("btrace:local:")
    assert result["tenant_id"] == TENANT and result["organization_id"] == ORG
    assert result["cohort_id"] == promoted["library_record"]["cohort_id"]


def test_n5_rolled_back_record_produces_bridge_hold(tmp_path):
    *_, library, promoted, adapter, store = _active(tmp_path)
    library.rollback(_rollback(promoted), tenant_id=TENANT, organization_id=ORG)
    result = adapter.retrieve(library, tenant_id=TENANT, organization_id=ORG, request_id="request:R458:rolled", purpose="SKILLUP_ANSWER", library_record_id=promoted["library_record"]["library_record_id"], allowed_rights_status=("owned",), max_items=1, role="STUDENT", course_id="course:R458:synthetic", module_id="module:R458:synthetic", binding_id="binding:R458:synthetic")
    assert result["result_status"] == "HOLD" and result["evidence_items"] == []
    assert result["hold_reason"] == "EVIDENCE_INVALIDATED"


def test_n6_missing_evidence_produces_hold(tmp_path):
    *_, library, promoted, adapter, store = _active(tmp_path)
    library.connection.execute("UPDATE evidence_pointers SET active=0")
    result = adapter.retrieve(library, tenant_id=TENANT, organization_id=ORG, request_id="request:R458:no-evidence", purpose="SKILLUP_ANSWER", library_record_id=promoted["library_record"]["library_record_id"], allowed_rights_status=("owned",), max_items=1, role="STUDENT", course_id="course:R458:synthetic", module_id="module:R458:synthetic", binding_id="binding:R458:synthetic")
    assert result["result_status"] == "HOLD" and result["hold_reason"] == "EVIDENCE_INVALIDATED"


def test_n7_rolled_back_trace_is_ineligible(tmp_path):
    *_, library, promoted, adapter, store = _active(tmp_path)
    row = library.connection.execute("SELECT payload_json FROM promotion_traces").fetchone()
    trace = json.loads(row[0]); trace["status"] = "rolled_back"
    text, digest = library_payload(trace)
    library.connection.execute("UPDATE promotion_traces SET status='rolled_back',payload_json=?,payload_hash=?", (text, digest))
    result = adapter.retrieve(library, tenant_id=TENANT, organization_id=ORG, request_id="request:R458:rolled-trace", purpose="SKILLUP_ANSWER", library_record_id=promoted["library_record"]["library_record_id"], allowed_rights_status=("owned",), max_items=1, role="STUDENT", course_id="course:R458:synthetic", module_id="module:R458:synthetic", binding_id="binding:R458:synthetic")
    assert result["result_status"] == "HOLD"


def test_n8_rights_not_allowed_and_n9_cross_scope_hold(tmp_path):
    *_, library, promoted, adapter, store = _active(tmp_path)
    common = dict(request_id="request:R458:scope", purpose="SKILLUP_ANSWER", library_record_id=promoted["library_record"]["library_record_id"], max_items=1, role="STUDENT", course_id="course:R458:synthetic", module_id="module:R458:synthetic", binding_id="binding:R458:synthetic")
    denied = adapter.retrieve(library, tenant_id=TENANT, organization_id=ORG, allowed_rights_status=("licensed",), **common)
    hidden = adapter.retrieve(library, tenant_id="tenant:other", organization_id=ORG, allowed_rights_status=("owned",), **common)
    assert denied["result_status"] == "HOLD" and hidden["result_status"] == "HOLD"
    assert hidden["hold_reason"] == "EVIDENCE_INVALIDATED"


def test_n10_hash_mismatch_is_controlled_hold(tmp_path):
    *_, library, promoted, adapter, store = _active(tmp_path)
    library.connection.execute("UPDATE evidence_pointers SET payload_hash=?", ("sha256:" + "0" * 64,))
    result = adapter.retrieve(library, tenant_id=TENANT, organization_id=ORG, request_id="request:R458:mismatch", purpose="SKILLUP_ANSWER", library_record_id=promoted["library_record"]["library_record_id"], allowed_rights_status=("owned",), max_items=1, role="STUDENT", course_id="course:R458:synthetic", module_id="module:R458:synthetic", binding_id="binding:R458:synthetic")
    assert result["result_status"] == "HOLD" and result["hold_reason"] == "BRIDGE_SOURCE_READ_FAILED"


def test_p9_to_p13_answer_feedback_and_safe_active_trace(tmp_path):
    *_, promoted, adapter, store, flow = _answer(tmp_path)
    assert flow["result_status"] == "ANSWERED"
    assert flow["skillup"]["answer_status"] == "ANSWERED"
    assert flow["feedback_result"]["status"] == "READY"
    trace = flow["trace_result"]["answer_release_trace"]
    assert trace["status"] == "ACTIVE"
    assert trace["library_record_id"] == promoted["library_record"]["library_record_id"]
    assert trace["evidence_id"] == promoted["library_record"]["evidence_id"]
    assert trace["promotion_trace_id"] == promoted["promotion_trace"]["promotion_trace_id"]
    assert "answer" not in trace and validate_answer_release_trace(trace)["valid"]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(trace)


def test_n11_n12_skillup_direct_storage_paths_are_impossible():
    result = skillup_answer_from_request({"request_id": "request:R458:direct"})
    assert result["answer_status"] == "HOLD"
    assert result["feedback_candidate_required"] is True


def test_p2_and_n2_same_scope_read_cross_tenant_hidden(tmp_path):
    *_, store, flow = _answer(tmp_path)
    trace_id = flow["trace_result"]["answer_release_trace"]["answer_release_id"]
    assert store.read_trace(trace_id, tenant_id=TENANT, organization_id=ORG)["found"]
    hidden = store.read_trace(trace_id, tenant_id="tenant:other", organization_id=ORG)
    assert not hidden["found"] and hidden["reason_code"] == "RECORD_NOT_FOUND_OR_NOT_VISIBLE"


def test_n4_unscoped_trace_access_forbidden(tmp_path):
    *_, store, flow = _answer(tmp_path)
    with pytest.raises(PromotedLibrarySkillupFlowError, match="TENANT_SCOPE_REQUIRED"):
        store.read_trace(flow["trace_result"]["answer_release_trace"]["answer_release_id"], tenant_id=None, organization_id=ORG)


def test_p14_to_p22_rollback_invalidation_feedback_and_hold(tmp_path):
    source, warehouse, item, candidate, plan, approval, library, promoted, adapter, store, flow = _answer(tmp_path)
    trace = flow["trace_result"]["answer_release_trace"]
    library.rollback(_rollback(promoted), tenant_id=TENANT, organization_id=ORG)
    invalidated = invalidate_after_rollback(trace_store=store, library_repository=library, promotion_trace_id=trace["promotion_trace_id"], tenant_id=TENANT, organization_id=ORG, expected_revision=1, idempotency_key="idem:R458:invalidate")
    assert invalidated["reason_code"] == "INVALIDATED"
    changed = store.read_trace(trace["answer_release_id"], tenant_id=TENANT, organization_id=ORG)["answer_release_trace"]
    assert changed["status"] == "INVALIDATED" and changed["invalidation_reason"] == INVALIDATION_REASON
    assert invalidated["feedback_result"]["status"] == "READY"
    assert not store.serve_trace(trace["answer_release_id"], library, tenant_id=TENANT, organization_id=ORG)["reusable"]
    replay = invalidate_after_rollback(trace_store=store, library_repository=library, promotion_trace_id=trace["promotion_trace_id"], tenant_id=TENANT, organization_id=ORG, expected_revision=1, idempotency_key="idem:R458:invalidate")
    assert replay["reason_code"] == "IDEMPOTENT_REPLAY"
    hold = execute_answer_or_hold(adapter=adapter, trace_store=store, library_repository=library, tenant_id=TENANT, organization_id=ORG, request_id="request:R458:1", library_record_id=promoted["library_record"]["library_record_id"], idempotency_key="idem:R458:postrollback")
    assert hold["result_status"] == "HOLD" and hold["skillup"]["answer_status"] == "HOLD"
    assert hold["feedback_result"]["status"] == "READY" and hold["trace_result"] is None


def test_n13_conflicting_invalidation_and_n16_revision_conflict(tmp_path):
    *context, store, flow = _answer(tmp_path)
    library, promoted = context[6], context[7]
    trace = flow["trace_result"]["answer_release_trace"]
    library.rollback(_rollback(promoted), tenant_id=TENANT, organization_id=ORG)
    conflict = store.invalidate_promotion(trace["promotion_trace_id"], tenant_id=TENANT, organization_id=ORG, expected_revision=1, idempotency_key="idem:R458:conflict", reason="OTHER_REASON")
    revision = store.invalidate_promotion(trace["promotion_trace_id"], tenant_id=TENANT, organization_id=ORG, expected_revision=99, idempotency_key="idem:R458:revision")
    assert conflict["reason_code"] == "INVALIDATION_CONFLICT"
    assert revision["reason_code"] == "REVISION_CONFLICT"
    assert store.read_trace(trace["answer_release_id"], tenant_id=TENANT, organization_id=ORG)["answer_release_trace"]["status"] == "ACTIVE"


def test_n3_wrong_organization_invalidation_hidden(tmp_path):
    *_, store, flow = _answer(tmp_path)
    trace = flow["trace_result"]["answer_release_trace"]
    with pytest.raises(PromotedLibrarySkillupFlowError, match="RECORD_NOT_FOUND_OR_NOT_VISIBLE"):
        store.invalidate_promotion(trace["promotion_trace_id"], tenant_id=TENANT, organization_id="organization:other", expected_revision=1, idempotency_key="idem:R458:wrong-org")


def test_p23_invalidation_write_failure_still_serve_time_blocks(tmp_path):
    *context, store, flow = _answer(tmp_path)
    library, promoted = context[6], context[7]
    trace = flow["trace_result"]["answer_release_trace"]
    library.rollback(_rollback(promoted), tenant_id=TENANT, organization_id=ORG)
    with pytest.raises(PromotedLibrarySkillupFlowError, match="INJECTED_INVALIDATION_FAILURE"):
        invalidate_after_rollback(trace_store=store, library_repository=library, promotion_trace_id=trace["promotion_trace_id"], tenant_id=TENANT, organization_id=ORG, expected_revision=1, idempotency_key="idem:R458:failed-invalidate", failure_injection="write")
    assert store.read_trace(trace["answer_release_id"], tenant_id=TENANT, organization_id=ORG)["answer_release_trace"]["status"] == "ACTIVE"
    assert not store.serve_trace(trace["answer_release_id"], library, tenant_id=TENANT, organization_id=ORG)["reusable"]


def test_p24_feedback_failure_does_not_change_source_eligibility(tmp_path):
    *context, adapter, store = _active(tmp_path)
    library, promoted = context[6], context[7]
    with pytest.raises(PromotedLibrarySkillupFlowError, match="FEEDBACK_VALIDATION_FAILED"):
        execute_answer_or_hold(adapter=adapter, trace_store=store, library_repository=library, tenant_id=TENANT, organization_id=ORG, request_id="request:R458:feedback-fail", library_record_id=promoted["library_record"]["library_record_id"], idempotency_key="idem:R458:feedback-fail", failure_injection="feedback")
    assert library.read_library_record(promoted["library_record"]["library_record_id"], tenant_id=TENANT, organization_id=ORG)["found"]


@pytest.mark.parametrize("failure", ["after_record", "after_evidence"])
def test_p25_bridge_read_failure_is_controlled_hold(tmp_path, failure):
    *context, adapter, store = _active(tmp_path)
    library, promoted = context[6], context[7]
    result = execute_answer_or_hold(adapter=adapter, trace_store=store, library_repository=library, tenant_id=TENANT, organization_id=ORG, request_id=f"request:R458:{failure}", library_record_id=promoted["library_record"]["library_record_id"], idempotency_key=f"idem:R458:{failure}", failure_injection=failure)
    assert result["result_status"] == "HOLD" and result["trace_result"] is None


@pytest.mark.parametrize("failure", ["skillup", "trace_insert"])
def test_required_adapter_and_trace_failures_do_not_mutate_library(tmp_path, failure):
    *context, adapter, store = _active(tmp_path)
    library, promoted = context[6], context[7]
    before = library.object_counts()
    with pytest.raises(PromotedLibrarySkillupFlowError):
        execute_answer_or_hold(adapter=adapter, trace_store=store, library_repository=library, tenant_id=TENANT, organization_id=ORG, request_id=f"request:R458:{failure}", library_record_id=promoted["library_record"]["library_record_id"], idempotency_key=f"idem:R458:{failure}", failure_injection=failure)
    assert library.object_counts() == before


def test_p26_close_reopen_preserves_invalidation(tmp_path):
    *context, store, flow = _answer(tmp_path)
    library, promoted = context[6], context[7]
    trace = flow["trace_result"]["answer_release_trace"]
    library.rollback(_rollback(promoted), tenant_id=TENANT, organization_id=ORG)
    invalidate_after_rollback(trace_store=store, library_repository=library, promotion_trace_id=trace["promotion_trace_id"], tenant_id=TENANT, organization_id=ORG, expected_revision=1, idempotency_key="idem:R458:reopen")
    path, approved = store.storage_path, store.storage_path.parent; store.close()
    reopened = LocalAnswerReleaseTraceStore(storage_path=path, approved_local_root=approved, environment="local_nonproduction", explicit_allow_local_trace_storage=True, clock=lambda: NOW)
    assert reopened.read_trace(trace["answer_release_id"], tenant_id=TENANT, organization_id=ORG)["answer_release_trace"]["status"] == "INVALIDATED"


def test_p27_corrupt_trace_db_fails_closed(tmp_path):
    root = tmp_path / "trace-root"; root.mkdir(); storage = root / "answer_traces"; storage.mkdir()
    (storage / "answer_release_traces.sqlite").write_bytes(b"corrupt")
    with pytest.raises(PromotedLibrarySkillupFlowError, match="TRACE_STORAGE_OPEN_FAILED"):
        LocalAnswerReleaseTraceStore(storage_path=storage, approved_local_root=root, environment="local_nonproduction", explicit_allow_local_trace_storage=True)


def test_wrong_trace_schema_version_fails_closed(tmp_path):
    *_, store = _active(tmp_path)
    path, approved = store.storage_path, store.storage_path.parent
    store.connection.execute("UPDATE metadata SET value='999' WHERE key='schema_version'"); store.close()
    with pytest.raises(PromotedLibrarySkillupFlowError, match="TRACE_SCHEMA_VERSION_MISMATCH"):
        LocalAnswerReleaseTraceStore(storage_path=path, approved_local_root=approved, environment="local_nonproduction", explicit_allow_local_trace_storage=True)


def test_n14_and_p28_to_p32_repromotion_recovery(tmp_path):
    source, warehouse, item, candidate, plan, approval, library, promoted, adapter, store, flow = _answer(tmp_path)
    old_trace = flow["trace_result"]["answer_release_trace"]
    library.rollback(_rollback(promoted), tenant_id=TENANT, organization_id=ORG)
    invalidate_after_rollback(trace_store=store, library_repository=library, promotion_trace_id=old_trace["promotion_trace_id"], tenant_id=TENANT, organization_id=ORG, expected_revision=1, idempotency_key="idem:R458:recovery-invalidate")
    with pytest.raises(Exception, match="REUSED_APPROVAL_NOT_ALLOWED"):
        library.promote(warehouse, source, plan, approval, tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R458:old-approval")
    repromoted = library.promote(warehouse, source, plan, _approval(plan, "R458-new"), tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R458:repromotion")
    assert repromoted["promotion_trace"]["promotion_trace_id"] != promoted["promotion_trace"]["promotion_trace_id"]
    recovered = execute_answer_or_hold(adapter=adapter, trace_store=store, library_repository=library, tenant_id=TENANT, organization_id=ORG, request_id="request:R458:recovered", library_record_id=repromoted["library_record"]["library_record_id"], idempotency_key="idem:R458:recovered-answer")
    new_trace = recovered["trace_result"]["answer_release_trace"]
    assert recovered["result_status"] == "ANSWERED" and recovered["feedback_result"]["status"] == "READY"
    assert new_trace["answer_release_id"] != old_trace["answer_release_id"]
    assert store.read_trace(old_trace["answer_release_id"], tenant_id=TENANT, organization_id=ORG)["answer_release_trace"]["status"] == "INVALIDATED"
    library.rollback(_rollback(repromoted, "R458-new"), tenant_id=TENANT, organization_id=ORG)


def test_p33_same_trace_replay_and_n15_conflict_are_nonmutating(tmp_path):
    *context, adapter, store, first = _answer(tmp_path)
    library, promoted = context[6], context[7]
    replay = execute_answer_or_hold(adapter=adapter, trace_store=store, library_repository=library, tenant_id=TENANT, organization_id=ORG, request_id="request:R458:1", library_record_id=promoted["library_record"]["library_record_id"], idempotency_key="idem:R458:1")
    assert replay["trace_result"]["reason_code"] == "IDEMPOTENT_REPLAY"
    before = store.read_trace(first["trace_result"]["answer_release_trace"]["answer_release_id"], tenant_id=TENANT, organization_id=ORG)
    conflict = execute_answer_or_hold(adapter=adapter, trace_store=store, library_repository=library, tenant_id=TENANT, organization_id=ORG, request_id="request:R458:different", library_record_id=promoted["library_record"]["library_record_id"], idempotency_key="idem:R458:1")
    assert conflict["trace_result"]["reason_code"] == "IDEMPOTENCY_CONFLICT"
    assert store.read_trace(first["trace_result"]["answer_release_trace"]["answer_release_id"], tenant_id=TENANT, organization_id=ORG) == before


def test_p35_to_p40_privacy_boundary_and_source_nonmutation(tmp_path):
    source, warehouse, item, candidate, plan, approval, library, promoted, adapter, store, flow = _answer(tmp_path)
    warehouse_before = warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)
    source_before = source.list_records(tenant_id=TENANT, organization_id=ORG)
    persisted = "\n".join(row[0] for row in store.connection.execute("SELECT payload_json FROM answer_release_traces")).lower()
    for marker in ("raw_query", "raw_body", "answer_text", "evidence_text", "standard_text", "user_id", "api_key"):
        assert marker not in persisted
    assert "safe_summary" not in persisted and "pointer_uri" not in persisted
    assert warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG) == warehouse_before
    assert source.list_records(tenant_id=TENANT, organization_id=ORG) == source_before
    assert not (store.storage_path / "graph.db").exists()


def test_p42_canonical_serialization_is_deterministic():
    assert canonical_payload({"b": 2, "a": 1}) == canonical_payload({"a": 1, "b": 2})


def test_feedback_vocabulary_supports_all_three_events():
    for event, status, evidence, hold in (
        ("answer_rendered", "ANSWERED", "evidence:R458:1", None),
        ("invalidated_answer", "INVALIDATED", "evidence:R458:1", None),
        ("hold_created", "HOLD", None, "EVIDENCE_INVALIDATED"),
    ):
        payload = build_feedback_payload(event_type=event, answer_status=status, tenant_id=TENANT, organization_id=ORG, request_id=f"request:R458:{event}", bridge_trace_id=f"btrace:R458:{event}", evidence_id=evidence, hold_reason=hold)
        assert validate_feedback_event(payload)["status"] == "READY"


def test_p41_runtime_cleanup_can_remove_closed_task_store(tmp_path):
    *context, adapter, store = _active(tmp_path)
    path = store.storage_path; store.close(); context[6].close(); context[1].close(); context[0].close()
    shutil.rmtree(path)
    assert not path.exists()
