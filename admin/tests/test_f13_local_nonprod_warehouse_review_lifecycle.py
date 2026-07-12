import copy
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from admin.f13_analytics_improvement_candidate_contract import (
    map_consent_analytics_to_warehouse_candidate,
)
from admin.f13_local_nonprod_analytics_repository import (
    ENVIRONMENT_LOCAL_NONPRODUCTION,
    LocalNonprodAnalyticsRepository,
)
from admin.f13_local_nonprod_warehouse_review_lifecycle import (
    DECISIONS,
    ITEM_STATUSES,
    LocalNonprodWarehouseReviewLifecycle,
    WarehouseLifecycleError,
    build_warehouse_idempotency_key,
    canonical_payload,
    validate_review_event,
    validate_warehouse_item,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_local_nonprod_warehouse_review_record.schema.json"
BASE_TIME = datetime(2026, 7, 12, tzinfo=timezone.utc)
TENANT = "TEN-R455-SYNTHETIC"
ORG = "ORG-R455-SYNTHETIC"
COHORT = "COH-R455-SYNTHETIC"
QUERY_HASH = "sha256:" + "b" * 64
USER_HASH = "sha256:" + "a" * 64


def _schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _event(suffix="1", **overrides):
    event = {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "event_id": f"EVT-R455-SYNTHETIC-{suffix}",
        "tenant_id": TENANT,
        "organization_id": ORG,
        "cohort_id": COHORT,
        "user_id_hash": USER_HASH,
        "event_type": "hold_created",
        "request_id": f"REQ-R455-SYNTHETIC-{suffix}",
        "trace_id": f"btrace:R455:SYNTHETIC:{suffix}",
        "query_hash": QUERY_HASH,
        "query_summary": "hold_event",
        "raw_query_stored": False,
        "risk_flags": ["evidence_missing"],
        "occurred_at": "2026-07-12T00:00:00Z",
    }
    event.update(overrides)
    return event


def _allowed(suffix="1", **overrides):
    return {
        "policy_status": "ALLOW",
        "reason_code": "CONSENT_ALLOWED",
        "analytics_event_present": True,
        "analytics_event": _event(suffix, **overrides),
    }


def _candidate_result(suffix="1", **overrides):
    return map_consent_analytics_to_warehouse_candidate(
        _allowed(suffix, **overrides),
        improvement_trigger="evidence_hold",
        created_at="2026-07-12T00:00:00Z",
    )


def _source_repository(tmp_path, *, suffix="1", clock=None, retention="2026-07-14T00:00:00Z"):
    root = tmp_path / f"source-{suffix}"
    repository = LocalNonprodAnalyticsRepository(
        root / "source.sqlite",
        approved_local_root=root,
        environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
        explicit_allow_local_durable=True,
        clock=clock or (lambda: BASE_TIME),
    )
    repository.store_analytics_event(
        _allowed(suffix), retention_policy_id="RET-R455-SYNTHETIC", retention_until=retention
    )
    stored = repository.store_improvement_candidate(
        _candidate_result(suffix), retention_policy_id="RET-R455-SYNTHETIC", retention_until=retention
    )
    return repository, stored["record_id"], _candidate_result(suffix)["candidate"]


def _warehouse(tmp_path, *, name="warehouse.sqlite", environment=ENVIRONMENT_LOCAL_NONPRODUCTION, allow=True):
    root = tmp_path / "warehouse-root"
    return LocalNonprodWarehouseReviewLifecycle(
        root / name,
        approved_local_root=root,
        environment=environment,
        explicit_allow_local_durable=allow,
        clock=lambda: BASE_TIME,
    )


def _provenance(candidate):
    return {
        "provider_type": "analytics",
        "provider_ref": candidate["candidate_id"],
        "source_event_id": candidate["source_event_id"],
        "source_trace_id": candidate["source_trace_id"],
        "collection_reason": candidate["improvement_trigger"],
    }


def _intake(warehouse, source, record_id, candidate, *, idem_suffix="intake", **overrides):
    values = {
        "tenant_id": TENANT,
        "organization_id": ORG,
        "provenance": _provenance(candidate),
        "rights_status": "owned",
        "sensitivity": "internal",
        "classification": {"domain": "quality", "visibility": "internal_only"},
        "idempotency_key": build_warehouse_idempotency_key(TENANT, ORG, idem_suffix),
        "created_at": "2026-07-12T00:00:00Z",
    }
    values.update(overrides)
    return warehouse.intake_candidate(source, record_id, **values)


def _transition(warehouse, item, next_status, *, source=None, suffix=None, approval_event_id=None, **overrides):
    values = {
        "tenant_id": TENANT,
        "organization_id": ORG,
        "expected_status": item["current_status"],
        "next_status": next_status,
        "expected_revision": item["revision"],
        "actor_id": "reviewer:synthetic:curator",
        "actor_role": "CURATOR",
        "reason_code": f"R455_{next_status.upper()}",
        "idempotency_key": build_warehouse_idempotency_key(TENANT, ORG, suffix or next_status),
        "transitioned_at": "2026-07-12T00:10:00Z",
        "source_repository": source,
        "approval_event_id": approval_event_id,
    }
    values.update(overrides)
    return warehouse.transition_item(item["warehouse_item_id"], **values)


def _ready(warehouse, source, item):
    item = _transition(warehouse, item, "classified")["item"]
    item = _transition(warehouse, item, "source_verified", source=source)["item"]
    return _transition(warehouse, item, "review_ready")["item"]


def _review(warehouse, item, decision, *, role="CURATOR", suffix=None, approval=None, **overrides):
    values = {
        "tenant_id": TENANT,
        "organization_id": ORG,
        "review_event_id": f"review:R455:SYNTHETIC:{suffix or decision}",
        "reviewer_id": f"reviewer:synthetic:{role.lower()}",
        "reviewer_role": role,
        "decision": decision,
        "decision_reason_code": f"R455_{decision}",
        "expected_revision": item["revision"],
        "approval_event_id": approval,
        "reviewed_at": "2026-07-12T00:20:00Z",
        "idempotency_key": build_warehouse_idempotency_key(TENANT, ORG, suffix or decision),
    }
    values.update(overrides)
    return warehouse.review_item(item["warehouse_item_id"], **values)


def _captured(tmp_path, *, suffix="1", **intake_overrides):
    source, record_id, candidate = _source_repository(tmp_path, suffix=suffix)
    warehouse = _warehouse(tmp_path)
    result = _intake(warehouse, source, record_id, candidate, idem_suffix=f"intake-{suffix}", **intake_overrides)
    return source, warehouse, result["item"], candidate, record_id


def test_p1_explicit_local_nonproduction_activation_and_schema(tmp_path):
    warehouse = _warehouse(tmp_path)
    tables = {row[0] for row in warehouse.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"warehouse_repository_metadata", "warehouse_intake_items", "warehouse_review_events"} <= tables
    assert warehouse.integrity_check()["integrity_ok"] is True


@pytest.mark.parametrize("environment", ["production", "prod", "staging"])
def test_n1_production_environment_rejected(tmp_path, environment):
    with pytest.raises(Exception):
        _warehouse(tmp_path, environment=environment)


def test_n2_missing_explicit_allow_rejected(tmp_path):
    with pytest.raises(Exception):
        _warehouse(tmp_path, allow=False)


def test_n3_outside_root_path_rejected(tmp_path):
    with pytest.raises(Exception):
        LocalNonprodWarehouseReviewLifecycle(
            tmp_path / "outside.sqlite", approved_local_root=tmp_path / "approved",
            environment=ENVIRONMENT_LOCAL_NONPRODUCTION, explicit_allow_local_durable=True,
        )


@pytest.mark.parametrize("path", ["relative.sqlite", r"\\server\share\warehouse.sqlite", r"C:\production\warehouse.sqlite"])
def test_n4_relative_unc_and_production_like_paths_rejected(tmp_path, path):
    with pytest.raises(Exception):
        LocalNonprodWarehouseReviewLifecycle(
            path, approved_local_root=tmp_path,
            environment=ENVIRONMENT_LOCAL_NONPRODUCTION, explicit_allow_local_durable=True,
        )


def test_p2_to_p8_valid_candidate_becomes_bounded_captured_item(tmp_path):
    source, warehouse, item, candidate, _record_id = _captured(tmp_path)
    assert item["current_status"] == "captured"
    assert item["revision"] == 1
    assert item["review_required"] is True
    assert item["approved_for_library"] is False
    assert item["auto_promote"] is False
    assert (item["source_candidate_id"], item["source_event_id"], item["source_trace_id"], item["query_hash"]) == (
        candidate["candidate_id"], candidate["source_event_id"], candidate["source_trace_id"], candidate["query_hash"]
    )
    assert (item["tenant_id"], item["organization_id"], item["cohort_id"]) == (TENANT, ORG, COHORT)
    assert validate_warehouse_item(item)["valid"] is True
    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(item)
    source.close(); warehouse.close()


def test_p9_close_reopen_retains_item(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    path, root = warehouse.database_path, warehouse.approved_local_root
    warehouse.close()
    reopened = LocalNonprodWarehouseReviewLifecycle(
        path, approved_local_root=root, environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
        explicit_allow_local_durable=True, clock=lambda: BASE_TIME,
    )
    assert reopened.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"] == item
    source.close(); reopened.close()


def test_n5_missing_candidate_is_hidden(tmp_path):
    source, _record_id, candidate = _source_repository(tmp_path)
    warehouse = _warehouse(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match="CANDIDATE_NOT_ACTIVE_OR_VISIBLE"):
        _intake(warehouse, source, "record:candidate:missing", candidate)


def test_n6_expired_candidate_rejected(tmp_path):
    source, record_id, candidate = _source_repository(
        tmp_path, retention="2026-07-13T00:00:00Z"
    )
    source._clock = lambda: BASE_TIME + timedelta(days=2)
    warehouse = _warehouse(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match="CANDIDATE_NOT_ACTIVE_OR_VISIBLE"):
        _intake(warehouse, source, record_id, candidate)


def test_n7_deleted_candidate_rejected(tmp_path):
    source, record_id, candidate = _source_repository(tmp_path)
    source.delete_record(record_id, tenant_id=TENANT, organization_id=ORG)
    warehouse = _warehouse(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match="CANDIDATE_NOT_ACTIVE_OR_VISIBLE"):
        _intake(warehouse, source, record_id, candidate)


class _InvalidSource:
    def __init__(self, candidate): self.candidate = candidate
    def read_record(self, *_args, **_kwargs): return {"found": True, "domain_object": self.candidate}
    def list_records(self, **_kwargs): return {"records": []}


def test_n8_invalid_candidate_rejected_without_echo(tmp_path):
    candidate = _candidate_result()["candidate"]
    candidate = {**candidate, "status": "INVALID"}
    warehouse = _warehouse(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match="CANDIDATE_INVALID"):
        _intake(warehouse, _InvalidSource(candidate), "record:candidate:invalid", candidate)


@pytest.mark.parametrize("tenant,organization", [("TEN-OTHER", ORG), (TENANT, "ORG-OTHER")])
def test_n9_n10_cross_scope_candidate_is_hidden(tmp_path, tenant, organization):
    source, record_id, candidate = _source_repository(tmp_path)
    warehouse = _warehouse(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match="CANDIDATE_NOT_ACTIVE_OR_VISIBLE"):
        _intake(warehouse, source, record_id, candidate, tenant_id=tenant, organization_id=organization)


@pytest.mark.parametrize(
    "field,reason", [("provenance", "PROVENANCE_REQUIRED"), ("rights_status", "RIGHTS_STATUS_REQUIRED"), ("sensitivity", "SENSITIVITY_REQUIRED")]
)
def test_n11_to_n13_mandatory_intake_metadata(field, reason, tmp_path):
    source, record_id, candidate = _source_repository(tmp_path)
    warehouse = _warehouse(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match=reason):
        _intake(warehouse, source, record_id, candidate, **{field: None})


def test_n14_promotion_marked_candidate_rejected(tmp_path):
    candidate = copy.deepcopy(_candidate_result()["candidate"])
    candidate["auto_promote"] = True
    warehouse = _warehouse(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match="CANDIDATE_INVALID"):
        _intake(warehouse, _InvalidSource(candidate), "record:candidate:promotion-marked", candidate)


def test_p10_to_p15_selected_progression_and_exact_revision(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    revisions = [item["revision"]]
    for status in ("classified", "source_verified", "review_ready"):
        item = _transition(warehouse, item, status, source=source if status == "source_verified" else None)["item"]
        revisions.append(item["revision"])
    reviewed = _review(
        warehouse, item, "APPROVE_WAREHOUSE", approval="approval:R455:SYNTHETIC:1"
    )
    assert validate_review_event(reviewed["review_event"])["valid"] is True
    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(reviewed["review_event"])
    item = reviewed["item"]
    revisions.append(item["revision"])
    item = _transition(
        warehouse, item, "approved_for_warehouse", approval_event_id="approval:R455:SYNTHETIC:1"
    )["item"]
    revisions.append(item["revision"])
    assert item["current_status"] == "approved_for_warehouse"
    assert revisions == [1, 2, 3, 4, 5, 6]
    source.close(); warehouse.close()


@pytest.mark.parametrize(
    "expected_status,next_status",
    [("captured", "approved_for_warehouse"), ("source_verified", "approved_for_warehouse"), ("review_ready", "approved_for_library"), ("reviewed", "promoted")],
)
def test_n15_to_n18_forbidden_transitions(expected_status, next_status, tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    item = dict(item); item["current_status"] = expected_status
    with pytest.raises(WarehouseLifecycleError):
        _transition(warehouse, item, next_status)


def test_n19_wrong_revision_fails_without_mutation(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match="REVISION_CONFLICT"):
        _transition(warehouse, item, "classified", expected_revision=99)
    assert warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"] == item


@pytest.mark.parametrize("role", ["SYSTEM", "LLM", "AUTOMATED_AGENT"])
def test_n20_n21_automated_or_invalid_actor_cannot_transition_or_approve(role, tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match="ACTOR_ROLE_NOT_ALLOWED"):
        _transition(warehouse, item, "classified", actor_role=role)


@pytest.mark.parametrize(
    "decision,role,expected",
    [
        ("REQUEST_MORE_EVIDENCE", "CURATOR", "hold_review_needed"),
        ("REQUEST_RIGHTS_REVIEW", "RIGHTS_REVIEWER", "hold_copyright"),
        ("REQUEST_DOMAIN_REVIEW", "DOMAIN_EXPERT", "hold_review_needed"),
        ("REJECT", "CURATOR", "rejected"),
        ("QUARANTINE", "CURATOR", "quarantined"),
    ],
)
def test_p17_to_p21_distinct_human_decisions(decision, role, expected, tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    item = _ready(warehouse, source, item)
    result = _review(warehouse, item, decision, role=role)
    assert result["review_event"]["decision"] == decision
    assert result["item"]["current_status"] == expected
    assert result["review_event"]["approval_event_id"] is None


def test_p16_p22_curator_approval_requires_and_persists_approval_event(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    item = _ready(warehouse, source, item)
    result = _review(warehouse, item, "APPROVE_WAREHOUSE", approval="approval:R455:SYNTHETIC:2")
    assert result["review_event"]["approval_event_id"] == "approval:R455:SYNTHETIC:2"
    assert result["item"]["current_status"] == "reviewed"


def test_n22_approval_without_event_rejected(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    item = _ready(warehouse, source, item)
    with pytest.raises(WarehouseLifecycleError, match="APPROVAL_EVENT_REQUIRED"):
        _review(warehouse, item, "APPROVE_WAREHOUSE")


def test_n23_library_promotion_decision_is_not_in_contract(tmp_path):
    assert "APPROVE_LIBRARY_PROMOTION" not in DECISIONS
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    item = _ready(warehouse, source, item)
    with pytest.raises(WarehouseLifecycleError, match="REVIEW_DECISION_INVALID"):
        _review(warehouse, item, "APPROVE_LIBRARY_PROMOTION")


@pytest.mark.parametrize("field,reason", [("decision_reason_code", "DECISION_REASON_REQUIRED"), ("reviewer_id", "REVIEWER_ID_REQUIRED")])
def test_n24_n25_required_review_identity_and_reason(field, reason, tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    item = _ready(warehouse, source, item)
    with pytest.raises(WarehouseLifecycleError, match=reason):
        _review(warehouse, item, "REJECT", **{field: ""})


def test_p23_intake_replay_and_n26_conflict_are_nonmutating(tmp_path):
    source, record_id, candidate = _source_repository(tmp_path)
    warehouse = _warehouse(tmp_path)
    first = _intake(warehouse, source, record_id, candidate)
    replay = _intake(warehouse, source, record_id, candidate)
    conflict = _intake(warehouse, source, record_id, candidate, sensitivity="restricted")
    assert replay["reason_code"] == "IDEMPOTENT_REPLAY"
    assert conflict["reason_code"] == "IDEMPOTENCY_CONFLICT"
    assert warehouse.count_items(tenant_id=TENANT, organization_id=ORG) == 1
    assert warehouse.read_item(first["item"]["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"] == first["item"]


def test_p24_n27_review_replay_and_conflict_are_nonmutating(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    item = _ready(warehouse, source, item)
    first = _review(warehouse, item, "REJECT", suffix="review-replay")
    replay = _review(warehouse, item, "REJECT", suffix="review-replay")
    conflict = _review(warehouse, item, "QUARANTINE", suffix="review-replay")
    assert replay["reason_code"] == "IDEMPOTENT_REPLAY"
    assert conflict["reason_code"] == "IDEMPOTENCY_CONFLICT"
    assert warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"] == first["item"]


def test_p26_same_scope_read_and_list(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    assert warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["found"] is True
    assert warehouse.list_items(tenant_id=TENANT, organization_id=ORG)["count"] == 1


@pytest.mark.parametrize("tenant,organization", [("TEN-OTHER", ORG), (TENANT, "ORG-OTHER")])
def test_n28_n31_cross_scope_read_hidden(tenant, organization, tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    result = warehouse.read_item(item["warehouse_item_id"], tenant_id=tenant, organization_id=organization)
    assert result == {"found": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "item": None}


def test_n29_n30_cross_scope_transition_and_review_hidden(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    transitioned = _transition(warehouse, item, "classified", tenant_id="TEN-OTHER")
    assert transitioned["reason_code"] == "RECORD_NOT_FOUND_OR_NOT_VISIBLE"
    item = _ready(warehouse, source, item)
    reviewed = _review(warehouse, item, "REJECT", tenant_id="TEN-OTHER")
    assert reviewed["reason_code"] == "RECORD_NOT_FOUND_OR_NOT_VISIBLE"


@pytest.mark.parametrize("method", ["read", "list", "delete"])
def test_n32_unscoped_operations_forbidden(method, tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    with pytest.raises(WarehouseLifecycleError):
        if method == "read": warehouse.read_item(item["warehouse_item_id"], tenant_id="", organization_id=ORG)
        elif method == "list": warehouse.list_items(tenant_id=TENANT, organization_id="")
        else: warehouse.delete_item(item["warehouse_item_id"], tenant_id="", organization_id=ORG)


def test_p27_intake_rollback_on_injected_failure(tmp_path):
    source, record_id, candidate = _source_repository(tmp_path)
    warehouse = _warehouse(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match="INJECTED_INTAKE_FAILURE"):
        _intake(warehouse, source, record_id, candidate, inject_failure=True)
    assert warehouse.count_items(tenant_id=TENANT, organization_id=ORG) == 0


def test_p28_transition_rollback_on_injected_failure(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    with pytest.raises(WarehouseLifecycleError, match="INJECTED_TRANSITION_FAILURE"):
        _transition(warehouse, item, "classified", inject_failure=True)
    assert warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"] == item


def test_p29_review_rollback_on_injected_failure(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    item = _ready(warehouse, source, item)
    with pytest.raises(WarehouseLifecycleError, match="INJECTED_REVIEW_FAILURE"):
        _review(warehouse, item, "REJECT", inject_failure=True)
    assert warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"] == item
    assert warehouse.connection.execute("SELECT COUNT(*) FROM warehouse_review_events").fetchone()[0] == 0


def test_p30_close_reopen_integrity_after_review(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    item = _ready(warehouse, source, item)
    item = _review(warehouse, item, "REJECT")["item"]
    path, root = warehouse.database_path, warehouse.approved_local_root
    warehouse.close()
    reopened = LocalNonprodWarehouseReviewLifecycle(
        path, approved_local_root=root, environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
        explicit_allow_local_durable=True, clock=lambda: BASE_TIME,
    )
    assert reopened.integrity_check()["integrity_ok"] is True
    assert reopened.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"] == item


def test_p31_wrong_schema_version_fails_closed(tmp_path):
    warehouse = _warehouse(tmp_path)
    path, root = warehouse.database_path, warehouse.approved_local_root
    warehouse.connection.execute("UPDATE warehouse_repository_metadata SET metadata_value='999' WHERE metadata_key='repository_schema_version'")
    warehouse.close()
    with pytest.raises(WarehouseLifecycleError, match="WAREHOUSE_SCHEMA_VERSION_UNSUPPORTED"):
        LocalNonprodWarehouseReviewLifecycle(
            path, approved_local_root=root, environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
            explicit_allow_local_durable=True,
        )


def test_p32_corrupt_file_fails_closed_and_is_preserved(tmp_path):
    root = tmp_path / "warehouse-root"; root.mkdir()
    path = root / "warehouse.sqlite"; original = b"not-a-database"
    path.write_bytes(original)
    with pytest.raises(WarehouseLifecycleError, match="CORRUPT_OR_NON_SQLITE_STORAGE"):
        LocalNonprodWarehouseReviewLifecycle(
            path, approved_local_root=root, environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
            explicit_allow_local_durable=True,
        )
    assert path.read_bytes() == original


def test_p33_to_p38_persisted_boundary_counters_zero(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    item = _ready(warehouse, source, item)
    _review(warehouse, item, "APPROVE_WAREHOUSE", approval="approval:R455:SYNTHETIC:boundary")
    stored = "\n".join(row[0] for row in warehouse.connection.execute(
        "SELECT payload_json FROM warehouse_intake_items UNION ALL SELECT payload_json FROM warehouse_review_events"
    ))
    forbidden_storage_markers = (
        '"raw_query"', '"raw_body"', '"answer_text"', '"evidence_text"',
        '"user_id"', '"personal_name"', '"library_id"', '"promotion_trace_id"',
    )
    assert all(marker not in stored for marker in forbidden_storage_markers)
    assert item["approved_for_library"] is False and item["auto_promote"] is False


def test_p39_canonical_serialization_is_deterministic():
    left = {"b": 2, "a": {"d": 4, "c": 3}}
    right = {"a": {"c": 3, "d": 4}, "b": 2}
    assert canonical_payload(left) == canonical_payload(right)


def test_p40_conflicts_do_not_echo_rejected_payload(tmp_path):
    source, record_id, candidate = _source_repository(tmp_path)
    warehouse = _warehouse(tmp_path)
    _intake(warehouse, source, record_id, candidate)
    conflict = _intake(warehouse, source, record_id, candidate, sensitivity="restricted")
    assert conflict == {"write_performed": False, "reason_code": "IDEMPOTENCY_CONFLICT", "item": None}


def test_scoped_delete_cleanup_is_hidden_and_idempotent(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    hidden = warehouse.delete_item(item["warehouse_item_id"], tenant_id="TEN-OTHER", organization_id=ORG)
    deleted = warehouse.delete_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)
    repeated = warehouse.delete_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)
    assert hidden["deleted_count"] == 0 and deleted["deleted_count"] == 1 and repeated["deleted_count"] == 0


def test_contract_contains_no_library_or_promotion_state():
    assert "approved_for_library" not in ITEM_STATUSES
    assert "promoted" not in ITEM_STATUSES
    assert all("LIBRARY" not in decision and "PROMOTION" not in decision for decision in DECISIONS)
