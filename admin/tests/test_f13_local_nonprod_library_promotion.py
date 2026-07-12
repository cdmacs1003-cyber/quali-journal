import copy
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from admin.f13_local_nonprod_library_promotion import (
    LocalLibraryPromotionError,
    LocalNonprodLibraryPromotion,
    canonical_payload,
    validate_library_approval,
    validate_local_library_root,
    validate_rollback_authorization,
)
from admin.f13_local_nonprod_promotion_dry_run import PromotionDryRunPlanner
from admin.tests.test_f13_local_nonprod_promotion_dry_run import (
    APPROVAL as WAREHOUSE_APPROVAL,
    ORG,
    TENANT,
    _approved,
    _plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_local_nonprod_library_promotion_record.schema.json"
FIXED_NOW = datetime(2026, 7, 12, 4, 0, tzinfo=timezone.utc)


def _context(tmp_path):
    source, warehouse, item, candidate, _ = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    approved_root = tmp_path / "runtime"
    approved_root.mkdir()
    library = LocalNonprodLibraryPromotion(
        library_root=approved_root / "library",
        approved_local_root=approved_root,
        environment="local_nonproduction",
        explicit_allow_actual_local_promotion=True,
        clock=lambda: FIXED_NOW,
    )
    return source, warehouse, item, candidate, plan, library


def _approval(plan, suffix="1", **overrides):
    values = {
        "approval_record_id": f"approval:library:R457:{suffix}",
        "target_object_type": "promotion_plan",
        "target_object_id": plan["promotion_plan_id"],
        "approval_type": "APPROVE_LIBRARY_PROMOTION",
        "approver_id": "owner:synthetic:R457",
        "approver_role": "OWNER",
        "approved_at": "2026-07-12T02:00:00Z",
        "approval_scope": "LOCAL_NONPRODUCTION_REFERENCE",
        "approval_comment_code": "POINTER_ONLY_APPROVED",
        "evidence_id": plan["evidence_pointer"]["evidence_id"],
        "proofpack_id": "proofpack:R457:synthetic",
    }
    values.update(overrides)
    return values


def _promote(tmp_path, suffix="1"):
    source, warehouse, item, candidate, plan, library = _context(tmp_path)
    approval = _approval(plan, suffix)
    result = library.promote(
        warehouse, source, plan, approval,
        tenant_id=TENANT, organization_id=ORG,
        idempotency_key=f"idem:R457:{suffix}", expected_plan_hash=plan["plan_hash"],
    )
    return source, warehouse, item, candidate, plan, approval, library, result


def _rollback(result, suffix="1", **overrides):
    values = {
        "rollback_authorization_id": f"rollback:R457:{suffix}",
        "promotion_trace_id": result["promotion_trace"]["promotion_trace_id"],
        "library_record_id": result["library_record"]["library_record_id"],
        "actor_id": "owner:synthetic:R457",
        "actor_role": "OWNER",
        "reason_code": "LOCAL_ROLLBACK_REHEARSAL",
        "approved_at": "2026-07-12T03:00:00Z",
        "proofpack_id": "proofpack:R457:synthetic",
    }
    values.update(overrides)
    return values


def test_p1_explicit_activation_and_closed_schema(tmp_path):
    *_, library = _context(tmp_path)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert library.database_path.is_file()


@pytest.mark.parametrize("environment", ["production", "prod", "staging"])
def test_n1_production_like_environment_rejected(tmp_path, environment):
    root = tmp_path / "runtime"; root.mkdir()
    with pytest.raises(LocalLibraryPromotionError, match="LOCAL_NONPRODUCTION_ENVIRONMENT_REQUIRED"):
        LocalNonprodLibraryPromotion(library_root=root / "library", approved_local_root=root, environment=environment, explicit_allow_actual_local_promotion=True)


def test_n2_missing_explicit_allow_rejected(tmp_path):
    root = tmp_path / "runtime"; root.mkdir()
    with pytest.raises(LocalLibraryPromotionError, match="EXPLICIT_LOCAL_PROMOTION_ALLOW_REQUIRED"):
        LocalNonprodLibraryPromotion(library_root=root / "library", approved_local_root=root, environment="local_nonproduction", explicit_allow_actual_local_promotion=False)


@pytest.mark.parametrize("path", ["relative/library", r"\\server\share\library"])
def test_n3_relative_or_unc_path_rejected(tmp_path, path):
    with pytest.raises(LocalLibraryPromotionError):
        validate_local_library_root(path, approved_local_root=tmp_path, environment="local_nonproduction", explicit_allow_actual_local_promotion=True)


def test_n3_outside_root_and_n4_escape_rejected(tmp_path):
    root = tmp_path / "runtime"; root.mkdir()
    for path in (tmp_path / "outside", root / ".." / "escape"):
        with pytest.raises(LocalLibraryPromotionError):
            validate_local_library_root(path, approved_local_root=root, environment="local_nonproduction", explicit_allow_actual_local_promotion=True)


def test_n3_production_like_root_rejected(tmp_path):
    root = tmp_path / "runtime"; root.mkdir()
    with pytest.raises(LocalLibraryPromotionError, match="PRODUCTION_LIKE_ROOT_FORBIDDEN"):
        validate_local_library_root(root / "prod" / "library", approved_local_root=root, environment="local_nonproduction", explicit_allow_actual_local_promotion=True)


def test_p2_to_p6_eligibility_continuity_and_pointer_policy(tmp_path):
    source, warehouse, item, _, plan, approval, library, result = _promote(tmp_path)
    record = result["library_record"]
    assert record["warehouse_item_revision"] == item["revision"]
    for field in ("tenant_id", "organization_id", "cohort_id", "source_event_id", "source_request_id", "source_trace_id", "query_hash"):
        assert record[field] == plan[field]
    assert record["raw_text_policy"] == "POINTER_ONLY"
    assert result["approval_record"] == approval


def test_n5_nonapproved_item_and_n6_stale_revision_rejected(tmp_path):
    source, warehouse, item, _, plan, library = _context(tmp_path)
    approval = _approval(plan)

    class View:
        def __init__(self, changes): self.changes = changes
        def read_item(self, *args, **kwargs):
            result = copy.deepcopy(warehouse.read_item(*args, **kwargs)); result["item"].update(self.changes); return result
        def read_review_event_by_approval(self, *args, **kwargs): return warehouse.read_review_event_by_approval(*args, **kwargs)

    for changes, reason in (({"current_status": "review_ready"}, "WAREHOUSE_ITEM_NOT_APPROVED"), ({"revision": 999}, "REVISION_CONFLICT")):
        with pytest.raises(LocalLibraryPromotionError, match=reason):
            library.promote(View(changes), source, plan, approval, tenant_id=TENANT, organization_id=ORG, idempotency_key=f"idem:R457:{reason}")


def test_n7_missing_warehouse_approval_event_rejected(tmp_path):
    source, warehouse, _, _, plan, library = _context(tmp_path)

    class View:
        def read_item(self, *args, **kwargs): return warehouse.read_item(*args, **kwargs)
        def read_review_event_by_approval(self, *args, **kwargs): return {"found": False}

    with pytest.raises(LocalLibraryPromotionError, match="WAREHOUSE_APPROVAL_EVENT_REQUIRED"):
        library.promote(View(), source, plan, _approval(plan), tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:missing-event")


def test_n8_missing_library_approval_rejected(tmp_path):
    source, warehouse, _, _, plan, library = _context(tmp_path)
    with pytest.raises(LocalLibraryPromotionError, match="PROMOTION_INPUT_INVALID"):
        library.promote(warehouse, source, plan, None, tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:no-approval")


@pytest.mark.parametrize("changes", [
    {"target_object_id": "promotion:plan:other"},
    {"approver_role": "SYSTEM"},
    {"approval_type": "APPROVE_WAREHOUSE"},
    {"evidence_id": "evidence:pointer:other"},
    {"proofpack_id": ""},
])
def test_n9_to_n12_object_approval_policy_rejected(tmp_path, changes):
    source, warehouse, _, _, plan, library = _context(tmp_path)
    with pytest.raises(LocalLibraryPromotionError, match="LIBRARY_APPROVAL_INVALID"):
        library.promote(warehouse, source, plan, _approval(plan, **changes), tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:bad-approval")


def test_n13_stale_plan_hash_rejected(tmp_path):
    source, warehouse, _, _, plan, library = _context(tmp_path)
    with pytest.raises(LocalLibraryPromotionError, match="PROMOTION_PLAN_CONFLICT"):
        library.promote(warehouse, source, plan, _approval(plan), tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:plan-conflict", expected_plan_hash="sha256:" + "0" * 64)


def test_n14_deleted_candidate_and_expired_event_are_revalidated(tmp_path):
    source, warehouse, item, _, plan, library = _context(tmp_path)
    source.delete_record(item["source_candidate_record_id"], tenant_id=TENANT, organization_id=ORG)
    with pytest.raises(LocalLibraryPromotionError, match="SOURCE_CANDIDATE_NOT_ACTIVE_OR_VALID"):
        library.promote(warehouse, source, plan, _approval(plan), tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:deleted-source")

    expired_root = tmp_path / "expired"; expired_root.mkdir()
    source2, warehouse2, _, _, plan2, library2 = _context(expired_root)
    source2.connection.execute("UPDATE local_repository_records SET retention_until='2026-07-11T00:00:00Z' WHERE record_type='ANALYTICS_EVENT'")
    with pytest.raises(LocalLibraryPromotionError, match="SOURCE_EVENT_NOT_ACTIVE_OR_VALID"):
        library2.promote(warehouse2, source2, plan2, _approval(plan2, "expired"), tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:expired-source")


def test_n15_cross_tenant_and_organization_are_hidden(tmp_path):
    source, warehouse, _, _, plan, library = _context(tmp_path)
    for tenant, org in (("tenant:other", ORG), (TENANT, "organization:other")):
        with pytest.raises(LocalLibraryPromotionError, match="RECORD_NOT_FOUND_OR_NOT_VISIBLE"):
            library.promote(warehouse, source, plan, _approval(plan), tenant_id=tenant, organization_id=org, idempotency_key=f"idem:R457:{tenant}:{org}")


def test_n16_n17_rights_and_sensitivity_fail_closed(tmp_path):
    source, warehouse, _, _, plan, library = _context(tmp_path)

    class View:
        def __init__(self, changes): self.changes = changes
        def read_item(self, *args, **kwargs):
            result = copy.deepcopy(warehouse.read_item(*args, **kwargs)); result["item"].update(self.changes); return result
        def read_review_event_by_approval(self, *args, **kwargs): return warehouse.read_review_event_by_approval(*args, **kwargs)

    for changes in ({"rights_status": "unknown"}, {"sensitivity": "secret"}):
        with pytest.raises(LocalLibraryPromotionError, match="RIGHTS_OR_SENSITIVITY_NOT_ELIGIBLE"):
            library.promote(View(changes), source, plan, _approval(plan), tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:policy")


def test_p7_to_p20_reference_evidence_trace_card_index_and_schema(tmp_path):
    *_, library, result = _promote(tmp_path)
    record = result["library_record"]
    assert record["doc_kind"] == "REFERENCE" and record["canonical_lang"] == "EN"
    assert record["status"] == "ACTIVE" and record["revision"] == 1
    assert result["evidence_pointer"]["raw_text_policy"] == "POINTER_ONLY"
    assert result["promotion_trace"]["status"] == "PROMOTED_LOCAL_NONPRODUCTION"
    assert library.object_counts() == {"library_records": 1, "evidence_pointers": 1, "promotion_traces": 1, "approval_records": 1, "rollback_audits": 0}
    assert len(list(library.cards_root.glob("*.json"))) == 1
    assert json.loads(library.index_path.read_text(encoding="utf-8"))["active_records"][0]["node_id"] == record["node_id"]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)
    assert library.verify_readback(record["library_record_id"], tenant_id=TENANT, organization_id=ORG)["valid"]


def test_p18_close_reopen_preserves_integrity(tmp_path):
    source, warehouse, _, _, plan, approval, library, result = _promote(tmp_path)
    root = library.library_root; approved = root.parent; record_id = result["library_record"]["library_record_id"]
    library.close()
    reopened = LocalNonprodLibraryPromotion(library_root=root, approved_local_root=approved, environment="local_nonproduction", explicit_allow_actual_local_promotion=True, clock=lambda: FIXED_NOW)
    assert reopened.verify_readback(record_id, tenant_id=TENANT, organization_id=ORG)["valid"]


def test_p21_p22_identical_replay_has_same_ids_and_counts(tmp_path):
    source, warehouse, _, _, plan, approval, library, first = _promote(tmp_path)
    before = library.object_counts()
    replay = library.promote(warehouse, source, plan, approval, tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:1", expected_plan_hash=plan["plan_hash"])
    assert replay["reason_code"] == "IDEMPOTENT_REPLAY"
    assert replay["library_record"]["library_record_id"] == first["library_record"]["library_record_id"]
    assert replay["promotion_trace"]["promotion_trace_id"] == first["promotion_trace"]["promotion_trace_id"]
    assert library.object_counts() == before


def test_n18_idempotency_conflict_is_nonmutating(tmp_path):
    source, warehouse, _, _, plan, _, library, _ = _promote(tmp_path)
    before = library.object_counts()
    result = library.promote(warehouse, source, plan, _approval(plan, "2"), tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:1")
    assert result["reason_code"] == "IDEMPOTENCY_CONFLICT" and library.object_counts() == before


def test_n19_target_collision_fails_without_overwrite(tmp_path):
    source, warehouse, _, _, plan, _, library, result = _promote(tmp_path)
    auth = _rollback(result); library.rollback(auth, tenant_id=TENANT, organization_id=ORG)
    row = library.connection.execute("SELECT * FROM library_records").fetchone()
    payload = json.loads(row["payload_json"]); payload["source_candidate_id"] = "candidate:collision"
    text, digest = canonical_payload(payload)
    library.connection.execute("UPDATE library_records SET payload_json=?,payload_hash=?", (text, digest))
    with pytest.raises(LocalLibraryPromotionError, match="TARGET_ID_CONFLICT"):
        library.promote(warehouse, source, plan, _approval(plan, "2"), tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:collision")


@pytest.mark.parametrize("failure", [
    "before_transaction", "after_approval_verification", "after_library_record", "after_evidence",
    "after_trace", "before_card_finalization", "before_index_finalization", "during_atomic_rename",
    "during_transaction_commit", "during_readback_verification",
])
def test_p24_to_p34_atomic_failure_points_leave_no_partial_state(tmp_path, failure):
    source, warehouse, _, _, plan, library = _context(tmp_path)
    with pytest.raises(LocalLibraryPromotionError):
        library.promote(warehouse, source, plan, _approval(plan), tenant_id=TENANT, organization_id=ORG, idempotency_key=f"idem:R457:{failure}", failure_injection=failure)
    assert library.object_counts() == {"library_records": 0, "evidence_pointers": 0, "promotion_traces": 0, "approval_records": 0, "rollback_audits": 0}
    assert not list(library.cards_root.glob("*.json"))
    if library.index_path.exists(): assert json.loads(library.index_path.read_text(encoding="utf-8"))["active_records"] == []
    assert not list(library.library_root.glob(".*-stage-*"))


def test_p35_to_p43_valid_rollback_removes_visibility_and_keeps_audit(tmp_path):
    source, warehouse, item, candidate, plan, approval, library, result = _promote(tmp_path)
    source_before = source.list_records(tenant_id=TENANT, organization_id=ORG)
    item_before = warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)
    authorization = _rollback(result)
    rolled = library.rollback(authorization, tenant_id=TENANT, organization_id=ORG)
    assert rolled["reason_code"] == "ROLLED_BACK"
    assert not library.read_library_record(result["library_record"]["library_record_id"], tenant_id=TENANT, organization_id=ORG)["found"]
    assert library.read_promotion_trace(result["promotion_trace"]["promotion_trace_id"], tenant_id=TENANT, organization_id=ORG)["promotion_trace"]["status"] == "rolled_back"
    assert not list(library.cards_root.glob("*.json"))
    assert json.loads(library.index_path.read_text(encoding="utf-8"))["active_records"] == []
    assert library.object_counts()["rollback_audits"] == 1
    assert source.list_records(tenant_id=TENANT, organization_id=ORG) == source_before
    assert warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG) == item_before
    assert library.rollback(authorization, tenant_id=TENANT, organization_id=ORG)["reason_code"] == "IDEMPOTENT_REPLAY"


def test_n22_n23_cross_scope_rollback_hidden(tmp_path):
    *_, library, result = _promote(tmp_path)
    for tenant, org in (("tenant:other", ORG), (TENANT, "organization:other")):
        with pytest.raises(LocalLibraryPromotionError, match="RECORD_NOT_FOUND_OR_NOT_VISIBLE"):
            library.rollback(_rollback(result, tenant.replace(":", "-")), tenant_id=tenant, organization_id=org)


def test_n24_automated_rollback_actor_rejected(tmp_path):
    *_, library, result = _promote(tmp_path)
    with pytest.raises(LocalLibraryPromotionError, match="ROLLBACK_AUTHORIZATION_INVALID"):
        library.rollback(_rollback(result, actor_role="SYSTEM"), tenant_id=TENANT, organization_id=ORG)


def test_n25_conflicting_rollback_authorization_rejected(tmp_path):
    *_, library, result = _promote(tmp_path)
    authorization = _rollback(result); library.rollback(authorization, tenant_id=TENANT, organization_id=ORG)
    conflicting = dict(authorization); conflicting["promotion_trace_id"] = "promotion:trace:conflict"
    with pytest.raises(LocalLibraryPromotionError, match="ROLLBACK_CONFLICT"):
        library.rollback(conflicting, tenant_id=TENANT, organization_id=ORG)


@pytest.mark.parametrize("failure", ["before_projection_finalization", "before_commit", "during_readback"])
def test_p45_rollback_failures_preserve_active_promotion(tmp_path, failure):
    *_, library, result = _promote(tmp_path)
    with pytest.raises(LocalLibraryPromotionError, match="INJECTED_ROLLBACK_FAILURE"):
        library.rollback(_rollback(result, failure), tenant_id=TENANT, organization_id=ORG, failure_injection=failure)
    record_id = result["library_record"]["library_record_id"]
    assert library.read_library_record(record_id, tenant_id=TENANT, organization_id=ORG)["found"]
    assert library.verify_readback(record_id, tenant_id=TENANT, organization_id=ORG)["valid"]
    assert library.object_counts()["rollback_audits"] == 0


def test_n26_old_approval_blocked_and_p46_to_p49_new_approval_repromotes(tmp_path):
    source, warehouse, _, _, plan, approval, library, first = _promote(tmp_path)
    library.rollback(_rollback(first), tenant_id=TENANT, organization_id=ORG)
    with pytest.raises(LocalLibraryPromotionError, match="REUSED_APPROVAL_NOT_ALLOWED"):
        library.promote(warehouse, source, plan, approval, tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:old-approval-new-key")
    second = library.promote(warehouse, source, plan, _approval(plan, "2"), tenant_id=TENANT, organization_id=ORG, idempotency_key="idem:R457:repromotion")
    assert second["library_record"]["revision"] == 2
    assert second["promotion_trace"]["promotion_trace_id"] != first["promotion_trace"]["promotion_trace_id"]
    assert library.verify_readback(second["library_record"]["library_record_id"], tenant_id=TENANT, organization_id=ORG)["valid"]
    assert library.rollback(_rollback(second, "2"), tenant_id=TENANT, organization_id=ORG)["rolled_back"]


def test_p50_to_p58_privacy_boundary_determinism_and_non_echo(tmp_path):
    *_, library, result = _promote(tmp_path)
    persisted = []
    for table in ("library_records", "evidence_pointers", "promotion_traces", "approval_records"):
        persisted.extend(row[0] for row in library.connection.execute(f"SELECT payload_json FROM {table}"))
    joined = "\n".join(persisted).lower()
    for marker in ("raw_query", "raw_body", "answer_text", "evidence_text", "standard_text", "user_id", "api_key"):
        assert marker not in joined
    assert "standard" not in {result["library_record"]["doc_kind"].lower()}
    assert canonical_payload({"b": 2, "a": 1}) == canonical_payload({"a": 1, "b": 2})
    assert not (library.library_root / "graph.db").exists()
    assert result["library_record"]["scope_code"] == "LOCAL_NONPRODUCTION"


def test_validation_contracts_do_not_echo_rejected_values(tmp_path):
    _, _, _, _, plan, library = _context(tmp_path)
    approval = _approval(plan); approval["raw_query"] = "forbidden-value"
    validation = validate_library_approval(approval, plan)
    assert not validation["valid"] and "forbidden-value" not in json.dumps(validation)
    authorization = _rollback({"promotion_trace": {"promotion_trace_id": "promotion:trace:x"}, "library_record": {"library_record_id": "library:record:x"}})
    authorization["actor_role"] = "LLM"
    assert not validate_rollback_authorization(authorization)["valid"]


def test_corrupt_database_fails_closed(tmp_path):
    *_, library, result = _promote(tmp_path)
    library.connection.execute("UPDATE library_records SET payload_hash=?", ("sha256:" + "0" * 64,))
    with pytest.raises(LocalLibraryPromotionError, match="LIBRARY_RECORD_INTEGRITY_FAILED"):
        library.read_library_record(result["library_record"]["library_record_id"], tenant_id=TENANT, organization_id=ORG)


def test_wrong_schema_version_fails_closed_on_reopen(tmp_path):
    *_, library = _context(tmp_path)
    root = library.library_root; approved = root.parent
    library.connection.execute("UPDATE metadata SET value='999' WHERE key='schema_version'")
    library.close()
    with pytest.raises(LocalLibraryPromotionError, match="LOCAL_LIBRARY_SCHEMA_VERSION_MISMATCH"):
        LocalNonprodLibraryPromotion(library_root=root, approved_local_root=approved, environment="local_nonproduction", explicit_allow_actual_local_promotion=True)
