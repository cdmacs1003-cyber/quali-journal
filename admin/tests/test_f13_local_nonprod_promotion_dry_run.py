import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from admin.f13_local_nonprod_promotion_dry_run import (
    ALLOWED_FILES,
    PromotionDryRunError,
    PromotionDryRunPlanner,
    build_promotion_idempotency_key,
    canonical_payload,
    cleanup_materialization,
    materialize_pointer_only_evidence,
    validate_evidence_pointer,
    validate_promotion_plan,
    verify_materialization,
)
from admin.tests.test_f13_local_nonprod_warehouse_review_lifecycle import (
    BASE_TIME,
    ORG,
    TENANT,
    _captured,
    _review,
    _transition,
    _ready,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_local_nonprod_promotion_plan.schema.json"
APPROVAL = "approval:R456:SYNTHETIC:1"
CREATOR = "reviewer:synthetic:owner"


def _schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _approved(tmp_path, **intake_overrides):
    source, warehouse, item, candidate, record_id = _captured(
        tmp_path, **intake_overrides
    )
    item = _ready(warehouse, source, item)
    item = _review(
        warehouse, item, "APPROVE_WAREHOUSE", approval=APPROVAL,
        suffix="R456-approve",
    )["item"]
    item = _transition(
        warehouse, item, "approved_for_warehouse", approval_event_id=APPROVAL,
        suffix="R456-final",
    )["item"]
    return source, warehouse, item, candidate, record_id


def _plan(planner, warehouse, source, item, **overrides):
    values = {
        "tenant_id": TENANT,
        "organization_id": ORG,
        "expected_revision": item["revision"],
        "approval_event_id": APPROVAL,
        "idempotency_key": build_promotion_idempotency_key(TENANT, ORG, "R456-plan"),
        "created_by": CREATOR,
        "created_at": "2026-07-12T01:00:00Z",
    }
    values.update(overrides)
    return planner.create_plan(warehouse, source, item["warehouse_item_id"], **values)


class _WarehouseView:
    def __init__(self, wrapped, *, item_changes=None, event_changes=None, missing_event=False):
        self.wrapped = wrapped
        self.item_changes = item_changes or {}
        self.event_changes = event_changes or {}
        self.missing_event = missing_event

    def read_item(self, *args, **kwargs):
        result = copy.deepcopy(self.wrapped.read_item(*args, **kwargs))
        if result.get("item"):
            result["item"].update(self.item_changes)
        return result

    def read_review_event_by_approval(self, *args, **kwargs):
        if self.missing_event:
            return {"found": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "review_event": None}
        result = copy.deepcopy(self.wrapped.read_review_event_by_approval(*args, **kwargs))
        if result.get("review_event"):
            result["review_event"].update(self.event_changes)
        return result


class _SourceView:
    def __init__(self, wrapped, *, event_changes=None):
        self.wrapped = wrapped
        self.event_changes = event_changes or {}

    def list_records(self, **kwargs):
        return self.wrapped.list_records(**kwargs)

    def read_record(self, *args, **kwargs):
        result = copy.deepcopy(self.wrapped.read_record(*args, **kwargs))
        obj = result.get("domain_object")
        if isinstance(obj, dict) and "event_id" in obj:
            obj.update(self.event_changes)
        return result


def test_p1_to_p6_approved_item_event_and_pointer_rights_are_eligible(tmp_path):
    source, warehouse, item, candidate, _record_id = _approved(tmp_path)
    result = _plan(PromotionDryRunPlanner(), warehouse, source, item)
    plan = result["plan"]
    assert result["reason_code"] == "DRY_RUN_PLAN_READY"
    assert plan["warehouse_item_revision"] == item["revision"]
    assert plan["approval_event_id"] == APPROVAL
    assert plan["cohort_id"] == item["cohort_id"]
    assert (plan["source_candidate_id"], plan["source_event_id"], plan["source_trace_id"], plan["query_hash"]) == (
        candidate["candidate_id"], candidate["source_event_id"], candidate["source_trace_id"], candidate["query_hash"]
    )
    assert validate_evidence_pointer(plan["evidence_pointer"])["valid"] is True


def test_n1_nonapproved_warehouse_state_rejected(tmp_path):
    source, warehouse, item, _candidate, _record_id = _captured(tmp_path)
    with pytest.raises(PromotionDryRunError, match="WAREHOUSE_ITEM_NOT_APPROVED"):
        _plan(PromotionDryRunPlanner(), warehouse, source, item)


def test_n2_stale_revision_rejected(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    with pytest.raises(PromotionDryRunError, match="STALE_WAREHOUSE_REVISION"):
        _plan(PromotionDryRunPlanner(), warehouse, source, item, expected_revision=item["revision"] - 1)


def test_n3_missing_approval_event_rejected(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    with pytest.raises(PromotionDryRunError, match="APPROVAL_EVENT_NOT_FOUND"):
        _plan(PromotionDryRunPlanner(), _WarehouseView(warehouse, missing_event=True), source, item)


def test_n4_approval_event_identity_mismatch_rejected(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    view = _WarehouseView(warehouse, event_changes={"approval_event_id": "approval:R456:SYNTHETIC:other"})
    with pytest.raises(PromotionDryRunError, match="APPROVAL_EVENT_MISMATCH"):
        _plan(PromotionDryRunPlanner(), view, source, item)


def test_n5_invalid_reviewer_role_rejected(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    view = _WarehouseView(warehouse, event_changes={"reviewer_role": "SYSTEM"})
    with pytest.raises(PromotionDryRunError, match="APPROVAL_REVIEWER_ROLE_INVALID"):
        _plan(PromotionDryRunPlanner(), view, source, item)


def test_n6_deleted_source_candidate_rejected(tmp_path):
    source, warehouse, item, _candidate, record_id = _approved(tmp_path)
    source.delete_record(record_id, tenant_id=TENANT, organization_id=ORG)
    with pytest.raises(PromotionDryRunError, match="SOURCE_CANDIDATE_NOT_ACTIVE_OR_VALID"):
        _plan(PromotionDryRunPlanner(), warehouse, source, item)


def test_n7_expired_source_event_rejected(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    source._clock = lambda: BASE_TIME + timedelta(days=3)
    with pytest.raises(PromotionDryRunError, match="SOURCE_CANDIDATE_NOT_ACTIVE_OR_VALID"):
        _plan(PromotionDryRunPlanner(), warehouse, source, item)


@pytest.mark.parametrize("tenant,organization", [("TEN-OTHER", ORG), (TENANT, "ORG-OTHER")])
def test_n8_n9_cross_scope_item_hidden(tenant, organization, tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    with pytest.raises(PromotionDryRunError, match="WAREHOUSE_ITEM_NOT_FOUND_OR_NOT_VISIBLE"):
        _plan(PromotionDryRunPlanner(), warehouse, source, item, tenant_id=tenant, organization_id=organization)


@pytest.mark.parametrize(
    "event_changes",
    [{"trace_id": "btrace:R456:SYNTHETIC:changed"}, {"query_hash": "sha256:" + "c" * 64}],
)
def test_n10_n11_trace_or_query_hash_mismatch_rejected(event_changes, tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    with pytest.raises(PromotionDryRunError, match="SOURCE_CONTINUITY_INVALID"):
        _plan(PromotionDryRunPlanner(), warehouse, _SourceView(source, event_changes=event_changes), item)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"rights_status": "unknown"}, "RIGHTS_NOT_POINTER_ELIGIBLE"),
        ({"sensitivity": "secret"}, "SENSITIVITY_NOT_DRY_RUN_ELIGIBLE"),
        ({"approved_for_library": True}, "PROMOTION_MARKER_FORBIDDEN"),
        ({"auto_promote": True}, "PROMOTION_MARKER_FORBIDDEN"),
    ],
)
def test_n12_to_n15_policy_markers_rejected(changes, reason, tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    with pytest.raises(PromotionDryRunError, match=reason):
        _plan(PromotionDryRunPlanner(), _WarehouseView(warehouse, item_changes=changes), source, item)


def test_p7_to_p16_closed_plan_contract_and_deterministic_identity(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    first = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    second = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    assert first == second
    assert first["status"] == "dry_run_ready"
    assert first["human_approval_required"] is True
    assert first["actual_promotion_performed"] is False
    assert first["library_write_performed"] is False
    assert first["target_doc_kind"] == "REFERENCE"
    assert first["target_canonical_lang"] == "EN"
    assert first["raw_text_policy"] == "POINTER_ONLY"
    assert first["target_library_node_id"].startswith("QLIB:ANALYTICS_REF:")
    assert validate_promotion_plan(first)["valid"] is True
    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(first)


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_doc_kind", "STANDARD"),
        ("target_canonical_lang", "KO"),
        ("raw_text_policy", "SAFE_SUMMARY_ONLY"),
    ],
)
def test_n16_to_n18_fixed_plan_boundary_rejected(field, value, tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    changed = {**plan, field: value}
    assert validate_promotion_plan(changed)["valid"] is False


def test_n19_unexpected_plan_property_rejected(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    plan = {**plan, "unexpected": "marker"}
    assert validate_promotion_plan(plan)["valid"] is False


def test_target_identity_collision_fails_closed(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    target = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]["target_library_node_id"]
    with pytest.raises(PromotionDryRunError, match="TARGET_ID_CONFLICT"):
        _plan(PromotionDryRunPlanner(), warehouse, source, item, existing_target_ids=[target])


def test_plan_idempotent_replay_and_conflict(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    planner = PromotionDryRunPlanner()
    first = _plan(planner, warehouse, source, item)
    replay = _plan(planner, warehouse, source, item)
    conflict = _plan(planner, warehouse, source, item, expected_revision=item["revision"] - 1)
    assert first["plan_created"] is True
    assert replay["reason_code"] == "IDEMPOTENT_REPLAY"
    assert conflict == {"plan_created": False, "reason_code": "IDEMPOTENCY_CONFLICT", "plan": None}


def _materialized(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    root = tmp_path / "runtime"
    output = root / "materialized" / "set"
    result = materialize_pointer_only_evidence(plan, output, approved_runtime_root=root)
    return source, warehouse, item, plan, root, output, result


def test_p17_to_p20_exact_files_utf8_manifest_and_hashes(tmp_path):
    _source, _warehouse, _item, _plan_value, _root, output, result = _materialized(tmp_path)
    assert sorted(path.name for path in output.iterdir()) == sorted(ALLOWED_FILES)
    assert all(path.read_text(encoding="utf-8") for path in output.iterdir())
    verified = verify_materialization(output)
    assert result["verified"] is True and verified["verified"] is True
    manifest = json.loads((output / "materialization_manifest.json").read_text(encoding="utf-8"))
    assert sorted(entry["path"] for entry in manifest["files"]) == sorted(ALLOWED_FILES)


def test_p21_same_input_two_directories_has_identical_hashes(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    root = tmp_path / "runtime"
    one = materialize_pointer_only_evidence(plan, root / "materialized" / "one", approved_runtime_root=root)
    two = materialize_pointer_only_evidence(plan, root / "materialized" / "two", approved_runtime_root=root)
    assert one["file_hashes"] == two["file_hashes"]


def test_p22_identical_existing_output_is_replay(tmp_path):
    _source, _warehouse, _item, plan, root, output, first = _materialized(tmp_path)
    replay = materialize_pointer_only_evidence(plan, output, approved_runtime_root=root)
    assert first["materialized"] is True
    assert replay["reason_code"] == "IDEMPOTENT_REPLAY"
    assert replay["file_hashes"] == first["file_hashes"]


def test_n20_conflicting_existing_output_not_overwritten(tmp_path):
    _source, _warehouse, _item, plan, root, output, _first = _materialized(tmp_path)
    before = (output / "README_POINTER_ONLY.md").read_bytes()
    (output / "README_POINTER_ONLY.md").write_text("conflict\n", encoding="utf-8")
    changed = (output / "README_POINTER_ONLY.md").read_bytes()
    with pytest.raises(PromotionDryRunError, match="MATERIALIZATION_CONFLICT"):
        materialize_pointer_only_evidence(plan, output, approved_runtime_root=root)
    assert changed != before and (output / "README_POINTER_ONLY.md").read_bytes() == changed


def test_n21_extra_materialized_file_causes_conflict(tmp_path):
    _source, _warehouse, _item, plan, root, output, _first = _materialized(tmp_path)
    (output / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PromotionDryRunError, match="MATERIALIZATION_CONFLICT"):
        materialize_pointer_only_evidence(plan, output, approved_runtime_root=root)


def test_n22_prohibited_plan_content_creates_nothing(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    changed = {**plan, "raw_text": "blocked-marker"}
    root = tmp_path / "runtime"; output = root / "materialized" / "set"
    with pytest.raises(PromotionDryRunError, match="PROMOTION_PLAN_INVALID"):
        materialize_pointer_only_evidence(changed, output, approved_runtime_root=root)
    assert not output.exists()


def test_n23_injected_hash_mismatch_cleans_temp(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    root = tmp_path / "runtime"; output = root / "materialized" / "set"
    with pytest.raises(PromotionDryRunError, match="MATERIALIZATION_HASH_MISMATCH"):
        materialize_pointer_only_evidence(plan, output, approved_runtime_root=root, inject_failure="hash_mismatch_readback")
    assert not output.exists() and not any(root.rglob("*.tmp-*"))


def test_n24_path_escape_rejected(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    root = tmp_path / "runtime"
    with pytest.raises(PromotionDryRunError, match="MATERIALIZATION_PATH_OUTSIDE_APPROVED_ROOT"):
        materialize_pointer_only_evidence(plan, tmp_path / "outside", approved_runtime_root=root)


def test_n25_production_library_path_rejected(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    root = tmp_path / "runtime"
    with pytest.raises(PromotionDryRunError, match="PRODUCTION_LIBRARY_PATH_FORBIDDEN"):
        materialize_pointer_only_evidence(plan, root / "library" / "set", approved_runtime_root=root)


@pytest.mark.parametrize(
    "failure",
    ["before_directory_creation", "after_one_temporary_file", "before_manifest_finalization", "during_final_atomic_rename"],
)
def test_p23_to_p26_injected_failures_leave_no_final_or_temp(failure, tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    root = tmp_path / "runtime"; output = root / "materialized" / "set"
    before = warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"]
    with pytest.raises(PromotionDryRunError):
        materialize_pointer_only_evidence(plan, output, approved_runtime_root=root, inject_failure=failure)
    after = warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"]
    assert not output.exists() and not any(root.rglob("*.tmp-*")) and before == after


def test_p26_atomic_failure_preserves_prior_valid_set(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    root = tmp_path / "runtime"; valid = root / "materialized" / "valid"; failed = root / "materialized" / "failed"
    before = materialize_pointer_only_evidence(plan, valid, approved_runtime_root=root)["file_hashes"]
    with pytest.raises(PromotionDryRunError, match="INJECTED_DURING_ATOMIC_RENAME"):
        materialize_pointer_only_evidence(plan, failed, approved_runtime_root=root, inject_failure="during_final_atomic_rename")
    assert verify_materialization(valid)["file_hashes"] == before and not failed.exists()


def test_p27_invalid_plan_creates_nothing(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    plan = {**plan, "status": "invalid"}
    root = tmp_path / "runtime"; output = root / "materialized" / "set"
    with pytest.raises(PromotionDryRunError, match="PROMOTION_PLAN_INVALID"):
        materialize_pointer_only_evidence(plan, output, approved_runtime_root=root)
    assert not output.exists()


def test_p28_p29_warehouse_unchanged_and_zero_library_write(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    before = warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"]
    plan = _plan(PromotionDryRunPlanner(), warehouse, source, item)["plan"]
    root = tmp_path / "runtime"; output = root / "materialized" / "set"
    materialize_pointer_only_evidence(plan, output, approved_runtime_root=root)
    after = warehouse.read_item(item["warehouse_item_id"], tenant_id=TENANT, organization_id=ORG)["item"]
    assert before == after
    assert plan["library_write_performed"] is False and plan["actual_promotion_performed"] is False


def test_p30_to_p36_materialized_boundary_counts_zero(tmp_path):
    _source, _warehouse, _item, _plan_value, _root, output, _result = _materialized(tmp_path)
    text = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    forbidden = ('"raw_query"', '"raw_body"', '"answer_text"', '"evidence_text"', '"user_id"', '"personal_name"', '"promotion_commit"')
    assert all(marker not in text for marker in forbidden)
    assert '"actual_promotion_performed": false' in text
    assert '"library_write_performed": false' in text


def test_p37_cleanup_removes_materialized_set(tmp_path):
    _source, _warehouse, _item, _plan_value, root, output, _result = _materialized(tmp_path)
    assert cleanup_materialization(output, approved_runtime_root=root)["deleted_count"] == 1
    assert cleanup_materialization(output, approved_runtime_root=root)["deleted_count"] == 0
    assert not output.exists()


def test_p38_canonical_serialization_deterministic():
    left = {"z": 1, "a": {"b": 2, "a": 1}}
    right = {"a": {"a": 1, "b": 2}, "z": 1}
    assert canonical_payload(left) == canonical_payload(right)


def test_r455_scoped_approval_read_contract(tmp_path):
    source, warehouse, item, _candidate, _record_id = _approved(tmp_path)
    found = warehouse.read_review_event_by_approval(
        item["warehouse_item_id"], APPROVAL, tenant_id=TENANT, organization_id=ORG
    )
    hidden = warehouse.read_review_event_by_approval(
        item["warehouse_item_id"], APPROVAL, tenant_id="TEN-OTHER", organization_id=ORG
    )
    assert found["review_event"]["decision"] == "APPROVE_WAREHOUSE"
    assert hidden == {"found": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "review_event": None}
